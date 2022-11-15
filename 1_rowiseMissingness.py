# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 23:47:19 2022

@author: João Araújo
"""
# Imports
import lightgbm as lg
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")

# Custom MAE evaluation function
def NMAE_eval(y_hat, y):
    return np.mean(np.abs(y-y_hat))

# Explore different types of missingness patterns. save them so we can build several models
def getMissingness(row: pd.Series, missingness_type: list, missingness_n: list):
    missing = list(row.loc[row.isna()].index)
    if len(missing) > 0:
        if missing not in missingness_type:
            missingness_type.append(missing)
            missingness_n.append(1)
        else:
            idx = missingness_type.index(missing)
            missingness_n[idx] += 1

    return missingness_type, missingness_n

# Load initial training  and test datasets
#########################################################
print('Loading data and building test dataset...')
df = pd.read_csv(r"C:\Users\João Araújo\Desktop\genentech-404-challenge\dev_set.csv").sort_values(
    ['RID_HASH', 'VISCODE'], ignore_index=True)
df_A = pd.read_csv(r"C:\Users\João Araújo\Desktop\genentech-404-challenge\test_A.csv").sort_values(
    ['RID_HASH', 'VISCODE'], ignore_index=True)
df_B = pd.read_csv(r"C:\Users\João Araújo\Desktop\genentech-404-challenge\test_B.csv").sort_values(
    ['RID_HASH', 'VISCODE'], ignore_index=True)

features = df.columns

df_Test = pd.concat([df_A, df_B], axis=0, ignore_index=True)


# Fill in deducible data assuming that gender, APOE4 alleles, education and
# baseline age do not change across time
# Get 1940 entries filled in with MAE = 0
#########################################################
print('Auto-filling test dataset with deducible data...',end="")
missingness_type = []
missingness_n = []
subj_unique = df_Test['RID_HASH'].unique()
auto_entries = 0
for i in range(subj_unique.shape[0]):

    subj_data = df_Test[df_Test['RID_HASH'] == subj_unique[i]].copy()

    if subj_data.shape[0] > 1:

        ## AGE ##
        # Estimate missing age entries from subject's data if we have at least one entry
        if np.sum(subj_data['AGE'].isna()) > 0 and np.sum(subj_data['AGE'].isna()) < subj_data.shape[0]:
            auto_entries += np.sum(subj_data['AGE'].isna())

            age_val = subj_data['AGE'][subj_data['AGE'].notnull()].values[0]
            viscode_val = subj_data['VISCODE'][subj_data['AGE'].notnull(
            )].values[0]

            age_baseline = age_val - viscode_val/12
            subj_data['AGE'][subj_data['AGE'].isna()] = age_baseline + \
                subj_data['VISCODE'][subj_data['AGE'].isna()]/12
        else:
            pass

        ## GENDER ##
        # Estimate missing gender entries from subject's data if we have at least one entry
        if np.sum(subj_data['PTGENDER_num'].isna()) > 0 and np.sum(subj_data['PTGENDER_num'].isna()) < subj_data.shape[0]:
            auto_entries += np.sum(subj_data['PTGENDER_num'].isna())
            gender_val = subj_data['PTGENDER_num'][subj_data['PTGENDER_num'].notnull(
            )].values[0]
            subj_data['PTGENDER_num'][subj_data['PTGENDER_num'].isna()
                                      ] = gender_val
        else:
            pass

        ## EDUCATION ##
        # Estimate missing education entries from subject's data if we have at least one entry
        if np.sum(subj_data['PTEDUCAT'].isna()) > 0 and np.sum(subj_data['PTEDUCAT'].isna()) < subj_data.shape[0]:
            auto_entries += np.sum(subj_data['PTEDUCAT'].isna())
            educat_val = subj_data['PTEDUCAT'][subj_data['PTEDUCAT'].notnull(
            )].values[0]
            subj_data['PTEDUCAT'][subj_data['PTEDUCAT'].isna()] = educat_val
        else:
            pass

        ## Number of APOE alleles ##
        # Estimate missing APOE entries from subject's data if we have at least one entry
        if np.sum(subj_data['APOE4'].isna()) > 0 and np.sum(subj_data['APOE4'].isna()) < subj_data.shape[0]:
            auto_entries += np.sum(subj_data['APOE4'].isna())
            apoe_val = subj_data['APOE4'][subj_data['APOE4'].notnull()
                                          ].values[0]
            subj_data['APOE4'][subj_data['APOE4'].isna()] = apoe_val
        else:
            pass

        # Update our estimations DataFrame
        df_Test[df_Test['RID_HASH'] == subj_unique[i]] = subj_data

print(str(auto_entries)+ ' entries filled.')
#########################################################
print('Identifying missingness types...', end="")
for i in range(len(df_Test)):
    missingness_type, missingness_n = getMissingness(
        df_Test.loc[i], missingness_type, missingness_n)

# order by most frequency missingness types
missing_freq_order = list(np.argsort(missingness_n)[::-1])
missingness_type = [missingness_type[i] for i in missing_freq_order]
missingness_n = [missingness_n[i] for i in missing_freq_order]

# get proportion of missingness types
missingness_prop = missingness_n / np.sum(missingness_n)

plt.plot(np.cumsum(missingness_prop))
plt.show()
print('\tTotal missingness patterns: '+str(len(missingness_n)))


# Add 250 full entries from the test sets to our training set
################################################################
print('Adding full entries from the test set to training set...')
test_entries = []
for i in range(df_Test.shape[0]):
    if(np.sum(df_Test.iloc[i].isna()) == 0):
        test_entries.append(i)

df = pd.concat([df, df_Test.iloc[test_entries]], axis=0, ignore_index=True)
print('Added '+str(len(test_entries))+' entries to the training set.')

dataframe_dict = {'dataset_train': df}

with open('extended_df.pickle', 'wb') as handle:
    pickle.dump(dataframe_dict, handle)

# Train and save the best models/mae for each rowise missingness
# pattern for each variable
################################################################
print('Creating whole-data models for MD estimation...')

# group k-fold will ensure that no data from the same patient
# that was used in the training sets gets leaked to the test sets
group_kfold = GroupKFold(n_splits=5)
g_splits = list(group_kfold.split(df, df, df['RID_HASH']))


missingness_mae = []
missingness_models = []
for i in range(len(missingness_type)):

    print_str = '\t[Models for missingness pattern ' + \
        str((i+1))+'/'+str(len(missingness_type))+']'
    print(print_str)

    # Create our own missingness dataset
    nan_cols = missingness_type[i]

    df_X = df.copy()
    df_Y = df_X[nan_cols].copy()
    df_X = df_X.drop(['RID_HASH'], axis=1)
    df_X = df_X.drop(nan_cols, axis=1)

    missing_pattern_mae = []
    missing_pattern_models = []

    for j in range(len(nan_cols)):

        df_y = df_Y[nan_cols[j]].copy()
        mae_pred_mean_all = []
        mae_pred_linear_all = []
        mae_pred_all = []
        mae_pred_train_all = []

        for train_idc, test_idc in g_splits:
            
            X_train = df_X.iloc[train_idc].values
            y_train = df_y.iloc[train_idc].values
            X_test = df_X.iloc[test_idc].values
            y_test = df_y.iloc[test_idc].values
            
            y_pred_mean = np.ones(len(y_test)) * np.mean(y_train)
            
            # Input for LGBM
            boosts = 1000
            num_ensembles = 10
            y_pred = 0.0

            # Choose regression or classification models according to the variable
            if nan_cols[j] == 'PTGENDER_num' or nan_cols[j] == 'APOE4' or nan_cols[j] == 'DX_num':
                
                
                cat_features = []
                if X_train[:, df_X.columns == 'PTGENDER_num'].shape[1] > 0:
                    cat_features.append(
                        int(np.argwhere(df_X.columns == 'PTGENDER_num')))
                if X_train[:, df_X.columns == 'APOE4'].shape[1] > 0:
                    cat_features.append(
                        int(np.argwhere(df_X.columns == 'APOE4')))
                if X_train[:, df_X.columns == 'DX_num'].shape[1] > 0:
                    cat_features.append(
                        int(np.argwhere(df_X.columns == 'DX_num')))

                # load data into train lightgbm dataset
                if len(cat_features) > 0:
                    train = lg.Dataset(
                        X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
                else:
                    train = lg.Dataset(X_train, y_train)
                
                if nan_cols[j] == 'PTGENDER_num':
                    # hyperparameters for the model
                    parameters = {'num_leaves': 7,
                                  'learning_rate': 0.005,
                                  'bagging_fraction': 0.5,
                                  'bagging_freq': 2,
                                  'colsample_bytree': 0.6,
                                  'force_col_wise': 'true',
                                  'objective': 'binary',
                                  'verbose': -1}
                else:
                    parameters = {'num_leaves': 7,
                                  'learning_rate': 0.005,
                                  'bagging_fraction': 0.5,
                                  'bagging_freq': 2,
                                  'colsample_bytree': 0.6,
                                  'force_col_wise': 'true',
                                  'objective': 'softmax',
                                  'num_class': 3,
                                  'verbose': -1}

                for a in (range(num_ensembles)):
                    model = lg.train(parameters, train_set=train,
                                      num_boost_round=boosts+a+10)
                    y_pred += model.predict(data=X_test)
                y_pred /= num_ensembles
                if nan_cols[j] == 'APOE4' or nan_cols[j] == 'DX_num':
                    y_pred = np.argmax(y_pred,axis = 1)
                

                model_linear = LogisticRegression(random_state=0,
                                            solver = 'liblinear',
                                            penalty='l1',
                                            C=100,
                                            max_iter=10000)
                
                model_linear.fit(X_train, y_train)

                y_pred_linear = model_linear.predict(X_test)
            else:

                cat_features = []
                if X_train[:, df_X.columns == 'PTGENDER_num'].shape[1] > 0:
                    cat_features.append(
                        int(np.argwhere(df_X.columns == 'PTGENDER_num')))
                if X_train[:, df_X.columns == 'APOE4'].shape[1] > 0:
                    cat_features.append(
                        int(np.argwhere(df_X.columns == 'APOE4')))
                if X_train[:, df_X.columns == 'DX_num'].shape[1] > 0:
                    cat_features.append(
                        int(np.argwhere(df_X.columns == 'DX_num')))
                
                # load data into train lightgbm dataset
                if len(cat_features) > 0:
                    train = lg.Dataset(
                        X_train, y_train, categorical_feature=cat_features, free_raw_data=False)
                else:
                    train = lg.Dataset(X_train, y_train)

                # hyperparameters for the model
                parameters = {'num_leaves': 7,
                              'learning_rate': 0.005,
                              'bagging_fraction': 0.5,
                              'bagging_freq': 2,
                              'colsample_bytree': 0.6,
                              'force_col_wise': 'true',
                              'objective': 'mae',
                              'verbose': -1}

                for a in (range(num_ensembles)):
                    model = lg.train(parameters, 
                                     train_set=train,
                                     num_boost_round=boosts+a+10,
                                     )
                    y_pred += model.predict(data=X_test)
                y_pred /= num_ensembles

                model_linear = BayesianRidge()
                model_linear.fit(X_train,y_train)

                y_pred_linear = model_linear.predict(X_test)
                y_pred_train_linear = model_linear.predict(X_train)

            mae_pred = NMAE_eval(y_pred, y_test)
            mae_pred_mean = NMAE_eval(y_pred_mean, y_test)
            mae_pred_linear = NMAE_eval(y_pred_linear, y_test)
            mae_pred_all.append(mae_pred)
            mae_pred_linear_all.append(mae_pred_linear)
            mae_pred_mean_all.append(mae_pred_mean)

        performances = [np.mean(mae_pred_all),np.mean(mae_pred_linear_all),np.mean(mae_pred_mean_all)]
        ###########################################
        # Save winning model and respective mae
        model_final = []
        mae_final = np.min(performances)
        
        
        if np.argmin(performances) == 0: #lgbm wins
            print('LGBM wins best performance! Feature: '+nan_cols[j]+' | MAE: '+str(performances[0]))
            if len(cat_features) > 0:
                train = lg.Dataset(
                    df_X.values, df_y.values, categorical_feature=cat_features, free_raw_data=False)
            else:
                train = lg.Dataset(df_X.values, df_y.values)
            for a in range(num_ensembles):
                model = lg.train(parameters, 
                                 train_set=train,
                                 num_boost_round=boosts+a+10,
                                 )      
                model_final.append(model)
        elif np.argmin(performances) == 1: #linear wins
            print('Linear wins best performance! Feature: '+nan_cols[j]+' | MAE: '+str(performances[1]))
            # Check if it is a classifier for logistic regression   
            if nan_cols[j] == 'PTGENDER_num' or nan_cols[j] == 'APOE4' or nan_cols[j] == 'DX_num':
                model = LogisticRegression(random_state=0,
                                            solver = 'liblinear',
                                            penalty='l1',
                                            C=100,
                                            max_iter=10000)
            else:
                
                model = BayesianRidge()
            
            model.fit(df_X.values, df_y.values)
            model_final = model
        else: # mean is better
            print('Mean wins best performance! Feature: '+nan_cols[j]+' | MAE: '+str(performances[2]))
            model_final = 'mean'
                
                
        missing_pattern_models.append(model_final)
        missing_pattern_mae.append(mae_final)
        
    missingness_models.append(missing_pattern_models)
    missingness_mae.append(missing_pattern_mae)

#############
print('Saving data and models in pickle files...')

dataframe_dict = {'dataset': df_Test}

with open('df_test.pickle', 'wb') as handle:
    pickle.dump(dataframe_dict, handle)


missingness_dict = {'models': missingness_models, 'type': missingness_type, 'mae': missingness_mae}

with open('missingness_models.pickle', 'wb') as handle:
    pickle.dump(missingness_dict, handle)
    
print('Done')