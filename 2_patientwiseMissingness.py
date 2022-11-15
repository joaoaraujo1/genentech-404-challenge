# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 13:31:21 2022

@author: João Araújo
"""
import lightgbm as lg
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import warnings
import pickle
import time 
warnings.filterwarnings("ignore")

print('Loading datasets...')
# Load extended df with full testA and testB entries
df = pd.read_pickle(r"C:\Users\João Araújo\Desktop\genentech-404-challenge\extended_df.pickle")
df = df['dataset_train']

# Load completed df_Test
df_Test = pd.read_pickle(r"C:\Users\João Araújo\Desktop\genentech-404-challenge\df_test.pickle")
df_Test = df_Test['dataset']

## DEALING WITH PATIENTS WITH A LOT OF LONGITUDINAL DATA
# Divide them in "2 patients": one with every odd entry and other with every
# even entry
# After this we end up with 863 "test participants" to find out missing data from
subj_unique = df_Test['RID_HASH'].unique()
for i in range(subj_unique.shape[0]):
    subj_data = df_Test[df_Test['RID_HASH'] == subj_unique[i]].copy()
    if(subj_data.shape[0] > 7):
        for j in range(subj_data.shape[0]):
            if j % 2 == 1:
                subj_data.iloc[j,0] = subj_data.iloc[j,0] + '##2'
        df_Test[df_Test['RID_HASH'] == subj_unique[i]] = subj_data
subj_unique = df_Test['RID_HASH'].unique()   

## DATA ENHANCEMENT: Sliding Window
# Make the most of our patient data by creating "new patients"
# dataframes with subsets of their entries
# Eg. patient_test has 2 trials patient_train has 8 trials 
# Use a sliding window on patient_train entries to get 6 "extra" training patients

# Create structure for easy access to full data df subjects
subj_unique_df = df['RID_HASH'].unique()
subj_entry_n_df = []

for i in range(subj_unique_df.shape[0]):
    subj_data = df[df['RID_HASH'] == subj_unique_df[i]].copy()
    subj_entry_n_df.append(subj_data.shape[0])

subj_entry_n_df = np.array(subj_entry_n_df)
print('Building training dataset trial structure...')
trials_subject_structure = np.unique(subj_entry_n_df)
trials_subject_structure = trials_subject_structure[1:]
trials_name_list_structure = [[]]*(trials_subject_structure.shape[0])
df_entry_structure = {'entry_n' : trials_subject_structure , \
                          'DF_LIST': trials_name_list_structure}
    
original_rid_hash = df['RID_HASH'].copy()
subj_unique_df = df['RID_HASH'].unique()   
for i in range(len(df_entry_structure['entry_n'])):
    test_entries = df_entry_structure['entry_n'][i]
    df_entry = []
    for s in range(len(subj_unique_df)):
        name_entry = df.loc[df['RID_HASH'] == subj_unique_df[s]].copy()
        
        if name_entry.shape[0] >= test_entries:
            division = 0
            # Sliding window method
            while True:
                sliding_entry = name_entry.iloc[division:division+test_entries,:].copy()
                new_name = subj_unique_df[s] + '#' + str(division)
                sliding_entry.iloc[:,0] = new_name
                
                if len(df_entry) == 0:
                    df_entry = sliding_entry
                else:
                    df_entry = df_entry.append(sliding_entry,ignore_index = True)
                
                if test_entries + division >= name_entry.shape[0]:
                    break
                else:
                    division +=1
                    
            
    df_entry_structure['DF_LIST'][i] = df_entry


# LOAD ROWISE MISSINGNESS MODELS (WHOLEDATA MODELS)
print('Loading wholedata models and missingness patterns...')
WHOLEDATA_INFO = []
with (open(r"C:\Users\João Araújo\Desktop\genentech-404-challenge\missingness_models.pickle", "rb")) as openfile:
    while True:
        try:
            WHOLEDATA_INFO.append(pickle.load(openfile))
        except EOFError:
            break


"""
MAIN IDEA

1) Check df_Test for each patient's data entries

2) Create similar missingness patterns on the training DataFrane using training 
   patients with the same number of entries

3) Train models using all available variables to predict every nan slot. 
   
4) Compare performance with whole data models using cross-validation

4) Train the best performing model with all the data available and predict the
   missing variable
"""
# Custom MAE evaluation function
def NMAE_eval(y_hat, y):
    return np.mean(np.abs(y-y_hat))


all_missing_patterns = []

saving_clock = time.time()
for i in range(subj_unique.shape[0]):
    subj_data = df_Test[df_Test['RID_HASH'] == subj_unique[i]].copy()
    subj_data = subj_data.drop(['RID_HASH'], axis=1)
    if np.sum(np.sum(subj_data.isna()).values) > 0:
        subj_data_to_edit = subj_data.values
        print('Solving nans for subject '+str(i+1)+'/'+str(subj_unique.shape[0])+' ('+str(subj_data.shape[0])+' records)')
        
        # Put subject data in vector form for prediction
        X_vector_predict = []
        for j in range(subj_data.shape[1]):
            
            x = subj_data.iloc[:,j][subj_data.iloc[:,j].notnull().values].values
        
            X_vector_predict = X_vector_predict + list(x)
        
        X_vector_predict = np.expand_dims(X_vector_predict,axis = 0)
        
        # 1-entry subjects should be predicted w/ wholedata models
        test_entries = subj_data.shape[0] 
        if test_entries > 1: 
            
            # Check if we have at least one subject with similar number of entries
            # using our previously built structure
            df_subjects_names_index, = np.where(df_entry_structure['entry_n'] == test_entries)
            df_entryopt = df_entry_structure['DF_LIST'][int(df_subjects_names_index)]
            df_subjects = df_entryopt['RID_HASH'].unique()
            if df_subjects.shape[0] > 0:

                # Get missingness pattern
                missing_pattern = []
                for column in range(subj_data.shape[1]):
                    for row in range(subj_data.shape[0]):
                        if np.isnan(subj_data.iloc[row,column]):
                            missing_pattern.append([row,column])
                # Build X and Y
                X = np.zeros((subj_data.shape[0]*subj_data.shape[1]-len(missing_pattern),len(df_subjects))).T
                Y = np.zeros((len(missing_pattern),len(df_subjects))).T
                isCategorical_X = []
                varname_Y = []
                for s in range(len(df_subjects)):
                    df_subject_data = df_entryopt[df_entryopt['RID_HASH'] == df_subjects[s]].copy()
                    df_subject_data = df_subject_data.iloc[:test_entries,:]
                    df_subject_data = df_subject_data.drop(['RID_HASH'], axis=1)
                    X_vector = []
                    y_vector = []
                    for j in range(df_subject_data.shape[1]):
                                                
                        x = df_subject_data.iloc[:,j][subj_data.iloc[:,j].notnull().values].values
                        
                        y = df_subject_data.iloc[:,j][subj_data.iloc[:,j].isna().values].values
                            
                        X_vector = X_vector + list(x)
                        y_vector = y_vector + list(y)
                        
                        if s == 0:
                            if df_subject_data.iloc[:,j].name == 'PTGENDER_num' or df_subject_data.iloc[:,j].name == 'DX_num' or df_subject_data.iloc[:,j].name == 'APOE4':
                                isCategorical_X = isCategorical_X + list(np.ones(len(x)))
                            else:
                                isCategorical_X = isCategorical_X + list(np.zeros(len(x)))
                            varname_Y = varname_Y + [df_subject_data.iloc[:,j].name] * len(y)
            
                    X[s,:] = X_vector
                    Y[s,:] = y_vector
                    if s == 0:
                        categorical_vars = np.where(np.array(isCategorical_X) == 1)
                        # Code to deal with formatting issues of 1-sized np arrays list conversion
                        if np.size(categorical_vars) == 1:
                            categorical_vars = [int(np.squeeze(categorical_vars))]
                        else:
                            categorical_vars = list(np.squeeze(categorical_vars))
                        
                        for cv in range(len(categorical_vars)):
                            categorical_vars[cv] = int(categorical_vars[cv])
                    
                # We have our dataset to optimize for this subject. ML time
                # Here we use K-fold and not group k-fold because each different
                # vector corresponds to one different patient. So we achieve a sort
                # of Group Kfold effect in a way by using the KFold method here
                n_samples = X.shape[0]
                if n_samples < 10:
                    kfold = KFold(n_splits=n_samples)
                else:
                    kfold = KFold(n_splits=10)
                splits = list(kfold.split(X))
                
                missing_pattern_mae = []
                missing_pattern_models = []
                # Predict for each y in Y sequentially
                for j in range(Y.shape[1]):
                    y = Y[:,j]
                    mae_pred_mean_all = []
                    mae_pred_linear_all = []
                    mae_pred_all = []
                    mae_pred_train_all = []
                    
                    for train_idc, test_idc in splits:
                        X_train = X[train_idc,:]
                        y_train = y[train_idc]
                        X_test = X[test_idc,:]
                        y_test = y[test_idc]
                        
                        y_pred_mean = np.ones(len(y_test)) * np.median(y_train)
                        
                        # Input for LGBM
                        boosts = 1000
                        num_ensembles = 10
                        y_pred = 0.0
                        
                        ## Categorical variables
                        # Note: I used regression for gender as it worked best
                        # on the public leaderboard for reasons I never understood
                        if varname_Y[j] == 'APOE14' or varname_Y[j] == 'DX_num':
                            # load data into train lightgbm dataset
                            if len(categorical_vars) > 0:
                                train = lg.Dataset(
                                    X_train, y_train, categorical_feature=categorical_vars, free_raw_data=False)
                            else:
                                train = lg.Dataset(X_train, y_train)

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
                                                  num_boost_round=boosts+a*10)
                                y_pred += model.predict(data=X_test)
                            y_pred /= num_ensembles
                            if varname_Y[j] == 'APOE4' or varname_Y[j] == 'DX_num':
                                y_pred = np.argmax(y_pred,axis = 1)
                            
                            model_linear = LogisticRegression(random_state=0,
                                                        solver = 'liblinear',
                                                        penalty='l1',
                                                        C=100)
                            
                            model_linear.fit(X_train, y_train)
    
                            y_pred_linear = model_linear.predict(X_test)
                        
                        ## Continuous variables
                        else:
                            # load data into train lightgbm dataset
                            if len(categorical_vars) > 0:
                                train = lg.Dataset(
                                    X_train, y_train, categorical_feature=categorical_vars, free_raw_data=False)
                            else:
                                train = lg.Dataset(X_train, y_train)
                            
                            
                            # hyperparameters for the model
                            parameters = {'num_leaves': 7,
                                          'learning_rate': 0.005,
                                          'bagging_fraction': 0.5,
                                          'bagging_freq': 2,
                                          'colsample_bytree': .6,
                                          'force_col_wise': 'true',
                                          'objective': 'mae',
                                          'verbose': -1}
                            for a in (range(num_ensembles)):
                                model = lg.train(parameters, 
                                                 train_set=train,
                                                 num_boost_round=boosts+a*10,
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
                    
                    # Find out missingness pattern of the line to retrieve the rowise missingness model
                    missingness_y_entry = list(subj_data.columns[subj_data.iloc[missing_pattern[j][0],:].isna().values])
                    for my in range(len(WHOLEDATA_INFO[0]['type'])):
                        if WHOLEDATA_INFO[0]['type'][my] == missingness_y_entry:
                            missing_whole = my
                            mae_whole = WHOLEDATA_INFO[0]['mae'][my][WHOLEDATA_INFO[0]['type'][my].index(varname_Y[j])]
                            model_whole = WHOLEDATA_INFO[0]['models'][my][WHOLEDATA_INFO[0]['type'][my].index(varname_Y[j])]
                            break
                    
                    performances = [np.mean(mae_pred_all),np.mean(mae_pred_linear_all),np.mean(mae_pred_mean_all),mae_whole]
                    # Save winning model and respective mae
                    model_final = []
                    mae_final = np.min(performances)
                    
                    if np.argmin(performances) == 0: #lgbm wins
                        y_pred_final = 0.0
                        print('LGBM wins best performance! Feature: '+varname_Y[j]+' | MAE: '+str(performances[0])[:7]+' (Wholedata MAE: '+str(performances[3])[:7]+', MType '+str(missing_whole)+')')
                        if len(categorical_vars) > 0:
                            train = lg.Dataset(
                                X, y, categorical_feature=categorical_vars, free_raw_data=False)
                        else:
                            train = lg.Dataset(X, y)
                        for a in range(num_ensembles):
                            model = lg.train(parameters, 
                                             train_set=train,
                                             num_boost_round=boosts+a*10,
                                             )      
                            y_pred_final += model.predict(data=X_vector_predict)
                        y_pred_final /= num_ensembles
                        if len(y_pred_final.shape) > 1:
                            if y_pred_final.shape[1]> 1: #multiclass classification
                                y_pred_final = np.argmax(y_pred_final,axis = 1)
                        
                    elif np.argmin(performances) == 1: #linear wins
                        print('Linear wins best performance! Feature: '+varname_Y[j]+' | MAE: '+str(performances[1])[:7]+' (Wholedata MAE: '+str(performances[3])[:7]+', MType '+str(missing_whole)+')')
                        # Check if it is a classifier for logistic regression   
                        if varname_Y[j] == 'APOE4' or varname_Y[j] == 'DX_num':
                            model = LogisticRegression(random_state=0,
                                                        solver = 'liblinear',
                                                        penalty='l1',
                                                        C=100)
                            
                        else:
                            
                            model = BayesianRidge()
                            
                        
                        model.fit(X, y)
                        y_pred_final = model.predict(X_vector_predict)
                    elif np.argmin(performances) == 2: # median is better
                        print('Median wins best performance! Feature: '+varname_Y[j]+' | MAE: '+str(performances[2])[:7]+' (Wholedata MAE: '+str(performances[3])[:7]+', MType '+str(missing_whole)+')')
                        #y_pred_final = np.median(y)
                        # I think I used the code below to get the median results, however the
                        # right idea would have been to use the commented code above
                        y_pred_final = df[varname_Y[j]].median()
                        
                        
                    else: # wholebrain models work best. we need to rebuild X_vector
                        print('Wholedata model wins best performance! Feature: '+varname_Y[j]+' | MAE: '+str(performances[3])[:7]+', MType '+str(missing_whole)+')')
                        #Create row-wise x-vector
                        X_vector_predict_new = np.expand_dims(np.array(subj_data.iloc[missing_pattern[j][0]][subj_data.iloc[missing_pattern[j][0]].notnull()].values),axis=0)
                        model_set = model_whole
                        if type(model_set) == str: # median
                            y_pred_final = df[varname_Y[j]].median()
                        elif type(model_set) != list: # linear
                            model = model_set
                            y_pred_final = model.predict(X_vector_predict_new)
                            if len(y_pred_final.shape) > 1:
                                if y_pred_final.shape[1]> 1: #multiclass classification
                                    y_pred_final = np.argmax(y_pred_final,axis = 1)
                        else: # ensemble of gboost
                            y_pred_final = 0.0
                            for a in range(len(model_set)):
                                y_pred_final += model_set[a].predict(data=X_vector_predict_new)
                            y_pred_final /= len(model_set)
                            if len(y_pred_final.shape) > 1:
                                if y_pred_final.shape[1]> 1: #multiclass classification
                                    y_pred_final = np.argmax(y_pred_final,axis = 1)
                    subj_data_to_edit[missing_pattern[j][0]][missing_pattern[j][1]] = np.squeeze(y_pred_final)
                    
        
        
        # Single entry: use Wholedata model
        else:
            # Find out missingness pattern of the line to load rowise missingness (wholedata) model
            missingness_y_entry = list(subj_data.columns[subj_data.isna().any()])
                
            for my in range(len(WHOLEDATA_INFO[0]['type'])):
                if WHOLEDATA_INFO[0]['type'][my] == missingness_y_entry:
                    missing_whole = my
                    maes = WHOLEDATA_INFO[0]['mae'][my]
                    models = WHOLEDATA_INFO[0]['models'][my]
                    break
            
            for j in range(len(missingness_y_entry)):
                column_idx = subj_data.columns.get_loc(missingness_y_entry[j])
                model_whole = models[j]
                expected_mae = maes[j]
            
                model_set = model_whole
                if type(model_set) == str: # mean
                    y_pred_final = df[missingness_y_entry[j]].mean()
                elif type(model_set) != list: # linear
                    model = model_set
                    y_pred_final = model.predict(X_vector_predict)
                    if len(y_pred_final.shape) > 1:
                        if y_pred_final.shape[1]> 1: #multiclass classification
                            y_pred_final = np.argmax(y_pred_final,axis = 1)
                else: # ensemble of gboost
                    y_pred_final = 0.0
                    for a in range(len(model_set)):
                        y_pred_final += model_set[a].predict(data=X_vector_predict)
                    y_pred_final /= len(model_set)
                    if len(y_pred_final.shape) > 1:
                        if y_pred_final.shape[1]> 1: #multiclass classification
                            y_pred_final = np.argmax(y_pred_final,axis = 1)
                
                subj_data_to_edit[0][column_idx] = np.squeeze(y_pred_final)
                #print('Single entry - using Wholedata model. Feature: '+missingness_y_entry[j]+' | Expected MAE: '+str(expected_mae)[:7]+')')
    
                
                            
        # Create new dataframe with our edited data
        subj_data = pd.DataFrame(subj_data_to_edit,columns = subj_data.columns) 
        # Add our entries to the test dataframe
        for col in subj_data.columns:
            df_Test.loc[df_Test['RID_HASH'] == subj_unique[i],col] = subj_data[col].values
        
        # Keep saving after at least half an hour of work
        if time.time() - saving_clock > 60*30:
            
            print('Saving dataset (pickle)...')

            dataframe_dict = {'dataset': df_Test}

            with open('df_test_sopt.pickle', 'wb') as handle:
                pickle.dump(dataframe_dict, handle)
            
            saving_clock = time.time()#reset clock
            

print('Saving final dataset (pickle)...')

dataframe_dict = {'dataset': df_Test}

with open('df_test_sopt.pickle', 'wb') as handle:
    pickle.dump(dataframe_dict, handle)
                      


        
        
        
    
