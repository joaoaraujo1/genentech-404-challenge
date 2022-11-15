# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:28:32 2022

@author: João Araújo
"""

import numpy as np
import pandas as pd

# We will edit the first sample submission file we were given
df_Submission = pd.read_csv(r"C:\Users\João Araújo\Desktop\genentech-404-challenge\first_sample_submission.csv")

# Load completed df_Test
df_Test = pd.read_pickle(r"C:\Users\João Araújo\Desktop\genentech-404-challenge\df_test_sopt.pickle")
df_Test = df_Test['dataset']

# Remove markers for subjects with large numbers of trials
for i in range(len(df_Test)):
    name = df_Test.iloc[i,0]
    if name[-3:] == "##2":
        name = name[:-3]
        df_Test.iloc[i,0] = name

# Rename columns to conform with the submission sample format
df_Test.rename(columns={'PTGENDER_num':'PTGENDER', 'DX_num':'DX'},inplace=True)

# Final MAE optimizations: For each subject with more than 1 entry,
# replace predicted APOE4 alleles, Gender, Age (baseline) and Education with their median
subj_unique = df_Test['RID_HASH'].unique()
for i in range(subj_unique.shape[0]):
    subj_data = df_Test[df_Test['RID_HASH'] == subj_unique[i]].copy()
    if subj_data.shape[0] > 1:
        # APOE4 alleles
        subj_data['APOE4'] = subj_data['APOE4'].median()  
        
        # Gender
        subj_data['PTGENDER'] = subj_data['PTGENDER'].median()
        
        # Education
        subj_data['PTEDUCAT'] = subj_data['PTEDUCAT'].median()
        
        # Age
        baseline_age_dist = subj_data['AGE'] - subj_data['VISCODE'] / 12
        subj_data['AGE'] = baseline_age_dist.median() + subj_data['VISCODE'] / 12
        
        df_Test[df_Test['RID_HASH'] == subj_unique[i]] = subj_data


# Remove negative values as none of the variables takes them
df_Test_vals = df_Test.iloc[:,1:]
df_Test_vals[df_Test_vals < 0] = 0.0
df_Test.iloc[:,1:] = df_Test_vals

# Populate the submission with our own predictions
for i in range(df_Submission.shape[0]):
   entry_string = df_Submission.iloc[i,0]
   ridhash = entry_string[:entry_string.index('_')]
   entry_string = entry_string[entry_string.index('_')+1:]
   viscode = int(entry_string[:entry_string.index('_')])
   entry_string = entry_string[entry_string.index('_')+1:]
   variable = entry_string[:entry_string.index('_')]
   mask = (df_Test['RID_HASH'] == ridhash) & (df_Test['VISCODE'] == viscode)
   if not np.isnan(np.squeeze(df_Test[mask][variable].values)):
       df_Submission.iloc[i,1] = df_Test[mask][variable].values
   else:
       print('ERROR!! NANs still present on the dataset!!')
       
# Save to CSV for submission
df_Submission.to_csv('first_sample_submission_edited.csv',index=False)