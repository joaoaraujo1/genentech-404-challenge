# Genentech 404 Challenge (Kaggle) - 2nd place entry
## Kaggle's task description
To develop an imputation algorithm capable of handling different types of missingness in tabular datasets.

First, the organizers will release the development data consisting of a complete dataset along with three missingness masks meant to simulate different patterns of missing data. Participants are encouraged to split this dataset as they see fit for training and validation.

Finally, two held out test sets of equal size will be released, each with a different, unspecified missingness mechanism. The public leaderboard will include 20% of each test set and the private leaderboard will include the remaining 80% of each test set.

We expect the winning submissions will likely be an algorithm that can automatically characterize the pattern of missing data and adapt its imputation approach to the missingness type.

## Data Description
Each row in the csv corresponds to a single subject visit. The data is longitudinal and all datasets (training set, public leaderboard set, and private leaderboard set) were split by subject. Ordinal or categorical variables will be treated as continuous during evaluation. Performance will be measured using range normalized mean absolute error (MAE).

Variables that are always observed:

- `RID_hash` - unique subject ID
- `VISCODE` - visit code, referring to the number of months

Variables that may have missingness:

- `AGE` - age at that particular visit
- `PTGENDER_num` - sex {0: 'Male', 1: 'Female'}
- `PTEDUCAT` - education level
- `DX_num` - diagnosis {0: 'Cognitively Normal', 1: 'Mild Cognitive Impairment', 2: 'Dementia'}
- `APOE4` - number of APOE e4 alleles
- `CDRSB` - cognitive score, Clinical Dementia Rating Sum of Boxes
- `MMSE` - cognitive score, Mini‐Mental State Examination
- `ADAS13` - cognitive score, Alzheimer's Disease Assessment Scale-Cognitive Subscale
- `Ventricles` - ventricle volume
- `Hippocampus` - hippocampus volume
- `WholeBrain` - whole brain volume
- `Entorhinal` - entorhinal volume
- `Fusiform` - fusiform volume
- `MidTemp` - midtemp volume

## My solution (TL;DR)
Let *y* be a missing entry of test patient *n*, not easily deducible by its missingness context. Evaluate the “row-wise missingness” in the row entry where *y* is missing (i.e. how many variables are present/missing in that row) and the “patient-wise missingness” (i.e. how many variables are present/missing in the full data matrix of the patient n). Build datasets to train linear/non-linear models and centrality metrics for variable *y* for both missingness contexts. Estimate the MAE for all models and centrality metrics using grouped k-fold cross-validation. Pick the best solution and train a new model (or estimate a centrality metric if the models don’t work better) with all available training data. Estimate *y*. Repeat for all entries and test patients.

## My solution (Detailed)
### Free lunch: Easily deducible variables
Concatenate testA and testB datasets. Assume that gender, APOE4 alleles, level of education and baseline age (AGE @ VISCODE = 0) will not change in this senior population. In that case, for each patient, if you have even a single entry of one of these columns, you can fill any of their missing values. Special attention with AGE, as you need to add VISCODE / 12 to every missing row once you get its baseline. This will allow you to fill close to 2000 entries with an estimated MAE = 0.

### Training data enhancement (1): Add testA and testB full entries to the dataset
You will get an extra 250 entries in your training dataset (6.1% enhancement)

### Train row-wise missingness models (aka Wholedata models)
Let’s get an upper bound for our MAEs by assuming each variable can be somewhat accurately predicted by its row-wise missingness context. Get the number of possible row-wise missingness patterns (90 in total). Estimate the model or centrality metric that minimizes cross-validated MAE. I chose to train linear models (Bayesian Ridge Regression for continuous variables, Logistic Regression with L1 regularization for categorical variables), non-linear models (LGBM ensemble with very basic parameters to avoid overfitting with L1 objective for continuous variables and softmax/logistic objective for categorical variables) and mean as centrality metric. I validated each model using grouped K-fold cross validation (group = RID_HASH) to avoid intra-patient info to be leaked in the validation process. All non-missing variables (except RID_HASH) were part of the training set *X* and the missing variables became my prediction target *y*’s. The model that best minimized MAE was picked as the best and trained on the whole training dataset (hence I called these Wholedata models in my code). The row-wise missing variable patterns, best models and their respective MAEs were saved (3.5GB size) in the working directory.

### Combining row-wise and patient-wise optimization
Complementary to the idea of the row-wise (or entry-wise) missingness, I had the idea of optimizing the solution using patient-wise missingness. The idea is simple: Take a matrix consisting of all the entries from a specific test patient and register which variables are missing. When building the training set, include patients that have at least as many entries as this test participant on the training set. Create your training dataset X using the variables that are not missing in the test patient and your targets y as the missing entries.  For each target y, assess the best model for its prediction using K-fold cross validation. Compare the best MAE from this patient-wise optimization with the best MAE from row-wise optimization and pick the model who will give you the best MAE. Predict each missing entry with the best model trained on the full X dataset. Repeat for every test patient.

### Dealing with test set patients with large numbers of entries
The number of patients with larger numbers of entries is particularly lacking on the training dataset which could be a problem when trying to predict missing variables from test patients with large numbers of entries. I went around this problem by renaming every other entry of a test patient with number of entries > 7, creating 2 artificial test patients with around half of the original test patient’s entries each.

### Training data enhancement (2): Sliding window sampling for patient-wise optimization
To maximize the number of entries for each training set, I sampled the training patients data using a sliding window with step_size = 1 and window_size = number of test_participant entries. This allows us to sample from a single training patient multiple times and artificially increase the size of our training set quite considerably. Let’s say we have a 6-entry participant and we want to build a training dataset for a 2-entry test participant. Using this sliding window method, we transform our single 6-entry training participant into 5 “new” training patients with 2-entries each with 50% data overlap.

### Training patient-wise missingness models 
This is similar to the row-wise missingness but using the idea of patient-wise optimization. The model parameters were kept somewhat similar. Main differences are: I used regression rather than classification models for PT_GENDER and used median instead of mean as centrality metric. Also increased the boosting on LGBM a bit. **NOTE**: If you keep PT_GENDER as a categorical variable, you will indeed reach a solution with MAE < 0.066 on the private leaderboard.

### Final optimization: PT_GENDER, APOE4, PTEDUCAT, (baseline) AGE
While different models might have held different predictions for these variables on each row, their value within each patient matrix should not change across entries (AGE changes are only directly related with VISCODE, so we can still put it within this optimization realm). Therefore, for each patient (with n_entries > 1), I used the median of these predicted variables (as it should minimize MAE) as the final value for their entries across VISCODEs.

## Running this code
The code assumes the CSV training and test sets are in the working directory as well as the sample submission file. Run the scripts in their order (1,2,3) to replicate my solution.







