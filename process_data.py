import pandas as pd
import numpy as np
import tensorflow as tf

# 1. Load the raw Synthea files
patients = pd.read_csv('patients.csv')
obs = pd.read_csv('observations.csv')
conds = pd.read_csv('conditions.csv')

# 2. Create the Label: Does the patient have Diabetes?
# Look for the SNOMED code for Diabetes (44054006 is a common one)
conds['is_diabetic'] = conds['DESCRIPTION'].str.contains('Diabetes', case=False).astype(int)
labels = conds.groupby('PATIENT')['is_diabetic'].max().reset_index()

# 3. Get Features: Most recent BMI and Glucose
# Filter observations for 'Body Mass Index' and 'Glucose'
bmi_obs = obs[obs['DESCRIPTION'] == 'Body Mass Index'].sort_values('DATE').groupby('PATIENT').last()
glucose_obs = obs[obs['DESCRIPTION'] == 'Glucose'].sort_values('DATE').groupby('PATIENT').last()

# 4. Join everything together
final_df = patients[['Id', 'BIRTHDATE', 'GENDER']].merge(labels, left_on='Id', right_on='PATIENT', how='left')
final_df = final_df.merge(bmi_obs[['VALUE']], left_on='Id', right_on='PATIENT', how='left')
final_df = final_df.rename(columns={'VALUE': 'bmi'})
final_df['is_diabetic'] = final_df['is_diabetic'].fillna(0) # If not found, assume negative

# 5. Clean & Normalize (The DP Requirement)
final_df = final_df.dropna(subset=['bmi']) # Remove patients without medical checks
final_df['bmi'] = (final_df['bmi'].astype(float) - 15) / (45 - 15) # Normalize BMI 15-45 range
final_df['bmi'] = final_df['bmi'].clip(0, 1)

print(f"✅ Preprocessing complete. Ready for training with {len(final_df)} patients.")
print(final_df[['Id', 'bmi', 'is_diabetic']].head())