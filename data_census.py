import pandas as pd

# 1. Load the conditions and patients
conds = pd.read_csv('conditions.csv')
patients = pd.read_csv('patients.csv')

total_patients = patients['Id'].nunique()

# 2. Count the occurrences of every condition
# We group by the 'DESCRIPTION' of the medical condition
condition_counts = conds.groupby('DESCRIPTION')['PATIENT'].nunique().sort_values(ascending=False)

# 3. Calculate Prevalence (%)
# This is vital for your thesis to show the "Baseline" before privacy is added
prevalence = (condition_counts / total_patients) * 100

# 4. Format the "Serious Research" Table
census_table = pd.DataFrame({
    'Condition': condition_counts.index,
    'Patient Count': condition_counts.values,
    'Prevalence (%)': prevalence.values
})

print(f"--- Data Census for {total_patients} Patients ---")
print(census_table.head(30)) # Look at the top 15 candidates