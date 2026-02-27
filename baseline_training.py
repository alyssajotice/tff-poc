import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data():
    print("[*] Loading and merging data from /app/data...")
    
    try:
        patients = pd.read_csv('data/patients.csv')
        observations = pd.read_csv('data/observations.csv')
        conditions = pd.read_csv('data/conditions.csv')

        # 1. Feature Engineering: Age
        patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
        patients['age'] = 2026 - patients['BIRTHDATE'].dt.year

        # 2. Feature Engineering: BMI
        bmi_data = observations[observations['DESCRIPTION'].str.contains('Body Mass Index', na=False, case=False)]
        bmi_data = bmi_data.sort_values('DATE').groupby('PATIENT').tail(1)
        bmi_data = bmi_data[['PATIENT', 'VALUE']].rename(columns={'PATIENT': 'Id', 'VALUE': 'bmi'})

        # 3. Create the CLINICAL LABEL (The Ground Truth)
        # We find all unique patient IDs that have the Prediabetes code
        prediabetic_ids = conditions[conditions['CODE'] == 714628002]['PATIENT'].unique()
        
        # 4. Merge Features
        df = patients.merge(bmi_data, on='Id')
        df['gender'] = df['GENDER'].map({'M': 0, 'F': 1})
        
        # 5. Assign Label: 1 if their ID is in the prediabetic list, else 0
        df['label'] = df['Id'].isin(prediabetic_ids).astype(int)

        # 6. Data Minimization & Type Casting
        final_df = df[['bmi', 'age', 'gender', 'label']].dropna()
        final_df['bmi'] = final_df['bmi'].astype(float)
        
        # Audit: How many positives vs negatives?
        positives = final_df['label'].sum()
        print(f"[*] Audit: Found {positives} cases of Prediabetes out of {len(final_df)} patients.")
        
        return final_df
    
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return None

        
def df_to_dataset(dataframe, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('label')
    features = df.values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.shuffle(buffer_size=len(dataframe)).batch(batch_size)
    return ds

# --- STEP 2: ARCHITECTURE DEFINITION ---
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- STEP 3: BASELINE EXECUTION ---
if __name__ == "__main__":
    print("\n" + "="*40)
    print("🚀 STARTING BASELINE TRAINING")
    print("="*40)
    
    processed_df = load_and_preprocess_data()
    
    if processed_df is not None:
        print(f"[*] Data Cleaned. Training on {len(processed_df)} records.")
        train_ds = df_to_dataset(processed_df)
        
        model = create_keras_model()
        history = model.fit(train_ds, epochs=20, verbose=1)
        
        final_acc = history.history['accuracy'][-1]
        print("\n" + "="*40)
        print(f"✅ BASELINE ACCURACY: {final_acc:.4f}")
        print("="*40)
        
        model.save('baseline_model.h5')