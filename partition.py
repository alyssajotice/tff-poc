import pandas as pd
import numpy as np

def create_federated_partitions(df, num_clients=5):
    print(f"[*] Partitioning {len(df)} patients into {num_clients} even clinics...")
    
    # 1. Shuffle the data to ensure random distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Assign a Client ID (0 to 4) to each row
    # np.array_split handles cases where the division isn't perfect (1137 / 5)
    partitions = np.array_split(df, num_clients)
    
    federated_data = {}
    for i, partition in enumerate(partitions):
        client_id = f"clinic_{i}"
        federated_data[client_id] = partition
        
        # Audit each clinic
        positives = partition['label'].sum()
        print(f"    - {client_id}: {len(partition)} patients ({positives} pre-diabetic)")
    
    return federated_data

# --- EXECUTION ---
if __name__ == "__main__":
    # Assuming 'processed_df' is the output from your baseline script
    # For this demonstration, we'll reload your cleaned data logic
    from baseline_training import load_and_preprocess_data
    
    df = load_and_preprocess_data()
    if df is not None:
        clinics = create_federated_partitions(df)
        print("\n✅ Partitioning Complete. Ready for TFF simulation.")