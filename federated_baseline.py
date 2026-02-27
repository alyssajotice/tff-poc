import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np
import collections

# 1. Initialize the TFF Execution Environment early to prevent the Popen error
tff.backends.native.set_sync_local_cpp_execution_context()

# --- STEP 1: LOAD & PARTITION DATA ---
from baseline_training import load_and_preprocess_data

def get_federated_data():
    df = load_and_preprocess_data()
    # Shuffle and split into 5 even clinics
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    partitions = np.array_split(df, 5)
    
    federated_train_data = []
    for partition in partitions:
        # TFF needs an Ordered Dictionary of tensors
        labels = partition.pop('label').values.astype(np.int32).reshape(-1, 1)
        features = partition.values.astype(np.float32)
        
        # Create a dataset that TFF can iterate over
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        federated_train_data.append(ds.batch(20))
    
    return federated_train_data

# --- STEP 2: DEFINE THE MODEL FOR TFF ---
def model_fn():
    # We use the exact same architecture as your 73% baseline
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # We wrap it for TFF
    return tff.learning.models.from_keras_model(
        keras_model,
        # We must define the input spec so TFF knows the "shape" of a patient
        input_spec=(
            tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.int32)
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

# --- STEP 3: EXECUTION ---
if __name__ == "__main__":
    print("\n🚀 STARTING FEDERATED SIMULATION (NO PRIVACY YET)")
    
    # Load the partitioned data
    train_data = get_federated_data()
    
    # Build the Federated Averaging process
    # Client optimizer: How each clinic learns
    # Server optimizer: How the central server averages the clinics
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    # Initialize the global model state
    state = iterative_process.initialize()

    # Run for 20 rounds (similar to 20 epochs)
    for round_num in range(1, 21):
        result = iterative_process.next(state, train_data)
        state = result.state
        metrics = result.metrics['client_work']['train']
        print(f'Round {round_num:2d}, Loss: {metrics["loss"]:.4f}, Accuracy: {metrics["binary_accuracy"]:.4f}')

    print("\n✅ Federated Training Complete.")