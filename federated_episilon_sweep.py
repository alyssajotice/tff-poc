import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import numpy as np

# --- INITIALIZE TFF ---
tff.backends.native.set_sync_local_cpp_execution_context()

from federated_baseline import get_federated_data, model_fn

def run_dp_experiment(epsilon_label, noise_val):
    print(f"\n--- Testing Epsilon Setting: {epsilon_label} (Noise: {noise_val}) ---")
    
    train_data = get_federated_data()
    
    # DP Aggregator
    dp_query = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=noise_val,
        clients_per_round=5,
        clip=1.0 # L2 Norm Clip
    )

    # Build Process
    iterative_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        model_aggregator=dp_query
    )

    state = iterative_process.initialize()
    
    final_accuracy = 0
    for round_num in range(1, 11): # 10 rounds per sweep to save time
        result = iterative_process.next(state, train_data)
        state = result.state
        final_accuracy = result.metrics['client_work']['train']['binary_accuracy']
    
    return final_accuracy

# --- MAIN SWEEP LOGIC ---
if __name__ == "__main__":
    # Mapping Noise to approximate Epsilon values
    # Higher noise = Stronger Privacy (Lower Epsilon)
    experiments = [
        {"label": "0.1 (Max Privacy)", "noise": 2.0},
        {"label": "1.0 (Strong Privacy)", "noise": 0.5},
        {"label": "10.0 (Moderate Privacy)", "noise": 0.1},
        {"label": "100.0 (Low Privacy)", "noise": 0.01}
    ]

    results = []

    for ex in experiments:
        acc = run_dp_experiment(ex['label'], ex['noise'])
        results.append({"Epsilon": ex['label'], "Accuracy": f"{acc:.4f}"})

    # Output Final Data Table
    print("\n" + "="*30)
    print("📊 FINAL THESIS DATA TABLE")
    print("="*30)
    print(pd.DataFrame(results))
    print("="*30)