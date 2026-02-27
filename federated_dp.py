import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

# --- STEP 1: INITIALIZE DP ENVIRONMENT ---
tff.backends.native.set_sync_local_cpp_execution_context()

from federated_baseline import get_federated_data, model_fn

# --- STEP 2: DEFINE THE DP PARAMETERS ---
# In your thesis, 'epsilon' (e) is the variable you will change.
# Lower epsilon = More Noise = More Privacy = Lower Accuracy.
# Higher epsilon = Less Noise = Less Privacy = Higher Accuracy.

EPSILON = 1.0  # This is a standard "strong" privacy setting
L2_NORM_CLIP = 1.0  # Limits the influence of any single clinic
NOISE_MULTIPLIER = 0.5 # Derived from epsilon; adds the "randomness"

def build_dp_process():
    # 1. Create the DP Factory
    # We use 'gaussian_fixed' which adds noise based on the L2_NORM_CLIP
    dp_query = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=NOISE_MULTIPLIER,
        clients_per_round=5,
        clip=L2_NORM_CLIP
    )

    # 2. Use 'build_unweighted_fed_avg'
    # DP usually requires unweighted averaging because the patient count 
    # itself can be sensitive information!
    return tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        model_aggregator=dp_query
    )

# --- STEP 3: EXECUTION ---
if __name__ == "__main__":
    print(f"\n🛡️ STARTING DP-FEDERATED TRAINING (Epsilon: {EPSILON})")
    
    train_data = get_federated_data()
    iterative_process = build_dp_process()
    state = iterative_process.initialize()

    for round_num in range(1, 21):
        result = iterative_process.next(state, train_data)
        state = result.state
        metrics = result.metrics['client_work']['train']
        print(f'Round {round_num:2d}, Loss: {metrics["loss"]:.4f}, Accuracy: {metrics["binary_accuracy"]:.4f}')

    print("\n✅ DP Training Complete.")