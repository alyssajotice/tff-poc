import os
import collections
import tensorflow as tf
import tensorflow_federated as tff

# 1. ARM64 Environment Stability Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tff.backends.native.set_sync_local_cpp_execution_context()

print("--- 🚀 STARTING FULL FEDERATED LEARNING TUTORIAL ---")

# 2. LOAD DATA (The "Decentralized" Data)
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def preprocess(dataset):
  def batch_format_fn(element):
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 784]),
        y=tf.reshape(element['label'], [-1, 1]))
  return dataset.repeat(1).shuffle(100).batch(20).map(batch_format_fn)

# We simulate 10 "Hospitals" or "Users"
sample_clients = emnist_train.client_ids[:10]
federated_train_data = [preprocess(emnist_train.create_tf_dataset_for_client(x)) for x in sample_clients]

# 3. DEFINE THE MODEL (The "Global Blueprint")
def model_fn():
  keras_model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(784,)),
      tf.keras.layers.Dense(10, kernel_initializer='zeros'),
      tf.keras.layers.Softmax(),
  ])
  return tff.learning.models.from_keras_model(
      keras_model,
      input_spec=federated_train_data[0].element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# 4. INITIALIZE THE ALGORITHM (Federated Averaging)
training_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

state = training_process.initialize()

# 5. RUN THE TRAINING ROUNDS
# In your thesis, each round is a "Communication Cycle"
NUM_ROUNDS = 11
print(f"✅ Training started on {len(sample_clients)} clients across {NUM_ROUNDS} rounds...")

for round_num in range(1, NUM_ROUNDS):
  result = training_process.next(state, federated_train_data)
  state = result.state
  metrics = result.metrics['client_work']['train']
  print(f'Round {round_num:2d}, loss={metrics["loss"]:.4f}, accuracy={metrics["sparse_categorical_accuracy"]:.4f}')

print("--- 🏁 FULL TUTORIAL COMPLETE ---")