import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import os
import multiprocessing

# Suppress TensorFlow C++ logs (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("--- 🚀 RESEARCH LAB STATUS ---")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tff.backends.native.set_sync_local_cpp_execution_context()

# Verify Hardware Allocation
logical_cpus = multiprocessing.cpu_count()
print(f"✅ Logical CPUs detected: {logical_cpus}")

# Verify TFF stack
print(f"✅ TFF Version: {tff.__version__}")

print("------------------------------")

