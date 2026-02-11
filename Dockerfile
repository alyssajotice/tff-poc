# Use Python 3.10 on amd64 for the best TFF compatibility
FROM --platform=linux/amd64 python:3.10-slim

# Step 1: System essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Modernize installer tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Step 3: THE JAX FIX - Essential for TFF 0.65.0
RUN pip install --no-cache-dir \
    jax==0.4.14 \
    jaxlib==0.4.14 \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Step 4: The Core Stack - Pinned for mathematical stability
RUN pip install --no-cache-dir tensorflow==2.15.0
RUN pip install --no-cache-dir tensorflow-federated==0.65.0
RUN pip install --no-cache-dir tensorflow-privacy==0.8.12

# Step 5: Research tools
RUN pip install --no-cache-dir pandas matplotlib jupyter

WORKDIR /app
EXPOSE 8888

# Step 6: STABILITY HARDENING
# We increase the timeout to 300s to handle the emulation latency on Mac Silicon
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.kernel_manager_class = 'jupyter_server.services.kernels.kernelmanager.MappingKernelManager'" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.MappingKernelManager.allowed_kernelspecs = ['python3']" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.KernelManager.control_connection_timeout = 300" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.KernelManager.request_info_timeout = 300" >> /root/.jupyter/jupyter_server_config.py

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]