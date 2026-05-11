FROM amdagi/training_ubuntu_rocm7.0.2_56_py312:v3_0427

WORKDIR /workspace/verl

# Pre-generate the GSM8K parquet files into /root/data/gsm8k
RUN python3 examples/data_preprocess/gsm8k.py --local_dir /root/data/gsm8k

# Pre-download the Qwen3-4B weights into the default HF cache (/root/.cache/huggingface)
RUN python3 -c "import transformers; transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B')"

# Install JupyterLab so the workshop notebook can be served from the container.
RUN pip install --no-cache-dir jupyterlab
