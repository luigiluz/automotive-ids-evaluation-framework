import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    print(f"Number of available GPUs: {num_gpus}")

    # Get information about each GPU
    for gpu_id in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # in GB

        print(f"GPU {gpu_id + 1}: {gpu_name}, Memory: {gpu_memory:.2f} GB")
else:
    print("No GPUs are available on this system.")
