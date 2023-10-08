import timing
import torch

import numpy as np

DEFAULT_NUMBER_OF_REPETITIONS = 500
DEFAULT_GPU_WARMUP_REPETITIONS = 50

def pytorch_inference_time_cpu(model, input_data, n_reps = DEFAULT_NUMBER_OF_REPETITIONS):
    """Compute a pytorch model inference time in a cpu device"""

    starter, ender = 0, 0
    timings = np.zeros((n_reps, 1))

    with torch.no_grad():
        for rep in range(n_reps):
            starter = time.time()
            _ = model(input_data)
            ender = time.time()

            elapsed_time = ender - starter
            timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)
    std_inference_time = np.std(timings)

    return mean_inference_time


def pytorch_inference_time_gpu(model, input_data, n_reps = DEFAULT_NUMBER_OF_REPETITIONS, n_gpu_warmups = DEFAULT_GPU_WARMUP_REPETITIONS):
    """Compute a pytorch model inference time in a gpu device"""
    # References:
    # https://deci.ai/blog/measure-inference-time-deep-neural-networks

    # https://discuss.pytorch.org/t/elapsed-time-units/29951 (time in milliseconds)

    # Init timer loggers
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((n_reps, 1))

    # GPU Warm-up
    for _ in range(n_gpu_warmups):
        _ = model(input_data)

    # Measure performance
    with torch.no_grad():
        for rep in range(n_reps):
            starter.record()
            _ = model(input_data)
            ender.record()
            # Wait for gpu to sync
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender)
            timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)
    std_inference_time = np.std(timings)

    return mean_inference_time


def sklearn_inference_time(model, input_data, n_reps = DEFAULT_NUMBER_OF_REPETITIONS):
    """Compute a sklearn model inference time in any device"""

    starter, ender = 0, 0
    timings = np.zeros((n_reps, 1))

    for rep in range(n_reps):
        starter = time.time()
        _ = model.predict(input_data)
        ender = time.time()

        elapsed_time = ender - starter
        timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)
    std_inference_time = np.std(timings)

    return mean_inference_time
