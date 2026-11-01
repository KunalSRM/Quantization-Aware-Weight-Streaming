
import os

import psutil
import torch


def memory_metrics():
    process = psutil.Process(os.getpid())

    output = {
        "ram_gb": (
            process.memory_info().rss / (1024 ** 3)
        )
    }

    if torch.cuda.is_available():
        output.update({
            "gpu_allocated_gb": (
                torch.cuda.memory_allocated() / (1024 ** 3)
            ),
            "gpu_reserved_gb": (
                torch.cuda.memory_reserved() / (1024 ** 3)
            ),
            "gpu_peak_gb": (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            ),
        })

    return output
