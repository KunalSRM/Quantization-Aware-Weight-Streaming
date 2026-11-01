import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parents[1]
    ),
)

import torch


try:

    import transformers
    import mpi4py

    print(
        f"Python: {sys.version.split()[0]}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Transformers: {transformers.__version__}"
    )

    print(
        f"mpi4py: {mpi4py.__version__}"
    )

    print(
        f"CUDA available: {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: {torch.version.cuda}"
        )

except Exception as e:

    print(
        f"Verification error: {e}"
    )