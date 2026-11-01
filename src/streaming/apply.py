
import torch
import torch.nn as nn

from src.streaming.streamed_linear import StreamedLinear


def replace_linear_layers(
    module: nn.Module,
    bits: int,
    prefix: str = "",
    skip_names=("lm_head",),
):
    count = 0

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        # Preserve selected layers in their original precision.
        if full_name in skip_names or name in skip_names:
            continue

        if isinstance(child, nn.Linear):
            setattr(module, name, StreamedLinear(child, bits))
            count += 1
        else:
            count += replace_linear_layers(
                child,
                bits,
                full_name,
                skip_names,
            )

    return count


def apply_streaming(
    model: nn.Module,
    bits: int,
    device: torch.device,
):
    model.cpu()

    count = replace_linear_layers(
        model,
        bits,
        skip_names=("lm_head",),
    )

    # Move ordinary parameters and buffers to the target device.
    # StreamedLinear keeps its packed weights on CPU.
    for module in model.modules():
        if isinstance(module, StreamedLinear):
            continue

        for name, parameter in list(module._parameters.items()):
            if parameter is not None:
                module._parameters[name].data = parameter.data.to(device)

        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.to(device)

    model.eval()
    return count
