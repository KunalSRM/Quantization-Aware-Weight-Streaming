
import torch

from src.models.model_loader import get_device, load_model
from src.streaming.apply import apply_streaming

PROMPT = "Explain how CPU to GPU weight streaming can reduce GPU memory pressure during LLM inference."

device = get_device("cuda")

# FP16
tok_fp, model_fp = load_model("Qwen/Qwen2.5-0.5B-Instruct", device)

# INT4
tok_int, model_int = load_model("Qwen/Qwen2.5-0.5B-Instruct", device)
apply_streaming(model_int, 4, device)

inputs = tok_fp(PROMPT, return_tensors="pt").to(device)

with torch.inference_mode():
    logits_fp = model_fp(**inputs).logits[:, -1, :]
    logits_int = model_int(**inputs).logits[:, -1, :]

fp_token = logits_fp.argmax(-1).item()
int_token = logits_int.argmax(-1).item()

print("FP16 token :", fp_token)
print("INT4 token :", int_token)

print("FP16 text :", tok_fp.decode([fp_token]))
print("INT4 text :", tok_int.decode([int_token]))

print("Mean logit diff :", (logits_fp-logits_int).abs().mean().item())
print("Max logit diff  :", (logits_fp-logits_int).abs().max().item())
