
import torch
from src.models.model_loader import load_model
from src.streaming.apply import apply_streaming

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = (
    "Explain how CPU to GPU weight streaming can reduce "
    "GPU memory pressure during LLM inference."
)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is disabled.")

    device = torch.device("cuda")
    print("Loading FP16 reference model...")

    tokenizer_fp, model_fp = load_model(MODEL_ID, device)
    model_fp.eval()

    inputs_fp = tokenizer_fp(
        PROMPT, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        logits_fp = model_fp(**inputs_fp).logits[:, -1, :].float()

    fp_token = logits_fp.argmax(dim=-1).item()
    print("FP16 next token:", repr(tokenizer_fp.decode([fp_token])))

    del inputs_fp
    del model_fp
    del tokenizer_fp
    torch.cuda.empty_cache()

    print("\nLoading fresh model for INT4...")
    tokenizer_q, model_q = load_model(MODEL_ID, device)
    model_q.eval()

    count = apply_streaming(model_q, bits=4, device=device)
    print("Replaced linear layers:", count)

    inputs_q = tokenizer_q(
        PROMPT, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        logits_q = model_q(**inputs_q).logits[:, -1, :].float()

    q_token = logits_q.argmax(dim=-1).item()
    print("INT4 next token:", repr(tokenizer_q.decode([q_token])))

    if logits_fp.shape != logits_q.shape:
        raise RuntimeError(
            f"Logit shape mismatch: {logits_fp.shape} vs {logits_q.shape}"
        )

    # Reload FP16 model for a true comparison if needed.
    # This diagnostic reports token differences and INT4 logit health.
    print("INT4 logits finite:", bool(torch.isfinite(logits_q).all()))
    print("INT4 logit range:", logits_q.min().item(), logits_q.max().item())
    print("FP16 and INT4 top token IDs:", fp_token, q_token)
    print(
        "INT4 top-1 vs top-2 margin:",
        (
            logits_q.topk(2, dim=-1).values[0, 0]
            - logits_q.topk(2, dim=-1).values[0, 1]
        ).item()
    )

    print(
        "\nNote: this script checks output health. "
        "It does not calculate a layer-wise error ranking."
    )

if __name__ == "__main__":
    main()
