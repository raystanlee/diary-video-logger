import torch
from transformers import AutoVideoProcessor
from transformers import VJEPA2ForVideoClassification

def main():
    print("=== Environment Check ===")

    # 1. Device check
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device detected: {device}")

    # 2. Load processor
    print("Loading video processor...")
    processor = AutoVideoProcessor.from_pretrained(
        "facebook/vjepa2-vitl-fpc16-256-ssv2"
    )
    print("✓ Processor loaded")

    # 3. Load model
    print("Loading V-JEPA 2 model...")
    model = VJEPA2ForVideoClassification.from_pretrained(
        "facebook/vjepa2-vitl-fpc16-256-ssv2"
    )

    model.eval()
    model.to(device)

    print("✓ Model loaded and moved to device")
    print("✓ V-JEPA 2 environment is ready")

if __name__ == "__main__":
    main()
