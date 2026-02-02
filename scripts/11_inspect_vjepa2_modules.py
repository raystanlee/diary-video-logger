from transformers import AutoModelForVideoClassification

MODEL = "facebook/vjepa2-vitl-fpc16-256-ssv2"

def show_children(prefix, module, max_items=40):
    items = list(module.named_children())
    print(f"{prefix} children: {len(items)}")
    for i, (name, child) in enumerate(items[:max_items]):
        print(f"  - {prefix}.{name:20s} => {child.__class__.__name__}")
    if len(items) > max_items:
        print(f"  ... ({len(items)-max_items} more)")

def main():
    m = AutoModelForVideoClassification.from_pretrained(
        MODEL,
        num_labels=80,
        ignore_mismatched_sizes=True
    )

    print("\n=== TOP LEVEL ===")
    show_children("model", m)

    # probe common candidates
    candidates = ["vjepa2", "vision_model", "backbone", "model", "encoder"]
    for c in candidates:
        if hasattr(m, c):
            obj = getattr(m, c)
            print(f"\n=== FOUND model.{c}: {obj.__class__.__name__} ===")
            show_children(f"model.{c}", obj)

if __name__ == "__main__":
    main()