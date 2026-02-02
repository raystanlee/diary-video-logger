from datasets import load_dataset

def main():
    # Load a small slice quickly
    ds = load_dataset("HuggingFaceM4/charades", split="train[:5]")

    print("=== Charades Inspect ===")
    print("Dataset type:", type(ds))
    print("Num rows:", len(ds))
    print("Features:\n", ds.features)

    ex = ds[0]
    print("\n=== Example keys ===")
    print(list(ex.keys()))

    print("\n=== Example preview (truncated) ===")
    for k, v in ex.items():
        if k in ("video", "videos"):
            # video objects can be large; just show type/meta
            print(f"{k}: {type(v)}")
        else:
            s = str(v)
            print(f"{k}: {s[:300]}{'...' if len(s) > 300 else ''}")

if __name__ == "__main__":
    main()
