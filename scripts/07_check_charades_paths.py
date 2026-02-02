import os
from datasets import load_dataset

def main():
    ds = load_dataset("HuggingFaceM4/charades", "480p", split="train[:2000]", trust_remote_code=True)
    paths = [ds[i]["video"] for i in range(len(ds))]

    exists = sum(os.path.exists(p) for p in paths)
    print("exists", exists, "out of", len(paths))
    print("example0:", paths[0], "|", os.path.exists(paths[0]))

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("\nfirst 5 missing:")
        for p in missing[:5]:
            print(" -", p)

if __name__ == "__main__":
    main()