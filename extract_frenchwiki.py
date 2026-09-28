from datasets import load_dataset
import os
from tqdm import tqdm

# ========== CONFIG ==========
TARGET_SIZE_GB = 10                  # Stop when we reach ~10 GB
OUTPUT_FILE = "lucie_french_10gb.txt"
CONFIG = "Wikipedia-fr"              # Clean & good size. Other options below
# ============================

MAX_BYTES = TARGET_SIZE_GB * 1024 * 1024 * 1024

print(f"Streaming {CONFIG} ...")
ds = load_dataset(
    "OpenLLM-France/Lucie-Training-Dataset",
    CONFIG,
    split="train",
    streaming=True
)

written_bytes = 0
docs = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in tqdm(ds, desc="Extracting"):
        text = sample.get("text", "").strip()
        if not text or len(text) < 100:          # skip very short docs
            continue

        # Write document + separator
        chunk = text + "\n\n"
        chunk_bytes = len(chunk.encode("utf-8"))

        if written_bytes + chunk_bytes > MAX_BYTES:
            break

        f.write(chunk)
        written_bytes += chunk_bytes
        docs += 1

print(f"\nDone!")
print(f"Documents written : {docs:,}")
print(f"Final size        : {written_bytes / (1024**3):.2f} GB")
print(f"Saved to          : {OUTPUT_FILE}")