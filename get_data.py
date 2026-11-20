# get_data.py - Step 2a: copy the Twin-2K-500 files into this project's "data" folder,
# so they are easy to find (and so Claude can read them too).

import os
from huggingface_hub import snapshot_download   # comes installed with "datasets"

snapshot_download(
    repo_id="LLM-Digital-Twin/Twin-2K-500",
    repo_type="dataset",
    # Only these parts: the README (instructions), the per-wave answers,
    # and the full personas. We skip "raw_data" (other researchers' AI runs).
    allow_patterns=["README.md", "wave_split/*", "full_persona/*"],
    local_dir="data",          # save into ./data inside talp-project
)

# Print what we got, with file sizes, so we can check it worked.
for folder, subfolders, files in os.walk("data"):
    if ".cache" in folder:
        continue
    for name in sorted(files):
        path = os.path.join(folder, name)
        size_mb = os.path.getsize(path) / 1_000_000
        print(f"{path}  ({size_mb:.1f} MB)")
