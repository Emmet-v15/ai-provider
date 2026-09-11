"""Download ramthrust model metadata from HF, then convert single-file checkpoint to diffusers format in cache."""
import os, sys, shutil
from pathlib import Path

CACHE_DIR = Path(os.path.expanduser("~/.cache/huggingface/hub"))
MODEL_ID = "John6666/ramthrusts-nsfw-pink-alchemy-mix-169-sdxl"
LOCAL_CKPT = r"C:\Users\Admin\Documents\Projects\image-gen\loras\RAMTHRUST_S-NSFW-PINK-ALCHEMY-MIX.safetensors"

# ── Step 1: download config-only snapshot ──
from huggingface_hub import snapshot_download
import json, requests

print("Step 1: Downloading config files from HF...")
snapshot_download(
    MODEL_ID,
    allow_patterns=["*.json", "*.txt", "*.md", ".gitattributes"],
    cache_dir=str(CACHE_DIR),
    resume_download=True,
)
print("  configs downloaded.")

# ── Step 2: figure out cache paths ──
# huggingface_hub stores as: hub/models--org--name/
# blobs/<sha256>, refs/main, snapshots/<commit-sha>/ (symlinks to blobs)

# Get repo info
api_url = f"https://huggingface.co/api/models/{MODEL_ID}"
r = requests.get(api_url, timeout=15)
data = r.json()
commit_sha = data["sha"]

model_dir = CACHE_DIR / f"models--{MODEL_ID.replace('/', '--')}"
snap_dir = model_dir / "snapshots" / commit_sha
blobs_dir = model_dir / "blobs"

print(f"  Commit: {commit_sha}")
print(f"  Model dir: {model_dir}")
print(f"  Snapshot dir: {snap_dir}")
print(f"  Blobs dir: {blobs_dir}")

# Write ref
(model_dir / "refs" / "main").write_text(commit_sha + "\n")

# ── Step 3: Find which .safetensors blob paths we need ──
need_symlinks = {}
for item in requests.get(f"https://huggingface.co/api/models/{MODEL_ID}/tree/main", timeout=15).json():
    path = item.get("path", "")
    if path.endswith(".safetensors"):
        # Get the LFS pointer file to find the expected sha256
        ptr = requests.get(f"https://huggingface.co/{MODEL_ID}/raw/main/{path}", timeout=15).text
        for line in ptr.splitlines():
            if line.startswith("oid sha256:"):
                expected_sha = line.split(":", 1)[1].strip()
                need_symlinks[path] = expected_sha
                print(f"  {path} -> sha256:{expected_sha[:16]}...")
                break

print(f"\nStep 2: Converting single-file checkpoint to diffusers format...")
print(f"  Loading {LOCAL_CKPT}...")

# Use diffusers from_single_file → save_pretrained into the snapshot dir
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_single_file(
    LOCAL_CKPT,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
print("  Model loaded.")

# Save in diffusers format directly into the HF cache snapshot dir
os.makedirs(snap_dir, exist_ok=True)
pipe.save_pretrained(snap_dir, safe_serialization=True)
print(f"  Model saved to {snap_dir}")

# ── Step 4: update blob symlinks for the newly saved files ──
# diffusers save_pretrained creates actual .safetensors files in the snapshot dir.
# We need to move them into blobs/ and symlink back.

import hashlib

for fpath in snap_dir.rglob("*.safetensors"):
    rel = fpath.relative_to(snap_dir)
    file_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
    blob_path = blobs_dir / file_hash
    
    # Move to blobs if not already there
    if not blob_path.exists():
        print(f"  Moving {rel} -> blobs/{file_hash[:16]}...")
        fpath.rename(blob_path)
    else:
        fpath.unlink()
    
    # Create symlink
    rel_link = os.path.relpath(blob_path, snap_dir)
    os.symlink(rel_link, fpath)

print("\nDone! Model is now cached at:", snap_dir)
