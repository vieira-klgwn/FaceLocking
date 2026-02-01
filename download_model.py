#!/usr/bin/env python3
"""Download ArcFace ONNX model for face recognition."""

import urllib.request
from pathlib import Path


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "embedder_arcface.onnx"

    if model_path.exists() and model_path.stat().st_size > 1000:
        print(f"Model exists: {model_path}")
        return

    urls = [
        "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx",
        "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx",
    ]
    for url in urls:
        try:
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, model_path)
            if model_path.stat().st_size > 1000:
                print(f"Saved to {model_path}")
                return
        except Exception as e:
            print(f"Failed: {e}")
    print("Manual: wget https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx -O models/embedder_arcface.onnx")


if __name__ == "__main__":
    main()
