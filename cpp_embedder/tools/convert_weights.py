#!/usr/bin/env python3
"""
Convert PyTorch/HuggingFace sentence-transformer weights to cpp_embedder binary format.

This script converts weights from HuggingFace models (e.g., sentence-transformers/all-MiniLM-L6-v2)
or local PyTorch checkpoints to the binary format used by cpp_embedder.

Binary Format:
    [4 bytes] magic "EMBD" (0x44424D45 little-endian)
    [4 bytes] version (1)
    [4 bytes] num_tensors
    For each tensor:
        [4 bytes] name length
        [N bytes] name (UTF-8)
        [4 bytes] num_dims
        [num_dims * 4 bytes] shape (each dim as uint32)
        [product(shape) * 4 bytes] float32 data (little-endian, row-major)
    [4 bytes] vocab_size
    For each vocab token:
        [4 bytes] token length
        [N bytes] token string (UTF-8)

Usage:
    python convert_weights.py --model sentence-transformers/all-MiniLM-L6-v2 --output model.bin
    python convert_weights.py --checkpoint ./model.pt --vocab ./vocab.txt --output model.bin
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MAGIC = b"EMBD"
VERSION = 1

REQUIRED_EMBEDDING_TENSORS = [
    "embeddings.word_embeddings.weight",
    "embeddings.position_embeddings.weight",
    "embeddings.token_type_embeddings.weight",
    "embeddings.LayerNorm.weight",
    "embeddings.LayerNorm.bias",
]

ENCODER_LAYER_TENSORS = [
    "encoder.layer.{N}.attention.self.query.weight",
    "encoder.layer.{N}.attention.self.query.bias",
    "encoder.layer.{N}.attention.self.key.weight",
    "encoder.layer.{N}.attention.self.key.bias",
    "encoder.layer.{N}.attention.self.value.weight",
    "encoder.layer.{N}.attention.self.value.bias",
    "encoder.layer.{N}.attention.output.dense.weight",
    "encoder.layer.{N}.attention.output.dense.bias",
    "encoder.layer.{N}.attention.output.LayerNorm.weight",
    "encoder.layer.{N}.attention.output.LayerNorm.bias",
    "encoder.layer.{N}.intermediate.dense.weight",
    "encoder.layer.{N}.intermediate.dense.bias",
    "encoder.layer.{N}.output.dense.weight",
    "encoder.layer.{N}.output.dense.bias",
    "encoder.layer.{N}.output.LayerNorm.weight",
    "encoder.layer.{N}.output.LayerNorm.bias",
]

EXPECTED_SHAPES = {
    "embeddings.word_embeddings.weight": [30522, 384],
    "embeddings.position_embeddings.weight": [512, 384],
    "embeddings.token_type_embeddings.weight": [2, 384],
    "embeddings.LayerNorm.weight": [384],
    "embeddings.LayerNorm.bias": [384],
}

NUM_LAYERS = 6


def get_required_tensor_names() -> List[str]:
    """Generate the full list of required tensor names."""
    names = list(REQUIRED_EMBEDDING_TENSORS)
    for layer_idx in range(NUM_LAYERS):
        for pattern in ENCODER_LAYER_TENSORS:
            names.append(pattern.replace("{N}", str(layer_idx)))
    return names


def load_huggingface_model(model_name: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load model weights and vocabulary from HuggingFace.

    Args:
        model_name: HuggingFace model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")

    Returns:
        Tuple of (state_dict, vocabulary) where state_dict maps tensor names to numpy arrays
        and vocabulary is a list of tokens.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("Error: transformers library is required. Install with: pip install transformers")
        sys.exit(1)

    print(f"Loading model from HuggingFace: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    state_dict = {}
    for name, param in model.state_dict().items():
        tensor = param.cpu().numpy().astype(np.float32)
        state_dict[name] = tensor

    vocab = tokenizer.get_vocab()
    vocab_list = [""] * len(vocab)
    for token, idx in vocab.items():
        if idx < len(vocab_list):
            vocab_list[idx] = token

    print(f"Loaded {len(state_dict)} tensors and {len(vocab_list)} vocabulary tokens")
    return state_dict, vocab_list


def load_checkpoint(checkpoint_path: str, vocab_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load model weights from a local PyTorch checkpoint and vocabulary from a text file.

    Args:
        checkpoint_path: Path to PyTorch checkpoint file (.pt or .pth)
        vocab_path: Path to vocabulary text file (one token per line)

    Returns:
        Tuple of (state_dict, vocabulary)
    """
    try:
        import torch
    except ImportError:
        print("Error: torch library is required. Install with: pip install torch")
        sys.exit(1)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    state_dict = {}
    for name, param in checkpoint.items():
        if hasattr(param, "numpy"):
            tensor = param.cpu().numpy().astype(np.float32)
        else:
            tensor = np.array(param, dtype=np.float32)
        state_dict[name] = tensor

    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_list = [line.rstrip("\n") for line in f]

    print(f"Loaded {len(state_dict)} tensors and {len(vocab_list)} vocabulary tokens")
    return state_dict, vocab_list


def validate_tensors(state_dict: Dict[str, np.ndarray]) -> bool:
    """
    Validate that all required tensors are present with correct shapes.

    Args:
        state_dict: Dictionary mapping tensor names to numpy arrays

    Returns:
        True if validation passes, False otherwise
    """
    required_names = get_required_tensor_names()
    missing = []
    shape_errors = []

    for name in required_names:
        if name not in state_dict:
            missing.append(name)
            continue

        if name in EXPECTED_SHAPES:
            expected = EXPECTED_SHAPES[name]
            actual = list(state_dict[name].shape)
            if actual != expected:
                shape_errors.append(f"{name}: expected {expected}, got {actual}")

    if missing:
        print(f"Error: Missing {len(missing)} required tensors:")
        for name in missing[:10]:
            print(f"  - {name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        return False

    if shape_errors:
        print("Error: Tensor shape mismatches:")
        for err in shape_errors:
            print(f"  - {err}")
        return False

    return True


def write_binary(
    output_path: str,
    state_dict: Dict[str, np.ndarray],
    vocab: List[str],
) -> None:
    """
    Write tensors and vocabulary to binary format.

    Args:
        output_path: Path to output binary file
        state_dict: Dictionary mapping tensor names to numpy arrays
        vocab: List of vocabulary tokens
    """
    required_names = get_required_tensor_names()
    tensors_to_write = [(name, state_dict[name]) for name in required_names]

    total_tensor_bytes = 0
    for name, tensor in tensors_to_write:
        name_bytes = name.encode("utf-8")
        total_tensor_bytes += 4 + len(name_bytes)
        total_tensor_bytes += 4
        total_tensor_bytes += 4 * len(tensor.shape)
        total_tensor_bytes += tensor.nbytes

    total_vocab_bytes = 4
    for token in vocab:
        token_bytes = token.encode("utf-8")
        total_vocab_bytes += 4 + len(token_bytes)

    print(f"Writing {len(tensors_to_write)} tensors to {output_path}")

    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(tensors_to_write)))

        for name, tensor in tensors_to_write:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            shape = tensor.shape
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))

            tensor_data = tensor.astype(np.float32, copy=False)
            if not tensor_data.flags["C_CONTIGUOUS"]:
                tensor_data = np.ascontiguousarray(tensor_data)
            f.write(tensor_data.tobytes())

        f.write(struct.pack("<I", len(vocab)))
        for token in vocab:
            token_bytes = token.encode("utf-8")
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)

    file_size = Path(output_path).stat().st_size
    print(f"Wrote {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MB)")


def print_statistics(state_dict: Dict[str, np.ndarray], vocab: List[str]) -> None:
    """Print statistics about the model weights and vocabulary."""
    required_names = get_required_tensor_names()

    total_params = 0
    total_bytes = 0

    print("\n=== Model Statistics ===")
    print(f"Total tensors: {len(required_names)}")

    print("\nEmbedding tensors:")
    for name in REQUIRED_EMBEDDING_TENSORS:
        if name in state_dict:
            tensor = state_dict[name]
            params = tensor.size
            total_params += params
            total_bytes += tensor.nbytes
            print(f"  {name}: {list(tensor.shape)} ({params:,} params)")

    print(f"\nEncoder layers: {NUM_LAYERS}")
    layer_params = 0
    for layer_idx in range(NUM_LAYERS):
        for pattern in ENCODER_LAYER_TENSORS:
            name = pattern.replace("{N}", str(layer_idx))
            if name in state_dict:
                tensor = state_dict[name]
                layer_params += tensor.size
                total_params += tensor.size
                total_bytes += tensor.nbytes

    print(f"  Parameters per layer: {layer_params // NUM_LAYERS:,}")
    print(f"  Total encoder params: {layer_params:,}")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size: {total_bytes:,} bytes ({total_bytes / (1024 * 1024):.2f} MB)")
    print(f"Vocabulary size: {len(vocab):,} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch/HuggingFace weights to cpp_embedder binary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model sentence-transformers/all-MiniLM-L6-v2 --output model.bin
  %(prog)s --checkpoint ./model.pt --vocab ./vocab.txt --output model.bin
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to local PyTorch checkpoint file",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        help="Path to vocabulary text file (required with --checkpoint)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output binary file path",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip tensor validation (use with caution)",
    )

    args = parser.parse_args()

    if args.model and args.checkpoint:
        parser.error("Cannot specify both --model and --checkpoint")

    if not args.model and not args.checkpoint:
        parser.error("Must specify either --model or --checkpoint")

    if args.checkpoint and not args.vocab:
        parser.error("--vocab is required when using --checkpoint")

    if args.model:
        state_dict, vocab = load_huggingface_model(args.model)
    else:
        state_dict, vocab = load_checkpoint(args.checkpoint, args.vocab)

    print_statistics(state_dict, vocab)

    if not args.skip_validation:
        print("\nValidating tensors...")
        if not validate_tensors(state_dict):
            print("Validation failed. Use --skip-validation to bypass.")
            sys.exit(1)
        print("Validation passed.")

    print(f"\nWriting binary file...")
    write_binary(args.output, state_dict, vocab)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
