"""Quick verification that the C++ embedder produces valid embeddings."""
import sys
import time
import ctypes
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("=" * 60)
    print("C++ EMBEDDER VERIFICATION")
    print("=" * 60)

    # Paths
    weights_path = Path(__file__).parent.parent / "data/models/all-MiniLM-L6-v2.bin"
    vocab_path = Path(__file__).parent.parent / "data/models/all-MiniLM-L6-v2.vocab"
    lib_path = Path(__file__).parent.parent / "cpp_embedder/build/lib/libcpp_embedder.dylib"

    if not lib_path.exists():
        lib_path = Path(__file__).parent.parent / "cpp_embedder/build/lib/libcpp_embedder.so"

    print(f"\nWeights: {weights_path}")
    print(f"Vocab: {vocab_path}")
    print(f"Library: {lib_path}")

    if not all(p.exists() for p in [weights_path, vocab_path, lib_path]):
        print("\nError: Missing required files")
        return

    # Load library
    print("\nLoading C++ library...")
    lib = ctypes.CDLL(str(lib_path))

    # Define signatures
    lib.embedder_create_with_vocab.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.embedder_create_with_vocab.restype = ctypes.c_void_p
    lib.embedder_destroy.argtypes = [ctypes.c_void_p]
    lib.embedder_embed.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.embedder_embed.restype = ctypes.c_int
    lib.embedder_get_error.restype = ctypes.c_char_p

    # Create embedder
    print("Creating embedder...")
    start = time.perf_counter()
    handle = lib.embedder_create_with_vocab(
        str(weights_path).encode('utf-8'),
        str(vocab_path).encode('utf-8')
    )
    load_time = time.perf_counter() - start

    if not handle:
        error = lib.embedder_get_error()
        print(f"Error: {error.decode() if error else 'unknown'}")
        return

    print(f"Embedder loaded in {load_time:.2f}s")

    # Test embedding
    test_text = "hello world"
    print(f"\nEmbedding text: '{test_text}'")

    output_buffer = (ctypes.c_float * 384)()
    start = time.perf_counter()
    result = lib.embedder_embed(handle, test_text.encode('utf-8'), output_buffer, 384)
    embed_time = time.perf_counter() - start

    if result != 0:
        error = lib.embedder_get_error()
        print(f"Error: {error.decode() if error else 'unknown'}")
        lib.embedder_destroy(handle)
        return

    embedding = np.array(output_buffer)
    print(f"Embedding generated in {embed_time:.2f}s")

    # Verify embedding properties
    print("\n--- Embedding Properties ---")
    print(f"Shape: {embedding.shape}")
    print(f"Norm: {np.linalg.norm(embedding):.6f} (should be ~1.0 if normalized)")
    print(f"Mean: {embedding.mean():.6f}")
    print(f"Std: {embedding.std():.6f}")
    print(f"Min: {embedding.min():.6f}")
    print(f"Max: {embedding.max():.6f}")
    print(f"Non-zero: {np.count_nonzero(embedding)}/{len(embedding)}")

    # Check for NaN/Inf
    has_nan = np.any(np.isnan(embedding))
    has_inf = np.any(np.isinf(embedding))
    print(f"Has NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")

    # First few values
    print(f"\nFirst 10 values: {embedding[:10]}")

    # Compare with Python implementation
    print("\n--- Python Comparison ---")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        py_embedding = model.encode([test_text], normalize_embeddings=True)[0]

        cosine_sim = np.dot(embedding, py_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(py_embedding))
        l2_dist = np.linalg.norm(embedding - py_embedding)

        print(f"Cosine similarity with Python: {cosine_sim:.6f}")
        print(f"L2 distance from Python: {l2_dist:.6f}")

        if cosine_sim > 0.99:
            print("\n✓ C++ embedder produces highly similar results to Python!")
        elif cosine_sim > 0.9:
            print("\n~ C++ embedder produces similar results (some numerical differences)")
        else:
            print("\n✗ Significant difference from Python implementation")

    except ImportError:
        print("sentence-transformers not available for comparison")

    # Cleanup
    lib.embedder_destroy(handle)
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
