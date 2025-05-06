# Technology Decisions

This document records the architectural decisions made for the cpp_embedder project.

---

## ADR-001: C++17 Standard

### Status
Accepted

### Context
We need to choose a C++ standard version that balances modern features with broad compiler support, while maintaining the "no external dependencies" constraint.

### Decision
Use **C++17** as the language standard.

### Rationale

**Features we use from C++17:**

| Feature | Usage |
|---------|-------|
| `std::string_view` | Zero-copy string handling in tokenizer |
| `std::optional` | Safe handling of missing values |
| `std::filesystem` | File path manipulation (weights, vocab) |
| Structured bindings | Cleaner iteration over maps |
| `if constexpr` | Compile-time branching (potential optimization) |
| Inline variables | Header-only friendly constants |

**Why not C++11/14:**
- Missing `string_view` would require more string copies
- Missing `optional` would require error-prone null pointers or exceptions
- Missing `filesystem` would require platform-specific code

**Why not C++20/23:**
- Concepts and modules not universally supported
- Ranges library adds complexity without necessity
- Broader compiler compatibility with C++17
- Target users may have older toolchains

### Consequences
- Requires GCC 7+, Clang 5+, or MSVC 19.14+
- Build instructions must specify `-std=c++17` flag

---

## ADR-002: Binary Weight Format

### Status
Accepted

### Context
Model weights (~90 MB) must be loaded efficiently. Options include:
1. Text formats (JSON, plaintext)
2. Standard binary formats (HDF5, NumPy .npy, SafeTensors)
3. Custom binary format

### Decision
Use a **custom simple binary format**.

### Format Specification

```
┌─────────────────────────────────────────────────────────────┐
│                     File Structure                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Offset 0:   HEADER                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Bytes 0-3:    Magic number (0x454D4244 = "EMBD")    │    │
│  │ Bytes 4-7:    Version (uint32_t, currently 1)       │    │
│  │ Bytes 8-11:   Number of tensors (uint32_t)          │    │
│  │ Bytes 12-15:  Header size (uint32_t)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Offset 16:  TENSOR INDEX (repeated for each tensor)        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Bytes 0-63:   Tensor name (null-padded string)      │    │
│  │ Bytes 64-67:  Number of dimensions (uint32_t)       │    │
│  │ Bytes 68-83:  Dimensions (4 × uint32_t, unused = 0) │    │
│  │ Bytes 84-91:  Data offset (uint64_t)                │    │
│  │ Bytes 92-99:  Data size in bytes (uint64_t)         │    │
│  └─────────────────────────────────────────────────────┘    │
│  (100 bytes per tensor entry)                               │
│                                                             │
│  Offset [header_size]:  TENSOR DATA                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Raw float32 values in row-major order               │    │
│  │ (little-endian, IEEE 754)                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Tensor Names

Following BERT/MiniLM naming convention:
```
embeddings.word_embeddings.weight
embeddings.position_embeddings.weight
embeddings.token_type_embeddings.weight
embeddings.LayerNorm.weight
embeddings.LayerNorm.bias
encoder.layer.0.attention.self.query.weight
encoder.layer.0.attention.self.query.bias
encoder.layer.0.attention.self.key.weight
encoder.layer.0.attention.self.key.bias
encoder.layer.0.attention.self.value.weight
encoder.layer.0.attention.self.value.bias
encoder.layer.0.attention.output.dense.weight
encoder.layer.0.attention.output.dense.bias
encoder.layer.0.attention.output.LayerNorm.weight
encoder.layer.0.attention.output.LayerNorm.bias
encoder.layer.0.intermediate.dense.weight
encoder.layer.0.intermediate.dense.bias
encoder.layer.0.output.dense.weight
encoder.layer.0.output.dense.bias
encoder.layer.0.output.LayerNorm.weight
encoder.layer.0.output.LayerNorm.bias
... (layers 1-5 follow same pattern)
```

### Rationale

**Why not HDF5:**
- Requires external library (libhdf5)
- Complex API for our simple use case
- Violates "no dependencies" constraint

**Why not NumPy .npy:**
- Requires parsing Python pickle protocol
- Multiple files needed for multiple tensors
- Numpy-specific format details

**Why not SafeTensors:**
- External parsing library needed
- More complex than necessary

**Why custom format:**
- Zero dependencies
- Direct memory mapping possible
- Simple to implement and debug
- Efficient loading (single read)
- Self-documenting with magic number

### Consequences
- Need Python script to convert from PyTorch weights
- Must document format for reproducibility
- Endianness assumption (little-endian)

---

## ADR-003: Memory Layout for Tensors

### Status
Accepted

### Context
Need consistent tensor storage format across all components.

### Decision
Use **row-major, contiguous float32 arrays** via `std::vector<float>`.

### Layout Convention

```
Matrix A with shape [M, N]:

Logical view:              Memory layout:
┌─────────────────┐        [a00, a01, a02, ..., a0(N-1),
│ a00 a01 a02 ... │         a10, a11, a12, ..., a1(N-1),
│ a10 a11 a12 ... │         ...
│ ...             │         a(M-1)0, a(M-1)1, ..., a(M-1)(N-1)]
└─────────────────┘

Element access: A[i][j] = data[i * N + j]
```

### 3D Tensor Convention (for batched operations)

```
Tensor T with shape [B, M, N]:

Element access: T[b][i][j] = data[b * M * N + i * N + j]
```

### Rationale

**Why row-major:**
- C++ standard for multi-dimensional arrays
- Matches typical loop patterns (outer loop over rows)
- Compatible with most linear algebra conventions

**Why float32:**
- Standard for neural network inference
- Sufficient precision for embeddings
- Matches original model weights

**Why std::vector:**
- Automatic memory management
- Bounds checking available (debug mode)
- Compatible with C APIs (data() pointer)
- No external dependency

### Consequences
- Shape must be tracked separately from data
- No automatic broadcasting (must implement explicitly)
- Cache-friendly access patterns require row-major iteration

---

## ADR-004: Build System

### Status
Accepted

### Context
Need cross-platform build system that's easy to use without external tools.

### Decision
Use **CMake** as the primary build system, with single-header fallback option.

### CMake Structure

```
cpp_embedder/
├── CMakeLists.txt              # Root build config
├── src/
│   ├── CMakeLists.txt          # Source build rules
│   ├── math/
│   │   └── CMakeLists.txt
│   ├── tokenizer/
│   │   └── CMakeLists.txt
│   └── model/
│       └── CMakeLists.txt
├── cli/
│   └── CMakeLists.txt          # CLI executable
├── bindings/
│   └── CMakeLists.txt          # Python bindings (optional)
└── tests/
    └── CMakeLists.txt          # Test suite
```

### Build Targets

| Target | Type | Description |
|--------|------|-------------|
| `cpp_embedder` | STATIC | Core library |
| `cpp_embedder_cli` | EXECUTABLE | Command-line tool |
| `cpp_embedder_python` | MODULE | Python extension (optional) |
| `cpp_embedder_tests` | EXECUTABLE | Test runner |

### Minimum CMake Version
CMake 3.14+ (for FetchContent, modern target features)

### Compiler Requirements

```cmake
target_compile_features(cpp_embedder PUBLIC cxx_std_17)
```

### Rationale

**Why CMake:**
- Industry standard for C++ projects
- Cross-platform (Windows, macOS, Linux)
- IDE integration (VS Code, CLion, Visual Studio)
- Handles dependency management (for optional Python bindings)

**Why also single-header option:**
- Some users prefer drop-in integration
- Simplifies embedding in other projects
- No build system knowledge required

### Consequences
- Users need CMake 3.14+ installed
- Must maintain both CMake and single-header builds
- Python bindings require additional setup (pybind11)

---

## ADR-005: Vocabulary File Format

### Status
Accepted

### Context
Need to load BERT/MiniLM vocabulary (30,522 tokens).

### Decision
Use **plain text format** matching HuggingFace's vocab.txt.

### Format

```
[PAD]
[unused0]
[unused1]
...
[UNK]
[CLS]
[SEP]
[MASK]
[unused2]
...
the
,
.
of
...
##s
##ing
##ed
...
```

- One token per line
- Line number (0-indexed) is the token ID
- UTF-8 encoding
- Special tokens at fixed positions (defined by BERT)

### Special Token IDs

| Token | ID | Purpose |
|-------|-----|---------|
| [PAD] | 0 | Padding |
| [UNK] | 100 | Unknown token |
| [CLS] | 101 | Classification/start |
| [SEP] | 102 | Separator/end |
| [MASK] | 103 | Masked LM (unused for embeddings) |

### Rationale

**Why plain text:**
- Directly compatible with HuggingFace tokenizers
- Human readable and editable
- No parsing library needed
- Small file size (~230 KB)

### Consequences
- Must handle UTF-8 correctly
- Line-by-line reading is fast enough
- Could add binary vocab cache for faster loading (future optimization)

---

## ADR-006: Error Handling Strategy

### Status
Accepted

### Context
Need consistent approach to error handling across all components.

### Decision
Use **exceptions for unrecoverable errors**, **return values for expected failures**.

### Error Categories

| Category | Handling | Example |
|----------|----------|---------|
| File not found | Exception | Missing weights file |
| Invalid format | Exception | Corrupted weight file |
| Invalid input | Return value | Empty input string |
| Out of memory | Exception (std::bad_alloc) | Model too large |
| Logic errors | Assertion (debug) | Invalid tensor dimensions |

### Exception Types

```
std::runtime_error
├── FileNotFoundError      (weights, vocab files)
├── InvalidFormatError     (corrupted files)
└── ModelError             (weight mismatch, etc.)
```

### Rationale

**Why exceptions for I/O errors:**
- File loading is rare (once at startup)
- Errors are truly exceptional
- Cleaner code than error codes

**Why return values for input validation:**
- Empty/invalid input is expected user error
- Allows graceful handling without try/catch

**Why assertions for logic errors:**
- Catches bugs during development
- Zero overhead in release builds
- Documents invariants

### Consequences
- Must document which functions throw
- C API wrapper must catch and convert to error codes
- Python bindings translate to Python exceptions

---

## ADR-007: No SIMD/Parallelization (Initial Version)

### Status
Accepted

### Context
Performance optimizations like SIMD and multi-threading add complexity.

### Decision
**No SIMD or threading** in initial implementation. Pure scalar C++ only.

### Rationale

**Priority is correctness and readability:**
- Easier to verify against reference implementation
- Simpler debugging
- More portable (no platform-specific intrinsics)
- Educational value (clear algorithm visibility)

**Performance is acceptable for target use case:**
- Single inference: ~100-500ms on modern CPU (acceptable)
- Batch processing: can parallelize at application level
- Not targeting real-time applications

**Future optimization path:**
1. Profile to identify bottlenecks
2. Add optional SIMD (runtime detection)
3. Add optional threading (batch level)
4. Consider GPU backend (separate module)

### Consequences
- 10-100x slower than optimized implementations
- Suitable for prototyping, testing, and educational use
- Production users may need to contribute optimizations

---

## ADR-008: Python Binding Approach

### Status
Accepted

### Context
Python bindings are required for integration with existing Python workflows.

### Decision
Support **two binding approaches**:
1. **pybind11** (recommended): Full-featured, Pythonic API
2. **ctypes** (fallback): Zero-dependency, C API wrapper

### pybind11 Approach

```cpp
// Example binding structure
PYBIND11_MODULE(cpp_embedder, m) {
    py::class_<Embedder>(m, "Embedder")
        .def(py::init<const std::string&>())
        .def("embed", &Embedder::embed)
        .def("embed_batch", &Embedder::embed_batch);
}
```

### ctypes Approach

```cpp
// C API for ctypes
extern "C" {
    void* embedder_create(const char* model_path);
    void embedder_destroy(void* handle);
    int embedder_embed(void* handle, const char* text, float* output);
}
```

### Rationale

**Why pybind11:**
- Automatic Python type conversion
- NumPy integration built-in
- Exception translation
- Well-maintained, widely used

**Why also ctypes:**
- No compilation needed
- Works with any Python version
- No pybind11 dependency
- Useful for quick testing

### Consequences
- pybind11 requires C++ compiler at install time (or pre-built wheels)
- ctypes API must be stable and documented
- Both approaches must produce identical results

---

## Summary Table

| Decision | Choice | Key Reason |
|----------|--------|------------|
| Language Standard | C++17 | Modern features, broad support |
| Weight Format | Custom binary | Zero dependencies, efficient |
| Memory Layout | Row-major float32 | Standard, cache-friendly |
| Build System | CMake | Cross-platform, IDE support |
| Vocab Format | Plain text | HuggingFace compatible |
| Error Handling | Exceptions + returns | Appropriate for error type |
| Optimization | None (initial) | Simplicity, correctness first |
| Python Bindings | pybind11 + ctypes | Flexibility, zero-dep option |
