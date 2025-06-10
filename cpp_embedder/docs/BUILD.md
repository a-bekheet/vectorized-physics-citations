# Build Instructions

This document covers building cpp_embedder from source on Linux, macOS, and Windows.

## Prerequisites

### Required

- **C++17 compiler:**
  - GCC 7+ (Linux)
  - Clang 5+ (macOS, Linux)
  - MSVC 19.14+ / Visual Studio 2017+ (Windows)

- **CMake 3.16+**

### Optional

- **Python 3.8+** with NumPy - for Python bindings
- **PyTorch** - for weight conversion script

## Building on Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential cmake

# Clone repository
git clone <repository-url>
cd cpp_embedder

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(nproc)

# Run tests
./cpp_embedder_tests

# Install (optional)
sudo cmake --install .
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug, Release, RelWithDebInfo) |
| `BUILD_SHARED_LIBS` | OFF | Build shared library instead of static |
| `BUILD_TESTS` | ON | Build test suite |
| `BUILD_CLI` | ON | Build command-line tool |
| `BUILD_PYTHON_BINDINGS` | OFF | Build Python bindings |

Example with options:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_PYTHON_BINDINGS=ON
```

## Building on macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake (via Homebrew)
brew install cmake

# Clone and build
git clone <repository-url>
cd cpp_embedder
mkdir build && cd build

# Configure (for native architecture)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(sysctl -n hw.ncpu)

# Run tests
./cpp_embedder_tests
```

### Apple Silicon Notes

The build automatically detects the architecture. To explicitly build for Apple Silicon:

```bash
cmake .. -DCMAKE_OSX_ARCHITECTURES=arm64
```

For Intel Macs:

```bash
cmake .. -DCMAKE_OSX_ARCHITECTURES=x86_64
```

## Building on Windows

### Using Visual Studio

1. Install Visual Studio 2019 or later with C++ workload
2. Install CMake (or use the one bundled with Visual Studio)

```powershell
# Clone repository
git clone <repository-url>
cd cpp_embedder

# Create build directory
mkdir build
cd build

# Configure (Visual Studio generator)
cmake .. -G "Visual Studio 16 2019" -A x64

# Build
cmake --build . --config Release

# Run tests
.\Release\cpp_embedder_tests.exe
```

### Using MSVC Command Line

```powershell
# Open Developer Command Prompt for VS 2019
mkdir build && cd build
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
nmake
```

## Build Artifacts

After building, you will find:

| File | Description |
|------|-------------|
| `libcpp_embedder.a` / `cpp_embedder.lib` | Static library |
| `libcpp_embedder.so` / `.dylib` / `.dll` | Shared library (if enabled) |
| `cpp_embed` / `cpp_embed.exe` | CLI executable |
| `cpp_embedder_tests` | Test executable |

## Installing Python Bindings

### Option 1: Install from build

After building with `-DBUILD_PYTHON_BINDINGS=ON`:

```bash
cd python
pip install .
```

### Option 2: Development install

```bash
cd python
pip install -e ".[dev]"
```

### Option 3: Manual setup

1. Build the shared library:

```bash
cmake .. -DBUILD_SHARED_LIBS=ON -DBUILD_PYTHON_BINDINGS=ON
cmake --build . -j
```

2. Set the library path:

```bash
# Linux
export CPP_EMBEDDER_LIB_PATH=/path/to/build/libcpp_embedder.so

# macOS
export CPP_EMBEDDER_LIB_PATH=/path/to/build/libcpp_embedder.dylib

# Windows
set CPP_EMBEDDER_LIB_PATH=C:\path\to\build\Release\cpp_embedder.dll
```

3. Install Python package:

```bash
cd python
pip install .
```

## Verifying the Build

### Run Tests

```bash
./cpp_embedder_tests
```

Expected output:

```
Running math tests...
Running tokenizer tests...
Running layer tests...
All tests passed!
```

### Test CLI

```bash
./cpp_embed --help
```

Expected output:

```
cpp_embed - C++ sentence embedder CLI

Usage: cpp_embed [options]

Options:
  -m, --model PATH      Path to weights file (required)
  -t, --text TEXT       Text to embed (can be repeated)
  ...
```

### Test Python Bindings

```python
import cpp_embedder
print(cpp_embedder.Embedder.__doc__)
```

## Troubleshooting

### CMake not finding compiler

Ensure your compiler is in PATH:

```bash
# Check compiler
which g++  # Linux
which clang++  # macOS
where cl.exe  # Windows (in Developer Command Prompt)
```

### C++17 features not available

Explicitly set the C++ standard:

```bash
cmake .. -DCMAKE_CXX_STANDARD=17
```

### Python bindings: library not found

Set the `CPP_EMBEDDER_LIB_PATH` environment variable to the full path of the shared library.

### Windows: missing DLL

Ensure the DLL is in the same directory as the executable or in PATH.

### macOS: code signing issues

If you get security warnings, allow the binary in System Preferences > Security & Privacy, or sign the binary:

```bash
codesign --sign - ./cpp_embed
```

## IDE Integration

### Visual Studio Code

Create `.vscode/c_cpp_properties.json`:

```json
{
    "configurations": [
        {
            "name": "cpp_embedder",
            "includePath": [
                "${workspaceFolder}/include/**"
            ],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```

### CLion

CMake projects are automatically detected. Open the root `cpp_embedder` directory.

### Visual Studio

Open the folder containing `CMakeLists.txt`. Visual Studio will detect it as a CMake project.

## Cross-Compilation

### Linux to Windows (MinGW)

```bash
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/mingw-toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

### Building for older glibc

Use an older distribution or Docker container matching your target system's glibc version.

## Continuous Integration

### GitHub Actions Example

```yaml
jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - name: Configure
        run: cmake -B build -DCMAKE_BUILD_TYPE=Release
      - name: Build
        run: cmake --build build -j
      - name: Test
        run: ./build/cpp_embedder_tests
```
