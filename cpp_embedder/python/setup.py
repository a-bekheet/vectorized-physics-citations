"""
Setup script for cpp_embedder Python package.

This script handles:
1. Building the C++ shared library
2. Installing the Python package with the compiled library
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.dist import Distribution

# Package metadata
PACKAGE_NAME = "cpp_embedder"
VERSION = "1.0.0"
DESCRIPTION = "Pure C++ sentence embeddings for Python"
AUTHOR = "cpp_embedder developers"
PYTHON_REQUIRES = ">=3.8"

# Dependencies
INSTALL_REQUIRES = [
    "numpy>=1.19",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=6.0",
        "pytest-cov",
        "black",
        "isort",
        "mypy",
    ],
}


def get_lib_name() -> str:
    """Get platform-specific library name."""
    system = platform.system()
    if system == "Darwin":
        return "libcpp_embedder.dylib"
    elif system == "Windows":
        return "cpp_embedder.dll"
    else:
        return "libcpp_embedder.so"


def get_cmake_generator() -> str:
    """Get the appropriate CMake generator for the platform."""
    system = platform.system()
    if system == "Windows":
        return "NMake Makefiles"  # or "Visual Studio 16 2019"
    else:
        return "Unix Makefiles"


class CMakeBuild(build_ext):
    """Custom build command that runs CMake to build the C++ library."""

    def run(self):
        """Build the C++ library using CMake."""
        # Get paths
        source_dir = Path(__file__).parent.parent.absolute()
        build_dir = source_dir / "build"
        lib_name = get_lib_name()

        # Create build directory
        build_dir.mkdir(exist_ok=True)

        # Check if CMakeLists.txt exists
        cmake_lists = source_dir / "CMakeLists.txt"
        if not cmake_lists.exists():
            print(f"Warning: CMakeLists.txt not found at {cmake_lists}")
            print("Skipping C++ build. Library must be pre-built.")
            return

        # Configure CMake
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DBUILD_SHARED_LIBS=ON",
            f"-DBUILD_PYTHON_BINDINGS=ON",
        ]

        # Add platform-specific arguments
        if platform.system() == "Darwin":
            # Build for current architecture on macOS
            cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={platform.machine()}")

        try:
            # Run CMake configure
            print(f"Configuring CMake in {build_dir}...")
            subprocess.run(
                ["cmake", str(source_dir)] + cmake_args,
                cwd=build_dir,
                check=True,
            )

            # Run CMake build
            print("Building C++ library...")
            subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "-j"],
                cwd=build_dir,
                check=True,
            )

            # Find and copy the library
            lib_path = None
            for search_dir in [build_dir, build_dir / "lib", build_dir / "Release"]:
                candidate = search_dir / lib_name
                if candidate.exists():
                    lib_path = candidate
                    break

            if lib_path:
                # Copy to package directory
                package_dir = Path(__file__).parent / PACKAGE_NAME
                dest_path = package_dir / lib_name
                print(f"Copying {lib_path} to {dest_path}")
                shutil.copy2(lib_path, dest_path)
            else:
                print(f"Warning: Built library {lib_name} not found in {build_dir}")

        except subprocess.CalledProcessError as e:
            print(f"CMake build failed: {e}")
            print("You may need to build the library manually.")
        except FileNotFoundError:
            print("CMake not found. Please install CMake or build the library manually.")

        # Call parent implementation
        super().run()


class BinaryDistribution(Distribution):
    """Distribution that includes platform-specific binaries."""

    def has_ext_modules(self):
        return True


class InstallWithLib(install):
    """Custom install that ensures the library is included."""

    def run(self):
        self.run_command("build_ext")
        super().run()


# Long description from README
def get_long_description() -> str:
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return DESCRIPTION


# Package data (include the shared library)
def get_package_data() -> dict:
    lib_name = get_lib_name()
    return {
        PACKAGE_NAME: [
            lib_name,
            "*.pyi",  # Type stubs
        ],
    }


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=PYTHON_REQUIRES,
    packages=find_packages(),
    package_data=get_package_data(),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    cmdclass={
        "build_ext": CMakeBuild,
        "install": InstallWithLib,
    },
    distclass=BinaryDistribution,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=[
        "embeddings",
        "sentence-transformers",
        "nlp",
        "machine-learning",
        "c++",
    ],
)
