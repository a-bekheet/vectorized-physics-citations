cmake_minimum_required(VERSION 3.16)

project(cpp_embedder
    VERSION 1.0.0
    DESCRIPTION "Lightweight C++ text embedding library"
    LANGUAGES CXX
)

# =============================================================================
# Build Options
# =============================================================================

option(CPP_EMBEDDER_BUILD_CLI "Build the CLI executable" ON)
option(CPP_EMBEDDER_BUILD_SHARED "Build the shared library for bindings" ON)
option(CPP_EMBEDDER_BUILD_TESTS "Build tests" ON)
option(CPP_EMBEDDER_INSTALL "Generate install targets" ON)

# =============================================================================
# C++ Standard and Compiler Settings
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Position independent code (needed for shared library)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wcast-qual
        -Wformat=2
        -Wundef
        -Werror=return-type
        -Wno-unused-parameter
    )
    # Debug flags
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    endif()
    # Release flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    endif()
elseif(MSVC)
    add_compile_options(
        /W4
        /permissive-
        /utf-8
    )
    # Disable specific warnings
    add_compile_options(/wd4100)  # unreferenced formal parameter
endif()

# =============================================================================
# Source Files
# =============================================================================

set(CPP_EMBEDDER_MATH_SOURCES
    src/math/tensor.cpp
    src/math/ops.cpp
)

set(CPP_EMBEDDER_TOKENIZER_SOURCES
    src/tokenizer/vocab.cpp
    src/tokenizer/tokenizer.cpp
)

set(CPP_EMBEDDER_MODEL_SOURCES
    src/model/layers.cpp
    src/model/weights.cpp
    src/model/embedder.cpp
)

set(CPP_EMBEDDER_CORE_SOURCES
    ${CPP_EMBEDDER_MATH_SOURCES}
    ${CPP_EMBEDDER_TOKENIZER_SOURCES}
    ${CPP_EMBEDDER_MODEL_SOURCES}
)

set(CPP_EMBEDDER_CLI_SOURCES
    src/cli/main.cpp
    src/cli/args.cpp
)

set(CPP_EMBEDDER_BINDINGS_SOURCES
    src/bindings/c_api.cpp
)

# =============================================================================
# Static Library Target
# =============================================================================

add_library(cpp_embedder STATIC ${CPP_EMBEDDER_CORE_SOURCES})

target_include_directories(cpp_embedder
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(cpp_embedder PROPERTIES
    OUTPUT_NAME cpp_embedder
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# =============================================================================
# Shared Library Target (for Python bindings)
# =============================================================================

if(CPP_EMBEDDER_BUILD_SHARED)
    add_library(cpp_embedder_shared SHARED
        ${CPP_EMBEDDER_CORE_SOURCES}
        ${CPP_EMBEDDER_BINDINGS_SOURCES}
    )

    target_include_directories(cpp_embedder_shared
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    # Define export macro for shared library
    target_compile_definitions(cpp_embedder_shared PRIVATE CPP_EMBEDDER_BUILD_SHARED)

    # Platform-specific shared library settings
    if(APPLE)
        set_target_properties(cpp_embedder_shared PROPERTIES
            OUTPUT_NAME cpp_embedder
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH TRUE
            MACOSX_RPATH TRUE
        )
    elseif(WIN32)
        set_target_properties(cpp_embedder_shared PROPERTIES
            OUTPUT_NAME cpp_embedder
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            WINDOWS_EXPORT_ALL_SYMBOLS OFF
        )
    else()
        set_target_properties(cpp_embedder_shared PROPERTIES
            OUTPUT_NAME cpp_embedder
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()

    # Set symbol visibility
    set_target_properties(cpp_embedder_shared PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

# =============================================================================
# CLI Executable Target
# =============================================================================

if(CPP_EMBEDDER_BUILD_CLI)
    add_executable(cpp_embed ${CPP_EMBEDDER_CLI_SOURCES})

    target_link_libraries(cpp_embed PRIVATE cpp_embedder)

    set_target_properties(cpp_embed PROPERTIES
        OUTPUT_NAME cpp_embed
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# =============================================================================
# Tests
# =============================================================================

if(CPP_EMBEDDER_BUILD_TESTS)
    enable_testing()

    # Find all test files
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")

    foreach(TEST_SOURCE ${TEST_SOURCES})
        # Extract test name from filename (e.g., test_math.cpp -> test_math)
        get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

        add_executable(${TEST_NAME} ${TEST_SOURCE})

        target_link_libraries(${TEST_NAME} PRIVATE cpp_embedder)

        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )

        # Register with CTest
        add_test(
            NAME ${TEST_NAME}
            COMMAND ${TEST_NAME}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
    endforeach()
endif()

# =============================================================================
# Install Rules
# =============================================================================

if(CPP_EMBEDDER_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install static library
    install(TARGETS cpp_embedder
        EXPORT CppEmbedderTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # Install shared library
    if(CPP_EMBEDDER_BUILD_SHARED)
        install(TARGETS cpp_embedder_shared
            EXPORT CppEmbedderTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    # Install CLI
    if(CPP_EMBEDDER_BUILD_CLI)
        install(TARGETS cpp_embed
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()

    # Install headers
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
    )

    # Generate and install package configuration files
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/CppEmbedderConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CppEmbedderConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CppEmbedderConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppEmbedder
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/CppEmbedderConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/CppEmbedderConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppEmbedder
    )

    install(EXPORT CppEmbedderTargets
        FILE CppEmbedderTargets.cmake
        NAMESPACE CppEmbedder::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CppEmbedder
    )
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "cpp_embedder configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build CLI:      ${CPP_EMBEDDER_BUILD_CLI}")
message(STATUS "  Build shared:   ${CPP_EMBEDDER_BUILD_SHARED}")
message(STATUS "  Build tests:    ${CPP_EMBEDDER_BUILD_TESTS}")
message(STATUS "  Install:        ${CPP_EMBEDDER_INSTALL}")
message(STATUS "")
