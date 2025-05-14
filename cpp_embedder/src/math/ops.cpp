#include "../../include/math/ops.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace cpp_embedder {
namespace math {

// =============================================================================
// Constants for GELU approximation
// =============================================================================

namespace {
    // sqrt(2/pi) for GELU approximation
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    // Coefficient for GELU approximation
    constexpr float GELU_COEF = 0.044715f;
}

// =============================================================================
// Matrix Multiplication
// =============================================================================

Tensor matmul(const Tensor& a, const Tensor& b) {
    // Handle different dimension cases

    // Case 1: Vector-Matrix multiplication (1D, 2D) -> 1D
    // a: (K,), b: (K, N) -> result: (N,)
    if (a.ndim() == 1 && b.ndim() == 2) {
        if (a.dim(0) != b.dim(0)) {
            throw std::invalid_argument(
                "matmul: vector length must match matrix rows. "
                "Got vector(" + std::to_string(a.dim(0)) + ") and matrix(" +
                std::to_string(b.dim(0)) + ", " + std::to_string(b.dim(1)) + ")");
        }

        Tensor::size_type K = a.dim(0);
        Tensor::size_type N = b.dim(1);
        Tensor result({N}, 0.0f);

        for (Tensor::size_type j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (Tensor::size_type k = 0; k < K; ++k) {
                sum += a[k] * b.at(k, j);
            }
            result[j] = sum;
        }
        return result;
    }

    // Case 2: Matrix-Vector multiplication (2D, 1D) -> 1D
    // a: (M, K), b: (K,) -> result: (M,)
    if (a.ndim() == 2 && b.ndim() == 1) {
        if (a.dim(1) != b.dim(0)) {
            throw std::invalid_argument(
                "matmul: matrix cols must match vector length. "
                "Got matrix(" + std::to_string(a.dim(0)) + ", " +
                std::to_string(a.dim(1)) + ") and vector(" +
                std::to_string(b.dim(0)) + ")");
        }

        Tensor::size_type M = a.dim(0);
        Tensor::size_type K = a.dim(1);
        Tensor result({M}, 0.0f);

        for (Tensor::size_type i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (Tensor::size_type k = 0; k < K; ++k) {
                sum += a.at(i, k) * b[k];
            }
            result[i] = sum;
        }
        return result;
    }

    // Case 3: Standard matrix multiplication (2D, 2D) -> 2D
    // a: (M, K), b: (K, N) -> result: (M, N)
    if (a.ndim() == 2 && b.ndim() == 2) {
        if (a.dim(1) != b.dim(0)) {
            throw std::invalid_argument(
                "matmul: incompatible shapes. "
                "Got (" + std::to_string(a.dim(0)) + ", " + std::to_string(a.dim(1)) +
                ") and (" + std::to_string(b.dim(0)) + ", " + std::to_string(b.dim(1)) + ")");
        }

        Tensor::size_type M = a.dim(0);
        Tensor::size_type K = a.dim(1);
        Tensor::size_type N = b.dim(1);
        Tensor result({M, N}, 0.0f);

        // Simple triple-nested loop (clarity over performance)
        for (Tensor::size_type i = 0; i < M; ++i) {
            for (Tensor::size_type j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (Tensor::size_type k = 0; k < K; ++k) {
                    sum += a.at(i, k) * b.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        return result;
    }

    // Case 4: Batched matrix multiplication (3D, 2D) -> 3D
    // a: (batch, M, K), b: (K, N) -> result: (batch, M, N)
    if (a.ndim() == 3 && b.ndim() == 2) {
        if (a.dim(2) != b.dim(0)) {
            throw std::invalid_argument(
                "matmul: batch matmul dimension mismatch");
        }

        Tensor::size_type batch = a.dim(0);
        Tensor::size_type M = a.dim(1);
        Tensor::size_type K = a.dim(2);
        Tensor::size_type N = b.dim(1);
        Tensor result({batch, M, N}, 0.0f);

        for (Tensor::size_type bi = 0; bi < batch; ++bi) {
            for (Tensor::size_type i = 0; i < M; ++i) {
                for (Tensor::size_type j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (Tensor::size_type k = 0; k < K; ++k) {
                        sum += a.at(bi, i, k) * b.at(k, j);
                    }
                    result.at(bi, i, j) = sum;
                }
            }
        }
        return result;
    }

    // Case 5: Batched matrix multiplication (3D, 3D) -> 3D
    // a: (batch, M, K), b: (batch, K, N) -> result: (batch, M, N)
    if (a.ndim() == 3 && b.ndim() == 3) {
        if (a.dim(0) != b.dim(0) || a.dim(2) != b.dim(1)) {
            throw std::invalid_argument(
                "matmul: batch matmul dimension mismatch");
        }

        Tensor::size_type batch = a.dim(0);
        Tensor::size_type M = a.dim(1);
        Tensor::size_type K = a.dim(2);
        Tensor::size_type N = b.dim(2);
        Tensor result({batch, M, N}, 0.0f);

        for (Tensor::size_type bi = 0; bi < batch; ++bi) {
            for (Tensor::size_type i = 0; i < M; ++i) {
                for (Tensor::size_type j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (Tensor::size_type k = 0; k < K; ++k) {
                        sum += a.at(bi, i, k) * b.at(bi, k, j);
                    }
                    result.at(bi, i, j) = sum;
                }
            }
        }
        return result;
    }

    throw std::invalid_argument(
        "matmul: unsupported tensor dimensions: " +
        std::to_string(a.ndim()) + "D and " + std::to_string(b.ndim()) + "D");
}

// =============================================================================
// Element-wise Operations
// =============================================================================

Tensor add(const Tensor& a, const Tensor& b) {
    if (!Tensor::shapes_equal(a, b)) {
        throw std::invalid_argument("add: shapes must match");
    }

    Tensor result = a.clone();
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        result[i] += b[i];
    }
    return result;
}

void add_inplace(Tensor& a, const Tensor& b) {
    if (!Tensor::shapes_equal(a, b)) {
        throw std::invalid_argument("add_inplace: shapes must match");
    }

    for (Tensor::size_type i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }
}

Tensor scale(const Tensor& a, float scalar) {
    Tensor result = a.clone();
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        result[i] *= scalar;
    }
    return result;
}

void scale_inplace(Tensor& a, float scalar) {
    for (Tensor::size_type i = 0; i < a.size(); ++i) {
        a[i] *= scalar;
    }
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    if (!Tensor::shapes_equal(a, b)) {
        throw std::invalid_argument("multiply: shapes must match");
    }

    Tensor result = a.clone();
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        result[i] *= b[i];
    }
    return result;
}

void multiply_inplace(Tensor& a, const Tensor& b) {
    if (!Tensor::shapes_equal(a, b)) {
        throw std::invalid_argument("multiply_inplace: shapes must match");
    }

    for (Tensor::size_type i = 0; i < a.size(); ++i) {
        a[i] *= b[i];
    }
}

// =============================================================================
// Activation Functions
// =============================================================================

Tensor gelu(const Tensor& x) {
    Tensor result = x.clone();
    gelu_inplace(result);
    return result;
}

void gelu_inplace(Tensor& x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    for (Tensor::size_type i = 0; i < x.size(); ++i) {
        float val = x[i];
        float x_cubed = val * val * val;
        float inner = SQRT_2_OVER_PI * (val + GELU_COEF * x_cubed);
        x[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }
}

Tensor relu(const Tensor& x) {
    Tensor result = x.clone();
    relu_inplace(result);
    return result;
}

void relu_inplace(Tensor& x) {
    for (Tensor::size_type i = 0; i < x.size(); ++i) {
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
    }
}

Tensor sigmoid(const Tensor& x) {
    Tensor result = x.clone();
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        result[i] = 1.0f / (1.0f + std::exp(-result[i]));
    }
    return result;
}

Tensor tanh_activation(const Tensor& x) {
    Tensor result = x.clone();
    for (Tensor::size_type i = 0; i < result.size(); ++i) {
        result[i] = std::tanh(result[i]);
    }
    return result;
}

// =============================================================================
// Normalization Functions
// =============================================================================

Tensor softmax(const Tensor& x, int axis) {
    // Handle negative axis
    int ndim = static_cast<int>(x.ndim());
    if (axis < 0) {
        axis = ndim + axis;
    }
    if (axis < 0 || axis >= ndim) {
        throw std::invalid_argument("softmax: invalid axis");
    }

    Tensor result = x.clone();

    // For 1D tensors, apply softmax across all elements
    if (x.ndim() == 1) {
        // Find max for numerical stability
        float max_val = result[0];
        for (Tensor::size_type i = 1; i < result.size(); ++i) {
            max_val = std::max(max_val, result[i]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (Tensor::size_type i = 0; i < result.size(); ++i) {
            result[i] = std::exp(result[i] - max_val);
            sum += result[i];
        }

        // Normalize
        for (Tensor::size_type i = 0; i < result.size(); ++i) {
            result[i] /= sum;
        }
        return result;
    }

    // For 2D tensors
    if (x.ndim() == 2) {
        Tensor::size_type rows = x.dim(0);
        Tensor::size_type cols = x.dim(1);

        if (axis == 1) {
            // Softmax along columns (per row)
            for (Tensor::size_type i = 0; i < rows; ++i) {
                // Find max in this row
                float max_val = result.at(i, 0);
                for (Tensor::size_type j = 1; j < cols; ++j) {
                    max_val = std::max(max_val, result.at(i, j));
                }

                // Compute exp(x - max) and sum
                float sum = 0.0f;
                for (Tensor::size_type j = 0; j < cols; ++j) {
                    result.at(i, j) = std::exp(result.at(i, j) - max_val);
                    sum += result.at(i, j);
                }

                // Normalize
                for (Tensor::size_type j = 0; j < cols; ++j) {
                    result.at(i, j) /= sum;
                }
            }
        } else {
            // Softmax along rows (per column)
            for (Tensor::size_type j = 0; j < cols; ++j) {
                // Find max in this column
                float max_val = result.at(0, j);
                for (Tensor::size_type i = 1; i < rows; ++i) {
                    max_val = std::max(max_val, result.at(i, j));
                }

                // Compute exp(x - max) and sum
                float sum = 0.0f;
                for (Tensor::size_type i = 0; i < rows; ++i) {
                    result.at(i, j) = std::exp(result.at(i, j) - max_val);
                    sum += result.at(i, j);
                }

                // Normalize
                for (Tensor::size_type i = 0; i < rows; ++i) {
                    result.at(i, j) /= sum;
                }
            }
        }
        return result;
    }

    // For 3D tensors (batch, seq, hidden)
    if (x.ndim() == 3) {
        Tensor::size_type batch = x.dim(0);
        Tensor::size_type seq = x.dim(1);
        Tensor::size_type hidden = x.dim(2);

        if (axis == 2) {
            // Softmax along last axis (most common for attention)
            for (Tensor::size_type b = 0; b < batch; ++b) {
                for (Tensor::size_type s = 0; s < seq; ++s) {
                    // Find max
                    float max_val = result.at(b, s, 0);
                    for (Tensor::size_type h = 1; h < hidden; ++h) {
                        max_val = std::max(max_val, result.at(b, s, h));
                    }

                    // Compute exp and sum
                    float sum = 0.0f;
                    for (Tensor::size_type h = 0; h < hidden; ++h) {
                        result.at(b, s, h) = std::exp(result.at(b, s, h) - max_val);
                        sum += result.at(b, s, h);
                    }

                    // Normalize
                    for (Tensor::size_type h = 0; h < hidden; ++h) {
                        result.at(b, s, h) /= sum;
                    }
                }
            }
        } else if (axis == 1) {
            // Softmax along sequence axis
            for (Tensor::size_type b = 0; b < batch; ++b) {
                for (Tensor::size_type h = 0; h < hidden; ++h) {
                    float max_val = result.at(b, 0, h);
                    for (Tensor::size_type s = 1; s < seq; ++s) {
                        max_val = std::max(max_val, result.at(b, s, h));
                    }

                    float sum = 0.0f;
                    for (Tensor::size_type s = 0; s < seq; ++s) {
                        result.at(b, s, h) = std::exp(result.at(b, s, h) - max_val);
                        sum += result.at(b, s, h);
                    }

                    for (Tensor::size_type s = 0; s < seq; ++s) {
                        result.at(b, s, h) /= sum;
                    }
                }
            }
        }
        return result;
    }

    throw std::invalid_argument("softmax: unsupported tensor dimension");
}

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                  float eps) {
    // Layer normalization normalizes over the last dimension
    // For each position, compute: gamma * (x - mean) / sqrt(var + eps) + beta

    if (x.ndim() < 1) {
        throw std::invalid_argument("layer_norm: input must be at least 1D");
    }

    Tensor::size_type last_dim = x.dim(x.ndim() - 1);

    // Verify gamma and beta shapes
    if (gamma.size() != last_dim || beta.size() != last_dim) {
        throw std::invalid_argument(
            "layer_norm: gamma and beta must have size matching last dimension");
    }

    Tensor result = x.clone();

    // Number of vectors to normalize (all dimensions except last)
    Tensor::size_type num_vectors = x.size() / last_dim;

    for (Tensor::size_type v = 0; v < num_vectors; ++v) {
        Tensor::size_type offset = v * last_dim;

        // Compute mean
        float mean = 0.0f;
        for (Tensor::size_type i = 0; i < last_dim; ++i) {
            mean += result[offset + i];
        }
        mean /= static_cast<float>(last_dim);

        // Compute variance
        float variance = 0.0f;
        for (Tensor::size_type i = 0; i < last_dim; ++i) {
            float diff = result[offset + i] - mean;
            variance += diff * diff;
        }
        variance /= static_cast<float>(last_dim);

        // Normalize and apply affine transform
        float inv_std = 1.0f / std::sqrt(variance + eps);
        for (Tensor::size_type i = 0; i < last_dim; ++i) {
            float normalized = (result[offset + i] - mean) * inv_std;
            result[offset + i] = gamma[i] * normalized + beta[i];
        }
    }

    return result;
}

// =============================================================================
// Reduction Operations
// =============================================================================

float sum(const Tensor& x) {
    float total = 0.0f;
    for (Tensor::size_type i = 0; i < x.size(); ++i) {
        total += x[i];
    }
    return total;
}

Tensor sum(const Tensor& x, int axis, bool keepdims) {
    int ndim = static_cast<int>(x.ndim());
    if (axis < 0) {
        axis = ndim + axis;
    }
    if (axis < 0 || axis >= ndim) {
        throw std::invalid_argument("sum: invalid axis");
    }

    // For 2D tensors
    if (x.ndim() == 2) {
        Tensor::size_type rows = x.dim(0);
        Tensor::size_type cols = x.dim(1);

        if (axis == 0) {
            // Sum along rows -> result shape: (cols,) or (1, cols)
            Tensor::Shape result_shape = keepdims ?
                Tensor::Shape{1, cols} : Tensor::Shape{cols};
            Tensor result(result_shape, 0.0f);

            for (Tensor::size_type j = 0; j < cols; ++j) {
                float total = 0.0f;
                for (Tensor::size_type i = 0; i < rows; ++i) {
                    total += x.at(i, j);
                }
                result[j] = total;
            }
            return result;
        } else {
            // Sum along cols -> result shape: (rows,) or (rows, 1)
            Tensor::Shape result_shape = keepdims ?
                Tensor::Shape{rows, 1} : Tensor::Shape{rows};
            Tensor result(result_shape, 0.0f);

            for (Tensor::size_type i = 0; i < rows; ++i) {
                float total = 0.0f;
                for (Tensor::size_type j = 0; j < cols; ++j) {
                    total += x.at(i, j);
                }
                result[i] = total;
            }
            return result;
        }
    }

    // For 1D tensors, sum reduces to scalar
    if (x.ndim() == 1) {
        Tensor result(keepdims ? Tensor::Shape{1} : Tensor::Shape{1});
        result[0] = sum(x);
        return result;
    }

    throw std::invalid_argument("sum along axis: unsupported tensor dimension");
}

float mean(const Tensor& x) {
    if (x.size() == 0) {
        return 0.0f;
    }
    return sum(x) / static_cast<float>(x.size());
}

Tensor mean(const Tensor& x, int axis, bool keepdims) {
    Tensor s = sum(x, axis, keepdims);
    Tensor::size_type count = x.dim(axis < 0 ? x.ndim() + axis : axis);
    scale_inplace(s, 1.0f / static_cast<float>(count));
    return s;
}

Tensor variance(const Tensor& x, int axis, bool keepdims) {
    // var(x) = mean((x - mean(x))^2)
    Tensor m = mean(x, axis, true); // Keep dims for broadcasting

    int ndim = static_cast<int>(x.ndim());
    if (axis < 0) {
        axis = ndim + axis;
    }

    // For 2D tensors
    if (x.ndim() == 2) {
        Tensor::size_type rows = x.dim(0);
        Tensor::size_type cols = x.dim(1);

        if (axis == 0) {
            Tensor::Shape result_shape = keepdims ?
                Tensor::Shape{1, cols} : Tensor::Shape{cols};
            Tensor result(result_shape, 0.0f);

            for (Tensor::size_type j = 0; j < cols; ++j) {
                float var_sum = 0.0f;
                float mean_val = m[j];
                for (Tensor::size_type i = 0; i < rows; ++i) {
                    float diff = x.at(i, j) - mean_val;
                    var_sum += diff * diff;
                }
                result[j] = var_sum / static_cast<float>(rows);
            }
            return result;
        } else {
            Tensor::Shape result_shape = keepdims ?
                Tensor::Shape{rows, 1} : Tensor::Shape{rows};
            Tensor result(result_shape, 0.0f);

            for (Tensor::size_type i = 0; i < rows; ++i) {
                float var_sum = 0.0f;
                float mean_val = m[i];
                for (Tensor::size_type j = 0; j < cols; ++j) {
                    float diff = x.at(i, j) - mean_val;
                    var_sum += diff * diff;
                }
                result[i] = var_sum / static_cast<float>(cols);
            }
            return result;
        }
    }

    throw std::invalid_argument("variance: unsupported tensor dimension");
}

float max(const Tensor& x) {
    if (x.size() == 0) {
        return -std::numeric_limits<float>::infinity();
    }
    float max_val = x[0];
    for (Tensor::size_type i = 1; i < x.size(); ++i) {
        max_val = std::max(max_val, x[i]);
    }
    return max_val;
}

float min(const Tensor& x) {
    if (x.size() == 0) {
        return std::numeric_limits<float>::infinity();
    }
    float min_val = x[0];
    for (Tensor::size_type i = 1; i < x.size(); ++i) {
        min_val = std::min(min_val, x[i]);
    }
    return min_val;
}

// =============================================================================
// Utility Operations
// =============================================================================

Tensor transpose(const Tensor& x) {
    if (x.ndim() != 2) {
        throw std::invalid_argument("transpose: only 2D tensors supported");
    }

    Tensor::size_type rows = x.dim(0);
    Tensor::size_type cols = x.dim(1);
    Tensor result(Tensor::Shape{cols, rows});

    for (Tensor::size_type i = 0; i < rows; ++i) {
        for (Tensor::size_type j = 0; j < cols; ++j) {
            result.at(j, i) = x.at(i, j);
        }
    }
    return result;
}

Tensor concatenate(const std::vector<Tensor>& tensors, int axis) {
    if (tensors.empty()) {
        return Tensor();
    }

    // For simplicity, only handle 1D and 2D concatenation along axis 0
    if (tensors[0].ndim() == 1 && axis == 0) {
        // Concatenate 1D tensors
        Tensor::size_type total_size = 0;
        for (const auto& t : tensors) {
            total_size += t.size();
        }

        Tensor result(Tensor::Shape{total_size});
        Tensor::size_type offset = 0;
        for (const auto& t : tensors) {
            for (Tensor::size_type i = 0; i < t.size(); ++i) {
                result[offset + i] = t[i];
            }
            offset += t.size();
        }
        return result;
    }

    if (tensors[0].ndim() == 2) {
        Tensor::size_type cols = tensors[0].dim(1);

        // Verify all tensors have same number of columns
        for (const auto& t : tensors) {
            if (t.dim(1) != cols) {
                throw std::invalid_argument(
                    "concatenate: all tensors must have same number of columns");
            }
        }

        if (axis == 0) {
            // Stack vertically
            Tensor::size_type total_rows = 0;
            for (const auto& t : tensors) {
                total_rows += t.dim(0);
            }

            Tensor result(Tensor::Shape{total_rows, cols});
            Tensor::size_type row_offset = 0;
            for (const auto& t : tensors) {
                for (Tensor::size_type i = 0; i < t.dim(0); ++i) {
                    for (Tensor::size_type j = 0; j < cols; ++j) {
                        result.at(row_offset + i, j) = t.at(i, j);
                    }
                }
                row_offset += t.dim(0);
            }
            return result;
        }
    }

    throw std::invalid_argument("concatenate: unsupported configuration");
}

std::vector<Tensor> split(const Tensor& x, int n_splits, int axis) {
    if (n_splits <= 0) {
        throw std::invalid_argument("split: n_splits must be positive");
    }

    int ndim = static_cast<int>(x.ndim());
    if (axis < 0) {
        axis = ndim + axis;
    }

    Tensor::size_type dim_size = x.dim(axis);
    if (dim_size % n_splits != 0) {
        throw std::invalid_argument(
            "split: dimension not evenly divisible by n_splits");
    }

    Tensor::size_type split_size = dim_size / n_splits;
    std::vector<Tensor> result;
    result.reserve(n_splits);

    if (x.ndim() == 1 && axis == 0) {
        for (int s = 0; s < n_splits; ++s) {
            Tensor part(Tensor::Shape{split_size});
            for (Tensor::size_type i = 0; i < split_size; ++i) {
                part[i] = x[s * split_size + i];
            }
            result.push_back(std::move(part));
        }
        return result;
    }

    if (x.ndim() == 2 && axis == 0) {
        Tensor::size_type cols = x.dim(1);
        for (int s = 0; s < n_splits; ++s) {
            result.push_back(x.slice(s * split_size, (s + 1) * split_size));
        }
        return result;
    }

    throw std::invalid_argument("split: unsupported configuration");
}

} // namespace math
} // namespace cpp_embedder
