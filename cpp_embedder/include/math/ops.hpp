#ifndef CPP_EMBEDDER_MATH_OPS_HPP
#define CPP_EMBEDDER_MATH_OPS_HPP

#include "tensor.hpp"

namespace cpp_embedder {
namespace math {

// =============================================================================
// Matrix/Vector Operations
// =============================================================================

/// Matrix multiplication: C = A @ B
/// A: (M, K), B: (K, N) -> C: (M, N)
/// Also supports batched: A: (batch, M, K), B: (K, N) -> C: (batch, M, N)
/// And vector-matrix: A: (K,), B: (K, N) -> C: (N,)
/// And matrix-vector: A: (M, K), B: (K,) -> C: (M,)
Tensor matmul(const Tensor& a, const Tensor& b);

/// Element-wise addition: C = A + B
/// Shapes must match exactly (no broadcasting for simplicity)
Tensor add(const Tensor& a, const Tensor& b);

/// In-place element-wise addition: A += B
void add_inplace(Tensor& a, const Tensor& b);

/// Scalar multiplication: B = A * scalar
Tensor scale(const Tensor& a, float scalar);

/// In-place scalar multiplication: A *= scalar
void scale_inplace(Tensor& a, float scalar);

/// Element-wise multiplication (Hadamard product): C = A * B
Tensor multiply(const Tensor& a, const Tensor& b);

/// In-place element-wise multiplication: A *= B
void multiply_inplace(Tensor& a, const Tensor& b);

// =============================================================================
// Activation Functions
// =============================================================================

/// GELU activation (Gaussian Error Linear Unit)
/// Approximate formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// Reference: https://arxiv.org/abs/1606.08415
Tensor gelu(const Tensor& x);

/// In-place GELU activation
void gelu_inplace(Tensor& x);

/// ReLU activation: max(0, x)
Tensor relu(const Tensor& x);

/// In-place ReLU activation
void relu_inplace(Tensor& x);

/// Sigmoid activation: 1 / (1 + exp(-x))
Tensor sigmoid(const Tensor& x);

/// Tanh activation
Tensor tanh_activation(const Tensor& x);

// =============================================================================
// Normalization
// =============================================================================

/// Softmax along specified axis
/// softmax(x)_i = exp(x_i) / sum(exp(x_j)) for all j along axis
/// Default axis=-1 means last axis
Tensor softmax(const Tensor& x, int axis = -1);

/// Layer normalization
/// Normalizes over the last dimension: (x - mean) / sqrt(var + eps)
/// Then applies affine transform: gamma * normalized + beta
/// gamma and beta should have shape matching the last dimension of x
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                  float eps = 1e-5f);

// =============================================================================
// Reduction Operations
// =============================================================================

/// Sum all elements
float sum(const Tensor& x);

/// Sum along axis
Tensor sum(const Tensor& x, int axis, bool keepdims = false);

/// Mean of all elements
float mean(const Tensor& x);

/// Mean along axis
Tensor mean(const Tensor& x, int axis, bool keepdims = false);

/// Variance along axis
Tensor variance(const Tensor& x, int axis, bool keepdims = false);

/// Max of all elements
float max(const Tensor& x);

/// Min of all elements
float min(const Tensor& x);

// =============================================================================
// Utility Operations
// =============================================================================

/// Transpose 2D tensor
Tensor transpose(const Tensor& x);

/// Concatenate tensors along axis
Tensor concatenate(const std::vector<Tensor>& tensors, int axis = 0);

/// Split tensor along axis into n equal parts
std::vector<Tensor> split(const Tensor& x, int n_splits, int axis = 0);

} // namespace math
} // namespace cpp_embedder

#endif // CPP_EMBEDDER_MATH_OPS_HPP
