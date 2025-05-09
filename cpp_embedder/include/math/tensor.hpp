#ifndef CPP_EMBEDDER_MATH_TENSOR_HPP
#define CPP_EMBEDDER_MATH_TENSOR_HPP

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>
#include <string>

namespace cpp_embedder {
namespace math {

/// Simple tensor class with value semantics.
/// Supports 1D, 2D, and 3D tensors with row-major storage.
/// Uses float32 exclusively for simplicity.
class Tensor {
public:
    // Type aliases
    using value_type = float;
    using size_type = std::size_t;
    using Shape = std::vector<size_type>;

    // Default constructor: creates empty tensor
    Tensor();

    // Construct tensor with given shape, initialized to zero
    explicit Tensor(const Shape& shape);

    // Construct tensor with given shape and fill value
    Tensor(const Shape& shape, value_type fill_value);

    // Construct tensor from shape and data vector
    Tensor(const Shape& shape, std::vector<value_type> data);

    // Construct 1D tensor from initializer list
    Tensor(std::initializer_list<value_type> values);

    // Construct 2D tensor from nested initializer list
    Tensor(std::initializer_list<std::initializer_list<value_type>> values);

    // Copy and move constructors/assignment (defaulted for value semantics)
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Shape information
    const Shape& shape() const noexcept { return shape_; }
    size_type ndim() const noexcept { return shape_.size(); }
    size_type size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    // Dimension accessors (return 1 for non-existent dimensions)
    size_type dim(size_type axis) const;

    // Element access (flat index)
    value_type& operator[](size_type index);
    const value_type& operator[](size_type index) const;

    // Element access with bounds checking (flat index)
    value_type& at(size_type index);
    const value_type& at(size_type index) const;

    // 2D element access (row, col)
    value_type& at(size_type row, size_type col);
    const value_type& at(size_type row, size_type col) const;

    // 3D element access (depth, row, col)
    value_type& at(size_type depth, size_type row, size_type col);
    const value_type& at(size_type depth, size_type row, size_type col) const;

    // Raw data access
    value_type* data() noexcept { return data_.data(); }
    const value_type* data() const noexcept { return data_.data(); }
    std::vector<value_type>& storage() noexcept { return data_; }
    const std::vector<value_type>& storage() const noexcept { return data_; }

    // Reshape operations
    Tensor reshape(const Shape& new_shape) const;
    void reshape_inplace(const Shape& new_shape);

    // Flatten to 1D
    Tensor flatten() const;

    // Slice along first axis: returns tensor with reduced first dimension
    // For 2D tensor: slice(start, end) returns rows [start, end)
    Tensor slice(size_type start, size_type end) const;

    // Get single row/slice along first axis
    Tensor row(size_type index) const;

    // Clone/copy
    Tensor clone() const;

    // Fill with value
    void fill(value_type value);

    // Utility: compute linear index from multi-dimensional indices
    size_type linear_index(size_type i) const { return i; }
    size_type linear_index(size_type row, size_type col) const;
    size_type linear_index(size_type depth, size_type row, size_type col) const;

    // Shape validation
    static bool shapes_equal(const Tensor& a, const Tensor& b);
    static bool shapes_broadcastable(const Tensor& a, const Tensor& b);

    // Debug string representation
    std::string to_string() const;

private:
    Shape shape_;
    std::vector<value_type> data_;

    // Compute total size from shape
    static size_type compute_size(const Shape& shape);

    // Validate shape (no zero dimensions unless empty)
    static void validate_shape(const Shape& shape);
};

} // namespace math
} // namespace cpp_embedder

#endif // CPP_EMBEDDER_MATH_TENSOR_HPP
