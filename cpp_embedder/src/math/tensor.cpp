#include "../../include/math/tensor.hpp"
#include <numeric>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace cpp_embedder {
namespace math {

// =============================================================================
// Static Helper Functions
// =============================================================================

Tensor::size_type Tensor::compute_size(const Shape& shape) {
    if (shape.empty()) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(),
                           size_type{1}, std::multiplies<size_type>{});
}

void Tensor::validate_shape(const Shape& shape) {
    for (auto dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Tensor dimensions cannot be zero");
        }
    }
}

// =============================================================================
// Constructors
// =============================================================================

Tensor::Tensor() : shape_{}, data_{} {}

Tensor::Tensor(const Shape& shape)
    : shape_(shape), data_(compute_size(shape), 0.0f) {
    if (!shape.empty()) {
        validate_shape(shape);
    }
}

Tensor::Tensor(const Shape& shape, value_type fill_value)
    : shape_(shape), data_(compute_size(shape), fill_value) {
    if (!shape.empty()) {
        validate_shape(shape);
    }
}

Tensor::Tensor(const Shape& shape, std::vector<value_type> data)
    : shape_(shape), data_(std::move(data)) {
    if (!shape.empty()) {
        validate_shape(shape);
    }
    if (data_.size() != compute_size(shape)) {
        throw std::invalid_argument(
            "Data size does not match shape: expected " +
            std::to_string(compute_size(shape)) + ", got " +
            std::to_string(data_.size()));
    }
}

Tensor::Tensor(std::initializer_list<value_type> values)
    : shape_{values.size()}, data_(values) {}

Tensor::Tensor(std::initializer_list<std::initializer_list<value_type>> values) {
    if (values.size() == 0) {
        shape_ = {};
        return;
    }

    size_type rows = values.size();
    size_type cols = values.begin()->size();

    // Verify all rows have same length
    for (const auto& row : values) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same length");
        }
    }

    shape_ = {rows, cols};
    data_.reserve(rows * cols);

    for (const auto& row : values) {
        for (auto val : row) {
            data_.push_back(val);
        }
    }
}

// =============================================================================
// Dimension Access
// =============================================================================

Tensor::size_type Tensor::dim(size_type axis) const {
    if (axis >= shape_.size()) {
        return 1; // Non-existent dimensions are treated as size 1
    }
    return shape_[axis];
}

// =============================================================================
// Element Access
// =============================================================================

Tensor::value_type& Tensor::operator[](size_type index) {
    return data_[index];
}

const Tensor::value_type& Tensor::operator[](size_type index) const {
    return data_[index];
}

Tensor::value_type& Tensor::at(size_type index) {
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor index out of range: " +
                                std::to_string(index) + " >= " +
                                std::to_string(data_.size()));
    }
    return data_[index];
}

const Tensor::value_type& Tensor::at(size_type index) const {
    if (index >= data_.size()) {
        throw std::out_of_range("Tensor index out of range: " +
                                std::to_string(index) + " >= " +
                                std::to_string(data_.size()));
    }
    return data_[index];
}

Tensor::value_type& Tensor::at(size_type row, size_type col) {
    if (ndim() != 2) {
        throw std::invalid_argument("2D indexing requires 2D tensor");
    }
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("2D index out of range");
    }
    return data_[row * shape_[1] + col];
}

const Tensor::value_type& Tensor::at(size_type row, size_type col) const {
    if (ndim() != 2) {
        throw std::invalid_argument("2D indexing requires 2D tensor");
    }
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("2D index out of range");
    }
    return data_[row * shape_[1] + col];
}

Tensor::value_type& Tensor::at(size_type depth, size_type row, size_type col) {
    if (ndim() != 3) {
        throw std::invalid_argument("3D indexing requires 3D tensor");
    }
    if (depth >= shape_[0] || row >= shape_[1] || col >= shape_[2]) {
        throw std::out_of_range("3D index out of range");
    }
    return data_[(depth * shape_[1] + row) * shape_[2] + col];
}

const Tensor::value_type& Tensor::at(size_type depth, size_type row, size_type col) const {
    if (ndim() != 3) {
        throw std::invalid_argument("3D indexing requires 3D tensor");
    }
    if (depth >= shape_[0] || row >= shape_[1] || col >= shape_[2]) {
        throw std::out_of_range("3D index out of range");
    }
    return data_[(depth * shape_[1] + row) * shape_[2] + col];
}

// =============================================================================
// Index Computation
// =============================================================================

Tensor::size_type Tensor::linear_index(size_type row, size_type col) const {
    return row * shape_[1] + col;
}

Tensor::size_type Tensor::linear_index(size_type depth, size_type row, size_type col) const {
    return (depth * shape_[1] + row) * shape_[2] + col;
}

// =============================================================================
// Reshape Operations
// =============================================================================

Tensor Tensor::reshape(const Shape& new_shape) const {
    Tensor result = *this;
    result.reshape_inplace(new_shape);
    return result;
}

void Tensor::reshape_inplace(const Shape& new_shape) {
    // Check for -1 (infer dimension)
    int infer_idx = -1;
    size_type known_size = 1;

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == static_cast<size_type>(-1)) {
            if (infer_idx != -1) {
                throw std::invalid_argument("Only one dimension can be inferred (-1)");
            }
            infer_idx = static_cast<int>(i);
        } else {
            known_size *= new_shape[i];
        }
    }

    Shape final_shape = new_shape;
    if (infer_idx >= 0) {
        if (data_.size() % known_size != 0) {
            throw std::invalid_argument("Cannot infer dimension: size not divisible");
        }
        final_shape[infer_idx] = data_.size() / known_size;
    }

    if (compute_size(final_shape) != data_.size()) {
        throw std::invalid_argument(
            "Cannot reshape: size mismatch. Current size: " +
            std::to_string(data_.size()) + ", new shape size: " +
            std::to_string(compute_size(final_shape)));
    }

    shape_ = final_shape;
}

Tensor Tensor::flatten() const {
    return Tensor({data_.size()}, data_);
}

// =============================================================================
// Slicing Operations
// =============================================================================

Tensor Tensor::slice(size_type start, size_type end) const {
    if (shape_.empty()) {
        throw std::invalid_argument("Cannot slice empty tensor");
    }
    if (start >= end || end > shape_[0]) {
        throw std::out_of_range("Invalid slice range");
    }

    // Calculate stride of first dimension
    size_type stride = 1;
    for (size_t i = 1; i < shape_.size(); ++i) {
        stride *= shape_[i];
    }

    // Create new shape with reduced first dimension
    Shape new_shape = shape_;
    new_shape[0] = end - start;

    // Copy data
    std::vector<value_type> new_data(
        data_.begin() + start * stride,
        data_.begin() + end * stride);

    return Tensor(new_shape, std::move(new_data));
}

Tensor Tensor::row(size_type index) const {
    if (shape_.empty() || index >= shape_[0]) {
        throw std::out_of_range("Row index out of range");
    }

    // Calculate stride
    size_type stride = 1;
    for (size_t i = 1; i < shape_.size(); ++i) {
        stride *= shape_[i];
    }

    // New shape: remove first dimension
    Shape new_shape(shape_.begin() + 1, shape_.end());
    if (new_shape.empty()) {
        new_shape = {1}; // Scalar becomes 1D with size 1
    }

    std::vector<value_type> new_data(
        data_.begin() + index * stride,
        data_.begin() + (index + 1) * stride);

    return Tensor(new_shape, std::move(new_data));
}

// =============================================================================
// Clone and Fill
// =============================================================================

Tensor Tensor::clone() const {
    return *this; // Value semantics: copy constructor does deep copy
}

void Tensor::fill(value_type value) {
    std::fill(data_.begin(), data_.end(), value);
}

// =============================================================================
// Shape Comparison
// =============================================================================

bool Tensor::shapes_equal(const Tensor& a, const Tensor& b) {
    return a.shape_ == b.shape_;
}

bool Tensor::shapes_broadcastable(const Tensor& a, const Tensor& b) {
    // For simplicity, only exact matches are broadcastable in this implementation
    return shapes_equal(a, b);
}

// =============================================================================
// Debug Output
// =============================================================================

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], data=[";

    // Print first few and last few elements for large tensors
    const size_type max_show = 6;
    if (data_.size() <= max_show * 2) {
        for (size_t i = 0; i < data_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << std::fixed << std::setprecision(4) << data_[i];
        }
    } else {
        for (size_t i = 0; i < max_show; ++i) {
            if (i > 0) oss << ", ";
            oss << std::fixed << std::setprecision(4) << data_[i];
        }
        oss << ", ...";
        for (size_t i = data_.size() - max_show; i < data_.size(); ++i) {
            oss << ", " << std::fixed << std::setprecision(4) << data_[i];
        }
    }
    oss << "])";
    return oss.str();
}

} // namespace math
} // namespace cpp_embedder
