#include "../include/math/tensor.hpp"
#include "../include/math/ops.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace cpp_embedder::math;

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;

    // Empty tensor
    Tensor empty;
    assert(empty.empty());
    assert(empty.ndim() == 0);

    // 1D tensor - use explicit Shape to avoid ambiguity with initializer_list<float>
    Tensor v1(Tensor::Shape{3});
    assert(v1.ndim() == 1);
    assert(v1.size() == 3);
    assert(v1.dim(0) == 3);

    // 2D tensor
    Tensor m1(Tensor::Shape{2, 3});
    assert(m1.ndim() == 2);
    assert(m1.size() == 6);
    assert(m1.dim(0) == 2);
    assert(m1.dim(1) == 3);

    // 3D tensor
    Tensor t1(Tensor::Shape{2, 3, 4});
    assert(t1.ndim() == 3);
    assert(t1.size() == 24);

    // From initializer list
    Tensor v2 = {1.0f, 2.0f, 3.0f};
    assert(v2.size() == 3);
    assert(v2[0] == 1.0f);
    assert(v2[1] == 2.0f);
    assert(v2[2] == 3.0f);

    // From nested initializer list
    Tensor m2 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    assert(m2.dim(0) == 2);
    assert(m2.dim(1) == 2);
    assert(m2.at(0, 0) == 1.0f);
    assert(m2.at(0, 1) == 2.0f);
    assert(m2.at(1, 0) == 3.0f);
    assert(m2.at(1, 1) == 4.0f);

    std::cout << "  Tensor creation: PASSED" << std::endl;
}

void test_matmul() {
    std::cout << "Testing matrix multiplication..." << std::endl;

    // 2x3 @ 3x2 -> 2x2
    Tensor a = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Tensor b = {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}};

    Tensor c = matmul(a, b);
    assert(c.dim(0) == 2);
    assert(c.dim(1) == 2);

    // Manual calculation:
    // c[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // c[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // c[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // c[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    assert(std::abs(c.at(0, 0) - 58.0f) < 1e-5f);
    assert(std::abs(c.at(0, 1) - 64.0f) < 1e-5f);
    assert(std::abs(c.at(1, 0) - 139.0f) < 1e-5f);
    assert(std::abs(c.at(1, 1) - 154.0f) < 1e-5f);

    // Matrix-vector multiplication
    Tensor v = {1.0f, 2.0f, 3.0f};
    Tensor mv = matmul(a, v);
    assert(mv.ndim() == 1);
    assert(mv.size() == 2);
    // mv[0] = 1*1 + 2*2 + 3*3 = 14
    // mv[1] = 4*1 + 5*2 + 6*3 = 32
    assert(std::abs(mv[0] - 14.0f) < 1e-5f);
    assert(std::abs(mv[1] - 32.0f) < 1e-5f);

    std::cout << "  Matrix multiplication: PASSED" << std::endl;
}

void test_add_scale() {
    std::cout << "Testing add and scale..." << std::endl;

    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {4.0f, 5.0f, 6.0f};

    Tensor c = add(a, b);
    assert(c[0] == 5.0f);
    assert(c[1] == 7.0f);
    assert(c[2] == 9.0f);

    Tensor d = scale(a, 2.0f);
    assert(d[0] == 2.0f);
    assert(d[1] == 4.0f);
    assert(d[2] == 6.0f);

    std::cout << "  Add and scale: PASSED" << std::endl;
}

void test_gelu() {
    std::cout << "Testing GELU activation..." << std::endl;

    Tensor x = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor y = gelu(x);

    // GELU(0) should be 0
    assert(std::abs(y[2]) < 1e-5f);

    // GELU is approximately linear for large positive values
    // GELU(x) ~ x for large x
    assert(y[4] > 1.9f && y[4] < 2.1f);

    // GELU(x) ~ 0 for large negative values
    assert(y[0] > -0.1f && y[0] < 0.1f);

    std::cout << "  GELU activation: PASSED" << std::endl;
}

void test_softmax() {
    std::cout << "Testing softmax..." << std::endl;

    // 1D softmax
    Tensor x = {1.0f, 2.0f, 3.0f};
    Tensor y = softmax(x);

    // Sum should be 1
    float s = y[0] + y[1] + y[2];
    assert(std::abs(s - 1.0f) < 1e-5f);

    // Values should be increasing since input was increasing
    assert(y[0] < y[1] && y[1] < y[2]);

    // 2D softmax along axis 1
    Tensor m = {{1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f}};
    Tensor sm = softmax(m, 1);

    // Each row should sum to 1
    float row0_sum = sm.at(0, 0) + sm.at(0, 1) + sm.at(0, 2);
    float row1_sum = sm.at(1, 0) + sm.at(1, 1) + sm.at(1, 2);
    assert(std::abs(row0_sum - 1.0f) < 1e-5f);
    assert(std::abs(row1_sum - 1.0f) < 1e-5f);

    // Second row should be uniform (all equal)
    assert(std::abs(sm.at(1, 0) - sm.at(1, 1)) < 1e-5f);
    assert(std::abs(sm.at(1, 1) - sm.at(1, 2)) < 1e-5f);

    std::cout << "  Softmax: PASSED" << std::endl;
}

void test_layer_norm() {
    std::cout << "Testing layer normalization..." << std::endl;

    // Simple 1D case with identity gamma and zero beta
    Tensor x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    Tensor gamma(Tensor::Shape{5}, 1.0f);  // Scale = 1
    Tensor beta(Tensor::Shape{5}, 0.0f);   // Shift = 0

    Tensor y = layer_norm(x, gamma, beta, 1e-5f);

    // Mean should be ~0 and variance ~1
    float mean_val = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        mean_val += y[i];
    }
    mean_val /= y.size();
    assert(std::abs(mean_val) < 1e-5f);

    float var_val = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) {
        var_val += y[i] * y[i];
    }
    var_val /= y.size();
    assert(std::abs(var_val - 1.0f) < 1e-4f);

    std::cout << "  Layer normalization: PASSED" << std::endl;
}

void test_reshape() {
    std::cout << "Testing reshape..." << std::endl;

    Tensor a(Tensor::Shape{2, 3}, 0.0f);
    for (size_t i = 0; i < 6; ++i) {
        a[i] = static_cast<float>(i);
    }

    // Reshape to 3x2
    Tensor b = a.reshape({3, 2});
    assert(b.dim(0) == 3);
    assert(b.dim(1) == 2);
    assert(b[0] == 0.0f);
    assert(b[5] == 5.0f);

    // Flatten
    Tensor c = a.flatten();
    assert(c.ndim() == 1);
    assert(c.size() == 6);

    std::cout << "  Reshape: PASSED" << std::endl;
}

void test_transpose() {
    std::cout << "Testing transpose..." << std::endl;

    Tensor a = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Tensor b = transpose(a);

    assert(b.dim(0) == 3);
    assert(b.dim(1) == 2);
    assert(b.at(0, 0) == 1.0f);
    assert(b.at(0, 1) == 4.0f);
    assert(b.at(1, 0) == 2.0f);
    assert(b.at(1, 1) == 5.0f);
    assert(b.at(2, 0) == 3.0f);
    assert(b.at(2, 1) == 6.0f);

    std::cout << "  Transpose: PASSED" << std::endl;
}

int main() {
    std::cout << "=== Math Primitives Test Suite ===" << std::endl << std::endl;

    test_tensor_creation();
    test_matmul();
    test_add_scale();
    test_gelu();
    test_softmax();
    test_layer_norm();
    test_reshape();
    test_transpose();

    std::cout << std::endl << "=== All tests PASSED ===" << std::endl;
    return 0;
}
