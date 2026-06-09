#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "slimt/Tensor.hh"

namespace slimt::qmm {

constexpr float kInt8Maxf = 127.0F;

namespace detail {

enum class Provider {
  Ruy,  //
};

template <enum Provider>
Tensor affine(const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name = "");

template <enum Provider>
Tensor prepare_bias(const Tensor& W, const Tensor& b, float a_quant,
                    float b_quant, const std::string& name = "");

template <enum Provider>
Tensor affine_with_prepared_bias(const Tensor& x, const Tensor& W,
                                 const Tensor& prepared_bias, float a_quant,
                                 float b_quant,
                                 const std::string& name = "");

template <enum Provider>
Tensor select_columns(const Tensor& W, const std::vector<uint32_t>& indices,
                      const std::string& name = "");

template <enum Provider>
Tensor dot(const Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

template <enum Provider>
void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows);
template <enum Provider>
void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols);

// Drops the calling thread's cached packs of heap-owned (standalone) weight
// tensors. Must run before those tensors are freed — the cache is keyed by
// data pointer, and a later allocation reusing the address would hit a stale
// pack.
template <enum Provider>
void clear_standalone_pack_cache();

}  // namespace detail

Tensor affine(const Tensor& x, const Tensor& W, const Tensor& b, float a_quant,
              float b_quant, const std::string& name = "");

Tensor prepare_bias(const Tensor& W, const Tensor& b, float a_quant,
                    float b_quant, const std::string& name = "");

Tensor affine_with_prepared_bias(const Tensor& x, const Tensor& W,
                                 const Tensor& prepared_bias, float a_quant,
                                 float b_quant,
                                 const std::string& name = "");

Tensor select_columns(const Tensor& W, const std::vector<uint32_t>& indices,
                      const std::string& name = "");

Tensor dot(const Tensor& x, const Tensor& W, float a_quant, float b_quant,
           const std::string& name = "");

void prepare_weight_transposed(const float* weights, int8_t* prepared,
                               float quantization_multiplier, size_t cols,
                               size_t rows);
void prepare_weight_quantized_transposed(const int8_t* input, int8_t* output,
                                         size_t rows, size_t cols);

void clear_standalone_pack_cache();

}  // namespace slimt::qmm
