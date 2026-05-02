#include "ruy/ruy.h"

#include <cmath>

namespace slimt::qmm::detail {

using Index = uint64_t;

void quantize(const float* input, float scale, Index rows, Index width,
              int8_t* output) {
  const Index size = rows * width;
  for (size_t i = 0; i < size; i++) {
    // Round to nearest after multiplying with scale.
    float value = roundf(scale * input[i]);

    // Since float can store bigger values, we threshold anything that's gone
    // higher and can't fit in int8.
    value = std::max<float>(-kInt8Maxf, value);
    value = std::min<float>(kInt8Maxf, value);

    // Finally a static cast.
    output[i] = static_cast<int8_t>(value);
  };
}

template <class Scalar>
void transpose(const Scalar* input, Index rows, Index cols, Scalar* output) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      output[j * rows + i] = input[i * cols + j];
    }
  }
}

void unquantize(const int32_t* input, float unquant_multiplier, Index rows_A,
                Index cols_B, float* output) {
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_B; j++) {
      Index idx = i * cols_B + j;
      output[idx] = (input[idx] * unquant_multiplier);
    }
  }
}

void unquantizeAddBias(const int32_t* input, const float* input_bias_prepared,
                       float unquant_multiplier, Index rows_A, Index cols_B,
                       float* output) {
  for (size_t i = 0; i < rows_A; i++) {
    for (size_t j = 0; j < cols_B; j++) {
      Index idx = i * cols_B + j;
      output[idx] = (input[idx] * unquant_multiplier) + input_bias_prepared[j];
    }
  }
}

// Ruy.
template <>
Tensor affine<Provider::Ruy>(const Tensor& x, const Tensor& W, const Tensor& b,
                             float a_quant, float b_quant,
                             const std::string& name) {
  const Tensor& A = x;  // NOLINT
  const Tensor& B = W;  // NOLINT
  const Tensor& bias = b;

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = B_rows;

  (void)name;
  // Prepare A: Quantize from f32 -> i8
  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT

  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  thread_local ruy::Context context;
  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(A_rows, width, ruy::Order::kRowMajor,
                        lhs.mutable_layout());
  lhs.set_data(prepared_A.data<int8_t>());

  // PrepareB: ?
  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, B_cols, ruy::Order::kColMajor,
                        rhs.mutable_layout());
  rhs.set_data(W.data<int8_t>());
  // Cache the RHS pack only when W's data pointer is model-lifetime stable.
  // Mmap-backed model weights satisfy this (Tensor::load takes a View,
  // never owns/frees the buffer — `standalone() == false`). Heap-owned
  // tensors (e.g. SelectedAffine.W from select_columns) get freed at
  // decode end; their address can be reused by the next decode's allocator
  // call, and Ruy's pointer-keyed cache would then return a stale pack
  // (garbage logits) plus grow unboundedly across decodes (OOM).
  if (!W.standalone()) {
    rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
  }

  // PrepareBias: ?
  // Actualyl there is no need.
  const Tensor& prepared_bias = bias;

  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(A_rows, B_cols, ruy::Order::kRowMajor,
                        dst.mutable_layout());

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, out_shape, name + "_out");  // NOLINT
  dst.set_data(AB.data<int32_t>());

  // Multiply C = AB;
  // When Dst is int32, mul_params is unused.
  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  // Unquantizes, then adds bias in a single statement on the output.
  Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantizeAddBias(AB.data<int32_t>(), prepared_bias.data<float>(),
                            unquant_multiplier, A_rows, B_cols,
                            y.data<float>());
  return y;
}

template <>
Tensor select_columns<Provider::Ruy>(const Tensor& W,
                                     const std::vector<uint32_t>& indices,
                                     const std::string& name) {
  // B is stored col-major as `width × cols`, so each output column lives in a
  // contiguous `width`-byte slab. Memcpy the requested columns into a fresh
  // `width × |indices|` block; the result has the same layout, so any
  // downstream affine call can treat it as a normal weight tensor.
  const Tensor& B = W;                // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT
  size_t width = B_rows;

  Tensor selected_B(Type::i8, Shape({width, indices.size()}),  // NOLINT
                    name.empty() ? "selected_B" : name);

  const auto* B_data = B.data<int8_t>();      // NOLINT
  auto* sB_data = selected_B.data<int8_t>();  // NOLINT
  for (size_t c = 0; c < indices.size(); ++c) {
    std::memcpy(&sB_data[c * width], &B_data[indices[c] * width], width);
  }
  return selected_B;
}

template <>
Tensor dot<Provider::Ruy>(const Tensor& x, const Tensor& W, float a_quant,
                          float b_quant, const std::string& name) {
  const Tensor& A = x;  // NOLINT
  const Tensor& B = W;  // NOLINT

  size_t A_cols = A.dim(-1);          // NOLINT
  size_t B_cols = B.dim(-1);          // NOLINT
  size_t A_rows = A.size() / A_cols;  // NOLINT
  size_t B_rows = B.size() / B_cols;  // NOLINT

  size_t width = B_rows;

  (void)name;
  // Prepare A: Quantize from f32 -> i8
  Tensor prepared_A(Type::i8, x.shape(), "prepared_A");  // NOLINT

  detail::quantize(x.data<float>(), a_quant, A_rows, A_cols,
                   prepared_A.data<int8_t>());

  thread_local ruy::Context context;
  ruy::Matrix<std::int8_t> lhs;
  ruy::MakeSimpleLayout(A_rows, width, ruy::Order::kRowMajor,
                        lhs.mutable_layout());
  lhs.set_data(prepared_A.data<int8_t>());

  // PrepareB: ?
  ruy::Matrix<std::int8_t> rhs;
  ruy::MakeSimpleLayout(width, B_cols, ruy::Order::kColMajor,
                        rhs.mutable_layout());
  rhs.set_data(W.data<int8_t>());
  rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);

  // PrepareBias: ?
  // Actualyl there is no need.
  ruy::Matrix<std::int32_t> dst;
  ruy::MakeSimpleLayout(A_rows, B_cols, ruy::Order::kRowMajor,
                        dst.mutable_layout());

  Shape out_shape = x.shape();
  out_shape.set_dim(-1, B_cols);
  Tensor AB(Type::i32, out_shape, name + "_out");  // NOLINT
  dst.set_data(AB.data<int32_t>());

  // Multiply C = AB;
  // When Dst is int32, mul_params is unused.
  ruy::MulParams<std::int32_t, std::int32_t> mul_params;
  ruy::Mul(lhs, rhs, mul_params, &context, &dst);

  // Unquantizes, then adds bias in a single statement on the output.
  Tensor y(Type::f32, out_shape, name + "_out");  // NOLINT
  float unquant_multiplier = 1.0F / (a_quant * b_quant);
  detail::unquantize(AB.data<int32_t>(), unquant_multiplier, A_rows, B_cols,
                     y.data<float>());
  return y;
}

template <>
void prepare_weight_transposed<Provider::Ruy>(const float* weights,
                                              int8_t* prepared,
                                              float quantization_multiplier,
                                              size_t cols, size_t rows) {
  detail::quantize(weights, quantization_multiplier, cols, rows, prepared);
}

template <>
void prepare_weight_quantized_transposed<Provider::Ruy>(const int8_t* input,
                                                        int8_t* output,
                                                        size_t rows,
                                                        size_t cols) {
  std::memcpy(output, input,
              /*count=*/sizeof(int8_t) * (rows * cols));
}

// The "prepared bias" optimization is intgemm-specific: it folds the
// shift-quantization compensation (column sums of W * a_quant_shift) into
// the bias once so subsequent affine calls can skip the per-call
// compensation pass. Ruy uses signed-signed int8 multiplication and applies
// the bias as a post-process, so no preparation is needed — pass the bias
// through unchanged and route the "with prepared bias" entry point to the
// regular one. Without these stubs, Modules.cc refers to undefined
// `prepare_bias<Ruy>` / `affine_with_prepared_bias<Ruy>`, producing a
// libslimt.a that silently fails to dlopen on ARM.
template <>
Tensor prepare_bias<Provider::Ruy>(const Tensor& W, const Tensor& b,
                                   float a_quant, float b_quant,
                                   const std::string& name) {
  (void)W;
  (void)a_quant;
  (void)b_quant;
  return b.clone(name.empty() ? "prepared_bias" : name);
}

template <>
Tensor affine_with_prepared_bias<Provider::Ruy>(const Tensor& x,
                                                const Tensor& W,
                                                const Tensor& prepared_bias,
                                                float a_quant, float b_quant,
                                                const std::string& name) {
  return affine<Provider::Ruy>(x, W, prepared_bias, a_quant, b_quant, name);
}
}  // namespace slimt::qmm::detail
