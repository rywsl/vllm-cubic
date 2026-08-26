// SPDX-License-Identifier: Apache-2.0
// Cubic W4-W8/A8 GEMV with one group-local carrier LUT per output.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace {

enum class CubicMetadataFormat : int {
  Expanded = 0,
  Int8Int4 = 1,
  E5M9Curve2 = 2,
  Int8Int4Codebook = 3
};

__device__ __forceinline__ int cubic_dp4a_w4_w8(int a, int b, int c) {
  int out;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
  return out;
}

__device__ __forceinline__ uint4 cubic_ldcs_u32x4(
    const uint4* __restrict__ address) {
  uint4 value;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address));
  return value;
}

template <CubicMetadataFormat Format>
__device__ __forceinline__ void cubic_load_metadata(
    const void* __restrict__ primary, const void* __restrict__ secondary,
    const void* __restrict__ tertiary,
    const float* __restrict__ scale_global,
    const float* __restrict__ a_global, const float* __restrict__ b_global,
    int64_t metadata_index, int64_t table_index, float& scale, float& a,
    float& b) {
  if constexpr (Format == CubicMetadataFormat::Expanded) {
    scale = static_cast<const float*>(primary)[metadata_index];
    a = __half2float(static_cast<const __half*>(secondary)[metadata_index]);
    b = __half2float(static_cast<const __half*>(tertiary)[metadata_index]);
  } else if constexpr (Format == CubicMetadataFormat::E5M9Curve2) {
    const uint16_t metadata =
        static_cast<const uint16_t*>(primary)[metadata_index];
    const int curve_id = metadata >> 14;
    const uint16_t scale_bits = (metadata & 0x3fffu) << 1;
    scale = __half2float(__ushort_as_half(scale_bits));
    const int64_t curve_index = table_index * 4 + curve_id;
    a = __half2float(static_cast<const __half*>(secondary)[curve_index]);
    b = __half2float(static_cast<const __half*>(tertiary)[curve_index]);
  } else {
    const int8_t scale_code =
        static_cast<const int8_t*>(primary)[metadata_index];
    const uint8_t packed_ab =
        static_cast<const uint8_t*>(secondary)[metadata_index];
    int a_code = packed_ab & 0xf;
    int b_code = packed_ab >> 4;
    if (a_code >= 8) a_code -= 16;
    if (b_code >= 8) b_code -= 16;
    scale = static_cast<float>(scale_code) * scale_global[table_index];
    a = __half2float(
        __float2half_rn(1.0f + static_cast<float>(a_code) * a_global[table_index]));
    b = __half2float(
        __float2half_rn(static_cast<float>(b_code) * b_global[table_index]));
  }
}

template <int Bits, int GroupSize, int ThreadsPerOutput,
          CubicMetadataFormat Format>
__global__ void cubic_w4_w8_compact_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const void* __restrict__ primary,
    const void* __restrict__ secondary, const void* __restrict__ tertiary,
    const float* __restrict__ scale_global,
    const float* __restrict__ a_global, const float* __restrict__ b_global,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int n, int k, int num_groups,
    int packed_k, int group_out, int top_k, int route_ctas,
    int num_valid_tokens, int output_partition_size, int num_partitions,
    bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int code_blocks_per_group = GroupSize / 32;
  constexpr int blocks_per_thread = code_blocks_per_group / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize * Bits / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr int levels = 1 << (Bits - 1);
  constexpr bool use_full_signed_lut = Bits <= 6;
  constexpr int lut_entries = use_full_signed_lut ? (1 << Bits) : levels;
  constexpr unsigned mask = 0xffffffffu;
  static_assert(Bits >= 4 && Bits <= 8);
  static_assert(code_blocks_per_group % ThreadsPerOutput == 0);
  extern __shared__ int8_t carrier_lut[];

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  const bool valid_output = out_channel < n;
  const int num_routes = *num_routes_ptr;
  const int input_words_per_row = k / 4;

  for (int route = blockIdx.y; route < num_routes; route += route_ctas) {
    const int token_id = token_ids[route];
    const int expert = expert_ids[route];
    if (token_id < 0 || token_id >= num_valid_tokens || expert < 0) continue;
    const int input_row = token_id / top_k;
    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* input_words = reinterpret_cast<const int*>(input) +
                             static_cast<int64_t>(input_row) * input_words_per_row;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base =
        (static_cast<int64_t>(expert) * (n / group_out) +
         out_channel / group_out) * num_groups;
    const int partition = min(out_channel / output_partition_size,
                              num_partitions - 1);
    const int64_t table_index =
        static_cast<int64_t>(expert) * num_partitions + partition;
    if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
      const int block_partition = min(
          (blockIdx.x * outputs_per_block) / output_partition_size,
          num_partitions - 1);
      const int64_t block_table_index =
          static_cast<int64_t>(expert) * num_partitions + block_partition;
      constexpr int table_entries = 256 * levels;
      const auto* table = static_cast<const int8_t*>(tertiary) +
                          block_table_index * table_entries;
      for (int entry = threadIdx.x; entry < table_entries;
           entry += kThreads) {
        carrier_lut[entry] = table[entry];
      }
      __syncthreads();
    }
    float accumulator = 0.0f;

    for (int group = 0; group < num_groups; ++group) {
      const int64_t meta = meta_base + group;
      float weight_scale = 0.0f;
      float a = 0.0f;
      float b = 0.0f;
      uint8_t packed_ab = 0;
      if (valid_output) {
        cubic_load_metadata<Format>(
            primary, secondary, tertiary, scale_global, a_global, b_global,
            meta, table_index, weight_scale, a, b);
        if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
          packed_ab = static_cast<const uint8_t*>(secondary)[meta];
        }
      }
      if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
        // The carrier is reconstructed directly for each consumed code below.
      } else {
        const float c = 1.0f - a - b;
        for (int level = lane; level < levels; level += ThreadsPerOutput) {
          const float t = static_cast<float>(level) / (levels - 1);
          const float normalized = t * (a + t * (b + t * c));
          const int8_t carrier =
              static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
          carrier_lut[output_in_block * lut_entries + level] = carrier;
          if constexpr (use_full_signed_lut) {
            if (level == 0)
              carrier_lut[output_in_block * lut_entries + levels] = 0;
            else
              carrier_lut[output_in_block * lut_entries + (1 << Bits) - level] =
                  -carrier;
          }
        }
        if constexpr (ThreadsPerOutput > 1) __syncwarp();
      }

      int dot = 0;
#pragma unroll
      for (int block_index = 0; block_index < blocks_per_thread;
           ++block_index) {
        const int code_block = lane + block_index * ThreadsPerOutput;
        unsigned packed_words[Bits];
#pragma unroll
        for (int word_index = 0; word_index < Bits; ++word_index) {
          packed_words[word_index] =
              valid_output ? weight_words[group * words_per_group +
                                          code_block * Bits + word_index]
                           : 0;
        }
#pragma unroll
        for (int quad = 0; quad < 8; ++quad) {
          unsigned carrier_word = 0;
#pragma unroll
          for (int code_in_quad = 0; code_in_quad < 4; ++code_in_quad) {
            constexpr unsigned code_mask = (1u << Bits) - 1;
            const int bit = (quad * 4 + code_in_quad) * Bits;
            const int word_index = bit >> 5;
            const int shift = bit & 31;
            unsigned raw = packed_words[word_index] >> shift;
            if constexpr (Bits > 4) {
              if (shift + Bits > 32)
                raw |= packed_words[word_index + 1] << (32 - shift);
            }
            raw &= code_mask;
            int carrier;
            if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
              constexpr int sign_bit = 1 << (Bits - 1);
              int signed_code = raw >= sign_bit
                                    ? static_cast<int>(raw) - (1 << Bits)
                                    : static_cast<int>(raw);
              if (signed_code == -sign_bit) signed_code = 0;
              const int magnitude = abs(signed_code);
              carrier = carrier_lut[static_cast<int>(packed_ab) * levels +
                                    magnitude];
              if (signed_code < 0) carrier = -carrier;
            } else if constexpr (use_full_signed_lut) {
              carrier = carrier_lut[output_in_block * lut_entries + raw];
            } else {
              constexpr int sign_bit = 1 << (Bits - 1);
              int signed_code = raw >= sign_bit
                                    ? static_cast<int>(raw) - (1 << Bits)
                                    : static_cast<int>(raw);
              if (signed_code == -sign_bit) signed_code = 0;
              carrier = carrier_lut[output_in_block * levels + abs(signed_code)];
              if (signed_code < 0) carrier = -carrier;
            }
            carrier_word |= (static_cast<unsigned>(carrier) & 0xff)
                            << (code_in_quad * 8);
          }
          const int input_index =
              group * input_words_per_group + code_block * 8 + quad;
          dot = cubic_dp4a_w4_w8(carrier_word, input_words[input_index], dot);
        }
      }
      for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1)
        dot += __shfl_down_sync(mask, dot, delta, ThreadsPerOutput);
      if (lane == 0 && valid_output) {
        accumulator += static_cast<float>(dot) * input_scale[input_row] *
                       weight_scale * (1.0f / 127.0f);
      }
      if constexpr (ThreadsPerOutput > 1) __syncwarp();
    }
    if (lane == 0 && valid_output) {
      if (multiply_routed_weight) accumulator *= topk_weights[token_id];
      output[static_cast<int64_t>(token_id) * n + out_channel] =
          __float2bfloat16(accumulator);
    }
  }
}

template <int Bits, int GroupSize, int RoutesPerBlock,
          CubicMetadataFormat Format>
__global__ void cubic_w4_w8_compact_group_parallel_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const void* __restrict__ primary,
    const void* __restrict__ secondary, const void* __restrict__ tertiary,
    const float* __restrict__ scale_global,
    const float* __restrict__ a_global, const float* __restrict__ b_global,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int n, int k, int num_groups,
    int packed_k, int top_k, int route_ctas, int num_valid_tokens,
    int output_partition_size, int num_partitions,
    bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int kLanesPerOutput = 8;
  constexpr int outputs_per_block = kThreads / kLanesPerOutput;
  constexpr int levels = 1 << (Bits - 1);
  constexpr bool use_full_signed_lut = Bits <= 6;
  constexpr int lut_entries = use_full_signed_lut ? (1 << Bits) : levels;
  constexpr bool use_shared_compact_codebook =
      Format == CubicMetadataFormat::Int8Int4 && Bits == 6;
  constexpr int code_blocks_per_group = GroupSize / 32;
  constexpr int words_per_group = GroupSize * Bits / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr unsigned mask = 0xffffffffu;
  extern __shared__ int8_t carrier_lut[];

  const int lane = threadIdx.x & (kLanesPerOutput - 1);
  const int output_in_block = threadIdx.x / kLanesPerOutput;
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  const bool valid_output = out_channel < n;
  const int num_route_blocks =
      (*num_routes_ptr + RoutesPerBlock - 1) / RoutesPerBlock;
  const int input_words_per_row = k / 4;
  int8_t* local_lut = carrier_lut + threadIdx.x * lut_entries;

  for (int route_block = blockIdx.y; route_block < num_route_blocks;
       route_block += route_ctas) {
    int token_ids_block[RoutesPerBlock];
    int input_rows[RoutesPerBlock];
    const int* input_words[RoutesPerBlock];
    bool valid_routes[RoutesPerBlock];
    float partials[RoutesPerBlock] = {};
#pragma unroll
    for (int route = 0; route < RoutesPerBlock; ++route) {
      const int route_index = route_block * RoutesPerBlock + route;
      const int token_id = token_ids[route_index];
      token_ids_block[route] = token_id;
      valid_routes[route] =
          static_cast<unsigned>(token_id) <
          static_cast<unsigned>(num_valid_tokens);
      input_rows[route] = valid_routes[route] ? token_id / top_k : 0;
      input_words[route] = reinterpret_cast<const int*>(input) +
                           static_cast<int64_t>(input_rows[route]) *
                               input_words_per_row;
    }
    const int expert = expert_ids[route_block];
    if (expert < 0) continue;
    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base =
        (static_cast<int64_t>(expert) * n + out_channel) * num_groups;
    const int partition = min(out_channel / output_partition_size,
                              num_partitions - 1);
    const int64_t table_index =
        static_cast<int64_t>(expert) * num_partitions + partition;
    if constexpr (use_shared_compact_codebook) {
      constexpr int table_entries = 256 * levels;
      const float a_step = a_global[table_index];
      const float b_step = b_global[table_index];
      for (int entry = threadIdx.x; entry < table_entries;
           entry += kThreads) {
        const int packed_ab = entry / levels;
        const int level = entry - packed_ab * levels;
        int a_code = packed_ab & 0xf;
        int b_code = packed_ab >> 4;
        if (a_code >= 8) a_code -= 16;
        if (b_code >= 8) b_code -= 16;
        const float a = __half2float(
            __float2half_rn(1.0f + static_cast<float>(a_code) * a_step));
        const float b = __half2float(
            __float2half_rn(static_cast<float>(b_code) * b_step));
        const float t = static_cast<float>(level) / (levels - 1);
        const float normalized =
            t * (a + t * (b + t * (1.0f - a - b)));
        carrier_lut[entry] =
            static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
      }
      __syncthreads();
    } else if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
      constexpr int table_entries = 256 * levels;
      const auto* table = static_cast<const int8_t*>(tertiary) +
                          table_index * table_entries;
      for (int entry = threadIdx.x; entry < table_entries;
           entry += kThreads) {
        carrier_lut[entry] = table[entry];
      }
      __syncthreads();
    }
    for (int group = lane; group < num_groups; group += kLanesPerOutput) {
      const int64_t meta = meta_base + group;
      float weight_scale = 0.0f;
      float a = 0.0f;
      float b = 0.0f;
      uint8_t packed_ab = 0;
      if (valid_output) {
        if constexpr (use_shared_compact_codebook) {
          const int8_t scale_code = static_cast<const int8_t*>(primary)[meta];
          weight_scale =
              static_cast<float>(scale_code) * scale_global[table_index];
          packed_ab = static_cast<const uint8_t*>(secondary)[meta];
        } else {
          cubic_load_metadata<Format>(
              primary, secondary, tertiary, scale_global, a_global, b_global,
              meta, table_index, weight_scale, a, b);
          if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
            packed_ab = static_cast<const uint8_t*>(secondary)[meta];
          }
        }
      }
      const float c = 1.0f - a - b;
      if constexpr (Format != CubicMetadataFormat::Int8Int4Codebook &&
                    !use_shared_compact_codebook) {
#pragma unroll
        for (int level = 0; level < levels; ++level) {
          const float t = static_cast<float>(level) / (levels - 1);
          const float normalized = t * (a + t * (b + t * c));
          const int8_t carrier =
              static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
          local_lut[level] = carrier;
          if constexpr (use_full_signed_lut) {
            if (level == 0)
              local_lut[levels] = 0;
            else
              local_lut[(1 << Bits) - level] = -carrier;
          }
        }
      }

      int dots[RoutesPerBlock] = {};
#pragma unroll
      for (int code_block = 0; code_block < code_blocks_per_group;
           ++code_block) {
        unsigned packed_words[Bits];
#pragma unroll
        for (int word_index = 0; word_index < Bits; ++word_index) {
          packed_words[word_index] =
              valid_output ? weight_words[group * words_per_group +
                                            code_block * Bits + word_index]
                           : 0;
        }
#pragma unroll
        for (int quad = 0; quad < 8; ++quad) {
          unsigned carrier_word = 0;
#pragma unroll
          for (int code_in_quad = 0; code_in_quad < 4; ++code_in_quad) {
            constexpr unsigned code_mask = (1u << Bits) - 1;
            const int bit = (quad * 4 + code_in_quad) * Bits;
            const int word_index = bit >> 5;
            const int shift = bit & 31;
            unsigned raw = packed_words[word_index] >> shift;
            if constexpr (Bits > 4) {
              if (shift + Bits > 32)
                raw |= packed_words[word_index + 1] << (32 - shift);
            }
            raw &= code_mask;
            int carrier;
            if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook ||
                          use_shared_compact_codebook) {
              constexpr int sign_bit = 1 << (Bits - 1);
              int signed_code = raw >= sign_bit
                                    ? static_cast<int>(raw) - (1 << Bits)
                                    : static_cast<int>(raw);
              if (signed_code == -sign_bit) signed_code = 0;
              carrier = carrier_lut[static_cast<int>(packed_ab) * levels +
                                    abs(signed_code)];
              if (signed_code < 0) carrier = -carrier;
            } else if constexpr (use_full_signed_lut) {
              carrier = local_lut[raw];
            } else {
              constexpr int sign_bit = 1 << (Bits - 1);
              int signed_code = raw >= sign_bit
                                    ? static_cast<int>(raw) - (1 << Bits)
                                    : static_cast<int>(raw);
              if (signed_code == -sign_bit) signed_code = 0;
              carrier = local_lut[abs(signed_code)];
              if (signed_code < 0) carrier = -carrier;
            }
            carrier_word |= (static_cast<unsigned>(carrier) & 0xff)
                            << (code_in_quad * 8);
          }
          const int input_index = group * input_words_per_group +
                                  code_block * 8 + quad;
#pragma unroll
          for (int route = 0; route < RoutesPerBlock; ++route) {
            if (valid_routes[route]) {
              dots[route] = cubic_dp4a_w4_w8(
                  carrier_word, input_words[route][input_index], dots[route]);
            }
          }
        }
      }
#pragma unroll
      for (int route = 0; route < RoutesPerBlock; ++route) {
        partials[route] += static_cast<float>(dots[route]) * weight_scale;
      }
    }
#pragma unroll
    for (int route = 0; route < RoutesPerBlock; ++route) {
      for (int delta = kLanesPerOutput / 2; delta > 0; delta >>= 1) {
        partials[route] += __shfl_down_sync(
            mask, partials[route], delta, kLanesPerOutput);
      }
    }
    if (lane == 0 && valid_output) {
#pragma unroll
      for (int route = 0; route < RoutesPerBlock; ++route) {
        if (!valid_routes[route]) continue;
        const int token_id = token_ids_block[route];
        float result = partials[route] * input_scale[input_rows[route]] *
                       (1.0f / 127.0f);
        if (multiply_routed_weight) result *= topk_weights[token_id];
        output[static_cast<int64_t>(token_id) * n + out_channel] =
            __float2bfloat16(result);
      }
    }
    if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook ||
                  use_shared_compact_codebook) {
      __syncthreads();
    }
  }
}

template <int Bits, int GroupSize, CubicMetadataFormat Format>
__global__ void cubic_w4_w8_compact_a8_fused_sum_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const void* __restrict__ primary,
    const void* __restrict__ secondary, const void* __restrict__ tertiary,
    const float* __restrict__ scale_global,
    const float* __restrict__ a_global, const float* __restrict__ b_global,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ topk_ids, const int* __restrict__ expert_map,
    int num_tokens, int top_k, int n, int k, int num_groups, int packed_k,
    int group_out, int output_partition_size, int num_partitions,
    bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int kLanesPerOutput = 8;
  constexpr int outputs_per_block = kThreads / kLanesPerOutput;
  constexpr int levels = 1 << (Bits - 1);
  constexpr bool use_full_signed_lut = Bits <= 6;
  constexpr int lut_entries = use_full_signed_lut ? (1 << Bits) : levels;
  constexpr int code_blocks_per_group = GroupSize / 32;
  constexpr int words_per_group = GroupSize * Bits / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr unsigned mask = 0xffffffffu;
  extern __shared__ int8_t carrier_lut[];

  const int lane = threadIdx.x & (kLanesPerOutput - 1);
  const int output_in_block = threadIdx.x / kLanesPerOutput;
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  const bool valid_output = out_channel < n;
  const int token = blockIdx.y;
  if (token >= num_tokens) return;
  const int input_words_per_row = k / 4;
  int8_t* local_lut = carrier_lut + threadIdx.x * lut_entries;
  float route_sum = 0.0f;

  for (int route_slot = 0; route_slot < top_k; ++route_slot) {
    const int route = token * top_k + route_slot;
    const int global_expert = topk_ids[route];
    const int expert = expert_map[global_expert];
    if (expert < 0) continue;
    const int* input_words = reinterpret_cast<const int*>(input) +
                             static_cast<int64_t>(route) * input_words_per_row;
    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base =
        (static_cast<int64_t>(expert) * (n / group_out) +
         out_channel / group_out) * num_groups;
    const int partition = min(out_channel / output_partition_size,
                              num_partitions - 1);
    const int64_t table_index =
        static_cast<int64_t>(expert) * num_partitions + partition;
    float partial = 0.0f;

    for (int group = lane; group < num_groups; group += kLanesPerOutput) {
      const int64_t meta = meta_base + group;
      float weight_scale = 0.0f;
      float a = 0.0f;
      float b = 0.0f;
      if (valid_output) {
        cubic_load_metadata<Format>(
            primary, secondary, tertiary, scale_global, a_global, b_global,
            meta, table_index, weight_scale, a, b);
      }
      const float c = 1.0f - a - b;
#pragma unroll
      for (int level = 0; level < levels; ++level) {
        const float t = static_cast<float>(level) / (levels - 1);
        const float normalized = t * (a + t * (b + t * c));
        const int8_t carrier =
            static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
        local_lut[level] = carrier;
        if constexpr (use_full_signed_lut) {
          if (level == 0)
            local_lut[levels] = 0;
          else
            local_lut[(1 << Bits) - level] = -carrier;
        }
      }

      int dot = 0;
#pragma unroll
      for (int code_block = 0; code_block < code_blocks_per_group;
           ++code_block) {
        unsigned packed_words[Bits];
#pragma unroll
        for (int word_index = 0; word_index < Bits; ++word_index) {
          packed_words[word_index] =
              valid_output ? weight_words[group * words_per_group +
                                            code_block * Bits + word_index]
                           : 0;
        }
#pragma unroll
        for (int quad = 0; quad < 8; ++quad) {
          unsigned carrier_word = 0;
#pragma unroll
          for (int code_in_quad = 0; code_in_quad < 4; ++code_in_quad) {
            constexpr unsigned code_mask = (1u << Bits) - 1;
            const int bit = (quad * 4 + code_in_quad) * Bits;
            const int word_index = bit >> 5;
            const int shift = bit & 31;
            unsigned raw = packed_words[word_index] >> shift;
            if constexpr (Bits > 4) {
              if (shift + Bits > 32)
                raw |= packed_words[word_index + 1] << (32 - shift);
            }
            raw &= code_mask;
            int carrier;
            if constexpr (use_full_signed_lut) {
              carrier = local_lut[raw];
            } else {
              constexpr int sign_bit = 1 << (Bits - 1);
              int signed_code = raw >= sign_bit
                                    ? static_cast<int>(raw) - (1 << Bits)
                                    : static_cast<int>(raw);
              if (signed_code == -sign_bit) signed_code = 0;
              carrier = local_lut[abs(signed_code)];
              if (signed_code < 0) carrier = -carrier;
            }
            carrier_word |= (static_cast<unsigned>(carrier) & 0xff)
                            << (code_in_quad * 8);
          }
          const int input_index = group * input_words_per_group +
                                  code_block * 8 + quad;
          dot = cubic_dp4a_w4_w8(carrier_word, input_words[input_index], dot);
        }
      }
      partial += static_cast<float>(dot) * weight_scale;
    }
    for (int delta = kLanesPerOutput / 2; delta > 0; delta >>= 1) {
      partial += __shfl_down_sync(mask, partial, delta, kLanesPerOutput);
    }
    if (lane == 0 && valid_output) {
      float result = partial * input_scale[route] * (1.0f / 127.0f);
      if (multiply_routed_weight) result *= topk_weights[route];
      route_sum += __bfloat162float(__float2bfloat16(result));
    }
  }
  if (lane == 0 && valid_output) {
    output[static_cast<int64_t>(token) * n + out_channel] =
        __float2bfloat16(route_sum);
  }
}

template <int Bits, int GroupSize, int ThreadsPerOutput, bool GroupwiseScale>
__global__ void cubic_w4_w8_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    const __half* __restrict__ cubic_a, const __half* __restrict__ cubic_b,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int n, int k, int num_groups,
    int packed_k, int group_out, int top_k, int route_ctas,
    bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int code_blocks_per_group = GroupSize / 32;
  constexpr int blocks_per_thread = code_blocks_per_group / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize * Bits / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr int levels = 1 << (Bits - 1);
  constexpr bool use_full_signed_lut = Bits <= 6;
  constexpr int lut_entries = use_full_signed_lut ? (1 << Bits) : levels;
  constexpr unsigned mask = 0xffffffffu;
  static_assert(Bits >= 4 && Bits <= 8);
  static_assert(code_blocks_per_group % ThreadsPerOutput == 0);
  extern __shared__ int8_t carrier_lut[];

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  const bool valid_output = out_channel < n;
  const int num_routes = *num_routes_ptr;
  const int input_words_per_row = k / 4;

  for (int route = blockIdx.y; route < num_routes; route += route_ctas) {
    const int token_id = token_ids[route];
    const int expert = expert_ids[route];
    const int input_row = token_id / top_k;
    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* input_words =
        reinterpret_cast<const int*>(input) +
        static_cast<int64_t>(input_row) * input_words_per_row;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base =
        (static_cast<int64_t>(expert) * (n / group_out) +
         out_channel / group_out) *
        num_groups;
    float accumulator = 0.0f;

    for (int group = 0; group < num_groups; ++group) {
      const int64_t meta = meta_base + group;
      float group_activation_scale = 0.0f;
      if constexpr (GroupwiseScale) {
        if constexpr (Bits == 4) {
          // W4 has enough output subgroups per warp for one shared load to
          // amortize the shuffle.  For W5-W8 the wider inner loop makes the
          // shuffle a net loss; only the accumulator lane needs the scale.
          if ((threadIdx.x & 31) == 0) {
            group_activation_scale =
                input_scale[static_cast<int64_t>(input_row) * num_groups +
                            group];
          }
          group_activation_scale = __shfl_sync(mask, group_activation_scale, 0);
        } else if (lane == 0) {
          group_activation_scale =
              input_scale[static_cast<int64_t>(input_row) * num_groups + group];
        }
      }
      const float a = valid_output ? __half2float(cubic_a[meta]) : 0.0f;
      const float b = valid_output ? __half2float(cubic_b[meta]) : 0.0f;
      const float c = 1.0f - a - b;
      for (int level = lane; level < levels; level += ThreadsPerOutput) {
        const float t = static_cast<float>(level) / (levels - 1);
        const float normalized = t * (a + t * (b + t * c));
        const int8_t carrier =
            static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
        carrier_lut[output_in_block * lut_entries + level] = carrier;
        if constexpr (use_full_signed_lut) {
          if (level == 0)
            carrier_lut[output_in_block * lut_entries + levels] = 0;
          else
            carrier_lut[output_in_block * lut_entries + (1 << Bits) - level] =
                -carrier;
        }
      }
      // An output subgroup never crosses a warp boundary, and its LUT slice is
      // private.  A warp fence is sufficient; unrelated outputs need not wait.
      __syncwarp();

      int dot = 0;
#pragma unroll
      for (int block_index = 0; block_index < blocks_per_thread;
           ++block_index) {
        const int code_block = lane + block_index * ThreadsPerOutput;
        unsigned packed_words[Bits];
#pragma unroll
        for (int word_index = 0; word_index < Bits; ++word_index) {
          packed_words[word_index] =
              valid_output ? weight_words[group * words_per_group +
                                          code_block * Bits + word_index]
                           : 0;
        }
#pragma unroll
        for (int quad = 0; quad < 8; ++quad) {
          unsigned carrier_word = 0;
#pragma unroll
          for (int code_in_quad = 0; code_in_quad < 4; ++code_in_quad) {
            constexpr unsigned code_mask = (1u << Bits) - 1;
            const int bit = (quad * 4 + code_in_quad) * Bits;
            const int word_index = bit >> 5;
            const int shift = bit & 31;
            unsigned raw = packed_words[word_index] >> shift;
            if constexpr (Bits > 4) {
              if (shift + Bits > 32)
                raw |= packed_words[word_index + 1] << (32 - shift);
            }
            raw &= code_mask;
            int carrier;
            if constexpr (use_full_signed_lut) {
              carrier = carrier_lut[output_in_block * lut_entries + raw];
            } else {
              constexpr int sign_bit = 1 << (Bits - 1);
              int signed_code = raw >= sign_bit
                                    ? static_cast<int>(raw) - (1 << Bits)
                                    : static_cast<int>(raw);
              if (signed_code == -sign_bit) signed_code = 0;
              const int magnitude = abs(signed_code);
              carrier = carrier_lut[output_in_block * levels + magnitude];
              if (signed_code < 0) carrier = -carrier;
            }
            carrier_word |= (static_cast<unsigned>(carrier) & 0xff)
                            << (code_in_quad * 8);
          }
          const int input_index =
              group * input_words_per_group + code_block * 8 + quad;
          dot = cubic_dp4a_w4_w8(carrier_word, input_words[input_index], dot);
        }
      }
      for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1)
        dot += __shfl_down_sync(mask, dot, delta, ThreadsPerOutput);
      if (lane == 0 && valid_output) {
        const float act_scale =
            GroupwiseScale ? group_activation_scale : input_scale[input_row];
        accumulator += static_cast<float>(dot) * act_scale *
                       weight_scale[meta] * (1.0f / 127.0f);
      }
      __syncwarp();
    }
    if (lane == 0 && valid_output) {
      if (multiply_routed_weight) accumulator *= topk_weights[token_id];
      output[static_cast<int64_t>(token_id) * n + out_channel] =
          __float2bfloat16(accumulator);
    }
  }
}

template <int Bits, int GroupSize, int ThreadsPerOutput>
__global__ void cubic_curve2_shared_lut_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ metadata,
    const __half* __restrict__ curve_a, const __half* __restrict__ curve_b,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int n, int k, int num_groups,
    int packed_k, int top_k, int route_ctas, int num_valid_tokens,
    int output_partition_size, int num_partitions,
    bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int code_blocks_per_group = GroupSize / 32;
  constexpr int blocks_per_thread = code_blocks_per_group / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize * Bits / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr int levels = 1 << (Bits - 1);
  constexpr bool use_full_signed_lut = Bits <= 6;
  constexpr int lut_entries = use_full_signed_lut ? (1 << Bits) : levels;
  constexpr unsigned mask = 0xffffffffu;
  static_assert(Bits >= 4 && Bits <= 8);
  static_assert(code_blocks_per_group % ThreadsPerOutput == 0);
  __shared__ int8_t carrier_lut[4 * lut_entries];

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  const bool valid_output = out_channel < n;
  const int num_routes = *num_routes_ptr;
  const int input_words_per_row = k / 4;

  for (int route = blockIdx.y; route < num_routes; route += route_ctas) {
    const int token_id = token_ids[route];
    const int expert = expert_ids[route];
    if (token_id < 0 || token_id >= num_valid_tokens || expert < 0) continue;
    const int input_row = token_id / top_k;
    const int block_first_output = blockIdx.x * outputs_per_block;
    const int partition =
        min(block_first_output / output_partition_size, num_partitions - 1);
    const int64_t table_index =
        static_cast<int64_t>(expert) * num_partitions + partition;
    for (int index = threadIdx.x; index < 4 * levels; index += kThreads) {
      const int curve = index / levels;
      const int level = index - curve * levels;
      const float a = __half2float(curve_a[table_index * 4 + curve]);
      const float b = __half2float(curve_b[table_index * 4 + curve]);
      const float t = static_cast<float>(level) / (levels - 1);
      const float normalized = t * (a + t * (b + t * (1.0f - a - b)));
      const int8_t carrier =
          static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
      carrier_lut[curve * lut_entries + level] = carrier;
      if constexpr (use_full_signed_lut) {
        if (level == 0)
          carrier_lut[curve * lut_entries + levels] = 0;
        else
          carrier_lut[curve * lut_entries + (1 << Bits) - level] = -carrier;
      }
    }
    __syncthreads();

    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* input_words = reinterpret_cast<const int*>(input) +
                             static_cast<int64_t>(input_row) *
                                 input_words_per_row;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base =
        (static_cast<int64_t>(expert) * n + out_channel) * num_groups;
    float accumulator = 0.0f;
    for (int group = 0; group < num_groups; ++group) {
      const uint16_t encoded =
          valid_output ? metadata[meta_base + group] : 0;
      const int curve = encoded >> 14;
      const uint16_t scale_bits = (encoded & 0x3fffu) << 1;
      const float weight_scale = __half2float(__ushort_as_half(scale_bits));
      int dot = 0;
#pragma unroll
      for (int block_index = 0; block_index < blocks_per_thread;
           ++block_index) {
        const int code_block = lane + block_index * ThreadsPerOutput;
        unsigned packed_words[Bits];
#pragma unroll
        for (int word_index = 0; word_index < Bits; ++word_index) {
          packed_words[word_index] =
              valid_output ? weight_words[group * words_per_group +
                                          code_block * Bits + word_index]
                           : 0;
        }
#pragma unroll
        for (int quad = 0; quad < 8; ++quad) {
          unsigned carrier_word = 0;
#pragma unroll
          for (int code_in_quad = 0; code_in_quad < 4; ++code_in_quad) {
            constexpr unsigned code_mask = (1u << Bits) - 1;
            const int bit = (quad * 4 + code_in_quad) * Bits;
            const int word_index = bit >> 5;
            const int shift = bit & 31;
            unsigned raw = packed_words[word_index] >> shift;
            if constexpr (Bits > 4) {
              if (shift + Bits > 32)
                raw |= packed_words[word_index + 1] << (32 - shift);
            }
            raw &= code_mask;
            int carrier;
            if constexpr (use_full_signed_lut) {
              carrier = carrier_lut[curve * lut_entries + raw];
            } else {
              constexpr int sign_bit = 1 << (Bits - 1);
              int signed_code = raw >= sign_bit
                                    ? static_cast<int>(raw) - (1 << Bits)
                                    : static_cast<int>(raw);
              if (signed_code == -sign_bit) signed_code = 0;
              carrier = carrier_lut[curve * levels + abs(signed_code)];
              if (signed_code < 0) carrier = -carrier;
            }
            carrier_word |= (static_cast<unsigned>(carrier) & 0xff)
                            << (code_in_quad * 8);
          }
          const int input_index =
              group * input_words_per_group + code_block * 8 + quad;
          dot = cubic_dp4a_w4_w8(carrier_word, input_words[input_index], dot);
        }
      }
      for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1)
        dot += __shfl_down_sync(mask, dot, delta, ThreadsPerOutput);
      if (lane == 0 && valid_output) {
        accumulator += static_cast<float>(dot) * input_scale[input_row] *
                       weight_scale * (1.0f / 127.0f);
      }
    }
    if (lane == 0 && valid_output) {
      if (multiply_routed_weight) accumulator *= topk_weights[token_id];
      output[static_cast<int64_t>(token_id) * n + out_channel] =
          __float2bfloat16(accumulator);
    }
    __syncthreads();
  }
}

template <int Bits, int GroupSize, int ThreadsPerOutput, bool Pair,
          bool GroupwiseScale>
__device__ __forceinline__ void cubic_w4_w8_compute_route_block(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    const __half* __restrict__ cubic_a, const __half* __restrict__ cubic_b,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    int8_t* carrier_lut, int output_in_block, int lane, int out_channel,
    int token0, int token1, int expert, int n, int k, int num_groups,
    int packed_k, int group_out, int top_k, bool multiply_routed_weight) {
  constexpr int code_blocks_per_group = GroupSize / 32;
  constexpr int blocks_per_thread = code_blocks_per_group / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize * Bits / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr int levels = 1 << (Bits - 1);
  constexpr bool use_full_signed_lut = Bits <= 6;
  constexpr int lut_entries = use_full_signed_lut ? (1 << Bits) : levels;
  constexpr unsigned mask = 0xffffffffu;
  static_assert(Bits >= 4 && Bits <= 8);
  static_assert(code_blocks_per_group % ThreadsPerOutput == 0);
  const bool valid_output = out_channel < n;
  const int input_words_per_row = k / 4;
  const int row0 = token0 / top_k;
  const int row1 = Pair ? token1 / top_k : 0;
  const int64_t expert_output = static_cast<int64_t>(expert) * n + out_channel;
  const int* input0 = reinterpret_cast<const int*>(input) +
                      static_cast<int64_t>(row0) * input_words_per_row;
  const int* input1 = reinterpret_cast<const int*>(input) +
                      static_cast<int64_t>(row1) * input_words_per_row;
  const int* weight_words =
      reinterpret_cast<const int*>(weight + expert_output * packed_k);
  const int64_t meta_base =
      (static_cast<int64_t>(expert) * (n / group_out) +
       out_channel / group_out) *
      num_groups;
  const float per_token_scale0 = GroupwiseScale ? 0.0f : input_scale[row0];
  const float per_token_scale1 =
      GroupwiseScale || !Pair ? 0.0f : input_scale[row1];
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;

  for (int group = 0; group < num_groups; ++group) {
    const int64_t meta = meta_base + group;
    float group_scale0 = per_token_scale0;
    float group_scale1 = per_token_scale1;
    if constexpr (GroupwiseScale) {
      if constexpr (Bits == 4) {
        if ((threadIdx.x & 31) == 0) {
          group_scale0 =
              input_scale[static_cast<int64_t>(row0) * num_groups + group];
          if constexpr (Pair) {
            group_scale1 =
                input_scale[static_cast<int64_t>(row1) * num_groups + group];
          }
        }
        group_scale0 = __shfl_sync(mask, group_scale0, 0);
        if constexpr (Pair) {
          group_scale1 = __shfl_sync(mask, group_scale1, 0);
        }
      } else if (lane == 0) {
        group_scale0 =
            input_scale[static_cast<int64_t>(row0) * num_groups + group];
        if constexpr (Pair) {
          group_scale1 =
              input_scale[static_cast<int64_t>(row1) * num_groups + group];
        }
      }
    }
    const float a = valid_output ? __half2float(cubic_a[meta]) : 0.0f;
    const float b = valid_output ? __half2float(cubic_b[meta]) : 0.0f;
    const float c = 1.0f - a - b;
    for (int level = lane; level < levels; level += ThreadsPerOutput) {
      const float t = static_cast<float>(level) / (levels - 1);
      const float normalized = t * (a + t * (b + t * c));
      const int8_t carrier =
          static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
      carrier_lut[output_in_block * lut_entries + level] = carrier;
      if constexpr (use_full_signed_lut) {
        if (level == 0)
          carrier_lut[output_in_block * lut_entries + levels] = 0;
        else
          carrier_lut[output_in_block * lut_entries + (1 << Bits) - level] =
              -carrier;
      }
    }
    __syncwarp();

    int dot0 = 0;
    int dot1 = 0;
#pragma unroll
    for (int block_index = 0; block_index < blocks_per_thread; ++block_index) {
      const int code_block = lane + block_index * ThreadsPerOutput;
      unsigned packed_words[Bits];
#pragma unroll
      for (int word_index = 0; word_index < Bits; ++word_index) {
        packed_words[word_index] =
            valid_output ? weight_words[group * words_per_group +
                                        code_block * Bits + word_index]
                         : 0;
      }
#pragma unroll
      for (int quad = 0; quad < 8; ++quad) {
        unsigned carrier_word = 0;
#pragma unroll
        for (int code_in_quad = 0; code_in_quad < 4; ++code_in_quad) {
          constexpr unsigned code_mask = (1u << Bits) - 1;
          const int bit = (quad * 4 + code_in_quad) * Bits;
          const int word_index = bit >> 5;
          const int shift = bit & 31;
          unsigned raw = packed_words[word_index] >> shift;
          if constexpr (Bits > 4) {
            if (shift + Bits > 32)
              raw |= packed_words[word_index + 1] << (32 - shift);
          }
          raw &= code_mask;
          int carrier;
          if constexpr (use_full_signed_lut) {
            carrier = carrier_lut[output_in_block * lut_entries + raw];
          } else {
            constexpr int sign_bit = 1 << (Bits - 1);
            int signed_code = raw >= sign_bit
                                  ? static_cast<int>(raw) - (1 << Bits)
                                  : static_cast<int>(raw);
            if (signed_code == -sign_bit) signed_code = 0;
            const int magnitude = abs(signed_code);
            carrier = carrier_lut[output_in_block * levels + magnitude];
            if (signed_code < 0) carrier = -carrier;
          }
          carrier_word |= (static_cast<unsigned>(carrier) & 0xff)
                          << (code_in_quad * 8);
        }
        const int input_index =
            group * input_words_per_group + code_block * 8 + quad;
        dot0 = cubic_dp4a_w4_w8(carrier_word, input0[input_index], dot0);
        if constexpr (Pair) {
          dot1 = cubic_dp4a_w4_w8(carrier_word, input1[input_index], dot1);
        }
      }
    }
    for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1) {
      dot0 += __shfl_down_sync(mask, dot0, delta, ThreadsPerOutput);
      if constexpr (Pair) {
        dot1 += __shfl_down_sync(mask, dot1, delta, ThreadsPerOutput);
      }
    }
    if (lane == 0 && valid_output) {
      const float ws = weight_scale[meta];
      accumulator0 +=
          static_cast<float>(dot0) * group_scale0 * ws * (1.0f / 127.0f);
      if constexpr (Pair) {
        accumulator1 +=
            static_cast<float>(dot1) * group_scale1 * ws * (1.0f / 127.0f);
      }
    }
    __syncwarp();
  }
  if (lane == 0 && valid_output) {
    if (multiply_routed_weight) accumulator0 *= topk_weights[token0];
    output[static_cast<int64_t>(token0) * n + out_channel] =
        __float2bfloat16(accumulator0);
    if constexpr (Pair) {
      if (multiply_routed_weight) accumulator1 *= topk_weights[token1];
      output[static_cast<int64_t>(token1) * n + out_channel] =
          __float2bfloat16(accumulator1);
    }
  }
}

template <int Bits, int GroupSize, int ThreadsPerOutput, bool GroupwiseScale>
__global__ void cubic_w4_w8_grouped2_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const float* __restrict__ weight_scale,
    const __half* __restrict__ cubic_a, const __half* __restrict__ cubic_b,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int num_valid_tokens, int n, int k,
    int num_groups, int packed_k, int group_out, int top_k, int route_ctas,
    bool multiply_routed_weight) {
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  extern __shared__ int8_t carrier_lut[];

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out_channel = blockIdx.x * outputs_per_block + output_in_block;
  const int num_route_blocks = (*num_routes_ptr + 1) / 2;

  for (int route_block = blockIdx.y; route_block < num_route_blocks;
       route_block += route_ctas) {
    const int token0 = token_ids[route_block * 2];
    const int token1 = token_ids[route_block * 2 + 1];
    const bool valid0 =
        static_cast<unsigned>(token0) < static_cast<unsigned>(num_valid_tokens);
    const bool valid1 =
        static_cast<unsigned>(token1) < static_cast<unsigned>(num_valid_tokens);
    const int expert = expert_ids[route_block];
    if (expert < 0 || (!valid0 && !valid1)) continue;
    const int single_token = valid0 ? token0 : token1;
    if (valid0 && valid1) {
      cubic_w4_w8_compute_route_block<Bits, GroupSize, ThreadsPerOutput, true,
                                      GroupwiseScale>(
          input, input_scale, weight, weight_scale, cubic_a, cubic_b, output,
          topk_weights, carrier_lut, output_in_block, lane, out_channel, token0,
          token1, expert, n, k, num_groups, packed_k, group_out, top_k,
          multiply_routed_weight);
    } else {
      cubic_w4_w8_compute_route_block<Bits, GroupSize, ThreadsPerOutput, false,
                                      GroupwiseScale>(
          input, input_scale, weight, weight_scale, cubic_a, cubic_b, output,
          topk_weights, carrier_lut, output_in_block, lane, out_channel,
          single_token, 0, expert, n, k, num_groups, packed_k, group_out, top_k,
          multiply_routed_weight);
    }
  }
}

template <int Bits, int GroupSize, int ThreadsPerOutput, int RoutesPerBlock,
          bool GroupwiseScale, CubicMetadataFormat Format,
          bool UsePairLut = false>
__global__ __launch_bounds__(128, 8) void cubic_w4_w8_grouped_a8_gemv_kernel(
    const int8_t* __restrict__ input, const float* __restrict__ input_scale,
    const uint8_t* __restrict__ weight, const void* __restrict__ primary,
    const void* __restrict__ secondary, const void* __restrict__ tertiary,
    const float* __restrict__ scale_global,
    const float* __restrict__ a_global, const float* __restrict__ b_global,
    __nv_bfloat16* __restrict__ output, const float* __restrict__ topk_weights,
    const int* __restrict__ token_ids, const int* __restrict__ expert_ids,
    const int* __restrict__ num_routes_ptr, int num_valid_tokens, int n, int k,
    int num_groups, int packed_k, int group_out, int top_k, int route_ctas,
    int output_partition_size, int num_partitions,
    bool multiply_routed_weight) {
  constexpr int routes_per_block = RoutesPerBlock;
  constexpr int kThreads = 128;
  constexpr int outputs_per_block = kThreads / ThreadsPerOutput;
  constexpr int code_blocks_per_group = GroupSize / 32;
  constexpr int blocks_per_thread = code_blocks_per_group / ThreadsPerOutput;
  constexpr int words_per_group = GroupSize * Bits / 32;
  constexpr int input_words_per_group = GroupSize / 4;
  constexpr int levels = 1 << (Bits - 1);
  constexpr bool use_full_signed_lut = Bits <= 6;
  constexpr int lut_entries = use_full_signed_lut ? (1 << Bits) : levels;
  constexpr int lut_bytes = outputs_per_block * lut_entries;
  constexpr int pair_lut_bytes = UsePairLut ? 256 * sizeof(uint16_t) : 0;
  static_assert(!UsePairLut || Bits == 4);
  constexpr unsigned mask = 0xffffffffu;
  static_assert(code_blocks_per_group % ThreadsPerOutput == 0);
  extern __shared__ int8_t carrier_lut[];
  uint16_t* carrier_pair_lut =
      reinterpret_cast<uint16_t*>(carrier_lut + lut_bytes);
  int* shared_inputs =
      reinterpret_cast<int*>(carrier_lut + lut_bytes + pair_lut_bytes);

  const int output_in_block = threadIdx.x / ThreadsPerOutput;
  const int lane = threadIdx.x & (ThreadsPerOutput - 1);
  const int out_channel = blockIdx.y * outputs_per_block + output_in_block;
  const bool valid_output = out_channel < n;
  const bool share_output_lut =
      group_out >= outputs_per_block && group_out % outputs_per_block == 0;
  const int num_route_blocks = (*num_routes_ptr + routes_per_block - 1) /
                               routes_per_block;
  const int input_words_per_row = k / 4;

  for (int route_block = blockIdx.x; route_block < num_route_blocks;
       route_block += route_ctas) {
    int tokens[routes_per_block];
    bool valid[routes_per_block];
    int rows[routes_per_block];
    const int* input_rows[routes_per_block];
    float activation_scales[routes_per_block];
    float accumulators[routes_per_block] = {};
#pragma unroll
    for (int route = 0; route < routes_per_block; ++route) {
      tokens[route] = token_ids[route_block * routes_per_block + route];
      valid[route] = static_cast<unsigned>(tokens[route]) <
                     static_cast<unsigned>(num_valid_tokens);
      rows[route] = valid[route] ? tokens[route] / top_k : 0;
      input_rows[route] = reinterpret_cast<const int*>(input) +
                          static_cast<int64_t>(rows[route]) *
                              input_words_per_row;
      if constexpr (!GroupwiseScale) {
        if (lane == 0) activation_scales[route] = input_scale[rows[route]];
      }
    }
    const int expert = expert_ids[route_block];
    bool any_valid = false;
#pragma unroll
    for (int route = 0; route < routes_per_block; ++route) {
      any_valid |= valid[route];
    }
    if (expert < 0 || !any_valid) continue;
    const int64_t expert_output =
        static_cast<int64_t>(expert) * n + out_channel;
    const int* weight_words =
        reinterpret_cast<const int*>(weight + expert_output * packed_k);
    const int64_t meta_base =
        (static_cast<int64_t>(expert) * (n / group_out) +
         out_channel / group_out) *
        num_groups;
    const int64_t shared_meta_base =
        (static_cast<int64_t>(expert) * (n / group_out) +
         (blockIdx.y * outputs_per_block) / group_out) *
        num_groups;
    const int partition =
        min(out_channel / output_partition_size, num_partitions - 1);
    const int64_t table_index =
        static_cast<int64_t>(expert) * num_partitions + partition;
    for (int group = 0; group < num_groups; ++group) {
      for (int route = 0; route < routes_per_block; ++route) {
        for (int word = threadIdx.x; word < input_words_per_group;
             word += kThreads) {
          shared_inputs[route * input_words_per_group + word] =
              valid[route]
                  ? input_rows[route][group * input_words_per_group + word]
                  : 0;
        }
      }
      __syncthreads();

      const int64_t meta =
          (share_output_lut ? shared_meta_base : meta_base) + group;
      float weight_scale = 0.0f;
      float cubic_a = 0.0f;
      float cubic_b = 0.0f;
      uint8_t packed_ab = 0;
      if (valid_output) {
        cubic_load_metadata<Format>(
            primary, secondary, tertiary, scale_global, a_global, b_global,
            meta, table_index, weight_scale, cubic_a, cubic_b);
        if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
          packed_ab = static_cast<const uint8_t*>(secondary)[meta];
        }
      }
      if constexpr (GroupwiseScale) {
        if constexpr (Bits == 4) {
          if ((threadIdx.x & 31) == 0) {
#pragma unroll
            for (int route = 0; route < routes_per_block; ++route) {
              activation_scales[route] =
                  input_scale[static_cast<int64_t>(rows[route]) * num_groups +
                              group];
            }
          }
#pragma unroll
          for (int route = 0; route < routes_per_block; ++route) {
            activation_scales[route] =
                __shfl_sync(mask, activation_scales[route], 0);
          }
        } else if (lane == 0) {
#pragma unroll
          for (int route = 0; route < routes_per_block; ++route) {
            activation_scales[route] =
                input_scale[static_cast<int64_t>(rows[route]) * num_groups +
                            group];
          }
        }
      }
      if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
        // The carrier is reconstructed directly for each consumed code below.
      } else if (share_output_lut) {
        const int level = threadIdx.x;
        if (level < levels) {
          const float a = cubic_a;
          const float b = cubic_b;
          const float t = static_cast<float>(level) / (levels - 1);
          const float normalized = t * (a + t * (b + t * (1.0f - a - b)));
          const int8_t carrier =
              static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
          carrier_lut[level] = carrier;
          if constexpr (use_full_signed_lut) {
            if (level == 0)
              carrier_lut[levels] = 0;
            else
              carrier_lut[(1 << Bits) - level] = -carrier;
          }
        }
        __syncthreads();
        if constexpr (UsePairLut) {
          for (int pair = threadIdx.x; pair < 256; pair += kThreads) {
            const unsigned low =
                static_cast<unsigned>(carrier_lut[pair & 0xf]) & 0xff;
            const unsigned high =
                static_cast<unsigned>(carrier_lut[pair >> 4]) & 0xff;
            carrier_pair_lut[pair] =
                static_cast<uint16_t>(low | (high << 8));
          }
          __syncthreads();
        }
      } else {
        const float a = cubic_a;
        const float b = cubic_b;
        const float c = 1.0f - a - b;
        for (int level = lane; level < levels; level += ThreadsPerOutput) {
          const float t = static_cast<float>(level) / (levels - 1);
          const float normalized = t * (a + t * (b + t * c));
          const int8_t carrier =
              static_cast<int8_t>(__float2int_rn(normalized * 127.0f));
          carrier_lut[output_in_block * lut_entries + level] = carrier;
          if constexpr (use_full_signed_lut) {
            if (level == 0)
              carrier_lut[output_in_block * lut_entries + levels] = 0;
            else
              carrier_lut[output_in_block * lut_entries + (1 << Bits) - level] =
                  -carrier;
          }
        }
        __syncwarp();
      }

      int dots[routes_per_block] = {};
#pragma unroll
      for (int block_index = 0; block_index < blocks_per_thread;
           ++block_index) {
        const int code_block = lane + block_index * ThreadsPerOutput;
        unsigned packed_words[Bits];
        if constexpr (Bits == 4) {
          const uint4 packed =
              valid_output
                  ? cubic_ldcs_u32x4(reinterpret_cast<const uint4*>(
                        weight_words + group * words_per_group +
                        code_block * Bits))
                  : make_uint4(0, 0, 0, 0);
          packed_words[0] = packed.x;
          packed_words[1] = packed.y;
          packed_words[2] = packed.z;
          packed_words[3] = packed.w;
        } else if constexpr (Bits == 8) {
          const int* source =
              weight_words + group * words_per_group + code_block * Bits;
          const uint4 packed0 =
              valid_output
                  ? cubic_ldcs_u32x4(reinterpret_cast<const uint4*>(source))
                  : make_uint4(0, 0, 0, 0);
          const uint4 packed1 =
              valid_output
                  ? cubic_ldcs_u32x4(reinterpret_cast<const uint4*>(source + 4))
                  : make_uint4(0, 0, 0, 0);
          packed_words[0] = packed0.x;
          packed_words[1] = packed0.y;
          packed_words[2] = packed0.z;
          packed_words[3] = packed0.w;
          packed_words[4] = packed1.x;
          packed_words[5] = packed1.y;
          packed_words[6] = packed1.z;
          packed_words[7] = packed1.w;
        } else {
#pragma unroll
          for (int word_index = 0; word_index < Bits; ++word_index) {
            packed_words[word_index] =
                valid_output
                    ? __ldcs(weight_words + group * words_per_group +
                             code_block * Bits + word_index)
                    : 0;
          }
        }
#pragma unroll
        for (int quad = 0; quad < 8; ++quad) {
          unsigned carrier_word = 0;
          if constexpr (UsePairLut) {
            const int shift = (quad & 1) * 16;
            const unsigned raw = (packed_words[quad >> 1] >> shift) & 0xffff;
            carrier_word =
                static_cast<unsigned>(carrier_pair_lut[raw & 0xff]) |
                (static_cast<unsigned>(carrier_pair_lut[raw >> 8]) << 16);
          } else {
#pragma unroll
            for (int code_in_quad = 0; code_in_quad < 4; ++code_in_quad) {
              constexpr unsigned code_mask = (1u << Bits) - 1;
              const int bit = (quad * 4 + code_in_quad) * Bits;
              const int word_index = bit >> 5;
              const int shift = bit & 31;
              unsigned raw = packed_words[word_index] >> shift;
              if constexpr (Bits > 4) {
                if (shift + Bits > 32)
                  raw |= packed_words[word_index + 1] << (32 - shift);
              }
              raw &= code_mask;
              int carrier;
              if constexpr (Format == CubicMetadataFormat::Int8Int4Codebook) {
                constexpr int sign_bit = 1 << (Bits - 1);
                int signed_code = raw >= sign_bit
                                      ? static_cast<int>(raw) - (1 << Bits)
                                      : static_cast<int>(raw);
                if (signed_code == -sign_bit) signed_code = 0;
                const int magnitude = abs(signed_code);
                const float t = static_cast<float>(magnitude) / (levels - 1);
                const float normalized =
                    t * (cubic_a + t * (cubic_b + t *
                        (1.0f - cubic_a - cubic_b)));
                carrier = __float2int_rn(normalized * 127.0f);
                if (signed_code < 0) carrier = -carrier;
              } else if constexpr (use_full_signed_lut) {
                carrier = carrier_lut[(share_output_lut
                                           ? 0
                                           : output_in_block * lut_entries) +
                                      raw];
              } else {
                constexpr int sign_bit = 1 << (Bits - 1);
                int signed_code = raw >= sign_bit
                                      ? static_cast<int>(raw) - (1 << Bits)
                                      : static_cast<int>(raw);
                if (signed_code == -sign_bit) signed_code = 0;
                const int magnitude = abs(signed_code);
                carrier = carrier_lut[(share_output_lut
                                           ? 0
                                           : output_in_block * levels) +
                                      magnitude];
                if (signed_code < 0) carrier = -carrier;
              }
              carrier_word |= (static_cast<unsigned>(carrier) & 0xff)
                              << (code_in_quad * 8);
            }
          }
          const int input_index = code_block * 8 + quad;
#pragma unroll
          for (int route = 0; route < routes_per_block; ++route) {
            dots[route] = cubic_dp4a_w4_w8(
                carrier_word,
                shared_inputs[route * input_words_per_group + input_index],
                dots[route]);
          }
        }
      }
      for (int delta = ThreadsPerOutput / 2; delta > 0; delta >>= 1) {
#pragma unroll
        for (int route = 0; route < routes_per_block; ++route) {
          dots[route] +=
              __shfl_down_sync(mask, dots[route], delta, ThreadsPerOutput);
        }
      }
      if (lane == 0 && valid_output) {
#pragma unroll
        for (int route = 0; route < routes_per_block; ++route) {
          accumulators[route] += static_cast<float>(dots[route]) *
                                 activation_scales[route] * weight_scale *
                                 (1.0f / 127.0f);
        }
      }
      __syncthreads();
    }
    if (lane == 0 && valid_output) {
#pragma unroll
      for (int route = 0; route < routes_per_block; ++route) {
        if (!valid[route]) continue;
        if (multiply_routed_weight)
          accumulators[route] *= topk_weights[tokens[route]];
        output[static_cast<int64_t>(tokens[route]) * n + out_channel] =
            __float2bfloat16(accumulators[route]);
      }
    }
  }
}

template <int Bits, int GroupSize, int ThreadsPerOutput, bool GroupwiseScale>
void launch_cubic_w4_w8_a8(const int8_t* input, const float* input_scale,
                           const uint8_t* weight, const float* weight_scale,
                           const __half* cubic_a, const __half* cubic_b,
                           __nv_bfloat16* output, const float* topk_weights,
                           const int* token_ids, const int* expert_ids,
                           const int* num_routes, int n, int k, int num_groups,
                           int packed_k, int group_out, int top_k,
                           int route_ctas,
                           bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  constexpr int levels = 1 << (Bits - 1);
  constexpr int lut_entries = Bits <= 6 ? (1 << Bits) : levels;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w4_w8_a8_gemv_kernel<Bits, GroupSize, ThreadsPerOutput, GroupwiseScale>
      <<<grid, 128, outputs_per_block * lut_entries, stream>>>(
          input, input_scale, weight, weight_scale, cubic_a, cubic_b, output,
          topk_weights, token_ids, expert_ids, num_routes, n, k, num_groups,
          packed_k, group_out, top_k, route_ctas, multiply_routed_weight);
}

template <int Bits, int GroupSize, int ThreadsPerOutput,
          CubicMetadataFormat Format>
void launch_cubic_w4_w8_compact_a8(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const void* primary, const void* secondary, const void* tertiary,
    const float* scale_global, const float* a_global, const float* b_global,
    __nv_bfloat16* output, const float* topk_weights, const int* token_ids,
    const int* expert_ids, const int* num_routes, int n, int k,
    int num_groups, int packed_k, int group_out, int top_k, int route_ctas,
    int num_valid_tokens, int output_partition_size, int num_partitions,
    bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  constexpr int levels = 1 << (Bits - 1);
  constexpr int lut_entries = Bits <= 6 ? (1 << Bits) : levels;
  constexpr int shared_bytes =
      Format == CubicMetadataFormat::Int8Int4Codebook
          ? 256 * levels
          : outputs_per_block * lut_entries;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w4_w8_compact_a8_gemv_kernel<Bits, GroupSize, ThreadsPerOutput,
                                      Format>
      <<<grid, 128, shared_bytes, stream>>>(
          input, input_scale, weight, primary, secondary, tertiary,
          scale_global, a_global, b_global, output, topk_weights, token_ids,
          expert_ids, num_routes, n, k, num_groups, packed_k, group_out, top_k,
          route_ctas, num_valid_tokens, output_partition_size, num_partitions,
          multiply_routed_weight);
}

template <int Bits, int GroupSize, int RoutesPerBlock,
          CubicMetadataFormat Format>
void launch_cubic_w4_w8_compact_group_parallel_a8(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const void* primary, const void* secondary, const void* tertiary,
    const float* scale_global, const float* a_global, const float* b_global,
    __nv_bfloat16* output, const float* topk_weights, const int* token_ids,
    const int* expert_ids, const int* num_routes, int n, int k,
    int num_groups, int packed_k, int top_k, int route_ctas,
    int num_valid_tokens, int output_partition_size, int num_partitions,
    bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / 8;
  constexpr int levels = 1 << (Bits - 1);
  constexpr int lut_entries = Bits <= 6 ? (1 << Bits) : levels;
  constexpr bool use_shared_compact_codebook =
      Format == CubicMetadataFormat::Int8Int4 && Bits == 6;
  constexpr int shared_bytes =
      Format == CubicMetadataFormat::Int8Int4Codebook ||
              use_shared_compact_codebook
          ? 256 * levels
          : 128 * lut_entries;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w4_w8_compact_group_parallel_a8_gemv_kernel<
      Bits, GroupSize, RoutesPerBlock, Format>
      <<<grid, 128, shared_bytes, stream>>>(
          input, input_scale, weight, primary, secondary, tertiary,
          scale_global, a_global, b_global, output, topk_weights, token_ids,
          expert_ids, num_routes, n, k, num_groups, packed_k, top_k,
          route_ctas, num_valid_tokens, output_partition_size, num_partitions,
          multiply_routed_weight);
}

template <int Bits, int GroupSize, CubicMetadataFormat Format>
void launch_cubic_w4_w8_compact_a8_fused_sum(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const void* primary, const void* secondary, const void* tertiary,
    const float* scale_global, const float* a_global, const float* b_global,
    __nv_bfloat16* output, const float* topk_weights, const int* topk_ids,
    const int* expert_map, int num_tokens, int top_k, int n, int k,
    int num_groups, int packed_k, int group_out, int output_partition_size,
    int num_partitions, bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / 8;
  constexpr int levels = 1 << (Bits - 1);
  constexpr int lut_entries = Bits <= 6 ? (1 << Bits) : levels;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, num_tokens);
  cubic_w4_w8_compact_a8_fused_sum_kernel<Bits, GroupSize, Format>
      <<<grid, 128, 128 * lut_entries, stream>>>(
          input, input_scale, weight, primary, secondary, tertiary,
          scale_global, a_global, b_global, output, topk_weights, topk_ids,
          expert_map, num_tokens, top_k, n, k, num_groups, packed_k, group_out,
          output_partition_size, num_partitions, multiply_routed_weight);
}

template <int Bits, int GroupSize, int ThreadsPerOutput, bool GroupwiseScale>
void launch_cubic_w4_w8_grouped2_a8(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const float* weight_scale, const __half* cubic_a, const __half* cubic_b,
    __nv_bfloat16* output, const float* topk_weights, const int* token_ids,
    const int* expert_ids, const int* num_routes, int num_valid_tokens, int n,
    int k, int num_groups, int packed_k, int group_out, int top_k,
    int route_ctas,
    bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  constexpr int levels = 1 << (Bits - 1);
  constexpr int lut_entries = Bits <= 6 ? (1 << Bits) : levels;
  dim3 grid((n + outputs_per_block - 1) / outputs_per_block, route_ctas);
  cubic_w4_w8_grouped2_a8_gemv_kernel<Bits, GroupSize, ThreadsPerOutput,
                                      GroupwiseScale>
      <<<grid, 128, outputs_per_block * lut_entries, stream>>>(
          input, input_scale, weight, weight_scale, cubic_a, cubic_b, output,
          topk_weights, token_ids, expert_ids, num_routes, num_valid_tokens, n,
          k, num_groups, packed_k, group_out, top_k, route_ctas,
          multiply_routed_weight);
}

template <int Bits, int GroupSize, int ThreadsPerOutput, int RoutesPerBlock,
          bool GroupwiseScale, CubicMetadataFormat Format>
void launch_cubic_w4_w8_grouped_a8(
    const int8_t* input, const float* input_scale, const uint8_t* weight,
    const void* primary, const void* secondary, const void* tertiary,
    const float* scale_global, const float* a_global, const float* b_global,
    __nv_bfloat16* output, const float* topk_weights, const int* token_ids,
    const int* expert_ids, const int* num_routes, int num_valid_tokens, int n,
    int k, int num_groups, int packed_k, int group_out, int top_k,
    int route_ctas, int output_partition_size, int num_partitions,
    bool multiply_routed_weight, cudaStream_t stream) {
  constexpr int outputs_per_block = 128 / ThreadsPerOutput;
  constexpr int levels = 1 << (Bits - 1);
  constexpr int lut_entries = Bits <= 6 ? (1 << Bits) : levels;
  dim3 grid(route_ctas,
            (n + outputs_per_block - 1) / outputs_per_block);
  if constexpr (Bits == 4) {
    const bool share_output_lut =
        group_out >= outputs_per_block && group_out % outputs_per_block == 0;
    if (share_output_lut) {
      cubic_w4_w8_grouped_a8_gemv_kernel<Bits, GroupSize, ThreadsPerOutput,
                                         RoutesPerBlock, GroupwiseScale, Format,
                                         false>
          <<<grid, 128,
             outputs_per_block * lut_entries + RoutesPerBlock * GroupSize,
             stream>>>(
              input, input_scale, weight, primary, secondary, tertiary,
              scale_global, a_global, b_global, output, topk_weights, token_ids,
              expert_ids, num_routes, num_valid_tokens, n, k, num_groups,
              packed_k, group_out, top_k, route_ctas, output_partition_size,
              num_partitions, multiply_routed_weight);
      return;
    }
  }
  cubic_w4_w8_grouped_a8_gemv_kernel<Bits, GroupSize, ThreadsPerOutput,
                                     RoutesPerBlock, GroupwiseScale, Format,
                                     false>
      <<<grid, 128,
         outputs_per_block * lut_entries + RoutesPerBlock * GroupSize,
         stream>>>(
          input, input_scale, weight, primary, secondary, tertiary,
          scale_global, a_global, b_global, output, topk_weights, token_ids,
          expert_ids, num_routes, num_valid_tokens, n, k, num_groups, packed_k,
          group_out, top_k, route_ctas, output_partition_size, num_partitions,
          multiply_routed_weight);
}


}  // namespace

void cubic_w4_w8_compact_a8_gemv(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& primary,
    const torch::stable::Tensor& secondary,
    const torch::stable::Tensor& tertiary,
    const torch::stable::Tensor& scale_global,
    const torch::stable::Tensor& a_global,
    const torch::stable::Tensor& b_global,
    torch::stable::Tensor& output, const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_routes, int64_t bits, int64_t group_size,
    int64_t group_out, int64_t top_k, bool multiply_routed_weight,
    int64_t route_ctas, int64_t num_valid_tokens,
    int64_t output_partition_size,
    int64_t num_partitions, int64_t metadata_format,
    int64_t routes_per_block) {
  STD_TORCH_CHECK(input.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w4_w8_compact_a8_gemv: tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w4_w8_compact_a8_gemv: invalid core tensor dtype");
  STD_TORCH_CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
                      weight.is_contiguous() && primary.is_contiguous() &&
                      secondary.is_contiguous() && tertiary.is_contiguous() &&
                      scale_global.is_contiguous() && a_global.is_contiguous() &&
                      b_global.is_contiguous() && output.is_contiguous(),
                  "cubic_w4_w8_compact_a8_gemv: tensors must be contiguous");
  const int n = weight.size(1);
  const int k = input.size(1);
  STD_TORCH_CHECK(
      bits >= 4 && bits <= 8 &&
          (group_size == 32 || group_size == 64 || group_size == 128 ||
           group_size == 256 || group_size == 512) &&
          group_out > 0 && n % group_out == 0 && k % group_size == 0 &&
          weight.size(2) == k * bits / 8 && route_ctas > 0 &&
          output_partition_size > 0 && num_partitions > 0 &&
          (metadata_format == 1 || metadata_format == 2 || metadata_format == 3) &&
          (routes_per_block == 1 || routes_per_block == 2 ||
           routes_per_block == 4 || routes_per_block == 8),
      "cubic_w4_w8_compact_a8_gemv: unsupported shape or format");
  const int groups = k / group_size;
  STD_TORCH_CHECK(
      primary.numel() ==
          weight.size(0) * static_cast<int64_t>(n / group_out) * groups,
      "cubic_w4_w8_compact_a8_gemv: invalid primary metadata shape");
  STD_TORCH_CHECK(input_scale.numel() == input.size(0),
                  "cubic_w4_w8_compact_a8_gemv: invalid input scale shape");
  const auto stream = get_current_cuda_stream(input.get_device_index());

#define COMPACT_ARGS                                                          \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), primary.const_data_ptr(),             \
      secondary.const_data_ptr(), tertiary.const_data_ptr(),                  \
      static_cast<const float*>(scale_global.const_data_ptr()),               \
      static_cast<const float*>(a_global.const_data_ptr()),                   \
      static_cast<const float*>(b_global.const_data_ptr()),                   \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),   \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, groups, weight.size(2), group_out, top_k, route_ctas,                \
      num_valid_tokens, output_partition_size, num_partitions,               \
      multiply_routed_weight, stream
#define COMPACT_GROUPED_ARGS                                                  \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), primary.const_data_ptr(),             \
      secondary.const_data_ptr(), tertiary.const_data_ptr(),                  \
      static_cast<const float*>(scale_global.const_data_ptr()),               \
      static_cast<const float*>(a_global.const_data_ptr()),                   \
      static_cast<const float*>(b_global.const_data_ptr()),                   \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),   \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(),     \
      num_valid_tokens, n, k, groups, weight.size(2), group_out, top_k,       \
      route_ctas, output_partition_size, num_partitions,                      \
      multiply_routed_weight, stream
  const bool disable_group_parallel =
      std::getenv("VLLM_CUBIC_DISABLE_GROUP_PARALLEL") != nullptr;
  const bool prefer_shared_curve2 =
      metadata_format == 2 && group_out == 1 && routes_per_block == 1 &&
      (((bits >= 5 && bits <= 7) && group_size == 32 &&
        output_partition_size % 128 == 0) ||
       (bits == 8 && (group_size == 256 || group_size == 512) &&
        output_partition_size % 16 == 0));
  if (!disable_group_parallel && !prefer_shared_curve2 && group_out == 1 &&
      group_size < 128 &&
      (metadata_format == 1 || metadata_format == 2 || metadata_format == 3)) {
#define LAUNCH_GROUP_PARALLEL_ROUTE(B, G, R, F)                             \
  launch_cubic_w4_w8_compact_group_parallel_a8<B, G, R, F>(                \
      input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),  \
      weight.const_data_ptr<uint8_t>(), primary.const_data_ptr(),           \
      secondary.const_data_ptr(), tertiary.const_data_ptr(),                \
      static_cast<const float*>(scale_global.const_data_ptr()),             \
      static_cast<const float*>(a_global.const_data_ptr()),                 \
      static_cast<const float*>(b_global.const_data_ptr()),                 \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),          \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(), \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(),   \
      n, k, groups, weight.size(2), top_k, route_ctas, num_valid_tokens,    \
      output_partition_size, num_partitions, multiply_routed_weight, stream)
#define LAUNCH_GROUP_PARALLEL(B, G, F)                    \
  do {                                                     \
    if (routes_per_block == 8)                             \
      LAUNCH_GROUP_PARALLEL_ROUTE(B, G, 8, F);             \
    else if (routes_per_block == 4)                        \
      LAUNCH_GROUP_PARALLEL_ROUTE(B, G, 4, F);             \
    else if (routes_per_block == 2)                        \
      LAUNCH_GROUP_PARALLEL_ROUTE(B, G, 2, F);             \
    else                                                   \
      LAUNCH_GROUP_PARALLEL_ROUTE(B, G, 1, F);             \
  } while (0)
#define DISPATCH_GROUP_PARALLEL_BITS(B, F) \
  do {                                         \
    if (group_size == 32)                      \
      LAUNCH_GROUP_PARALLEL(B, 32, F);         \
    else if (group_size == 64)                 \
      LAUNCH_GROUP_PARALLEL(B, 64, F);         \
    else if (group_size == 128)                \
      LAUNCH_GROUP_PARALLEL(B, 128, F);        \
    else if (group_size == 256)                \
      LAUNCH_GROUP_PARALLEL(B, 256, F);        \
    else                                       \
      LAUNCH_GROUP_PARALLEL(B, 512, F);        \
    return;                                    \
  } while (0)
#define DISPATCH_GROUP_PARALLEL_FORMAT(B)                                  \
  do {                                                                      \
    if (metadata_format == 1)                                               \
      DISPATCH_GROUP_PARALLEL_BITS(B, CubicMetadataFormat::Int8Int4);       \
    else if (metadata_format == 3)                                          \
      DISPATCH_GROUP_PARALLEL_BITS(                                         \
          B, CubicMetadataFormat::Int8Int4Codebook);                        \
    else                                                                    \
      DISPATCH_GROUP_PARALLEL_BITS(B, CubicMetadataFormat::E5M9Curve2);     \
  } while (0)
    if (bits == 4)
      DISPATCH_GROUP_PARALLEL_FORMAT(4);
    else if (bits == 5)
      DISPATCH_GROUP_PARALLEL_FORMAT(5);
    else if (bits == 6)
      DISPATCH_GROUP_PARALLEL_FORMAT(6);
    else if (bits == 7)
      DISPATCH_GROUP_PARALLEL_FORMAT(7);
    else
      DISPATCH_GROUP_PARALLEL_FORMAT(8);
#undef DISPATCH_GROUP_PARALLEL_FORMAT
#undef DISPATCH_GROUP_PARALLEL_BITS
#undef LAUNCH_GROUP_PARALLEL
#undef LAUNCH_GROUP_PARALLEL_ROUTE
  }
  if (metadata_format == 2 && group_out == 1 && routes_per_block == 1) {
#define LAUNCH_SHARED_CURVE2(B, G, T)                                      \
  do {                                                                      \
    dim3 grid((n + (128 / T) - 1) / (128 / T), route_ctas);                \
    cubic_curve2_shared_lut_a8_gemv_kernel<B, G, T>                         \
        <<<grid, 128, 0, stream>>>(                                         \
            input.const_data_ptr<int8_t>(),                                 \
            input_scale.const_data_ptr<float>(),                            \
            weight.const_data_ptr<uint8_t>(),                               \
            primary.const_data_ptr<uint16_t>(),                             \
            reinterpret_cast<const __half*>(secondary.const_data_ptr()),    \
            reinterpret_cast<const __half*>(tertiary.const_data_ptr()),     \
            reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),    \
            topk_weights.const_data_ptr<float>(),                           \
            token_ids.const_data_ptr<int>(), expert_ids.const_data_ptr<int>(), \
            num_routes.const_data_ptr<int>(), n, k, groups, weight.size(2), \
            top_k, route_ctas, num_valid_tokens, output_partition_size,     \
            num_partitions, multiply_routed_weight);                        \
    return;                                                                 \
  } while (0)
    if (bits == 5 && group_size == 32 && output_partition_size % 128 == 0)
      LAUNCH_SHARED_CURVE2(5, 32, 1);
    if (bits == 6 && group_size == 32 && output_partition_size % 128 == 0)
      LAUNCH_SHARED_CURVE2(6, 32, 1);
    if (bits == 7 && group_size == 32 && output_partition_size % 128 == 0)
      LAUNCH_SHARED_CURVE2(7, 32, 1);
    if (bits == 8 && group_size == 256 && output_partition_size % 16 == 0)
      LAUNCH_SHARED_CURVE2(8, 256, 8);
    if (bits == 8 && group_size == 512 && output_partition_size % 16 == 0)
      LAUNCH_SHARED_CURVE2(8, 512, 8);
#undef LAUNCH_SHARED_CURVE2
  }
#define LAUNCH_COMPACT_FORMAT(B, G, T, F)                       \
  do {                                                          \
    if (routes_per_block == 8)                                  \
      launch_cubic_w4_w8_grouped_a8<B, G, T, 8, false, F>(     \
          COMPACT_GROUPED_ARGS);                                 \
    else if (routes_per_block == 4)                             \
      launch_cubic_w4_w8_grouped_a8<B, G, T, 4, false, F>(     \
          COMPACT_GROUPED_ARGS);                                 \
    else if (routes_per_block == 2)                             \
      launch_cubic_w4_w8_grouped_a8<B, G, T, 2, false, F>(     \
          COMPACT_GROUPED_ARGS);                                 \
    else                                                        \
      launch_cubic_w4_w8_compact_a8<B, G, T, F>(COMPACT_ARGS); \
  } while (0)
#define LAUNCH_COMPACT_GROUP(B, G, T)                                  \
  do {                                                                 \
    if (metadata_format == 1)                                          \
      LAUNCH_COMPACT_FORMAT(B, G, T, CubicMetadataFormat::Int8Int4);   \
    else if (metadata_format == 3)                                     \
      LAUNCH_COMPACT_FORMAT(B, G, T,                                   \
                            CubicMetadataFormat::Int8Int4Codebook);    \
    else                                                               \
      LAUNCH_COMPACT_FORMAT(B, G, T, CubicMetadataFormat::E5M9Curve2); \
  } while (0)
#define LAUNCH_COMPACT_BITS(B)         \
  do {                                 \
    if (group_size == 32)              \
      LAUNCH_COMPACT_GROUP(B, 32, 1);  \
    else if (group_size == 64)         \
      LAUNCH_COMPACT_GROUP(B, 64, 2);  \
    else if (group_size == 128)        \
      LAUNCH_COMPACT_GROUP(B, 128, 4); \
    else if (group_size == 256)        \
      LAUNCH_COMPACT_GROUP(B, 256, 8); \
    else                               \
      LAUNCH_COMPACT_GROUP(B, 512, 8); \
  } while (0)
  if (bits == 4)
    LAUNCH_COMPACT_BITS(4);
  else if (bits == 5)
    LAUNCH_COMPACT_BITS(5);
  else if (bits == 6)
    LAUNCH_COMPACT_BITS(6);
  else if (bits == 7)
    LAUNCH_COMPACT_BITS(7);
  else
    LAUNCH_COMPACT_BITS(8);
#undef LAUNCH_COMPACT_BITS
#undef LAUNCH_COMPACT_GROUP
#undef LAUNCH_COMPACT_FORMAT
#undef COMPACT_GROUPED_ARGS
#undef COMPACT_ARGS
}

void cubic_w4_w8_compact_a8_fused_sum(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& primary,
    const torch::stable::Tensor& secondary,
    const torch::stable::Tensor& tertiary,
    const torch::stable::Tensor& scale_global,
    const torch::stable::Tensor& a_global,
    const torch::stable::Tensor& b_global,
    torch::stable::Tensor& output, const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& expert_map, int64_t bits,
    int64_t group_size, int64_t group_out, bool multiply_routed_weight,
    int64_t output_partition_size, int64_t num_partitions,
    int64_t metadata_format) {
  STD_TORCH_CHECK(input.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w4_w8_compact_a8_fused_sum: tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          topk_weights.scalar_type() == torch::headeronly::ScalarType::Float &&
          topk_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          expert_map.scalar_type() == torch::headeronly::ScalarType::Int,
      "cubic_w4_w8_compact_a8_fused_sum: invalid core tensor dtype");
  STD_TORCH_CHECK(
      input.is_contiguous() && input_scale.is_contiguous() &&
          weight.is_contiguous() && primary.is_contiguous() &&
          secondary.is_contiguous() && tertiary.is_contiguous() &&
          scale_global.is_contiguous() && a_global.is_contiguous() &&
          b_global.is_contiguous() && output.is_contiguous() &&
          topk_weights.is_contiguous() && topk_ids.is_contiguous() &&
          expert_map.is_contiguous(),
      "cubic_w4_w8_compact_a8_fused_sum: tensors must be contiguous");
  const int num_tokens = output.size(0);
  const int top_k = topk_ids.size(1);
  const int n = weight.size(1);
  const int k = input.size(1);
  STD_TORCH_CHECK(
      bits >= 4 && bits <= 8 &&
          (group_size == 32 || group_size == 64 || group_size == 128 ||
           group_size == 256 || group_size == 512) &&
          group_out > 0 && n % group_out == 0 && k % group_size == 0 &&
          weight.size(2) == k * bits / 8 && output.size(1) == n &&
          input.size(0) == num_tokens * top_k &&
          input_scale.numel() == input.size(0) &&
          topk_weights.numel() == num_tokens * top_k &&
          topk_ids.numel() == num_tokens * top_k &&
          output_partition_size > 0 && num_partitions > 0 &&
          (metadata_format == 1 || metadata_format == 2),
      "cubic_w4_w8_compact_a8_fused_sum: unsupported shape or format");
  const int groups = k / group_size;
  STD_TORCH_CHECK(
      primary.numel() ==
          weight.size(0) * static_cast<int64_t>(n / group_out) * groups,
      "cubic_w4_w8_compact_a8_fused_sum: invalid primary metadata shape");
  const auto stream = get_current_cuda_stream(input.get_device_index());

#define FUSED_SUM_ARGS                                                       \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),       \
      weight.const_data_ptr<uint8_t>(), primary.const_data_ptr(),            \
      secondary.const_data_ptr(), tertiary.const_data_ptr(),                 \
      static_cast<const float*>(scale_global.const_data_ptr()),              \
      static_cast<const float*>(a_global.const_data_ptr()),                  \
      static_cast<const float*>(b_global.const_data_ptr()),                  \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),           \
      topk_weights.const_data_ptr<float>(), topk_ids.const_data_ptr<int>(),  \
      expert_map.const_data_ptr<int>(), num_tokens, top_k, n, k, groups,    \
      weight.size(2), group_out, output_partition_size, num_partitions,      \
      multiply_routed_weight, stream
#define LAUNCH_FUSED_SUM_GROUP(B, G, F) \
  launch_cubic_w4_w8_compact_a8_fused_sum<B, G, F>(FUSED_SUM_ARGS)
#define DISPATCH_FUSED_SUM_GROUP(B, F)              \
  do {                                               \
    if (group_size == 32)                           \
      LAUNCH_FUSED_SUM_GROUP(B, 32, F);             \
    else if (group_size == 64)                      \
      LAUNCH_FUSED_SUM_GROUP(B, 64, F);             \
    else if (group_size == 128)                     \
      LAUNCH_FUSED_SUM_GROUP(B, 128, F);            \
    else if (group_size == 256)                     \
      LAUNCH_FUSED_SUM_GROUP(B, 256, F);            \
    else                                             \
      LAUNCH_FUSED_SUM_GROUP(B, 512, F);            \
  } while (0)
#define DISPATCH_FUSED_SUM_BITS(F)         \
  do {                                     \
    if (bits == 4)                         \
      DISPATCH_FUSED_SUM_GROUP(4, F);      \
    else if (bits == 5)                    \
      DISPATCH_FUSED_SUM_GROUP(5, F);      \
    else if (bits == 6)                    \
      DISPATCH_FUSED_SUM_GROUP(6, F);      \
    else if (bits == 7)                    \
      DISPATCH_FUSED_SUM_GROUP(7, F);      \
    else                                   \
      DISPATCH_FUSED_SUM_GROUP(8, F);      \
  } while (0)
  if (metadata_format == 1)
    DISPATCH_FUSED_SUM_BITS(CubicMetadataFormat::Int8Int4);
  else
    DISPATCH_FUSED_SUM_BITS(CubicMetadataFormat::E5M9Curve2);
#undef DISPATCH_FUSED_SUM_BITS
#undef DISPATCH_FUSED_SUM_GROUP
#undef LAUNCH_FUSED_SUM_GROUP
#undef FUSED_SUM_ARGS
}

void cubic_w4_w8_a8_gemv_impl(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& weight_scale,
    const torch::stable::Tensor& cubic_a, const torch::stable::Tensor& cubic_b,
    torch::stable::Tensor& output, const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_routes, int64_t bits, int64_t group_size,
    int64_t group_out, int64_t top_k, bool multiply_routed_weight,
    int64_t route_ctas,
    int64_t num_valid_tokens, int64_t routes_per_block, bool groupwise_scale) {
  STD_TORCH_CHECK(input.device().is_cuda() && weight.device().is_cuda() &&
                      output.device().is_cuda(),
                  "cubic_w4_w8_a8_gemv: tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Char &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weight_scale.scalar_type() == torch::headeronly::ScalarType::Float &&
          cubic_a.scalar_type() == torch::headeronly::ScalarType::Half &&
          cubic_b.scalar_type() == torch::headeronly::ScalarType::Half &&
          output.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "cubic_w4_w8_a8_gemv: invalid tensor dtype");
  STD_TORCH_CHECK(input.is_contiguous() && input_scale.is_contiguous() &&
                      weight.is_contiguous() && weight_scale.is_contiguous() &&
                      cubic_a.is_contiguous() && cubic_b.is_contiguous() &&
                      output.is_contiguous(),
                  "cubic_w4_w8_a8_gemv: tensors must be contiguous");
  const int n = weight.size(1);
  const int k = input.size(1);
  STD_TORCH_CHECK(
      bits >= 4 && bits <= 8 &&
          (group_size == 32 || group_size == 64 || group_size == 128 ||
           group_size == 256 || group_size == 512) &&
          group_out > 0 && n % group_out == 0 && k % group_size == 0 &&
          weight.size(2) == k * bits / 8,
      "cubic_w4_w8_a8_gemv: unsupported shape, bits, or group size");
  STD_TORCH_CHECK(
      route_ctas > 0 &&
          (routes_per_block == 1 || routes_per_block == 2 ||
           routes_per_block == 4 || routes_per_block == 8),
      "cubic_w4_w8_a8_gemv: invalid route_ctas");
  const int groups = k / group_size;
  STD_TORCH_CHECK(
      weight_scale.numel() ==
              weight.size(0) * static_cast<int64_t>(n / group_out) * groups &&
          cubic_a.numel() == weight_scale.numel() &&
          cubic_b.numel() == weight_scale.numel(),
      "cubic_w4_w8_a8_gemv: invalid weight metadata shape");
  STD_TORCH_CHECK(
      input_scale.numel() ==
          input.size(0) * static_cast<int64_t>(groupwise_scale ? groups : 1),
      "cubic_w4_w8_a8_gemv: invalid input scale shape");
  const int desired_subgroup =
      static_cast<int>(std::min<int64_t>(group_size / 32, 8));
  const auto stream = get_current_cuda_stream(input.get_device_index());

#define CUBIC_ARGS                                                            \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      reinterpret_cast<const __half*>(cubic_a.const_data_ptr()),              \
      reinterpret_cast<const __half*>(cubic_b.const_data_ptr()),              \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(), n,  \
      k, groups, weight.size(2), group_out, top_k, route_ctas,               \
      multiply_routed_weight,                                                \
      stream
#define CUBIC_GROUPED_ARGS                                                    \
  input.const_data_ptr<int8_t>(), input_scale.const_data_ptr<float>(),        \
      weight.const_data_ptr<uint8_t>(), weight_scale.const_data_ptr<float>(), \
      reinterpret_cast<const __half*>(cubic_a.const_data_ptr()),              \
      reinterpret_cast<const __half*>(cubic_b.const_data_ptr()),              \
      nullptr, nullptr, nullptr,                                              \
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),            \
      topk_weights.const_data_ptr<float>(), token_ids.const_data_ptr<int>(),  \
      expert_ids.const_data_ptr<int>(), num_routes.const_data_ptr<int>(),     \
      num_valid_tokens, n, k, groups, weight.size(2), group_out, top_k,       \
      route_ctas, n, 1, multiply_routed_weight, stream
#define LAUNCH_GROUP_MODE(B, G, GROUPWISE)                   \
  do {                                                       \
    if constexpr (G == 32) {                                \
      STD_TORCH_CHECK(routes_per_block == 1,                 \
                      "G32 supports singleton routes only"); \
      launch_cubic_w4_w8_a8<B, G, 1, GROUPWISE>(CUBIC_ARGS); \
    } else if (routes_per_block == 8) {                      \
      if constexpr (G == 64)                                 \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 2, 8, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else if constexpr (G == 128)                           \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 4, 8, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else if (desired_subgroup == 4)                        \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 4, 8, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else                                                   \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 8, 8, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
    } else if (routes_per_block == 4) {                      \
      if constexpr (G == 64)                                 \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 2, 4, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else if constexpr (G == 128)                           \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 4, 4, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else if (desired_subgroup == 4)                        \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 4, 4, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else                                                   \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 8, 4, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
    } else if (routes_per_block == 2) {                      \
      if constexpr (G == 64)                                 \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 2, 2, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else if constexpr (G == 128)                           \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 4, 2, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else if (desired_subgroup == 4)                        \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 4, 2, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
      else                                                   \
        launch_cubic_w4_w8_grouped_a8<                              \
            B, G, 8, 2, GROUPWISE, CubicMetadataFormat::Expanded>( \
            CUBIC_GROUPED_ARGS);                             \
    } else if constexpr (G == 64)                            \
      launch_cubic_w4_w8_a8<B, G, 2, GROUPWISE>(CUBIC_ARGS); \
    else if constexpr (G == 128)                             \
      launch_cubic_w4_w8_a8<B, G, 4, GROUPWISE>(CUBIC_ARGS); \
    else if (desired_subgroup == 4)                          \
      launch_cubic_w4_w8_a8<B, G, 4, GROUPWISE>(CUBIC_ARGS); \
    else                                                     \
      launch_cubic_w4_w8_a8<B, G, 8, GROUPWISE>(CUBIC_ARGS); \
  } while (0)
#define LAUNCH_GROUP(B, G)            \
  do {                                \
    if (groupwise_scale)              \
      LAUNCH_GROUP_MODE(B, G, true);  \
    else                              \
      LAUNCH_GROUP_MODE(B, G, false); \
  } while (0)
#define LAUNCH_BITS(B)          \
  do {                          \
    if (group_size == 32)       \
      LAUNCH_GROUP(B, 32);      \
    else if (group_size == 64)  \
      LAUNCH_GROUP(B, 64);      \
    else if (group_size == 128) \
      LAUNCH_GROUP(B, 128);     \
    else if (group_size == 256) \
      LAUNCH_GROUP(B, 256);     \
    else                        \
      LAUNCH_GROUP(B, 512);     \
  } while (0)
  if (bits == 4)
    LAUNCH_BITS(4);
  else if (bits == 5)
    LAUNCH_BITS(5);
  else if (bits == 6)
    LAUNCH_BITS(6);
  else if (bits == 7)
    LAUNCH_BITS(7);
  else
    LAUNCH_BITS(8);
#undef LAUNCH_BITS
#undef LAUNCH_GROUP
#undef LAUNCH_GROUP_MODE
#undef CUBIC_GROUPED_ARGS
#undef CUBIC_ARGS
}


void cubic_w4_w8_a8_gemv(const torch::stable::Tensor& input,
                         const torch::stable::Tensor& input_scale,
                         const torch::stable::Tensor& weight,
                         const torch::stable::Tensor& weight_scale,
                         const torch::stable::Tensor& cubic_a,
                         const torch::stable::Tensor& cubic_b,
                         torch::stable::Tensor& output,
                         const torch::stable::Tensor& topk_weights,
                         const torch::stable::Tensor& token_ids,
                         const torch::stable::Tensor& expert_ids,
                         const torch::stable::Tensor& num_routes, int64_t bits,
                         int64_t group_size, int64_t group_out, int64_t top_k,
                         bool multiply_routed_weight, int64_t route_ctas,
                         int64_t num_valid_tokens, int64_t routes_per_block) {
  cubic_w4_w8_a8_gemv_impl(input, input_scale, weight, weight_scale, cubic_a,
                           cubic_b, output, topk_weights, token_ids, expert_ids,
                           num_routes, bits, group_size, group_out, top_k,
                           multiply_routed_weight, route_ctas, num_valid_tokens,
                           routes_per_block, false);
}

void cubic_w4_w8_groupwise_a8_gemv(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& weight,
    const torch::stable::Tensor& weight_scale,
    const torch::stable::Tensor& cubic_a, const torch::stable::Tensor& cubic_b,
    torch::stable::Tensor& output, const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_routes, int64_t bits, int64_t group_size,
    int64_t group_out, int64_t top_k, bool multiply_routed_weight,
    int64_t route_ctas,
    int64_t num_valid_tokens, int64_t routes_per_block) {
  cubic_w4_w8_a8_gemv_impl(input, input_scale, weight, weight_scale, cubic_a,
                           cubic_b, output, topk_weights, token_ids, expert_ids,
                           num_routes, bits, group_size, group_out, top_k,
                           multiply_routed_weight, route_ctas, num_valid_tokens,
                           routes_per_block, true);
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("cubic_w4_w8_compact_a8_gemv",
         TORCH_BOX(&cubic_w4_w8_compact_a8_gemv));
  m.impl("cubic_w4_w8_compact_a8_fused_sum",
         TORCH_BOX(&cubic_w4_w8_compact_a8_fused_sum));
  m.impl("cubic_w4_w8_a8_gemv", TORCH_BOX(&cubic_w4_w8_a8_gemv));
  m.impl("cubic_w4_w8_groupwise_a8_gemv",
         TORCH_BOX(&cubic_w4_w8_groupwise_a8_gemv));
}
