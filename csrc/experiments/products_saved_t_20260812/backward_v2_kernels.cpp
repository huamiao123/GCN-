#include "../../amx/backward_v2_kernels.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <sys/syscall.h>
#include <unistd.h>

namespace tfs::backward_v2 {

namespace {

struct alignas(64) TileConfig {
  std::uint8_t palette = 1;
  std::uint8_t start_row = 0;
  std::uint8_t reserved[14]{};
  std::uint16_t colsb[16]{};
  std::uint8_t rows[16]{};
};

inline void load_a_tile(bool alternate, const bf16* ptr, int stride_bytes) {
  if (alternate) {
    _tile_loadd(5, ptr, stride_bytes);
  } else {
    _tile_loadd(4, ptr, stride_bytes);
  }
}

inline void load_b_and_dp(int q, bool alternate, const bf16* ptr,
                          bool a_alternate) {
  if (alternate) {
    _tile_loadd(7, ptr, 64);
    if (a_alternate) {
      switch (q) {
        case 0: _tile_dpbf16ps(0, 5, 7); break;
        case 1: _tile_dpbf16ps(1, 5, 7); break;
        case 2: _tile_dpbf16ps(2, 5, 7); break;
        default: _tile_dpbf16ps(3, 5, 7); break;
      }
    } else {
      switch (q) {
        case 0: _tile_dpbf16ps(0, 4, 7); break;
        case 1: _tile_dpbf16ps(1, 4, 7); break;
        case 2: _tile_dpbf16ps(2, 4, 7); break;
        default: _tile_dpbf16ps(3, 4, 7); break;
      }
    }
  } else {
    _tile_loadd(6, ptr, 64);
    if (a_alternate) {
      switch (q) {
        case 0: _tile_dpbf16ps(0, 5, 6); break;
        case 1: _tile_dpbf16ps(1, 5, 6); break;
        case 2: _tile_dpbf16ps(2, 5, 6); break;
        default: _tile_dpbf16ps(3, 5, 6); break;
      }
    } else {
      switch (q) {
        case 0: _tile_dpbf16ps(0, 4, 6); break;
        case 1: _tile_dpbf16ps(1, 4, 6); break;
        case 2: _tile_dpbf16ps(2, 4, 6); break;
        default: _tile_dpbf16ps(3, 4, 6); break;
      }
    }
  }
}

inline void load_c_tiles(float* c, int stride_floats, int active) {
  if (active >= 1) _tile_loadd(0, c, stride_floats * static_cast<int>(sizeof(float)));
  if (active >= 2) _tile_loadd(1, c + 16, stride_floats * static_cast<int>(sizeof(float)));
  if (active >= 3) _tile_loadd(2, c + 32, stride_floats * static_cast<int>(sizeof(float)));
  if (active >= 4) _tile_loadd(3, c + 48, stride_floats * static_cast<int>(sizeof(float)));
}

inline void store_c_tiles(float* c, int stride_floats, int active) {
  if (active >= 1) _tile_stored(0, c, stride_floats * static_cast<int>(sizeof(float)));
  if (active >= 2) _tile_stored(1, c + 16, stride_floats * static_cast<int>(sizeof(float)));
  if (active >= 3) _tile_stored(2, c + 32, stride_floats * static_cast<int>(sizeof(float)));
  if (active >= 4) _tile_stored(3, c + 48, stride_floats * static_cast<int>(sizeof(float)));
}

inline void store_c_to_scratch(float scratch[4][256], int active) {
  if (active >= 1) _tile_stored(0, scratch[0], 64);
  if (active >= 2) _tile_stored(1, scratch[1], 64);
  if (active >= 3) _tile_stored(2, scratch[2], 64);
  if (active >= 4) _tile_stored(3, scratch[3], 64);
}

}  // namespace

void configure_amx_tiles_16x64() {
  // AMX permission is a per-thread property and survives tile release.  Repeating
  // arch_prctl for every layer made the integrated PyTorch path pay the kernel
  // transition on every worker at every invocation.  Keep an opt-out for a
  // matched diagnostic A/B; the production default caches permission per worker.
  static const bool cache_permission = [] {
    const char* value = std::getenv("TFS_AMX_PERMISSION_CACHE");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  static thread_local bool permission_granted = false;
  if (!cache_permission || !permission_granted) {
    if (syscall(SYS_arch_prctl, 0x1023, 18) != 0) {
      throw std::runtime_error("AMX tile-data permission failed");
    }
    permission_granted = true;
  }
  TileConfig cfg;
  for (int tile = 0; tile < 8; ++tile) {
    cfg.rows[tile] = 16;
    cfg.colsb[tile] = 64;
  }
  _tile_loadconfig(&cfg);
}

std::vector<bf16> pack_rhs_baseline(const bf16* b, int reduction_padded,
                                    int output_padded) {
  const int reduction_blocks = reduction_padded / 32;
  const int output_blocks = output_padded / 16;
  std::vector<bf16> packed(
      static_cast<std::size_t>(reduction_blocks) * output_blocks * 512);
  for (int k = 0; k < reduction_padded; ++k) {
    for (int n = 0; n < output_padded; ++n) {
      const std::size_t base =
          (static_cast<std::size_t>(k / 32) * output_blocks + n / 16) * 512;
      packed[base + static_cast<std::size_t>((k % 32) / 2) * 32 +
             2 * (n % 16) + (k & 1)] =
          b[static_cast<std::size_t>(k) * output_padded + n];
    }
  }
  return packed;
}

std::vector<bf16> pack_rhs_reduction_baseline(const bf16* b,
                                              int reduction_padded,
                                              int output_padded,
                                              int reduction_rows) {
  const int blocks = reduction_padded / reduction_rows;
  const int output_blocks = output_padded / 16;
  std::vector<bf16> packed(
      static_cast<std::size_t>(blocks) * output_blocks * 512, bf16(0));
  for (int k = 0; k < reduction_padded; ++k) {
    for (int n = 0; n < output_padded; ++n) {
      const int local = k % reduction_rows;
      const std::size_t base =
          (static_cast<std::size_t>(k / reduction_rows) * output_blocks +
           n / 16) * 512;
      packed[base + static_cast<std::size_t>(local / 2) * 32 +
             2 * (n % 16) + (local & 1)] =
          b[static_cast<std::size_t>(k) * output_padded + n];
    }
  }
  return packed;
}

void pack_h_panel_direct(const bf16* hs, int hs_stride, int row0,
                         int valid_rows, int rows_padded, int k_padded,
                         bf16* packed_h) {
  const int output_blocks = k_padded / 16;
  std::fill(packed_h,
            packed_h + static_cast<std::size_t>(rows_padded / 32) *
                           output_blocks * 512,
            bf16(0));
  for (int row = 0; row < valid_rows; ++row) {
    for (int col = 0; col < k_padded; ++col) {
      const std::size_t base =
          (static_cast<std::size_t>(row / 32) * output_blocks + col / 16) *
          512;
      packed_h[base + static_cast<std::size_t>((row % 32) / 2) * 32 +
               2 * (col % 16) + (row & 1)] =
          hs[static_cast<std::size_t>(row0 + row) * hs_stride + col];
    }
  }
}

void pack_h_panel_direct_tail(const bf16* hs, int hs_stride, int logical_k,
                              int row0, int valid_rows, int rows_padded,
                              int k_padded, bf16* packed_h) {
  const int output_blocks = k_padded / 16;
  std::fill(packed_h,
            packed_h + static_cast<std::size_t>(rows_padded / 32) *
                           output_blocks * 512,
            bf16(0));
  for (int row = 0; row < valid_rows; ++row) {
    for (int col = 0; col < logical_k; ++col) {
      const std::size_t base =
          (static_cast<std::size_t>(row / 32) * output_blocks + col / 16) *
          512;
      packed_h[base + static_cast<std::size_t>((row % 32) / 2) * 32 +
               2 * (col % 16) + (row & 1)] =
          hs[static_cast<std::size_t>(row0 + row) * hs_stride + col];
    }
  }
}

void amx_gemm_1c_baseline(const bf16* a, int rows_padded,
                          int reduction_padded,
                          const std::vector<bf16>& packed_b,
                          int output_padded, float* c) {
  const int output_blocks = output_padded / 16;
  alignas(64) float tmp[256];
  for (int row = 0; row < rows_padded; row += 16) {
    for (int col = 0; col < output_padded; col += 16) {
      _tile_zero(0);
      for (int k = 0; k < reduction_padded; k += 32) {
        _tile_loadd(1, a + static_cast<std::size_t>(row) * reduction_padded + k,
                    reduction_padded * static_cast<int>(sizeof(bf16)));
        const bf16* b = packed_b.data() +
            (static_cast<std::size_t>(k / 32) * output_blocks + col / 16) *
                512;
        _tile_loadd(2, b, 64);
        _tile_dpbf16ps(0, 1, 2);
      }
      _tile_stored(0, tmp, 64);
      for (int i = 0; i < 16; ++i) {
        std::memcpy(c + static_cast<std::size_t>(row + i) * output_padded + col,
                    tmp + i * 16, 64);
      }
    }
  }
}

void amx_gemm_4c_epilogue(const bf16* a, int rows_padded,
                          int reduction_padded,
                          const std::vector<bf16>& packed_b,
                          int output_padded, int output_stride,
                          int global_row0, int valid_rows,
                          int logical_output, const float* scale,
                          const float* bias, float* c) {
  if (rows_padded % 16 != 0 || reduction_padded % 32 != 0 ||
      output_padded % 16 != 0 || valid_rows < 0 || valid_rows > rows_padded ||
      logical_output < 1 || logical_output > output_padded ||
      output_stride < logical_output) {
    throw std::invalid_argument("AMX 4c forward shape unsupported");
  }
  const int output_blocks = output_padded / 16;
  alignas(64) float tmp[4][256];
  for (int row = 0; row < rows_padded; row += 16) {
    for (int col = 0; col < output_padded; col += 64) {
      const int active = std::min(
          4, std::max(0, (logical_output - col + 15) / 16));
      if (active == 0) break;
      if (active >= 1) _tile_zero(0);
      if (active >= 2) _tile_zero(1);
      if (active >= 3) _tile_zero(2);
      if (active >= 4) _tile_zero(3);
      for (int k = 0; k < reduction_padded; k += 32) {
        _tile_loadd(4, a + static_cast<std::size_t>(row) * reduction_padded + k,
                    reduction_padded * static_cast<int>(sizeof(bf16)));
        const std::size_t block_base =
            static_cast<std::size_t>(k / 32) * output_blocks + col / 16;
        for (int q = 0; q < active; ++q) {
          const bf16* b = packed_b.data() + (block_base + q) * 512;
          _tile_loadd(6, b, 64);
          switch (q) {
            case 0: _tile_dpbf16ps(0, 4, 6); break;
            case 1: _tile_dpbf16ps(1, 4, 6); break;
            case 2: _tile_dpbf16ps(2, 4, 6); break;
            default: _tile_dpbf16ps(3, 4, 6); break;
          }
        }
      }
      if (active >= 1) _tile_stored(0, tmp[0], 64);
      if (active >= 2) _tile_stored(1, tmp[1], 64);
      if (active >= 3) _tile_stored(2, tmp[2], 64);
      if (active >= 4) _tile_stored(3, tmp[3], 64);
      for (int i = 0; i < 16 && row + i < valid_rows; ++i) {
        const __m512 sv = _mm512_set1_ps(scale[global_row0 + row + i]);
        for (int q = 0; q < active; ++q) {
          const int out_col = col + q * 16;
          if (out_col >= logical_output) break;
          const int lanes = std::min(16, logical_output - out_col);
          float* dst = c + static_cast<std::size_t>(row + i) * output_stride +
              out_col;
          const __m512 value = _mm512_load_ps(tmp[q] + i * 16);
          const __mmask16 mask = static_cast<__mmask16>(
              lanes == 16 ? 0xffffu : ((1u << lanes) - 1u));
          const __m512 bias_v = lanes == 16
              ? _mm512_loadu_ps(bias + out_col)
              : _mm512_maskz_loadu_ps(mask, bias + out_col);
          const __m512 result = _mm512_fmadd_ps(
              value, sv, bias_v);
          if (lanes == 16) {
            _mm512_storeu_ps(dst, result);
          } else {
            _mm512_mask_storeu_ps(dst, mask, result);
          }
        }
      }
    }
  }
}

void dh_amx_4c2a2b(const bf16* ybar, int rows_padded, int valid_rows,
                   int d_padded, const bf16* packed_wt, int logical_k,
                   int k_padded,
                   float* dh, int global_row0, const float* target_scale,
                   bool double_buffer, KernelCounters* counters) {
  const int output_blocks = k_padded / 16;
  alignas(64) float tmp[4][256];
  for (int row = 0; row < rows_padded; row += 16) {
    for (int col = 0; col < k_padded; col += 64) {
      const int active = std::min(4, std::max(0, (logical_k - col + 15) / 16));
      if (active == 0) break;
      if (active >= 1) _tile_zero(0);
      if (active >= 2) _tile_zero(1);
      if (active >= 3) _tile_zero(2);
      if (active >= 4) _tile_zero(3);
      bool a_alt = false;
      load_a_tile(false, ybar + static_cast<std::size_t>(row) * d_padded,
                  d_padded * static_cast<int>(sizeof(bf16)));
      if (counters != nullptr) {
        ++counters->a_tile_loads;
        counters->y_bytes_read += 1024;
      }
      for (int k = 0; k < d_padded; k += 32) {
        for (int q = 0; q < active; ++q) {
          const bool b_alt = double_buffer && ((q & 1) != 0);
          const bf16* b = packed_wt +
              (static_cast<std::size_t>(k / 32) * output_blocks +
               (col / 16) + q) * 512;
          load_b_and_dp(q, b_alt, b, a_alt);
          if (counters != nullptr) {
            ++counters->b_tile_loads;
            ++counters->dpbf16ps_calls;
          }
        }
        if (k + 32 < d_padded) {
          const bool next_alt = double_buffer ? !a_alt : false;
          load_a_tile(next_alt,
                      ybar + static_cast<std::size_t>(row) * d_padded + k + 32,
                      d_padded * static_cast<int>(sizeof(bf16)));
          a_alt = next_alt;
          if (counters != nullptr) {
            ++counters->a_tile_loads;
            counters->y_bytes_read += 1024;
          }
        }
      }
      if (target_scale == nullptr) {
        store_c_tiles(dh + static_cast<std::size_t>(global_row0 + row) *
                               k_padded + col,
                      k_padded, active);
      } else {
        store_c_to_scratch(tmp, active);
        for (int i = 0; i < 16 && row + i < valid_rows; ++i) {
          const __m512 scale =
              _mm512_set1_ps(target_scale[global_row0 + row + i]);
          for (int q = 0; q < active; ++q) {
            const __m512 value = _mm512_load_ps(tmp[q] + i * 16);
            _mm512_storeu_ps(
                dh + static_cast<std::size_t>(global_row0 + row + i) *
                         k_padded + col + q * 16,
                _mm512_mul_ps(value, scale));
          }
        }
      }
      if (counters != nullptr) {
        counters->c_tile_stores += active;
        counters->accumulator_bytes += static_cast<std::uint64_t>(active) * 1024;
      }
    }
  }
}

void dh_amx_4c2a2b_bf16(const bf16* ybar, int rows_padded, int valid_rows,
                        int d_padded, const bf16* packed_wt, int logical_k,
                        int k_padded, bf16* dh, int global_row0,
                        bool double_buffer, KernelCounters* counters) {
  const int output_blocks = k_padded / 16;
  alignas(64) float tmp[4][256];
  for (int row = 0; row < rows_padded; row += 16) {
    for (int col = 0; col < k_padded; col += 64) {
      const int active = std::min(4, std::max(0, (logical_k - col + 15) / 16));
      if (active == 0) break;
      if (active >= 1) _tile_zero(0);
      if (active >= 2) _tile_zero(1);
      if (active >= 3) _tile_zero(2);
      if (active >= 4) _tile_zero(3);
      bool a_alt = false;
      load_a_tile(false, ybar + static_cast<std::size_t>(row) * d_padded,
                  d_padded * static_cast<int>(sizeof(bf16)));
      if (counters != nullptr) {
        ++counters->a_tile_loads;
        counters->y_bytes_read += 1024;
      }
      for (int reduction = 0; reduction < d_padded; reduction += 32) {
        for (int q = 0; q < active; ++q) {
          const bool b_alt = double_buffer && ((q & 1) != 0);
          const bf16* b = packed_wt +
              (static_cast<std::size_t>(reduction / 32) * output_blocks +
               (col / 16) + q) * 512;
          load_b_and_dp(q, b_alt, b, a_alt);
          if (counters != nullptr) {
            ++counters->b_tile_loads;
            ++counters->dpbf16ps_calls;
          }
        }
        if (reduction + 32 < d_padded) {
          const bool next_alt = double_buffer ? !a_alt : false;
          load_a_tile(next_alt,
                      ybar + static_cast<std::size_t>(row) * d_padded +
                          reduction + 32,
                      d_padded * static_cast<int>(sizeof(bf16)));
          a_alt = next_alt;
          if (counters != nullptr) {
            ++counters->a_tile_loads;
            counters->y_bytes_read += 1024;
          }
        }
      }
      store_c_to_scratch(tmp, active);
      for (int i = 0; i < 16 && row + i < valid_rows; ++i) {
        bf16* dst = dh + static_cast<std::size_t>(global_row0 + row + i) *
                             k_padded + col;
        for (int q = 0; q < active; ++q) {
          const __m512 value = _mm512_load_ps(tmp[q] + i * 16);
          _mm256_storeu_si256(
              reinterpret_cast<__m256i*>(dst + q * 16),
              _mm512_cvtneps_pbh(value));
        }
      }
      if (counters != nullptr) {
        counters->c_tile_stores += active;
        counters->accumulator_bytes += static_cast<std::uint64_t>(active) * 1024;
      }
    }
  }
}

void dw_naive_baseline(const bf16* ybar_t, int d_padded, int rows_padded,
                       const std::vector<bf16>& packed_h, int k_padded,
                       float* local_dwt) {
  alignas(64) float tmp[256];
  const int output_blocks = k_padded / 16;
  for (int row = 0; row < rows_padded; row += 32) {
    for (int d = 0; d < d_padded; d += 16) {
      for (int col = 0; col < k_padded; col += 16) {
        _tile_zero(0);
        _tile_loadd(1, ybar_t + static_cast<std::size_t>(d) * rows_padded + row,
                    rows_padded * static_cast<int>(sizeof(bf16)));
        _tile_loadd(2, packed_h.data() +
                           (static_cast<std::size_t>(row / 32) * output_blocks +
                            col / 16) * 512,
                    64);
        _tile_dpbf16ps(0, 1, 2);
        _tile_stored(0, tmp, 64);
        for (int i = 0; i < 16; ++i) {
          for (int j = 0; j < 16; ++j) {
            local_dwt[static_cast<std::size_t>(d + i) * k_padded + col + j] +=
                tmp[i * 16 + j];
          }
        }
      }
    }
  }
}

void dw_twolevel_4c1a1b_baseline(
    const bf16* ybar_t, int d_padded, int rows_padded,
    const std::vector<bf16>& packed_h, int k_padded, int tile_panel,
    int reduction_rows, float* local_dwt) {
  alignas(64) float tmp[4][256];
  alignas(64) bf16 a_tmp[16 * 32];
  const int output_blocks = k_padded / 16;
  for (int d = 0; d < d_padded; d += 16) {
    for (int block0 = 0; block0 < output_blocks; block0 += tile_panel) {
      const int active = std::min(tile_panel, output_blocks - block0);
      if (active >= 1) _tile_zero(0);
      if (active >= 2) _tile_zero(1);
      if (active >= 3) _tile_zero(2);
      if (active >= 4) _tile_zero(3);
      for (int row = 0; row < rows_padded; row += reduction_rows) {
        if (reduction_rows == 32) {
          _tile_loadd(4,
                      ybar_t + static_cast<std::size_t>(d) * rows_padded + row,
                      rows_padded * static_cast<int>(sizeof(bf16)));
        } else {
          std::fill(a_tmp, a_tmp + 16 * 32, bf16(0));
          for (int i = 0; i < 16; ++i) {
            std::copy_n(ybar_t + static_cast<std::size_t>(d + i) * rows_padded +
                            row,
                        16, a_tmp + i * 32);
          }
          _tile_loadd(4, a_tmp, 64);
        }
        const bf16* b = packed_h.data() +
            (static_cast<std::size_t>(row / reduction_rows) * output_blocks +
             block0) * 512;
        for (int q = 0; q < active; ++q) {
          _tile_loadd(5, b + static_cast<std::size_t>(q) * 512, 64);
          switch (q) {
            case 0: _tile_dpbf16ps(0, 4, 5); break;
            case 1: _tile_dpbf16ps(1, 4, 5); break;
            case 2: _tile_dpbf16ps(2, 4, 5); break;
            default: _tile_dpbf16ps(3, 4, 5); break;
          }
        }
      }
      if (active >= 1) _tile_stored(0, tmp[0], 64);
      if (active >= 2) _tile_stored(1, tmp[1], 64);
      if (active >= 3) _tile_stored(2, tmp[2], 64);
      if (active >= 4) _tile_stored(3, tmp[3], 64);
      for (int q = 0; q < active; ++q) {
        for (int i = 0; i < 16; ++i) {
          for (int j = 0; j < 16; ++j) {
            local_dwt[static_cast<std::size_t>(d + i) * k_padded +
                      (block0 + q) * 16 + j] += tmp[q][i * 16 + j];
          }
        }
      }
    }
  }
}

void dw_amx_4c2a2b(const bf16* ybar_t, int d_padded, int rows_padded,
                   const bf16* packed_h, int logical_k, int k_padded,
                   float* local_dwt,
                   bool double_buffer, KernelCounters* counters) {
  const int output_blocks = k_padded / 16;
  for (int d = 0; d < d_padded; d += 16) {
    for (int col = 0; col < k_padded; col += 64) {
      const int active = std::min(4, std::max(0, (logical_k - col + 15) / 16));
      if (active == 0) break;
      float* c = local_dwt + static_cast<std::size_t>(d) * k_padded + col;
      load_c_tiles(c, k_padded, active);
      if (counters != nullptr) {
        counters->c_tile_loads += active;
        counters->accumulator_bytes += static_cast<std::uint64_t>(active) * 1024;
      }
      bool a_alt = false;
      load_a_tile(false, ybar_t + static_cast<std::size_t>(d) * rows_padded,
                  rows_padded * static_cast<int>(sizeof(bf16)));
      if (counters != nullptr) {
        ++counters->a_tile_loads;
        counters->y_bytes_read += 1024;
      }
      for (int row = 0; row < rows_padded; row += 32) {
        for (int q = 0; q < active; ++q) {
          const bool b_alt = double_buffer && ((q & 1) != 0);
          const bf16* b = packed_h +
              (static_cast<std::size_t>(row / 32) * output_blocks +
               col / 16 + q) * 512;
          load_b_and_dp(q, b_alt, b, a_alt);
          if (counters != nullptr) {
            ++counters->b_tile_loads;
            ++counters->dpbf16ps_calls;
            counters->h_bytes_read += 1024;
          }
        }
        if (row + 32 < rows_padded) {
          const bool next_alt = double_buffer ? !a_alt : false;
          load_a_tile(next_alt,
                      ybar_t + static_cast<std::size_t>(d) * rows_padded +
                          row + 32,
                      rows_padded * static_cast<int>(sizeof(bf16)));
          a_alt = next_alt;
          if (counters != nullptr) {
            ++counters->a_tile_loads;
            counters->y_bytes_read += 1024;
          }
        }
      }
      store_c_tiles(c, k_padded, active);
      if (counters != nullptr) {
        counters->c_tile_stores += active;
        counters->accumulator_bytes += static_cast<std::uint64_t>(active) * 1024;
      }
    }
  }
}

}  // namespace tfs::backward_v2
