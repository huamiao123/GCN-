#include "tfs/kernel/backward_v2.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

namespace tfs::backward_v2 {

void transpose_panel_scalar_t1(const bf16* y, int rows_padded, int d_padded,
                               bf16* y_t) {
  for (int row = 0; row < rows_padded; ++row) {
    for (int col = 0; col < d_padded; ++col) {
      y_t[static_cast<std::size_t>(col) * rows_padded + row] =
          y[static_cast<std::size_t>(row) * d_padded + col];
    }
  }
}

namespace {

// Transpose a dense 16x16 block of 16-bit values. AVX2 unpack operations are
// used inside an AVX-512-enabled translation unit because this shape maps
// exactly to sixteen 256-bit rows without masked tails.
inline void transpose16x16_u16(const bf16* src, int src_stride, bf16* dst,
                               int dst_stride) {
  __m256i x[16];
  __m256i a[16];
  __m256i b[16];
  __m256i c[16];
  __m256i out[16];
  for (int i = 0; i < 16; ++i) {
    x[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
        src + static_cast<std::size_t>(i) * src_stride));
  }
  for (int i = 0; i < 16; i += 2) {
    a[i] = _mm256_unpacklo_epi16(x[i], x[i + 1]);
    a[i + 1] = _mm256_unpackhi_epi16(x[i], x[i + 1]);
  }
  for (int i = 0; i < 16; i += 4) {
    b[i] = _mm256_unpacklo_epi32(a[i], a[i + 2]);
    b[i + 1] = _mm256_unpackhi_epi32(a[i], a[i + 2]);
    b[i + 2] = _mm256_unpacklo_epi32(a[i + 1], a[i + 3]);
    b[i + 3] = _mm256_unpackhi_epi32(a[i + 1], a[i + 3]);
  }
  for (int i = 0; i < 16; i += 8) {
    c[i] = _mm256_unpacklo_epi64(b[i], b[i + 4]);
    c[i + 1] = _mm256_unpackhi_epi64(b[i], b[i + 4]);
    c[i + 2] = _mm256_unpacklo_epi64(b[i + 1], b[i + 5]);
    c[i + 3] = _mm256_unpackhi_epi64(b[i + 1], b[i + 5]);
    c[i + 4] = _mm256_unpacklo_epi64(b[i + 2], b[i + 6]);
    c[i + 5] = _mm256_unpackhi_epi64(b[i + 2], b[i + 6]);
    c[i + 6] = _mm256_unpacklo_epi64(b[i + 3], b[i + 7]);
    c[i + 7] = _mm256_unpackhi_epi64(b[i + 3], b[i + 7]);
  }
  for (int i = 0; i < 8; ++i) {
    out[i] = _mm256_permute2x128_si256(c[i], c[i + 8], 0x20);
    out[i + 8] = _mm256_permute2x128_si256(c[i], c[i + 8], 0x31);
  }
  for (int i = 0; i < 16; ++i) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(
        dst + static_cast<std::size_t>(i) * dst_stride), out[i]);
  }
}

}  // namespace

void transpose_y_avx512_t2(const bf16* y, int rows_padded, int d_padded,
                           bf16* y_t) {
  if ((rows_padded % 16) != 0 || (d_padded % 16) != 0) {
    throw std::invalid_argument("T2 requires 16-element padded dimensions");
  }
  for (int row = 0; row < rows_padded; row += 16) {
    for (int col = 0; col < d_padded; col += 16) {
      transpose16x16_u16(y + static_cast<std::size_t>(row) * d_padded + col,
                         d_padded,
                         y_t + static_cast<std::size_t>(col) * rows_padded + row,
                         rows_padded);
    }
  }
}

namespace {

inline void transpose16x16_u16_avx512_gather(const bf16* src, int src_stride,
                                             bf16* dst, int dst_stride) {
  alignas(64) std::int32_t offsets_data[16];
  for (int row = 0; row < 16; ++row) offsets_data[row] = row * src_stride;
  const __m512i offsets = _mm512_load_si512(offsets_data);
  for (int col = 0; col < 16; col += 2) {
    const __m512i pairs = _mm512_i32gather_epi32(offsets, src + col, 2);
    const __m256i even = _mm512_cvtepi32_epi16(pairs);
    const __m256i odd =
        _mm512_cvtepi32_epi16(_mm512_srli_epi32(pairs, 16));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(
        dst + static_cast<std::size_t>(col) * dst_stride), even);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(
        dst + static_cast<std::size_t>(col + 1) * dst_stride), odd);
  }
}

}  // namespace

void transpose_y_avx512_gather_t2(const bf16* y, int rows_padded,
                                  int d_padded, bf16* y_t) {
  if ((rows_padded % 16) != 0 || (d_padded % 16) != 0) {
    throw std::invalid_argument("AVX-512 T2 requires padded dimensions");
  }
  for (int row = 0; row < rows_padded; row += 16) {
    for (int col = 0; col < d_padded; col += 16) {
      transpose16x16_u16_avx512_gather(
          y + static_cast<std::size_t>(row) * d_padded + col, d_padded,
          y_t + static_cast<std::size_t>(col) * rows_padded + row,
          rows_padded);
    }
  }
}

void transpose_y_block_to_panel_t3(const bf16* y_block, int rows_padded,
                                   int d_padded, bf16* y_t_panel,
                                   int panel_stride, int panel_row_offset) {
  if (rows_padded != 32 || (d_padded % 16) != 0 ||
      (panel_row_offset % 16) != 0) {
    throw std::invalid_argument("T3 requires a padded 32-row producer block");
  }
  for (int row = 0; row < rows_padded; row += 16) {
    for (int col = 0; col < d_padded; col += 16) {
      transpose16x16_u16(
          y_block + static_cast<std::size_t>(row) * d_padded + col, d_padded,
          y_t_panel + static_cast<std::size_t>(col) * panel_stride +
              panel_row_offset + row,
          panel_stride);
    }
  }
}

void scale_bf16_db_transpose_t3(const float* grad, const float* scale,
                                int global_row0, int valid_rows, int d,
                                int d_padded, int panel_stride,
                                int panel_row_offset, bf16* y_rowmajor,
                                bf16* y_t_panel, float* db_local) {
  if (valid_rows < 0 || valid_rows > 32 || d < 1 || d > d_padded ||
      (d_padded % 32) != 0 || (panel_stride % 16) != 0 ||
      (panel_row_offset % 16) != 0) {
    throw std::invalid_argument(
        "fused scale/transpose requires a padded 32-row block");
  }

  // Keep the conversion source in a tiny L1-resident block.  The caller still
  // receives the row-major block for dh_amx_4c2a2b, but the panel transpose no
  // longer rereads that block from the much larger worker scratch allocation.
  alignas(64) bf16 block[16][32];
  for (int row_base = 0; row_base < 32; row_base += 16) {
    for (int col = 0; col < d_padded; col += 32) {
      for (int r = 0; r < 16; ++r) {
        const int local_row = row_base + r;
        bf16* scratch = block[r];
        const int global_row = global_row0 + local_row;
        if (local_row >= valid_rows) {
          std::memset(scratch, 0, sizeof(block[r]));
        } else {
          const float* gr = grad + static_cast<std::size_t>(global_row) * d;
          const float sv = scale[global_row];
          const __m512 svv = _mm512_set1_ps(sv);
          const int n0 = std::max(0, std::min(16, d - col));
          const int n1 = std::max(0, std::min(16, d - (col + 16)));
          const __mmask16 m0 = static_cast<__mmask16>(
              n0 == 16 ? 0xffffu : ((1u << n0) - 1u));
          const __mmask16 m1 = static_cast<__mmask16>(
              n1 == 16 ? 0xffffu : ((1u << n1) - 1u));
          const float* p0 = gr + (col < d ? col : d);
          const float* p1 = gr + (col + 16 < d ? col + 16 : d);
          const __m512 g0 = _mm512_maskz_loadu_ps(m0, p0);
          const __m512 g1 = _mm512_maskz_loadu_ps(m1, p1);
          if (db_local != nullptr) {
            if (n0 != 0) {
              const __m512 old = _mm512_loadu_ps(db_local + col);
              _mm512_mask_storeu_ps(db_local + col, m0,
                                    _mm512_add_ps(old, g0));
            }
            if (n1 != 0) {
              const __m512 old = _mm512_loadu_ps(db_local + col + 16);
              _mm512_mask_storeu_ps(db_local + col + 16, m1,
                                    _mm512_add_ps(old, g1));
            }
          }
          _mm512_storeu_si512(
              reinterpret_cast<void*>(scratch),
              (__m512i)_mm512_cvtne2ps_pbh(
                  _mm512_mul_ps(g1, svv), _mm512_mul_ps(g0, svv)));
        }
        std::memcpy(y_rowmajor + static_cast<std::size_t>(local_row) *
                        d_padded + col,
                    scratch, sizeof(block[r]));
      }

      transpose16x16_u16(
          &block[0][0], 32,
          y_t_panel + static_cast<std::size_t>(col) * panel_stride +
              panel_row_offset + row_base,
          panel_stride);
      transpose16x16_u16(
          &block[0][16], 32,
          y_t_panel + static_cast<std::size_t>(col + 16) * panel_stride +
              panel_row_offset + row_base,
          panel_stride);
    }
  }
}

std::string_view y_layout_name(YLayoutMode mode) {
  switch (mode) {
    case YLayoutMode::T1Scalar: return "T1";
    case YLayoutMode::T2VectorBlocked: return "T2-vector";
    case YLayoutMode::T2Avx512Gather: return "T2-avx512";
    case YLayoutMode::T3ProducerDual: return "T3";
  }
  return "unknown";
}

}  // namespace tfs::backward_v2
