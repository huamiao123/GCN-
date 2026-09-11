#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tfs::backward_v2 {

using bf16 = std::uint16_t;

enum class YLayoutMode {
  T1Scalar,
  T2VectorBlocked,
  T2Avx512Gather,
  T3ProducerDual
};

struct BackwardV2Config {
  int row_panel = 512;
  int reduction_rows = 32;
  int tile_panel = 4;
  YLayoutMode y_layout = YLayoutMode::T2VectorBlocked;
  bool dh_double_buffer = true;
  bool dw_double_buffer = true;
  bool direct_dh_store = true;
  bool direct_h_pack = true;
};

struct KernelCounters {
  std::uint64_t a_tile_loads = 0;
  std::uint64_t b_tile_loads = 0;
  std::uint64_t c_tile_loads = 0;
  std::uint64_t c_tile_stores = 0;
  std::uint64_t dpbf16ps_calls = 0;
  std::uint64_t y_bytes_read = 0;
  std::uint64_t h_bytes_read = 0;
  std::uint64_t accumulator_bytes = 0;
};

void configure_amx_tiles_16x64();

// Baseline layout retained as the elementwise correctness oracle.
void transpose_panel_scalar_t1(const bf16* y, int rows_padded, int d_padded,
                               bf16* y_t);

// 16x16 blocked vector transpose. The input and output include zero padding.
void transpose_y_avx512_t2(const bf16* y, int rows_padded, int d_padded,
                           bf16* y_t);

// Transpose a logical row-major matrix with an unpadded source stride directly
// into a padded D x rows panel.  Only the final row/column tiles use a small
// zero-filled staging block; full 16x16 tiles never materialize a padded copy.
void transpose_y_avx512_tail_t4(const bf16* y, int valid_rows, int logical_d,
                                int rows_padded, int d_padded, bf16* y_t);

// Literal AVX-512 implementation used as an ISA-width ablation. The selector
// may still prefer the 256-bit unpack network when it is faster for 16 BF16s.
void transpose_y_avx512_gather_t2(const bf16* y, int rows_padded,
                                  int d_padded, bf16* y_t);

// T3 producer helper: transpose one freshly produced 32-row micro-panel into
// its final offset inside the panel-wide D x P layout.
void transpose_y_block_to_panel_t3(const bf16* y_block, int rows_padded,
                                   int d_padded, bf16* y_t_panel,
                                   int panel_stride, int panel_row_offset);

// Produce one 32-row BF16 gradient block and its transposed panel layout in
// one pass.  The row-major block is retained for the dP/dH AMX kernel, while
// the small 16x16 transpose is performed from an L1-sized conversion buffer
// instead of rereading the complete row-major block.  ``db_local`` may be
// null when the caller does not need a bias-gradient accumulation.
void scale_bf16_db_transpose_t3(const float* grad, const float* scale,
                                int global_row0, int valid_rows,
                                int grad_row_stride, int d,
                                int d_padded, int panel_stride,
                                int panel_row_offset, bf16* y_rowmajor,
                                bf16* y_t_panel, float* db_local);

std::string_view y_layout_name(YLayoutMode mode);

// Packs Hs[r0:r0+valid_rows, :] directly from the global row-major tensor.
// The output is VNNI-packed in [32 reduction rows, 16 output columns] blocks.
void pack_h_panel_direct(const bf16* hs, int hs_stride, int row0,
                         int valid_rows, int rows_padded, int k_padded,
                         bf16* packed_h);

// Computes ybar[rows_padded,D] * W^T[D,K]. dH uses global row stride
// k_padded. When target_scale is non-null, only valid rows are written and
// each row is scaled by target_scale[global_row].
void dh_amx_4c2a2b(const bf16* ybar, int rows_padded, int valid_rows,
                   int d_padded, const bf16* packed_wt, int logical_k,
                   int k_padded,
                   float* dh, int global_row0, const float* target_scale,
                   bool double_buffer, KernelCounters* counters,
                   bool accumulate = false);

// BF16-output sibling used by the High-D Q-first path.  The AMX accumulator
// is rounded once at the dense-to-sparse boundary, avoiding an N×K FP32
// materialization that would immediately be converted to BF16 by CSR pull.
void dh_amx_4c2a2b_bf16(const bf16* ybar, int rows_padded, int valid_rows,
                        int d_padded, const bf16* packed_wt, int logical_k,
                        int k_padded, bf16* dh, int global_row0,
                        bool double_buffer, KernelCounters* counters);

// Accumulates one row panel directly into a thread-private dW^T[D,K].
// C tiles are loaded from and stored to local_dwt once per output panel.
void dw_amx_4c2a2b(const bf16* ybar_t, int d_padded, int rows_padded,
                   const bf16* packed_h, int logical_k, int k_padded,
                   float* local_dwt,
                   bool double_buffer, KernelCounters* counters);

}  // namespace tfs::backward_v2
