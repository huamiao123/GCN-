#pragma once

#include <vector>

#include "tfs/kernel/backward_v2.h"

namespace tfs::backward_v2 {

std::vector<bf16> pack_rhs_baseline(const bf16* b, int reduction_padded,
                                    int output_padded);
std::vector<bf16> pack_rhs_reduction_baseline(const bf16* b,
                                              int reduction_padded,
                                              int output_padded,
                                              int reduction_rows);

void amx_gemm_1c_baseline(const bf16* a, int rows_padded,
                          int reduction_padded,
                          const std::vector<bf16>& packed_b,
                          int output_padded, float* c);

// Four-output-tile forward GEMM.  One A tile is reused for four adjacent
// 16-column output tiles; the FP32 accumulator is scaled and biased directly
// into C, avoiding the intermediate BF16 aggregate and a second epilogue
// pass.  The caller supplies a complete 16-row block (valid_rows may be a
// tail smaller than 16 when the source block is zero-padded).
void amx_gemm_4c_epilogue(const bf16* a, int rows_padded,
                          int reduction_padded,
                          const std::vector<bf16>& packed_b,
                          int output_padded, int output_stride,
                          int global_row0, int valid_rows,
                          int logical_output, const float* scale,
                          const float* bias, float* c);

void dw_naive_baseline(const bf16* ybar_t, int d_padded, int rows_padded,
                       const std::vector<bf16>& packed_h, int k_padded,
                       float* local_dwt);

void dw_twolevel_4c1a1b_baseline(
    const bf16* ybar_t, int d_padded, int rows_padded,
    const std::vector<bf16>& packed_h, int k_padded, int tile_panel,
    int reduction_rows, float* local_dwt);

}  // namespace tfs::backward_v2
