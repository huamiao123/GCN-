#pragma once

#include "../../amx/backward_v2_kernels.h"

namespace tfs::backward_v2 {

void pack_h_panel_direct_tail(const bf16* hs, int hs_stride, int logical_k,
                              int row0, int valid_rows, int rows_padded,
                              int k_padded, bf16* packed_h);

}  // namespace tfs::backward_v2
