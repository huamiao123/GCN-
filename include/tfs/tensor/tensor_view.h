#pragma once
#include <cstdint>
#include "tfs/common/types.h"
namespace tfs {
struct TensorView {
  void* data=nullptr; DType dtype=DType::kFP32; MatrixLayout layout=MatrixLayout::kRowMajor;
  NodeOrder node_order=NodeOrder::kOriginal; int64_t rows=0, cols=0, stride_bytes=0;
  bool IsContiguousRowMajor() const;
  Status Validate() const;
};
}

