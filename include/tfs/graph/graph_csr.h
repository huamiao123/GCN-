#pragma once
#include <cstdint>
#include <vector>
#include "tfs/common/types.h"
namespace tfs {
struct GraphCSR {
  int64_t num_nodes=0, num_edges=0; std::vector<int64_t> row_ptr; std::vector<int32_t> col_idx;
  std::vector<float> degree, inv_sqrt_degree; bool has_self_loops=false, is_bidirectional=false;
  bool is_symmetric_normalized=false; NodeOrder node_order=NodeOrder::kOriginal;
  Status ValidateStructure() const;
};
Status LoadGraphJson(const std::string& path, GraphCSR* graph);
}

