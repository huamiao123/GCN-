#pragma once
#include <cstdint>
#include <vector>
#include "tfs/graph/graph_csr.h"
namespace tfs {
struct NodeBlock { int64_t logical_begin=0; int32_t row_count=0; };
struct SchedulePlan {
  int32_t block_rows=16; std::vector<int32_t> schedule_order, inverse_schedule_order; std::vector<NodeBlock> blocks;
  std::vector<int32_t> low_degree_nodes, medium_degree_nodes, high_degree_nodes; NodeOrder data_order=NodeOrder::kOriginal;
  Status Validate(int64_t num_nodes) const;
};
Status BuildLogicalDegreeSchedule(const GraphCSR& graph, SchedulePlan* schedule);
}

