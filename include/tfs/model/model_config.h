#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "tfs/common/types.h"
namespace tfs {
struct GcnLayerConfig { int64_t input_dim=0, output_dim=0; bool use_bias=true, use_relu=false; double dropout=0.0; };
struct ModelConfig { int64_t num_nodes=0, num_classes=0; std::vector<GcnLayerConfig> layers; Status Validate() const; };
struct LayerShape { int64_t input_dim=0, output_dim=0; };
struct WeightState { LayerShape shape; uint64_t master_fp32_bytes=0, plain_bf16_bytes=0, packed_fwd_w_bytes=0, packed_bwd_wt_bytes=0, grad_fp32_bytes=0, adam_m_fp32_bytes=0, adam_v_fp32_bytes=0, version=0; Status ValidateMetadata() const; };
struct TrainConfig {
  std::string config_path, raw_json, project_name, run_name, graph_path, graph_format, run_root;
  int64_t seed=0, input_dim=0, hidden_dim=0, num_classes=0, num_layers=0; int threads=1;
  bool directed=false, bidirectional=false, full_batch=true, require_amx=true, allow_runtime_fallback=false, dry_run=true;
  std::string degree_sort_mode, optimizer; ZStrategy z_strategy=ZStrategy::kRecompute; QStrategy q_strategy=QStrategy::kHybrid;
  NormalizationStrategy normalization=NormalizationStrategy::kFactorizedOnTheFly;
  ModelConfig Model(int64_t nodes) const; Status Validate() const;
};
Status LoadConfigJson(const std::string& path, TrainConfig* config);
}

