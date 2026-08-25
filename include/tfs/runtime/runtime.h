#pragma once
#include <cstdint>
#include <string>
#include "tfs/model/model_config.h"
#include "tfs/graph/schedule_plan.h"
namespace tfs {
struct EnvironmentInfo { std::string hostname, kernel, cpu_model, compiler, build_flags; int online_cpus=0, numa_nodes=0; bool cpu_avx512=false, cpu_amx_tile=false, cpu_amx_bf16=false; };
EnvironmentInfo ProbeBasicEnvironment();
struct MemoryBudget {
  uint64_t graph_bytes=0,node_tensor_bytes=0,weight_state_bytes=0,weight_master_fp32_bytes=0,
    weight_plain_bf16_bytes=0,weight_pack_fwd_bytes=0,weight_pack_bwd_wt_bytes=0,
    grad_fp32_bytes=0,optimizer_m_fp32_bytes=0,optimizer_v_fp32_bytes=0,
    thread_private_bytes=0,z_strategy_extra_bytes=0,q_strategy_extra_bytes=0,estimated_peak_bytes=0;
  double estimated_peak_gib=0.0; std::string ToJson() const;
};
Status PlanMemory(const TrainConfig& config,const GraphCSR& graph,MemoryBudget* budget);
class RunRecorder {
 public: Status Create(const TrainConfig& config,const EnvironmentInfo& env,std::string* run_dir); Status WriteStatus(const std::string& status,const std::string& message); const std::string& run_dir() const{return run_dir_;}
 private: std::string run_dir_;
};
class Trainer {
 public: Status Prepare(const TrainConfig&,const GraphCSR&,const ModelConfig&,RunRecorder*); Status DryRun(); Status TrainStep(){return Status::NotImplemented("training.TrainStep: C0 kernel intentionally not implemented");}
 private: bool prepared_=false;
};
}
