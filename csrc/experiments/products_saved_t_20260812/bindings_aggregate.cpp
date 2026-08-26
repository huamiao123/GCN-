#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> c2_forward(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&);
std::vector<at::Tensor> c2_backward(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&);
std::vector<at::Tensor> c2_backward_selective(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,bool);
at::Tensor c2_pull_spmm(const at::Tensor&,const at::Tensor&,const at::Tensor&);
std::vector<at::Tensor> c3_backward_amx_v2(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);
std::vector<at::Tensor> c3_forward_amx_v2(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);
std::vector<at::Tensor> c3_forward_aggregate_saved_amx_v4(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
at::Tensor c3_prepare_static_hs_v1(const at::Tensor&,const at::Tensor&,int64_t);
at::Tensor c3_prepare_static_hs_padded_v2(const at::Tensor&,const at::Tensor&,int64_t,int64_t);
at::Tensor c3_replicate_static_hs_numa_v1(const at::Tensor&,int64_t);
at::Tensor c3_prepare_static_aggregate_v3(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_forward_cached_hs_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);
std::vector<at::Tensor> c3_forward_cached_hs_padded_amx_v2(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);
std::vector<at::Tensor> c3_forward_cached_aggregate_amx_v3(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_backward_cached_aggregate_amx_v3(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_forward_aggregate_wide_amx_v3(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_forward_aggregate_wide_cached_hs_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
at::Tensor c3_pull_only_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
at::Tensor c3_pull_only_bf16_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
at::Tensor c3_pull_only_bf16_scaled_fp32_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
at::Tensor c3_scale_grad_bf16_v1(const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_scale_grad_bf16_db_v2(const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_forward_wide_amx_v3(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_forward_wide_cached_hs_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_forward_saved_t_amx_v3(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_backward_saved_t_amx_v3(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t);
std::vector<at::Tensor> c3_backward_aggregate_saved_amx_v4(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);
std::vector<at::Tensor> c3_backward_wide_amx_v3(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);
std::vector<at::Tensor> c3_backward_aggregate_highd_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);
std::vector<at::Tensor> c3_backward_transform_highd_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);
std::vector<at::Tensor> c3_backward_transform_highd_single_scan_amx_v1(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,bool);

std::string architecture_version() { return "TFS-Train-v2-single-runtime"; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("architecture_version", &architecture_version);
  m.def("c2_pull_spmm", &c2_pull_spmm);
  m.def("c2_forward", &c2_forward);
  m.def("c2_backward", &c2_backward);
  m.def("c2_backward_selective", &c2_backward_selective);
  m.def("c3_backward_amx_v2", &c3_backward_amx_v2);
  m.def("c3_forward_amx_v2", &c3_forward_amx_v2);
  m.def("c3_forward_aggregate_saved_amx_v4", &c3_forward_aggregate_saved_amx_v4);
  m.def("c3_prepare_static_hs_v1", &c3_prepare_static_hs_v1);
  m.def("c3_prepare_static_hs_padded_v2", &c3_prepare_static_hs_padded_v2);
  m.def("c3_replicate_static_hs_numa_v1", &c3_replicate_static_hs_numa_v1);
  m.def("c3_prepare_static_aggregate_v3", &c3_prepare_static_aggregate_v3);
  m.def("c3_forward_cached_hs_amx_v1", &c3_forward_cached_hs_amx_v1);
  m.def("c3_forward_cached_hs_padded_amx_v2", &c3_forward_cached_hs_padded_amx_v2);
  m.def("c3_forward_cached_aggregate_amx_v3", &c3_forward_cached_aggregate_amx_v3);
  m.def("c3_backward_cached_aggregate_amx_v3", &c3_backward_cached_aggregate_amx_v3);
  m.def("c3_forward_aggregate_wide_amx_v3", &c3_forward_aggregate_wide_amx_v3);
  m.def("c3_forward_aggregate_wide_cached_hs_amx_v1", &c3_forward_aggregate_wide_cached_hs_amx_v1);
  m.def("c3_pull_only_amx_v1", &c3_pull_only_amx_v1);
  m.def("c3_pull_only_bf16_amx_v1", &c3_pull_only_bf16_amx_v1);
  m.def("c3_pull_only_bf16_scaled_fp32_amx_v1", &c3_pull_only_bf16_scaled_fp32_amx_v1);
  m.def("c3_scale_grad_bf16_v1", &c3_scale_grad_bf16_v1);
  m.def("c3_scale_grad_bf16_db_v2", &c3_scale_grad_bf16_db_v2);
  m.def("c3_forward_wide_amx_v3", &c3_forward_wide_amx_v3);
  m.def("c3_forward_wide_cached_hs_amx_v1", &c3_forward_wide_cached_hs_amx_v1);
  m.def("c3_backward_wide_amx_v3", &c3_backward_wide_amx_v3);
  m.def("c3_backward_aggregate_highd_amx_v1", &c3_backward_aggregate_highd_amx_v1);
  m.def("c3_backward_transform_highd_amx_v1", &c3_backward_transform_highd_amx_v1);
  m.def("c3_backward_transform_highd_single_scan_amx_v1", &c3_backward_transform_highd_single_scan_amx_v1);
  m.def("c3_forward_saved_t_amx_v3", &c3_forward_saved_t_amx_v3);
  m.def("c3_backward_saved_t_amx_v3", &c3_backward_saved_t_amx_v3);
  m.def("c3_backward_aggregate_saved_amx_v4", &c3_backward_aggregate_saved_amx_v4);
}
