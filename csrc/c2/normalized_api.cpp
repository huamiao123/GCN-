#include <torch/extension.h>
#include <vector>

static void check_graph(const at::Tensor& x,const at::Tensor& rowptr,const at::Tensor& colidx,const at::Tensor& s){
 TORCH_CHECK(x.device().is_cpu()&&rowptr.device().is_cpu()&&colidx.device().is_cpu()&&s.device().is_cpu(),"C2 CPU reference only");
 TORCH_CHECK(x.dim()==2&&rowptr.dim()==1&&colidx.dim()==1&&s.dim()==1,"invalid tensor rank");
 TORCH_CHECK(rowptr.scalar_type()==at::kLong&&colidx.scalar_type()==at::kLong,"CSR indices must be int64");
 TORCH_CHECK(rowptr.numel()==x.size(0)+1&&s.numel()==x.size(0),"CSR/node shape mismatch");
}

at::Tensor c2_pull_spmm(const at::Tensor& input,const at::Tensor& rowptr,const at::Tensor& colidx){
 auto x=input.contiguous(),rp=rowptr.contiguous(),ci=colidx.contiguous();auto out=x.clone();const auto* r=rp.data_ptr<int64_t>();const auto* c=ci.data_ptr<int64_t>();int64_t n=x.size(0),f=x.size(1);
 AT_DISPATCH_FLOATING_TYPES(x.scalar_type(),"c2_pull_spmm",[&]{const scalar_t* src=x.data_ptr<scalar_t>();scalar_t* dst=out.data_ptr<scalar_t>();for(int64_t i=0;i<n;i++)for(int64_t e=r[i];e<r[i+1];e++){int64_t j=c[e];TORCH_CHECK(j>=0&&j<n,"CSR column out of range");for(int64_t q=0;q<f;q++)dst[i*f+q]+=src[j*f+q];}});return out;
}

std::vector<at::Tensor> c2_forward(const at::Tensor& x,const at::Tensor& weight,const at::Tensor& bias,const at::Tensor& rowptr,const at::Tensor& colidx,const at::Tensor& s){
 check_graph(x,rowptr,colidx,s);TORCH_CHECK(weight.dim()==2&&weight.size(0)==x.size(1),"weight shape mismatch");TORCH_CHECK(bias.dim()==1&&bias.size(0)==weight.size(1),"bias shape mismatch");
 auto hs=s.unsqueeze(1)*x;auto t=c2_pull_spmm(hs,rowptr,colidx);auto p=s.unsqueeze(1)*at::matmul(t,weight)+bias;return{p,hs};
}

std::vector<at::Tensor> c2_backward_selective(const at::Tensor& grad,const at::Tensor& hs,const at::Tensor& weight,const at::Tensor& rowptr,const at::Tensor& colidx,const at::Tensor& s,bool compute_dx){
 check_graph(grad,rowptr,colidx,s);auto grad_scaled=s.unsqueeze(1)*grad;auto ybar=c2_pull_spmm(grad_scaled,rowptr,colidx);auto dw=at::matmul(hs.transpose(0,1),ybar);auto dx=compute_dx?s.unsqueeze(1)*at::matmul(ybar,weight.transpose(0,1)):at::empty({0},grad.options());auto db=grad.sum(0);return{dx,dw,db,ybar};
}

std::vector<at::Tensor> c2_backward(const at::Tensor& grad,const at::Tensor& hs,const at::Tensor& weight,const at::Tensor& rowptr,const at::Tensor& colidx,const at::Tensor& s){
 return c2_backward_selective(grad,hs,weight,rowptr,colidx,s,true);
}
