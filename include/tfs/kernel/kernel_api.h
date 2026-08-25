#pragma once
#include "tfs/runtime/runtime.h"
namespace tfs {
inline Status TfsForward(const GraphCSR&,const SchedulePlan&,const TensorView&,const WeightState&,TensorView,EnvironmentInfo&) { return Status::NotImplemented("kernel.TfsForward: deferred to C1"); }
}

