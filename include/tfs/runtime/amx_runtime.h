#pragma once
#include <string>
#include "tfs/common/status.h"
namespace tfs {
struct AmxCapability { bool cpu_amx_tile=false,cpu_amx_bf16=false,kernel_tilecfg=false,kernel_tiledata=false,permission_granted=false,tile_lifecycle=false; long permission_syscall=0; int permission_errno=0; };
class AmxRuntime { public: static AmxCapability Probe(); static Status RequestPermission(AmxCapability*); static Status SmokeTileLifecycle(AmxCapability*); };
std::string AmxCapabilityJson(const AmxCapability&);
}

