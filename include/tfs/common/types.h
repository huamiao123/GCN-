#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "tfs/common/status.h"

namespace tfs {
enum class DType { kBF16, kFP32, kFP64, kInt32, kInt64, kUInt8 };
enum class MatrixLayout { kRowMajor, kColMajor, kAmxBf16RhsVnni, kUnknown };
enum class NodeOrder { kOriginal, kPhysicalPermuted };
enum class ZStrategy { kSave, kRecompute };
enum class QStrategy { kMaterialize, kPush, kPrivate, kHybrid };
enum class NormalizationStrategy { kFactorizedOnTheFly, kExplicitEdgeWeight, kPreScaledFeature };
std::size_t DTypeBytes(DType value);
std::string ToString(DType value);
}

