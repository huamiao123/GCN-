#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace tfs { struct OwnedBuffer { std::vector<std::uint8_t> bytes; std::size_t alignment=64; }; }

