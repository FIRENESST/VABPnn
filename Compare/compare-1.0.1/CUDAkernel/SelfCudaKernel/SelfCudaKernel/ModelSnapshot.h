#pragma once
#include <cstddef>
#include <cuda_runtime.h>
namespace vabpnn {
// Model/optimizer storage only; frame buffers and transient activations are never checkpointed.
struct ModelPart { void* device; std::size_t bytes; };
}
