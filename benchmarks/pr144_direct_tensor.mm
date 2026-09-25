#define PY_SSIZE_T_CLEAN
#include <Python.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <cstdint>
#include <new>

// DLPack 0.x ABI. Metal data is an opaque id<MTLBuffer>, not its GPU address.
struct DLDevice { int32_t device_type; int32_t device_id; };
struct DLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };
struct DLTensor { void *data; DLDevice device; int32_t ndim; DLDataType dtype; int64_t *shape; int64_t *strides; uint64_t byte_offset; };
struct DLManagedTensor { DLTensor dl_tensor; void *manager_ctx; void (*deleter)(DLManagedTensor *); };
struct Owner { DLManagedTensor tensor; id<MTLBuffer> buffer; int64_t shape; };
static std::atomic<uint64_t> live_bytes{0};

static void destroy(DLManagedTensor *managed) {
  if (!managed) return;
  Owner *owner = static_cast<Owner *>(managed->manager_ctx);
  @autoreleasepool { [owner->buffer release]; }
  live_bytes.fetch_sub(static_cast<uint64_t>(owner->shape));
  delete owner;
}
static void capsule_destroy(PyObject *capsule) {
  if (PyCapsule_IsValid(capsule, "dltensor")) {
    destroy(static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor")));
  }
}
extern "C" uint64_t pr144_direct_live_bytes() { return live_bytes.load(); }
extern "C" PyObject *pr144_direct_allocate(uint64_t size) {
  if (size == 0 || size > ((1ULL << 31) - (1ULL << 20))) {
    PyErr_SetString(PyExc_ValueError, "Direct buffer size must be positive and below the MPS arena addressing limit");
    return nullptr;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { PyErr_SetString(PyExc_RuntimeError, "Metal device unavailable"); return nullptr; }
    id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
    [device release];
    if (!buffer) return PyErr_NoMemory();
    Owner *owner = new (std::nothrow) Owner{};
    if (!owner) { [buffer release]; return PyErr_NoMemory(); }
    owner->buffer = buffer;
    owner->shape = static_cast<int64_t>(size);
    owner->tensor.dl_tensor = {static_cast<void *>(buffer), {8, 0}, 1, {1, 8, 1}, &owner->shape, nullptr, 0};
    owner->tensor.manager_ctx = owner;
    owner->tensor.deleter = destroy;
    live_bytes.fetch_add(size);
    PyObject *capsule = PyCapsule_New(&owner->tensor, "dltensor", capsule_destroy);
    if (!capsule) destroy(&owner->tensor);
    return capsule;
  }
}
