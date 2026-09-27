// Native ownership for the render arena; no PyTorch or Quadrants C++ ABI.
#define Py_LIMITED_API 0x030A0000
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdint>
#include <new>

namespace {
// The stable, legacy DLManagedTensor ABI (DLPack 0.x). Use the legacy capsule
// for compatibility with the supported Torch releases. For kDLMetal, data is
// an opaque id<MTLBuffer>, NOT contents() or gpuAddress. Layout reference:
// https://dmlc.github.io/dlpack/latest/c_api.html
struct DLDevice { int32_t device_type; int32_t device_id; };
struct DLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };
struct DLTensor {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
};
struct DLManagedTensor {
  DLTensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(DLManagedTensor *);
};
constexpr int32_t kDLMetal = 8;
constexpr uint64_t kMaxArenaBytes = (1ULL << 31) - (1ULL << 20);
std::atomic<uint64_t> live_bytes{0};

struct Owner {
  DLManagedTensor tensor{};
  id<MTLBuffer> buffer = nil;
  int64_t shape = 0;
  int64_t stride = 1;
  uint64_t bytes = 0;
};

void destroy(DLManagedTensor *managed) noexcept {
  if (!managed) return;
  auto *owner = static_cast<Owner *>(managed->manager_ctx);
  // Torch calls this when the last STORAGE owner dies, not when the original
  // tensor or ManualMemory dies. The callback uses no Python or Torch APIs,
  // so it is safe off the GIL and during interpreter shutdown. Already-encoded
  // work retains its Metal resources through the ordinary retained-reference
  // command buffers; there is no allocator reuse or new synchronization here.
  @autoreleasepool { [owner->buffer release]; }
  live_bytes.fetch_sub(owner->bytes, std::memory_order_relaxed);
  delete owner;
}

void capsule_destroy(PyObject *capsule) {
  // A consumed capsule is renamed used_dltensor by Torch. Its storage deleter
  // now owns the release; releasing it here as well would double-free it.
  if (PyCapsule_IsValid(capsule, "dltensor")) {
    destroy(static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor")));
  }
}

PyObject *allocate(PyObject *, PyObject *size_object) {
  if (PyBool_Check(size_object)) {
    PyErr_SetString(PyExc_TypeError, "MPS arena size must be an integer byte count, not bool");
    return nullptr;
  }
  PyObject *index = PyNumber_Index(size_object);
  if (!index) return nullptr;
  long long size = PyLong_AsLongLong(index);
  Py_DECREF(index);
  if (size == -1 && PyErr_Occurred()) return nullptr;
  if (size <= 0 || static_cast<uint64_t>(size) > kMaxArenaBytes) {
    PyErr_SetString(PyExc_ValueError, "MPS arena size must be positive and below the MPS addressing limit");
    return nullptr;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      PyErr_SetString(PyExc_RuntimeError, "No Metal device is available for the MPS arena");
      return nullptr;
    }
    // Algan's MPS path targets Apple's unified-memory device, as does Torch.
    // Refuse discrete-memory devices: this owner deliberately creates shared
    // storage and is supported on the unified-memory Apple GPU only.
    if (![device hasUnifiedMemory]) {
      [device release];
      PyErr_SetString(PyExc_RuntimeError, "Native MPS arenas require a unified-memory Apple GPU");
      return nullptr;
    }
    if (static_cast<uint64_t>(size) > [device maxBufferLength]) {
      [device release];
      PyErr_SetString(PyExc_ValueError, "MPS arena exceeds Metal maxBufferLength");
      return nullptr;
    }
    // Deliberately NOT newHeap / newBuffer on an MTLHeap. Repeated large heap
    // allocations on the hosted paravirtual GPU reproduce a native GPU hang
    // without Algan or Quadrants. A standalone tracked buffer avoids that path.
    id<MTLBuffer> buffer = [device newBufferWithLength:static_cast<NSUInteger>(size)
        options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked];
    [device release];
    if (!buffer) {
      // memory_utils.is_cuda_oom recognizes this alongside Torch's OOMs.
      PyErr_SetString(PyExc_RuntimeError, "MPS backend out of memory allocating a standalone render arena");
      return nullptr;
    }
    auto *owner = new (std::nothrow) Owner{};
    if (!owner) {
      [buffer release];
      return PyErr_NoMemory();
    }
    owner->buffer = buffer;
    owner->shape = size;
    owner->bytes = [buffer length];
    owner->tensor.dl_tensor = {
        static_cast<void *>(buffer), {kDLMetal, 0}, 1, {1, 8, 1},
        &owner->shape, &owner->stride, 0};
    owner->tensor.manager_ctx = owner;
    owner->tensor.deleter = destroy;
    live_bytes.fetch_add(owner->bytes, std::memory_order_relaxed);
    PyObject *capsule = PyCapsule_New(&owner->tensor, "dltensor", capsule_destroy);
    if (!capsule) destroy(&owner->tensor);
    return capsule;
  }
}

PyObject *allocated_bytes(PyObject *, PyObject *) {
  return PyLong_FromUnsignedLongLong(live_bytes.load(std::memory_order_relaxed));
}

PyMethodDef methods[] = {
    {"allocate", allocate, METH_O, "Return one owning DLPack capsule over a standalone Metal byte buffer."},
    {"allocated_bytes", allocated_bytes, METH_NOARGS, "Bytes retained by live external tensor storage and unconsumed capsules."},
    {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_mps_arena_native",
    "Standalone Metal render-arena ownership (CPython stable ABI).",
    -1, methods, nullptr, nullptr, nullptr, nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__mps_arena_native() {
  PyObject *result = PyModule_Create(&module);
  if (result && PyModule_AddIntConstant(result, "MAX_ARENA_BYTES", kMaxArenaBytes) < 0) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}
