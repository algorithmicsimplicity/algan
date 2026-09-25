// Diagnostic-only observers: preserve calls, return values and sync boundaries.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

using Wait = void (*)(id, SEL);
using Pipeline = id (*)(id, SEL, id, NSError **);
static Wait original_wait = nullptr;
static Pipeline original_pipeline = nullptr;
static bool installed = false;

static void report(const char *stage, NSError *error, id function) {
  const char *test = std::getenv("PR144_CURRENT_TEST");
  fprintf(stderr, "PR144_NATIVE_ERROR pid=%d stage=%s test=%s function=%s error=%s userInfo=%s\n",
          getpid(), stage, test ? test : "session",
          function ? [[function name] UTF8String] : "none",
          error ? [[error description] UTF8String] : "nil",
          error ? [[[error userInfo] description] UTF8String] : "nil");
  fprintf(stderr, "PR144_NATIVE_STACK %s\n", [[[NSThread callStackSymbols] description] UTF8String]);
  fflush(stderr);
}

static void wait_observer(id self, SEL selector) {
  original_wait(self, selector);
  NSError *error = [(id<MTLCommandBuffer>)self error];
  if (error != nil) report("command-buffer-completed", error, nil);
}

static id pipeline_observer(id self, SEL selector, id function, NSError **error) {
  id result = original_pipeline(self, selector, function, error);
  if (result == nil) report("compute-pipeline-creation", error ? *error : nil, function);
  return result;
}

static IMP install_observer(Class cls, SEL selector, IMP observer) {
  Method method = class_getInstanceMethod(cls, selector);
  if (method == nullptr) return nullptr;
  IMP original = method_getImplementation(method);
  // Override on the concrete class rather than mutating an inherited method
  // shared by unrelated device/command-buffer implementations.
  if (!class_addMethod(cls, selector, observer, method_getTypeEncoding(method)))
    method_setImplementation(class_getInstanceMethod(cls, selector), observer);
  return original;
}

extern "C" int pr144_install() {
  if (installed) return 1;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 0;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    if (!queue || !buffer) { [queue release]; [device release]; return 0; }
    original_wait = reinterpret_cast<Wait>(install_observer(
        object_getClass(buffer), @selector(waitUntilCompleted), reinterpret_cast<IMP>(wait_observer)));
    original_pipeline = reinterpret_cast<Pipeline>(install_observer(
        object_getClass(device), @selector(newComputePipelineStateWithFunction:error:),
        reinterpret_cast<IMP>(pipeline_observer)));
    fprintf(stderr, "PR144_NATIVE_OBSERVERS device=%s command_buffer=%s wait=%d pipeline=%d\n",
            class_getName(object_getClass(device)), class_getName(object_getClass(buffer)),
            original_wait != nullptr, original_pipeline != nullptr);
    [queue release];
    [device release];
    installed = original_wait && original_pipeline;
  }
  return installed ? 1 : 0;
}
