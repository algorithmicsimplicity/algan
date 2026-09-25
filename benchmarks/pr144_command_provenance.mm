// Diagnostic only: retain bounded text metadata, never GPU resources.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unistd.h>

static char command_key, encoder_key, pipeline_key;
static std::atomic<unsigned long long> sequence{0};
static std::atomic<bool> failed{false}, enabled{false};
static std::mutex context_mutex;
static std::string current_test = "session", last_operation = "unlabelled";
static thread_local std::string operation = "unlabelled";
static id (*old_compute)(id, SEL);
static id (*old_blit)(id, SEL);
static void (*old_commit)(id, SEL);
static void (*old_wait)(id, SEL);
static void (*old_pipeline)(id, SEL, id);
static void (*old_buffer)(id, SEL, id, NSUInteger, NSUInteger);
static void (*old_dispatch)(id, SEL, MTLSize, MTLSize);
static void (*old_threads)(id, SEL, MTLSize, MTLSize);
static void (*old_copy)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger);
static void (*old_fill)(id, SEL, id, NSRange, uint8_t);
static id (*old_heap_buffer)(id, SEL, NSUInteger, MTLResourceOptions);
static id (*old_make_pipeline)(id, SEL, id, NSError **);

static NSString *context() {
  std::lock_guard<std::mutex> guard(context_mutex);
  return [NSString stringWithFormat:@"test=%s op=%s", current_test.c_str(), (operation == "unlabelled" ? last_operation : operation).c_str()];
}

static NSMutableDictionary *info(id buffer) {
  auto *result = (NSMutableDictionary *)objc_getAssociatedObject(buffer, &command_key);
  if (!result) {
    result = [NSMutableDictionary dictionaryWithObjectsAndKeys:
        @((unsigned long long)++sequence), @"sequence", context(), @"first_seen",
        [NSMutableArray array], @"encoded", nil];
    objc_setAssociatedObject(buffer, &command_key, result, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  }
  return result;
}

static void note(id encoder, NSString *description) {
  auto *data = (NSMutableDictionary *)objc_getAssociatedObject(encoder, &encoder_key);
  if (!data) return;
  @synchronized(data) {
    NSMutableArray *ops = data[@"encoded"];
    if ([ops count] >= 64) [ops removeObjectAtIndex:0];
    [ops addObject:[NSString stringWithFormat:@"%@ %@", context(), description]];
  }
}

static void report(id buffer, const char *stage) {
  NSError *error = [(id<MTLCommandBuffer>)buffer error];
  if (!error || failed.exchange(true)) return;
  @autoreleasepool {
    NSMutableDictionary *data = info(buffer);
    fprintf(stderr, "\nPR144_FIRST_COMMAND_ERROR stage=%s buffer=%p queue=%p status=%lu label=%s context=%s\n",
            stage, buffer, [(id<MTLCommandBuffer>)buffer commandQueue],
            (unsigned long)[(id<MTLCommandBuffer>)buffer status],
            [[(id<MTLCommandBuffer>)buffer label] UTF8String], [context() UTF8String]);
    fprintf(stderr, "PR144_ERROR_DETAIL %s\nPR144_COMMAND_HISTORY %s\n",
            [[error description] UTF8String], [[data description] UTF8String]);
    fprintf(stderr, "PR144_ERROR_USERINFO %s\nPR144_NATIVE_STACK %s\n",
            [[[error userInfo] description] UTF8String], [[[NSThread callStackSymbols] description] UTF8String]);
    fflush(stderr);
    // Preserve the first event and stop before later ignored submissions obscure it.
    if (std::getenv("PR144_EXIT_NATIVE_ERROR")) _exit(86);
  }
}

static id compute(id self, SEL selector) {
  id encoder = old_compute(self, selector);
  if (enabled.load() && encoder)
    objc_setAssociatedObject(encoder, &encoder_key, info(self), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  return encoder;
}
static id blit(id self, SEL selector) {
  id encoder = old_blit(self, selector);
  if (enabled.load() && encoder)
    objc_setAssociatedObject(encoder, &encoder_key, info(self), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  return encoder;
}
static void commit(id self, SEL selector) {
  if (enabled.load()) {
    @autoreleasepool {
      NSMutableDictionary *data = info(self);
      data[@"commit"] = context();
      fprintf(stderr, "PR144_COMMIT buffer=%p queue=%p history=%s\n", self,
              [(id<MTLCommandBuffer>)self commandQueue], [[data description] UTF8String]);
      fflush(stderr);
    }
  }
  old_commit(self, selector);
}
static void wait(id self, SEL selector) {
  old_wait(self, selector);
  report(self, "waitUntilCompleted");
}
static void pipeline(id self, SEL selector, id state) {
  if (enabled.load()) {
    NSString *name = objc_getAssociatedObject(state, &pipeline_key);
    note(self, [NSString stringWithFormat:@"pipeline=%p function=%@ label=%@", state, name,
                 [(id<MTLComputePipelineState>)state label]]);
  }
  old_pipeline(self, selector, state);
}
static void buffer(id self, SEL selector, id value, NSUInteger offset, NSUInteger index) {
  if (enabled.load())
    note(self, [NSString stringWithFormat:@"binding=%lu buffer=%p length=%lu offset=%lu storage=%lu hazards=%lu",
        (unsigned long)index, value, (unsigned long)[(id<MTLBuffer>)value length],
        (unsigned long)offset, (unsigned long)[(id<MTLBuffer>)value storageMode],
        (unsigned long)[(id<MTLBuffer>)value hazardTrackingMode]]);
  old_buffer(self, selector, value, offset, index);
}
static NSString *grid(NSString *kind, MTLSize size, MTLSize group) {
  return [NSString stringWithFormat:@"%@=(%lu,%lu,%lu) group=(%lu,%lu,%lu)", kind,
      size.width, size.height, size.depth, group.width, group.height, group.depth];
}
static void dispatch(id self, SEL selector, MTLSize size, MTLSize group) {
  if (enabled.load()) note(self, grid(@"groups", size, group));
  old_dispatch(self, selector, size, group);
}
static void threads(id self, SEL selector, MTLSize size, MTLSize group) {
  if (enabled.load()) note(self, grid(@"threads", size, group));
  old_threads(self, selector, size, group);
}
static void copy(id self, SEL selector, id src, NSUInteger src_offset, id dst, NSUInteger dst_offset, NSUInteger size) {
  if (enabled.load()) note(self, [NSString stringWithFormat:@"copy src=%p[%lu]+%lu dst=%p[%lu]+%lu bytes=%lu",
      src, (unsigned long)[(id<MTLBuffer>)src length], src_offset,
      dst, (unsigned long)[(id<MTLBuffer>)dst length], dst_offset, size]);
  old_copy(self, selector, src, src_offset, dst, dst_offset, size);
}
static void fill(id self, SEL selector, id dst, NSRange range, uint8_t value) {
  if (enabled.load()) note(self, [NSString stringWithFormat:@"fill buffer=%p length=%lu offset=%lu bytes=%lu value=%u",
      dst, (unsigned long)[(id<MTLBuffer>)dst length], range.location, range.length, value]);
  old_fill(self, selector, dst, range, value);
}
static id make_pipeline(id self, SEL selector, id function, NSError **error) {
  id result = old_make_pipeline(self, selector, function, error);
  if (result) objc_setAssociatedObject(result, &pipeline_key, [(id<MTLFunction>)function name], OBJC_ASSOCIATION_COPY_NONATOMIC);
  return result;
}
static id heap_buffer(id self, SEL selector, NSUInteger size, MTLResourceOptions options) {
  if (enabled.load()) {
    fprintf(stderr, "PR144_HEAP_ALLOC_BEGIN heap=%p size=%lu options=%lu context=%s\n", self, size, (unsigned long)options, [context() UTF8String]);
    fflush(stderr);
  }
  id result = old_heap_buffer(self, selector, size, options);
  if (enabled.load()) {
    fprintf(stderr, "PR144_HEAP_ALLOC_END heap=%p size=%lu buffer=%p\n", self, size, result);
    fflush(stderr);
  }
  return result;
}
static IMP observe(Class cls, SEL selector, IMP replacement) {
  Method method = class_getInstanceMethod(cls, selector);
  if (!method) return nullptr;
  IMP original = method_getImplementation(method);
  if (!class_addMethod(cls, selector, replacement, method_getTypeEncoding(method)))
    method_setImplementation(class_getInstanceMethod(cls, selector), replacement);
  return original;
}
extern "C" void pr144_test(const char *name, int active) {
  std::lock_guard<std::mutex> guard(context_mutex);
  current_test = name ? name : "session";
  enabled.store(active != 0);
}
extern "C" void pr144_op(const char *name) {
  operation = name ? name : "unlabelled";
  std::lock_guard<std::mutex> guard(context_mutex);
  last_operation = operation;
}
extern "C" int pr144_failed() { return failed.load() ? 1 : 0; }
extern "C" int pr144_install() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 0;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> ce = [command computeCommandEncoder];
    Class cc = object_getClass(command), ec = object_getClass(ce), dc = object_getClass(device);
    [ce endEncoding];
    id<MTLBlitCommandEncoder> be = [command blitCommandEncoder];
    Class bc = object_getClass(be);
    [be endEncoding];
    old_compute = (decltype(old_compute))observe(cc, @selector(computeCommandEncoder), (IMP)compute);
    old_blit = (decltype(old_blit))observe(cc, @selector(blitCommandEncoder), (IMP)blit);
    old_commit = (decltype(old_commit))observe(cc, @selector(commit), (IMP)commit);
    old_wait = (decltype(old_wait))observe(cc, @selector(waitUntilCompleted), (IMP)wait);
    old_pipeline = (decltype(old_pipeline))observe(ec, @selector(setComputePipelineState:), (IMP)pipeline);
    old_buffer = (decltype(old_buffer))observe(ec, @selector(setBuffer:offset:atIndex:), (IMP)buffer);
    old_dispatch = (decltype(old_dispatch))observe(ec, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)dispatch);
    old_threads = (decltype(old_threads))observe(ec, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)threads);
    old_copy = (decltype(old_copy))observe(bc, @selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:), (IMP)copy);
    old_fill = (decltype(old_fill))observe(bc, @selector(fillBuffer:range:value:), (IMP)fill);
    old_make_pipeline = (decltype(old_make_pipeline))observe(dc, @selector(newComputePipelineStateWithFunction:error:), (IMP)make_pipeline);
    MTLHeapDescriptor *descriptor = [[MTLHeapDescriptor alloc] init];
    descriptor.size = 4 * 1024 * 1024;
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLHeap> heap = [device newHeapWithDescriptor:descriptor];
    if (heap) old_heap_buffer = (decltype(old_heap_buffer))observe(object_getClass(heap), @selector(newBufferWithLength:options:), (IMP)heap_buffer);
    [heap release]; [descriptor release];
    fprintf(stderr, "PR144_PROVENANCE_CLASSES command=%s compute=%s blit=%s\n", class_getName(cc), class_getName(ec), class_getName(bc));
    [queue release]; [device release];
    return old_compute && old_blit && old_commit && old_wait && old_pipeline && old_buffer && old_dispatch && old_threads && old_copy && old_fill && old_make_pipeline;
  }
}
