// Diagnostic only: annotate command history; do not add waits, resets or retries.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unistd.h>

static char buffer_key, encoder_key, pipeline_key;
static std::atomic<unsigned long long> serial{0};
static std::atomic<unsigned> errors{0};
static std::atomic<bool> active{false};
static thread_local std::string context;
static std::mutex output_mutex;
static FILE *output = nullptr;
using Void = void (*)(id, SEL);
using MakeEncoder = id (*)(id, SEL);
using MakeCompute = id (*)(id, SEL, MTLDispatchType);
using SetPipeline = void (*)(id, SEL, id);
using Dispatch = void (*)(id, SEL, MTLSize, MTLSize);
using Event = void (*)(id, SEL, id, uint64_t);
using Copy = void (*)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger);
using Fill = void (*)(id, SEL, id, NSRange, uint8_t);
using Pipeline = id (*)(id, SEL, id, NSError **);
static Void old_wait, old_commit;
static MakeEncoder old_compute, old_blit;
static MakeCompute old_compute_type;
static SetPipeline old_set_pipeline;
static Dispatch old_dispatch, old_threads;
static Event old_signal, old_event_wait;
static Copy old_copy;
static Fill old_fill;
static Pipeline old_pipeline;
static bool installed = false;

static NSString *ctx() {
  return [NSString stringWithUTF8String:context.c_str()] ?: @"";
}
static NSMutableDictionary *metadata(id buffer) {
  NSMutableDictionary *value = objc_getAssociatedObject(buffer, &buffer_key);
  if (!value) {
    value = [NSMutableDictionary dictionaryWithObjectsAndKeys:
        @(++serial), @"id", [NSMutableArray array], @"work", @0, @"count", nil];
    objc_setAssociatedObject(buffer, &buffer_key, value, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  }
  return value;
}
static void append(NSMutableDictionary *meta, NSString *text) {
  if (!meta || !active.load()) return;
  @synchronized(meta) {
    NSUInteger count = [meta[@"count"] unsignedIntegerValue];
    meta[@"count"] = @(count + 1);
    NSMutableArray *work = meta[@"work"];
    if ([work count] >= 32) [work removeObjectAtIndex:8];
    [work addObject:[NSString stringWithFormat:@"%lu %@ | %@", (unsigned long)count, ctx(), text]];
  }
}
static id attach(id encoder, id buffer, const char *type) {
  if (encoder && active.load()) {
    NSMutableDictionary *meta = metadata(buffer);
    objc_setAssociatedObject(encoder, &encoder_key, meta, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    append(meta, [NSString stringWithFormat:@"encoder=%s", type]);
  }
  return encoder;
}
static void report(id buffer, NSError *error) {
  if (!error) return;
  unsigned index = errors.fetch_add(1);
  if (index >= 4) return;
  @autoreleasepool {
    NSMutableDictionary *meta = objc_getAssociatedObject(buffer, &buffer_key);
    const char *test = getenv("PR144_CURRENT_TEST");
    NSDictionary *row = @{
      @"error_index": @(index), @"test": @(test ? test : "session"),
      @"observed_context": ctx(), @"error": [error description] ?: @"nil",
      @"userinfo": [[error userInfo] description] ?: @"nil",
      @"command_buffer": [NSString stringWithFormat:@"%p", buffer],
      @"class": NSStringFromClass([buffer class]),
      @"queue": [NSString stringWithFormat:@"%p", [(id<MTLCommandBuffer>)buffer commandQueue]],
      @"metadata": meta ?: @{}, @"native_stack": [NSThread callStackSymbols]
    };
    NSData *data = [NSJSONSerialization dataWithJSONObject:row options:0 error:nil];
    std::lock_guard<std::mutex> lock(output_mutex);
    if (output && data) { fwrite(data.bytes, 1, data.length, output); fputc('\n', output); fflush(output); }
    fprintf(stderr, "COMMAND_TRACE_FAILURE index=%u test=%s context=%s error=%s metadata=%s\n", index,
      test ? test : "session", context.c_str(), [[error description] UTF8String],
      meta ? [[meta description] UTF8String] : "unobserved/internal command buffer");
    fflush(stderr);
  }
}
static void wait_call(id self, SEL selector) {
  old_wait(self, selector);
  report(self, [(id<MTLCommandBuffer>)self error]);
}
static void commit_call(id self, SEL selector) {
  if (active.load()) {
    NSMutableDictionary *meta = metadata(self);
    @synchronized(meta) {
      meta[@"commit_context"] = ctx();
      const char *test = getenv("PR144_CURRENT_TEST");
      meta[@"commit_test"] = @(test ? test : "session");
    }
  }
  old_commit(self, selector);
}
static id compute_call(id self, SEL selector) { return attach(old_compute(self, selector), self, "compute"); }
static id compute_type_call(id self, SEL selector, MTLDispatchType type) { return attach(old_compute_type(self, selector, type), self, "compute-type"); }
static id blit_call(id self, SEL selector) { return attach(old_blit(self, selector), self, "blit"); }
static id pipeline_call(id self, SEL selector, id function, NSError **error) {
  id result = old_pipeline(self, selector, function, error);
  if (result) {
    NSString *name = [NSString stringWithFormat:@"%@/%@", ctx(), [(id<MTLFunction>)function name]];
    objc_setAssociatedObject(result, &pipeline_key, name, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  }
  return result;
}
static void set_pipeline_call(id self, SEL selector, id state) {
  NSString *name = objc_getAssociatedObject(state, &pipeline_key);
  append(objc_getAssociatedObject(self, &encoder_key), [NSString stringWithFormat:@"pipeline=%@ label=%@", name ?: @"unobserved", [(id<MTLComputePipelineState>)state label]]);
  old_set_pipeline(self, selector, state);
}
static void dispatch_call(id self, SEL selector, MTLSize groups, MTLSize threads) {
  append(objc_getAssociatedObject(self, &encoder_key), [NSString stringWithFormat:@"dispatch groups=%lu,%lu,%lu threads=%lu,%lu,%lu", groups.width,groups.height,groups.depth,threads.width,threads.height,threads.depth]);
  old_dispatch(self, selector, groups, threads);
}
static void threads_call(id self, SEL selector, MTLSize grid, MTLSize group) {
  append(objc_getAssociatedObject(self, &encoder_key), [NSString stringWithFormat:@"dispatchThreads grid=%lu,%lu,%lu group=%lu,%lu,%lu",grid.width,grid.height,grid.depth,group.width,group.height,group.depth]);
  old_threads(self, selector, grid, group);
}
static void signal_call(id self, SEL selector, id event, uint64_t value) {
  append(metadata(self), [NSString stringWithFormat:@"signal event=%p value=%llu",event,(unsigned long long)value]);
  old_signal(self, selector, event, value);
}
static void event_wait_call(id self, SEL selector, id event, uint64_t value) {
  append(metadata(self), [NSString stringWithFormat:@"wait event=%p value=%llu",event,(unsigned long long)value]);
  old_event_wait(self, selector, event, value);
}
static void copy_call(id self, SEL selector, id src, NSUInteger so, id dst, NSUInteger to, NSUInteger size) {
  append(objc_getAssociatedObject(self, &encoder_key), [NSString stringWithFormat:@"copy src=%p/%lu+%lu dst=%p/%lu+%lu bytes=%lu",src,[(id<MTLBuffer>)src length],so,dst,[(id<MTLBuffer>)dst length],to,size]);
  old_copy(self, selector, src, so, dst, to, size);
}
static void fill_call(id self, SEL selector, id buffer, NSRange range, uint8_t value) {
  append(objc_getAssociatedObject(self, &encoder_key), [NSString stringWithFormat:@"fill buffer=%p/%lu range=%lu+%lu value=%u",buffer,[(id<MTLBuffer>)buffer length],range.location,range.length,value]);
  old_fill(self, selector, buffer, range, value);
}
static IMP hook(Class cls, SEL sel, IMP replacement) {
  Method method = class_getInstanceMethod(cls, sel);
  if (!method) return nullptr;
  IMP original = method_getImplementation(method);
  if (!class_addMethod(cls, sel, replacement, method_getTypeEncoding(method)))
    method_setImplementation(class_getInstanceMethod(cls, sel), replacement);
  return original;
}
extern "C" void pr144_context(const char *value) { context = value ? value : ""; }
extern "C" void pr144_enable(int enabled) { active.store(enabled != 0); }
extern "C" unsigned pr144_errors() { return errors.load(); }
extern "C" int pr144_install(const char *path) {
  if (installed) return 1;
  @autoreleasepool {
    output = fopen(path, "a");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> compute = [buffer computeCommandEncoder];
    Class cc = object_getClass(compute); [compute endEncoding];
    id<MTLBlitCommandEncoder> blit = [buffer blitCommandEncoder];
    Class bc = object_getClass(blit); [blit endEncoding];
    Class cb = object_getClass(buffer);
    old_wait = (Void)hook(cb,@selector(waitUntilCompleted),(IMP)wait_call);
    old_commit = (Void)hook(cb,@selector(commit),(IMP)commit_call);
    old_compute = (MakeEncoder)hook(cb,@selector(computeCommandEncoder),(IMP)compute_call);
    old_compute_type = (MakeCompute)hook(cb,@selector(computeCommandEncoderWithDispatchType:),(IMP)compute_type_call);
    old_blit = (MakeEncoder)hook(cb,@selector(blitCommandEncoder),(IMP)blit_call);
    old_signal = (Event)hook(cb,@selector(encodeSignalEvent:value:),(IMP)signal_call);
    old_event_wait = (Event)hook(cb,@selector(encodeWaitForEvent:value:),(IMP)event_wait_call);
    old_set_pipeline = (SetPipeline)hook(cc,@selector(setComputePipelineState:),(IMP)set_pipeline_call);
    old_dispatch = (Dispatch)hook(cc,@selector(dispatchThreadgroups:threadsPerThreadgroup:),(IMP)dispatch_call);
    old_threads = (Dispatch)hook(cc,@selector(dispatchThreads:threadsPerThreadgroup:),(IMP)threads_call);
    old_copy = (Copy)hook(bc,@selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:),(IMP)copy_call);
    old_fill = (Fill)hook(bc,@selector(fillBuffer:range:value:),(IMP)fill_call);
    old_pipeline = (Pipeline)hook(object_getClass(device),@selector(newComputePipelineStateWithFunction:error:),(IMP)pipeline_call);
    [queue release]; [device release];
    installed = output && old_wait && old_commit && old_compute && old_blit && old_dispatch && old_set_pipeline && old_pipeline && old_copy && old_fill;
    fprintf(stderr,"COMMAND_TRACE_INSTALLED ok=%d buffer=%s compute=%s blit=%s\n",installed,class_getName(cb),class_getName(cc),class_getName(bc));
  }
  return installed ? 1 : 0;
}
