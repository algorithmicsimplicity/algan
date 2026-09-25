#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static uint64_t rss() {
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    return 0;
  return info.resident_size;
}

static bool compile(id<MTLDevice> device, const std::string &source,
                    const std::string &name, unsigned iteration) {
  NSString *text = [[NSString alloc] initWithBytes:source.data()
                                          length:source.size()
                                        encoding:NSUTF8StringEncoding];
  MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
  options.languageVersion = MTLLanguageVersion3_0;
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:text options:options error:&error];
  [text release];
  [options release];
  if (!library) {
    fprintf(stderr, "LIBRARY_FAILURE iteration=%u source_bytes=%zu error=%s\n",
            iteration, source.size(), [[error description] UTF8String]);
    return false;
  }
  NSString *entry = [[NSString alloc] initWithUTF8String:name.c_str()];
  id<MTLFunction> function = [library newFunctionWithName:entry];
  [entry release];
  if (!function) { [library release]; return false; }
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) {
    fprintf(stderr, "PIPELINE_FAILURE iteration=%u source_bytes=%zu error=%s userInfo=%s\n",
            iteration, source.size(), [[error description] UTF8String],
            [[[error userInfo] description] UTF8String]);
  }
  bool ok = pipeline != nil;
  [pipeline release];
  [function release];
  [library release];
  return ok;
}

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  bool scoped = std::string(argv[2]) == "pooled";
  unsigned count = static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10));
  if (!count || count > 32) return 2;
  std::ifstream stream(argv[1]);
  std::ostringstream data;
  data << stream.rdbuf();
  std::string original = data.str();
  if (original.empty() || original.size() > 6000000) return 2;
  // The control keeps one pool until exit, as a Python main thread without an
  // Objective-C event loop would. Explicit library/function/PSO owners are
  // released on every iteration in BOTH arms.
  NSAutoreleasePool *outer = [[NSAutoreleasePool alloc] init];
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) return 3;
  fprintf(stdout, "DEVICE %s scoped=%d source_bytes=%zu\n", [[device name] UTF8String], scoped, original.size());
  fflush(stdout);
  int status = 0;
  for (unsigned i = 0; i < count; ++i) {
    NSAutoreleasePool *pool = scoped ? [[NSAutoreleasePool alloc] init] : nil;
    std::string source = original;
    std::string entry = "pr144_main_" + std::to_string(i);
    size_t position = source.find("main0(");
    if (position == std::string::npos) return 2;
    source.replace(position, 5, entry);
    bool ok = compile(device, source, entry, i);
    if (pool) [pool drain];
    fprintf(stdout, "NATIVE_COMPILE i=%u scoped=%d ok=%d rss=%llu metal=%llu\n", i, scoped, ok,
            (unsigned long long)rss(), (unsigned long long)[device currentAllocatedSize]);
    fflush(stdout);
    if (!ok) { status = 1; break; }
    if (rss() > 2ULL * 1024 * 1024 * 1024) { status = 4; break; }
  }
  std::string small = "#include <metal_stdlib>\nusing namespace metal;\nkernel void pr144_small(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {out[i]=37.0f;}\n";
  bool small_ok = compile(device, small, "pr144_small", count);
  fprintf(stdout, "SMALL_SENTINEL ok=%d rss=%llu\n", small_ok, (unsigned long long)rss());
  [outer drain];
  fprintf(stdout, "AFTER_OUTER_DRAIN rss=%llu metal=%llu\n", (unsigned long long)rss(),
          (unsigned long long)[device currentAllocatedSize]);
  [device release];
  return status ? status : (small_ok ? 0 : 1);
}
