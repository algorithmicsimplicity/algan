#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <cstdio>
#include <cstring>
#include <string>

static uint64_t rss() {
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
  return info.resident_size;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  std::string mode(argv[1]);
  if (mode != "automatic" && mode != "placement" && mode != "automatic-no-purge") return 2;
  const bool placement = mode == "placement";
  const bool purge = mode != "automatic-no-purge";
  constexpr NSUInteger requested = 1202590840ULL;
  constexpr NSUInteger round = 2ULL * 1024 * 1024;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 3;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
    MTLSizeAndAlign requirement = [device heapBufferSizeAndAlignWithLength:requested options:options];
    NSUInteger length = (requested + requirement.align - 1) & ~(requirement.align - 1);
    NSUInteger heapSize = round * ((requirement.size + round - 1) / round);
    MTLHeapDescriptor *descriptor = [[MTLHeapDescriptor alloc] init];
    descriptor.size = heapSize;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    descriptor.resourceOptions = options;
    descriptor.type = placement ? MTLHeapTypePlacement : MTLHeapTypeAutomatic;
    printf("NATIVE_HEAP device=%s mode=%s requested=%lu length=%lu heap=%lu\n", [[device name] UTF8String], mode.c_str(), requested, length, heapSize);
    fflush(stdout);
    for (unsigned i = 0; i < 768; ++i) {
      @autoreleasepool {
        if (rss() > 2ULL * 1024 * 1024 * 1024 || [device currentAllocatedSize] > 2ULL * 1024 * 1024 * 1024) {
          fprintf(stderr, "MEMORY_SAFETY_STOP i=%u rss=%llu metal=%llu\n", i, (unsigned long long)rss(), (unsigned long long)[device currentAllocatedSize]);
          return 4;
        }
        id<MTLHeap> heap = [device newHeapWithDescriptor:descriptor];
        if (!heap) { fprintf(stderr, "HEAP_ALLOCATION_FAILED i=%u\n", i); return 1; }
        [heap setPurgeableState:MTLPurgeableStateNonVolatile];
        id<MTLBuffer> buffer = placement ? [heap newBufferWithLength:length options:options offset:0] : [heap newBufferWithLength:length options:options];
        if (!buffer) { fprintf(stderr, "BUFFER_ALLOCATION_FAILED i=%u\n", i); [heap release]; return 1; }
        uint64_t address = [buffer gpuAddress];
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
        [blit fillBuffer:buffer range:NSMakeRange(0, 8) value:7];
        [blit fillBuffer:buffer range:NSMakeRange(requested - 8, 8) value:13];
        [blit endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if ([commands status] == MTLCommandBufferStatusError) {
          fprintf(stderr, "NATIVE_GPU_ERROR i=%u mode=%s address=%llx error=%s\n", i, mode.c_str(), (unsigned long long)address, [[[commands error] description] UTF8String]);
          return 1;
        }
        const unsigned char *bytes = static_cast<const unsigned char *>([buffer contents]);
        for (unsigned j = 0; j < 8; ++j) {
          if (bytes[j] != 7 || bytes[requested - 8 + j] != 13) {
            fprintf(stderr, "NATIVE_READBACK_FAILED i=%u j=%u\n", i, j);
            return 1;
          }
        }
        [buffer release];
        if (purge) [heap setPurgeableState:MTLPurgeableStateEmpty];
        [heap release];
        if (i % 32 == 31) {
          printf("NATIVE_HEAP_CHURN i=%u mode=%s address=%llx rss=%llu metal=%llu\n", i + 1, mode.c_str(), (unsigned long long)address, (unsigned long long)rss(), (unsigned long long)[device currentAllocatedSize]);
          fflush(stdout);
        }
      }
    }
    printf("NATIVE_HEAP_PASS mode=%s count=768\n", mode.c_str());
    [descriptor release]; [queue release]; [device release];
  }
  return 0;
}
