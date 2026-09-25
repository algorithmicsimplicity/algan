#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <cstdio>
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
  if (mode != "direct" && mode != "reuse-heap") return 2;
  constexpr NSUInteger requested = 1202590840ULL;
  constexpr NSUInteger rounding = 2ULL * 1024 * 1024;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 3;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked;
    id<MTLHeap> heap = nil;
    if (mode == "reuse-heap") {
      MTLSizeAndAlign requirement = [device heapBufferSizeAndAlignWithLength:requested options:options];
      MTLHeapDescriptor *descriptor = [[MTLHeapDescriptor alloc] init];
      descriptor.size = rounding * ((requirement.size + rounding - 1) / rounding);
      descriptor.storageMode = MTLStorageModeShared;
      descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
      descriptor.resourceOptions = options;
      descriptor.type = MTLHeapTypeAutomatic;
      heap = [device newHeapWithDescriptor:descriptor];
      [descriptor release];
      if (!heap) return 3;
    }
    printf("BUFFER_LIFETIME device=%s mode=%s bytes=%lu\n", [[device name] UTF8String], mode.c_str(), requested);
    fflush(stdout);
    for (unsigned i = 0; i < 768; ++i) {
      uint64_t address = 0;
      @autoreleasepool {
        if (rss() > 2ULL*1024*1024*1024 || [device currentAllocatedSize] > 2ULL*1024*1024*1024) return 4;
        id<MTLBuffer> buffer = heap ? [heap newBufferWithLength:requested options:options]
                                    : [device newBufferWithLength:requested options:options];
        if (!buffer) { fprintf(stderr, "ALLOCATION_FAILURE iteration=%u\n", i); return 1; }
        address = [buffer gpuAddress];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        [blit fillBuffer:buffer range:NSMakeRange(0, 8) value:7];
        [blit fillBuffer:buffer range:NSMakeRange(requested-8, 8) value:13];
        [blit endEncoding];
        [command commit]; [command waitUntilCompleted];
        if ([command status] == MTLCommandBufferStatusError) {
          fprintf(stderr, "GPU_ERROR iteration=%u address=%llx error=%s\n",i,(unsigned long long)address,[[[command error] description] UTF8String]);
          return 1;
        }
        const unsigned char *bytes = static_cast<const unsigned char *>([buffer contents]);
        for (unsigned j=0;j<8;++j) if (bytes[j]!=7 || bytes[requested-8+j]!=13) return 1;
        [buffer release];
      }
      if (i%32==31) {
        printf("BUFFER_LIFETIME_STEP i=%u mode=%s address=%llx rss=%llu metal=%llu\n",i+1,mode.c_str(),(unsigned long long)address,(unsigned long long)rss(),(unsigned long long)[device currentAllocatedSize]);
        fflush(stdout);
      }
    }
    printf("BUFFER_LIFETIME_PASS mode=%s count=768\n", mode.c_str());
    [heap release]; [queue release]; [device release];
  }
  return 0;
}
