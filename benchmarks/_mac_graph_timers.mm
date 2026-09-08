// Diagnostic only. Time existing Objective-C calls, without submitting work,
// changing execution descriptors, or adding synchronization. Original IMPs
// receive their exact original arguments. Method ABIs are checked at install.
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
using Obj0 = id (*)(id, SEL);
using Obj5 = id (*)(id, SEL, id, id, id, id, id);
using Void5 = void (*)(id, SEL, id, id, id, id, id);
Obj0 original_init = nullptr;
Obj5 original_compile = nullptr;
Obj5 original_encode_obj = nullptr;
Void5 original_encode_void = nullptr;
std::atomic<int> phase{-1};
struct Row { uint64_t count = 0; double total = 0, self = 0, maximum = 0; };
Row rows[5][3];
std::mutex mutex;
thread_local double nested = 0;
std::string serialized;

struct Span {
  int p, kind;
  double old_nested;
  Clock::time_point start;
  Span(int k) : p(phase.load(std::memory_order_relaxed)), kind(k), old_nested(nested) {
    if (p >= 0) { start = Clock::now(); nested = 0; }
  }
  ~Span() {
    if (p < 0) return;
    double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    double own = seconds - nested;
    nested = old_nested + seconds;
    std::lock_guard<std::mutex> guard(mutex);
    Row &r = rows[p][kind];
    ++r.count; r.total += seconds; r.self += own;
    r.maximum = std::max(r.maximum, seconds);
  }
};
id timed_init(id self, SEL sel) {
  Span span(0); return original_init(self, sel);
}
id timed_compile(id self, SEL sel, id a, id b, id c, id d, id e) {
  Span span(1); return original_compile(self, sel, a, b, c, d, e);
}
id timed_encode_obj(id self, SEL sel, id a, id b, id c, id d, id e) {
  Span span(2); return original_encode_obj(self, sel, a, b, c, d, e);
}
void timed_encode_void(id self, SEL sel, id a, id b, id c, id d, id e) {
  Span span(2); original_encode_void(self, sel, a, b, c, d, e);
}
Method checked(Class cls, const char *selector, unsigned count, char result) {
  Method m = class_getInstanceMethod(cls, sel_registerName(selector));
  if (!m || method_getNumberOfArguments(m) != count) return nullptr;
  char type[128]; method_getReturnType(m, type, sizeof(type));
  if (type[0] != result) return nullptr;
  for (unsigned i = 2; i < count; ++i) {
    method_getArgumentType(m, i, type, sizeof(type));
    if (type[0] != '@') return nullptr;
  }
  return m;
}
}

extern "C" int algan_graph_install() {
  Class cls = NSClassFromString(@"MPSGraph");
  if (!cls) return 0;
  int installed = 0;
  if (Method m = checked(cls, "init", 2, '@')) {
    // class_addMethod avoids replacing NSObject's inherited init, if this OS
    // does not supply its own implementation on MPSGraph.
    original_init = reinterpret_cast<Obj0>(method_getImplementation(m));
    if (!class_addMethod(cls, sel_registerName("init"), reinterpret_cast<IMP>(timed_init), method_getTypeEncoding(m)))
      method_setImplementation(m, reinterpret_cast<IMP>(timed_init));
    installed |= 1;
  }
  if (Method m = checked(cls, "compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:", 7, '@')) {
    original_compile = reinterpret_cast<Obj5>(method_setImplementation(m, reinterpret_cast<IMP>(timed_compile)));
    installed |= 2;
  }
  const char *encode = "encodeToCommandBuffer:feeds:targetOperations:resultsDictionary:executionDescriptor:";
  if (Method m = checked(cls, encode, 7, '@')) {
    original_encode_obj = reinterpret_cast<Obj5>(method_setImplementation(m, reinterpret_cast<IMP>(timed_encode_obj)));
    installed |= 4;
  } else if (Method m = checked(cls, encode, 7, 'v')) {
    original_encode_void = reinterpret_cast<Void5>(method_setImplementation(m, reinterpret_cast<IMP>(timed_encode_void)));
    installed |= 4;
  }
  return installed;
}

extern "C" void algan_graph_phase(int value) {
  phase.store(value >= 0 && value < 5 ? value : -1, std::memory_order_relaxed);
}

extern "C" const char *algan_graph_stats() {
  std::lock_guard<std::mutex> guard(mutex);
  const char *names[] = {"graph.init", "graph.compile", "graph.encode"};
  const char *phases[] = {"prelude", "chunk1", "after_chunk1", "chunk2", "after_chunk2"};
  std::ostringstream out; out << '[';
  bool comma = false;
  for (int p = 0; p < 5; ++p) for (int k = 0; k < 3; ++k) {
    const Row &r = rows[p][k];
    if (!r.count) continue;
    if (comma) out << ','; comma = true;
    out << "{\"phase\":\"" << phases[p] << "\",\"name\":\"" << names[k]
        << "\",\"count\":" << r.count << ",\"seconds\":" << r.total
        << ",\"self_seconds\":" << r.self << ",\"maximum\":" << r.maximum << '}';
  }
  out << ']'; serialized = out.str(); return serialized.c_str();
}
