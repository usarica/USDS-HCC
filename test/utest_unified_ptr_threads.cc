#include "common_test_defs.h"

#include "std_ivy/IvyMemory.h"

#include <thread>
#include <vector>


using std_ivy::IvyMemoryType;


// Regression test for the thread-safety of IvyUnifiedPtr's reference count.
// Many threads concurrently copy (increment) and destroy (decrement) copies of a
// single shared root pointer. With a non-atomic counter this races and corrupts
// the heap (confirmed previously via TSan/ASan). With an atomic counter the final
// use_count() must return exactly 1 and no memory errors must occur.
static bool test_concurrent_copy_destroy(IvyGPUStream& stream){
  bool success = true;

  std_mem::shared_ptr<double> root = std_mem::make_shared<double>(IvyMemoryType::Host, &stream, 3.14);

  constexpr int nthreads = 8;
  constexpr int niter = 20000;

  std::vector<std::thread> ts;
  ts.reserve(nthreads);
  for (int t=0; t<nthreads; ++t){
    ts.emplace_back([&root](){
      for (int i=0; i<niter; ++i){
        // Copy increments the shared count; destruction at scope exit decrements it.
        std_mem::shared_ptr<double> local = root;
        volatile double v = *local; (void) v;
      }
    });
  }
  for (auto& th : ts) th.join();

  auto const final_count = root.use_count();
  __PRINT_INFO__("test_concurrent_copy_destroy: final use_count = %llu (expected 1)\n", __STATIC_CAST__(unsigned long long, final_count));
  if (final_count != __STATIC_CAST__(std_mem::shared_ptr<double>::counter_type, 1)){
    __PRINT_ERROR__("test_concurrent_copy_destroy FAILED: use_count = %llu, expected 1\n", __STATIC_CAST__(unsigned long long, final_count));
    success = false;
  }

  return success;
}

// Threads each hold their own long-lived copy for the duration, so the shared count
// should be observed as (nthreads + 1) while they are alive and return to 1 afterwards.
static bool test_concurrent_shared_count(IvyGPUStream& stream){
  bool success = true;

  std_mem::shared_ptr<double> root = std_mem::make_shared<double>(IvyMemoryType::Host, &stream, 2.71);

  constexpr int nthreads = 8;
  constexpr int nrepeat = 5000;

  std::vector<std::thread> ts;
  ts.reserve(nthreads);
  for (int t=0; t<nthreads; ++t){
    ts.emplace_back([&root](){
      for (int i=0; i<nrepeat; ++i){
        std_mem::shared_ptr<double> a = root;
        std_mem::shared_ptr<double> b = a;
        std_mem::shared_ptr<double> c = b;
        c.reset();
        b.reset();
        a.reset();
      }
    });
  }
  for (auto& th : ts) th.join();

  auto const final_count = root.use_count();
  __PRINT_INFO__("test_concurrent_shared_count: final use_count = %llu (expected 1)\n", __STATIC_CAST__(unsigned long long, final_count));
  if (final_count != __STATIC_CAST__(std_mem::shared_ptr<double>::counter_type, 1)){
    __PRINT_ERROR__("test_concurrent_shared_count FAILED: use_count = %llu, expected 1\n", __STATIC_CAST__(unsigned long long, final_count));
    success = false;
  }

  return success;
}


void utest(IvyGPUStream& stream){
  __PRINT_INFO__("|*** Benchmarking IvyUnifiedPtr thread safety... ***|\n");

  bool success = true;
  success &= test_concurrent_copy_destroy(stream);
  success &= test_concurrent_shared_count(stream);

  if (!success){
    __PRINT_ERROR__("IvyUnifiedPtr thread-safety test FAILED.\n");
    assert(false);
  }

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}

int main(){
  IvyGPUStream* stream = IvyStreamUtils::make_global_gpu_stream();

  utest(*stream);

  IvyStreamUtils::destroy_stream(stream);

  return 0;
}
