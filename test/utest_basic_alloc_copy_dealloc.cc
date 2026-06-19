#include "common_test_defs.h"

#include "std_ivy/IvyMemory.h"


using std_ivy::IvyMemoryType;


struct set_doubles : public kernel_base_noprep_nofin{
  static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void kernel_unified_unit(IvyTypes::size_t i, IvyTypes::size_t n, double* ptr, unsigned char is){
    ptr[i] = (i+2)*(is+1);
    if (i<3 || i==n-1) printf("ptr[%llu] = %f in stream %u\n", static_cast<unsigned long long int>(i), ptr[i], static_cast<unsigned int>(is));
  }
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, double* ptr, unsigned char is){
    if (kernel_check_dims<set_doubles>::check_dims(i, n)) kernel_unified_unit(i, n, ptr, is);
  }
};


template<unsigned int nvars> void utest(IvyGPUStream& stream, unsigned int i_st){
  __PRINT_INFO__("|*** Benchmarking basic data allocation, transfer, and deallocation... ***|\n");

  typedef std_mem::allocator<double> obj_allocator;
  typedef std_mem::allocator<int> obj_allocator_i;

  int* ptr_i = nullptr;

  auto ptr_h = obj_allocator::build(nvars, IvyMemoryType::Host, stream);
  __PRINT_INFO__("ptr_h = %p\n", ptr_h);
  ptr_h[0] = 1.0;
  ptr_h[1] = 2.0;
  ptr_h[2] = 3.0;
  __PRINT_INFO__("ptr_h values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
  obj_allocator::destroy(ptr_h, nvars, IvyMemoryType::Host, stream);

  __PRINT_INFO__("Trying device...\n");

  IvyGPUEvent ev_build(IvyGPUEvent::EventFlags::Default); ev_build.record(stream);
  auto ptr_d = obj_allocator::build(nvars, IvyMemoryType::GPU, stream);
  IvyGPUEvent ev_build_end(IvyGPUEvent::EventFlags::Default); ev_build_end.record(stream);
  ev_build_end.synchronize();
  auto time_build = ev_build_end.elapsed_time(ev_build);
  __PRINT_INFO__("Construction time = %f ms\n", time_build);
  __PRINT_INFO__("ptr_d = %p\n", ptr_d);

  ptr_h = obj_allocator::build(nvars, IvyMemoryType::Host, stream);
  __PRINT_INFO__("ptr_h new = %p\n", ptr_h);

  {
    IvyGPUEvent ev_set(IvyGPUEvent::EventFlags::Default); ev_set.record(stream);
    run_kernel<set_doubles>(0, stream).parallel_1D(nvars, ptr_d, i_st);
    IvyGPUEvent ev_set_end(IvyGPUEvent::EventFlags::Default); ev_set_end.record(stream);
    ev_set_end.synchronize();
    auto time_set = ev_set_end.elapsed_time(ev_set);
    __PRINT_INFO__("Set time = %f ms\n", time_set);
  }

  IvyGPUEvent ev_transfer(IvyGPUEvent::EventFlags::Default); ev_transfer.record(stream);
  obj_allocator::transfer(ptr_h, ptr_d, nvars, IvyMemoryType::Host, IvyMemoryType::GPU, stream);
  IvyGPUEvent ev_transfer_end(IvyGPUEvent::EventFlags::Default); ev_transfer_end.record(stream);
  ev_transfer_end.synchronize();
  auto time_transfer = ev_transfer_end.elapsed_time(ev_transfer);
  __PRINT_INFO__("Transfer time = %f ms\n", time_transfer);

  IvyGPUEvent ev_cp(IvyGPUEvent::EventFlags::Default); ev_cp.record(stream);
  IvyMemoryHelpers::copy_data(ptr_i, ptr_h, 0, nvars, nvars, IvyMemoryType::Host, IvyMemoryType::Host, stream);
  IvyGPUEvent ev_cp_end(IvyGPUEvent::EventFlags::Default); ev_cp_end.record(stream);
  ev_cp_end.synchronize();
  auto time_cp = ev_cp_end.elapsed_time(ev_cp);
  __PRINT_INFO__("Copy time = %f ms\n", time_cp);

  __PRINT_INFO__("ptr_h new values: %f, %f, %f, ..., %f\n", ptr_h[0], ptr_h[1], ptr_h[2], ptr_h[nvars-1]);
  __PRINT_INFO__("ptr_i new values: %d, %d, %d, ..., %d\n", ptr_i[0], ptr_i[1], ptr_i[2], ptr_i[nvars-1]);
  obj_allocator::destroy(ptr_h, nvars, IvyMemoryType::Host, stream);
  obj_allocator_i::destroy(ptr_i, nvars, IvyMemoryType::Host, stream);

  IvyGPUEvent ev_destroy(IvyGPUEvent::EventFlags::Default); ev_destroy.record(stream);
  obj_allocator::destroy(ptr_d, nvars, IvyMemoryType::GPU, stream);
  IvyGPUEvent ev_destroy_end(IvyGPUEvent::EventFlags::Default); ev_destroy_end.record(stream);
  ev_destroy_end.synchronize();
  auto time_destroy = ev_destroy_end.elapsed_time(ev_destroy);
  __PRINT_INFO__("Destruction time = %f ms\n", time_destroy);

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}


int main(){
  constexpr unsigned char nStreams = 3;
  constexpr unsigned int nvars = 10000; // Can be up to ~700M

  IvyGPUStream* streams[nStreams]{
    IvyStreamUtils::make_global_gpu_stream(),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking)
  };

  for (unsigned char i = 0; i < nStreams; i++){
    __PRINT_INFO__("**********\n");

    auto& stream = *(streams[i]);
    __PRINT_INFO__("Stream %i (%p, %p, size in bytes = %llu) computing...\n", i, &stream, (void*) &stream.stream(), __STATIC_CAST__(unsigned long long, sizeof(&stream)));

    utest<nvars>(stream, i);

    __PRINT_INFO__("**********\n");
  }

  // Clean up local streams
  for (unsigned char i = 0; i < nStreams; i++) IvyStreamUtils::destroy_stream(streams[i]);

  return 0;
}
