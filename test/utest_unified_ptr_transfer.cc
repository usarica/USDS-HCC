#include "common_test_defs.h"

#include "std_ivy/IvyMemory.h"


using std_ivy::IvyMemoryType;


class dummy_B{
public:
  dummy_B() = default;
};
class dummy_D : public dummy_B{
public:
  double a;
  __HOST_DEVICE__ dummy_D() : dummy_B(){}
  __HOST_DEVICE__ dummy_D(double a_) : dummy_B(), a(a_){}
  __HOST_DEVICE__ dummy_D(dummy_D const& other) : a(other.a){}
  //__HOST_DEVICE__ ~dummy_D(){ printf("dummy D destructor...\n"); }
};


typedef std_mem::allocator<std_mem::unique_ptr<dummy_D>> uniqueptr_allocator;
typedef std_mem::allocator_traits<uniqueptr_allocator> uniqueptr_allocator_traits;
typedef std_mem::allocator<std_mem::shared_ptr<dummy_D>> sharedptr_allocator;
typedef std_mem::allocator_traits<sharedptr_allocator> sharedptr_allocator_traits;

typedef std_mem::allocator<std_mem::unique_ptr<std_mem::unique_ptr<dummy_D>>> uniqueptr_uniqueptr_allocator;
typedef std_mem::allocator_traits<uniqueptr_uniqueptr_allocator> uniqueptr_uniqueptr_allocator_traits;
typedef std_mem::allocator<std_mem::shared_ptr<std_mem::shared_ptr<dummy_D>>> sharedptr_sharedptr_allocator;
typedef std_mem::allocator_traits<sharedptr_sharedptr_allocator> sharedptr_sharedptr_allocator_traits;

template<typename T> struct test_unifiedptr_ptr : public kernel_base_noprep_nofin{
  static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void kernel_unified_unit(T* ptr){
    if (ptr){
      printf("Calling test_unifiedptr_ptr...\n");
      printf("test_unifiedptr_ptr: Device ptr ptrs: address, ptr, mem_type, stream, size, counter, exec_mem = %p, %p, %p, %p, %p, %p, %u\n", ptr, ptr->get(), ptr->get_memory_type_ptr(), ptr->gpu_stream(), ptr->size_ptr(), ptr->counter(), __STATIC_CAST__(unsigned int, ptr->get_exec_memory_type()));
      if (ptr->get()) printf("test_unifiedptr_ptr: Device ptr no. of copies, counter val., memory type, dummy_D.a: %llu, %llu, %u, %f\n", ptr->use_count(), *(ptr->counter()), __STATIC_CAST__(unsigned int, ptr->get_memory_type()), (*ptr)->a);
      else printf("test_unifiedptr_ptr: Device ptr is null.\n");
    }
    else printf("test_unifiedptr_ptr: Pointer is null.\n");
  }
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, T* ptr){
    if (kernel_check_dims<test_unifiedptr_ptr<T>>::check_dims(i, n)) kernel_unified_unit(ptr);
  }
};
template<typename T, std_mem::IvyPointerType IPT> struct test_unifiedptr_fcn{
  static __HOST_DEVICE__ void prepare(...){
    printf("Preparing for test_unifiedptr_fcn...\n");
  }
  static __HOST_DEVICE__ void finalize(...){
    printf("Finalizing test_unifiedptr_fcn...\n");
  }
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, std_mem::IvyUnifiedPtr<T, IPT>* ptr){
    if (kernel_check_dims<test_unifiedptr_fcn<T, IPT>>::check_dims(i, n)){
      printf("Inside test_unifiedptr_fcn now...\n");
      //ptr->reset();
      printf("test_unifiedptr_fcn: get = %p\n", ptr->get());
    }
  }
};


void utest(IvyGPUStream& stream){
  __PRINT_INFO__("|*** Benchmarking IvyUnifiedPtr transfer functionalities... ***|\n");

  std_mem::shared_ptr<dummy_D> h_shared_transferable = std_mem::make_shared<dummy_D>(IvyMemoryType::Host, &stream, 1.);
  __PRINT_INFO__("Allocating h_ptr_shared...\n");
  std_mem::shared_ptr<dummy_D>* h_ptr_shared = sharedptr_allocator_traits::allocate(1, IvyMemoryType::Host, stream);
  __PRINT_INFO__("Transferring h_shared_transferable to h_ptr_shared...\n");
  sharedptr_allocator_traits::transfer(h_ptr_shared, &h_shared_transferable, 1, IvyMemoryType::Host, IvyMemoryType::Host, stream);
  __PRINT_INFO__("h_shared_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
  __PRINT_INFO__("h_ptr_shared no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_ptr_shared->use_count(), h_ptr_shared->get(), (*h_ptr_shared)->a);
  __PRINT_INFO__("Destroying h_ptr_shared...\n");
  sharedptr_allocator_traits::destroy(h_ptr_shared, 1, IvyMemoryType::Host, stream);
  __PRINT_INFO__("h_shared_transferable (after destroying h_ptr_shared) no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
  __PRINT_INFO__("Allocating d_ptr_shared...\n");
  std_mem::shared_ptr<dummy_D>* d_ptr_shared = sharedptr_allocator_traits::allocate(1, IvyMemoryType::GPU, stream);
  __PRINT_INFO__("Allocating d_ptr_shared_dtransfer...\n");
  std_mem::shared_ptr<dummy_D>* d_ptr_shared_dtransfer = sharedptr_allocator_traits::allocate(1, IvyMemoryType::GPU, stream);
  __PRINT_INFO__("Transfering h_shared_transferable to d_ptr_shared...\n");
  sharedptr_allocator_traits::transfer(d_ptr_shared, &h_shared_transferable, 1, IvyMemoryType::GPU, IvyMemoryType::Host, stream);
  stream.synchronize();
  __PRINT_INFO__("h_shared_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
  __PRINT_INFO__("Running kernel test on d_ptr_shared...\n");
  run_kernel<test_unifiedptr_ptr<std_mem::shared_ptr<dummy_D>>>(0, stream).parallel_1D(1, d_ptr_shared);
  run_kernel<test_unifiedptr_fcn<dummy_D, std_mem::IvyPointerType::shared>>(0, stream).parallel_1D(1, d_ptr_shared);
  stream.synchronize();

  __PRINT_INFO__("Transfering d_ptr_shared to d_ptr_shared_dtransfer...\n");
  sharedptr_allocator_traits::transfer(d_ptr_shared_dtransfer, d_ptr_shared, 1, IvyMemoryType::GPU, IvyMemoryType::GPU, stream);
  stream.synchronize();
  __PRINT_INFO__("h_shared_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
  __PRINT_INFO__("Running kernel test on d_ptr_shared_dtransfer...\n");
  run_kernel<test_unifiedptr_ptr<std_mem::shared_ptr<dummy_D>>>(0, stream).parallel_1D(1, d_ptr_shared_dtransfer);
  run_kernel<test_unifiedptr_fcn<dummy_D, std_mem::IvyPointerType::shared>>(0, stream).parallel_1D(1, d_ptr_shared_dtransfer);
  stream.synchronize();
  __PRINT_INFO__("Destroying d_ptr_shared...\n");
  sharedptr_allocator_traits::destroy(d_ptr_shared, 1, IvyMemoryType::GPU, stream);
  __PRINT_INFO__("Destroying d_ptr_shared_dtransfer...\n");
  sharedptr_allocator_traits::destroy(d_ptr_shared_dtransfer, 1, IvyMemoryType::GPU, stream);

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}


int main(){
  constexpr unsigned char nStreams = 3;

  IvyGPUStream* streams[nStreams]{
    IvyStreamUtils::make_global_gpu_stream(),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking)
  };

  for (unsigned char i = 0; i < nStreams; i++){
    __PRINT_INFO__("**********\n");

    auto& stream = *(streams[i]);
    __PRINT_INFO__("Stream %i (%p, %p, size in bytes = %llu) computing...\n", i, &stream, (void*) &stream.stream(), __STATIC_CAST__(unsigned long long, sizeof(&stream)));

    utest(stream);

    __PRINT_INFO__("**********\n");
  }

  for (unsigned char i = 0; i < nStreams; i++) IvyStreamUtils::destroy_stream(streams[i]);

  return 0;
}
