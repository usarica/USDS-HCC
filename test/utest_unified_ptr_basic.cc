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


struct print_dummy_D : public kernel_base_noprep_nofin{
  static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void kernel_unified_unit(dummy_D* ptr){
    if (ptr) printf("print_dummy_D: dummy_D address = %p, a = %f\n", ptr, ptr->a);
    else printf("print_dummy_D: dummy_D is null.\n");
  }
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, dummy_D* ptr){
    if (kernel_check_dims<print_dummy_D>::check_dims(i, n)) kernel_unified_unit(ptr);
  }
};

struct print_dummy_B_as_D : public kernel_base_noprep_nofin{
  static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void kernel_unified_unit(dummy_B* ptr){
    if (ptr) printf("print_dummy_B_as_D: dummy_B address = %p, a = %f\n", ptr, __STATIC_CAST__(dummy_D*, ptr)->a);
    else printf("print_dummy_B_as_D: dummy_B is null.\n");
  }
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, dummy_B* ptr){
    if (kernel_check_dims<print_dummy_B_as_D>::check_dims(i, n)) kernel_unified_unit(ptr);
  }
};

template<std_mem::IvyPointerType IPT> struct print_pointer_to_dummy_D : public kernel_base_noprep_nofin{
  static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void kernel_unified_unit(std_mem::IvyUnifiedPtr<dummy_D, IPT>* ptr){
    if (ptr){
      printf("print_pointer_to_dummy_D: IvyUnifiedPtr<dummy_D, %d> address = %p, size = %llu\n", IPT, ptr, ptr->size());
      if (ptr->get()){
        for (size_t i=0; i<ptr->size(); ++i){
          auto p = (ptr->get()+i);
          printf("print_pointer_to_dummy_D: IvyUnifiedPtr<dummy_D, %d>[%llu] address = %p\n", IPT, i, p);
          if (p) printf("print_pointer_to_dummy_D: dummy_D address = %p, a = %f\n", p, p->a);
          else printf("print_pointer_to_dummy_D: dummy_D is null.\n");
        }
      }
    }
    else printf("print_pointer_to_dummy_D: Unified pointer of dummy_D is null.\n");
  }
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, std_mem::IvyUnifiedPtr<dummy_D, IPT>* ptr){
    if (kernel_check_dims<print_pointer_to_dummy_D<IPT>>::check_dims(i, n)) kernel_unified_unit(ptr);
  }
};

template<std_mem::IvyPointerType IPT> struct print_unifiedptr_cx_to_dummy_D : public kernel_base_noprep_nofin{
  static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void kernel_unified_unit(std_mem::IvyUnifiedPtr<std_mem::IvyUnifiedPtr<dummy_D, IPT>, IPT>* ptr){
    if (ptr){
      printf("kernel_print_unifiedptr_cx_to_dummy_D: IvyUnifiedPtr<IvyUnifiedPtr<dummy_D, %d>> address = %p, size = %llu\n", IPT, ptr, ptr->size());
      if (ptr->get()){
        for (size_t i=0; i<ptr->size(); ++i){
          auto p = (ptr->get()+i);
          printf("kernel_print_unifiedptr_cx_to_dummy_D: IvyUnifiedPtr<dummy_D, %d>[%llu] address = %p, size = %llu\n", IPT, i, p, p->size());
          if (p->get()){
            for (size_t j=0; j<p->size(); ++j){
              auto q = (p->get()+j);
              printf("kernel_print_unifiedptr_cx_to_dummy_D: IvyUnifiedPtr<dummy_D, %d>[%llu][%llu] address = %p\n", IPT, i, j, q);
              if (q) printf("kernel_print_unifiedptr_cx_to_dummy_D: dummy_D address = %p, a = %f\n", q, q->a);
              else printf("kernel_print_unifiedptr_cx_to_dummy_D: dummy_D is null.\n");
            }
          }
        }
      }
      else printf("kernel_print_unifiedptr_cx_to_dummy_D: Unified pointer of dummy_D is null.\n");
    }
    else printf("kernel_print_unifiedptr_cx_to_dummy_D: Unified cx pointer of dummy_D is null.\n");
  }
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, std_mem::IvyUnifiedPtr<std_mem::IvyUnifiedPtr<dummy_D, IPT>, IPT>* ptr){
    if (kernel_check_dims<print_unifiedptr_cx_to_dummy_D<IPT>>::check_dims(i, n)) kernel_unified_unit(ptr);
  }
};


void utest(IvyGPUStream& stream){
  __PRINT_INFO__("|*** Benchmarking IvyUnifiedPtr basic functionality... ***|\n");

  {
    __PRINT_INFO__("Testing shared_ptr...\n");
    __PRINT_INFO__("  Size of shared_ptr<dummy_D>: %llu\n", sizeof(std_mem::shared_ptr<dummy_D>));
    std_mem::shared_ptr<dummy_D> ptr_shared = std_mem::make_shared<dummy_D>(IvyMemoryType::GPU, &stream, 1.);
    std_mem::shared_ptr<dummy_B> ptr_shared_copy = ptr_shared; ptr_shared_copy.reset(); ptr_shared_copy = ptr_shared;
    __PRINT_INFO__("  ptr_shared no. of copies: %i\n", ptr_shared.use_count());
    __PRINT_INFO__("  ptr_shared_copy no. of copies: %i\n", ptr_shared_copy.use_count());
    run_kernel<print_dummy_D>(0, stream).parallel_1D(1, ptr_shared.get());
    run_kernel<print_dummy_B_as_D>(0, stream).parallel_1D(1, ptr_shared_copy.get());
    ptr_shared_copy.reset();
    __PRINT_INFO__("ptr_shared no. of copies after reset: %i\n", ptr_shared.use_count());
    __PRINT_INFO__("ptr_shared_copy no. of copies after reset: %i\n", ptr_shared_copy.use_count());
  }
  {
    __PRINT_INFO__("Testing unique_ptr...\n");
    __PRINT_INFO__("  Size of unique_ptr<dummy_D>: %llu\n", sizeof(std_mem::unique_ptr<dummy_D>));
    std_mem::unique_ptr<dummy_D> ptr_unique = std_mem::make_unique<dummy_D>(IvyMemoryType::GPU, &stream, 1.);
    std_mem::unique_ptr<dummy_B> ptr_unique_copy = ptr_unique;
    __PRINT_INFO__("ptr_unique no. of copies: %i\n", ptr_unique.use_count());
    __PRINT_INFO__("ptr_unique_copy no. of copies: %i\n", ptr_unique_copy.use_count());
    run_kernel<print_dummy_D>(0, stream).parallel_1D(1, ptr_unique.get());
    run_kernel<print_dummy_B_as_D>(0, stream).parallel_1D(1, ptr_unique_copy.get());
    ptr_unique_copy.reset();
    __PRINT_INFO__("ptr_unique no. of copies after reset: %i\n", ptr_unique.use_count());
    __PRINT_INFO__("ptr_unique_copy no. of copies after reset: %i\n", ptr_unique_copy.use_count());
  }
  {
    __PRINT_INFO__("\t- Testing IvyUnifiedPtr memory allocation...\n");
    std_mem::unique_ptr<dummy_D> h_unique_transferable = std_mem::make_unique<dummy_D>(IvyMemoryType::Host, &stream, 1.);
    __PRINT_INFO__("Allocating h_ptr_unique...\n");
    std_mem::unique_ptr<dummy_D>* h_ptr_unique = uniqueptr_allocator_traits::allocate(1, IvyMemoryType::Host, stream);
    uniqueptr_allocator_traits::transfer(h_ptr_unique, &h_unique_transferable, 1, IvyMemoryType::Host, IvyMemoryType::Host, stream);
    __PRINT_INFO__("h_unique_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_unique_transferable.use_count(), h_unique_transferable.get(), h_unique_transferable->a);
    __PRINT_INFO__("h_ptr_unique no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_ptr_unique->use_count(), h_ptr_unique->get(), (*h_ptr_unique)->a);
    __PRINT_INFO__("Destroying h_ptr_unique...\n");
    uniqueptr_allocator_traits::destroy(h_ptr_unique, 1, IvyMemoryType::Host, stream);
  }
  {
    __PRINT_INFO__("\t- Testing IvyUnifiedPtr<IvyUnifiedPtr> memory allocation...\n");
    std_mem::shared_ptr<dummy_D> ptr_shared = std_mem::make_shared<dummy_D>(3, 4, IvyMemoryType::GPU, &stream, 1.);
    sharedptr_allocator::transfer_internal_memory(&ptr_shared, 1, IvyMemoryType::Host, IvyMemoryType::GPU, stream, true);
    std_mem::shared_ptr<std_mem::shared_ptr<dummy_D>> ptr_cx_shared = std_mem::make_shared<std_mem::shared_ptr<dummy_D>>(2, 3, IvyMemoryType::GPU, &stream, ptr_shared);
    std_mem::shared_ptr<std_mem::shared_ptr<dummy_D>>* d_ptr_cx_shared_copy = sharedptr_sharedptr_allocator_traits::allocate(1, IvyMemoryType::GPU, stream);
    sharedptr_sharedptr_allocator_traits::transfer(d_ptr_cx_shared_copy, &ptr_cx_shared, 1, IvyMemoryType::GPU, IvyMemoryType::Host, stream);
    stream.synchronize();
    printf("Calling print_pointer_to_dummy_D...\n");
    run_kernel<print_pointer_to_dummy_D<std_mem::pointer_traits<std_mem::shared_ptr<dummy_D>>::ivy_ptr_type>>(0, stream).parallel_1D(1, ptr_cx_shared.get());
    stream.synchronize();
    printf("Calling print_unifiedptr_cx_to_dummy_D...\n");
    run_kernel<print_unifiedptr_cx_to_dummy_D<std_mem::pointer_traits<std_mem::shared_ptr<dummy_D>>::ivy_ptr_type>>(0, stream).parallel_1D(1, d_ptr_cx_shared_copy);
    stream.synchronize();
    printf("Destroying d_ptr_cx_shared_copy...\n");
    sharedptr_sharedptr_allocator_traits::destroy(d_ptr_cx_shared_copy, 1, IvyMemoryType::GPU, stream);
    printf("Done destroying d_ptr_cx_shared_copy...\n");
  }

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
