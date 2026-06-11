#include "common_test_defs.h"

#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyIterator.h"


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


template<typename T> struct test_IvyContiguousIterator : public kernel_base_noprep_nofin{
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, std_iter::IvyContiguousIteratorBuilder<T>* it_builder){
    if (kernel_check_dims<test_IvyContiguousIterator<T>>::check_dims(i, n)){
      printf("Inside test_IvyContiguousIterator now...\n");
      auto& chain = it_builder->chain;
      printf("chain address = %p, size_ptr = %p, capacity_ptr = %p, mem_type_ptr = %p, stream = %p\n", &chain, chain.size_ptr(), chain.capacity_ptr(), chain.get_memory_type_ptr(), chain.gpu_stream());
      auto const n_size = chain.size();
      auto const n_capacity = chain.capacity();
      printf("test_IvyContiguousIterator: chain n_size = %llu, n_capacity = %llu, mem_type = %d\n", n_size, n_capacity, int(chain.get_memory_type()));

      for (auto const& v:(*it_builder)){
        printf("test_IvyContiguousIterator: v.a = %f\n", v.a);
      }

      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      std_iter::IvyContiguousIteratorBuilder<T> it_builder_copy(std_mem::addressof(*(it_builder->begin())), n_size-2, def_mem_type, nullptr);
      for (auto const& v:it_builder_copy) printf("test_IvyContiguousIterator: it_builder_copy.a = %f\n", v.a);
    }
  }
};


void utest(IvyGPUStream& stream){
  using namespace std_ivy;

  typedef std_iter::IvyContiguousIteratorBuilder<dummy_D> iterator_builder_t;
  typedef std_iter::IvyContiguousIteratorBuilder<dummy_D const> const_iterator_builder_t;
  typedef std_mem::allocator<iterator_builder_t> allocator_iterator_builder_t;
  typedef std_mem::allocator<const_iterator_builder_t> allocator_const_iterator_builder_t;
  typedef std_mem::allocator_traits<allocator_iterator_builder_t> allocator_iterator_builder_traits_t;
  typedef std_mem::allocator_traits<allocator_const_iterator_builder_t> allocator_const_iterator_builder_traits_t;

  __PRINT_INFO__("|*** Benchmarking IvyContiguousIterator basic functionality... ***|\n");

  constexpr unsigned int ndata = 10;
  std_mem::unique_ptr<dummy_D> ptr_unique = std_mem::make_unique<dummy_D>(ndata, IvyMemoryType::Host, &stream, 1.);
  for (size_t i=0; i<ndata; i++){
    ptr_unique[i].a += i;
    printf("ptr_unique[%llu].a = %f\n", i, ptr_unique[i].a);
  }

  IvyContiguousIteratorBuilder<dummy_D> it_builder;
  IvyContiguousIteratorBuilder<dummy_D const> cit_builder;
  it_builder.reset(ptr_unique.get(), ndata, ptr_unique.get_memory_type(), ptr_unique.gpu_stream());
  cit_builder.reset(ptr_unique.get(), ndata, ptr_unique.get_memory_type(), ptr_unique.gpu_stream());

  {
    auto it_begin = it_builder.begin();
    auto it_end = it_builder.end();
    auto it = it_begin;
    while (it != it_end){
      printf("it.a = %f\n", it->a);
      ++it;
    }
  }

  printf("Testing range-based for-loop...\n");
  for (auto const& obj:it_builder) printf("obj.a = %f\n", obj.a);

  ptr_unique.pop_back();
  it_builder.pop_back();
  printf("After pop_back...\n");
  {
    auto it_begin = it_builder.begin();
    auto it_end = it_builder.end();
    auto it = it_begin;
    while (it != it_end){
      printf("it.a = %f\n", it->a);
      ++it;
    }
  }

  ptr_unique.push_back(dummy_D(-7.));
  it_builder.push_back(std_mem::addressof(ptr_unique[ptr_unique.size()-1]), ptr_unique.get_memory_type(), ptr_unique.gpu_stream());
  printf("After push_back...\n");
  for (auto const& obj:it_builder) printf("obj.a = %f\n", obj.a);

  {
    auto it_middle = *(it_builder.find_pointable(ptr_unique.get()+3));
    auto cit_middle = *(cit_builder.find_pointable(ptr_unique.get()+3));
    ptr_unique.erase(3);
    it_builder.erase(3);
    cit_builder.erase(3);
    printf("After erase...\n");
    {
      auto it_begin = it_builder.begin();
      auto it_end = it_builder.end();
      auto it = it_begin;
      while (it != it_end){
        printf("it.a = %f\n", it->a);
        ++it;
      }
      if (it_middle.is_valid()) printf("it_middle.a = %f\n", it_middle->a);
      else printf("it_middle is invalid.\n");
    }
  }

  ptr_unique.insert(3, dummy_D(-11.));
  it_builder.insert(3, std_mem::addressof(ptr_unique[3]), ptr_unique.get_memory_type(), ptr_unique.gpu_stream());
  printf("After insert...\n");
  for (auto const& obj:it_builder) printf("obj.a = %f\n", obj.a);
  for (size_t i=0; i<ndata; i++){
    printf("ptr_unique[%llu].a = %f\n", i, ptr_unique[i].a);
  }

  printf("Distance end-begin of iterators: %lld\n", std_iter::distance(it_builder.begin(), it_builder.end()));

  printf("Testing empty data...\n");
  std_mem::unique_ptr<dummy_D> h_data_empty;
  IvyContiguousIteratorBuilder<dummy_D> it_builder_empty(h_data_empty.get(), 0, h_data_empty.get_memory_type(), h_data_empty.gpu_stream());
  {
    auto it_begin = it_builder_empty.begin();
    auto it_end = it_builder_empty.end();
    auto it = it_begin;
    while (it != it_end){
      printf("it.a = %f\n", it->a);
      ++it;
    }
  }

  printf("Testing GPU usage...\n");
  constexpr IvyMemoryType mem_gpu = IvyMemoryType::GPU;

  std_mem::unique_ptr<dummy_D>* h_d_ptr_unique = uniqueptr_allocator_traits::allocate(1, IvyMemoryType::Host, stream);
  uniqueptr_allocator_traits::transfer(h_d_ptr_unique, &ptr_unique, 1, IvyMemoryType::Host, IvyMemoryType::Host, stream);
  uniqueptr_allocator::transfer_internal_memory(h_d_ptr_unique, 1, IvyMemoryType::Host, mem_gpu, stream, true);
  auto ptr_h_d_ptr_unique = h_d_ptr_unique->get();
  auto n_h_d_ptr_unique = ptr_unique.size();

  printf("Allocating d_it_builder...\n");
  iterator_builder_t* d_it_builder = allocator_iterator_builder_traits_t::allocate(1, mem_gpu, stream);
  printf("Building h_d_it_builder...\n");
  iterator_builder_t* h_d_it_builder = allocator_iterator_builder_traits_t::build(
    1, IvyMemoryType::Host, stream,
    ptr_h_d_ptr_unique, n_h_d_ptr_unique, mem_gpu, &stream
  );
  printf("Transferring h_d_it_builder -> d_it_builder...\n");
  allocator_iterator_builder_traits_t::transfer(d_it_builder, h_d_it_builder, 1, mem_gpu, IvyMemoryType::Host, stream);
  printf("Destroying h_d_it_builder...\n");
  allocator_iterator_builder_traits_t::destroy(h_d_it_builder, 1, IvyMemoryType::Host, stream);

  printf("d_it_builder address: %p\n", d_it_builder);
  printf("Running the test kernel on d_it_builder...\n");
  if (IvyMemoryHelpers::use_device_acc(mem_gpu)) run_kernel<test_IvyContiguousIterator<dummy_D>>(0, stream).parallel_1D(1, d_it_builder);
  stream.synchronize();
  printf("Destroying d_it_builder...\n");
  allocator_iterator_builder_traits_t::destroy(d_it_builder, 1, mem_gpu, stream);

  uniqueptr_allocator_traits::destroy(h_d_ptr_unique, 1, IvyMemoryType::Host, stream);

  {
    printf("Testing iterator builder over GPU on stack...\n");
    iterator_builder_t d_it_builder_stack(ptr_h_d_ptr_unique, n_h_d_ptr_unique, mem_gpu, &stream);
    allocator_iterator_builder_t::transfer_internal_memory(&d_it_builder_stack, 1, IvyMemoryType::Host, mem_gpu, stream, true);
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
