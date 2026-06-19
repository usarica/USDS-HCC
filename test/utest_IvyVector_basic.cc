#include "common_test_defs.h"

#include "std_ivy/IvyIterator.h"
#include "std_ivy/IvyVector.h"


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

typedef std_mem::allocator<std_vec::vector<dummy_D>> vector_allocator;
typedef std_mem::allocator_traits<vector_allocator> vector_allocator_traits;


template<typename T> struct test_IvyVector : public kernel_base_noprep_nofin{
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, std_vec::vector<T>* ptr){
    if (kernel_check_dims<test_IvyVector<T>>::check_dims(i, n)){
      printf("Inside test_IvyVector now...\n");
      printf("test_IvyVector: ptr = %p, size = %llu, capacity = %llu\n", ptr, ptr->size(), ptr->capacity());

      printf("Obtaining data container...\n");
      auto const& data = ptr->get_data_container();
      printf("data address = %p, size_ptr = %p, capacity_ptr = %p, mem_type_ptr = %p\n", data.get(), data.size_ptr(), data.capacity_ptr(), data.get_memory_type_ptr());

      printf("Obtaining it_builder...\n");
      auto const& it_builder = ptr->get_iterator_builder();
      printf(
        "chain head = %p, size_ptr = %p, capacity_ptr = %p, mem_type_ptr = %p\n",
        it_builder.chain.get(), it_builder.chain.size_ptr(), it_builder.chain.capacity_ptr(), it_builder.chain.get_memory_type_ptr()
      );

      printf("Obtaining begin...\n");
      auto it_begin = ptr->begin();
      printf("Obtaining end...\n");
      auto it_end = ptr->end();
      printf("Looping from begin to end...\n");
      size_t j = 0;
      for (auto it=it_begin; it!=it_end; ++it){
        printf("test_IvyVector: ptr[%llu] = %f ?= %f\n", j, it->a, ptr->at(j).a);
        printf("  it.prev = %p, it.next = %p\n", it.prev(), it.next());
        ++j;
      }
      j = 0;
      printf("Range-based version:\n");
      for (auto const& v:(*ptr)){
        printf("test_IvyVector: ptr[%llu] = %f ?= %f\n", j, v.a, ptr->at(j).a);
        ++j;
      }
    }
  }
};


void utest(IvyGPUStream& stream){
  __PRINT_INFO__("|*** Benchmarking IvyVector basic functionality... ***|\n");

  constexpr size_t n_initial = 10;
  std_vec::vector<dummy_D> h_vec(n_initial, IvyMemoryType::Host, &stream, 1.);
  { unsigned short j=0; for (auto& v:h_vec){ v.a += j; ++j; } }
  for (auto const& v:h_vec) __PRINT_INFO__("h_vec.a = %f\n", v.a);
  //std_vec::vector<dummy_D>* h_vec = vector_allocator_traits::build(1, IvyMemoryType::Host, stream, n_initial, IvyMemoryType::Host, &stream, 1.);
  //{ unsigned short j=0; for (auto& v:*h_vec){ v.a += j; ++j; } }
  //for (auto const& v:*h_vec) __PRINT_INFO__("h_vec.a = %f\n", v.a);

  h_vec.push_back(IvyMemoryType::Host, &stream, dummy_D(11.));
  __PRINT_INFO__("h_vec after push_back #1...\n");
  for (auto const& v:h_vec) __PRINT_INFO__("h_vec.a = %f\n", v.a);

  h_vec.reserve(h_vec.capacity()+2);
  h_vec.push_back(IvyMemoryType::Host, &stream, dummy_D(12.));
  __PRINT_INFO__("h_vec after push_back #2...\n");
  for (auto const& v:h_vec) __PRINT_INFO__("h_vec.a = %f\n", v.a);

  {
    std_ilist::initializer_list<dummy_D> ilist{ dummy_D(-3.), dummy_D(-5.), dummy_D(-7.) };
    auto it_ins = h_vec.insert(h_vec.cbegin()+3, ilist, IvyMemoryType::Host, &stream);
    __PRINT_INFO__("First inserted element (loop #1): it_ins->a = %f\n", it_ins->a);
  }
  __PRINT_INFO__("h_vec after insert #1...\n");
  for (auto const& v:h_vec) __PRINT_INFO__("h_vec.a = %f\n", v.a);
  {
    std_vec::vector<dummy_D> h_ins_vec(3, IvyMemoryType::Host, &stream, -100.);
    { unsigned short j=0; for (auto& v:h_ins_vec){ v.a -= j; ++j; } }
    auto it_ins = h_vec.insert(h_vec.cbegin()+2, h_ins_vec.begin(), h_ins_vec.end(), IvyMemoryType::Host, &stream);
    __PRINT_INFO__("First inserted element (loop #2): it_ins->a = %f\n", it_ins->a);
  }
  __PRINT_INFO__("h_vec after insert #2...\n");
  for (auto const& v:h_vec) __PRINT_INFO__("h_vec.a = %f\n", v.a);

  {
    h_vec.erase(h_vec.cbegin()+6);
    h_vec.erase(h_vec.cbegin()+2);
    auto it_ers = h_vec.erase(h_vec.cbegin()+6);
    __PRINT_INFO__("First element after all erases: it_ers->a = %f\n", it_ers->a);
  }
  __PRINT_INFO__("h_vec after erase...\n");
  for (auto const& v:h_vec) __PRINT_INFO__("h_vec.a = %f\n", v.a);

  h_vec.pop_back(); h_vec.pop_back();
  __PRINT_INFO__("h_vec after pop_back...\n");
  for (auto const& v:h_vec) __PRINT_INFO__("h_vec.a = %f\n", v.a);

  __PRINT_INFO__("Testing vector iterator difference between begin+1 and begin+4...\n");
  {
    auto it_begin1 = h_vec.begin()+1;
    auto it_begin4 = h_vec.begin()+4;
    __PRINT_INFO__("it_begin1->a = %f, it_begin4->a = %f\n", it_begin1->a, it_begin4->a);
    __PRINT_INFO__("it_begin4-it_begin1 = %lld\n", it_begin4-it_begin1);
  }

  __PRINT_INFO__("Testing vector functionality on GPU...\n");
  __PRINT_INFO__("Allocating d_vec...\n");
  std_vec::vector<dummy_D>* d_vec = vector_allocator_traits::allocate(1, IvyMemoryType::GPU, stream);
  __PRINT_INFO__("Transferring h_vec to d_vec...\n");
  vector_allocator_traits::transfer(d_vec, &h_vec, 1, IvyMemoryType::GPU, IvyMemoryType::Host, stream);
  __PRINT_INFO__("Running the test kernel on d_vec...\n");
  run_kernel<test_IvyVector<dummy_D>>(0, stream).parallel_1D(1, d_vec);
  stream.synchronize();
  __PRINT_INFO__("Destroying d_vec...\n");
  vector_allocator_traits::destroy(d_vec, 1, IvyMemoryType::GPU, stream);
  stream.synchronize();
  //__PRINT_INFO__("Destroying h_vec...\n");
  //vector_allocator_traits::destroy(h_vec, 1, IvyMemoryType::Host, stream);

  __PRINT_INFO__("Empty vector check...\n");
  std_vec::vector<dummy_D> e_vec;
  for (auto const& v: e_vec){}
  __PRINT_INFO__("Empty vector check completed.\n");

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
