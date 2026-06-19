#include "common_test_defs.h"

#include "std_ivy/IvyUnorderedMap.h"


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
template<> struct std_ivy::value_printout<dummy_D>{ static __HOST_DEVICE__ void print(dummy_D const& x){ std_ivy::print_value(x.a, false); } };


typedef std_mem::allocator<std_umap::unordered_map<double, dummy_D>> umap_allocator;
typedef std_mem::allocator_traits<umap_allocator> umap_allocator_traits;


template<typename T, typename U> struct test_IvyUMap : public kernel_base_noprep_nofin{
  static __HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, std_umap::unordered_map<T, U>* ptr){
    if (kernel_check_dims<test_IvyUMap<T, U>>::check_dims(i, n)){
      printf("Inside test_IvyUMap now...\n");
      printf("test_IvyUMap: ptr = %p, size = %llu, capacity = %llu\n", ptr, ptr->size(), ptr->capacity());

      printf("Obtaining begin...\n");
      auto it_begin = ptr->begin();
      printf("Obtaining end...\n");
      auto it_end = ptr->end();
      printf("Looping from begin to end...\n");
      size_t j = 0;
      for (auto it=it_begin; it!=it_end; ++it){
        printf("test_IvyUMap: ptr[%llu] = %f\n", j, it->second.a);
        printf("  it.prev = %p, it.next = %p\n", it.prev(), it.next());
        ++j;
      }
      j = 0;
      printf("Range-based version:\n");
      for (auto const& v:(*ptr)){
        printf("test_IvyUMap: ptr[%llu] = %f\n", j, v.second.a);
        ++j;
      }
    }
  }
};


void utest(IvyGPUStream& stream){
  using namespace std_ivy;
  typedef double key_type;
  typedef dummy_D mapped_type;

  __PRINT_INFO__("|*** Benchmarking IvyUnorderedMap basic functionality... ***|\n");

  constexpr size_t ndata = 10;
  std_mem::unique_ptr<mapped_type> raw_data = std_mem::make_unique<mapped_type>(ndata, IvyMemoryType::Host, &stream, 1.);
  std_umap::unordered_map<key_type, mapped_type> h_umap;
  __PRINT_INFO__("Adding key-value pairs to h_umap...\n");
  for (size_t i=0; i<raw_data.size(); ++i){
    raw_data[i].a += i;
    h_umap.emplace(IvyMemoryType::Host, &stream, raw_data[i].a, raw_data[i]);
  }

  __PRINT_INFO__("h_umap[3].a = %f\n", h_umap[3].a);
  __PRINT_INFO__("h_umap[-1].a = %f\n", h_umap(-1, dummy_D(-1)).a);

  __PRINT_INFO__("Extracting h_umap iterators...\n");
  auto it_begin = h_umap.begin();
  auto it_end = h_umap.end();
  __PRINT_INFO__("Iterating over h_umap...\n");
  for (auto it=it_begin; it!=it_end; ++it){ __PRINT_INFO__("(iterator loop) h_umap[%f].a = %f\n", it->first, it->second.a); }
  for (auto const& kv:h_umap) __PRINT_INFO__("(range-based loop) h_umap[%f].a = %f\n", kv.first, kv.second.a);
  __PRINT_INFO__("Printing h_umap via print_value...\n");
  print_value(h_umap);

  __PRINT_INFO__("Testing unordered_map functionality on GPU...\n");
  __PRINT_INFO__("Allocating d_umap...\n");
  std_umap::unordered_map<key_type, mapped_type>* d_umap = umap_allocator_traits::allocate(1, IvyMemoryType::GPU, stream);
  __PRINT_INFO__("Transferring h_umap to d_umap...\n");
  umap_allocator_traits::transfer(d_umap, &h_umap, 1, IvyMemoryType::GPU, IvyMemoryType::Host, stream);
  __PRINT_INFO__("Running the test kernel on d_umap...\n");
  run_kernel<test_IvyUMap<key_type, mapped_type>>(0, stream).parallel_1D(1, d_umap);
  stream.synchronize();
  __PRINT_INFO__("Destroying d_umap...\n");
  umap_allocator_traits::destroy(d_umap, 1, IvyMemoryType::GPU, stream);
  stream.synchronize();

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
