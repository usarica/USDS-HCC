#include "common_test_defs.h"

#include "std_ivy/IvyIterator.h"
#include "std_ivy/IvyMemory.h"
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


typedef std_mem::allocator<std_mem::unique_ptr<dummy_D>> uniqueptr_allocator;
typedef std_mem::allocator_traits<uniqueptr_allocator> uniqueptr_allocator_traits;
typedef std_mem::allocator<std_mem::shared_ptr<dummy_D>> sharedptr_allocator;
typedef std_mem::allocator_traits<sharedptr_allocator> sharedptr_allocator_traits;

typedef std_mem::allocator<std_mem::unique_ptr<std_mem::unique_ptr<dummy_D>>> uniqueptr_uniqueptr_allocator;
typedef std_mem::allocator_traits<uniqueptr_uniqueptr_allocator> uniqueptr_uniqueptr_allocator_traits;
typedef std_mem::allocator<std_mem::shared_ptr<std_mem::shared_ptr<dummy_D>>> sharedptr_sharedptr_allocator;
typedef std_mem::allocator_traits<sharedptr_sharedptr_allocator> sharedptr_sharedptr_allocator_traits;


void utest(IvyGPUStream& stream){
  using namespace std_ivy;

  typedef double key_type;
  typedef dummy_D mapped_type;
  typedef std_util::pair<key_type, mapped_type> value_type;
  typedef std_ivy::hash<key_type> hasher;
  typedef hasher::result_type hash_result_type;
  typedef std_iter::IvyBucketedIteratorBuilder<key_type, value_type, std_ivy::hash<key_type>> iterator_builder_t;
  typedef std_mem::allocator<iterator_builder_t> allocator_iterator_builder_t;
  typedef std_mem::allocator_traits<allocator_iterator_builder_t> allocator_iterator_builder_traits_t;
  typedef std_util::pair<
    hash_result_type,
    std_mem::unique_ptr<value_type>
  > bucket_element_type;
  typedef std_mem::allocator<bucket_element_type> allocator_bucket_element_type;
  typedef std_mem::unique_ptr<bucket_element_type> data_type;
  typedef std_mem::allocator<data_type> allocator_data_type;

  __PRINT_INFO__("|*** Benchmarking IvyBucketedIteratorBuilder basic functionality... ***|\n");

  constexpr size_t ndata = 5;
  size_t const bucket_capacity = IvyHashEqualEvalDefault<hash_result_type>::bucket_size(ndata, ndata);
  size_t const preferred_data_capacity = IvyHashEqualEvalDefault<hash_result_type>::preferred_data_capacity(bucket_capacity);
  size_t const max_n_bucket_elements = (preferred_data_capacity+1)/bucket_capacity;
  std_mem::unique_ptr<mapped_type> raw_data = std_mem::make_unique<mapped_type>(ndata, preferred_data_capacity, IvyMemoryType::Host, &stream, 1.);
  std_mem::unique_ptr<bucket_element_type> ptr_buckets = std_mem::make_unique<bucket_element_type>(0, bucket_capacity, IvyMemoryType::Host, &stream);
  for (size_t i=0; i<ndata; i++){
    auto& raw_data_val = raw_data[i];
    raw_data_val.a += i;
    printf("raw_data[%llu].a = %f\n", i, raw_data_val.a);
    key_type key = raw_data_val.a + 100; // Add an arbitrary offset that is not confusing.
    hash_result_type hash = hasher()(key);
    bool is_found = false;
    for (size_t ib=0; ib<ptr_buckets.size(); ++ib){
      auto& bucket_element = ptr_buckets[ib];
      if (IvyHashEqualEvalDefault<hash_result_type>::eval(ndata, preferred_data_capacity, bucket_element.first, hash)){
        auto& data_bucket = bucket_element.second;
        for (size_t jd=0; jd<data_bucket.size(); ++jd){
          if (IvyKeyEqualEvalDefault<key_type>::eval(ndata, preferred_data_capacity, data_bucket[jd].first, key)){
            data_bucket[jd].second = raw_data_val;
            is_found = true;
            break;
          }
        }
        if (!is_found){
          data_bucket.emplace_back(key, raw_data_val);
          is_found = true;
          break;
        }
      }
      if (is_found) break;
    }
    if (!is_found) ptr_buckets.emplace_back(std_util::make_pair(hash, std_mem::make_unique<value_type>(1, max_n_bucket_elements, IvyMemoryType::Host, &stream, key, raw_data_val)));
  }

  __PRINT_INFO__("Data size = %llu, capacity = %llu\n", raw_data.size(), raw_data.capacity());
  __PRINT_INFO__("Bucket size = %llu, capacity = %llu\n", ptr_buckets.size(), ptr_buckets.capacity());
  for (size_t ib=0; ib<ptr_buckets.size(); ++ib){
    auto& bucket_element = ptr_buckets[ib];
    auto& data_bucket = bucket_element.second;
    printf("Bucket %llu: hash = %llu, data size = %llu, capacity = %llu\n", ib, bucket_element.first, data_bucket.size(), data_bucket.capacity());
    for (size_t jd=0; jd<data_bucket.size(); ++jd){
      auto& data_element = data_bucket[jd];
      printf("  Data %llu: key = %f, value.a = %f\n", jd, data_element.first, data_element.second.a);
    }
  }

  std_iter::IvyBucketedIteratorBuilder<key_type, mapped_type, hasher> it_builder;
  it_builder.reset(ptr_buckets.get(), ptr_buckets.size(), raw_data.capacity(), ptr_buckets.get_memory_type(), ptr_buckets.gpu_stream());
  {
    auto it_begin = it_builder.begin();
    auto it_end = it_builder.end();
    auto it = it_begin;
    while (it != it_end){
      printf("it->first = %f, it->second.a = %f\n", it->first, it->second.a);
      ++it;
    }
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
