/**
 * @file IvyUnorderedMapImpl.h
 * @brief Method definitions for the std_ivy unordered_map implementation.
 */
#ifndef IVYUNORDEREDMAPIMPL_H
#define IVYUNORDEREDMAPIMPL_H


#include "std_ivy/unordered_map/IvyUnorderedMapImpl.hh"


#define __UMAPTPLARGSINIT__ <typename Key, typename T, typename Hash, typename KeyEqual, typename HashEqual, typename Allocator>
#define __UMAPTPLARGS__ <Key, T, Hash, KeyEqual, HashEqual, Allocator>


namespace std_ivy{
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::IvyUnorderedMap(){}
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::IvyUnorderedMap(IvyUnorderedMap const& v){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = v._data.get_gpu_stream();
    data_container* ptr_data = std_mem::addressof(_data);
    data_container* ptr_v_data = __CONST_CAST__(data_container*, std_mem::addressof(v._data));
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        allocator_data_container_traits::transfer(ptr_data, ptr_v_data, 1, def_mem_type, def_mem_type, ref_stream);
      )
    );
    this->reset_iterator_builder();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::IvyUnorderedMap(IvyUnorderedMap&& v) :
    _data(std_util::move(v._data)),
    _iterator_builder(std_util::move(v._iterator_builder))
  {}
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::~IvyUnorderedMap(){
    this->destroy_iterator_builder();
    _data.reset();
  }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__& IvyUnorderedMap __UMAPTPLARGS__::operator=(IvyUnorderedMap __UMAPTPLARGS__ const& v){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = v._data.get_gpu_stream();
    _data.reset();
    data_container* ptr_data = std_mem::addressof(_data);
    data_container* ptr_v_data = __CONST_CAST__(data_container*, std_mem::addressof(v._data));
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        allocator_data_container_traits::transfer(ptr_data, ptr_v_data, 1, def_mem_type, def_mem_type, ref_stream);
      )
    );
    this->reset_iterator_builder();
    return *this;
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__& IvyUnorderedMap __UMAPTPLARGS__::operator=(IvyUnorderedMap __UMAPTPLARGS__&& v){
    _data = std_util::move(v._data);
    _iterator_builder = std_util::move(v._iterator_builder);
    return *this;
  }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::data_container const& IvyUnorderedMap __UMAPTPLARGS__::get_data_container() const{ return _data; }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyMemoryType IvyUnorderedMap __UMAPTPLARGS__::get_memory_type() const{ return _data.get_memory_type(); }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyGPUStream* IvyUnorderedMap __UMAPTPLARGS__::gpu_stream() const{ return _data.gpu_stream(); }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::destroy_iterator_builder(){
    _iterator_builder.reset();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::reset_iterator_builder(){
    auto const mem_type = _data.get_memory_type();
    auto const& stream = _data.gpu_stream();
    auto const s = this->bucket_count();
    auto const c = hash_equal::preferred_data_capacity(this->bucket_capacity());
    auto ptr = _data.get();
    _iterator_builder.reset(ptr, s, c, mem_type, stream);
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ bool IvyUnorderedMap __UMAPTPLARGS__::transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old){
    bool res = true;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = _data.gpu_stream();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        res &= allocator_data_container::transfer_internal_memory(&_data, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        res &= allocator_iterator_builder_t::transfer_internal_memory(&_iterator_builder, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        this->reset_iterator_builder();
      )
    );
    return res;
  }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::reference IvyUnorderedMap __UMAPTPLARGS__::front(){
    return *(_iterator_builder.front());
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reference IvyUnorderedMap __UMAPTPLARGS__::front() const{
    return *(_iterator_builder.cfront());
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::reference IvyUnorderedMap __UMAPTPLARGS__::back(){
    return *(_iterator_builder.back());
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reference IvyUnorderedMap __UMAPTPLARGS__::back() const{
    return *(_iterator_builder.cback());
  }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::begin(){
    return _iterator_builder.begin();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::end(){
    return _iterator_builder.end();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::rbegin(){
    return _iterator_builder.rbegin();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::rend(){
    return _iterator_builder.rend();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::begin() const{
    return _iterator_builder.cbegin();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::cbegin() const{
    return _iterator_builder.cbegin();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::end() const{
    return _iterator_builder.cend();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::cend() const{
    return _iterator_builder.cend();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::rbegin() const{
    return _iterator_builder.crbegin();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::crbegin() const{
    return _iterator_builder.crbegin();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::rend() const{
    return _iterator_builder.crend();
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::crend() const{
    return _iterator_builder.crend();
  }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ bool IvyUnorderedMap __UMAPTPLARGS__::empty() const{ return this->size()==0; }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::size() const{ return _iterator_builder.n_valid_iterators(); }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ constexpr IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::max_size() const{ return std_limits::numeric_limits<size_type>::max(); }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::capacity() const{ return _iterator_builder.n_capacity_valid_iterators(); }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::clear(){
    _data.reset();
    this->reset_iterator_builder();
  }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::get_predicted_bucket_count() const{
    return KeyEqual::bucket_size(this->size(), this->capacity());
  }

  template __UMAPTPLARGSINIT__
    __HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::swap(IvyUnorderedMap __UMAPTPLARGS__& v){
    std_util::swap(_data, v._data);
    std_util::swap(_iterator_builder, v._iterator_builder);
  }

  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::bucket_count() const{
    return _data.size();
  }
  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::bucket_capacity() const{
    return _data.capacity();
  }
  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ constexpr IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::max_bucket_count() const{
    return std_limits::numeric_limits<size_type>::max();
  }

  // find functions
  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::find_iterator(Key const& key) const{
    // Implemented in terms of _iterator_builder.begin/end because this is a const function, and const overloads of begin/end return const_iterator, not iterator.
    for (auto it = _iterator_builder.begin(); it!=_iterator_builder.end(); ++it){ if (it->first==key) return it; }
    return _iterator_builder.end();
  }
  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::find_const_iterator(Key const& key) const{
    for (auto it = this->cbegin(); it!=this->cend(); ++it){ if (it->first==key) return it; }
    return this->cend();
  }
  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::find(Key const& key){
    return this->find_iterator(key);
  }
  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::find(Key const& key) const{
    return this->find_const_iterator(key);
  }

  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::calculate_data_size_capacity(IvyUnorderedMap __UMAPTPLARGS__::size_type& n_size, IvyUnorderedMap __UMAPTPLARGS__::size_type& n_capacity) const{
    n_size = n_capacity = 0;
    if (!_data) return;

    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    IvyMemoryType const mem_type = _data.get_memory_type();
    IvyGPUStream* stream = _data.gpu_stream();
    size_type const current_n_size_buckets = this->bucket_count();

    memview<bucket_element, allocator_bucket_element> data_view(_data.get(), current_n_size_buckets, mem_type, stream, false);
    bucket_element* tr_data_ptr = data_view;
    for (size_type ib=0; ib<current_n_size_buckets; ++ib){
      auto& data_bucket = tr_data_ptr->second;
      n_size += data_bucket.size();
      n_capacity += data_bucket.capacity();
      ++tr_data_ptr;
    }
  }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__
  IvyUnorderedMap __UMAPTPLARGS__::data_container IvyUnorderedMap __UMAPTPLARGS__::get_rehashed_data(IvyUnorderedMap __UMAPTPLARGS__::size_type new_n_buckets) const{
    size_type const current_n_size_buckets = this->bucket_count();

    size_type const preferred_data_capacity = hash_equal::preferred_data_capacity(new_n_buckets);
    size_type const max_n_bucket_elements = (preferred_data_capacity+1)/new_n_buckets;

    allocator_type a;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    IvyMemoryType const mem_type = _data.get_memory_type();
    IvyGPUStream* stream = _data.gpu_stream();
    size_type const n_size = this->size();
    size_type const n_capacity = this->capacity();

    // Make a new data container to which we can copy the data with size = 0 and capacity = new_n_buckets.
    data_container new_data = std_mem::make_unique<bucket_element>(
      0, new_n_buckets,
      mem_type, stream
    );

    {
      bool const is_same_mem_type = (mem_type==def_mem_type);
      build_GPU_stream_reference_from_pointer(stream, ref_stream);

      memview<bucket_element, allocator_bucket_element> data_view(_data.get(), current_n_size_buckets, mem_type, &ref_stream, false);
      {
        bucket_element* data_ptr = data_view;
        for (size_type ib=0; ib<current_n_size_buckets; ++ib){
          auto& data_bucket = data_ptr->second;
          size_type const n_size_data_bucket = data_bucket.size();
          memview<value_type, allocator_type> data_bucket_view(data_bucket.get(), n_size_data_bucket, mem_type, &ref_stream, false);
          {
            value_type* data_bucket_ptr = data_bucket_view;
            for (size_type jd=0; jd<n_size_data_bucket; ++jd){
              auto const& data_value = *data_bucket_ptr;
              auto const& key = data_value.first;
              auto const& value = data_value.second;
              auto const& hash = hasher()(key);

              bool is_inserted = false;
              size_type const new_current_n_size_buckets = new_data.size();
              auto& d_new_data_ptr = new_data.get();
              memview<bucket_element, allocator_bucket_element> new_data_view(d_new_data_ptr, new_current_n_size_buckets, mem_type, &ref_stream, false);
              {
                bucket_element* new_data_ptr = new_data_view;
                for (size_type jb=0; jb<new_current_n_size_buckets; ++jb){
                  auto& new_bucket_element = *new_data_ptr;
                  auto const& new_bucket_hash = new_bucket_element.first;
                  if (hash_equal::eval(n_size, n_capacity, hash, new_bucket_hash)){
                    auto& new_data_bucket = new_bucket_element.second;
                    if (new_data_bucket.capacity()==0) new_data_bucket.reserve(max_n_bucket_elements, mem_type, stream);
                    new_data_bucket.emplace_back(key, value);
                    if (!is_same_mem_type){
                      auto tmp_tgt = d_new_data_ptr + jb;
                      IvyMemoryHelpers::transfer_memory(tmp_tgt, new_data_ptr, 1, mem_type, def_mem_type, ref_stream);
                    }
                    is_inserted = true;
                    break;
                  }
                  ++new_data_ptr;
                }
                if (!is_inserted){
                  // If we stil have not found the bucket, we need to create a new one.
                  new_data.emplace_back(
                    hash,
                    std_mem::build_unique<value_type>(
                      a, 1, max_n_bucket_elements,
                      mem_type, stream, data_value
                    )
                  );
                }
              }
              ++data_bucket_ptr;
            }
          }
          ++data_ptr;
        }
      }

      destroy_GPU_stream_reference_from_pointer(stream);
    }

    return new_data;
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__
  void IvyUnorderedMap __UMAPTPLARGS__::rehash(IvyUnorderedMap __UMAPTPLARGS__::size_type new_n_buckets){
    if (!_data) return;
    size_type const current_n_capacity_buckets = this->bucket_capacity();
    if (new_n_buckets<=current_n_capacity_buckets) return;
    _data = this->get_rehashed_data(new_n_buckets);
    this->reset_iterator_builder();
  }

  // insert functions
  template __UMAPTPLARGSINIT__ template<typename... Args>
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::emplace(
    IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args
  ){ return this->insert(mem_type, stream, key, args...); }
  template __UMAPTPLARGSINIT__ template<typename... Args>
  __HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::insert_impl(
    IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args
  ){
    size_type const current_size = this->size();
    size_type const current_capacity = this->capacity();
    size_type const current_bucket_size = this->bucket_count();
    size_type const current_bucket_capacity = this->bucket_capacity();
    size_type const preferred_current_data_capacity = hash_equal::preferred_data_capacity(current_bucket_capacity);
    size_type const current_max_n_bucket_elements = (current_bucket_capacity==0 ? 0 : (preferred_current_data_capacity+1)/current_bucket_capacity);

    size_type const new_size = current_size + 1;
    size_type const new_capacity = (new_size>current_capacity ? new_size + 1 : current_capacity);
    size_type const new_bucket_capacity = hash_equal::bucket_size(new_size, new_capacity);
    size_type const preferred_new_data_capacity = hash_equal::preferred_data_capacity(new_bucket_capacity);
    size_type const new_max_n_bucket_elements = (preferred_new_data_capacity+1)/new_bucket_capacity;

    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    hash_result_type const hash = hasher()(key);
    allocator_type a;
    bool trigger_it_reset = false;

    if (!_data || _data.get_memory_type()!=mem_type){
      _data = std_mem::make_unique<bucket_element>(
        1, new_bucket_capacity,
        mem_type, stream,
        std_util::pair<hash_result_type, bucket_data_type>(
          hash,
          std_mem::build_unique<value_type>(
            a, 1, new_max_n_bucket_elements,
            mem_type, stream,
            value_type(key, mapped_type(std_util::forward<Args>(args)...))
          )
        )
      );
      this->reset_iterator_builder();
    }
    else{
      bool is_found = false;
      value_type* mem_loc_pos = nullptr;
      {
        bool const is_same_mem_type = (mem_type==def_mem_type);
        build_GPU_stream_reference_from_pointer(stream, ref_stream);

        auto& data_ptr = _data.get();
        memview<bucket_element, allocator_bucket_element> data_view(data_ptr, current_bucket_size, mem_type, &ref_stream, false);
        bucket_element* tr_h_data_ptr = data_view;
        for (size_type ib=0; ib<current_bucket_size; ++ib){
          auto& current_bucket_element = *tr_h_data_ptr;
          auto const& bucket_hash = current_bucket_element.first;
          auto& data_bucket = current_bucket_element.second;
          if (hash_equal::eval(current_size, current_capacity, hash, bucket_hash)){
            auto& bucket_data_ptr = data_bucket.get();
            size_type const n_size_data_bucket = data_bucket.size();
            memview<value_type, allocator_type> bucket_data_view(bucket_data_ptr, n_size_data_bucket, mem_type, &ref_stream, false);
            {
              value_type* tr_h_bucket_data_ptr = bucket_data_view;
              for (size_t jd=0; jd<n_size_data_bucket; ++jd){
                value_type& data_value = *tr_h_bucket_data_ptr;
                if (key_equal::eval(current_size, current_capacity, key, data_value.first)){
                  data_value.second = mapped_type(std_util::forward<Args>(args)...);
                  if (!is_same_mem_type){
                    auto tmp_tgt = bucket_data_ptr + jd;
                    IvyMemoryHelpers::transfer_memory(tmp_tgt, tr_h_bucket_data_ptr, 1, mem_type, def_mem_type, ref_stream);
                  }
                  is_found = true;
                  break;
                }
                ++tr_h_bucket_data_ptr;
              }
            }
            if (!is_found){
              trigger_it_reset = (data_bucket.size()==data_bucket.capacity());
              data_bucket.emplace_back(key, mapped_type(std_util::forward<Args>(args)...));
              mem_loc_pos = data_bucket.get() + data_bucket.size() - 1;
              is_found = true;
            }
          }
          if (is_found){
            if (mem_type!=def_mem_type){
              auto tmp_tgt = data_ptr + ib;
              IvyMemoryHelpers::transfer_memory(tmp_tgt, tr_h_data_ptr, 1, mem_type, def_mem_type, ref_stream);
            }
            break;
          }
          ++tr_h_data_ptr;
        }
        if (!is_found){
          // If we stil have not found the bucket, we need to create a new one.
          _data.emplace_back(
            hash,
            std_mem::build_unique<value_type>(
              a, 1, current_max_n_bucket_elements,
              mem_type, stream, value_type(key, mapped_type(std_util::forward<Args>(args)...))
            )
          );
          memview<bucket_element, allocator_bucket_element> data_view_last(_data.get() + current_bucket_size, mem_type, &ref_stream, false);
          auto& data_bucket = (*data_view_last).second;
          mem_loc_pos = data_bucket.get();
        }

        destroy_GPU_stream_reference_from_pointer(stream);
      }

      // We need to update the iterators if a new element is inserted.
      // We do this even if we rehash later, which will invalidate the iterators.
      // The reason is to make sure size() and capacity() are updated, which are need to be up-to-date in rehashing.
      // Notice that the correct test for genuine new insertion is to check if mem_loc_pos is not a nullptr
      // because it is also possible to find the inserted key to exist and just update the corresponding value.
      if (mem_loc_pos){
        // If the new size is greater than the current capacity, we need to reset the iterator builder;
        // the push_back function does not invalidate existing iterators if the existing capacity of the iterator builder is exceeded.
        // Otherwise, we just push the new element to its back.
        if (new_size>current_capacity || trigger_it_reset) this->reset_iterator_builder();
        else _iterator_builder.push_back(mem_loc_pos, mem_type, stream);
      }
    }
  }
  template __UMAPTPLARGSINIT__ template<typename... Args>
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::insert(
    IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args
  ){
    this->insert_impl(mem_type, stream, key, args...);

    // Rehash if needed
    this->rehash(hash_equal::bucket_size(this->size(), this->capacity()));

    return find_iterator(key);
  }
  template __UMAPTPLARGSINIT__ template<typename InputIterator>
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::insert(
    InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream
  ){
    InputIterator it = first;
    while (it!=last){
      auto const& v = *it;
      this->insert_impl(mem_type, stream, v.first, v.second);
      ++it;
    }

    // Rehash if needed
    this->rehash(hash_equal::bucket_size(this->size(), this->capacity()));

    return find_iterator(first->first);
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::insert(
    std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream
  ){
    if (!_data) return iterator();
    if (ilist.size()==0) return this->end();
    for (auto const& v:ilist) this->insert_impl(mem_type, stream, v.first, v.second);

    // Rehash if needed
    this->rehash(hash_equal::bucket_size(this->size(), this->capacity()));

    return find_iterator(ilist.begin()->first);
  }

  // erase functions
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::erase_impl(Key const& key, IvyUnorderedMap __UMAPTPLARGS__::size_type& n_erased){
    if (!_data) return iterator();
    IvyUnorderedMap __UMAPTPLARGS__::iterator res = this->end();
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    IvyMemoryType const mem_type = _data.get_memory_type();
    IvyGPUStream* stream = _data.gpu_stream();

    // Note that current_size and current_capacity come from the size and capacity of the iterator builder.
    // Since erase_impl does not update iterators, the actual values of the size and capacity of the _data container may be different.
    // However, since erase_impl also does not launch rehashing, hash_equal::eval and key_equal::eval NEED current_size and current_capacity
    // to correspond to the _data shape after the last rehashing operation.
    // For that reason, we are using current_size and current_capacity correctly here.
    size_type const current_size = this->size();
    size_type const current_capacity = this->capacity();
    size_type const current_bucket_size = this->bucket_count();

    hash_result_type const hash = hasher()(key);
    allocator_type a;
    {
      bool const is_same_mem_type = (mem_type==def_mem_type);
      build_GPU_stream_reference_from_pointer(stream, ref_stream);

      bucket_element*& data_ptr = _data.get();
      memview<bucket_element, allocator_bucket_element> data_view(data_ptr, current_bucket_size, mem_type, &ref_stream, false);
      {
        bucket_element* tr_h_data_ptr = data_view.get() + current_bucket_size - 1;
        for (size_type rib=0; rib<current_bucket_size; ++rib){
          size_type const ib = current_bucket_size-1-rib;
          auto& current_bucket_element = *tr_h_data_ptr;
          auto const& bucket_hash = current_bucket_element.first;
          auto& data_bucket = current_bucket_element.second;
          size_type const n_size_data_bucket = data_bucket.size();
          if (n_size_data_bucket>0 && hash_equal::eval(current_size, current_capacity, hash, bucket_hash)){
            bool data_bucket_modified = false;

            value_type*& bucket_data_ptr = data_bucket.get();
            memview<value_type, allocator_type> bucket_data_view(bucket_data_ptr, n_size_data_bucket, mem_type, &ref_stream, false);
            {
              value_type* tr_h_bucket_data_ptr = bucket_data_view.get() + n_size_data_bucket - 1;
              for (size_t rjd=0; rjd<n_size_data_bucket; ++rjd){
                size_t const jd = n_size_data_bucket-1-rjd;
                if (key_equal::eval(current_size, current_capacity, key, tr_h_bucket_data_ptr->first)){
                  data_bucket.erase(jd);
                  data_bucket_modified = true;
                  ++n_erased;
                }
                --tr_h_bucket_data_ptr;
              }
            }
            if (data_bucket_modified && !is_same_mem_type){
              auto tmp_tgt = data_ptr + ib;
              IvyMemoryHelpers::transfer_memory(tmp_tgt, tr_h_data_ptr, 1, mem_type, def_mem_type, ref_stream);
            }
          }
          if (data_bucket.size()==0) _data.erase(ib);
          --tr_h_data_ptr;
        }
      }

      destroy_GPU_stream_reference_from_pointer(stream);
    }
  }
  template __UMAPTPLARGSINIT__
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::erase(Key const& key){
    size_type n_erased = 0;
    this->erase_impl(key, n_erased);

    // Because data is not fully contiguous, iterators cannot be kept after an erase operation over _iterator_builder.
    // The surest way is to make a clean set of them.
    this->reset_iterator_builder();
    // Rehash if needed
    this->rehash(hash_equal::bucket_size(this->size(), this->capacity()));

    return n_erased;
  }
  template __UMAPTPLARGSINIT__ template<typename PosIterator>
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::erase(PosIterator pos){
    size_type n_erased = 0;
    if (!_data || !pos.is_valid()) return n_erased;
    key_type const& key = pos->first;
    this->erase_impl(key, n_erased);

    // Because data is not fully contiguous, iterators cannot be kept after an erase operation over _iterator_builder.
    // The surest way is to make a clean set of them.
    this->reset_iterator_builder();
    // Rehash if needed
    this->rehash(hash_equal::bucket_size(this->size(), this->capacity()));

    return n_erased;
  }
  template __UMAPTPLARGSINIT__ template<typename PosIterator>
  __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::erase(PosIterator first, PosIterator last){
    size_type n_erased = 0;
    if (!_data) return n_erased;
    while (first!=last){
      n_erased += this->erase(first);
      ++first;
    }

    // Because data is not fully contiguous, iterators cannot be kept after an erase operation over _iterator_builder.
    // The surest way is to make a clean set of them.
    this->reset_iterator_builder();
    // Rehash if needed
    this->rehash(hash_equal::bucket_size(this->size(), this->capacity()));

    return n_erased;
  }

  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::mapped_type const& IvyUnorderedMap __UMAPTPLARGS__::operator[](Key const& key) const{
    if (!_data){
      __PRINT_ERROR__("IvyUnorderedMap::operator[] cannot be called on an empty map.");
      assert(false);
    }
    if (_data.get_memory_type()!=IvyMemoryHelpers::get_execution_default_memory()){
      __PRINT_ERROR__("IvyUnorderedMap::operator[] cannot be called for data that resides in another device.");
      assert(false);
    }

    size_type const current_size = this->size();
    size_type const current_capacity = this->capacity();
    size_type const current_bucket_size = this->bucket_count();

    auto const hash = hasher()(key);
    auto data_ptr = _data.get();
    for (size_type ib=0; ib<current_bucket_size; ++ib){
      auto& current_bucket_element = *data_ptr;
      auto const& bucket_hash = current_bucket_element.first;
      auto& data_bucket = current_bucket_element.second;
      if (hash_equal::eval(current_size, current_capacity, hash, bucket_hash)){
        size_t n_size_data_bucket = data_bucket.size();
        auto data_bucket_ptr = data_bucket.get();
        for (size_t jd=0; jd<n_size_data_bucket; ++jd){
          if (key_equal::eval(current_size, current_capacity, key, data_bucket_ptr->first)) return data_bucket_ptr->second;
          ++data_bucket_ptr;
        }
        ++data_ptr;
      }
    }

    __PRINT_ERROR__("IvyUnorderedMap::operator[] cannot find the key.\n");
    assert(false);
    return this->begin()->second;
  }
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::mapped_type& IvyUnorderedMap __UMAPTPLARGS__::operator[](Key const& key){
    if (!_data){
      __PRINT_ERROR__("IvyUnorderedMap::operator[] cannot be called on an empty map.");
      assert(false);
    }
    if (_data.get_memory_type()!=IvyMemoryHelpers::get_execution_default_memory()){
      __PRINT_ERROR__("IvyUnorderedMap::operator[] cannot be called for data that resides in another device.");
      assert(false);
    }

    size_type const current_size = this->size();
    size_type const current_capacity = this->capacity();
    size_type const current_bucket_size = this->bucket_count();

    auto const hash = hasher()(key);
    auto data_ptr = _data.get();
    for (size_type ib=0; ib<current_bucket_size; ++ib){
      auto& current_bucket_element = *data_ptr;
      auto const& bucket_hash = current_bucket_element.first;
      auto& data_bucket = current_bucket_element.second;
      if (hash_equal::eval(current_size, current_capacity, hash, bucket_hash)){
        size_t n_size_data_bucket = data_bucket.size();
        auto data_bucket_ptr = data_bucket.get();
        for (size_t jd=0; jd<n_size_data_bucket; ++jd){
          if (key_equal::eval(current_size, current_capacity, key, data_bucket_ptr->first)) return data_bucket_ptr->second;
          ++data_bucket_ptr;
        }
        ++data_ptr;
      }
    }

    __PRINT_ERROR__("IvyUnorderedMap::operator[] cannot find the key.\n");
    assert(false);
    return this->begin()->second;
  }
  template __UMAPTPLARGSINIT__ template<typename... Args> __HOST_DEVICE__
  IvyUnorderedMap __UMAPTPLARGS__::mapped_type& IvyUnorderedMap __UMAPTPLARGS__::operator()(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args){
    this->emplace(mem_type, stream, key, args...);
    return this->operator[](key);
  }
  template __UMAPTPLARGSINIT__ template<typename... Args> __HOST_DEVICE__
  IvyUnorderedMap __UMAPTPLARGS__::mapped_type& IvyUnorderedMap __UMAPTPLARGS__::operator()(Key const& key, Args&&... args){
    return this->operator()(_data.get_memory_type(), _data.gpu_stream(), key, args...);
  }

  template __UMAPTPLARGSINIT__  struct value_printout<IvyUnorderedMap __UMAPTPLARGS__>{
    static __HOST_DEVICE__ void print(IvyUnorderedMap __UMAPTPLARGS__ const& x){ print_value(x.get_data_container(), false); }
  };

}
namespace std_util{
  template __UMAPTPLARGSINIT__ __HOST_DEVICE__ void swap(std_ivy::IvyUnorderedMap __UMAPTPLARGS__& a, std_ivy::IvyUnorderedMap __UMAPTPLARGS__& b){
    a.swap(b);
  }
}


#undef __UMAPTPLARGS__
#undef __UMAPTPLARGSINIT__


#endif
