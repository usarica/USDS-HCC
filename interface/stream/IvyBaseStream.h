#ifndef IVYBASESTREAM_HH
#define IVYBASESTREAM_HH

/**
 * @file IvyBaseStream.h
 * @brief Base abstraction for backend stream wrappers.
 */


/**
  IvyBaseStream:
  This class is a base class for any stream, requiring certain operations to be present.
*/


#include "config/IvyCompilerConfig.h"
#include "stream/IvyBaseStreamEvent.h"


namespace IvyStreamUtils{
  /** @brief Construct a backend raw stream object. */
  template<typename RawStream_t> __HOST_DEVICE__ void buildRawStream(RawStream_t& st, unsigned int flags, unsigned int priority);
  /** @brief Destroy a backend raw stream object. */
  template<typename RawStream_t> __HOST_DEVICE__ void destroyRawStream(RawStream_t& st);

  /** @brief Create a typed stream object from typed stream flags. */
  template<typename Stream_t> __HOST__ void make_stream(Stream_t*& stream, typename Stream_t::StreamFlags flags, unsigned int priority=0);
  /** @brief Create a typed stream object from raw backend flags. */
  template<typename Stream_t> __HOST__ void make_stream(Stream_t*& stream, unsigned int flags, unsigned int priority=0);
  /** @brief Wrap an existing raw stream handle in a typed stream object. */
  template<typename Stream_t> __HOST_DEVICE__ void make_stream(Stream_t*& stream, typename Stream_t::RawStream_t st, bool is_owned);
  /** @brief Destroy a previously created typed stream object. */
  template<typename Stream_t> __HOST_DEVICE__ void destroy_stream(Stream_t*& stream);

  // Calls with different arguments
  template<typename Stream_t> __HOST__ Stream_t* make_stream(unsigned int flags, unsigned int priority=0){ Stream_t* res = nullptr; make_stream(res, flags, priority); return res; }
  template<typename Stream_t> __HOST__ Stream_t* make_stream(typename Stream_t::StreamFlags flags, unsigned int priority=0){ Stream_t* res = nullptr; make_stream(res, flags, priority); return res; }
  template<typename Stream_t> __HOST_DEVICE__ Stream_t* make_stream(typename Stream_t::RawStream_t st, bool is_owned){ Stream_t* res = nullptr; make_stream(res, st, is_owned); return res; }
}

/**
 * @brief Polymorphic base class for stream wrappers.
 * @tparam S Backend raw stream type.
 */
template<typename S> class IvyBaseStream{
public:
  /** @brief Backend raw stream type. */
  using RawStream_t = S;
  /** @brief Backend raw event type associated with RawStream_t. */
  using RawEvent_t = IvyStreamUtils::StreamEvent_t<RawStream_t>;

protected:
  bool is_owned_ = false;
  unsigned int flags_ = 0;
  int priority_ = 0;
  RawStream_t stream_{};

public:
  /** @brief Default constructor. */
  IvyBaseStream() = default;
  /** @brief Construct from explicit ownership, flags, priority, and raw stream handle. */
  __HOST_DEVICE__ IvyBaseStream(bool const& is_owned, unsigned int const& flags, int const& priority, RawStream_t const& st) :
    is_owned_(is_owned), flags_(flags), priority_(priority), stream_(st)
  {}
  /** @brief Copy constructor is disabled to avoid ambiguous ownership semantics. */
  __HOST_DEVICE__ IvyBaseStream(IvyBaseStream const&) = delete;
  /** @brief Move-copy constructor syntax variant is disabled. */
  __HOST_DEVICE__ IvyBaseStream(IvyBaseStream const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyBaseStream(){ if (is_owned_) IvyStreamUtils::destroyRawStream(this->stream_); }

  __HOST_DEVICE__ bool const& is_owned() const{ return this->is_owned_; }
  __HOST_DEVICE__ unsigned int const& flags() const{ return this->flags_; }
  __HOST_DEVICE__ int const& priority() const{ return this->priority_; }
  __HOST_DEVICE__ RawStream_t const& stream() const{ return this->stream_; }
  __HOST_DEVICE__ operator RawStream_t const& () const{ return this->stream_; }

  __HOST_DEVICE__ bool& is_owned(){ return this->is_owned_; }
  __HOST_DEVICE__ unsigned int& flags(){ return this->flags_; }
  __HOST_DEVICE__ int& priority(){ return this->priority_; }
  __HOST_DEVICE__ RawStream_t& stream(){ return this->stream_; }
  __HOST_DEVICE__ operator RawStream_t& (){ return this->stream_; }

  /** @brief Synchronize host execution with this stream. */
  virtual __HOST__ void synchronize() = 0;
  /** @brief Wait on an event before continuing stream execution. */
  virtual __HOST__ void wait(RawEvent_t& event, unsigned int wait_flags) = 0;

  __HOST_DEVICE__ void swap(IvyBaseStream& other){
    std_util::swap(this->is_owned_, other.is_owned_);
    std_util::swap(this->flags_, other.flags_);
    std_util::swap(this->priority_, other.priority_);
    std_util::swap(this->stream_, other.stream_);
  }
};


#endif
