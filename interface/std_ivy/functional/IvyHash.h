/**
 * @file IvyHash.h
 * @brief Hash functors and helpers implemented under std_ivy.
 */
#ifndef IVYHASH_H
#define IVYHASH_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyTypeTraits.h"
#include "IvyBasicTypes.h"


namespace ivy_hash_impl{
  /**
   * @brief Generic hash functor that folds object bytes into IvyTypes::size_t.
   * @tparam T Input type.
   */
  template<typename T> struct IvyHash{
    /** @brief Hash result type. */
    using result_type = IvyTypes::size_t;
    /** @brief Input argument type with cv-qualifiers removed. */
    using argument_type = std_ttraits::remove_cv_t<T>;

    /** @brief Compute hash value for an object. */
    __HOST_DEVICE__ constexpr result_type operator()(argument_type const& v) const{
      constexpr result_type nb_T = sizeof(argument_type);
      constexpr result_type size_partition = sizeof(result_type);
      constexpr result_type nbits_partition = size_partition*8;
      constexpr result_type nparts_full = nb_T / size_partition;
      constexpr result_type remainder = nb_T % size_partition;
      constexpr result_type part_full_shift_offset = (remainder==0 ? 0 : 1);

      result_type res = 0;
      result_type const* prc = __REINTERPRET_CAST__(result_type const*, &__CONST_CAST__(char&, __REINTERPRET_CAST__(char const volatile&, v)));
      if constexpr (nparts_full>0){
        for (result_type i=0; i<nparts_full; ++i){
          result_type const& pv = prc[i];
          res ^= (pv<<((i+part_full_shift_offset)%nbits_partition));
        }
      }
      if constexpr (remainder>0){
        result_type pv = 0;
        char const* prch = __REINTERPRET_CAST__(char const*, &__CONST_CAST__(char&, __REINTERPRET_CAST__(const volatile char&, prc[nparts_full])));
        for (result_type i=0; i<remainder; ++i) pv |= (__STATIC_CAST__(result_type, __STATIC_CAST__(unsigned char, prch[i]))<<(i*8));
        res ^= pv;
      }
      return res;
    }
  };
  /** @brief Hash specialization for C-string pointers with null-terminated traversal. */
  template<> struct IvyHash<char const*>{
    /** @brief Hash result type. */
    using result_type = IvyTypes::size_t;
    /** @brief Input argument type. */
    using argument_type = char const*;

    /** @brief Compute hash value for a null-terminated const char string. */
    __HOST_DEVICE__ constexpr result_type operator()(argument_type const& v) const{
      constexpr result_type size_partition = sizeof(result_type);
      constexpr result_type nbits_partition = size_partition*8;
      constexpr result_type nbits_arg_el = sizeof(char)*8;
      result_type res = 0;
      {
        argument_type vv = v;
        for (result_type i=0; *vv; ++i){ result_type const shift = ((i%size_partition)*nbits_arg_el)%nbits_partition; res ^= (__STATIC_CAST__(result_type, __STATIC_CAST__(unsigned char, *vv))<<shift); ++vv; }
      }
      return res;
    }
  };
  /** @brief Hash specialization for mutable C-string pointers with null-terminated traversal. */
  template<> struct IvyHash<char*>{
    /** @brief Hash result type. */
    using result_type = IvyTypes::size_t;
    /** @brief Input argument type. */
    using argument_type = char*;

    /** @brief Compute hash value for a null-terminated mutable char string. */
    __HOST_DEVICE__ constexpr result_type operator()(argument_type const& v) const{
      constexpr result_type size_partition = sizeof(result_type);
      constexpr result_type nbits_partition = size_partition*8;
      constexpr result_type nbits_arg_el = sizeof(char)*8;
      result_type res = 0;
      {
        argument_type vv = v;
        for (result_type i=0; *vv; ++i){ result_type const shift = ((i%size_partition)*nbits_arg_el)%nbits_partition; res ^= (__STATIC_CAST__(result_type, __STATIC_CAST__(unsigned char, *vv))<<shift); ++vv; }
      }
      return res;
    }
  };
}
namespace std_ivy{
  /** @brief Public hash alias used throughout std_ivy containers. */
  template<typename T> using hash = ivy_hash_impl::IvyHash<T>;
}


#endif
