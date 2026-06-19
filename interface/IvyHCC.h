/**
 * @file IvyHCC.h
 * @brief Umbrella include for the full IvyHCC public API.
 *
 * Include order matters: IvyVector and IvyUnorderedMap must come before any
 * header that transitively includes <cstring> (e.g. IvyConstant → IvyCstring),
 * otherwise the bare `size_t` used internally in the unordered_map
 * implementation becomes ambiguous between ::size_t and IvyTypes::size_t.
 */
#ifndef IVYHCC_H
#define IVYHCC_H
#include "std_ivy/IvyVector.h"
#include "std_ivy/IvyUnorderedMap.h"
#include "autodiff/basic_nodes/IvyConstant.h"
#include "autodiff/basic_nodes/IvyVariable.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
#include "autodiff/basic_nodes/IvyTensor.h"
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"
#include "stream/IvyStream.h"
#endif
