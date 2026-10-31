/** @file CanonicalScalarEncoding.hh
 * @brief Private sink-based emission of finite canonical scalar words.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include <cstdint>

namespace SimpleFluid::PLIC::details
{
/** The caller supplies a finite value and retains its word storage policy. */
template<class WordSink>
SIMPLEFLUID_LOCAL void emit_canonical_scalar(long double value, WordSink& sink) noexcept(
    noexcept(sink.word(std::uint64_t{})));
} // namespace SimpleFluid::PLIC::details

#include "geometry/PLIC/details/CanonicalScalarEncoding.ipp"
