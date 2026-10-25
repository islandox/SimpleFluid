/** @file GeometryInstrumentation.hh
 * @brief Explicit backend operation instrumentation.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/GeometryBackendTypes.hh"
#include <cstdint>
namespace SimpleFluid::PLIC
{
template<ArithmeticMode Mode> SIMPLEFLUID_LOCAL void enable_backend_statistics(bool);
template<ArithmeticMode Mode> SIMPLEFLUID_LOCAL void reset_backend_statistics();
template<ArithmeticMode Mode> [[nodiscard]] SIMPLEFLUID_LOCAL BackendStatistics<Mode> backend_statistics();
template<ArithmeticMode Mode> [[nodiscard]] SIMPLEFLUID_LOCAL const void* backend_statistics_domain();
} // namespace SimpleFluid::PLIC
#include "geometry/PLIC/details/exact/GeometryInstrumentation.hh"
