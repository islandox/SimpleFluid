/** @file NativeConstructionWireAccess.hh
 * @brief Allocation-free mode-neutral scalar preflight for native constructors.
 */
#pragma once
#include "utils/export_helper.hh"
#include <cstddef>
#include <span>
namespace SimpleFluid::PLIC::details
{
class SIMPLEFLUID_LOCAL NativeConstructionWireAccess
{
public:
    /** Uses the same Ordinary extrusion scalar reader and binary64 conversion
     * as full decode. On success advances cursor without constructing a packet,
     * vector, geometry, source identity or mutable geometry cache.
     */
    [[nodiscard]] static double ordinary_scalar(std::span<const char>, std::size_t& cursor);
};
} // namespace SimpleFluid::PLIC::details
