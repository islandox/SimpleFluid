/** Private first-proof validation using the original Ordinary box constructor. */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/details/NativeConstructionAdmission.hh"

namespace SimpleFluid::PLIC::details
{
class SIMPLEFLUID_LOCAL NativeBoxSourceAccess final
{
public:
    /** Complete requested-C++ upper bound for the fixed original constructor,
     * including controls, numerical ledger and temporary container growth.
     * This is a library scratch profile, separate from actual native extents.
     */
    [[nodiscard]] static std::size_t scratch_bound();
    /** Callback and its temporary faces must not let a box child escape. Only
     * scalar validation facts may survive. All numerical owners die before
     * the profile retires, including the outer numerical ledger's pins.
     */
    static void verify(GeometryPoint lower, GeometryPoint upper, const geometry_detail::NativeConstructionAdmission&,
        void*, void (*)(void*, const ConvexCellGeometry<ArithmeticMode::Ordinary>&));
};
} // namespace SimpleFluid::PLIC::details
