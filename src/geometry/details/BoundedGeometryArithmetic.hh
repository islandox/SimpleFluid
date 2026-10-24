/** @file BoundedGeometryArithmetic.hh
 * @brief Fixed finite arithmetic policies over shared geometry storage.
 */
#pragma once
#include "geometry/details/BoundedConvexGeometryInternal.hh"
namespace SimpleFluid::Meshes
{
namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> struct ArithmeticPolicy
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    static constexpr bool ordinary() { return Mode == ArithmeticMode::Ordinary; }
    static void operation() { bounded_geometry_detail::operation(); }
    static void interval_operation()
    {
        if (active_work)
        {
            if constexpr (Mode == ArithmeticMode::BoundedCertification)
                ++statistics.interval_operations;
            if (++active_work->operations > active_work->operation_limit)
                bounded_geometry_detail::reject("interval scalar operation cap exhausted.");
        }
        else
            ++statistics.interval_operations;
    }
    static void finite(long double value)
    {
        if (!std::isfinite(value))
            bounded_geometry_detail::reject("arithmetic range unresolved (overflow or subnormal wider-FP result).");
        if constexpr (Mode == ArithmeticMode::BoundedCertification)
            if (value != 0 && !std::isnormal(value))
                bounded_geometry_detail::reject("arithmetic range unresolved (overflow or subnormal wider-FP result).");
    }
    [[noreturn]] static void reject(const char* reason) { bounded_geometry_detail::reject(reason); }
};
} // namespace bounded_geometry_detail SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::Meshes
