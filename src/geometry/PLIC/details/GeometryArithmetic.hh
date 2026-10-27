/** @file GeometryArithmetic.hh
 * @brief Fixed finite arithmetic policies over shared geometry storage.
 */
#pragma once
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
namespace SimpleFluid::PLIC
{
namespace details SIMPLEFLUID_LOCAL
{
template<ArithmeticMode Mode> struct ArithmeticPolicy
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    static constexpr bool ordinary() { return Mode == ArithmeticMode::Ordinary; }
    static void operation() { details::operation(); }
    static void scalar_operation()
    {
        static_assert(Mode == ArithmeticMode::Ordinary);
        if (active_work)
        {
            if (active_work->operations >= active_work->operation_limit)
            {
                ++statistics.rejections;
                throw GeometryError(GeometryErrorCode::ResourceLimit,
                    "Ordinary FP geometry: scalar operation work limit exhausted.");
            }
            ++active_work->operations;
        }
    }
    static void interval_operation()
    {
        static_assert(Mode == ArithmeticMode::BoundedCertification);
        if (active_work)
        {
            ++statistics.interval_operations;
            if (++active_work->operations > active_work->operation_limit)
                details::reject(Mode, "interval scalar operation cap exhausted.");
        }
        else
            ++statistics.interval_operations;
    }
    static void finite(long double value)
    {
        if (!std::isfinite(value))
            details::reject(Mode, "nonfinite arithmetic result.");
        if constexpr (Mode == ArithmeticMode::BoundedCertification)
            if (value != 0 && !std::isnormal(value))
                details::reject(Mode, "arithmetic range unresolved (overflow or subnormal wider-FP result).");
    }
    [[noreturn]] static void reject(const char* reason) { details::reject(Mode, reason); }
};
} // namespace details SIMPLEFLUID_LOCAL
} // namespace SimpleFluid::PLIC
