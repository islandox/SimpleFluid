/**
 * @file CertifiedGeometryProperties.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief Mode-typed published geometry measures and their conversion contracts.
 * @version 0.1
 * @date 2026-10-25
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/typedefs.hh"
#include "utils/ArithmeticMode.hh"

namespace SimpleFluid::PLIC
{

using NativeScalar = long double;

using GeometryPoint = Vec3D<double>;

/** Published geometry measure in physical units. This is an output view, not
 * the exact backend's retained rational scalar. Ordinary carries a nominal
 * value only; bounded and exact views retain their conversion/error enclosure.
 * Publication rejects overflow and disappearance of a nonzero measure. */
template<ArithmeticMode Mode> struct CertifiedGeometryValue;

template<> struct CertifiedGeometryValue<ArithmeticMode::Ordinary>
{
    static constexpr ArithmeticMode arithmetic_mode = ArithmeticMode::Ordinary;
    NativeScalar value = 0;
    [[nodiscard]] double as_double() const;
    /** Outward bound on narrowing this stored scalar only. Ordinary arithmetic
     * does not track accumulated construction or arithmetic uncertainty. */
    [[nodiscard]] NativeScalar binary64_absolute_error() const;
};

template<> struct CertifiedGeometryValue<ArithmeticMode::BoundedCertification>
{
    static constexpr ArithmeticMode arithmetic_mode = ArithmeticMode::BoundedCertification;
    NativeScalar value = 0, absolute_error = 0;
    [[nodiscard]] NativeScalar lower() const;
    [[nodiscard]] NativeScalar upper() const;
    [[nodiscard]] double as_double() const;
    [[nodiscard]] bool relative_error_at_most(NativeScalar target, double tolerance) const;
    /** Enclosure including additional checked binary64 narrowing. */
    [[nodiscard]] NativeScalar binary64_absolute_error() const;
};

template<> struct CertifiedGeometryValue<ArithmeticMode::Exact>
{
    static constexpr ArithmeticMode arithmetic_mode = ArithmeticMode::Exact;
    NativeScalar value = 0, absolute_error = 0;
    /** The retained rational is nonzero even if this output interval contains zero. */
    bool exact_nonzero = false;
    [[nodiscard]] double as_double() const;
    [[nodiscard]] NativeScalar binary64_absolute_error() const;
};

template<ArithmeticMode Mode> using GeometryValue = CertifiedGeometryValue<Mode>;

template<ArithmeticMode Mode> struct GeometryMoments
{
    CertifiedGeometryValue<Mode> volume; ///< m^3
    GeometryPoint origin{};
    std::array<CertifiedGeometryValue<Mode>, 3> first_moment; ///< integral (x-origin) dV, m^4
};

} // namespace SimpleFluid::PLIC
