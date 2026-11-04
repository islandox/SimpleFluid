/** @file NativeFaceConstraintAccess.hh
 * @brief Private trusted original-plane constraint issuance from immutable views.
 */
#pragma once
#include "geometry/PLIC/CellGeometryView.hh"
namespace SimpleFluid::PLIC::details
{
class NativeFaceConstraintAccess
{
public:
    [[nodiscard]] static DynamicOriginalFaceConstraint make(
        const CellGeometryView<ArithmeticMode::Ordinary>&, std::size_t);
    [[nodiscard]] static const DynamicOriginalFaceConstraint::Facts& facts(
        const DynamicOriginalFaceConstraint& value) noexcept
    {
        return value.facts_;
    }
};
} // namespace SimpleFluid::PLIC::details
