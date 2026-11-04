/** @file CellGeometryConstraints.cc
 * @brief Trusted original native plane facts without general shell ownership.
 */
#include "geometry/PLIC/CellGeometryConstraints.hh"
#include "geometry/PLIC/details/NativeFaceConstraintAccess.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
namespace SimpleFluid::PLIC::details
{
DynamicOriginalFaceConstraint NativeFaceConstraintAccess::make(
    const CellGeometryView<ArithmeticMode::Ordinary>& view, std::size_t face)
{
    const auto source = NativeCellGeometrySourceAccess::original_face(view, face);
    DynamicOriginalFaceConstraint::Facts facts{.support = view.source_support(face)};
    facts.normal = source.normal;
    facts.anchor = source.anchor;
    facts.frame_origin = source.frame_origin;
    facts.frame_exponent = source.frame_exponent;
    facts.orientation = source.orientation;
    facts.offset = source.offset;
    return DynamicOriginalFaceConstraint(std::move(facts));
}
} // namespace SimpleFluid::PLIC::details
