/** @file GeometryIdentity.hh
 * @brief Private native Exact identity tuple issuance and access.
 */
#pragma once
#include "geometry/PLIC/details/exact/ConvexCellGeometryInternal.hh"

namespace SimpleFluid::PLIC::details
{
[[nodiscard]] GeometryIdentityBackingPtr import_exact_identity_backing(
    GeometryIdentityKind, std::span<const char> checked_key, std::size_t byte_budget);
[[nodiscard]] std::size_t exact_identity_import_storage_bound(GeometryIdentityKind, std::span<const char> checked_key);
} // namespace SimpleFluid::PLIC::details
namespace SimpleFluid::PLIC::exact_geometry_detail
{
/** Canonical borrowed-body parser without full integer/key byte temporaries. */
[[nodiscard]] Q read_identity_rational(WireReader&);
[[nodiscard]] PointIdentity<ArithmeticMode::Exact> make_point_identity(const P&);
[[nodiscard]] DisplacementIdentity<ArithmeticMode::Exact> make_displacement_identity(const P&);
[[nodiscard]] LineIdentity<ArithmeticMode::Exact> make_line_identity(const P& direction, const P& moment);
[[nodiscard]] PlaneSupportIdentity<ArithmeticMode::Exact> make_support_identity(const std::array<Q, 4>&);
[[nodiscard]] const std::array<Q, 6>& line_coefficients(const LineIdentity<ArithmeticMode::Exact>&);
[[nodiscard]] const std::array<Q, 4>& support_coefficients(const PlaneSupportIdentity<ArithmeticMode::Exact>&);
/** Root, copied native tuple limbs and transient emission scratch.
 * This bound excludes allocator/control blocks, as does payload admission. */
[[nodiscard]] std::size_t point_identity_storage_bound(const P&) noexcept;
} // namespace SimpleFluid::PLIC::exact_geometry_detail
