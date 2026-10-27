/** @file SupportClassification.hh
 * @brief Original-support contact and represented-coordinate boundary classification.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"

namespace SimpleFluid::PLIC
{
/** Membership consults the sealed coordinate snapshot; arbitrary coordinate
 * mutation invalidates it. Keys identify geometric supports only. The caller
 * authenticates geometry revision, ownership and transaction independently. */
[[nodiscard]] bool has_support(const DynamicPoint&, const PlaneAuthority&);
[[nodiscard]] bool has_any_support(const DynamicPoint&, std::span<const PlaneAuthority>);
/** Uses the identical original native support packet as source-point export. */
[[nodiscard]] PlaneAuthority original_support_authority(const PlanarFace<ArithmeticMode::Ordinary>&);

enum class OrdinaryBoundaryClass
{
    Inside,
    ConstructionContact,
    CoordinateContact,
    Outside,
    Unresolved
};

/** Retains the immutable original cell plane, including its frame and outward
 * incidence. Clipped/rounded face tiles must not reconstruct that plane from
 * their represented vertices. Inside names the retained cell half-space.
 * Construction contact is authoritative only within a caller-authenticated
 * source/geometry transaction. An unresolved predicate never establishes contact.
 * Coordinate predicates preserve the caller's floating-point environment. */
class OrdinaryBoundary
{
public:
    explicit OrdinaryBoundary(FaceConstraint<ArithmeticMode::Ordinary>);
    [[nodiscard]] std::span<const PlaneAuthority> authorities() const;
    [[nodiscard]] OrdinaryBoundaryClass classify(const Point<ArithmeticMode::Ordinary>&) const;
    /** Wrapper and its owned fallback key only; retained immutable cell storage
     * belongs to the caller's geometry accounting. */
    [[nodiscard]] std::size_t storage_bytes() const
    {
        return sizeof(*this) + native_authority_.key.capacity() * sizeof(std::uint64_t);
    }

private:
    FaceConstraint<ArithmeticMode::Ordinary> boundary_;
    PlaneAuthority native_authority_;
};
} // namespace SimpleFluid::PLIC
