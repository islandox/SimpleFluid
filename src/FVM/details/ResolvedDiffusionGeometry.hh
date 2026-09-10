/** @file ResolvedDiffusionGeometry.hh
 * @brief Present one resolved face to the existing two-point coefficient formulas.
 */
#pragma once

#include <cstdint>

namespace SimpleFluid::FVM::detail
{
/**
 * @brief A borrowed, single-face adapter with canonical cell identifiers.
 *
 * The numerical coefficient functions consume exactly the same metrics and
 * arithmetic as the generic mesh path. All accessors use the already resolved
 * value; none performs canonical-to-native decoding or interface searches.
 */
template<class Face, class Scalar> class ResolvedDiffusionGeometry
{
public:
    using scalar_type = Scalar;
    using local_ordinal_type = std::uint64_t;
    explicit ResolvedDiffusionGeometry(const Face& face) : d_face(face) {}
    auto face_normal_outward(local_ordinal_type, local_ordinal_type cell) const
    { return d_face.face_normal_outward(cell); }
    auto face_area(local_ordinal_type) const { return d_face.area(); }
    auto face_centroid(local_ordinal_type) const { return d_face.centroid; }
    auto cell_centroid(local_ordinal_type cell) const
    { return cell == d_face.owner ? d_face.owner_centroid : d_face.neighbor_centroid; }
    auto cell_center_vector(local_ordinal_type, local_ordinal_type cell) const
    { return d_face.cell_center_vector(cell); }
    auto face_cell_center_distance(local_ordinal_type) const
    { return d_face.interior() ? d_face.owner_to_neighbor.norm() : 0.0; }
    auto cell_to_face_distance(local_ordinal_type, local_ordinal_type cell) const
    { return d_face.face_center_vector(cell).norm(); }
private:
    const Face& d_face;
};
} // namespace SimpleFluid::FVM::detail
