/** @file ResolvedTransportGeometry.hh
 * @brief Borrow one resolved region face using the existing field-local IDs.
 */
#pragma once

#include "FVM/details/ResolvedDiffusionGeometry.hh"

namespace SimpleFluid::FVM::detail
{
/**
 * @brief Face-local metric adapter for the existing transport formulas.
 *
 * The row and adjacent field IDs retain their owned/overlap-map meaning. Only
 * the two incident cells are translated, by comparison, into the already
 * resolved canonical geometry. The adapter and face expire after the callback.
 */
template<class MeshType, class Face> class ResolvedTransportGeometry
{
public:
    using scalar_type = typename MeshType::scalar_type;
    using local_ordinal_type = typename MeshType::local_ordinal_type;
    using Vec3 = typename MeshType::Vec3;
    using canonical_id_type = decltype(Face::owner);

    ResolvedTransportGeometry(const Face& face, local_ordinal_type cell, canonical_id_type canonical_cell)
        : d_face(face), d_cell(cell), d_canonical_cell(canonical_cell), d_geometry(face) {}

    auto face_normal(local_ordinal_type) const { return d_face.normal(); }
    auto face_normal_outward(local_ordinal_type, local_ordinal_type cell) const
    { return d_face.face_normal_outward(canonical_cell(cell)); }
    auto face_area_vector_outward(local_ordinal_type, local_ordinal_type cell) const
    { return d_face.face_area_vector_outward(canonical_cell(cell)); }
    auto face_area(local_ordinal_type) const { return d_face.area(); }
    auto face_centroid(local_ordinal_type) const { return d_face.centroid; }
    auto cell_centroid(local_ordinal_type cell) const
    { return d_geometry.cell_centroid(canonical_cell(cell)); }
    auto cell_center_vector(local_ordinal_type, local_ordinal_type cell) const
    { return d_face.cell_center_vector(canonical_cell(cell)); }
    auto face_center_vector(local_ordinal_type, local_ordinal_type cell) const
    { return d_face.face_center_vector(canonical_cell(cell)); }
    auto face_cell_center_distance(local_ordinal_type) const
    { return d_geometry.face_cell_center_distance(d_face.face); }
    auto cell_to_face_distance(local_ordinal_type, local_ordinal_type cell) const
    { return d_geometry.cell_to_face_distance(d_face.face, canonical_cell(cell)); }

private:
    canonical_id_type canonical_cell(local_ordinal_type cell) const
    { return cell == d_cell ? d_canonical_cell : d_face.opposite_cell(d_canonical_cell); }
    const Face& d_face;
    local_ordinal_type d_cell;
    canonical_id_type d_canonical_cell;
    ResolvedDiffusionGeometry<Face, scalar_type> d_geometry;
};
} // namespace SimpleFluid::FVM::detail
