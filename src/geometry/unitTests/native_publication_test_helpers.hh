/** Complete native publication parity against the public materializer. */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include <algorithm>
#include <bit>
#include <gtest/gtest.h>
#include <vector>

namespace SimpleFluid::test::native_publication
{
namespace A = Meshes::AMR;
inline void same_bits(double first, double second)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(first), std::bit_cast<std::uint64_t>(second));
}
inline void same_vector(MeshUtils::Vec3 first, MeshUtils::Vec3 second)
{
    same_bits(first.x, second.x);
    same_bits(first.y, second.y);
    same_bits(first.z, second.z);
}
template<class Range> std::vector<std::uint64_t> ids(const Range& range)
{
    return {range.begin(), range.end()};
}
enum class ExplicitOrdinals
{
    Preserve,
    Repacked
};

inline void complete_parity(const A::AdaptiveNativePublicationView& view, const A::AdaptiveNativePublication& oracle,
    ExplicitOrdinals ordinals = ExplicitOrdinals::Preserve)
{
    const auto& actual = *view.partition()->geometry();
    const auto& expected = *oracle.partition->geometry();
    const auto explicit_repacked = [&](std::size_t actual_region, std::size_t expected_region)
    {
        using Family = Meshes::RegionLayout::Family;
        return ordinals == ExplicitOrdinals::Repacked &&
               (actual.region_layout(actual_region).family == Family::Explicit ||
                   expected.region_layout(expected_region).family == Family::Explicit);
    };
    EXPECT_EQ(view.partition()->plan().cell_ids(), oracle.partition->plan().cell_ids());
    EXPECT_EQ(view.partition()->plan().face_ids(), oracle.partition->plan().face_ids());
    EXPECT_EQ(view.partition()->plan().node_ids(), oracle.partition->plan().node_ids());
    for (const auto cell : view.partition()->plan().cell_ids())
    {
        EXPECT_EQ(view.local_keys().at(cell), oracle.local_keys.at(cell));
        same_bits(actual.cell_volume(cell), expected.cell_volume(cell));
        same_vector(actual.cell_centroid(cell), expected.cell_centroid(cell));
        const auto [ar, ac] = actual.native_cell(cell);
        const auto [er, ec] = expected.native_cell(cell);
        auto actual_faces = ids(actual.cell_faces(cell)), expected_faces = ids(expected.cell_faces(cell));
        auto actual_nodes = ids(actual.cell_nodes(cell)), expected_nodes = ids(expected.cell_nodes(cell));
        if (explicit_repacked(ar, er))
        {
            std::sort(actual_faces.begin(), actual_faces.end());
            std::sort(expected_faces.begin(), expected_faces.end());
            std::sort(actual_nodes.begin(), actual_nodes.end());
            std::sort(expected_nodes.begin(), expected_nodes.end());
        }
        else
        {
            EXPECT_EQ(ac, ec);
            EXPECT_EQ(std::visit([](const auto& r) { return r.name(); }, actual.region(ar)),
                std::visit([](const auto& r) { return r.name(); }, expected.region(er)));
        }
        EXPECT_EQ(actual_faces, expected_faces);
        EXPECT_EQ(actual_nodes, expected_nodes);
    }
    EXPECT_EQ(actual.boundary_batch_ids(), expected.boundary_batch_ids());
    for (const auto boundary : actual.boundary_batch_ids())
        EXPECT_EQ(actual.boundary_batch_name(boundary), expected.boundary_batch_name(boundary));
    for (const auto face : view.partition()->plan().face_ids())
    {
        EXPECT_EQ(ids(actual.face_nodes(face)), ids(expected.face_nodes(face)));
        EXPECT_EQ(actual.owner_cell(face), expected.owner_cell(face));
        EXPECT_EQ(actual.neighbor_cell(face), expected.neighbor_cell(face));
        EXPECT_EQ(actual.boundary_id(face), expected.boundary_id(face));
        same_bits(actual.face_area(face), expected.face_area(face));
        same_vector(actual.face_centroid(face), expected.face_centroid(face));
        same_vector(actual.face_area_vector(face), expected.face_area_vector(face));
        const auto af = actual.native_face(face), ef = expected.native_face(face);
        if (!explicit_repacked(af.region, ef.region))
        {
            EXPECT_EQ(af.face, ef.face);
            EXPECT_EQ(std::visit([](const auto& r) { return r.name(); }, actual.region(af.region)),
                std::visit([](const auto& r) { return r.name(); }, expected.region(ef.region)));
        }
        same_bits(actual.native_flux_orientation(af, face), expected.native_flux_orientation(ef, face));
        if (!actual.is_exterior_face(face))
        {
            same_vector(actual.cell_center_vector(face, actual.owner_cell(face)),
                expected.cell_center_vector(face, expected.owner_cell(face)));
            same_vector(actual.cell_center_vector(face, actual.neighbor_cell(face)),
                expected.cell_center_vector(face, expected.neighbor_cell(face)));
        }
    }
    for (const auto node : view.partition()->plan().node_ids())
        same_vector(actual.node_coordinates(node), expected.node_coordinates(node));
}
} // namespace SimpleFluid::test::native_publication
