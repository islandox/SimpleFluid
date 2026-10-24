/** @file testDecomposedOrthogonalTopology.cc
 * @brief Virtual-rank exact halo, incidence, periodic and compact storage checks.
 */
#include "geometry/mesh/DecomposedOrthogonalTopology.hh"
#include <algorithm>
#include <gtest/gtest.h>
#include <set>

namespace
{
using Topology = SimpleFluid::Meshes::DecomposedOrthogonalTopology;
using Source = SimpleFluid::Meshes::OrthoMeshTopo;
using Indexer = SimpleFluid::Meshes::OrthogonalIndexer;
using Point = Topology::Coordinates;

std::shared_ptr<Source> source(Point n, Point periodic = {0, 0, 0})
{
    return std::make_shared<Source>(n[0], n[1], n[2], periodic[0], periodic[1], periodic[2],
        Source::BoundaryNames{"imin", "imax", "jmin", "jmax", "kmin", "kmax"});
}

void verify(Point n, Point periodic, Point begin, Point end, size_t depth)
{
    auto mesh = source(n, periodic);
    Topology local(mesh, begin, end, depth);
    const auto& ix = mesh->indexer();
    std::set<Indexer::CellID> owned, expected;
    for (unsigned k = begin[2]; k < end[2]; ++k)
        for (unsigned j = begin[1]; j < end[1]; ++j)
            for (unsigned i = begin[0]; i < end[0]; ++i)
                owned.insert({i, j, k});
    expected = owned;
    auto frontier = owned;
    for (size_t layer = 0; layer < depth; ++layer)
    {
        std::set<Indexer::CellID> next;
        for (auto cell : frontier)
        {
            const auto neighbors = mesh->neighbor_cells(cell);
            for (unsigned t = 0; t < neighbors.num; ++t)
                if (!expected.contains(neighbors.neighbors[t]))
                    next.insert(neighbors.neighbors[t]);
        }
        expected.insert(next.begin(), next.end());
        frontier = std::move(next);
    }
    EXPECT_EQ(local.num_owned_cells(), owned.size());
    EXPECT_EQ(local.num_local_cells(), expected.size());
    std::set<Indexer::CellID> actual;
    std::set<Indexer::FaceID> expected_faces, owned_faces, actual_faces;
    std::set<Indexer::NodeID> expected_nodes, actual_nodes;
    for (size_t row = 0; row < local.num_local_cells(); ++row)
    {
        const auto c = local.source_cell(row);
        EXPECT_EQ(local.cell_local_id(c), row);
        EXPECT_EQ(owned.contains(c), row < local.num_owned_cells());
        actual.insert(c);
        EXPECT_EQ(local.cell_faces(row).size(), 6);
        for (auto face : local.cell_faces(row))
        {
            ASSERT_LT(face, local.num_local_faces());
            const auto f = local.source_face(face);
            EXPECT_TRUE(mesh->owner_cell(f) == c || mesh->neighbor_cell(f) == c);
        }
    }
    EXPECT_EQ(actual, expected);
    // Independent incidence oracle scans all source faces, rather than local boxes.
    for (size_t raw = 0; raw < ix.total_faces(); ++raw)
    {
        const auto f = ix.face_id(raw);
        if (owned.contains(mesh->owner_cell(f)))
            owned_faces.insert(f);
        if (expected.contains(mesh->owner_cell(f)) || expected.contains(mesh->neighbor_cell(f)))
            expected_faces.insert(f);
    }
    for (const auto c : expected)
        for (unsigned mask = 0; mask < 8; ++mask)
        {
            Point p{c.i + ((mask >> 0) & 1), c.j + ((mask >> 1) & 1), c.k + ((mask >> 2) & 1)};
            for (unsigned a = 0; a < 3; ++a)
                if (periodic[a])
                    p[a] %= n[a];
            expected_nodes.insert(
                {static_cast<unsigned>(p[0]), static_cast<unsigned>(p[1]), static_cast<unsigned>(p[2])});
        }
    EXPECT_EQ(local.num_owned_faces(), owned_faces.size());
    for (size_t row = 0; row < local.num_local_faces(); ++row)
    {
        const auto f = local.source_face(row);
        EXPECT_EQ(local.face_local_id(f), row);
        EXPECT_EQ(owned_faces.contains(f), row < local.num_owned_faces());
        EXPECT_EQ(local.boundary_id(row), mesh->boundary_id(f));
        EXPECT_EQ(local.owner_cell(row), local.cell_local_id(mesh->owner_cell(f)));
        EXPECT_EQ(local.neighbor_cell(row), local.cell_local_id(mesh->neighbor_cell(f)));
        actual_faces.insert(f);
    }
    EXPECT_EQ(actual_faces, expected_faces);
    for (size_t row = 0; row < local.num_local_nodes(); ++row)
    {
        const auto node = local.source_node(row);
        EXPECT_EQ(local.node_local_id(node), row);
        actual_nodes.insert(node);
    }
    EXPECT_EQ(actual_nodes, expected_nodes);
    size_t previous = 0;
    for (size_t row = 0; row < local.num_owned_cells(); ++row)
    {
        const auto raw = ix.cell_ordinal(local.source_cell(row));
        if (row)
            EXPECT_GT(raw, previous);
        previous = raw;
    }
    unsigned orientation = 0;
    previous = 0;
    for (size_t row = 0; row < local.num_owned_faces(); ++row)
    {
        const auto f = local.source_face(row);
        const auto raw =
            f.i + ix.num_faces_per_dim_per_orientation[f.orientation][0] *
                      (f.j + static_cast<size_t>(ix.num_faces_per_dim_per_orientation[f.orientation][1]) * f.k);
        if (row && orientation == f.orientation)
            EXPECT_GT(raw, previous);
        EXPECT_GE(f.orientation, orientation);
        orientation = f.orientation;
        previous = raw;
    }
    // Ghost/node order is the public deterministic box concatenation order.
    // Independent BFS and global incidence sets above remain the membership oracle.
    const auto check_boxes = [&](auto boxes, size_t base, auto get)
    {
        size_t offset = 0;
        for (const auto& box : boxes)
        {
            EXPECT_EQ(box.offset, offset);
            for (size_t k = box.begin[2]; k < box.end[2]; ++k)
                for (size_t j = box.begin[1]; j < box.end[1]; ++j)
                    for (size_t i = box.begin[0]; i < box.end[0]; ++i)
                    {
                        const auto id = get(base + offset++);
                        EXPECT_EQ(id.i, i);
                        EXPECT_EQ(id.j, j);
                        EXPECT_EQ(id.k, k);
                    }
        }
        return offset;
    };
    EXPECT_EQ(
        check_boxes(local.cell_boxes(false), local.num_owned_cells(), [&](size_t id) { return local.source_cell(id); }),
        local.num_local_cells() - local.num_owned_cells());
    size_t face_base = local.num_owned_faces();
    for (unsigned o = 0; o < 3; ++o)
        face_base +=
            check_boxes(local.face_boxes(o, false), face_base, [&](size_t id) { return local.source_face(id); });
    EXPECT_EQ(face_base, local.num_local_faces());
    EXPECT_EQ(
        check_boxes(local.node_boxes(), 0, [&](size_t id) { return local.source_node(id); }), local.num_local_nodes());
    EXPECT_EQ(local.lookup_storage_estimate(), 0);
    EXPECT_EQ(local.cell_local_id({}), Topology::invalid_local_id());
    EXPECT_EQ(local.face_local_id({}), Topology::invalid_local_id());
    EXPECT_THROW(local.source_cell(local.num_local_cells()), std::out_of_range);
}

TEST(DecomposedOrthogonalTopologyTest, ExactAdjacencyHalosAcrossVirtualBlocks)
{
    for (size_t depth : {0U, 1U, 2U, 4U})
        for (unsigned mask = 0; mask < 8; ++mask)
        {
            const Point periodic{mask & 1U, (mask >> 1) & 1U, (mask >> 2) & 1U};
            verify({7, 6, 5}, periodic, {0, 0, 0}, {3, 2, 2}, depth);
            verify({7, 6, 5}, periodic, {3, 2, 2}, {7, 6, 5}, depth);
            verify({7, 6, 5}, periodic, {2, 2, 2}, {4, 4, 3}, depth);
        }
}
TEST(DecomposedOrthogonalTopologyTest, ThinPeriodicAxesAndEmptyOwners)
{
    for (size_t depth : {0U, 1U, 2U, 8U})
    {
        verify({1, 2, 7}, {1, 1, 1}, {0, 0, 2}, {1, 1, 4}, depth);
        verify({1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, depth);
        verify({7, 6, 5}, {1, 0, 1}, {2, 2, 2}, {2, 4, 4}, depth);
    }
}
TEST(DecomposedOrthogonalTopologyTest, MetadataDoesNotScaleWithCoreVolume)
{
    Topology small(source({20, 20, 20}), {4, 4, 4}, {12, 12, 12}, 2);
    Topology large(source({2000, 2000, 2000}), {400, 400, 400}, {1200, 1200, 1200}, 2);
    EXPECT_EQ(small.storage_bytes(), large.storage_bytes());
    EXPECT_GT(large.num_owned_cells(), small.num_owned_cells() * 100000);
    auto retained = source({4, 4, 4});
    const auto* identity = retained.get();
    Topology local(retained, {0, 0, 0}, {2, 2, 2}, 1);
    retained.reset();
    EXPECT_EQ(&local.source_topology(), identity);
}
TEST(DecomposedOrthogonalTopologyTest, RejectsInvalidBounds)
{
    EXPECT_THROW((Topology(nullptr, {0, 0, 0}, {1, 1, 1}, 1)), std::invalid_argument);
    EXPECT_THROW((Topology(source({3, 3, 3}), {2, 0, 0}, {1, 1, 1}, 1)), std::invalid_argument);
    EXPECT_THROW((Topology(source({3, 3, 3}), {0, 0, 0}, {4, 1, 1}, 1)), std::invalid_argument);
}
} // namespace
