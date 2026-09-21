/** @file testSweptRZRegionProviders.cc
 * @brief Exact compact sweep metrics, coincident seams, MPI, and affine ALE.
 */
#include <gtest/gtest.h>
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/SweptRZRegionProviders.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <cmath>
#include <numbers>
#include <set>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Handle = MeshHandle<>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

Arr<MeshUtils::Vec3> base_nodes()
{
    return {{1, 0, 0}, {1, 1, 0}, {2, 1, 0}, {2, 0, 0}, {3, 0, 0}};
}

ArrReal angles(unsigned count)
{
    ArrReal result;
    for (unsigned k = 0; k <= count; ++k)
        result.push_back(2 * std::numbers::pi_v<real_t> * k / count);
    return result;
}

std::shared_ptr<const ExtrudedTopology> topology(unsigned layers)
{
    return std::make_shared<const ExtrudedTopology>(5,
        Arr<Arr<unsigned>>{{0, 1, 2, 3}, {3, 2, 4}}, layers);
}

ExplicitConformingInterface seam(const ExtrudedTopology& t)
{
    ExplicitConformingInterface result;
    result.first_region = result.second_region = 0;
    const auto& indexer = t.indexer();
    const auto first = indexer.face_ordinal({0, 0, SemiStructuredIndexer::AXIAL});
    const auto last = indexer.face_ordinal({0, indexer.num_layers, SemiStructuredIndexer::AXIAL});
    result.first_boundary = t.boundary_id(first);
    result.second_boundary = t.boundary_id(last);
    for (unsigned cell = 0; cell < indexer.num_cells_per_layer; ++cell)
        result.faces.emplace_back(indexer.face_ordinal({cell, 0, SemiStructuredIndexer::AXIAL}),
            indexer.face_ordinal({cell, indexer.num_layers, SemiStructuredIndexer::AXIAL}));
    return result;
}

SP<MultiRegionMesh> full_sweep(unsigned layers = 16)
{
    const auto t = topology(layers);
    return std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{
        swept_rz_region("corner", t, base_nodes(), angles(layers))},
        std::vector<MultiRegionMesh::Interface>{seam(*t)});
}
} // namespace

TEST(SweptRZRegionProvidersTest, ExactMomentsAgreeWithPlanarHexAndWedgeDecomposition)
{
    const auto t = topology(4);
    const SweptRZGeometry geometry(t, base_nodes(), {0, 0.15, 0.5, 1.1, 1.6});
    for (size_t cell = 0; cell < t->layout().cells; ++cell)
    {
        std::vector<MeshUtils::Vec3> nodes;
        for (const auto node : t->cell_nodes(cell)) nodes.push_back(geometry.node_coordinates(node));
        const bool hex = nodes.size() == 8;
        const auto volume = hex ? MeshUtils::hex_volume(nodes) : MeshUtils::wedge_volume(nodes);
        const auto center = hex ? MeshUtils::hex_centroid(nodes) : MeshUtils::wedge_centroid(nodes);
        EXPECT_NEAR(geometry.cell_volume(cell), volume, 2e-14);
        EXPECT_NEAR((geometry.cell_centroid(cell) - center).norm(), 0, 2e-14);
        const auto base_side = hex ? 3U : 2U;
        EXPECT_GT((nodes[1] - nodes[0]).cross(nodes[base_side] - nodes[0])
            .dot(nodes[nodes.size() / 2] - nodes[0]), 0);
        MeshUtils::Vec3 closure{};
        for (const auto face : t->cell_faces(cell))
            closure = closure + geometry.face_area_vector(face) * (t->owner_cell(face) == cell ? 1.0 : -1.0);
        EXPECT_NEAR(closure.norm(), 0, 2e-14);
    }
    for (size_t face = 0; face < t->layout().faces; ++face)
    {
        std::vector<MeshUtils::Vec3> nodes;
        for (const auto node : t->face_nodes(face)) nodes.push_back(geometry.node_coordinates(node));
        auto vector = MeshUtils::face_area_vector(nodes);
        const auto center = MeshUtils::face_centroid(nodes);
        if (vector.dot(center - geometry.cell_centroid(t->owner_cell(face))) < 0) vector = vector * -1;
        EXPECT_NEAR((geometry.face_area_vector(face) - vector).norm(), 0, 2e-14);
        EXPECT_NEAR((geometry.face_centroid(face) - center).norm(), 0, 2e-14);
    }
    std::vector<MeshUtils::Vec3> first_nodes;
    for (const auto node : t->cell_nodes(0)) first_nodes.push_back(geometry.node_coordinates(node));
    EXPECT_GT((geometry.cell_centroid(0) - MeshUtils::average(first_nodes)).norm(), 0.02);
}

TEST(SweptRZRegionProvidersTest, StoresBaseMetricsAndAngleEdgesWithoutExpandedGeometry)
{
    const SweptRZGeometry small(topology(8), base_nodes(), angles(8));
    const SweptRZGeometry large(topology(64), base_nodes(), angles(64));
    EXPECT_EQ(small.layout().cells * 8, large.layout().cells);
    EXPECT_EQ(large.storage_report().geometry - small.storage_report().geometry,
        (large.theta_edges().capacity() - small.theta_edges().capacity()) * sizeof(real_t));
    EXPECT_EQ(large.axial_bounds(), (std::array<real_t, 2>{0, 1}));
}

TEST(SweptRZRegionProvidersTest, BoundaryAliasesSeparateAngularCapsFromPhysicalBottom)
{
    const Arr<Arr<unsigned>> cells{{0, 1, 2, 3}};
    const Arr<SemiStructMeshTopo::BoundaryEdge> boundaries{
        {0, 1, "inner_wall"}, {1, 2, "upper_join"},
        {2, 3, "outer_join"}, {3, 0, "bottom_wall"}};
    const std::map<std::string, std::string> aliases{
        {"zmin", "theta_min"}, {"zmax", "theta_max"}, {"bottom_wall", "zmin"}};
    const ExtrudedTopology t(4, cells, 8, boundaries, aliases);
    std::set<std::string> names;
    for (const auto boundary : t.boundary_batch_ids()) names.insert(t.boundary_batch_name(boundary));
    EXPECT_EQ(names, (std::set<std::string>{"inner_wall", "upper_join", "outer_join",
        "zmin", "theta_min", "theta_max"}));
    const auto cap = t.indexer().face_ordinal({0, 0, SemiStructuredIndexer::AXIAL});
    EXPECT_EQ(t.boundary_batch_name(t.boundary_id(cap)), "theta_min");
    EXPECT_EQ(t.native_topology().boundary_batch_name(t.boundary_id(cap)), "zmin");
    EXPECT_GT(t.storage_report().topology, ExtrudedTopology(4, cells, 8, boundaries).storage_report().topology);
    EXPECT_THROW((ExtrudedTopology(4, cells, 8, boundaries, {{"missing", "new"}})), std::invalid_argument);
    EXPECT_THROW((ExtrudedTopology(4, cells, 8, boundaries, {{"bottom_wall", ""}})), std::invalid_argument);
    EXPECT_THROW((ExtrudedTopology(4, cells, 8, boundaries, {{"bottom_wall", "zmin"}})), std::invalid_argument);
}

TEST(SweptRZRegionProvidersTest, RejectsInvalidGeometryAndWrongWinding)
{
    const auto t = topology(8);
    auto nodes = base_nodes();
    nodes[0].x = 0;
    EXPECT_THROW((SweptRZGeometry(t, nodes, angles(8))), std::invalid_argument);
    nodes = base_nodes(); nodes[0].z = 1;
    EXPECT_THROW((SweptRZGeometry(t, nodes, angles(8))), std::invalid_argument);
    const auto ccw = std::make_shared<const ExtrudedTopology>(4,
        Arr<Arr<unsigned>>{{0, 3, 2, 1}}, 8);
    nodes = base_nodes(); nodes.resize(4);
    EXPECT_THROW((SweptRZGeometry(ccw, nodes, angles(8))), std::invalid_argument);
    auto bad_angles = angles(8); bad_angles[1] = bad_angles[0];
    EXPECT_THROW((SweptRZGeometry(t, base_nodes(), bad_angles)), std::invalid_argument);
    bad_angles = angles(8); bad_angles.back() += 0.1;
    EXPECT_THROW((SweptRZGeometry(t, base_nodes(), bad_angles)), std::invalid_argument);
    EXPECT_THROW((SweptRZGeometry(topology(2), base_nodes(), {0, std::numbers::pi, 2 * std::numbers::pi})),
        std::invalid_argument);
}

TEST(SweptRZRegionProvidersTest, CoincidentSelfSeamRetainsChecksAndCanonicalIncidence)
{
    const auto mesh = full_sweep();
    EXPECT_EQ(mesh->num_cells(), 32U);
    EXPECT_EQ(mesh->num_faces(), 128U);
    const auto& source = std::get<SweptRZRegion>(mesh->regions()[0]).topology();
    for (const auto& [first, second] : seam(source).faces)
    {
        const auto canonical = mesh->canonical_face({0, first});
        EXPECT_EQ(canonical, mesh->canonical_face({0, second}));
        EXPECT_EQ(mesh->owner_cell(canonical), source.owner_cell(first));
        EXPECT_EQ(mesh->neighbor_cell(canonical), source.owner_cell(second));
    }
    for (size_t cell = 0; cell < mesh->num_cells(); ++cell)
    {
        MeshUtils::Vec3 closure{};
        for (const auto face : mesh->cell_faces(cell))
        {
            closure = closure + mesh->face_area_vector_outward(face, cell);
            if (mesh->neighbor_cell(face) != MultiRegionMesh::invalid_cell_id())
                EXPECT_NE(mesh->owner_cell(face), mesh->neighbor_cell(face));
        }
        EXPECT_NEAR(closure.norm(), 0, 3e-14);
    }
    const auto t = topology(16);
    std::vector<MultiRegionMesh::Region> regions{
        swept_rz_region("corner", t, base_nodes(), angles(16))};
    auto mapping = seam(*t);
    std::swap(mapping.faces[0].second, mapping.faces[1].second);
    EXPECT_THROW((MultiRegionMesh(regions, {mapping})), std::invalid_argument);
    mapping = seam(*t); mapping.faces.pop_back();
    EXPECT_THROW((MultiRegionMesh(regions, {mapping})), std::invalid_argument);
    mapping = seam(*t);
    mapping.second_boundary = mapping.first_boundary;
    for (auto& [first, second] : mapping.faces) second = first;
    EXPECT_THROW((MultiRegionMesh(regions, {mapping})), std::invalid_argument);
    auto open_angles = angles(16); open_angles.back() -= 0.1;
    regions = {swept_rz_region("open", t, base_nodes(), open_angles)};
    EXPECT_THROW((MultiRegionMesh(regions, {seam(*t)})), std::invalid_argument);
}

TEST(SweptRZRegionProvidersTest, CompactOwnershipAndAffineMotionPreserveGCLAndRollback)
{
    const auto geometry = full_sweep();
    auto mesh = std::make_shared<Handle>(geometry);
    EXPECT_EQ(mesh->owned_cell_map()->getGlobalNumElements(), geometry->num_cells());
    EXPECT_EQ(mesh->owned_face_map()->getGlobalNumElements(), geometry->num_faces());
    EXPECT_FALSE(mesh->has_materialized_indexer());
    EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
    const auto comm = Tpetra::getDefaultComm();
    real_t before_local = 0, before = 0;
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell) before_local += mesh->cell_volume(cell);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &before_local, &before);
    const auto original = geometry->node_coordinates(0);
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.1, 0.25);
    real_t after_local = 0, after = 0;
    for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell) after_local += mesh->cell_volume(cell);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &after_local, &after);
    EXPECT_NEAR(after, 1.1 * before, 1e-13);
    const auto& region = std::get<SweptRZRegion>(geometry->regions()[0]);
    EXPECT_EQ(region.geometry().axial_bounds(), (std::array<real_t, 2>{0, 1}));
    motion.rollback_trial();
    EXPECT_EQ(geometry->axial_edges(), (ArrReal{0, 1}));
    EXPECT_EQ(geometry->node_coordinates(0), original);
}

TEST(SweptRZRegionProvidersTest, RankDivergentSweepDescriptorsFailCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1) GTEST_SKIP() << "Requires two ranks.";
    const auto t = topology(16);
    auto nodes = base_nodes();
    if (comm->getRank() != 0)
        for (auto& node : nodes) node.y += 0.1;
    EXPECT_THROW((MultiRegionMesh({swept_rz_region("corner", t, nodes, angles(16))}, {seam(*t)})),
        std::invalid_argument);
}
