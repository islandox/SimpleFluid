/** @file testSparseNativeMesh.cc
 * @brief Native sparse dispatch, actual maps, halos and metric-only neighbors.
 */
#include "fields/FieldStored.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include "parallel/CompositePartition.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <set>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using LO = DefaultTpetraTypes::local_ordinal_type;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
enum class Corruption
{
    None,
    SwapEndpoints,
    WrongIntroducedCoordinate,
    WrongNativeAlias,
    WrongLoopOrder
};
struct Fixture
{
    SP<MultiRegionMesh> shard;
    std::vector<std::uint64_t> owned, ghosts;
    std::weak_ptr<const MultiRegionMesh> old;
    Fixture(Corruption corruption = Corruption::None, bool excessive_faces = false, bool excessive_nodes = false)
    {
        auto comm = Tpetra::getDefaultComm();
        ::SimpleFluid::Meshes::AMR::AMRGrid grid;
        grid.block_cells = {2, 2, 2};
        auto source = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(grid);
        const auto mesh = source->geometry();
        old = mesh;
        auto cell_gid = [](std::uint64_t cell) { return cell * 3 % 8; };
        auto face_gid = [](std::uint64_t face) { return face * 5 % 36; };
        auto node_gid = [](std::uint64_t node) { return node * 5 % 27; };
        auto owner = [&](std::uint64_t cell) { return comm->getSize() == 1 ? 0 : (cell < 4 ? 0 : 1); };
        std::set<std::uint64_t> full, metrics, faces, nodes;
        for (std::uint64_t cell = 0; cell < 8; ++cell)
            if (owner(cell) == comm->getRank())
                full.insert(cell);
        auto first = full;
        for (auto cell : first)
            for (auto face : mesh->cell_faces(cell))
            {
                auto other = mesh->opposite_cell(face, cell);
                if (other != MultiRegionMesh::invalid_cell_id())
                    full.insert(other);
            }
        metrics = full;
        for (auto cell : full)
            for (auto face : mesh->cell_faces(cell))
            {
                faces.insert(face);
                for (auto incident : {mesh->owner_cell(face), mesh->neighbor_cell(face)})
                    if (incident != MultiRegionMesh::invalid_cell_id())
                        metrics.insert(incident);
                for (auto node : mesh->face_nodes(face))
                    nodes.insert(node);
            }
        SparseNativeMeshInput input;
        input.global_cells = 8;
        input.global_faces =
            excessive_faces ? std::uint64_t(std::numeric_limits<DefaultTpetraTypes::global_ordinal_type>::max()) + 1
                            : 36;
        input.global_nodes =
            excessive_nodes ? std::uint64_t(std::numeric_limits<DefaultTpetraTypes::global_ordinal_type>::max()) + 1
                            : 27;
        if (!full.empty())
            input.fragments = mesh->regions();
        for (auto boundary : mesh->boundary_batch_ids())
            input.boundaries.emplace(boundary, mesh->boundary_batch_name(boundary));
        for (auto cell : metrics)
        {
            auto p = mesh->cell_centroid(cell);
            input.cells.push_back(
                {cell_gid(cell), owner(cell), 0, cell, full.contains(cell), mesh->cell_volume(cell), {p.x, p.y, p.z}});
            if (full.contains(cell))
                (owner(cell) == comm->getRank() ? owned : ghosts).push_back(cell_gid(cell));
        }
        std::map<SimpleFluid::PLIC::GeometryPoint, std::uint64_t> native_points;
        for (auto node : nodes)
        {
            const auto p = mesh->node_coordinates(node);
            native_points[{p.x, p.y, p.z}] = node;
            input.nodes.push_back({node_gid(node), std::pair<std::size_t, std::uint64_t>{0, node}, {}});
            input.native_nodes.emplace(std::pair<std::size_t, std::uint64_t>{0, node}, node_gid(node));
        }
        for (auto face : faces)
        {
            SparseNativeFace route;
            route.gid = face_gid(face);
            route.first = cell_gid(mesh->owner_cell(face));
            const auto neighbor = mesh->neighbor_cell(face);
            if (neighbor != MultiRegionMesh::invalid_cell_id())
                route.second = cell_gid(neighbor);
            route.native = {0, face};
            route.boundary = mesh->boundary_id(face);
            route.override_geometry =
                SimpleFluid::PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Ordinary>::from_native_face(*mesh, face);
            for (const auto& point : route.override_geometry->vertex_identities())
            {
                const auto gid = node_gid(
                    native_points.at(SimpleFluid::PLIC::point_display<SimpleFluid::ArithmeticMode::Ordinary>(point)));
                route.override_nodes.push_back(gid);
                input.node_certificates.emplace(gid, point);
            }
            input.faces.push_back(std::move(route));
        }
        if (comm->getRank() == 0)
        {
            if (corruption == Corruption::SwapEndpoints)
            {
                const auto face = std::find_if(input.faces.begin(), input.faces.end(),
                    [](const auto& f) { return f.second != MultiRegionMesh::invalid_cell_id(); });
                if (face == input.faces.end())
                    throw std::logic_error("Sparse test needs an internal face.");
                std::swap(face->first, face->second);
            }
            if (corruption == Corruption::WrongIntroducedCoordinate)
            {
                auto& node = input.nodes.front();
                node.native.reset();
                node.introduced_coordinate = {.125, .25, .375};
            }
            if (corruption == Corruption::WrongNativeAlias)
            {
                auto alias = input.native_nodes.begin();
                const auto next = std::next(alias);
                alias->second = next->second;
            }
            if (corruption == Corruption::WrongLoopOrder)
            {
                auto& nodes = input.faces.front().override_nodes;
                std::swap(nodes[0], nodes[1]);
            }
        }
        shard = MultiRegionMesh::from_sparse_native(std::move(input), comm);
    }
};
TEST(SparseNativeMesh, NoncontiguousRoutesNativeFieldsAndNoConnectivityCache)
{
    auto comm = Tpetra::getDefaultComm();
    Fixture fixture;
    EXPECT_TRUE(fixture.old.expired());
    ASSERT_TRUE(fixture.shard->sparse_native());
    EXPECT_FALSE(fixture.shard->supports_axial_motion());
    auto partition = CompositePartition<>::from_sparse_native(fixture.shard, fixture.owned, fixture.ghosts);
    auto handle = std::make_shared<MeshHandle<>>(partition);
    ScalarCellFieldStored<> field(handle, "sparse_gid");
    for (std::size_t c = 0; c < handle->num_owned_cells(); ++c)
        field.set_owned_value(LO(c), double(handle->cell_global_id(LO(c))) + .25);
    field.sync_ghosts();
    for (std::size_t c = 0; c < handle->num_cells(); ++c)
        EXPECT_EQ(field.local_value(LO(c)), double(handle->cell_global_id(LO(c))) + .25);
    EXPECT_EQ(handle->connectivity_storage_bytes(), 0u);
    EXPECT_FALSE(handle->has_materialized_indexer());
    std::size_t visited = 0;
    handle->visit_owned_region_cells(
        [&](LO local, std::uint64_t gid, std::uint64_t native, const auto& geometry)
        {
            ++visited;
            EXPECT_EQ(handle->cell_global_id(local), gid);
            EXPECT_EQ(gid, native * 3 % 8);
            MeshUtils::Vec3 sum{};
            geometry.visit_cell_faces(
                native, [&](const auto& face) { sum = sum + face.face_area_vector_outward(gid); });
            EXPECT_NEAR(sum.norm(), 0, 1e-15);
            EXPECT_EQ(fixture.shard->certified_sparse_cell_geometry(gid)->moments().volume.as_double(), .125);
        });
    EXPECT_EQ(visited, fixture.owned.size());
    int local_empty = fixture.owned.empty(), empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty);
    EXPECT_EQ(empty, std::max(0, comm->getSize() - 2));
    if (comm->getRank() >= 2)
    {
        EXPECT_TRUE(fixture.shard->regions().empty());
        EXPECT_EQ(handle->num_cells(), 0u);
    }
}
TEST(SparseNativeMesh, ForgedSignedIncidenceAndExactNodeRoutesRejectCollectively)
{
    Fixture accepted;
    const auto epoch = accepted.shard->geometry_epoch();
    for (auto corruption : {Corruption::SwapEndpoints, Corruption::WrongIntroducedCoordinate,
             Corruption::WrongNativeAlias, Corruption::WrongLoopOrder})
        EXPECT_THROW((void) Fixture(corruption), std::invalid_argument);
    EXPECT_EQ(accepted.shard->geometry_epoch(), epoch);
    for (auto cell : accepted.owned)
        EXPECT_EQ(accepted.shard->cell_volume(cell), .125);
}
TEST(SparseNativeMesh, TypedPartitionRejectsUnrepresentableFaceAndNodeDomains)
{
    for (bool faces : {true, false})
    {
        Fixture oversized(Corruption::None, faces, !faces);
        EXPECT_THROW(CompositePartition<>::from_sparse_native(oversized.shard, oversized.owned, oversized.ghosts),
            std::invalid_argument);
    }
    Fixture valid;
    EXPECT_NO_THROW(CompositePartition<>::from_sparse_native(valid.shard, valid.owned, valid.ghosts));
}
} // namespace
