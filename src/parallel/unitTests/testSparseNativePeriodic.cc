/** @file testSparseNativePeriodic.cc
 * @brief Physical Cartesian periodic images, exact sparse certificates and native node routes.
 */
#include "fields/FieldStored.hh"
#include "geometry/ExactPlanarFace.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include "parallel/CompositePartition.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <gtest/gtest.h>
#include <set>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct PeriodicInput
{
    SparseNativeMeshInput input;
    std::vector<std::uint64_t> owned, ghosts;
    std::map<std::uint64_t, ConvexGeometryPtr> physical;
};
PeriodicInput periodic_input(bool empty_owners = false)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::AMRGrid config;
    config.block_cells = {2, 2, 2};
    config.periodic[0] = true;
    config.origin = {.1, .2, .3};
    config.extent = {.7, .8, .9};
    ::SimpleFluid::Meshes::AMR::CartesianAMR source(config);
    auto mesh = source.geometry();
    PeriodicInput result;
    auto owner = [&](std::uint64_t cell) { return empty_owners ? 0 : int(cell % comm->getSize()); };
    std::set<std::uint64_t> full, metrics, faces;
    for (std::uint64_t cell = 0; cell < 8; ++cell)
        if (owner(cell) == comm->getRank())
            full.insert(cell);
    auto initial = full;
    for (auto cell : initial)
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
        }
    result.input.global_cells = mesh->num_cells();
    result.input.global_faces = mesh->num_faces();
    result.input.global_nodes = mesh->num_nodes();
    if (!full.empty())
        result.input.fragments = mesh->regions();
    for (auto boundary : mesh->boundary_batch_ids())
        result.input.boundaries.emplace(boundary, mesh->boundary_batch_name(boundary));
    const auto& region = std::get<CartesianRegion>(mesh->regions().front());
    for (auto cell : metrics)
    {
        // The strict sparse factory binds metrics to the retained provider.
        // Legacy composite identity-map arithmetic may round z once more.
        auto center = region.geometry().cell_centroid(cell);
        result.input.cells.push_back({cell, owner(cell), 0, cell, full.contains(cell),
            region.geometry().cell_volume(cell), {center.x, center.y, center.z}});
        if (full.contains(cell))
        {
            (owner(cell) == comm->getRank() ? result.owned : result.ghosts).push_back(cell);
            auto box = source.cell_box(cell);
            result.physical.emplace(cell, ConvexCellGeometry::box(box.lower, box.upper));
            for (auto node : region.topology().cell_nodes(cell))
                result.input.native_nodes.emplace(std::pair<std::size_t, std::uint64_t>{0, node}, node);
        }
    }
    std::map<GeometryPoint, std::uint64_t> point_nodes;
    for (std::uint64_t node = 0; node < mesh->num_nodes(); ++node)
    {
        auto p = mesh->node_coordinates(node);
        point_nodes.emplace(GeometryPoint{p.x, p.y, p.z}, node);
    }
    const auto address = [&](std::uint64_t cell, std::uint64_t logical)
    {
        for (auto native : region.topology().cell_faces(cell))
            for (auto candidate : mesh->logical_faces({0, native}))
                if (candidate == logical)
                    return RegionFace{0, native};
        throw std::logic_error("Periodic fixture has no actual native occurrence.");
    };
    for (auto face : faces)
    {
        SparseNativeFace route;
        route.gid = face;
        route.first = mesh->owner_cell(face);
        route.second = mesh->neighbor_cell(face);
        route.boundary = mesh->boundary_id(face);
        auto first = address(route.first, face);
        std::optional<RegionFace> second;
        if (route.second != MultiRegionMesh::invalid_cell_id())
            second = address(route.second, face);
        route.native_is_second = !full.contains(route.first);
        route.native = route.native_is_second ? *second : first;
        if (second && *second != first && full.contains(route.first) && full.contains(route.second))
            route.second_native = *second;
        route.override_geometry = ExactPlanarFace::from_native_face(*mesh, face);
        if (second)
        {
            auto a = region.geometry().face_centroid(first.face), b = region.geometry().face_centroid(second->face);
            route.exact_second_image = exact_displacement({b.x, b.y, b.z}, {a.x, a.y, a.z});
            route.second_image = exact_displacement_display(route.exact_second_image);
        }
        for (const auto& point : route.override_geometry->vertex_identities())
        {
            auto gid = point_nodes.at(exact_point_display(point));
            route.override_nodes.push_back(gid);
            result.input.node_certificates.emplace(gid, point);
            if (second)
            {
                auto physical =
                    exact_point_translated(point, exact_displacement_difference({}, route.exact_second_image));
                auto second_gid = point_nodes.at(exact_point_display(physical));
                route.second_node_gids.push_back(second_gid);
                result.input.node_certificates.emplace(second_gid, std::move(physical));
            }
        }
        result.input.faces.push_back(std::move(route));
    }
    for (const auto& [node, proof] : result.input.node_certificates)
    {
        SparseNativeNode route;
        route.gid = node;
        if (result.input.native_nodes.contains({0, node}))
            route.native = std::pair<std::size_t, std::uint64_t>{0, node};
        else
            route.introduced_coordinate = exact_point_display(proof);
        result.input.nodes.push_back(std::move(route));
    }
    return result;
}
TEST(SparseNativePeriodic, PhysicalShellsNodesSignedFluxAndNeighborVectors)
{
    auto comm = Tpetra::getDefaultComm();
    for (bool empty : {false, true})
    {
        auto fixture = periodic_input(empty);
        auto physical = fixture.physical;
        std::vector<std::uint64_t> local_faces;
        for (const auto& face : fixture.input.faces)
            local_faces.push_back(face.gid);
        auto mesh = MultiRegionMesh::from_sparse_native(std::move(fixture.input), comm);
        auto partition = CompositePartition<>::from_sparse_native(mesh, fixture.owned, fixture.ghosts);
        EXPECT_EQ(partition.plan().owned_cells().size(), fixture.owned.size());
        for (const auto& [cell, certificate] : physical)
        {
            EXPECT_TRUE(mesh->certified_sparse_cell_geometry(cell)->equivalent(*certificate));
            EXPECT_EQ(mesh->cell_type(cell), MeshUtils::CellType::HEXAHEDRON);
            EXPECT_EQ(mesh->cell_nodes(cell).size(), 8);
            MeshUtils::Vec3 sum{};
            auto center = mesh->cell_centroid(cell);
            for (auto node : mesh->cell_nodes(cell))
            {
                auto p = mesh->node_coordinates(node);
                EXPECT_LT((p - center).norm(), 1.0);
            }
            for (auto face : mesh->cell_faces(cell))
                sum =
                    sum + mesh->face_normal(face) * (mesh->face_area(face) * (mesh->owner_cell(face) == cell ? 1 : -1));
            EXPECT_NEAR(sum.norm(), 0, 1e-14);
        }
        auto view = mesh->acquire_execution_view();
        for (auto face : local_faces)
        {
            auto resolved = view.resolve_face(face);
            if (resolved.interior())
            {
                auto normal = mesh->face_normal(face);
                EXPECT_GT(resolved.owner_to_neighbor.dot(normal), 0);
                EXPECT_GT(resolved.owner_to_face.dot(normal), 0);
            }
        }
        if (empty && comm->getRank() != 0)
        {
            EXPECT_TRUE(fixture.owned.empty());
            EXPECT_TRUE(physical.empty());
        }
    }
}
TEST(SparseNativePeriodic, HierarchyRefineMigrationCoarsenPublishesNativeImagesAndFields)
{
    auto comm = Tpetra::getDefaultComm();
    ::SimpleFluid::Meshes::AMR::AMRGrid config;
    config.block_cells = {2, 2, 2};
    config.max_level = 1;
    config.periodic[0] = true;
    config.origin = {.1, .2, .3};
    config.extent = {.7, .8, .9};
    auto original = std::make_shared<::SimpleFluid::Meshes::AMR::CartesianAMR>(config);
    std::weak_ptr<const MultiRegionMesh> old = original->geometry();
    std::vector<std::uint64_t> owned;
    if (comm->getRank() == 0)
        for (std::uint64_t c = 0; c < 8; ++c)
            owned.push_back(c);
    const auto root_key = original->cell_key(0);
    auto detached = original->detach_owned(owned, {'p'});
    original.reset();
    EXPECT_TRUE(old.expired());
    auto kernel = ::SimpleFluid::Meshes::AMR::cartesian_adaptive_kernel(config);
    auto schema = kernel->descriptor();
    auto state = std::make_shared<const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
    const auto value = [](const ::SimpleFluid::Meshes::AMR::AdaptiveCellKey& key)
    {
        const auto& cell = std::get<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(key.ancestry);
        return double(cell.index[0] + 16 * (cell.index[1] + 16 * cell.index[2]) + 4096 * cell.level);
    };
    const auto publish = [&](const ::SimpleFluid::Meshes::AMR::DistributedAdaptiveHierarchy& hierarchy)
    {
        auto publication = ::SimpleFluid::Meshes::AMR::assemble_adaptive_native(hierarchy, schema);
        auto geometry = publication.partition->geometry();
        EXPECT_EQ(publication.mesh->num_owned_cells(), hierarchy.owned().size());
        for (const auto& [key, gid] : publication.owned_gids)
        {
            const auto& leaf = hierarchy.owned().at(key);
            EXPECT_TRUE(geometry->certified_sparse_cell_geometry(gid)->equivalent(*kernel->geometry(leaf)));
            auto metric = kernel->metrics(leaf);
            auto center = geometry->cell_centroid(gid);
            EXPECT_EQ(geometry->cell_volume(gid), metric.volume);
            EXPECT_EQ((GeometryPoint{center.x, center.y, center.z}), metric.centroid);
        }
        ScalarCellFieldStored<> field(publication.mesh, "periodic_native_key");
        using LO = DefaultTpetraTypes::local_ordinal_type;
        for (std::size_t c = 0; c < publication.mesh->num_owned_cells(); ++c)
        {
            const auto gid = publication.mesh->cell_global_id(LO(c));
            field.set_owned_value(LO(c), value(publication.local_keys.at(gid)));
        }
        field.sync_ghosts();
        for (std::size_t c = 0; c < publication.mesh->num_cells(); ++c)
        {
            const auto gid = publication.mesh->cell_global_id(LO(c));
            EXPECT_EQ(field.local_value(LO(c)), value(publication.local_keys.at(gid)));
        }
        std::size_t visited = 0;
        publication.mesh->visit_owned_region_cells(
            [&](LO, std::uint64_t gid, std::uint64_t native, const auto& native_geometry)
            {
                ++visited;
                MeshUtils::Vec3 sum{};
                native_geometry.visit_cell_faces(native,
                    [&](const auto& face)
                    {
                        sum = sum + face.face_area_vector_outward(gid);
                        if (face.interior())
                            EXPECT_GT(face.owner_to_neighbor.dot(face.area_vector), 0);
                    });
                EXPECT_NEAR(sum.norm(), 0, 1e-14);
            });
        EXPECT_EQ(visited, hierarchy.owned().size());
        if (hierarchy.owned().empty())
        {
            EXPECT_EQ(publication.mesh->num_cells(), 0);
            EXPECT_TRUE(geometry->regions().empty());
        }
    };
    publish(*state);
    EXPECT_EQ(state->owned().empty(), comm->getRank() != 0);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    if (comm->getRank() == 0)
        marks.push_back({root_key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic});
    auto refine = state->plan_adaptation(marks);
    auto refined = state->apply(refine);
    publish(*refined);
    std::vector<int> destinations;
    for (const auto& [key, leaf] : refined->owned())
    {
        const auto& cell = std::get<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(key.ancestry);
        destinations.push_back(int((cell.index[0] + 4 * (cell.index[1] + 4 * cell.index[2])) % comm->getSize()));
    }
    auto migration = refined->plan_migration(destinations);
    auto moved = refined->apply(migration);
    publish(*moved);
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveBytes> inverse;
    if (comm->getRank() == 0)
        inverse.push_back(::SimpleFluid::Meshes::AMR::adaptive_family_key(
            root_key, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic));
    auto coarsen = moved->plan_adaptation({}, inverse);
    auto restored = moved->apply(coarsen);
    publish(*restored);
    std::uint64_t count = restored->owned().size(), global = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &count, &global);
    EXPECT_EQ(global, 8);
    EXPECT_EQ(state->owned().size(), comm->getRank() == 0 ? 8 : 0);
}
TEST(SparseNativePeriodic, InconsistentDisplayImageRejectsCollectively)
{
    auto comm = Tpetra::getDefaultComm();
    auto fixture = periodic_input(true);
    if (comm->getRank() == 0)
    {
        auto found = std::find_if(fixture.input.faces.begin(), fixture.input.faces.end(),
            [](const auto& face) { return !face.exact_second_image.empty(); });
        ASSERT_NE(found, fixture.input.faces.end());
        found->second_image[0] += .125;
    }
    try
    {
        (void) MultiRegionMesh::from_sparse_native(std::move(fixture.input), comm);
        FAIL() << "Expected exact/display image rejection.";
    }
    catch (const std::invalid_argument& error)
    {
        if (comm->getRank() == 0)
            EXPECT_NE(std::string(error.what()).find("inconsistent exact/display image"), std::string::npos);
    }
}
TEST(SparseNativePeriodic, WrongSecondPhysicalNodeRejectsCollectively)
{
    auto comm = Tpetra::getDefaultComm();
    auto fixture = periodic_input(true);
    if (comm->getRank() == 0)
    {
        auto found = std::find_if(fixture.input.faces.begin(), fixture.input.faces.end(),
            [](const auto& face) { return !face.exact_second_image.empty(); });
        ASSERT_NE(found, fixture.input.faces.end());
        found->second_node_gids = found->override_nodes;
    }
    try
    {
        (void) MultiRegionMesh::from_sparse_native(std::move(fixture.input), comm);
        FAIL() << "Expected second physical node rejection.";
    }
    catch (const std::invalid_argument& error)
    {
        if (comm->getRank() == 0)
            EXPECT_NE(std::string(error.what()).find("second physical node proof"), std::string::npos);
    }
}
} // namespace
