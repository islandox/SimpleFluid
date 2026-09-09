/**
 * @file testMeshReorderingFactory.cc
 * @brief Tests for selected-first mesh layouts and range-backed solid views.
 */

#include <gtest/gtest.h>

#include "geometry/MeshReorderingFactory.hh"
#include "geometry/SolidSubdomain.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Access.hpp>
#include <Tpetra_CombineMode.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <set>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Factory = SimpleFluid::MeshReorderingFactory<Pack>;
using Subdomain = SimpleFluid::SolidSubdomain<Pack>;
using Cartesian = SimpleFluid::Meshes::OrthogonalCartesian3D;
using Cylindrical = SimpleFluid::Meshes::OrthogonalCylindrial3D;
using LocalOrdinal = Pack::local_ordinal_type;
using GlobalOrdinal = Pack::global_ordinal_type;
using Vec3 = Handle::Vec3;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

constexpr double tolerance = 1.0e-12;

LocalOrdinal local(size_t value)
{
    return static_cast<LocalOrdinal>(value);
}

bool in_box(const Vec3& center)
{
    return center.x >= -1.0 && center.x < 1.0 && center.y >= -1.0 && center.y < 1.0 && center.z >= 0.0 &&
           center.z < 2.0;
}

bool in_cylinder(const Vec3& center)
{
    return std::hypot(center.x, center.y) < 2.0 && center.z >= 0.0 && center.z < 2.0;
}

bool in_tube(const Vec3& center)
{
    const auto radius = std::hypot(center.x, center.y);
    return radius >= 2.0 && radius < 4.0;
}

SimpleFluid::SP<Handle> make_box_parent()
{
    auto mesh = std::make_shared<Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{
        {{-2.0, -1.0, 0.0, 1.0, 2.0}, {-2.0, -1.0, 0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}});
    return std::make_shared<Handle>(std::move(mesh));
}

SimpleFluid::SP<Handle> make_cylinder_parent()
{
    auto mesh = std::make_shared<Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{
        {{-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}, {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0}}});
    return std::make_shared<Handle>(std::move(mesh));
}

SimpleFluid::SP<Handle> make_tube_parent()
{
    constexpr auto pi = std::numbers::pi_v<double>;
    auto mesh = std::make_shared<Cylindrical>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{
        {{1.0, 2.0, 3.0, 4.0, 5.0}, {0.0, 0.5 * pi, pi, 1.5 * pi, 2.0 * pi}, {0.0, 1.0, 2.0}}});
    return std::make_shared<Handle>(std::move(mesh));
}

SimpleFluid::SP<Handle> make_partitioned_line_parent()
{
    auto mesh = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    return std::make_shared<Handle>(std::move(mesh));
}

struct CellIdentity
{
    GlobalOrdinal gid{};
    Vec3 centroid{};
    double volume = 0.0;
    bool owned = false;
    std::vector<GlobalOrdinal> face_gids;
};

std::map<GlobalOrdinal, CellIdentity> capture_cells(const Handle& mesh)
{
    std::map<GlobalOrdinal, CellIdentity> cells;
    for (size_t index = 0; index < mesh.num_local_cells(); ++index)
    {
        const auto cell = local(index);
        CellIdentity identity{.gid = mesh.cell_global_id(cell),
            .centroid = mesh.cell_centroid(cell),
            .volume = mesh.cell_volume(cell),
            .owned = mesh.is_owned_cell(cell)};
        for (const auto face : mesh.faces(cell))
        {
            identity.face_gids.push_back(mesh.face_global_id(face));
        }
        std::ranges::sort(identity.face_gids);
        EXPECT_TRUE(cells.emplace(mesh.cell_geometry_global_id(cell), std::move(identity)).second);
    }
    return cells;
}

void expect_cells_preserved(const std::map<GlobalOrdinal, CellIdentity>& before, const Handle& reordered)
{
    ASSERT_EQ(reordered.num_local_cells(), before.size());
    for (size_t index = 0; index < reordered.num_local_cells(); ++index)
    {
        const auto cell = local(index);
        const auto found = before.find(reordered.cell_geometry_global_id(cell));
        ASSERT_NE(found, before.end());
        const auto& expected = found->second;
        EXPECT_EQ(reordered.cell_global_id(cell), expected.gid);
        EXPECT_EQ(reordered.is_owned_cell(cell), expected.owned);
        EXPECT_DOUBLE_EQ(reordered.cell_volume(cell), expected.volume);
        const auto center = reordered.cell_centroid(cell);
        EXPECT_DOUBLE_EQ(center.x, expected.centroid.x);
        EXPECT_DOUBLE_EQ(center.y, expected.centroid.y);
        EXPECT_DOUBLE_EQ(center.z, expected.centroid.z);

        std::vector<GlobalOrdinal> faces;
        for (const auto face : reordered.faces(cell))
        {
            faces.push_back(reordered.face_global_id(face));
        }
        std::ranges::sort(faces);
        EXPECT_EQ(faces, expected.face_gids);
        EXPECT_EQ(reordered.overlap_cell_map()->getGlobalElement(cell), expected.gid);
    }
}

std::optional<LocalOrdinal> find_x_face(const Handle& mesh, double x)
{
    for (size_t index = 0; index < mesh.num_faces(); ++index)
    {
        const auto face = local(index);
        const auto center = mesh.face_centroid(face);
        const auto normal = mesh.face_normal(face);
        if (std::abs(center.x - x) < tolerance && std::abs(center.y - 0.5) < tolerance &&
            std::abs(center.z - 0.5) < tolerance && std::abs(std::abs(normal.x) - 1.0) < tolerance &&
            std::abs(normal.y) < tolerance && std::abs(normal.z) < tolerance)
        {
            return face;
        }
    }
    return std::nullopt;
}

void expect_selected_prefix(const Factory::SelectedCellLayout& layout, const std::function<bool(const Vec3&)>& selected)
{
    const auto& mesh = *layout.mesh;
    const auto owned = layout.selected_owned_range();
    const auto ghosts = layout.selected_ghost_range();
    EXPECT_EQ(owned, (Factory::LocalCellRange{0, layout.selected_owned_cells}));
    EXPECT_EQ(ghosts,
        (Factory::LocalCellRange{mesh.num_owned_cells(), mesh.num_owned_cells() + layout.selected_ghost_cells}));
    for (size_t index = 0; index < mesh.num_local_cells(); ++index)
    {
        const bool expected = owned.contains(index) || ghosts.contains(index);
        EXPECT_EQ(selected(mesh.cell_centroid(local(index))), expected) << "cell local ordinal " << index;
    }
}

} // namespace

TEST(MeshReorderingFactoryTest, LocalCellRangeUsesHalfOpenArithmetic)
{
    constexpr Factory::LocalCellRange range{3, 8};
    static_assert(range.size() == 5);
    static_assert(!range.empty());
    static_assert(!range.contains(2));
    static_assert(range.contains(3));
    static_assert(range.contains(7));
    static_assert(!range.contains(8));

    constexpr Factory::LocalCellRange empty{4, 4};
    static_assert(empty.empty());
    static_assert(empty.size() == 0);
}

TEST(MeshReorderingFactoryTest, ReturnsInputHandleWhenSelectionIsAlreadyAPrefix)
{
    auto parent = make_box_parent();
    const auto* parent_address = parent.get();
    const auto owned_cells = parent->num_owned_cells();
    const auto ghost_cells = parent->num_local_cells() - owned_cells;
    const auto layout =
        Factory::selected_cells_first(std::move(parent), [](GlobalOrdinal, const Vec3&) { return true; });

    EXPECT_FALSE(parent);
    EXPECT_EQ(layout.mesh.get(), parent_address);
    EXPECT_FALSE(layout.mesh->has_reordered_cells());
    EXPECT_EQ(layout.selected_owned_cells, owned_cells);
    EXPECT_EQ(layout.selected_ghost_cells, ghost_cells);
}

TEST(MeshReorderingFactoryTest, RejectsSharedHandleWhenPermutationIsRequired)
{
    auto parent = make_box_parent();
    const auto retained_alias = parent;

    EXPECT_THROW(static_cast<void>(Factory::selected_cells_first(
                     std::move(parent), [](GlobalOrdinal, const Vec3& center) { return in_box(center); })),
        std::invalid_argument);
    EXPECT_TRUE(parent);
    EXPECT_TRUE(retained_alias);
}

TEST(MeshReorderingFactoryTest, RejectsOverlapWithoutCommunicatorOwner)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0, 3.0, 4.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto parent = std::make_shared<Handle>(
        std::move(geometry), Handle::DistributionOptions{.ghost_layers = 1, .partition = 0, .partitions = 2});

    EXPECT_THROW(static_cast<void>(
                     Factory::selected_cells_first(std::move(parent), [](GlobalOrdinal, const Vec3&) { return true; })),
        std::invalid_argument);
}

TEST(MeshReorderingFactoryTest, BoxSelectionIsStableAndPreservesMeshIdentity)
{
    auto parent = make_box_parent();
    const auto* parent_address = parent.get();
    const auto before = capture_cells(*parent);

    const auto layout = Factory::selected_cells_first(
        std::move(parent), [](GlobalOrdinal, const Vec3& center) { return in_box(center); });

    ASSERT_TRUE(layout.mesh);
    EXPECT_FALSE(parent);
    EXPECT_EQ(layout.mesh.get(), parent_address);
    EXPECT_TRUE(layout.mesh->has_reordered_cells());
    EXPECT_EQ(layout.selected_owned_cells, 8U);
    EXPECT_EQ(layout.selected_ghost_cells, 0U);
    expect_selected_prefix(layout, in_box);
    expect_cells_preserved(before, *layout.mesh);

    const Subdomain solid(layout);
    ASSERT_EQ(solid.num_owned_cells(), 8U);
    ASSERT_EQ(solid.num_local_cells(), 8U);
    for (size_t index = 0; index < solid.num_local_cells(); ++index)
    {
        EXPECT_EQ(solid.parent_cell_lid(local(index)), local(index));
        EXPECT_EQ(solid.subdomain_cell_lid(local(index)), local(index));
        EXPECT_TRUE(in_box(solid.cell_centroid(local(index))));
    }
    for (size_t parent_lid = 8; parent_lid < layout.mesh->num_local_cells(); ++parent_lid)
    {
        EXPECT_EQ(solid.subdomain_cell_lid(local(parent_lid)), Subdomain::invalid_local_id());
    }
}

TEST(MeshReorderingFactoryTest, ReordersLegacyStorageWithoutChangingGeometryIdentity)
{
    const auto legacy = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_two_hex_database());
    auto parent = std::make_shared<Handle>(legacy);
    const auto* parent_address = parent.get();
    const auto before = capture_cells(*parent);

    const auto layout = Factory::selected_cells_first(
        std::move(parent), [](GlobalOrdinal, const Vec3& center) { return center.x > 1.0; });

    ASSERT_TRUE(layout.mesh);
    EXPECT_FALSE(parent);
    EXPECT_EQ(layout.mesh.get(), parent_address);
    EXPECT_TRUE(layout.mesh->is_stk());
    EXPECT_EQ(layout.selected_owned_cells, 1U);
    EXPECT_EQ(layout.selected_ghost_cells, 0U);
    EXPECT_GT(layout.mesh->cell_centroid(0).x, 1.0);
    expect_cells_preserved(before, *layout.mesh);

    const Subdomain solid(layout);
    ASSERT_EQ(solid.num_owned_cells(), 1U);
    EXPECT_EQ(solid.parent_cell_lid(0), 0);
    EXPECT_EQ(solid.cell_geometry_global_id(0), legacy->cell_global_id(1));
    ASSERT_EQ(solid.interface_faces().size(), 1U);
    EXPECT_DOUBLE_EQ(solid.face_normal(solid.interface_faces().front().face_lid).x, -1.0);
}

TEST(MeshReorderingFactoryTest, ReordersUnstructuredStorageAndIndexerTogether)
{
    const auto geometry = SimpleFluid::test::make_unstructured_hex_line(2);
    auto parent = std::make_shared<Handle>(geometry);
    const auto* parent_address = parent.get();
    const auto before = capture_cells(*parent);
    const auto expected_first_gid = parent->indexer().cell_global_id(1);

    const auto layout = Factory::selected_cells_first(
        std::move(parent), [](GlobalOrdinal, const Vec3& center) { return center.x > 1.0; });

    ASSERT_TRUE(layout.mesh);
    EXPECT_FALSE(parent);
    EXPECT_EQ(layout.mesh.get(), parent_address);
    EXPECT_FALSE(layout.mesh->is_stk());
    EXPECT_EQ(layout.selected_owned_cells, 1U);
    EXPECT_GT(layout.mesh->cell_centroid(0).x, 1.0);
    expect_cells_preserved(before, *layout.mesh);
    EXPECT_EQ(layout.mesh->indexer().cell_global_id(0), expected_first_gid);

    const Subdomain solid(layout);
    ASSERT_EQ(solid.num_owned_cells(), 1U);
    ASSERT_EQ(solid.interface_faces().size(), 1U);
    EXPECT_DOUBLE_EQ(solid.face_normal(solid.interface_faces().front().face_lid).x, -1.0);
}

TEST(MeshReorderingFactoryTest, SparseSelectionMapsOnlyRetainedFaces)
{
    SimpleFluid::ArrReal x_edges(1025);
    for (size_t edge = 0; edge < x_edges.size(); ++edge)
    {
        x_edges[edge] = static_cast<double>(edge);
    }
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{std::move(x_edges), {0.0, 1.0}, {0.0, 1.0}}});
    auto parent = std::make_shared<Handle>(std::move(geometry));
    const auto layout = Factory::selected_cells_first(
        std::move(parent), [](GlobalOrdinal, const Vec3& center) { return center.x >= 510.0 && center.x < 513.0; });
    const Subdomain solid(layout);

    ASSERT_EQ(layout.selected_owned_cells, 3U);
    ASSERT_EQ(solid.num_local_cells(), 3U);
    for (size_t index = 0; index < 3; ++index)
    {
        EXPECT_EQ(solid.parent_cell_lid(local(index)), local(index));
        EXPECT_EQ(solid.subdomain_cell_lid(local(index)), local(index));
    }
    for (size_t parent_lid = 3; parent_lid < layout.mesh->num_local_cells(); ++parent_lid)
    {
        EXPECT_EQ(solid.subdomain_cell_lid(local(parent_lid)), Subdomain::invalid_local_id());
    }

    size_t retained_faces = 0;
    size_t omitted_faces = 0;
    for (size_t parent_face = 0; parent_face < layout.mesh->num_faces(); ++parent_face)
    {
        const auto parent_face_lid = local(parent_face);
        const auto subdomain_face = solid.subdomain_face_lid(parent_face_lid);
        if (subdomain_face == Subdomain::invalid_local_id())
        {
            ++omitted_faces;
        }
        else
        {
            ++retained_faces;
            EXPECT_EQ(solid.parent_face_lid(subdomain_face), parent_face_lid);
        }
    }
    EXPECT_EQ(retained_faces, solid.num_faces());
    EXPECT_GT(omitted_faces, 1000U);
}

TEST(MeshReorderingFactoryTest, CylinderSelectionFormsOneOutwardInterface)
{
    auto parent = make_cylinder_parent();
    const auto layout = Factory::selected_cells_first(
        std::move(parent), [](GlobalOrdinal, const Vec3& center) { return in_cylinder(center); });
    expect_selected_prefix(layout, in_cylinder);
    ASSERT_EQ(layout.selected_owned_cells, 24U);

    const Subdomain solid(layout);
    ASSERT_EQ(solid.interface_faces().size(), 32U);
    for (const auto& interface : solid.interface_faces())
    {
        const auto outward = layout.mesh->cell_centroid(interface.outside_parent_cell_lid) -
                             solid.cell_centroid(interface.solid_cell_lid);
        EXPECT_GT(solid.face_normal(interface.face_lid).dot(outward), 0.0);
    }
}

TEST(MeshReorderingFactoryTest, TubeSelectionPreservesPeriodicTopologyAndWallNormals)
{
    auto parent = make_tube_parent();
    const auto layout = Factory::selected_cells_first(
        std::move(parent), [](GlobalOrdinal, const Vec3& center) { return in_tube(center); });
    expect_selected_prefix(layout, in_tube);
    ASSERT_EQ(layout.selected_owned_cells, 16U);

    const Subdomain solid(layout);
    ASSERT_EQ(solid.interface_faces().size(), 16U);
    size_t inner_faces = 0;
    size_t outer_faces = 0;
    for (const auto& interface : solid.interface_faces())
    {
        const auto solid_center = solid.cell_centroid(interface.solid_cell_lid);
        const auto outside_center = layout.mesh->cell_centroid(interface.outside_parent_cell_lid);
        const auto outward = outside_center - solid_center;
        EXPECT_GT(solid.face_normal(interface.face_lid).dot(outward), 0.0);
        if (std::hypot(outside_center.x, outside_center.y) < 2.0)
        {
            ++inner_faces;
        }
        else
        {
            ++outer_faces;
        }
    }
    EXPECT_EQ(inner_faces, 8U);
    EXPECT_EQ(outer_faces, 8U);

    std::set<std::string> boundary_names;
    for (const auto& [batch_id, batch] : solid.boundary_batches())
    {
        static_cast<void>(batch);
        boundary_names.insert(solid.boundary_batch_name(batch_id));
    }
    EXPECT_EQ(boundary_names, (std::set<std::string>{"solid_interface", "zmin", "zmax"}));
}

TEST(MeshReorderingFactoryMultiRankTest, SolidOwnsPartitionCutFaceOnSelectedCellRank)
{
    SKIP_SINGLE_RANK(SolidOwnsPartitionCutFaceOnSelectedCellRank);
    auto parent = make_partitioned_line_parent();
    const auto comm = parent->owned_cell_map()->getComm();
    ASSERT_EQ(comm->getSize(), 2);
    const auto rank = comm->getRank();
    const auto original_cut = find_x_face(*parent, 2.0);
    ASSERT_TRUE(original_cut.has_value());
    const auto cut_gid = parent->face_global_id(*original_cut);
    EXPECT_EQ(parent->is_owned_face(*original_cut), rank == 0);

    const auto layout = Factory::selected_cells_first(
        std::move(parent), [](GlobalOrdinal, const Vec3& center) { return center.x >= 2.0 && center.x < 3.0; });
    ASSERT_TRUE(layout.mesh);
    EXPECT_EQ(layout.selected_owned_cells, rank == 1 ? 1U : 0U);
    EXPECT_EQ(layout.selected_ghost_cells, rank == 0 ? 1U : 0U);
    EXPECT_EQ(layout.selected_owned_range(), (Factory::LocalCellRange{0, rank == 1 ? 1U : 0U}));
    EXPECT_EQ(layout.selected_ghost_range(), (Factory::LocalCellRange{2, rank == 0 ? 3U : 2U}));
    expect_selected_prefix(layout, [](const Vec3& center) { return center.x >= 2.0 && center.x < 3.0; });

    // Reordering changes only cell LIDs. Parent face identity and ownership stay
    // intact; SolidSubdomain deliberately assigns a cut face to its solid cell.
    const auto reordered_cut = find_x_face(*layout.mesh, 2.0);
    ASSERT_TRUE(reordered_cut.has_value());
    EXPECT_EQ(layout.mesh->face_global_id(*reordered_cut), cut_gid);
    EXPECT_EQ(layout.mesh->is_owned_face(*reordered_cut), rank == 0);

    const Subdomain solid(layout);
    EXPECT_EQ(solid.num_owned_cells(), rank == 1 ? 1U : 0U);
    ASSERT_EQ(solid.num_local_cells(), 1U);
    EXPECT_EQ(solid.parent_cell_lid(0), rank == 1 ? 0 : 2);
    EXPECT_EQ(solid.subdomain_cell_lid(rank == 1 ? 0 : 2), 0);

    const auto solid_cut = solid.subdomain_face_lid(*reordered_cut);
    ASSERT_NE(solid_cut, Subdomain::invalid_local_id());
    EXPECT_TRUE(solid.is_interface_face(solid_cut));
    EXPECT_EQ(solid.is_owned_face(solid_cut), rank == 1);
    const int local_owner = solid.is_owned_face(solid_cut) ? 1 : 0;
    int global_owners = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_owner, &global_owners);
    EXPECT_EQ(global_owners, 1);

    const auto interface_iter = std::find_if(solid.interface_faces().begin(), solid.interface_faces().end(),
        [&](const auto& candidate) { return candidate.face_lid == solid_cut; });
    ASSERT_NE(interface_iter, solid.interface_faces().end());
    const auto& interface = *interface_iter;
    const auto outward =
        layout.mesh->cell_centroid(interface.outside_parent_cell_lid) - solid.cell_centroid(interface.solid_cell_lid);
    EXPECT_GT(solid.face_normal(solid_cut).dot(outward), 0.0);

    typename Pack::vector_type owned(layout.mesh->owned_cell_map(), true);
    typename Pack::vector_type overlap(layout.mesh->overlap_cell_map(), true);
    for (size_t index = 0; index < layout.mesh->num_owned_cells(); ++index)
    {
        const auto cell = local(index);
        owned.replaceLocalValue(cell, static_cast<Pack::scalar_type>(layout.mesh->cell_geometry_global_id(cell)));
    }
    typename Pack::import_type importer(layout.mesh->owned_cell_map(), layout.mesh->overlap_cell_map());
    overlap.doImport(owned, importer, Tpetra::REPLACE);
    const auto values = overlap.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t index = 0; index < layout.mesh->num_local_cells(); ++index)
    {
        EXPECT_DOUBLE_EQ(
            values(local(index), 0), static_cast<double>(layout.mesh->cell_geometry_global_id(local(index))));
    }
}

TEST(MeshReorderingFactoryMultiRankTest, SupportsRankWithoutSelectedOwnedOrGhostCells)
{
    SKIP_SINGLE_RANK(SupportsRankWithoutSelectedOwnedOrGhostCells);
    auto parent = make_partitioned_line_parent();
    const auto comm = parent->owned_cell_map()->getComm();
    ASSERT_EQ(comm->getSize(), 2);
    const auto rank = comm->getRank();

    const auto layout = Factory::selected_cells_first(
        std::move(parent), [](GlobalOrdinal, const Vec3& center) { return center.x >= 3.0; });
    EXPECT_EQ(layout.selected_owned_cells, rank == 1 ? 1U : 0U);
    EXPECT_EQ(layout.selected_ghost_cells, 0U);

    const Subdomain solid(layout);
    EXPECT_EQ(solid.num_owned_cells(), rank == 1 ? 1U : 0U);
    EXPECT_EQ(solid.num_local_cells(), rank == 1 ? 1U : 0U);
    EXPECT_EQ(solid.owned_cell_map()->getGlobalNumElements(), 1U);
    EXPECT_EQ(solid.overlap_cell_map()->getGlobalNumElements(), 1U);
    if (rank == 0)
    {
        EXPECT_TRUE(layout.selected_owned_range().empty());
        EXPECT_TRUE(layout.selected_ghost_range().empty());
        EXPECT_EQ(solid.num_faces(), 0U);
    }
}
