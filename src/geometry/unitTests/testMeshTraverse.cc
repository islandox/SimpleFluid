/** @file testMeshTraverse.cc
 * @brief Selection, ordering, geometry lifetime, and partition contracts for mesh traverses.
 */
#include <gtest/gtest.h>

#include "geometry/MeshReorderingFactory.hh"
#include "geometry/MeshTraverse.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Cartesian = Meshes::OrthogonalCartesian3D;
using Local = Pack::local_ordinal_type;

static_assert(!std::is_constructible_v<MeshTraversePlan<Handle>, Handle&&>);
static_assert(!std::is_constructible_v<MeshTraverseExecution<Handle>, MeshTraversePlan<Handle>&&>);
static_assert(!std::is_constructible_v<FaceBatchProxy<Handle>, MeshTraverseExecution<Handle>&&>);
static_assert(!std::is_constructible_v<CellFaceBatchProxy<Handle>, const MeshTraverseExecution<Handle>&, Local>);

testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

auto cartesian(size_t nx, size_t ny, size_t nz)
{
    Vec3D<ArrReal> edges;
    for (const auto [axis, count] :
         {std::pair{0U, nx}, std::pair{1U, ny}, std::pair{2U, nz}})
    {
        edges[axis].resize(count + 1);
        std::iota(edges[axis].begin(), edges[axis].end(), 0.0);
    }
    return std::make_shared<Cartesian>(edges);
}

std::vector<Local> ordinals(size_t count)
{
    std::vector<Local> result(count);
    std::iota(result.begin(), result.end(), Local{0});
    return result;
}

template<class Selection>
std::vector<Local> selected_ids(const Selection& selection)
{
    std::vector<Local> ids;
    selection.apply([&](const auto& entity) { ids.push_back(entity.id()); });
    return ids;
}

void expect_vector_equal(const auto& actual, const auto& expected)
{
    EXPECT_DOUBLE_EQ(actual.x, expected.x);
    EXPECT_DOUBLE_EQ(actual.y, expected.y);
    EXPECT_DOUBLE_EQ(actual.z, expected.z);
}

/** Compare against canonical topology while keeping field ordinals explicit. */
void expect_handle_selections(const Handle& mesh)
{
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto traverse = plan.acquire();
    std::vector<Local> interior_cells, boundary_cells, interior_faces, boundary_faces;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto cell = static_cast<Local>(c);
        bool boundary = false;
        for (const auto face : mesh.faces(cell))
            boundary = boundary || mesh.is_geometry_exterior_face(face);
        (boundary ? boundary_cells : interior_cells).push_back(cell);
    }
    for (size_t f = 0; f < mesh.num_owned_faces(); ++f)
    {
        const auto face = static_cast<Local>(f);
        (mesh.is_geometry_exterior_face(face) ? boundary_faces : interior_faces).push_back(face);
    }

    EXPECT_EQ(selected_ids(traverse.cells().all()), ordinals(mesh.num_owned_cells()));
    EXPECT_EQ(selected_ids(traverse.cells().interior()), interior_cells);
    EXPECT_EQ(selected_ids(traverse.cells().boundary()), boundary_cells);
    EXPECT_EQ(selected_ids(traverse.faces().all()), ordinals(mesh.num_owned_faces()));
    EXPECT_EQ(selected_ids(traverse.faces().interior()), interior_faces);
    EXPECT_EQ(selected_ids(traverse.faces().boundary()), boundary_faces);
    EXPECT_EQ(selected_ids(traverse.cells().all().local()), ordinals(mesh.num_local_cells()));
    EXPECT_EQ(selected_ids(traverse.faces().all().local()), ordinals(mesh.num_faces()));
    std::vector<Local> local_interior_faces, local_boundary_faces;
    for (size_t f = 0; f < mesh.num_faces(); ++f)
    {
        const auto face = static_cast<Local>(f);
        (mesh.is_geometry_exterior_face(face) ? local_boundary_faces : local_interior_faces).push_back(face);
    }
    EXPECT_EQ(selected_ids(traverse.faces().interior().local()), local_interior_faces);
    EXPECT_EQ(selected_ids(traverse.faces().boundary().local()), local_boundary_faces);

    traverse.cells().all().apply([&](const auto& cell)
    {
        using Ref = std::remove_cvref_t<decltype(cell)>;
        EXPECT_EQ(Ref::is_interior, std::ranges::binary_search(interior_cells, cell.id()));
        EXPECT_DOUBLE_EQ(cell.volume(), mesh.cell_volume(cell.id()));
        expect_vector_equal(cell.centroid(), mesh.cell_centroid(cell.id()));
    });
    traverse.faces().all().apply([&](const auto& face)
    {
        using Ref = std::remove_cvref_t<decltype(face)>;
        EXPECT_EQ(Ref::is_interior, !mesh.is_geometry_exterior_face(face.id()));
        EXPECT_EQ(face.owner(), mesh.owner_cell(face.id()));
        if constexpr (Ref::is_interior)
            EXPECT_EQ(face.neighbor(), mesh.neighbor_cell(face.id()));
        else
            EXPECT_EQ(face.boundary_id(), mesh.boundary_id(face.id()));
        EXPECT_DOUBLE_EQ(face.area(), mesh.face_area(face.id()));
        expect_vector_equal(face.centroid(), mesh.face_centroid(face.id()));
        expect_vector_equal(face.normal(), mesh.face_normal(face.id()));
    });

    std::vector<Local> from_batches;
    traverse.faces().all().visit_batches([&](const auto& batch)
    {
        const auto before = from_batches.size();
        using Batch = std::remove_cvref_t<decltype(batch)>;
        batch.apply([&](const auto& face)
        {
            using Ref = std::remove_cvref_t<decltype(face)>;
            static_assert(Batch::is_interior == Ref::is_interior);
            from_batches.push_back(face.id());
        });
        EXPECT_GT(batch.size(), 0U);
        EXPECT_EQ(from_batches.size() - before, batch.size());
    });
    EXPECT_EQ(from_batches, ordinals(mesh.num_owned_faces()));

    auto independent = selected_ids(traverse.faces().all().independent());
    std::ranges::sort(independent);
    EXPECT_EQ(independent, ordinals(mesh.num_owned_faces()));
    for (const auto& [patch, ignored] : mesh.boundary_batches())
    {
        std::vector<Local> expected;
        for (const auto face : boundary_faces)
            if (mesh.boundary_id(face) == patch) expected.push_back(face);
        EXPECT_EQ(selected_ids(traverse.faces().boundary(patch)), expected);
    }
    EXPECT_TRUE(selected_ids(traverse.faces().boundary(987654)).empty());
}
} // namespace

TEST(MeshTraverseTest, SelectionsCoverCellsAndFacesAndDeduplicateBoundaryCorners)
{
    Handle mesh(cartesian(3, 3, 3));
    expect_handle_selections(mesh);
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto traverse = plan.acquire();
    EXPECT_EQ(selected_ids(traverse.cells().interior()), (std::vector<Local>{13}));
    EXPECT_EQ(selected_ids(traverse.cells().boundary()).size(), 26U);
    EXPECT_EQ(selected_ids(traverse.faces().interior()).size(), 54U);
    EXPECT_EQ(selected_ids(traverse.faces().boundary()).size(), 54U);
}

TEST(MeshTraverseTest, DefaultOrderPreservesReadsOfEarlierWritesAcrossCategoryChanges)
{
    Handle mesh(cartesian(3, 3, 3));
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto traverse = plan.acquire();
    std::vector<double> cells(mesh.num_owned_cells(), -1.0);
    traverse.cells().all().apply([&](const auto& cell)
    {
        const auto id = static_cast<size_t>(cell.id());
        cells[id] = id == 0 ? 1.0 : cells[id - 1] + 1.0;
    });
    for (size_t i = 0; i < cells.size(); ++i) EXPECT_DOUBLE_EQ(cells[i], i + 1.0);
    std::vector<double> faces(mesh.num_owned_faces(), -1.0);
    traverse.faces().all().apply([&](const auto& face)
    {
        const auto id = static_cast<size_t>(face.id());
        faces[id] = id == 0 ? 1.0 : faces[id - 1] + 1.0;
    });
    for (size_t i = 0; i < faces.size(); ++i) EXPECT_DOUBLE_EQ(faces[i], i + 1.0);
}

TEST(MeshTraverseTest, SupportsMoveOnlyOperationsAndBatchVisitors)
{
    Handle mesh(cartesian(3, 3, 3));
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto traverse = plan.acquire();
    size_t visited = 0;
    auto operation = [state = std::make_unique<size_t>(0), &visited](const auto&) mutable
    {
        visited = ++*state;
    };
    static_assert(!std::is_copy_constructible_v<decltype(operation)>);
    traverse.cells().all().apply(std::move(operation));
    EXPECT_EQ(visited, mesh.num_owned_cells());

    visited = 0;
    auto visitor = [state = std::make_unique<size_t>(0), &visited](const auto& batch) mutable
    {
        *state += batch.size();
        visited = *state;
    };
    static_assert(!std::is_copy_constructible_v<decltype(visitor)>);
    traverse.faces().all().visit_batches(std::move(visitor));
    EXPECT_EQ(visited, mesh.num_owned_faces());
}

TEST(MeshTraverseTest, InvalidatedOrCountChangedPlansRejectAcquisition)
{
    auto geometry = cartesian(1, 1, 1);
    auto plan = make_mesh_traverse_plan(*geometry);
    {
        const auto traverse = plan.acquire();
        const auto cells = traverse.cells().all();
        const auto faces = traverse.faces().all();
        plan.invalidate();
        EXPECT_THROW(static_cast<void>(traverse.cells()), std::logic_error);
        EXPECT_THROW(static_cast<void>(traverse.faces()), std::logic_error);
        EXPECT_THROW(static_cast<void>(cells.size()), std::logic_error);
        size_t callbacks = 0;
        EXPECT_THROW(cells.apply([&](const auto&) { ++callbacks; }), std::logic_error);
        EXPECT_THROW(faces.visit_batches([&](const auto&) { ++callbacks; }), std::logic_error);
        EXPECT_EQ(callbacks, 0U);
    }
    EXPECT_THROW(static_cast<void>(plan.acquire()), std::logic_error);

    const auto replacement_plan = make_mesh_traverse_plan(*geometry);
    EXPECT_NO_THROW(static_cast<void>(replacement_plan.acquire()));
    *geometry = Cartesian(Vec3D<ArrReal>{{{0, 1, 2}, {0, 1}, {0, 1}}});
    EXPECT_THROW(static_cast<void>(replacement_plan.acquire()), std::logic_error);
    const auto rebuilt = make_mesh_traverse_plan(*geometry);
    const auto traverse = rebuilt.acquire();
    EXPECT_EQ(selected_ids(traverse.cells().all()), ordinals(2));
}

TEST(MeshTraverseTest, SameCountBackendReplacementRejectsThePreviousPlan)
{
    Handle mesh(cartesian(1, 1, 1));
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto replacement = cartesian(1, 1, 1);
    {
        const auto traverse = plan.acquire();
        const auto faces = traverse.faces().all();
        mesh = Handle(replacement);
        EXPECT_THROW(static_cast<void>(traverse.faces()), std::logic_error);
        EXPECT_THROW(static_cast<void>(faces.size()), std::logic_error);
        size_t callbacks = 0;
        EXPECT_THROW(faces.apply([&](const auto&) { ++callbacks; }), std::logic_error);
        EXPECT_EQ(callbacks, 0U);
    }
    EXPECT_THROW(static_cast<void>(plan.acquire()), std::logic_error);
    const auto rebuilt = make_mesh_traverse_plan(mesh);
    const auto traverse = rebuilt.acquire();
    EXPECT_EQ(selected_ids(traverse.cells().all()), ordinals(1));
}

TEST(MeshTraverseTest, ReplacedCompositeBackendOutlivesItsActiveExecutionLease)
{
    std::weak_ptr<Meshes::MultiRegionMesh> previous_backend;
    Handle mesh([&]
    {
        auto geometry = test::two_regions();
        previous_backend = geometry;
        return geometry;
    }());
    const auto plan = make_mesh_traverse_plan(mesh);
    ASSERT_FALSE(previous_backend.expired());
    {
        const auto traverse = plan.acquire();
        const auto faces = traverse.faces().all();
        mesh = Handle(test::two_regions());
        EXPECT_FALSE(previous_backend.expired());
        EXPECT_THROW(static_cast<void>(traverse.faces()), std::logic_error);
        EXPECT_THROW(static_cast<void>(faces.size()), std::logic_error);
        size_t callbacks = 0;
        EXPECT_THROW(faces.apply([&](const auto&) { ++callbacks; }), std::logic_error);
        EXPECT_EQ(callbacks, 0U);
    }
    EXPECT_TRUE(previous_backend.expired());
    EXPECT_THROW(static_cast<void>(plan.acquire()), std::logic_error);
    const auto rebuilt = make_mesh_traverse_plan(mesh);
    const auto traverse = rebuilt.acquire();
    EXPECT_EQ(selected_ids(traverse.cells().all()), ordinals(mesh.num_owned_cells()));
}

TEST(MeshTraverseTest, ThinAndSingleCellMeshesHaveNoInteriorCells)
{
    for (const auto& geometry : {cartesian(1, 1, 1), cartesian(4, 1, 2)})
    {
        Handle mesh(geometry);
        expect_handle_selections(mesh);
        const auto plan = make_mesh_traverse_plan(mesh);
        const auto traverse = plan.acquire();
        EXPECT_TRUE(selected_ids(traverse.cells().interior()).empty());
        EXPECT_EQ(selected_ids(traverse.cells().boundary()).size(), mesh.num_owned_cells());
    }
}

TEST(MeshTraverseTest, NativeMeshesPreserveLocalOrdinalsAndAcceptAnEmptyPartition)
{
    const auto geometry = cartesian(3, 3, 3);
    const auto plan = make_mesh_traverse_plan(*geometry);
    const auto traverse = plan.acquire();
    EXPECT_EQ(selected_ids(traverse.cells().all()), ordinals(27));
    EXPECT_EQ(selected_ids(traverse.cells().interior()), (std::vector<Local>{13}));
    traverse.cells().all().apply([&](const auto& cell)
    {
        const auto native_id = geometry->cell_id(cell.id());
        EXPECT_DOUBLE_EQ(cell.volume(), geometry->cell_volume(native_id));
        expect_vector_equal(cell.centroid(), geometry->cell_centroid(native_id));
    });

    const auto empty = Meshes::UnstructuredMesh::empty_partition();
    const auto empty_plan = make_mesh_traverse_plan(empty);
    const auto empty_traverse = empty_plan.acquire();
    EXPECT_TRUE(selected_ids(empty_traverse.cells().all()).empty());
    EXPECT_TRUE(selected_ids(empty_traverse.cells().interior()).empty());
    EXPECT_TRUE(selected_ids(empty_traverse.cells().boundary()).empty());
    EXPECT_TRUE(selected_ids(empty_traverse.faces().all()).empty());
    EXPECT_TRUE(selected_ids(empty_traverse.faces().interior()).empty());
    EXPECT_TRUE(selected_ids(empty_traverse.faces().boundary()).empty());
    size_t batches = 0;
    empty_traverse.cells().all().visit_batches([&](const auto&) { ++batches; });
    empty_traverse.faces().all().visit_batches([&](const auto&) { ++batches; });
    EXPECT_EQ(batches, 0U);
}

TEST(MeshTraverseTest, LegacyAndPartitionedWrappersExposeFieldOrdinalsAndGeometry)
{
    const auto legacy = test::build_mesh<Pack>(test::make_two_hex_database());
    auto check = [](const auto& mesh)
    {
        const auto plan = make_mesh_traverse_plan(mesh);
        const auto traverse = plan.acquire();
        EXPECT_EQ(selected_ids(traverse.cells().all()), ordinals(2));
        EXPECT_EQ(selected_ids(traverse.faces().all()), ordinals(11));
        EXPECT_EQ(selected_ids(traverse.faces().interior()).size(), 1U);
        EXPECT_EQ(selected_ids(traverse.faces().boundary()).size(), 10U);
        EXPECT_NO_THROW(traverse.faces().all().require_local_adjacency());
        traverse.cells().all().apply([&](const auto& cell)
        {
            EXPECT_DOUBLE_EQ(cell.volume(), mesh.cell_volume(cell.id()));
            expect_vector_equal(cell.centroid(), mesh.cell_centroid(cell.id()));
        });
        traverse.faces().all().apply([&](const auto& face)
        {
            EXPECT_DOUBLE_EQ(face.area(), mesh.face_area(face.id()));
            EXPECT_EQ(face.owner(), mesh.owner_cell(face.id()));
            if constexpr (std::remove_cvref_t<decltype(face)>::is_interior)
                EXPECT_EQ(face.neighbor(), mesh.neighbor_cell(face.id()));
        });
    };
    check(*legacy);

    const auto native = test::make_unstructured_hex_line(2);
    using Partitioned = Meshes::PartitionedMesh<Meshes::UnstructuredMesh, Pack>;
    using Indexer = Partitioned::indexer_type;
    std::vector<Indexer::cell_id_t> cells;
    std::vector<Indexer::face_id_t> faces;
    std::vector<Indexer::node_id_t> nodes;
    for (size_t c = 0; c < native->num_cells(); ++c) cells.push_back(native->cell_id(c));
    for (size_t f = 0; f < native->num_faces(); ++f) faces.push_back(native->face_id(f));
    for (size_t n = 0; n < native->num_nodes(); ++n) nodes.push_back(native->node_id(n));
    Partitioned partition(native, Indexer(std::move(cells), {}, std::move(faces), {}, std::move(nodes)));
    check(partition);
}

TEST(MeshTraverseTest, UntaggedPhysicalExteriorStillBelongsToBoundarySelection)
{
    using U = Meshes::UnstructuredMesh;
    auto geometry = std::make_shared<U>(
        Arr<U::Vec3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                     {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}},
        Arr<U::CellDefinition>{{U::CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}}});
    Handle mesh(geometry);
    expect_handle_selections(mesh);
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto traverse = plan.acquire();
    EXPECT_EQ(selected_ids(traverse.faces().boundary()).size(), 6U);
    EXPECT_TRUE(selected_ids(traverse.faces().interior()).empty());
    traverse.faces().boundary().apply([](const auto& face)
    {
        EXPECT_EQ(face.boundary_id(), Handle::invalid_boundary_id);
    });
}

TEST(MeshTraverseTest, FullyPeriodicMeshHasOnlyInteriorCellsAndFaces)
{
    auto geometry = std::make_shared<Cartesian>(
        Vec3D<ArrReal>{{{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}}},
        Vec3D<bool>{{true, true, true}});
    Handle mesh(geometry);
    expect_handle_selections(mesh);
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto traverse = plan.acquire();
    EXPECT_TRUE(selected_ids(traverse.cells().boundary()).empty());
    EXPECT_TRUE(selected_ids(traverse.faces().boundary()).empty());
    EXPECT_EQ(selected_ids(traverse.cells().interior()).size(), 27U);
    EXPECT_EQ(selected_ids(traverse.faces().interior()).size(), mesh.num_owned_faces());
}

TEST(MeshTraverseTest, ReorderedCellsUseFieldOrdinalsForSelectionsAndFaceAdjacency)
{
    auto parent = std::make_shared<Handle>(cartesian(3, 3, 3));
    const auto old_plan = make_mesh_traverse_plan(*parent);
    auto layout = MeshReorderingFactory<Pack>::selected_cells_first(
        std::move(parent), [](auto, const auto& center)
        { return center.x == 1.5 && center.y == 1.5 && center.z == 1.5; });
    ASSERT_TRUE(layout.mesh->has_reordered_cells());
    EXPECT_THROW(static_cast<void>(old_plan.acquire()), std::logic_error);
    expect_handle_selections(*layout.mesh);
    const auto plan = make_mesh_traverse_plan(*layout.mesh);
    const auto traverse = plan.acquire();
    EXPECT_EQ(selected_ids(traverse.cells().interior()), (std::vector<Local>{0}));
}

TEST(MeshTraverseTest, CompositeInterfacesUseCanonicalFacesWithoutConnectivityMaterialization)
{
    for (const auto& geometry :
         {test::coarse_fine_regions(), test::periodic_regions(), test::independent_extruded_regions()})
    {
        Handle mesh(geometry);
        expect_handle_selections(mesh);
        const auto plan = make_mesh_traverse_plan(mesh);
        const auto traverse = plan.acquire();
        size_t interfaces = 0;
        traverse.faces().interior().apply([&](const auto& face)
        {
            EXPECT_NE(face.owner(), face.neighbor());
            EXPECT_GT(face.area(), 0.0);
            ++interfaces;
        });
        EXPECT_GT(interfaces, 0U);
        EXPECT_EQ(mesh.connectivity_storage_bytes(), 0U);
        EXPECT_FALSE(mesh.has_materialized_indexer());
    }
}

TEST(MeshTraverseTest, ReusedPlanRefreshesGeometryAcrossALETrialAndRollback)
{
    auto mesh = std::make_shared<Handle>(test::two_regions());
    const auto plan = make_mesh_traverse_plan(*mesh);
    PlanarALEMeshMotion<Pack> motion(mesh);
    auto capture = [&]
    {
        std::vector<double> volumes;
        const auto traverse = plan.acquire();
        traverse.cells().all().apply([&](const auto& cell)
        {
            volumes.push_back(cell.volume());
            EXPECT_DOUBLE_EQ(cell.volume(), mesh->cell_volume(cell.id()));
            expect_vector_equal(cell.centroid(), mesh->cell_centroid(cell.id()));
        });
        traverse.faces().all().apply([&](const auto& face)
        {
            EXPECT_DOUBLE_EQ(face.area(), mesh->face_area(face.id()));
            expect_vector_equal(face.centroid(), mesh->face_centroid(face.id()));
        });
        return volumes;
    };
    const auto before = capture();
    {
        const auto traverse = plan.acquire();
        EXPECT_THROW(motion.begin_trial(2.0, 0.1), std::logic_error);
        EXPECT_FALSE(motion.has_active_trial());
    }
    motion.begin_trial(2.0, 0.1);
    const auto trial = capture();
    ASSERT_EQ(trial.size(), before.size());
    for (size_t i = 0; i < before.size(); ++i) EXPECT_DOUBLE_EQ(trial[i], 2.0 * before[i]);
    motion.rollback_trial();
    EXPECT_EQ(capture(), before);
}

TEST(MeshTraverseTest, OptionalCellFaceBatchesPreserveCanonicalFineSubfaceIncidence)
{
    Handle mesh(test::coarse_fine_regions());
    {
        const auto plan = make_mesh_traverse_plan(mesh);
        const auto traverse = plan.acquire();
        EXPECT_THROW(static_cast<void>(traverse.cell_faces(0)), std::logic_error);
    }
    const auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    const auto traverse = plan.acquire();
    EXPECT_EQ(mesh.faces(0).size(), 7U);
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto cell = static_cast<Local>(c);
        std::vector<Local> expected, interior, boundary;
        for (const auto face : mesh.faces(cell))
        {
            expected.push_back(face);
            (mesh.is_geometry_exterior_face(face) ? boundary : interior).push_back(face);
        }
        EXPECT_EQ(selected_ids(traverse.cell_faces(cell).all()), expected);
        EXPECT_EQ(selected_ids(traverse.cell_faces(cell).interior()), interior);
        EXPECT_EQ(selected_ids(traverse.cell_faces(cell).boundary()), boundary);
        traverse.cell_faces(cell).all().apply([&](const auto& face)
        {
            EXPECT_EQ(face.cell_id(), cell);
            EXPECT_EQ(face.orientation(), mesh.owner_cell(face.id()) == cell ? 1 : -1);
            EXPECT_DOUBLE_EQ(face.area(), mesh.face_area(face.id()));
        });
    }
    EXPECT_EQ(mesh.connectivity_storage_bytes(), 0U);
    EXPECT_FALSE(mesh.has_materialized_indexer());
}

TEST(MeshTraverseMultiRankTest, ZeroHaloDoesNotTurnPartitionInterfacesIntoPhysicalBoundaries)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two ranks.";
    Handle mesh(cartesian(4, 3, 3), Handle::DistributionOptions{.ghost_layers = 0});
    expect_handle_selections(mesh);
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto traverse = plan.acquire();
    int absent_adjacency = 0;
    traverse.faces().interior().local().apply([&](const auto& face)
    {
        EXPECT_FALSE(mesh.is_geometry_exterior_face(face.id()));
        if (face.owner() == Handle::invalid_local_id() || face.neighbor() == Handle::invalid_local_id())
            ++absent_adjacency;
    });
    int global_absent = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &absent_adjacency, &global_absent);
    EXPECT_GT(global_absent, 0);
    EXPECT_THROW(traverse.faces().interior().local().require_local_adjacency(), std::invalid_argument);
    EXPECT_NO_THROW(traverse.faces().boundary().local().require_local_adjacency());
    int interior = static_cast<int>(selected_ids(traverse.cells().interior()).size());
    int global_interior = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &interior, &global_interior);
    EXPECT_EQ(global_interior, 2);

    const auto incidence_plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    const auto incidence = incidence_plan.acquire();
    int unowned_incidence = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto cell = static_cast<Local>(c);
        std::vector<Local> expected;
        for (const auto face : mesh.faces(cell))
        {
            expected.push_back(face);
            if (!mesh.is_owned_face(face)) ++unowned_incidence;
        }
        EXPECT_EQ(selected_ids(incidence.cell_faces(cell).all()), expected);
    }
    int global_unowned = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &unowned_incidence, &global_unowned);
    EXPECT_GT(global_unowned, 0);
}

TEST(MeshTraverseMultiRankTest, OwnedAndLocalSelectionsHandleGhostsAndEmptyRanks)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two ranks.";
    const auto geometry = cartesian(4, 3, 3);
    Handle with_ghosts(geometry);
    expect_handle_selections(with_ghosts);
    EXPECT_GT(with_ghosts.num_local_cells(), with_ghosts.num_owned_cells());
    {
        const auto plan = make_mesh_traverse_plan(with_ghosts);
        const auto traverse = plan.acquire();
        EXPECT_NO_THROW(traverse.faces().all().require_local_adjacency());
        std::vector<Local> interior, boundary;
        for (size_t c = 0; c < with_ghosts.num_local_cells(); ++c)
        {
            const auto cell = static_cast<Local>(c);
            const auto native = geometry->cell_id(with_ghosts.cell_geometry_global_id(cell));
            bool exterior = false;
            for (const auto face : geometry->faces(native))
                exterior = exterior || geometry->is_exterior_face(face);
            (exterior ? boundary : interior).push_back(cell);
        }
        EXPECT_EQ(selected_ids(traverse.cells().interior().local()), interior);
        EXPECT_EQ(selected_ids(traverse.cells().boundary().local()), boundary);
    }

    Handle sparse(cartesian(1, 1, 1),
                  Handle::DistributionOptions{.ghost_layers = 0, .allow_empty_partitions = true});
    expect_handle_selections(sparse);
    int empty = sparse.num_owned_cells() == 0 ? 1 : 0;
    int empty_ranks = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &empty_ranks);
    EXPECT_EQ(empty_ranks, 1);
}
