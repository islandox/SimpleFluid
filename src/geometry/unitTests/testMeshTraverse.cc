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
#include <cmath>
#include <map>
#include <limits>
#include <string>
#include <tuple>
#include <memory>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <type_traits>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Cartesian = Meshes::OrthogonalCartesian3D;
using Cylindrical = Meshes::OrthogonalCylindrical3D;
using Local = Pack::local_ordinal_type;

static_assert(!std::is_constructible_v<MeshTraversePlan<Handle>, Handle&&>);
static_assert(!std::is_constructible_v<MeshTraverseExecution<Handle>, MeshTraversePlan<Handle>&&>);
static_assert(!std::is_constructible_v<FaceBatchProxy<Handle>, MeshTraverseExecution<Handle>&&>);
static_assert(!std::is_constructible_v<CellFaceBatchProxy<Handle>, const MeshTraverseExecution<Handle>&, Local>);

template<class Proxy, class Access>
concept AcceptsBoundTraverseAccess = requires(const Proxy& proxy, const Access& access)
{
    proxy.apply(access, [](const auto&) {});
    proxy.visit_batches(access, [](const auto&) {});
};
static_assert(AcceptsBoundTraverseAccess<CellFaceBatchProxy<Handle>,
    detail::TraverseAccess<Handle, Cartesian>>);
static_assert(!AcceptsBoundTraverseAccess<CellFaceBatchProxy<Handle>,
    detail::TraverseAccess<Cartesian, Cartesian>>);
static_assert(!AcceptsBoundTraverseAccess<CellFaceBatchProxy<Handle>, int>);

template<class Access> struct FaceRefStorageMirror
{
    const Access* access;
    typename Access::local_ordinal_type id;
    int batch;
    typename Access::local_ordinal_type cell;
    int orientation;
};
template<class Access> struct FaceBatchStorageMirror
{
    const Access* access;
    std::span<const detail::MeshTraverseRun> runs;
    std::span<const detail::MeshTraverseIncidence<typename Access::local_ordinal_type>> incidence;
    typename Access::local_ordinal_type cell;
};
template<class Access> constexpr bool face_storage_unchanged()
{
    constexpr auto Entity=MeshTraverseEntity::Face;
    static_assert(sizeof(MeshTraverseRef<Access,Entity,true,false>) == sizeof(FaceRefStorageMirror<Access>));
    static_assert(sizeof(MeshTraverseRef<Access,Entity,false,false>) == sizeof(FaceRefStorageMirror<Access>));
    static_assert(sizeof(MeshTraverseRef<Access,Entity,true,true>) == sizeof(FaceRefStorageMirror<Access>));
    static_assert(sizeof(MeshTraverseRef<Access,Entity,false,true>) == sizeof(FaceRefStorageMirror<Access>));
    static_assert(sizeof(MeshTraverseBatch<Access,Entity,true,false>) == sizeof(FaceBatchStorageMirror<Access>));
    static_assert(sizeof(MeshTraverseBatch<Access,Entity,false,false>) == sizeof(FaceBatchStorageMirror<Access>));
    static_assert(sizeof(MeshTraverseBatch<Access,Entity,true,true>) == sizeof(FaceBatchStorageMirror<Access>));
    static_assert(sizeof(MeshTraverseBatch<Access,Entity,false,true>) == sizeof(FaceBatchStorageMirror<Access>));
    return true;
}
static_assert(face_storage_unchanged<detail::TraverseAccess<Handle,Cartesian>>());
static_assert(face_storage_unchanged<detail::TraverseAccess<Cartesian,Cartesian>>());
static_assert(face_storage_unchanged<detail::LayeredTraverseAccess<detail::TraverseAccess<Handle,Meshes::SemiStructuredXY_Z>>>());
static_assert(face_storage_unchanged<detail::LayeredTraverseAccess<detail::TraverseAccess<Meshes::SemiStructuredXY_Z,Meshes::SemiStructuredXY_Z>>>());

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

/** The public point queries are independent of the structured range generator. */
template<class Mesh>
void expect_structured_plan(const Mesh& mesh, const Handle& reference)
{
    const auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    ASSERT_TRUE(plan.uses_structured_ranges());
    const auto traverse = plan.acquire();
    std::vector<Local> interior_cells, boundary_cells, interior_faces, boundary_faces;
    for (const auto cell : ordinals(reference.num_owned_cells()))
    {
        bool boundary = false;
        for (const auto face : reference.faces(cell))
            boundary = boundary || reference.is_geometry_exterior_face(face);
        (boundary ? boundary_cells : interior_cells).push_back(cell);
    }
    for (const auto face : ordinals(reference.num_owned_faces()))
        (reference.is_geometry_exterior_face(face) ? boundary_faces : interior_faces).push_back(face);

    const auto check_selection = [](const auto& selection, const auto& expected)
    {
        EXPECT_EQ(selection.size(), expected.size());
        EXPECT_EQ(selection.empty(), expected.empty());
        EXPECT_EQ(selected_ids(selection), expected);
        std::vector<Local> batched;
        selection.visit_batches([&](const auto& batch)
        {
            const auto before = batched.size();
            batch.apply([&](const auto& entity)
            {
                static_assert(std::remove_cvref_t<decltype(batch)>::is_interior ==
                              std::remove_cvref_t<decltype(entity)>::is_interior);
                batched.push_back(entity.id());
            });
            EXPECT_GT(batch.size(), 0U);
            EXPECT_EQ(batched.size() - before, batch.size());
        });
        EXPECT_EQ(batched, expected);
        auto independent = selected_ids(selection.independent());
        auto sorted = expected;
        std::ranges::sort(independent);
        std::ranges::sort(sorted);
        EXPECT_EQ(independent, sorted);
    };
    check_selection(traverse.cells(), ordinals(reference.num_owned_cells()));
    check_selection(traverse.cells().interior(), interior_cells);
    check_selection(traverse.cells().boundary(), boundary_cells);
    check_selection(traverse.cells().local(), ordinals(reference.num_local_cells()));
    check_selection(traverse.faces(), ordinals(reference.num_owned_faces()));
    check_selection(traverse.faces().interior(), interior_faces);
    check_selection(traverse.faces().boundary(), boundary_faces);
    check_selection(traverse.faces().local(), ordinals(reference.num_faces()));
    for (int patch = 0; patch < 6; ++patch)
    {
        std::vector<Local> expected;
        for (const auto face : boundary_faces)
            if (reference.boundary_id(face) == patch) expected.push_back(face);
        check_selection(traverse.faces().boundary(patch), expected);
    }
    check_selection(traverse.faces().boundary(987654), std::vector<Local>{});
    auto grouped_cells = interior_cells;
    grouped_cells.insert(grouped_cells.end(), boundary_cells.begin(), boundary_cells.end());
    EXPECT_EQ(selected_ids(traverse.cells().independent()), grouped_cells);
    const auto grouped_faces = [&](auto interior, auto boundary)
    {
        std::ranges::stable_sort(boundary, [&](auto a, auto b)
        { return reference.boundary_id(a) < reference.boundary_id(b); });
        interior.insert(interior.end(), boundary.begin(), boundary.end());
        return interior;
    };
    EXPECT_EQ(selected_ids(traverse.faces().independent()), grouped_faces(interior_faces, boundary_faces));

    traverse.cells().apply([&](const auto& cell)
    {
        EXPECT_EQ(std::remove_cvref_t<decltype(cell)>::is_interior,
                  std::ranges::binary_search(interior_cells, cell.id()));
        EXPECT_DOUBLE_EQ(cell.volume(), reference.cell_volume(cell.id()));
        expect_vector_equal(cell.centroid(), reference.cell_centroid(cell.id()));
        std::vector<Local> all, interior, boundary;
        for (const auto face : reference.faces(cell.id()))
        {
            all.push_back(face);
            (reference.is_geometry_exterior_face(face) ? boundary : interior).push_back(face);
        }
        const auto incidence = traverse.cell_faces(cell.id());
        check_selection(incidence, all);
        check_selection(incidence.interior(), interior);
        check_selection(incidence.boundary(), boundary);
        EXPECT_EQ(selected_ids(incidence.independent()), grouped_faces(interior, boundary));
        for (int patch = 0; patch < 6; ++patch)
        {
            std::vector<Local> expected;
            for (const auto face : boundary)
                if (reference.boundary_id(face) == patch) expected.push_back(face);
            check_selection(incidence.boundary(patch), expected);
        }
        std::vector<Local> borrowed;
        incidence.apply(cell.geometry(), [&](const auto& face)
        {
            using Ref = std::remove_cvref_t<decltype(face)>;
            borrowed.push_back(face.id());
            EXPECT_EQ(Ref::is_interior, !reference.is_geometry_exterior_face(face.id()));
            EXPECT_EQ(face.cell_id(), cell.id());
            EXPECT_EQ(face.owner(), reference.owner_cell(face.id()));
            if constexpr (Ref::is_interior)
                EXPECT_EQ(face.neighbor(), reference.neighbor_cell(face.id()));
            else
                EXPECT_EQ(face.boundary_id(), reference.boundary_id(face.id()));
            EXPECT_EQ(face.orientation(), reference.owner_cell(face.id()) == cell.id() ? 1 : -1);
            EXPECT_DOUBLE_EQ(face.area(), reference.face_area(face.id()));
            expect_vector_equal(face.centroid(), reference.face_centroid(face.id()));
            expect_vector_equal(face.outward_normal(), reference.face_normal_outward(face.id(), cell.id()));
        });
        EXPECT_EQ(borrowed, all);
    });
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

    visited = 0;
    traverse.cells().all().visit_batches([&](const auto& batch)
    {
        using Ref = typename std::remove_cvref_t<decltype(batch)>::reference;
        EXPECT_NO_THROW(batch.apply(&Ref::volume));
        struct RvalueOperation
        {
            size_t& count;
            void operator()(Ref&&) { ++count; }
        };
        batch.apply(RvalueOperation{visited});
    });
    EXPECT_EQ(visited, mesh.num_owned_cells());
    visited = 0;
    auto wrapped = [&](const auto&) { ++visited; };
    traverse.cells().apply(std::ref(wrapped));
    EXPECT_EQ(visited, mesh.num_owned_cells());
}

TEST(MeshTraverseTest, BoundAccessPreservesCellFaceOrderCategoriesAndGeometry)
{
    Handle mesh(cartesian(3, 3, 3));
    const auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    const auto traverse = plan.acquire();
    traverse.cells().apply([&](const auto& cell)
    {
        const auto faces = traverse.cell_faces(cell.id());
        std::vector<std::pair<Local, bool>> expected, actual, from_batches;
        faces.apply([&](const auto& face)
        {
            expected.emplace_back(face.id(), std::remove_cvref_t<decltype(face)>::is_interior);
        });
        faces.apply(cell.geometry(), [&](const auto& face)
        {
            actual.emplace_back(face.id(), std::remove_cvref_t<decltype(face)>::is_interior);
            EXPECT_DOUBLE_EQ(face.area(), mesh.face_area(face.id()));
            expect_vector_equal(face.outward_normal(), mesh.face_normal_outward(face.id(), cell.id()));
        });
        faces.visit_batches(cell.geometry(), [&](const auto& batch)
        {
            batch.apply([&](const auto& face)
            {
                using Batch = std::remove_cvref_t<decltype(batch)>;
                static_assert(Batch::is_interior == std::remove_cvref_t<decltype(face)>::is_interior);
                from_batches.emplace_back(face.id(), Batch::is_interior);
            });
        });
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(from_batches, expected);
    });
}

TEST(MeshTraverseTest, BoundAccessRejectsForeignMeshAndInvalidatedPlanBeforeCallbacks)
{
    const auto geometry = cartesian(1, 1, 1);
    Handle mesh(geometry), other(geometry);
    auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    const auto other_plan = make_mesh_traverse_plan(other);
    const auto traverse = plan.acquire();
    const auto other_traverse = other_plan.acquire();
    const auto faces = traverse.cell_faces(0);
    size_t callbacks = 0;
    const auto operation = [&](const auto&) { ++callbacks; };
    other_traverse.cells().apply([&](const auto& cell)
    {
        EXPECT_THROW(faces.apply(cell.geometry(), operation), std::invalid_argument);
        EXPECT_THROW(faces.visit_batches(cell.geometry(), operation), std::invalid_argument);
    });
    EXPECT_EQ(callbacks, 0U);
    traverse.cells().apply([&](const auto& cell)
    {
        plan.invalidate();
        EXPECT_THROW(faces.apply(cell.geometry(), operation), std::logic_error);
        EXPECT_THROW(faces.visit_batches(cell.geometry(), operation), std::logic_error);
    });
    EXPECT_EQ(callbacks, 0U);
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
    EXPECT_FALSE(plan.uses_structured_ranges());
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
        EXPECT_FALSE(plan.uses_structured_ranges());
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

TEST(MeshTraverseTest, StructuredCartesianRangesPreserveAllPeriodicMasks)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    for (unsigned mask = 0; mask < 8; ++mask)
    {
        SCOPED_TRACE(mask);
        auto geometry = std::make_shared<Cartesian>(
            Vec3D<ArrReal>{{{0, 0.5, 2, 4}, {0, 0.25, 1, 2, 5}, {0, 2, 7}}},
            Vec3D<bool>{{bool(mask & 1U), bool(mask & 2U), bool(mask & 4U)}});
        Handle mesh(geometry);
        expect_structured_plan(*geometry, mesh);
        expect_structured_plan(mesh, mesh);
    }
}

TEST(MeshTraverseTest, StructuredCartesianRangesPreserveThinAndSingleCellAxes)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    for (const auto shape : {std::array<size_t, 3>{1, 1, 1}, {1, 4, 2}, {4, 1, 2}, {4, 2, 1}, {2, 2, 2}})
    {
        SCOPED_TRACE(::testing::PrintToString(shape));
        auto geometry = cartesian(shape[0], shape[1], shape[2]);
        Handle mesh(geometry);
        expect_structured_plan(*geometry, mesh);
        expect_structured_plan(mesh, mesh);
    }
}

TEST(MeshTraverseTest, StructuredCylindricalRangesPreserveSectorsAndPeriodicTheta)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    const auto pi = std::numbers::pi_v<real_t>;
    for (const auto& edges : {
        Vec3D<ArrReal>{{{1, 2}, {0, 0.5}, {0, 1}}},
        Vec3D<ArrReal>{{{1, 2, 4}, {0, 0.2, 0.7, 1.5}, {0, 0.5, 3}}},
        Vec3D<ArrReal>{{{1, 2}, {0, pi, 2 * pi}, {0, 1}}},
        Vec3D<ArrReal>{{{1, 2, 4}, {0, pi / 2, pi, 3 * pi / 2, 2 * pi}, {0, 0.5, 2, 3}}}})
    {
        auto geometry = std::make_shared<Cylindrical>(edges);
        SCOPED_TRACE(geometry->is_theta_periodic());
        Handle mesh(geometry);
        expect_structured_plan(*geometry, mesh);
        expect_structured_plan(mesh, mesh);
    }
}

TEST(MeshTraverseTest, StructuredPlanStorageDoesNotGrowWithCellOrFaceCount)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    const auto small = cartesian(3, 3, 3), large = cartesian(24, 24, 24);
    Handle small_handle(small), large_handle(large);
    for (const auto entities : {MeshTraverseEntities::Cells, MeshTraverseEntities::Faces,
                               MeshTraverseEntities::All, MeshTraverseEntities::AllWithCellFaces})
    {
        const auto small_native_plan = make_mesh_traverse_plan(*small, entities);
        const auto large_native_plan = make_mesh_traverse_plan(*large, entities);
        const auto small_handle_plan = make_mesh_traverse_plan(small_handle, entities);
        const auto large_handle_plan = make_mesh_traverse_plan(large_handle, entities);
        EXPECT_TRUE(small_native_plan.uses_structured_ranges());
        EXPECT_TRUE(large_native_plan.uses_structured_ranges());
        EXPECT_TRUE(small_handle_plan.uses_structured_ranges());
        EXPECT_TRUE(large_handle_plan.uses_structured_ranges());
        EXPECT_EQ(small_native_plan.storage_bytes(), large_native_plan.storage_bytes());
        EXPECT_EQ(small_handle_plan.storage_bytes(), large_handle_plan.storage_bytes());
    }
}

TEST(MeshTraverseTest, StructuredPlansRetainTopologyAndRefreshGeometryAcrossALE)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    const auto check = [](const auto& geometry)
    {
        auto mesh = std::make_shared<Handle>(geometry);
        const auto plan = make_mesh_traverse_plan(*mesh, MeshTraverseEntities::AllWithCellFaces);
        const auto native_plan = make_mesh_traverse_plan(*geometry, MeshTraverseEntities::AllWithCellFaces);
        ASSERT_TRUE(plan.uses_structured_ranges());
        ASSERT_TRUE(native_plan.uses_structured_ranges());
        const auto storage = plan.storage_bytes();
        const auto capture = [&](const auto& retained_plan)
        {
            std::vector<real_t> metrics;
            const auto traverse = retained_plan.acquire();
            traverse.cells().apply([&](const auto& cell)
            {
                EXPECT_DOUBLE_EQ(cell.volume(), mesh->cell_volume(cell.id()));
                metrics.push_back(cell.volume());
                std::vector<Local> expected;
                for (const auto face : mesh->faces(cell.id())) expected.push_back(face);
                EXPECT_EQ(selected_ids(traverse.cell_faces(cell.id())), expected);
                traverse.cell_faces(cell.id()).apply(cell.geometry(), [&](const auto& face)
                {
                    EXPECT_DOUBLE_EQ(face.area(), mesh->face_area(face.id()));
                    expect_vector_equal(face.centroid(), mesh->face_centroid(face.id()));
                    expect_vector_equal(face.outward_normal(), mesh->face_normal_outward(face.id(), cell.id()));
                    metrics.push_back(face.area());
                });
            });
            return metrics;
        };
        const auto before = capture(plan);
        EXPECT_EQ(capture(native_plan), before);
        PlanarALEMeshMotion<Pack> motion(mesh);
        motion.begin_trial(4.0, 0.1);
        const auto during = capture(plan);
        EXPECT_NE(during, before);
        EXPECT_EQ(capture(native_plan), during);
        EXPECT_EQ(plan.storage_bytes(), storage);
        motion.rollback_trial();
        EXPECT_EQ(capture(plan), before);
        EXPECT_EQ(capture(native_plan), before);
        EXPECT_EQ(plan.storage_bytes(), storage);
    };
    check(cartesian(3, 2, 2));
    check(std::make_shared<Cylindrical>(
        Vec3D<ArrReal>{{{1, 2, 3}, {0, 0.5, 1}, {0, 1, 2}}}));
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
    EXPECT_FALSE(plan.uses_structured_ranges());
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

namespace
{
using IncidenceRunKey = std::tuple<size_t, size_t, bool, int>;

// Build the oracle only from public point queries. Retained XY runs and the
// arithmetic layout are deliberately absent from this classification.
std::vector<IncidenceRunKey> reference_incidence_runs(const Handle& reference,
    Local cell, detail::StructuredTraverseSelection selection, bool independent,
    std::optional<int> patch)
{
    struct Entry { size_t slot; bool interior; int patch; };
    std::vector<Entry> entries;
    size_t slot = 0;
    for (auto face : reference.faces(cell))
    {
        const bool interior = !reference.is_geometry_exterior_face(face);
        const int id = interior ? -1 : reference.boundary_id(face);
        const bool selected = selection == detail::StructuredTraverseSelection::All
            || (selection == detail::StructuredTraverseSelection::Interior) == interior;
        if (selected && (!patch || selection == detail::StructuredTraverseSelection::Interior || id == *patch))
            entries.push_back({slot, interior, id});
        ++slot;
    }
    if (independent)
        std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b)
        {
            if (a.interior != b.interior) return a.interior;
            return a.patch < b.patch;
        });
    std::vector<IncidenceRunKey> runs;
    for (const auto& entry : entries)
    {
        if (!runs.empty() && std::get<1>(runs.back()) == entry.slot
            && std::get<2>(runs.back()) == entry.interior && std::get<3>(runs.back()) == entry.patch)
            ++std::get<1>(runs.back());
        else runs.emplace_back(entry.slot, entry.slot + 1, entry.interior, entry.patch);
    }
    return runs;
}

template<class Incidences, class Proxy>
void expect_layered_incidence_runs(const Incidences& indexed, const Proxy& proxy,
    const Handle& reference, Local cell)
{
    using Selection = detail::StructuredTraverseSelection;
    std::vector<Local> face_ids;
    std::vector<std::optional<int>> patches{std::nullopt, 987654};
    for (auto face : reference.faces(cell))
    {
        face_ids.push_back(face);
        if (reference.is_geometry_exterior_face(face)) patches.push_back(reference.boundary_id(face));
    }
    std::sort(patches.begin(), patches.end());
    patches.erase(std::unique(patches.begin(), patches.end()), patches.end());
    auto check = [&](Selection selection, bool independent, std::optional<int> patch, const auto& selected_proxy)
    {
        const auto expected = reference_incidence_runs(reference, cell, selection, independent, patch);
        std::vector<IncidenceRunKey> actual;
        indexed.visit_runs(selection, independent, patch, [&](detail::StructuredTraverseRun& run)
        {
            actual.emplace_back(run.begin, run.end, run.interior, run.batch);
            // A nested visit must leave the outer descriptor unchanged.
            std::vector<IncidenceRunKey> nested;
            indexed.visit_runs(selection, independent, patch, [&](const auto& inner)
            { nested.emplace_back(inner.begin, inner.end, inner.interior, inner.batch); });
            EXPECT_EQ(nested, expected);
        });
        EXPECT_EQ(actual, expected);
        std::vector<std::vector<Local>> expected_batches, actual_batches;
        for (const auto& [begin, end, interior, batch] : expected)
            expected_batches.emplace_back(face_ids.begin() + begin, face_ids.begin() + end);
        selected_proxy.visit_batches([&](const auto& batch)
        {
            std::vector<Local> ids;
            batch.apply([&](const auto& face) { ids.push_back(face.id()); });
            EXPECT_EQ(batch.size(), ids.size());
            actual_batches.push_back(std::move(ids));
            using Ref = typename std::remove_cvref_t<decltype(batch)>::reference;
            batch.apply(&Ref::id); // Preserve member-pointer invocation.
        });
        EXPECT_EQ(actual_batches, expected_batches);
        std::vector<Local> move_only_ids;
        selected_proxy.apply([state = std::make_unique<unsigned>(0), &move_only_ids](auto&& face) mutable
        { ++*state; move_only_ids.push_back(face.id()); });
        std::vector<Local> flattened;
        for (const auto& batch : expected_batches) flattened.insert(flattened.end(), batch.begin(), batch.end());
        EXPECT_EQ(move_only_ids, flattened);
    };
    for (bool independent : {false, true})
    {
        const auto selected_proxy = independent ? proxy.independent() : proxy;
        check(Selection::All, independent, {}, selected_proxy);
        check(Selection::Interior, independent, {}, selected_proxy.interior());
        check(Selection::Boundary, independent, {}, selected_proxy.boundary());
        for (const auto patch : patches)
            if (patch) check(Selection::Boundary, independent, patch, selected_proxy.boundary(*patch));
    }
}

template<class Mesh>
void expect_layered_plan(const Mesh& mesh, const Handle& reference)
{
    const auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    ASSERT_TRUE(plan.uses_layered_ranges());
    EXPECT_FALSE(plan.uses_structured_ranges());
    ASSERT_TRUE(plan.layered_layout());
    const auto traverse = plan.acquire();
    std::vector<Local> interior_cells, boundary_cells, interior_faces, boundary_faces;
    std::map<int,std::vector<Local>> patches;
    for (const auto cell : ordinals(reference.num_owned_cells()))
    {
        bool boundary=false;
        for (auto face : reference.faces(cell)) boundary=boundary || reference.is_geometry_exterior_face(face);
        (boundary ? boundary_cells : interior_cells).push_back(cell);
    }
    for (const auto face : ordinals(reference.num_owned_faces()))
    {
        const bool boundary=reference.is_geometry_exterior_face(face);
        (boundary ? boundary_faces : interior_faces).push_back(face);
        if (boundary) patches[reference.boundary_id(face)].push_back(face);
    }
    EXPECT_EQ(selected_ids(traverse.cells()),ordinals(reference.num_owned_cells()));
    EXPECT_EQ(selected_ids(traverse.cells().interior()),interior_cells);
    EXPECT_EQ(selected_ids(traverse.cells().boundary()),boundary_cells);
    EXPECT_EQ(traverse.cells().interior().size(),interior_cells.size());
    EXPECT_EQ(selected_ids(traverse.faces()),ordinals(reference.num_owned_faces()));
    EXPECT_EQ(selected_ids(traverse.faces().interior()),interior_faces);
    EXPECT_EQ(selected_ids(traverse.faces().boundary()),boundary_faces);
    std::vector<Local> grouped=interior_faces;
    for (const auto& [id, faces] : patches)
    {
        EXPECT_EQ(selected_ids(traverse.faces().boundary(id)),faces);
        EXPECT_EQ(traverse.faces().boundary(id).size(),faces.size());
        grouped.insert(grouped.end(),faces.begin(),faces.end());
    }
    EXPECT_EQ(selected_ids(traverse.faces().independent()),grouped);
    traverse.cells().apply([&](const auto& cell)
    {
        EXPECT_DOUBLE_EQ(cell.volume(),reference.cell_volume(cell.id()));
        expect_vector_equal(cell.centroid(),reference.cell_centroid(cell.id()));
        std::vector<Local> expected,inner;
        std::map<int,std::vector<Local>> boundary;
        for (auto face : reference.faces(cell.id()))
        {
            expected.push_back(face);
            if (reference.is_geometry_exterior_face(face)) boundary[reference.boundary_id(face)].push_back(face);
            else inner.push_back(face);
        }
        const auto incidence=traverse.cell_faces(cell.id());
        const bool fully_interior = boundary.empty();
        EXPECT_EQ(plan.layered_layout()->cell_faces(cell.id()).all_interior(), fully_interior);
        if (fully_interior)
        {
            size_t batches = 0;
            incidence.interior().independent().visit_batches([&](const auto& batch)
            { ++batches; EXPECT_EQ(batch.size(), expected.size()); });
            EXPECT_EQ(batches, 1U);
            EXPECT_TRUE(selected_ids(incidence.boundary()).empty());
            EXPECT_TRUE(selected_ids(incidence.boundary(987654).independent()).empty());
        }
        // Existing batch apply passes a temporary reference. A constrained
        // rvalue-only callable catches accidental lvalue dispatch for both
        // native and canonical Handle layered incidence.
        std::vector<Local> rvalue_faces;
        const auto rvalue_only = [&]<class Ref>(Ref&& face)
            requires (!std::is_lvalue_reference_v<Ref>)
        { rvalue_faces.push_back(face.id()); };
        incidence.apply(rvalue_only);
        EXPECT_EQ(rvalue_faces, expected);
        rvalue_faces.clear();
        incidence.apply(cell.geometry(), rvalue_only);
        EXPECT_EQ(rvalue_faces, expected);
        EXPECT_EQ(selected_ids(incidence),expected);
        EXPECT_EQ(incidence.size(),expected.size());
        EXPECT_EQ(plan.layered_layout()->cell_faces(cell.id()).size(),expected.size());
        const auto indexed_faces = plan.layered_layout()->cell_faces(cell.id());
        expect_layered_incidence_runs(indexed_faces, incidence, reference, static_cast<Local>(cell.id()));
        for (size_t slot = 0; slot < expected.size(); ++slot)
        {
            EXPECT_EQ(indexed_faces.face_ordinal(slot), static_cast<size_t>(expected[slot]));
            using Access = std::remove_cvref_t<decltype(cell.geometry())>;
            if constexpr (detail::is_layered_traverse_access<Access>)
            {
                const auto resolved = cell.geometry().resolve_incidence(indexed_faces, slot);
                const auto face = expected[slot];
                const auto owner = reference.owner_cell(face);
                const auto neighbor = reference.neighbor_cell(face);
                const auto& indexer = plan.layered_layout()->indexer();
                EXPECT_EQ(resolved.incidence.face, static_cast<size_t>(face));
                EXPECT_EQ(resolved.incidence.orientation, owner == cell.id() ? 1 : -1);
                EXPECT_EQ(resolved.incidence.interior, !reference.is_geometry_exterior_face(face));
                EXPECT_EQ(resolved.incidence.batch, resolved.incidence.interior ? -1 : reference.boundary_id(face));
                EXPECT_EQ(resolved.owner, static_cast<size_t>(owner));
                EXPECT_EQ(resolved.neighbor, neighbor == Handle::invalid_local_id()
                    ? std::numeric_limits<size_t>::max() : static_cast<size_t>(neighbor));
                EXPECT_EQ(resolved.cell, indexer.cell_id(cell.id()));
                const auto other = owner == cell.id() ? neighbor : owner;
                EXPECT_EQ(resolved.other, other == Handle::invalid_local_id()
                    ? Meshes::SemiStructuredIndexer::CellID{} : indexer.cell_id(other));
            }
        }
        incidence.apply(cell.geometry(),[&](const auto& face)
        {
            EXPECT_EQ(face.orientation(),reference.owner_cell(face.id())==cell.id() ? 1 : -1);
            EXPECT_EQ(face.owner(),reference.owner_cell(face.id()));
            using Ref = std::remove_cvref_t<decltype(face)>;
            using Access = std::remove_cvref_t<decltype(face.geometry())>;
            if constexpr (Ref::is_interior)
            {
                EXPECT_EQ(face.neighbor(),reference.neighbor_cell(face.id()));
                if constexpr (detail::is_layered_traverse_access<Access>)
                {
                    static_assert(requires { face.cell_center_vector(); });
                    expect_vector_equal(face.cell_center_vector(),reference.cell_center_vector(face.id(),cell.id()));
                }
            }
            EXPECT_DOUBLE_EQ(face.area(),reference.face_area(face.id()));
            expect_vector_equal(face.centroid(),reference.face_centroid(face.id()));
            expect_vector_equal(face.normal(),reference.face_normal(face.id()));
            // Nested borrowed incidence must not overwrite shared scratch.
            EXPECT_EQ(selected_ids(traverse.cell_faces(cell.id())),expected);
        });
        for (const auto& [id,faces] : boundary) inner.insert(inner.end(),faces.begin(),faces.end());
        EXPECT_EQ(selected_ids(incidence.independent()),inner);
        std::vector<Local> grouped_boundary;
        for (const auto& [id,faces] : boundary) grouped_boundary.insert(grouped_boundary.end(),faces.begin(),faces.end());
        EXPECT_EQ(selected_ids(incidence.boundary().independent()),grouped_boundary);
    });
}
}

TEST(MeshTraverseTest, LayeredPolygonIncidenceAndPatchOrder)
{
    using Layered=Meshes::SemiStructuredXY_Z;
    for (unsigned degree : {3U,4U,7U,300U})
        for (unsigned layers : {1U,2U,5U})
        {
            Arr<Layered::Vec3> nodes; Arr<unsigned> polygon; Arr<Layered::BoundaryEdge> edges;
            for (unsigned i=0;i<degree;++i)
            {
                const double angle=2*std::acos(-1.0)*i/degree;
                nodes.push_back({std::cos(angle),std::sin(angle),0}); polygon.push_back(i);
                edges.push_back({i,(i+1)%degree,"patch"+std::to_string((i/2)%11)});
            }
            ArrReal z{0}; for (unsigned k=0;k<layers;++k) z.push_back(z.back()+0.5+k*0.125);
            auto native=std::make_shared<Layered>(nodes,Arr<Arr<unsigned>>{polygon},z,edges);
            const Handle handle(native);
            expect_layered_plan(*native,handle); expect_layered_plan(handle,handle);
        }
}

TEST(MeshTraverseTest, LayeredMixedCellsAndStorageIndependentOfLayers)
{
    using Layered=Meshes::SemiStructuredXY_Z;
    const Arr<Layered::Vec3> nodes{{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0}};
    const Arr<Arr<unsigned>> cells{{0,1,4,3},{1,2,4},{2,5,4}};
    size_t storage=0;
    for (unsigned layers : {1U,2U,8U,100U})
    {
        ArrReal z{0}; for (unsigned k=0;k<layers;++k) z.push_back(z.back()+1.0+k*0.01);
        auto native=std::make_shared<Layered>(nodes,cells,z);
        const Handle handle(native);
        const auto plan=make_mesh_traverse_plan(*native,MeshTraverseEntities::AllWithCellFaces);
        if (storage==0) storage=plan.storage_bytes();
        EXPECT_EQ(plan.storage_bytes(),storage);
        if (layers<100) { expect_layered_plan(*native,handle); expect_layered_plan(handle,handle); }
    }
    // A three-by-three XY grid has a physical interior prism on inner Z layers.
    Arr<Layered::Vec3> grid; Arr<Arr<unsigned>> quads;
    for (unsigned j=0;j<4;++j) for (unsigned i=0;i<4;++i) grid.push_back({double(i),double(j),0});
    for (unsigned j=0;j<3;++j) for (unsigned i=0;i<3;++i) { auto b=i+4*j; quads.push_back({b,b+1,b+5,b+4}); }
    auto native=std::make_shared<Layered>(grid,quads,ArrReal{0,0.5,2,4});
    const Handle handle(native); expect_layered_plan(*native,handle); expect_layered_plan(handle,handle);
}

TEST(MeshTraverseTest, ReorderedLayeredHandleRetainsCanonicalFallback)
{
    using Layered=Meshes::SemiStructuredXY_Z;
    auto native=std::make_shared<Layered>(
        Arr<Layered::Vec3>{{0,0,0},{1,0,0},{1,1,0},{0,1,0}},
        Arr<Arr<unsigned>>{{0,1,3},{1,2,3}},ArrReal{0,1,3});
    auto handle=std::make_shared<Handle>(native);
    auto reordered=MeshReorderingFactory<Pack>::selected_cells_first(
        std::move(handle),[](auto,const auto& center){return center.z>1;});
    const auto plan=make_mesh_traverse_plan(*reordered.mesh,MeshTraverseEntities::AllWithCellFaces);
    EXPECT_FALSE(plan.uses_layered_ranges()); EXPECT_FALSE(plan.uses_structured_ranges());
    expect_handle_selections(*reordered.mesh);
    const auto traverse=plan.acquire();
    traverse.cells().apply([&](const auto& cell)
    {
        std::vector<Local> expected;
        for (auto face : reordered.mesh->faces(cell.id())) expected.push_back(face);
        EXPECT_EQ(selected_ids(traverse.cell_faces(cell.id())),expected);
    });
}

TEST(MeshTraverseTest, LayeredIncidenceRunsMergeAxialPrefixWithFirstSideRun)
{
    using Layered = Meshes::SemiStructuredXY_Z;
    // Start the second polygon at its shared edge. On the bottom Z layer,
    // upper axial incidence and first side incidence form one interior run.
    const Arr<Layered::Vec3> nodes{{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0}};
    const Arr<Arr<unsigned>> cells{{0,1,4,3},{4,1,2,5}};
    size_t native_storage = 0, handle_storage = 0;
    for (unsigned layers : {1U,2U,7U,40U})
    {
        ArrReal z{0};
        for (unsigned k = 0; k < layers; ++k) z.push_back(z.back() + 0.25 + k*0.03);
        auto native = std::make_shared<Layered>(nodes, cells, z);
        const Handle reference(native);
        const auto native_plan = make_mesh_traverse_plan(*native, MeshTraverseEntities::AllWithCellFaces);
        const auto handle_plan = make_mesh_traverse_plan(reference, MeshTraverseEntities::AllWithCellFaces);
        if (native_storage == 0)
        { native_storage = native_plan.storage_bytes(); handle_storage = handle_plan.storage_bytes(); }
        EXPECT_EQ(native_plan.storage_bytes(), native_storage);
        EXPECT_EQ(handle_plan.storage_bytes(), handle_storage);
        expect_layered_plan(*native, reference);
        expect_layered_plan(reference, reference);
        if (layers > 1)
        {
            const auto expected = reference_incidence_runs(reference, 1,
                detail::StructuredTraverseSelection::All, false, {});
            ASSERT_GE(expected.size(), 2U);
            EXPECT_EQ(expected[1], (IncidenceRunKey{1, 3, true, -1}));
        }
    }
}

TEST(MeshTraverseTest, LayeredTopologyPeriodicIncidenceRuns)
{
    using Topology = Meshes::SemiStructMeshTopo;
    using Selection = detail::StructuredTraverseSelection;
    const Arr<Arr<unsigned>> cells{{0,1,4,3},{4,1,2,5}};
    size_t storage = 0;
    for (unsigned layers : {1U,2U,5U})
    {
        const Topology topology(6, cells, layers, {}, true);
        const detail::LayeredTraverseLayout layout(topology);
        if (storage == 0) storage = layout.storage_bytes();
        EXPECT_EQ(layout.storage_bytes(), storage);
        const auto& indexer = topology.indexer();
        for (size_t ordinal = 0; ordinal < indexer.total_cells(); ++ordinal)
        {
            const auto cell = indexer.cell_id(ordinal);
            const auto incidences = layout.cell_faces(ordinal);
            struct Entry { size_t slot; bool interior; int batch; };
            std::vector<Entry> entries;
            std::vector<std::optional<int>> patches{std::nullopt, 987654};
            size_t slot = 0;
            for (auto face : topology.cell_faces(cell))
            {
                const bool interior = topology.neighbor_cell(face) != Topology::CellID{};
                const int batch = interior ? -1 : topology.boundary_id(face);
                const auto actual = incidences[slot];
                const auto resolved = incidences.resolved(slot, topology);
                const auto owner = topology.owner_cell(face), neighbor = topology.neighbor_cell(face);
                EXPECT_EQ(resolved.incidence.face, actual.face);
                EXPECT_EQ(resolved.incidence.orientation, actual.orientation);
                EXPECT_EQ(resolved.incidence.interior, actual.interior);
                EXPECT_EQ(resolved.incidence.batch, actual.batch);
                EXPECT_EQ(resolved.owner, indexer.cell_ordinal(owner));
                EXPECT_EQ(resolved.neighbor, neighbor == Topology::CellID{}
                    ? std::numeric_limits<size_t>::max() : indexer.cell_ordinal(neighbor));
                EXPECT_EQ(resolved.cell, cell);
                EXPECT_EQ(resolved.other, owner == cell ? neighbor : owner);
                EXPECT_EQ(actual.face, indexer.face_ordinal(face));
                EXPECT_EQ(incidences.face_ordinal(slot), indexer.face_ordinal(face));
                EXPECT_EQ(actual.orientation, topology.owner_cell(face) == cell ? 1 : -1);
                EXPECT_EQ(actual.interior, interior);
                EXPECT_EQ(actual.interior, !topology.is_boundary_face(face));
                EXPECT_EQ(actual.batch, batch);
                entries.push_back({slot++, interior, batch});
                if (!interior) patches.push_back(batch);
            }
            EXPECT_EQ(incidences.size(), entries.size());
            if (layers == 1)
            {
                EXPECT_EQ(incidences[0].face, incidences[1].face);
                EXPECT_EQ(incidences[0].orientation, 1);
                EXPECT_EQ(incidences[1].orientation, 1);
            }
            std::sort(patches.begin(), patches.end());
            patches.erase(std::unique(patches.begin(), patches.end()), patches.end());
            for (Selection selection : {Selection::All, Selection::Interior, Selection::Boundary})
                for (bool independent : {false, true})
                    for (auto patch : patches)
                    {
                        if (selection != Selection::Boundary && patch) continue;
                        std::vector<Entry> selected;
                        for (const auto entry : entries)
                            if ((selection == Selection::All || (selection == Selection::Interior) == entry.interior)
                                && (!patch || entry.batch == *patch)) selected.push_back(entry);
                        if (independent)
                            std::stable_sort(selected.begin(), selected.end(), [](const Entry& a, const Entry& b)
                            {
                                if (a.interior != b.interior) return a.interior;
                                return a.batch < b.batch;
                            });
                        std::vector<IncidenceRunKey> expected;
                        for (const auto entry : selected)
                        {
                            if (!expected.empty() && std::get<1>(expected.back()) == entry.slot
                                && std::get<2>(expected.back()) == entry.interior && std::get<3>(expected.back()) == entry.batch)
                                ++std::get<1>(expected.back());
                            else expected.emplace_back(entry.slot, entry.slot + 1, entry.interior, entry.batch);
                        }
                        std::vector<IncidenceRunKey> actual;
                        incidences.visit_runs(selection, independent, patch, [&](const auto& run)
                        { actual.emplace_back(run.begin, run.end, run.interior, run.batch); });
                        EXPECT_EQ(actual, expected);
                    }
        }
    }
}

TEST(MeshTraverseTest, LayeredFullyInteriorPeriodicRowsProduceOneRun)
{
    using Topology = Meshes::SemiStructMeshTopo;
    using Selection = detail::StructuredTraverseSelection;
    Arr<Arr<unsigned>> cells;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 3; ++x)
        { const auto p = x + 4*y; cells.push_back({p, p+1, p+5, p+4}); }
    for (unsigned layers : {1U, 2U, 5U})
    {
        const Topology topology(16, cells, layers, {}, true);
        const detail::LayeredTraverseLayout layout(topology);
        for (unsigned layer = 0; layer < layers; ++layer)
        {
            const auto cell = layout.indexer().cell_ordinal({4, layer});
            const auto faces = layout.cell_faces(cell);
            ASSERT_TRUE(faces.all_interior());
            for (bool independent : {false, true})
                for (auto selection : {Selection::All, Selection::Interior})
                {
                    std::vector<IncidenceRunKey> runs;
                    faces.visit_runs(selection, independent, {}, [&](const auto& run)
                    { runs.emplace_back(run.begin, run.end, run.interior, run.batch); });
                    EXPECT_EQ(runs, (std::vector<IncidenceRunKey>{{0, 6, true, -1}}));
                }
            for (auto patch : {std::optional<int>{}, std::optional<int>{987654}})
            {
                size_t batches = 0;
                faces.visit_runs(Selection::Boundary, true, patch, [&](const auto&) { ++batches; });
                EXPECT_EQ(batches, 0U);
            }
            if (layers == 1)
            {
                EXPECT_EQ(faces[0].face, faces[1].face);
                EXPECT_EQ(faces[0].orientation, 1);
                EXPECT_EQ(faces[1].orientation, 1);
            }
        }
    }
}

namespace
{
auto layered_interior_fixture(unsigned layers)
{
    using Layered = Meshes::SemiStructuredXY_Z;
    Arr<Layered::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
    for (unsigned j=0;j<4;++j) for (unsigned i=0;i<4;++i) nodes.push_back({double(i),double(j),0});
    for (unsigned j=0;j<3;++j) for (unsigned i=0;i<3;++i)
    { const auto b=i+4*j; cells.push_back({b,b+1,b+5,b+4}); }
    ArrReal z{0};
    for (unsigned k=0;k<layers;++k) z.push_back(z.back()+0.2+k*0.07);
    return std::make_shared<Layered>(nodes,cells,z);
}

template<class Mesh>
void expect_layered_typed_cells(const Mesh& mesh)
{
    const auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto foreign_plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto traversal=plan.acquire();
    const auto another_execution=plan.acquire();
    const auto foreign=foreign_plan.acquire();
    traversal.cells().apply([&](const auto& cell)
    {
        using Cell=std::remove_cvref_t<decltype(cell)>;
        using Access=std::remove_cvref_t<decltype(cell.geometry())>;
        const auto typed=traversal.cell_faces(cell);
        const auto by_id=traversal.cell_faces(cell.id());
        EXPECT_NO_THROW(typed.require_local_adjacency());
        EXPECT_NO_THROW(by_id.require_local_adjacency());
        if constexpr (Cell::is_interior && detail::is_layered_traverse_access<Access>)
        {
            static_assert(std::remove_cvref_t<decltype(typed)>::is_layered_interior_range);
            EXPECT_EQ(selected_ids(typed),selected_ids(by_id));
            EXPECT_EQ(selected_ids(typed.interior()),selected_ids(by_id));
            EXPECT_EQ(selected_ids(typed.independent()),selected_ids(by_id));
            EXPECT_EQ(typed.size(),by_id.size());
            EXPECT_EQ(typed.boundary().size(),0U);
            EXPECT_EQ(typed.boundary(987654).size(),0U);
            EXPECT_NO_THROW(typed.boundary().require_local_adjacency());
            EXPECT_NO_THROW(typed.boundary(987654).require_local_adjacency());
            size_t empty_callbacks=0;
            typed.boundary().apply([&](const auto&){++empty_callbacks;});
            typed.boundary(987654).visit_batches([&](const auto&){++empty_callbacks;});
            EXPECT_EQ(empty_callbacks,0U);
            size_t batches=0;
            typed.visit_batches([&](const auto& batch)
            {
                using Batch=std::remove_cvref_t<decltype(batch)>;
                static_assert(Batch::is_interior);
                using Ref=typename Batch::reference;
                ++batches;
                EXPECT_EQ(batch.size(),by_id.size());
                batch.apply(&Ref::id);
            });
            EXPECT_EQ(batches,1U);
            std::vector<Local> ids;
            typed.apply([&]<class Ref>(Ref&& face) requires (!std::is_lvalue_reference_v<Ref>)
            { ids.push_back(face.id()); });
            EXPECT_EQ(ids,selected_ids(by_id));
            ids.clear();
            typed.apply([state=std::make_unique<unsigned>(0),&ids](auto&& face) mutable
            { ++*state; ids.push_back(face.id()); });
            EXPECT_EQ(ids,selected_ids(by_id));
            std::vector<std::pair<Local,int>> expected,actual;
            by_id.apply([&](const auto& face){expected.emplace_back(face.id(),face.orientation());});
            typed.apply([&](const auto& face)
            {
                actual.emplace_back(face.id(),face.orientation());
                using Ref = std::remove_cvref_t<decltype(face)>;
                static_assert(Ref::is_interior);
                static_assert(requires { face.cell_center_vector(); });
                EXPECT_EQ(face.owner(),face.geometry().owner_cell(face.id()));
                EXPECT_EQ(face.neighbor(),face.geometry().neighbor_cell(face.id()));
                expect_vector_equal(face.cell_center_vector(),face.geometry().cell_center_vector(face.id(),cell.id()));
                EXPECT_EQ(selected_ids(another_execution.cell_faces(cell)),selected_ids(by_id));
                EXPECT_EQ(selected_ids(traversal.cell_faces(cell)),selected_ids(by_id));
            });
            EXPECT_EQ(actual,expected);
        }
        else EXPECT_EQ(selected_ids(typed),selected_ids(by_id));
        EXPECT_THROW(static_cast<void>(foreign.cell_faces(cell)),std::invalid_argument);
        const Cell forged(cell.geometry(),cell.id());
        EXPECT_THROW(static_cast<void>(traversal.cell_faces(forged)),std::invalid_argument);
    });
    const auto cells_only=make_mesh_traverse_plan(mesh,MeshTraverseEntities::Cells);
    const auto cell_execution=cells_only.acquire();
    cell_execution.cells().apply([&](const auto& cell)
    { EXPECT_THROW(static_cast<void>(cell_execution.cell_faces(cell)),std::logic_error); });
}
}

TEST(MeshTraverseTest, LayeredTypedInteriorCellProofPreservesIncidenceContracts)
{
    for (unsigned layers : {1U,2U,3U,7U})
    {
        const auto native=layered_interior_fixture(layers);
        const Handle handle(native);
        expect_layered_typed_cells(*native);
        expect_layered_typed_cells(handle);
    }
}

TEST(MeshTraverseTest, LayeredTypedCellProofRejectsForeignMesh)
{
    const auto native=layered_interior_fixture(3);
    const auto other_native=layered_interior_fixture(3);
    const Handle mesh(native),other_mesh(other_native);
    const auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto other_plan=make_mesh_traverse_plan(other_mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto traversal=plan.acquire(),other=other_plan.acquire();
    other.cells().apply([&](const auto& cell)
    { EXPECT_THROW(static_cast<void>(traversal.cell_faces(cell)),std::invalid_argument); });
}

TEST(MeshTraverseTest, LayeredTypedInteriorRangeRejectsInvalidatedPlan)
{
    const auto native=layered_interior_fixture(3);
    const Handle mesh(native);
    auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto traversal=plan.acquire();
    traversal.cells().interior().apply([&](const auto& cell)
    {
        using Access=std::remove_cvref_t<decltype(cell.geometry())>;
        if constexpr (detail::is_layered_traverse_access<Access>)
        {
            const auto range=traversal.cell_faces(cell);
            plan.invalidate();
            EXPECT_THROW(static_cast<void>(range.size()),std::logic_error);
            EXPECT_THROW(range.require_local_adjacency(),std::logic_error);
            EXPECT_THROW(range.boundary().require_local_adjacency(),std::logic_error);
            EXPECT_THROW(range.boundary(987654).require_local_adjacency(),std::logic_error);
            EXPECT_THROW(static_cast<void>(range.empty()),std::logic_error);
            size_t callbacks=0;
            EXPECT_THROW(range.apply([&](const auto&){++callbacks;}),std::logic_error);
            EXPECT_THROW(range.visit_batches([&](const auto&){++callbacks;}),std::logic_error);
            EXPECT_THROW(static_cast<void>(range.boundary().size()),std::logic_error);
            EXPECT_EQ(callbacks,0U);
        }
    });
}

TEST(MeshTraverseTest, LayeredTypedInteriorRangeRejectsBackendReplacement)
{
    const auto native=layered_interior_fixture(3);
    Handle mesh(native);
    const auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto traversal=plan.acquire();
    traversal.cells().interior().apply([&](const auto& cell)
    {
        using Access=std::remove_cvref_t<decltype(cell.geometry())>;
        if constexpr (detail::is_layered_traverse_access<Access>)
        {
            const auto range=traversal.cell_faces(cell);
            mesh=Handle(layered_interior_fixture(3));
            EXPECT_THROW(static_cast<void>(range.size()),std::logic_error);
            EXPECT_THROW(range.require_local_adjacency(),std::logic_error);
            EXPECT_THROW(range.boundary().require_local_adjacency(),std::logic_error);
            EXPECT_THROW(range.boundary(987654).require_local_adjacency(),std::logic_error);
            size_t callbacks=0;
            EXPECT_THROW(range.apply([&](const auto&){++callbacks;}),std::logic_error);
            EXPECT_THROW(range.visit_batches([&](const auto&){++callbacks;}),std::logic_error);
            EXPECT_THROW(static_cast<void>(range.boundary(987654).size()),std::logic_error);
            EXPECT_EQ(callbacks,0U);
        }
    });
}

TEST(MeshTraverseTest, LayeredResolvedAdjacencyRetainsLiveGeometryAcrossAxialMotion)
{
    auto native=layered_interior_fixture(3);
    const Handle mesh(native);
    const auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto layout=plan.layered_layout();
    const auto check=[&]
    {
        const auto traversal=plan.acquire();
        traversal.cells().apply([&](const auto& cell)
        {
            const auto check_faces=[&](const auto& range)
            {
                range.interior().apply([&](const auto& face)
                {
                    using Access=std::remove_cvref_t<decltype(face.geometry())>;
                    if constexpr (detail::is_layered_traverse_access<Access>)
                    {
                        EXPECT_EQ(face.owner(),mesh.owner_cell(face.id()));
                        EXPECT_EQ(face.neighbor(),mesh.neighbor_cell(face.id()));
                        expect_vector_equal(face.cell_center_vector(),mesh.cell_center_vector(face.id(),cell.id()));
                    }
                });
            };
            check_faces(traversal.cell_faces(cell.id()));
            check_faces(traversal.cell_faces(cell));
        });
    };
    check();
    PlanarALEMeshMotion<Pack> motion(std::make_shared<Handle>(mesh));
    motion.begin_trial(1.2, 0.2);
    EXPECT_EQ(plan.layered_layout().get(),layout.get());
    check();
    motion.rollback_trial();
    check();
}
