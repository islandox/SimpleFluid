#pragma once

// Helpers retain translation-unit-local linkage. Register the test
// environment only in the executable's top-level source.

/** @file test_helpers.hh
 * @brief Translation-unit-local mesh traversal fixtures and reference checks.
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
    // Independently compiled cylindrical query paths can cancel to zero on
    // one path and leave a few rounding bits on the other.
    const auto scale = std::max({real_t{1}, actual.norm(), expected.norm()});
    const auto tolerance = 8 * std::numeric_limits<real_t>::epsilon() * scale;
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
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

