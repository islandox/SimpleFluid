#include <gtest/gtest.h>
#include "geometry/MeshHandle.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using ID = MultiRegionMesh::ID;
using Vec3 = MeshUtils::Vec3;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

void same_vector(Vec3 actual, Vec3 expected)
{
    EXPECT_NEAR(actual.x, expected.x, 1e-13);
    EXPECT_NEAR(actual.y, expected.y, 1e-13);
    EXPECT_NEAR(actual.z, expected.z, 1e-13);
}

void same_face(const MultiRegionMesh& mesh, const MultiRegionMesh::ResolvedFace& face)
{
    const auto f = face.face;
    ASSERT_LT(f, mesh.num_faces());
    EXPECT_EQ(face.owner, mesh.owner_cell(f));
    EXPECT_EQ(face.neighbor, mesh.neighbor_cell(f));
    EXPECT_EQ(face.interior(), mesh.is_interior_face(f));
    EXPECT_DOUBLE_EQ(face.area(), mesh.face_area(f));
    same_vector(face.normal(), mesh.face_normal(f));
    same_vector(face.centroid, mesh.face_centroid(f));
    same_vector(face.owner_centroid, mesh.cell_centroid(face.owner));
    for (const auto cell : {face.owner, face.neighbor})
    {
        if (cell == MultiRegionMesh::invalid_cell_id()) continue;
        EXPECT_EQ(face.opposite_cell(cell), mesh.opposite_cell(f, cell));
        same_vector(face.face_normal_outward(cell), mesh.face_normal_outward(f, cell));
        same_vector(face.face_area_vector_outward(cell), mesh.face_area_vector_outward(f, cell));
        same_vector(face.face_center_vector(cell), mesh.face_center_vector(f, cell));
        if (face.interior()) same_vector(face.cell_center_vector(cell), mesh.cell_center_vector(f, cell));
        else EXPECT_THROW(face.cell_center_vector(cell), std::invalid_argument);
    }
    if (face.interior()) same_vector(face.neighbor_centroid, mesh.cell_centroid(face.neighbor));
    EXPECT_THROW(face.opposite_cell(mesh.num_cells()), std::invalid_argument);
    EXPECT_THROW(face.face_center_vector(mesh.num_cells()), std::invalid_argument);
    EXPECT_THROW(face.face_normal_outward(mesh.num_cells()), std::invalid_argument);
    EXPECT_THROW(face.cell_center_vector(mesh.num_cells()), std::invalid_argument);
}

void same_geometry(const std::shared_ptr<MultiRegionMesh>& mesh)
{
    const auto execution = mesh->acquire_execution_view();
    size_t visited_cells = 0;
    execution.visit_region_geometry([&](size_t r, ID begin, const auto& geometry)
    {
        EXPECT_EQ(geometry.layout(), mesh->region_layout(r));
        for (ID native = 0; native < geometry.layout().cells; ++native)
        {
            const auto cell = begin + native;
            ++visited_cells;
            EXPECT_DOUBLE_EQ(geometry.cell_volume(native), mesh->cell_volume(cell));
            same_vector(geometry.cell_centroid(native), mesh->cell_centroid(cell));
            const auto checked = mesh->cell_faces(cell);
            size_t index = 0;
            geometry.visit_cell_faces(native, [&](const auto& face)
            {
                ASSERT_LT(index, checked.size());
                EXPECT_EQ(face.face, checked[index++]);
                same_face(*mesh, face);
            });
            EXPECT_EQ(index, checked.size());
        }
        EXPECT_THROW(geometry.cell_volume(geometry.layout().cells), std::out_of_range);
        EXPECT_THROW(geometry.cell_centroid(geometry.layout().cells), std::out_of_range);
        EXPECT_THROW(geometry.visit_cell_faces(geometry.layout().cells, [](const auto&) {}), std::out_of_range);
    });
    EXPECT_EQ(visited_cells, mesh->num_cells());
    for (ID face = 0; face < mesh->num_faces(); ++face) same_face(*mesh, execution.resolve_face(face));
    EXPECT_THROW(execution.resolve_face(mesh->num_faces()), std::out_of_range);
}

std::shared_ptr<MultiRegionMesh> native_periodic_cylinder()
{
    const auto pi = std::acos(-1.0);
    auto native = std::make_shared<OrthogonalCylindrial3D>(
        Vec3D<ArrReal>{{{1,1.5,2},{0,pi/2,pi,3*pi/2,2*pi},{0,0.5,1}}});
    EXPECT_TRUE(native->is_theta_periodic());
    return std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{native_region("cylinder", native)},
        std::vector<MultiRegionMesh::Interface>{});
}

std::shared_ptr<MultiRegionMesh> self_periodic_region()
{
    StructuredPatchInterface periodic{{0,0},{0,1}};
    periodic.periodic_translation = Vec3{1,0,0};
    return std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{
        cartesian_region("periodic", {{{0,0.5,1},{0,0.5,1},{0,0.5,1}}})},
        std::vector<MultiRegionMesh::Interface>{periodic});
}

std::vector<std::shared_ptr<MultiRegionMesh>> fixtures()
{
    return {test::two_regions(), test::coarse_fine_regions(), test::periodic_regions(),
        test::mixed_regions(), test::independent_extruded_regions(), test::cylindrical_regions(),
        native_periodic_cylinder(), self_periodic_region()};
}
}

TEST(ResolvedRegionGeometryTest, NativeTraversalAndResolvedFacesMatchCheckedGeometry)
{
    for (const auto& geometry : fixtures()) same_geometry(geometry);
}

TEST(ResolvedRegionGeometryTest, AcceptedMotionAndRollbackUseCompositeCoordinates)
{
    for (const auto& geometry : fixtures())
    {
        auto mesh = std::make_shared<MeshHandle<>>(geometry);
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.2, 0.1);
        motion.accept_trial();
        const auto accepted_volume = geometry->cell_volume(0);
        same_geometry(geometry);
        motion.begin_trial(1.35, 0.1);
        same_geometry(geometry);
        motion.rollback_trial();
        EXPECT_DOUBLE_EQ(geometry->cell_volume(0), accepted_volume);
        same_geometry(geometry);
        EXPECT_EQ(mesh->connectivity_storage_bytes(), 0U);
        EXPECT_FALSE(mesh->has_materialized_indexer());
    }
}

TEST(ResolvedRegionGeometryTest, ExplicitNativeRegionRetainsCanonicalGeometry)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Global explicit constituents are serial.";
    auto native = test::explicit_reference(*test::mixed_regions());
    auto geometry = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{native_region("explicit", native)},
        std::vector<MultiRegionMesh::Interface>{});
    same_geometry(geometry);
}

TEST(ResolvedRegionGeometryTest, ExecutionKeepsLeaseAndRejectsStaleConstituents)
{
    const Vec3D<ArrReal> edges{{{0,0.5,1},{0,1},{0,1}}};
    auto native = std::make_shared<OrthogonalCartesian3D>(edges);
    auto geometry = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{native_region("native", native)},
        std::vector<MultiRegionMesh::Interface>{});
    auto mesh = std::make_shared<MeshHandle<>>(geometry);
    PlanarALEMeshMotion<> motion(mesh);
    {
        const auto execution = geometry->acquire_execution_view();
        execution.visit_region_geometry([&](size_t, ID, const auto& region)
        {
            EXPECT_THROW(*native = OrthogonalCartesian3D(edges), std::logic_error);
            EXPECT_THROW(motion.begin_trial(1.2, 0.1), std::logic_error);
            region.visit_cell_faces(0, [&](const auto& face) { same_face(*geometry, face); });
        });
    }
    *native = OrthogonalCartesian3D(edges);
    EXPECT_THROW(static_cast<void>(geometry->acquire_execution_view()), std::logic_error);
    EXPECT_THROW(geometry->face_area(0), std::logic_error);
}

TEST(ResolvedRegionGeometryTest, EmptyExecutionViewRejectsResolvedQueries)
{
    const MultiRegionMesh::ExecutionView execution(nullptr);
    EXPECT_THROW(execution.resolve_face(0), std::logic_error);
    EXPECT_THROW(execution.visit_region_geometry([](size_t, ID, const auto&) {}), std::logic_error);
}
