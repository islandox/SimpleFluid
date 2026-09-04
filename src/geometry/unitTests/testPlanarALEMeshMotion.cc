/**
 * @file testPlanarALEMeshMotion.cc
 * @brief Tests transactional structured planar geometry motion and GCL data.
 */

#include <gtest/gtest.h>

#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Motion = SimpleFluid::PlanarALEMeshMotion<Pack>;
using Cartesian = Handle::Cartesian;
using Cylindrical = Handle::Cylindrical;
using SemiStructured = Handle::SemiStructured;
using local_ordinal_type = Pack::local_ordinal_type;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);

double maximum_local_gcl_residual(const Handle& mesh, const SimpleFluid::MeshMotionModel& motion)
{
    double maximum{};
    const auto old_volumes = motion.old_cell_volumes();
    const auto new_volumes = motion.new_cell_volumes();
    const auto mesh_fluxes = motion.face_mesh_fluxes();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        double balance{};
        for (const auto face : mesh.faces(cell))
        {
            balance += mesh.owner_cell(face) == cell ? mesh_fluxes[static_cast<size_t>(face)]
                                                     : -mesh_fluxes[static_cast<size_t>(face)];
        }
        const auto rate = (new_volumes[owned] - old_volumes[owned]) / motion.diagnostics().time_step;
        maximum = std::max(maximum, std::abs(rate - balance));
    }
    return maximum;
}

std::vector<Pack::global_ordinal_type> cell_geometry_ids(const Handle& mesh)
{
    std::vector<Pack::global_ordinal_type> result;
    result.reserve(mesh.num_local_cells());
    for (size_t local = 0; local < mesh.num_local_cells(); ++local)
    {
        result.push_back(mesh.cell_geometry_global_id(static_cast<local_ordinal_type>(local)));
    }
    return result;
}

std::vector<Pack::global_ordinal_type> face_ids(const Handle& mesh)
{
    std::vector<Pack::global_ordinal_type> result;
    result.reserve(mesh.num_faces());
    for (size_t local = 0; local < mesh.num_faces(); ++local)
    {
        result.push_back(mesh.face_global_id(static_cast<local_ordinal_type>(local)));
    }
    return result;
}

std::map<std::string, std::vector<Pack::global_ordinal_type>> boundary_signature(const Handle& mesh)
{
    std::map<std::string, std::vector<Pack::global_ordinal_type>> result;
    for (const auto& [batch_id, batch] : mesh.boundary_batches())
    {
        auto& ids = result[mesh.boundary_batch_name(batch_id)];
        ids.reserve(batch.face_lids.size());
        for (const auto face : batch.face_lids)
        {
            ids.push_back(mesh.face_global_id(face));
        }
        std::ranges::sort(ids);
    }
    return result;
}

} // namespace

TEST(PlanarALEMeshMotionTest, CartesianTrialExposesExactVolumesSweptFluxAndRollback)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 2.0}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    const auto cells_before = cell_geometry_ids(*mesh);
    const auto faces_before = face_ids(*mesh);
    const auto boundaries_before = boundary_signature(*mesh);
    const auto topology_before = mesh->vtu_topology();
    const auto cell_map_before = mesh->owned_cell_map();
    const auto face_map_before = mesh->owned_face_map();

    Motion motion(mesh);
    EXPECT_EQ(motion.mesh_family(), "OrthogonalCartesian3D");
    EXPECT_EQ(mesh->geometry_epoch(), 0U);

    motion.begin_trial(3.0, 2.0);

    EXPECT_TRUE(motion.has_active_trial());
    EXPECT_EQ(mesh->geometry_epoch(), 1U);
    EXPECT_EQ(motion.diagnostics().old_geometry_epoch, 0U);
    EXPECT_EQ(motion.diagnostics().new_geometry_epoch, 1U);
    EXPECT_DOUBLE_EQ(motion.diagnostics().old_surface_elevation, 2.0);
    EXPECT_DOUBLE_EQ(motion.diagnostics().new_surface_elevation, 3.0);
    ASSERT_EQ(motion.old_cell_volumes().size(), 2U);
    ASSERT_EQ(motion.new_cell_volumes().size(), 2U);
    EXPECT_DOUBLE_EQ(motion.old_cell_volumes()[0], 2.0);
    EXPECT_DOUBLE_EQ(motion.old_cell_volumes()[1], 2.0);
    EXPECT_DOUBLE_EQ(motion.new_cell_volumes()[0], 3.0);
    EXPECT_DOUBLE_EQ(motion.new_cell_volumes()[1], 3.0);
    EXPECT_LE(maximum_local_gcl_residual(*mesh, motion), 1.0e-14);
    EXPECT_LE(motion.diagnostics().maximum_absolute_gcl_residual, 1.0e-14);

    const auto interior_z_face = geometry->indexer().face_ordinal(Cartesian::FaceID{0, 0, 1, Cartesian::Z_FACE});
    const auto top_z_face = geometry->indexer().face_ordinal(Cartesian::FaceID{0, 0, 2, Cartesian::Z_FACE});
    for (size_t local = 0; local < mesh->num_faces(); ++local)
    {
        const auto face = static_cast<local_ordinal_type>(local);
        const auto global = static_cast<size_t>(mesh->face_global_id(face));
        const auto expected = global == interior_z_face ? 0.5 : global == top_z_face ? 1.0 : 0.0;
        EXPECT_DOUBLE_EQ(motion.face_mesh_fluxes()[local], expected);
    }

    EXPECT_EQ(cell_geometry_ids(*mesh), cells_before);
    EXPECT_EQ(face_ids(*mesh), faces_before);
    EXPECT_EQ(boundary_signature(*mesh), boundaries_before);
    const auto topology_after = mesh->vtu_topology();
    EXPECT_EQ(topology_after->connectivity, topology_before->connectivity);
    EXPECT_EQ(topology_after->cell_offsets, topology_before->cell_offsets);
    EXPECT_EQ(topology_after->cell_types, topology_before->cell_types);
    EXPECT_EQ(mesh->owned_cell_map(), cell_map_before);
    EXPECT_EQ(mesh->owned_face_map(), face_map_before);

    motion.rollback_trial();
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(mesh->geometry_epoch(), 2U);
    EXPECT_EQ(geometry->cell_edges()[Cartesian::Z], (SimpleFluid::ArrReal{0.0, 1.0, 2.0}));
    EXPECT_TRUE(std::ranges::all_of(motion.face_mesh_fluxes(), [](double value) { return value == 0.0; }));
}

TEST(PlanarALEMeshMotionTest, BufferedMotionAcceptsAndRepeatedCycleRecoversReferenceGeometry)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0, 3.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    SimpleFluid::PlanarALEMeshMotionOptions options;
    options.deformation_start_elevation = 1.0;
    Motion motion(mesh, options);

    motion.begin_trial(4.0, 1.0);
    EXPECT_EQ(geometry->cell_edges()[Cartesian::Z], (SimpleFluid::ArrReal{0.0, 1.0, 2.5, 4.0}));
    EXPECT_LE(maximum_local_gcl_residual(*mesh, motion), 1.0e-14);
    motion.accept_trial();
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(mesh->geometry_epoch(), 1U);
    EXPECT_DOUBLE_EQ(motion.diagnostics().new_surface_elevation, 4.0);
    const auto accepted_edges = geometry->cell_edges()[Cartesian::Z];
    EXPECT_THROW(motion.rollback_trial(), std::logic_error);
    EXPECT_EQ(geometry->cell_edges()[Cartesian::Z], accepted_edges);
    EXPECT_EQ(mesh->geometry_epoch(), 1U);

    motion.begin_trial(3.5, 1.0);
    EXPECT_NE(geometry->cell_edges()[Cartesian::Z], accepted_edges);
    motion.rollback_trial();
    EXPECT_EQ(geometry->cell_edges()[Cartesian::Z], accepted_edges);
    EXPECT_DOUBLE_EQ(motion.diagnostics().new_surface_elevation, 4.0);
    EXPECT_EQ(mesh->geometry_epoch(), 3U);

    motion.begin_trial(3.0, 1.0);
    EXPECT_EQ(geometry->cell_edges()[Cartesian::Z], (SimpleFluid::ArrReal{0.0, 1.0, 2.0, 3.0}));
    EXPECT_LE(maximum_local_gcl_residual(*mesh, motion), 1.0e-14);
    motion.accept_trial();
    EXPECT_EQ(mesh->geometry_epoch(), 4U);
}

TEST(PlanarALEMeshMotionTest, SharedGeometryPublishesOneEpochAndAllowsOneController)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    auto first_handle = std::make_shared<Handle>(geometry);
    auto alias_handle = std::make_shared<Handle>(geometry);
    EXPECT_EQ(first_handle->geometry_identity(), geometry.get());
    EXPECT_EQ(alias_handle->geometry_identity(), geometry.get());

    {
        Motion motion(first_handle);
        EXPECT_THROW({ Motion duplicate(alias_handle); }, std::invalid_argument);
        auto independent_geometry = std::make_shared<Cartesian>(*geometry);
        auto independent_handle = std::make_shared<Handle>(independent_geometry);
        EXPECT_NE(independent_handle->geometry_identity(),
                  first_handle->geometry_identity());
        EXPECT_NO_THROW({ Motion independent(independent_handle); });
        motion.begin_trial(3.0, 1.0);
        EXPECT_EQ(first_handle->geometry_epoch(), 1U);
        EXPECT_EQ(alias_handle->geometry_epoch(), 1U);
        motion.accept_trial();
    }

    EXPECT_NO_THROW({ Motion replacement(alias_handle); });
}

TEST(PlanarALEMeshMotionTest, ContractingMeshFluxUsesOwnerNormalSigns)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    Motion motion(mesh);

    motion.begin_trial(1.0, 1.0);
    const auto interior_z_face = geometry->indexer().face_ordinal(Cartesian::FaceID{0, 0, 1, Cartesian::Z_FACE});
    const auto top_z_face = geometry->indexer().face_ordinal(Cartesian::FaceID{0, 0, 2, Cartesian::Z_FACE});
    for (size_t local = 0; local < mesh->num_faces(); ++local)
    {
        const auto face = static_cast<local_ordinal_type>(local);
        const auto global = static_cast<size_t>(mesh->face_global_id(face));
        const auto expected = global == interior_z_face ? -0.5 : global == top_z_face ? -1.0 : 0.0;
        EXPECT_DOUBLE_EQ(motion.face_mesh_fluxes()[local], expected);
    }
    EXPECT_LE(maximum_local_gcl_residual(*mesh, motion), 1.0e-14);
}

TEST(PlanarALEMeshMotionTest, CartesianNonZAxisIsExactAndLevelChangeLimitIsPreMutation)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    SimpleFluid::PlanarALEMeshMotionOptions options;
    options.axis = SimpleFluid::Dimension::X;
    options.maximum_level_change = 0.6;
    Motion motion(mesh, options);

    motion.begin_trial(2.5, 0.5);
    EXPECT_EQ(geometry->cell_edges()[Cartesian::X], (SimpleFluid::ArrReal{0.0, 1.25, 2.5}));
    EXPECT_LE(maximum_local_gcl_residual(*mesh, motion), 1.0e-14);
    motion.accept_trial();

    const auto accepted_edges = geometry->cell_edges()[Cartesian::X];
    EXPECT_THROW(motion.begin_trial(3.2, 0.5), std::invalid_argument);
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(mesh->geometry_epoch(), 1U);
    EXPECT_EQ(geometry->cell_edges()[Cartesian::X], accepted_edges);
}

TEST(PlanarALEMeshMotionTest, CylindricalAxialMotionUpdatesSideGeometryAndSatisfiesGCL)
{
    auto geometry = std::make_shared<Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{1.0, 2.0}, {0.0, std::numbers::pi}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    const Cylindrical::FaceID radial_face{1, 0, 0, Cylindrical::R_FACE};
    const auto old_radial_area = geometry->face_area(radial_face);
    Motion motion(mesh);

    motion.begin_trial(3.0, 1.0);
    EXPECT_DOUBLE_EQ(geometry->cell_edges()[Cylindrical::AXIAL].back(), 3.0);
    EXPECT_NEAR(geometry->face_area(radial_face), 1.5 * old_radial_area, 1.0e-14);
    EXPECT_LE(maximum_local_gcl_residual(*mesh, motion), 2.0e-14);
    motion.accept_trial();
}

TEST(PlanarALEMeshMotionTest, SemiStructuredAxialMotionPreservesXYTopologyAndSatisfiesGCL)
{
    auto geometry = std::make_shared<SemiStructured>(
        SimpleFluid::Arr<SemiStructured::Vec3>{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}},
        SimpleFluid::Arr<SimpleFluid::Arr<unsigned>>{{0, 1, 2}}, SimpleFluid::ArrReal{0.0, 1.0, 2.0});
    auto mesh = std::make_shared<Handle>(geometry);
    const auto xy_nodes = geometry->xy_nodes();
    const auto xy_cells = geometry->xy_cell_nodes();
    Motion motion(mesh);

    motion.begin_trial(3.0, 1.0);
    EXPECT_EQ(geometry->xy_nodes(), xy_nodes);
    EXPECT_EQ(geometry->xy_cell_nodes(), xy_cells);
    EXPECT_EQ(geometry->z_edges(), (SimpleFluid::ArrReal{0.0, 1.5, 3.0}));
    EXPECT_LE(maximum_local_gcl_residual(*mesh, motion), 1.0e-14);
    motion.accept_trial();
}

TEST(PlanarALEMeshMotionTest, QualityFailureRollsBackBeforeReturningTheError)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    SimpleFluid::PlanarALEMeshMotionOptions options;
    options.quality_limits.maximum_aspect_ratio = 10.0;
    Motion motion(mesh, options);

    EXPECT_THROW(motion.begin_trial(100.0, 1.0), std::runtime_error);
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(mesh->geometry_epoch(), 2U);
    EXPECT_EQ(geometry->cell_edges()[Cartesian::Z], (SimpleFluid::ArrReal{0.0, 1.0, 2.0}));
    EXPECT_TRUE(std::ranges::equal(motion.old_cell_volumes(), motion.new_cell_volumes()));
    EXPECT_TRUE(std::ranges::all_of(motion.face_mesh_fluxes(), [](double value) { return value == 0.0; }));
}

TEST(PlanarALEMeshMotionTest, DestructorRollsBackAnUnresolvedTrial)
{
    auto geometry =
        std::make_shared<Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    {
        Motion motion(mesh);
        motion.begin_trial(2.0, 1.0);
        ASSERT_DOUBLE_EQ(geometry->cell_edges()[Cartesian::Z].back(), 2.0);
    }
    EXPECT_DOUBLE_EQ(geometry->cell_edges()[Cartesian::Z].back(), 1.0);
    EXPECT_EQ(mesh->geometry_epoch(), 2U);
}

TEST(PlanarALEMeshMotionTest, DestructorDoesNotOverwriteExternallyReplacedGeometry)
{
    auto geometry =
        std::make_shared<Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    auto motion = std::make_unique<Motion>(mesh);
    motion->begin_trial(2.0, 1.0);

    *geometry = Cartesian(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.5, 3.0}}});
    EXPECT_THROW(motion->accept_trial(), std::logic_error);
    motion.reset();

    EXPECT_EQ(geometry->cell_edges()[Cartesian::Z], (SimpleFluid::ArrReal{0.0, 1.5, 3.0}));
    EXPECT_EQ(mesh->geometry_epoch(), 0U);
}

TEST(PlanarALEMeshMotionTest, RejectsConstUnsupportedAndNonAxialGeometry)
{
    auto cartesian =
        std::make_shared<Cartesian>(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    Handle::CartesianPtr read_only = cartesian;
    auto const_backed = std::make_shared<Handle>(std::move(read_only));
    EXPECT_THROW({ Motion candidate(const_backed); }, std::invalid_argument);

    auto unstructured = SimpleFluid::test::make_unstructured_hex_line(1);
    auto unsupported = std::make_shared<Handle>(unstructured);
    EXPECT_THROW({ Motion candidate(unsupported); }, std::invalid_argument);

    auto cylindrical = std::make_shared<Cylindrical>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{1.0, 2.0}, {0.0, std::numbers::pi}, {0.0, 1.0}}});
    auto cylindrical_handle = std::make_shared<Handle>(cylindrical);
    SimpleFluid::PlanarALEMeshMotionOptions options;
    options.axis = SimpleFluid::Dimension::X;
    EXPECT_THROW(Motion(cylindrical_handle, options), std::invalid_argument);

    options.axis = static_cast<SimpleFluid::Dimension>(3);
    auto mutable_cartesian = std::make_shared<Handle>(cartesian);
    EXPECT_THROW(Motion(mutable_cartesian, options), std::invalid_argument);
}

TEST(PlanarALEMeshMotionTest, DetectsRawGeometryMutationOutsideTheEpochContract)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    Motion motion(mesh);

    *geometry = Cartesian(SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.1, 2.0}}});
    EXPECT_THROW(motion.begin_trial(2.5, 1.0), std::invalid_argument);
    EXPECT_EQ(mesh->geometry_epoch(), 0U);
    EXPECT_FALSE(motion.has_active_trial());
}

TEST(PlanarALEMeshMotionMultiRankTest, PartitionFaceFluxIsConsistentAndTargetsAreCollective)
{
    auto geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0, 3.0, 4.0}}});
    auto mesh = std::make_shared<Handle>(geometry);
    const auto comm = mesh->owned_cell_map()->getComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }
    Motion motion(mesh);

    const auto divergent_target = comm->getRank() == 0 ? 5.0 : 5.5;
    EXPECT_THROW(motion.begin_trial(divergent_target, 1.0), std::invalid_argument);
    EXPECT_EQ(mesh->geometry_epoch(), 0U);

    const auto divergent_time_step = comm->getRank() == 0 ? 1.0 : 2.0;
    EXPECT_THROW(motion.begin_trial(5.0, divergent_time_step), std::invalid_argument);
    EXPECT_EQ(mesh->geometry_epoch(), 0U);

    motion.begin_trial(5.0, 1.0);
    EXPECT_LE(maximum_local_gcl_residual(*mesh, motion), 1.0e-14);

    if (comm->getRank() == 0)
    {
        EXPECT_THROW(motion.accept_trial(), std::logic_error);
    }
    else
    {
        EXPECT_THROW(motion.rollback_trial(), std::logic_error);
    }
    EXPECT_TRUE(motion.has_active_trial());
    EXPECT_EQ(mesh->geometry_epoch(), 1U);

    const auto shared_geometry_face = geometry->indexer().face_ordinal(Cartesian::FaceID{0, 0, 2, Cartesian::Z_FACE});
    int local_found = 0;
    double local_flux = {};
    for (size_t local = 0; local < mesh->num_faces(); ++local)
    {
        const auto face_lid = static_cast<local_ordinal_type>(local);
        if (static_cast<size_t>(mesh->face_global_id(face_lid)) == shared_geometry_face)
        {
            local_found = 1;
            local_flux = motion.face_mesh_fluxes()[local];
            break;
        }
    }
    int global_found = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_found, &global_found);
    ASSERT_EQ(global_found, 2);
    double minimum_flux{};
    double maximum_flux{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_flux, &minimum_flux);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_flux, &maximum_flux);
    EXPECT_DOUBLE_EQ(minimum_flux, maximum_flux);
    EXPECT_DOUBLE_EQ(minimum_flux, 0.5);

    motion.accept_trial();
    unsigned long long local_epoch = mesh->geometry_epoch();
    unsigned long long minimum_epoch{};
    unsigned long long maximum_epoch{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_epoch, &minimum_epoch);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_epoch, &maximum_epoch);
    EXPECT_EQ(minimum_epoch, maximum_epoch);
    EXPECT_EQ(minimum_epoch, 1U);
}

TEST(PlanarALEMeshMotionMultiRankTest, RejectsTransverseGeometryAndInvalidAxisMismatch)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This test requires exactly two MPI ranks.";
    }

    const auto transverse_extent = comm->getRank() == 0 ? 1.0 : 2.0;
    auto divergent_geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, transverse_extent}, {0.0, 1.0}, {0.0, 1.0, 2.0, 3.0, 4.0}}});
    auto divergent_mesh = std::make_shared<Handle>(divergent_geometry);
    EXPECT_THROW({ Motion inconsistent_geometry(divergent_mesh); }, std::invalid_argument);

    auto common_geometry = std::make_shared<Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0, 3.0, 4.0}}});
    auto common_mesh = std::make_shared<Handle>(common_geometry);
    SimpleFluid::PlanarALEMeshMotionOptions options;
    options.axis = comm->getRank() == 0 ? SimpleFluid::Dimension::Z : static_cast<SimpleFluid::Dimension>(3);
    EXPECT_THROW(Motion(common_mesh, options), std::invalid_argument);
}
