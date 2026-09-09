/** @file testPlanarALEBoundary.cc @brief Moving planar boundary contract tests. */

#include <gtest/gtest.h>

#include "FVM/FaceFlux.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "solvers/PlanarALEBoundary.hh"
#include "solvers/VolumeContinuityModel.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
using Mesh = SimpleFluid::MeshHandle<Pack>;
using Motion = SimpleFluid::PlanarALEMeshMotion<Pack>;
using Flux = SimpleFluid::ScalarFaceFieldStored<Pack>;
using Velocity = SimpleFluid::VectorCellFieldStored<Pack>;
using FaceVelocity = SimpleFluid::VectorFaceFieldStored<Pack>;

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new KokkosEnvironment);
} // namespace

TEST(PlanarALEBoundaryTest, ValidatesGeometryPreservesTangentialSlipAndEnforcesZeroRelativeTopFlux)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    SimpleFluid::ConstantAreaVesselVolumeMap volume_map(0.0, 4.0, 1.0);
    SimpleFluid::PlanarALEBoundary<Pack> boundary(
        mesh, "zmax", SimpleFluid::Dimension::Z, volume_map, 1.0e-12, 1.0e-10);
    EXPECT_DOUBLE_EQ(boundary.diagnostics().surface_elevation, 2.0);
    EXPECT_DOUBLE_EQ(boundary.diagnostics().global_area, 1.0);
    EXPECT_DOUBLE_EQ(boundary.diagnostics().global_mesh_volume, 2.0);

    Motion motion(mesh);
    motion.begin_trial(3.0, 1.0);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    boundary.refresh(volume_map);

    auto cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        std::shared_ptr<const Mesh>(mesh), SimpleFluid::BoundaryConditionSet{});
    boundary.apply_kinematic_velocity(ale, cache);
    EXPECT_EQ(cache.type_by_name.at("zmax"), SimpleFluid::BoundaryConditionType::Slip);

    Velocity velocity(mesh, SimpleFluid::vec3<double>{2.0, -3.0, 7.0}, "tangential_velocity");
    velocity.sync_ghosts();
    FaceVelocity face_velocity(mesh, "slip_face_velocity");
    SimpleFluid::FVM::face_velocities(velocity, cache, face_velocity);

    int local_owned_top_faces = 0;
    if (boundary.batch_id() >= 0)
    {
        EXPECT_EQ(cache.type.at(boundary.batch_id()), SimpleFluid::BoundaryConditionType::Slip);
        const auto& boundary_values = cache.value.at(boundary.batch_id());
        ASSERT_EQ(boundary_values.size(), mesh->boundary_face_batch(boundary.batch_id()).face_lids.size());
        for (size_t in_batch = 0; in_batch < boundary_values.size(); ++in_batch)
        {
            const auto face_lid = mesh->boundary_face_batch(boundary.batch_id()).face_lids[in_batch];
            const auto expected_mesh_velocity =
                mesh->face_normal(face_lid) *
                (ale.face_mesh_fluxes()[static_cast<size_t>(face_lid)] / mesh->face_area(face_lid));
            EXPECT_DOUBLE_EQ(boundary_values[in_batch].x, expected_mesh_velocity.x);
            EXPECT_DOUBLE_EQ(boundary_values[in_batch].y, expected_mesh_velocity.y);
            EXPECT_DOUBLE_EQ(boundary_values[in_batch].z, expected_mesh_velocity.z);
            if (!face_velocity.is_owned_face(face_lid))
            {
                continue;
            }
            ++local_owned_top_faces;
            const auto owner = mesh->owner_cell(face_lid);
            const auto normal = mesh->face_normal_outward(face_lid, owner);
            const auto owner_velocity = velocity.local_value(owner);
            const auto expected_slip_velocity = owner_velocity - normal * owner_velocity.dot(normal);
            const auto actual = face_velocity.value(face_lid);
            EXPECT_DOUBLE_EQ(actual.x, expected_slip_velocity.x);
            EXPECT_DOUBLE_EQ(actual.y, expected_slip_velocity.y);
            EXPECT_DOUBLE_EQ(actual.z, expected_slip_velocity.z);
            EXPECT_DOUBLE_EQ(actual.x, 2.0);
            EXPECT_DOUBLE_EQ(actual.y, -3.0);
            EXPECT_DOUBLE_EQ(actual.dot(normal), 0.0);
        }
    }
    int global_owned_top_faces = 0;
    Teuchos::reduceAll(
        *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_owned_top_faces, &global_owned_top_faces);
    EXPECT_GT(global_owned_top_faces, 0);

    Flux absolute(mesh, 0.0, "absolute");
    Flux relative(mesh, 0.0, "relative");
    boundary.enforce_kinematic_flux(ale, absolute);
    SimpleFluid::FVM::mesh_relative_face_fluxes(absolute, ale, relative);
    int local_owned_flux_faces = 0;
    if (boundary.batch_id() >= 0)
    {
        for (const auto face_lid : mesh->boundary_face_batch(boundary.batch_id()).face_lids)
        {
            EXPECT_DOUBLE_EQ(absolute.local_value(face_lid), motion.face_mesh_fluxes()[static_cast<size_t>(face_lid)]);
            EXPECT_DOUBLE_EQ(relative.local_value(face_lid), 0.0);
            local_owned_flux_faces += absolute.is_owned_face(face_lid) ? 1 : 0;
        }
    }
    int global_owned_flux_faces = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_owned_flux_faces,
        &global_owned_flux_faces);
    EXPECT_GT(global_owned_flux_faces, 0);
    motion.rollback_trial();
}

TEST(PlanarALEBoundaryTest, RejectsMissingNonPlanarOrVesselInconsistentPatch)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    SimpleFluid::ConstantAreaVesselVolumeMap valid(0.0, 4.0, 1.0);
    EXPECT_THROW(
        (SimpleFluid::PlanarALEBoundary<Pack>(mesh, "missing", SimpleFluid::Dimension::Z, valid, 1.0e-12, 1.0e-10)),
        std::invalid_argument);
    SimpleFluid::ConstantAreaVesselVolumeMap wrong_area(0.0, 4.0, 2.0);
    EXPECT_THROW(
        (SimpleFluid::PlanarALEBoundary<Pack>(mesh, "zmax", SimpleFluid::Dimension::Z, wrong_area, 1.0e-12, 1.0e-10)),
        std::invalid_argument);
    EXPECT_THROW(
        (SimpleFluid::PlanarALEBoundary<Pack>(mesh, "xmax", SimpleFluid::Dimension::Z, valid, 1.0e-12, 1.0e-10)),
        std::invalid_argument);
}

TEST(PlanarALEBoundaryTest, ConvertsVolumeToleranceBeforeCheckingPatchArea)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    // At the current one-metre surface, the 0.05 m^3 volume mismatch is
    // inside the 0.1 m^3 tolerance. The 100 m vessel height converts that
    // tolerance to 0.001 m^2, so the 0.05 m^2 area mismatch must still fail.
    SimpleFluid::ConstantAreaVesselVolumeMap map(0.0, 100.0, 1.05);
    EXPECT_THROW((SimpleFluid::PlanarALEBoundary<Pack>(mesh, "zmax", SimpleFluid::Dimension::Z, map, 0.1, 0.0)),
        std::invalid_argument);
}

TEST(PlanarALEBoundaryTest, CollectivelyRejectsRankDivergentControlsAndForeignFields)
{
    const auto communicator = Tpetra::getDefaultComm();
    if (communicator->getSize() != 2)
    {
        GTEST_SKIP() << "This collective validation test requires two ranks.";
    }
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 0.5, 1.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    SimpleFluid::ConstantAreaVesselVolumeMap volume_map(0.0, 2.0, 1.0);

    auto every_rank_rejects = [&](auto&& operation, std::string_view expected_diagnostic)
    {
        int local_rejected = 0;
        std::string local_diagnostic;
        try
        {
            operation();
        }
        catch (const std::invalid_argument& error)
        {
            local_rejected = 1;
            local_diagnostic = error.what();
        }
        catch (const std::exception& error)
        {
            local_diagnostic = error.what();
        }
        int all_rejected = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_rejected, &all_rejected);
        EXPECT_EQ(all_rejected, 1);
        EXPECT_NE(local_diagnostic.find(expected_diagnostic), std::string::npos) << local_diagnostic;
    };

    every_rank_rejects(
        [&]
        {
            SimpleFluid::PlanarALEBoundary<Pack> boundary(mesh, communicator->getRank() == 0 ? "zmax" : "ymax",
                SimpleFluid::Dimension::Z, volume_map, 1.0e-12, 1.0e-10);
        },
        "boundary name must match exactly");
    every_rank_rejects(
        [&]
        {
            SimpleFluid::PlanarALEBoundary<Pack> boundary(mesh, communicator->getRank() == 0 ? "zmax" : "top",
                SimpleFluid::Dimension::Z, volume_map, 1.0e-12, 1.0e-10);
        },
        "boundary name must match exactly");
    every_rank_rejects(
        [&]
        {
            SimpleFluid::PlanarALEBoundary<Pack> boundary(mesh, "zmax",
                communicator->getRank() == 0 ? SimpleFluid::Dimension::Z : SimpleFluid::Dimension::Y, volume_map,
                1.0e-12, 1.0e-10);
        },
        "axis and tolerances must match exactly");
    every_rank_rejects(
        [&]
        {
            SimpleFluid::PlanarALEBoundary<Pack> boundary(mesh, "zmax", SimpleFluid::Dimension::Z, volume_map,
                communicator->getRank() == 0 ? 1.0e-12 : 2.0e-12, 1.0e-10);
        },
        "axis and tolerances must match exactly");
    every_rank_rejects(
        [&]
        {
            SimpleFluid::PlanarALEBoundary<Pack> boundary(mesh, communicator->getRank() == 0 ? "zmax" : "",
                SimpleFluid::Dimension::Z, volume_map, 1.0e-12, 1.0e-10);
        },
        "requires one non-empty boundary name");
    every_rank_rejects(
        [&]
        {
            SimpleFluid::ConstantAreaVesselVolumeMap rank_local_map(0.0, 2.0, communicator->getRank() == 0 ? 1.0 : 1.1);
            SimpleFluid::PlanarALEBoundary<Pack> boundary(
                mesh, "zmax", SimpleFluid::Dimension::Z, rank_local_map, 1.0e-12, 1.0e-10);
        },
        "vessel-map values and range policy must match exactly");
    every_rank_rejects(
        [&]
        {
            SimpleFluid::ConstantAreaVesselVolumeMap rank_local_map(
                0.0, communicator->getRank() == 0 ? 2.0 : 0.75, 1.0);
            SimpleFluid::PlanarALEBoundary<Pack> boundary(
                mesh, "zmax", SimpleFluid::Dimension::Z, rank_local_map, 1.0e-12, 1.0e-10);
        },
        "vessel map must be valid at the moving surface");
    every_rank_rejects(
        [&]
        {
            SimpleFluid::ConstantAreaVesselVolumeMap rank_local_map(0.0, 2.0, 1.0,
                communicator->getRank() == 0 ? SimpleFluid::FreeSurfaceRangePolicy::Error
                                             : SimpleFluid::FreeSurfaceRangePolicy::ClampAndReport);
            SimpleFluid::PlanarALEBoundary<Pack> boundary(
                mesh, "zmax", SimpleFluid::Dimension::Z, rank_local_map, 1.0e-12, 1.0e-10);
        },
        "vessel-map values and range policy must match exactly");

    SimpleFluid::PlanarALEBoundary<Pack> boundary(
        mesh, "zmax", SimpleFluid::Dimension::Z, volume_map, 1.0e-12, 1.0e-10);
    const auto accepted_diagnostics = boundary.diagnostics();
    SimpleFluid::ConstantAreaVesselVolumeMap rank_local_refresh_map(0.0, 2.0, communicator->getRank() == 0 ? 1.0 : 1.1);
    every_rank_rejects(
        [&] { boundary.refresh(rank_local_refresh_map); }, "vessel-map values and range policy must match exactly");
    EXPECT_DOUBLE_EQ(boundary.diagnostics().surface_elevation, accepted_diagnostics.surface_elevation);
    EXPECT_DOUBLE_EQ(boundary.diagnostics().global_area, accepted_diagnostics.global_area);
    EXPECT_DOUBLE_EQ(boundary.diagnostics().global_mesh_volume, accepted_diagnostics.global_mesh_volume);
    EXPECT_DOUBLE_EQ(boundary.diagnostics().mapped_pool_volume, accepted_diagnostics.mapped_pool_volume);
    EXPECT_DOUBLE_EQ(boundary.diagnostics().volume_mismatch, accepted_diagnostics.volume_mismatch);

    Motion motion(mesh);
    motion.begin_trial(1.25, 1.0);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    auto other_mesh = std::make_shared<Mesh>(geometry);
    auto matching_cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        std::shared_ptr<const Mesh>(mesh), SimpleFluid::BoundaryConditionSet{});
    auto foreign_cache = SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
        std::shared_ptr<const Mesh>(other_mesh), SimpleFluid::BoundaryConditionSet{});
    auto& selected_cache = communicator->getRank() == 0 ? matching_cache : foreign_cache;
    const auto accepted_cache = selected_cache;
    every_rank_rejects(
        [&] { boundary.apply_kinematic_velocity(ale, selected_cache); }, "velocity cache belongs to another mesh");
    EXPECT_EQ(selected_cache.type_by_name, accepted_cache.type_by_name);
    EXPECT_EQ(selected_cache.type, accepted_cache.type);
    EXPECT_EQ(selected_cache.value, accepted_cache.value);

    Flux matching_flux(mesh, 7.0, "matching");
    Flux foreign_flux(other_mesh, 7.0, "foreign");
    auto& selected_flux = communicator->getRank() == 0 ? matching_flux : foreign_flux;
    every_rank_rejects(
        [&] { boundary.enforce_kinematic_flux(ale, selected_flux); }, "face flux belongs to another mesh");
    for (size_t face = 0; face < selected_flux.num_local_faces(); ++face)
    {
        EXPECT_DOUBLE_EQ(selected_flux.local_value(static_cast<Pack::local_ordinal_type>(face)), 7.0);
    }
    motion.rollback_trial();
}

TEST(VolumeContinuityModelTest, UniformExpansionProducesExactIntegratedTargetAndClosure)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    Motion motion(mesh);
    motion.begin_trial(2.0, 1.0);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    Flux relative(mesh, 0.0, "relative");
    const std::vector<double> old_material{1.0};
    const std::vector<double> new_material{2.0};
    SimpleFluid::VolumeContinuityModel<Pack> model(mesh);
    const auto trial = model.preview({.ale = ale,
                                         .old_material_volume = old_material,
                                         .new_material_volume = new_material,
                                         .carrier_relative_flux = relative,
                                         .old_pool_volume = 1.0,
                                         .new_pool_volume = 2.0},
        1);
    EXPECT_DOUBLE_EQ(trial.target().integrated_rate(0), 1.0);
    EXPECT_DOUBLE_EQ(trial.diagnostics().global_material_source, 1.0);
    EXPECT_DOUBLE_EQ(trial.diagnostics().source_pool_closure_residual, 0.0);
    model.commit(trial);
    EXPECT_DOUBLE_EQ(model.continuity_target_field().value(0), 1.0);
    motion.rollback_trial();
}

TEST(VolumeContinuityModelTest, UniformContractionProducesExactNegativeTargetAndClosure)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    Motion motion(mesh);
    motion.begin_trial(0.5, 1.0);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    Flux relative(mesh, 0.0, "relative");
    const std::vector<double> old_material{1.0};
    const std::vector<double> new_material{0.5};
    SimpleFluid::VolumeContinuityModel<Pack> model(mesh);
    const auto trial = model.preview({.ale = ale,
                                         .old_material_volume = old_material,
                                         .new_material_volume = new_material,
                                         .carrier_relative_flux = relative,
                                         .old_pool_volume = 1.0,
                                         .new_pool_volume = 0.5},
        1);
    EXPECT_DOUBLE_EQ(trial.target().integrated_rate(0), -0.5);
    EXPECT_DOUBLE_EQ(trial.diagnostics().global_material_source, -0.5);
    EXPECT_DOUBLE_EQ(trial.diagnostics().source_pool_closure_residual, 0.0);
    motion.rollback_trial();
}

TEST(VolumeContinuityModelTest, ConvertsAbsoluteVolumeToleranceToRateWithTheTrialTimeStep)
{
    auto run = [](double time_step, double pool_volume_change, bool expect_acceptance)
    {
        auto geometry = std::make_shared<Mesh::Cartesian>(
            SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
        auto mesh = std::make_shared<Mesh>(geometry);
        Motion motion(mesh);
        motion.begin_trial(1.0, time_step);
        const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
        Flux relative(mesh, 0.0, "relative");
        const std::vector<double> material{1.0};
        SimpleFluid::VolumeContinuityModel<Pack> model(mesh, 0.1, 0.0);
        auto preview = [&]
        {
            return model.preview({.ale = ale,
                                     .old_material_volume = material,
                                     .new_material_volume = material,
                                     .carrier_relative_flux = relative,
                                     .old_pool_volume = 1.0,
                                     .new_pool_volume = 1.0 + pool_volume_change},
                1);
        };
        if (expect_acceptance)
        {
            EXPECT_NO_THROW(static_cast<void>(preview()));
        }
        else
        {
            EXPECT_THROW(static_cast<void>(preview()), std::runtime_error);
        }
        motion.rollback_trial();
    };

    // |Delta V|/dt = 0.075 m^3/s exceeds 0.1/2 = 0.05 m^3/s.
    run(2.0, 0.15, false);
    // 0.15 m^3/s is below 0.1/0.5 = 0.2 m^3/s.
    run(0.5, 0.075, true);
}

TEST(VolumeContinuityModelTest, BubbleEscapeIsRemovedOnceFromCarrierTarget)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    Motion motion(mesh);
    motion.begin_trial(0.5, 1.0);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    Flux relative(mesh, 0.0, "relative");
    Flux slip(mesh, 0.0, "bubble_slip_volume_flux");
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) == "zmax")
        {
            for (const auto face_lid : batch.face_lids)
            {
                if (slip.is_owned_face(face_lid))
                {
                    slip.set_owned_value(face_lid, 0.25);
                }
            }
        }
    }
    slip.sync_ghosts();
    const std::vector<double> old_material{1.0};
    const std::vector<double> new_material{0.5};
    SimpleFluid::VolumeContinuityModel<Pack> model(mesh);
    const auto trial = model.preview({.ale = ale,
                                         .old_material_volume = old_material,
                                         .new_material_volume = new_material,
                                         .carrier_relative_flux = relative,
                                         .bubble_slip_volume_flux = &slip,
                                         .old_pool_volume = 1.0,
                                         .new_pool_volume = 0.5,
                                         .bubble_escape_volume_rate = 0.25},
        2);
    EXPECT_DOUBLE_EQ(trial.diagnostics().global_material_source, -0.25);
    EXPECT_DOUBLE_EQ(trial.diagnostics().global_bubble_slip_divergence, 0.25);
    EXPECT_DOUBLE_EQ(trial.bubble_slip_contribution()[0], -0.25);
    EXPECT_DOUBLE_EQ(trial.target().integrated_rate(0), -0.5);
    EXPECT_DOUBLE_EQ(trial.diagnostics().source_pool_closure_residual, 0.0);
    motion.rollback_trial();
}

TEST(VolumeContinuityModelTest, ImplicitNonuniformCarrierAdvectionDoesNotCreateMaterialSource)
{
    if (Tpetra::getDefaultComm()->getSize() != 1)
    {
        GTEST_SKIP() << "This local analytic ordering check is serial.";
    }
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0, 2.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    Motion motion(mesh);
    constexpr double time_step = 0.25;
    motion.begin_trial(2.0, time_step);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);

    ASSERT_EQ(mesh->num_owned_cells(), 2U);
    const auto low =
        mesh->cell_centroid(0).z < mesh->cell_centroid(1).z ? Pack::local_ordinal_type{0} : Pack::local_ordinal_type{1};
    const auto high = low == 0 ? Pack::local_ordinal_type{1} : Pack::local_ordinal_type{0};
    Pack::local_ordinal_type interior = -1;
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid = static_cast<Pack::local_ordinal_type>(face);
        if (mesh->is_interior_face(face_lid))
        {
            interior = face_lid;
            break;
        }
    }
    ASSERT_GE(interior, 0);

    constexpr double flux_rate = 0.5;
    Flux relative(mesh, 0.0, "relative");
    relative.set_owned_value(interior, mesh->owner_cell(interior) == low ? flux_rate : -flux_rate);
    relative.sync_ghosts();

    std::vector<double> old_material(mesh->num_local_cells());
    std::vector<double> new_material(mesh->num_local_cells());
    old_material[static_cast<size_t>(low)] = 0.25;
    old_material[static_cast<size_t>(high)] = 0.75;
    const auto transported_low = 0.25 / (1.0 + time_step * flux_rate);
    new_material[static_cast<size_t>(low)] = transported_low;
    new_material[static_cast<size_t>(high)] = 0.75 + time_step * flux_rate * transported_low;

    SimpleFluid::VolumeContinuityModel<Pack> model(mesh);
    const auto trial = model.preview({.ale = ale,
                                         .old_material_volume = old_material,
                                         .new_material_volume = new_material,
                                         .carrier_relative_flux = relative,
                                         .old_pool_volume = 1.0,
                                         .new_pool_volume = 1.0},
        3);
    EXPECT_NEAR(trial.material_source()[static_cast<size_t>(low)], 0.0, 1.0e-15);
    EXPECT_NEAR(trial.material_source()[static_cast<size_t>(high)], 0.0, 1.0e-15);
    EXPECT_NEAR(trial.target().integrated_rate(low), 0.0, 1.0e-15);
    EXPECT_NEAR(trial.target().integrated_rate(high), 0.0, 1.0e-15);
    EXPECT_NEAR(trial.diagnostics().global_material_source, 0.0, 1.0e-15);
    EXPECT_NEAR(trial.diagnostics().source_pool_closure_residual, 0.0, 1.0e-15);
    motion.rollback_trial();
}

TEST(VolumeContinuityModelTest, TrialAndSnapshotGenerationsRejectForeignStaleAndRolledBackState)
{
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    Motion motion(mesh);
    motion.begin_trial(0.5, 1.0);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    Flux relative(mesh, 0.0, "relative");
    Flux slip(mesh, 0.0, "bubble_slip_volume_flux");
    for (const auto& [batch_id, batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) != "zmax")
        {
            continue;
        }
        for (const auto face_lid : batch.face_lids)
        {
            if (slip.is_owned_face(face_lid))
            {
                slip.set_owned_value(face_lid, 0.25);
            }
        }
    }
    slip.sync_ghosts();

    const std::vector<double> old_material{1.0};
    const std::vector<double> accepted_material{0.5};
    const std::vector<double> changed_material{0.25};
    const typename SimpleFluid::VolumeContinuityModel<Pack>::Inputs accepted_inputs{.ale = ale,
        .old_material_volume = old_material,
        .new_material_volume = accepted_material,
        .carrier_relative_flux = relative,
        .bubble_slip_volume_flux = &slip,
        .old_pool_volume = 1.0,
        .new_pool_volume = 0.5,
        .bubble_escape_volume_rate = 0.25};
    const typename SimpleFluid::VolumeContinuityModel<Pack>::Inputs changed_inputs{.ale = ale,
        .old_material_volume = old_material,
        .new_material_volume = changed_material,
        .carrier_relative_flux = relative,
        .old_pool_volume = 1.0,
        .new_pool_volume = 0.25};

    SimpleFluid::VolumeContinuityModel<Pack> model(mesh);
    SimpleFluid::VolumeContinuityModel<Pack> other(mesh);
    const auto foreign_trial = other.preview(accepted_inputs, 1);
    EXPECT_THROW(model.commit(foreign_trial), std::invalid_argument);

    const auto superseded = model.preview(accepted_inputs, 1);
    const auto accepted_trial = model.preview(accepted_inputs, 2);
    EXPECT_THROW(model.commit(superseded), std::logic_error);
    EXPECT_NO_THROW(model.commit(accepted_trial));
    EXPECT_THROW(model.commit(accepted_trial), std::logic_error);

    EXPECT_EQ(model.generation(), 2U);
    EXPECT_DOUBLE_EQ(model.material_source_field().value(0), -0.25);
    EXPECT_DOUBLE_EQ(model.slip_contribution_field().value(0), -0.25);
    EXPECT_DOUBLE_EQ(model.continuity_target_field().value(0), -0.5);
    const auto accepted_diagnostics = model.diagnostics();
    const auto accepted_snapshot = model.snapshot();

    const auto changed_trial = model.preview(changed_inputs, 3);
    model.commit(changed_trial);
    EXPECT_DOUBLE_EQ(model.material_source_field().value(0), -0.75);
    EXPECT_DOUBLE_EQ(model.slip_contribution_field().value(0), 0.0);
    EXPECT_DOUBLE_EQ(model.continuity_target_field().value(0), -0.75);

    const auto rollback_trial = model.preview(changed_inputs, 4);
    model.restore(accepted_snapshot);
    EXPECT_THROW(model.commit(rollback_trial), std::logic_error);
    EXPECT_EQ(model.generation(), 2U);
    EXPECT_DOUBLE_EQ(model.material_source_field().value(0), -0.25);
    EXPECT_DOUBLE_EQ(model.slip_contribution_field().value(0), -0.25);
    EXPECT_DOUBLE_EQ(model.continuity_target_field().value(0), -0.5);
    EXPECT_DOUBLE_EQ(model.diagnostics().old_material_volume, accepted_diagnostics.old_material_volume);
    EXPECT_DOUBLE_EQ(model.diagnostics().new_material_volume, accepted_diagnostics.new_material_volume);
    EXPECT_DOUBLE_EQ(model.diagnostics().global_material_source, accepted_diagnostics.global_material_source);
    EXPECT_DOUBLE_EQ(
        model.diagnostics().global_bubble_slip_divergence, accepted_diagnostics.global_bubble_slip_divergence);
    EXPECT_DOUBLE_EQ(model.diagnostics().bubble_escape_volume_rate, accepted_diagnostics.bubble_escape_volume_rate);
    EXPECT_DOUBLE_EQ(model.diagnostics().other_outflow_volume_rate, accepted_diagnostics.other_outflow_volume_rate);
    EXPECT_DOUBLE_EQ(
        model.diagnostics().source_pool_closure_residual, accepted_diagnostics.source_pool_closure_residual);
    EXPECT_DOUBLE_EQ(model.diagnostics().normalized_source_pool_closure_residual,
        accepted_diagnostics.normalized_source_pool_closure_residual);
    EXPECT_DOUBLE_EQ(model.diagnostics().maximum_target_change, accepted_diagnostics.maximum_target_change);

    const auto foreign_snapshot = other.snapshot();
    EXPECT_THROW(model.restore(foreign_snapshot), std::invalid_argument);
    const auto retry = model.preview(changed_inputs, 3);
    EXPECT_NO_THROW(model.commit(retry));
    EXPECT_EQ(model.generation(), 3U);
    motion.rollback_trial();
}

TEST(VolumeContinuityModelTest, CollectivelyRejectsRankDivergentGlobalAndOptionalInputs)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2)
    {
        GTEST_SKIP() << "This collective validation test requires two ranks.";
    }
    auto geometry = std::make_shared<Mesh::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0}, {0.0, 1.0}, {0.0, 0.25, 0.5, 0.75, 1.0}}});
    auto mesh = std::make_shared<Mesh>(geometry);
    Motion motion(mesh);
    motion.begin_trial(1.0, 1.0);
    const auto ale = SimpleFluid::FVM::make_ale_control_volume_state(*mesh, motion);
    Flux relative(mesh, 0.0, "relative");
    Flux exact_material_flux(mesh, 0.0, "exact_material_flux");
    std::vector<double> old_material(ale.old_cell_volumes().begin(), ale.old_cell_volumes().end());
    const std::vector<double> new_material(ale.new_cell_volumes().begin(), ale.new_cell_volumes().end());
    SimpleFluid::VolumeContinuityModel<Pack> model(mesh);

    auto every_rank_rejects = [&](const auto& inputs)
    {
        int local_rejected = 0;
        try
        {
            static_cast<void>(model.preview(inputs, 1));
        }
        catch (const std::exception&)
        {
            local_rejected = 1;
        }
        int all_rejected = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_rejected, &all_rejected);
        EXPECT_EQ(all_rejected, 1);
    };

    every_rank_rejects(typename SimpleFluid::VolumeContinuityModel<Pack>::Inputs{.ale = ale,
        .old_material_volume = old_material,
        .new_material_volume = new_material,
        .carrier_relative_flux = relative,
        .old_pool_volume = 1.0,
        .new_pool_volume = comm->getRank() == 0 ? 1.0 : 1.1});

    every_rank_rejects(typename SimpleFluid::VolumeContinuityModel<Pack>::Inputs{.ale = ale,
        .old_material_volume = old_material,
        .new_material_volume = new_material,
        .carrier_relative_flux = relative,
        .carrier_material_volume_flux = comm->getRank() == 0 ? &exact_material_flux : nullptr,
        .old_pool_volume = 1.0,
        .new_pool_volume = 1.0});

    if (comm->getRank() != 0)
    {
        old_material.pop_back();
    }
    every_rank_rejects(typename SimpleFluid::VolumeContinuityModel<Pack>::Inputs{.ale = ale,
        .old_material_volume = old_material,
        .new_material_volume = new_material,
        .carrier_relative_flux = relative,
        .old_pool_volume = 1.0,
        .new_pool_volume = 1.0});
    motion.rollback_trial();
}
