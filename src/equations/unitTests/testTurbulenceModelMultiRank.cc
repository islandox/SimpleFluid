/**
 * @file testTurbulenceModelMultiRank.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief MPI regressions for rank-coherent turbulence validation and advance.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/turbulence/TurbulenceModel.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using Model = SimpleFluid::TurbulenceModel<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_distributed_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_box_database(4, 4, 4, 0.25));
}

SimpleFluid::MaterialPropertyFields<Pack> make_material(SimpleFluid::SP<const MeshType> mesh)
{
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions options;
    options.reference_density = 1.0;
    options.density = 1.0;
    options.specific_heat_capacity = 1.0;
    options.dynamic_viscosity = 1.0e-2;
    options.thermal_conductivity = 1.0e-1;
    return {std::move(mesh), options, time_options};
}

void require_multiple_ranks(const MeshType& mesh)
{
    if (mesh.owned_cell_map()->getComm()->getSize() < 2)
    {
        GTEST_SKIP() << "This test requires at least two MPI ranks.";
    }
}

template <class Field> void expect_positive_finite_and_uniform(const Field& field)
{
    using scalar_type = typename Pack::scalar_type;
    auto local_minimum = std::numeric_limits<scalar_type>::max();
    auto local_maximum = std::numeric_limits<scalar_type>::lowest();
    for (size_t local = 0; local < field.num_local_cells(); ++local)
    {
        const auto value = field.local_value(static_cast<Pack::local_ordinal_type>(local));
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GT(value, scalar_type{});
    }
    for (size_t owned = 0; owned < field.num_owned_cells(); ++owned)
    {
        const auto value = field.value(static_cast<Pack::local_ordinal_type>(owned));
        local_minimum = std::min(local_minimum, value);
        local_maximum = std::max(local_maximum, value);
    }
    scalar_type global_minimum = {};
    scalar_type global_maximum = {};
    const auto communicator = field.mesh().owned_cell_map()->getComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_minimum, &global_minimum);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_maximum, &global_maximum);
    EXPECT_NEAR(global_minimum, global_maximum, 1.0e-11);
}

/** @brief Verifies coherent model rollback after a rank-local material failure. */
TEST(TurbulenceModelMultiRankTest, RankLocalMaterialFailureThrowsCoherentlyWithoutEnablingModel)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    auto material = make_material(mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    ASSERT_GT(mesh->num_owned_cells(), 0U);
    if (rank == 0)
    {
        material.dynamic_viscosity.set_owned_value(0, std::numeric_limits<double>::quiet_NaN());
    }

    SimpleFluid::BoundaryConditionSet boundaries;
    Model model(mesh, boundaries);
    SimpleFluid::TurbulenceModelOptions options;
    options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    EXPECT_ANY_THROW(model.configure(options, material, 1.0));
    EXPECT_FALSE(model.enabled());
}

/** @brief Verifies global accepted-state preservation after a rank-local source failure. */
TEST(TurbulenceModelMultiRankTest, RankLocalScalarSourceFailurePreservesAcceptedStateOnEveryRank)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    FieldType state(mesh, 1.0, "accepted_state");
    FieldType diffusivity(mesh, 0.1, "diffusivity");
    SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "face_flux");
    SimpleFluid::TurbulenceScalarTransportEquation<Pack> equation(mesh);
    const auto source = [rank](Pack::local_ordinal_type cell_lid)
    { return rank == 0 && cell_lid == 0 ? -1.0 : 0.0; };
    const auto zero = [](Pack::local_ordinal_type) { return 0.0; };

    EXPECT_ANY_THROW(
        equation.advance(state, zero_flux, 0.1, diffusivity, state, source, zero, 1.0e-12));
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        EXPECT_DOUBLE_EQ(state.value(static_cast<Pack::local_ordinal_type>(owned)), 1.0);
    }
}

/** @brief Verifies rejection of rank-inconsistent model selection before allocation. */
TEST(TurbulenceModelMultiRankTest, RejectsRankInconsistentModelSelectionBeforeStateAllocation)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    auto material = make_material(mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::BoundaryConditionSet boundaries;
    Model model(mesh, boundaries);
    SimpleFluid::TurbulenceModelOptions options;
    options.model = rank == 0 ? SimpleFluid::TurbulenceModelType::Laminar
                              : SimpleFluid::TurbulenceModelType::StandardKEpsilon;

    EXPECT_ANY_THROW(model.configure(options, material, 1.0));
    EXPECT_FALSE(model.enabled());
}

/** @brief Verifies rejection of rank-inconsistent wall treatment before allocation. */
TEST(TurbulenceModelMultiRankTest, RejectsRankInconsistentWallTreatmentBeforeStateAllocation)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    auto material = make_material(mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::BoundaryConditionSet boundaries;
    boundaries.velocity["xmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    Model model(mesh, boundaries);
    SimpleFluid::TurbulenceModelOptions options;
    options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    if (rank == 0)
    {
        options.wall_treatment =
            SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
        options.wall_options.boundary_names = {"xmin"};
    }

    EXPECT_ANY_THROW(model.configure(options, material, 1.0));
    EXPECT_FALSE(model.enabled());
}

/** @brief Verifies rejection of rank-inconsistent wall constants before allocation. */
TEST(TurbulenceModelMultiRankTest, RejectsRankInconsistentWallConstantsBeforeStateAllocation)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    auto material = make_material(mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::BoundaryConditionSet boundaries;
    boundaries.velocity["xmin"] = {
        SimpleFluid::BoundaryConditionType::NoSlip, {}};
    Model model(mesh, boundaries);
    SimpleFluid::TurbulenceModelOptions options;
    options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    options.wall_treatment =
        SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon;
    options.wall_options.boundary_names = {"xmin"};
    if (rank == 0)
        options.wall_options.c_mu = 0.08;

    EXPECT_ANY_THROW(model.configure(options, material, 1.0));
    EXPECT_FALSE(model.enabled());
}

/** @brief Verifies coherent failure propagation after a rank-local database parse error. */
TEST(TurbulenceModelMultiRankTest, RankLocalDatabaseParseFailureThrowsCoherently)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    auto material = make_material(mesh);
    const auto rank = mesh->owned_cell_map()->getComm()->getRank();
    SimpleFluid::BoundaryConditionSet boundaries;
    Model model(mesh, boundaries);
    SimpleFluid::Database database;
    database.set(
        "turbulence_model",
        rank == 0 ? std::string{"notAModel"}
                  : std::string{"standardKEpsilon"});

    EXPECT_ANY_THROW(model.configure(database, material, 1.0));
    EXPECT_FALSE(model.enabled());
}

/** @brief Verifies uniform distributed advances for standard k-epsilon and SST. */
TEST(TurbulenceModelMultiRankTest, StandardKEpsilonAndSSTAdvanceUniformDistributedState)
{
    for (const auto type : {SimpleFluid::TurbulenceModelType::StandardKEpsilon,
                            SimpleFluid::TurbulenceModelType::SSTKOmega})
    {
        auto mesh = make_distributed_mesh();
        require_multiple_ranks(*mesh);
        auto material = make_material(mesh);
        SimpleFluid::BoundaryConditionSet boundaries;
        Model model(mesh, boundaries);
        SimpleFluid::TurbulenceModelOptions options;
        options.model = type;
        options.initial_turbulent_kinetic_energy = 0.1;
        options.initial_dissipation_rate = 0.01;
        options.initial_specific_dissipation_rate = 1.0;
        if (type == SimpleFluid::TurbulenceModelType::SSTKOmega)
        {
            options.initial_wall_distance = 0.25;
        }
        model.configure(options, material, 1.0);

        SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
        SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "projected_face_flux");
        const auto boundary_cache =
            SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
        const auto summary = model.advance(velocity, zero_flux, boundary_cache, 1.0e-3, material,
                                           1.0, SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);

        EXPECT_TRUE(summary.converged);
        EXPECT_EQ(summary.solves, 2);
        expect_positive_finite_and_uniform(model.turbulent_kinetic_energy());
        const auto* secondary = type == SimpleFluid::TurbulenceModelType::StandardKEpsilon
                                    ? model.dissipation_rate()
                                    : model.specific_dissipation_rate();
        ASSERT_NE(secondary, nullptr);
        expect_positive_finite_and_uniform(*secondary);
        expect_positive_finite_and_uniform(model.turbulent_kinematic_viscosity());
    }
}

/**
 * @brief Wall-distance replacement ignores stale caller ghosts and imports owned values.
 */
TEST(TurbulenceModelMultiRankTest, WallDistanceReplacementSynchronizesOwnedInputBeforeValidation)
{
    auto mesh = make_distributed_mesh();
    require_multiple_ranks(*mesh);
    auto material = make_material(mesh);
    SimpleFluid::BoundaryConditionSet boundaries;
    Model model(mesh, boundaries);
    SimpleFluid::TurbulenceModelOptions options;
    options.model = SimpleFluid::TurbulenceModelType::SSTKOmega;
    options.initial_turbulent_kinetic_energy = 0.1;
    options.initial_specific_dissipation_rate = 1.0;
    options.initial_wall_distance = 0.25;
    model.configure(options, material, 1.0);

    FieldType replacement(mesh, 0.4, "replacement_wall_distance");
    long long local_stale_ghosts = 0;
    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(local);
        if (mesh->is_owned_cell(cell_lid))
        {
            continue;
        }
        replacement.overlap_data().replaceLocalValue(
            cell_lid, std::numeric_limits<double>::quiet_NaN());
        ++local_stale_ghosts;
    }
    long long global_stale_ghosts = 0;
    Teuchos::reduceAll(
        *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1,
        &local_stale_ghosts, &global_stale_ghosts);
    ASSERT_GT(global_stale_ghosts, 0);

    ASSERT_NO_THROW(
        model.set_wall_distance(replacement, material, 1.0));
    ASSERT_NE(model.wall_distance(), nullptr);
    for (size_t local = 0; local < mesh->num_local_cells(); ++local)
    {
        const auto cell_lid =
            static_cast<Pack::local_ordinal_type>(local);
        EXPECT_DOUBLE_EQ(
            model.wall_distance()->local_value(cell_lid), 0.4);
    }
}

/** @brief Verifies wall-treatment/closure pairings across partitioned walls. */
TEST(TurbulenceModelMultiRankTest,
     WallTreatmentClosurePairingsAdvanceAcrossPartitionedWallBatches)
{
    /** @brief Wall-treatment case and the closure used to exercise it. */
    struct WallCase
    {
        SimpleFluid::TurbulenceModelType model;
        SimpleFluid::TurbulenceWallTreatmentType wall;
    };
    const WallCase cases[] = {
        {SimpleFluid::TurbulenceModelType::StandardKEpsilon,
         SimpleFluid::TurbulenceWallTreatmentType::StandardHighReKEpsilon},
        {SimpleFluid::TurbulenceModelType::StandardKEpsilon,
         SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReKEpsilon},
        {SimpleFluid::TurbulenceModelType::RealizableKEpsilon,
         SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReKEpsilon},
        {SimpleFluid::TurbulenceModelType::SSTKOmega,
         SimpleFluid::TurbulenceWallTreatmentType::ResolvedLowReSST}};

    for (const auto wall_case : cases)
    {
        auto mesh = make_distributed_mesh();
        require_multiple_ranks(*mesh);
        auto material = make_material(mesh);
        SimpleFluid::BoundaryConditionSet boundaries;
        boundaries.velocity["xmin"] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
        Model model(mesh, boundaries);
        SimpleFluid::TurbulenceModelOptions options;
        options.model = wall_case.model;
        options.initial_turbulent_kinetic_energy = 0.1;
        options.initial_dissipation_rate = 0.01;
        options.initial_specific_dissipation_rate = 1.0;
        options.wall_treatment = wall_case.wall;
        options.wall_options.boundary_names = {"xmin"};
        if (wall_case.model == SimpleFluid::TurbulenceModelType::SSTKOmega)
            options.initial_wall_distance = 0.25;
        model.configure(options, material, 1.0);

        SimpleFluid::VectorCellField<Pack> velocity(mesh, "velocity");
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            velocity.set_owned_value(
                static_cast<Pack::local_ordinal_type>(owned), {0.0, 1.0, 0.0});
        }
        velocity.sync_ghosts();
        SimpleFluid::FaceField<Pack> zero_flux(mesh, 0.0, "projected_face_flux");
        const auto boundary_cache =
            SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
        const auto summary = model.advance(
            velocity, zero_flux, boundary_cache, 1.0e-4, material, 1.0,
            SimpleFluid::FVM::NonOrthogonalTreatment::Explicit);

        EXPECT_TRUE(summary.converged);
        ASSERT_NE(model.wall_y_plus(), nullptr);
        ASSERT_NE(model.effective_dynamic_viscosity_boundary_cache(), nullptr);
        double local_max_y_plus = 0.0;
        for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            const auto lid = static_cast<Pack::local_ordinal_type>(owned);
            EXPECT_TRUE(std::isfinite(model.turbulent_kinetic_energy().value(lid)));
            EXPECT_GT(model.turbulent_kinetic_energy().value(lid), 0.0);
            EXPECT_TRUE(std::isfinite(model.wall_y_plus()->value(lid)));
            EXPECT_GE(model.wall_y_plus()->value(lid), 0.0);
            local_max_y_plus = std::max(
                local_max_y_plus, model.wall_y_plus()->value(lid));
        }
        double global_max_y_plus = 0.0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX,
                           1, &local_max_y_plus, &global_max_y_plus);
        EXPECT_GT(global_max_y_plus, 0.0);
    }
}

} // namespace
