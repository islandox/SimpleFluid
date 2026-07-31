/**
 * @file testPhysicalEquations.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for separated physical equation classes.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "equations/BoussinesqMomentumEquation.hh"
#include "equations/PressureProjectionEquation.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "fields/FaceField.hh"
#include "fields/VectorCellField.hh"
#include "FVM/Operators.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "geometry/unitTests/test_skewed_prism_mesh_helpers.hh"
#include "utils/ErrorNorms.hh"
#include "utils/testing_environment.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace SimpleFluid::detail
{

/**
 * @brief Test-only access to linear-solver preconditioner setup counts.
 * @tparam Pack Tpetra type pack used by the solver.
 */
template<TpetraTypePack Pack>
struct BelosLinearSolverTestAccess
{
    static std::size_t preconditioner_setup_count(
        const BelosLinearSolver<Pack>& solver) noexcept
    {
        return solver.d_preconditioner_setup_count;
    }
};

/**
 * @brief Test-only access to cached pressure-projection state.
 * @tparam Pack Tpetra type pack used by the equation.
 */
template<TpetraTypePack Pack>
struct PressureProjectionEquationTestAccess
{
    static std::size_t preconditioner_setup_count(
        const PressureProjectionEquation<Pack>& equation) noexcept
    {
        return BelosLinearSolverTestAccess<Pack>::
            preconditioner_setup_count(equation.d_linear_solver);
    }

    static std::size_t predictor_flux_reuse_count(
        const PressureProjectionEquation<Pack>& equation) noexcept
    {
        return equation.d_cached_predictor_flux_reuse_count;
    }

    static real_t rhs_norm_reference(
        const PressureProjectionEquation<Pack>& equation) noexcept
    {
        return equation.d_rhs_norm_reference;
    }

    static auto project_reusing_cached_predictor(
        PressureProjectionEquation<Pack>& equation,
        CellField<Pack>& pressure,
        CellField<Pack>& pressure_correction,
        typename Pack::scalar_type time_step,
        typename Pack::scalar_type reference_density,
        const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
        VectorCellField<Pack>& velocity)
        -> typename PressureProjectionEquation<Pack>::ProjectionResult
    {
        return equation.project_reusing_cached_predictor(
            pressure,
            pressure_correction,
            time_step,
            reference_density,
            velocity_boundary_cache,
            velocity);
    }
};

} // namespace SimpleFluid::detail

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;
using VectorFieldType = SimpleFluid::VectorCellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

SimpleFluid::SP<MeshType> make_single_hex_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_single_hex_database());
}

SimpleFluid::SP<MeshType> make_2x2x2_mesh()
{
    return SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_2x2x2_database());
}

std::vector<Pack::scalar_type> local_values(const FieldType& field)
{
    std::vector<Pack::scalar_type> values(field.num_local_cells());
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(field.num_local_cells());
         ++lid)
    {
        values[static_cast<size_t>(lid)] = field.local_value(lid);
    }

    return values;
}

int advance_explicit_diffusion_until_converged(
    const SimpleFluid::TemperatureDiffusionEquation<Pack>& equation,
    FieldType& temperature,
    Pack::scalar_type time_step,
    Pack::scalar_type diffusivity,
    Pack::scalar_type update_tolerance,
    int max_steps)
{
    for (int step = 0; step < max_steps; ++step)
    {
        const auto old_temperature = local_values(temperature);
        equation.advance_explicit(old_temperature, time_step, diffusivity,
                                  temperature);

        Pack::scalar_type max_update = 0.0;
        for (MeshType::local_ordinal_type lid = 0;
             lid < static_cast<MeshType::local_ordinal_type>(
                       temperature.num_owned_cells());
             ++lid)
        {
            const auto update = std::abs(
                temperature.value(lid)
              - old_temperature[static_cast<size_t>(lid)]);
            if (update > max_update)
            {
                max_update = update;
            }
        }

        if (max_update < update_tolerance)
        {
            return step + 1;
        }
    }

    return max_steps;
}

double weighted_sum(const FieldType& field)
{
    const auto& mesh = field.mesh();
    double sum = 0.0;
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh.num_owned_cells());
         ++lid)
    {
        sum += field.value(lid) * mesh.cell_volume(lid);
    }

    return sum;
}

bool cell_has_exterior_face(const MeshType& mesh,
                            MeshType::local_ordinal_type cell_lid)
{
    for (const auto face_lid : mesh.faces(cell_lid))
    {
        if (mesh.is_exterior_face(face_lid))
        {
            return true;
        }
    }

    return false;
}

} // namespace

/** @brief Verifies enforcement of a Dirichlet boundary in temperature diffusion. */
TEST(PhysicalEquationsTest, TemperatureDiffusionAppliesDirichletBoundary)
{
    auto mesh = make_single_hex_mesh();
    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 0.0);
    temperature.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    equation.advance_explicit(local_values(temperature), 0.1, 1.0, temperature);

    EXPECT_NEAR(temperature.value(0), 0.2, 1.0e-12);
}

/** @brief Verifies explicit diffusion applies a prescribed outward gradient. */
TEST(PhysicalEquationsTest,
     TemperatureExplicitDiffusionAppliesNonzeroNeumannGradient)
{
    auto mesh = make_single_hex_mesh();
    FieldType temperature(mesh, 300.0, "temperature");

    constexpr double time_step = 0.1;
    constexpr double diffusivity = 0.5;
    constexpr double outward_gradient = 2.0;
    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["zmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann,
        outward_gradient};

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    equation.advance_explicit(
        local_values(temperature), time_step, diffusivity, temperature);

    const auto zmax_id = [&]
    {
        for (const auto& [batch_id, batch] : mesh->boundary_batches())
        {
            (void)batch;
            if (mesh->boundary_batch_name(batch_id) == "zmax")
            {
                return batch_id;
            }
        }
        return -1;
    }();
    ASSERT_GE(zmax_id, 0);
    const auto face_lid =
        mesh->boundary_batches().at(zmax_id).face_lids.front();
    const auto expected =
        300.0 + time_step * diffusivity * outward_gradient
              * mesh->face_area(face_lid) / mesh->cell_volume(0);
    EXPECT_NEAR(temperature.value(0), expected, 1.0e-12);
}

/** @brief Verifies that an explicit temperature step includes its source term. */
TEST(PhysicalEquationsTest, TemperatureExplicitStepAddsSourceTerm)
{
    auto mesh = make_single_hex_mesh();
    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 2.0);
    temperature.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    auto source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 3.0;
    };

    equation.advance_explicit(local_values(temperature), 0.1, 0.0,
                              temperature, source);

    EXPECT_NEAR(temperature.value(0), 2.3, 1.0e-12);
}

/**
 * @brief Verifies all three velocity components advance under Boussinesq buoyancy with gravity.
 */
TEST(PhysicalEquationsTest, BoussinesqMomentumAdvancesAllVelocityComponents)
{
    auto mesh = make_single_hex_mesh();
    FieldType temperature(mesh, "temperature");
    VectorFieldType velocity(mesh, "velocity");

    temperature.set_value(0, 1.0);
    velocity.set_value(0, {});
    temperature.sync_ghosts();
    velocity.sync_ghosts();

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.kinematic_viscosity = 0.0;
    options.thermal_expansion = 2.0;
    options.gravity_x = -10.0;
    options.gravity_y = -20.0;
    options.gravity_z = -30.0;
    options.reference_temperature = 0.5;

    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::BoussinesqMomentumEquation<Pack> equation(mesh);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    equation.advance_velocity(velocity,
                              zero_fluxes,
                              temperature,
                              cache,
                              options,
                              velocity);

    EXPECT_NEAR(velocity.value(0).x, 1.0, 1.0e-10);
    EXPECT_NEAR(velocity.value(0).y, 2.0, 1.0e-10);
    EXPECT_NEAR(velocity.value(0).z, 3.0, 1.0e-10);
}

/**
 * @brief Confirms the Boussinesq momentum equation accepts a caller-provided source term.
 */
TEST(PhysicalEquationsTest, BoussinesqMomentumAddsCallerSourceTerm)
{
    auto mesh = make_single_hex_mesh();
    FieldType temperature(mesh, "temperature");
    VectorFieldType velocity(mesh, "velocity");

    temperature.set_value(0, 0.5);
    velocity.set_value(0, {});
    temperature.sync_ghosts();
    velocity.sync_ghosts();

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.kinematic_viscosity = 0.0;
    options.thermal_expansion = 0.0;
    options.reference_temperature = 0.5;

    SimpleFluid::BoundaryConditionSet bcs;
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    auto source =
        [](MeshType::local_ordinal_type) -> SimpleFluid::vec3<Pack::scalar_type>
    {
        return {1.0, 2.0, 3.0};
    };

    SimpleFluid::BoussinesqMomentumEquation<Pack> equation(mesh);
    equation.advance_velocity(velocity,
                              zero_fluxes,
                              temperature,
                              cache,
                              options,
                              velocity,
                              source);

    EXPECT_NEAR(velocity.value(0).x, 0.1, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).y, 0.2, 1.0e-12);
    EXPECT_NEAR(velocity.value(0).z, 0.3, 1.0e-12);
}

/**
 * @brief Runs semi-implicit temperature advection in X, Y, and Z directions and verifies changes.
 */
TEST(PhysicalEquationsTest, TemperatureSemiImplicitAdvectionRunsInEachDirection)
{
    auto mesh = make_2x2x2_mesh();

    for (size_t component = 0; component < 3; ++component)
    {
        FieldType temperature(mesh, "temperature");
        VectorFieldType velocity(mesh, "velocity");

        for (MeshType::local_ordinal_type lid = 0;
             lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
             ++lid)
        {
            const auto& center = mesh->cell_centroid(lid);
            const auto value = component == 0 ? center.x
                             : (component == 1 ? center.y : center.z);
            temperature.set_value(lid, value);
            velocity.set_value(lid,
                               {component == 0 ? 1.0 : 0.0,
                                component == 1 ? 1.0 : 0.0,
                                component == 2 ? 1.0 : 0.0});
        }
        temperature.sync_ghosts();
        velocity.sync_ghosts();

        const auto old_temperature = local_values(temperature);
        SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
        SimpleFluid::FVM::face_fluxes(velocity, fluxes);

        SimpleFluid::BoundaryConditionSet bcs;
        SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
        equation.advance_semi_implicit(temperature, fluxes, 0.1, 0.0,
                                       temperature);

        bool changed = false;
        for (MeshType::local_ordinal_type lid = 0;
             lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
             ++lid)
        {
            EXPECT_TRUE(std::isfinite(temperature.value(lid)));
            changed = changed
                   || std::abs(temperature.value(lid)
                             - old_temperature[static_cast<size_t>(lid)])
                      > 1.0e-12;
        }
        EXPECT_TRUE(changed);
    }
}

/** @brief Verifies that a semi-implicit temperature step includes its source term. */
TEST(PhysicalEquationsTest, TemperatureSemiImplicitStepAddsSourceTerm)
{
    auto mesh = make_single_hex_mesh();
    FieldType temperature(mesh, "temperature");
    temperature.set_value(0, 2.0);
    temperature.sync_ghosts();

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    auto source =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 3.0;
    };

    equation.advance_semi_implicit(temperature, zero_fluxes, 0.1, 0.0,
                                   temperature, source);

    EXPECT_NEAR(temperature.value(0), 2.3, 1.0e-12);
}

/**
 * @brief An accepted temperature that already solves the new transport
 *        system is passed to Belos as the initial guess.
 */
TEST(PhysicalEquationsTest,
     TemperatureSemiImplicitWarmStartsFromAcceptedField)
{
    auto mesh = make_single_hex_mesh();
    FieldType old_temperature(mesh, 2.0, "old_temperature");
    FieldType temperature(mesh, 9.0, "temperature");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    SimpleFluid::LinearSolverOptions options;
    options.max_iterations = 1;
    options.tolerance = 1.0e-12;

    const auto statistics = equation.advance_semi_implicit(
        old_temperature, zero_fluxes, 0.1, 0.0, temperature, options);

    EXPECT_TRUE(statistics.converged);
    EXPECT_EQ(statistics.iterations, 0);
    EXPECT_DOUBLE_EQ(temperature.value(0), 2.0);
}

/**
 * @brief Physical temperature transport also seeds its candidate from the
 *        accepted field rather than from caller output storage.
 */
TEST(PhysicalEquationsTest,
     TemperaturePhysicalWarmStartsFromAcceptedField)
{
    auto mesh = make_single_hex_mesh();
    FieldType old_temperature(mesh, 275.0, "old_temperature");
    FieldType temperature(mesh, 9.0, "temperature");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions model_options;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, time_options);
    auto zero_power =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 1;
    linear_options.tolerance = 1.0e-12;

    const auto statistics = equation.advance_physical(
        old_temperature, zero_fluxes, 0.1, material, temperature,
        zero_power,
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
        linear_options);

    EXPECT_TRUE(statistics.converged);
    EXPECT_EQ(statistics.iterations, 0);
    EXPECT_DOUBLE_EQ(temperature.value(0), 275.0);
}

/** @brief Verifies that a rejected semi-implicit solve preserves aliased accepted state. */
TEST(PhysicalEquationsTest,
     TemperatureSemiImplicitRejectionPreservesAliasedAcceptedField)
{
    auto mesh = make_2x2x2_mesh();
    FieldType temperature(mesh, 0.0, "temperature");
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_owned_cells());
         ++cell_lid)
    {
        const auto index = static_cast<double>(cell_lid + 1);
        temperature.set_value(
            cell_lid, 1.0 + index * index);
    }
    temperature.sync_ghosts();
    const auto accepted_temperature = local_values(temperature);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    SimpleFluid::LinearSolverOptions options;
    options.max_iterations = 1;
    options.tolerance = 1.0e-14;

    try
    {
        equation.advance_semi_implicit(
            temperature,
            zero_fluxes,
            1.0,
            1.0,
            temperature,
            options);
        FAIL() << "Expected the under-iterated temperature solve to fail.";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(
            std::string(error.what()).find(
                "transport solve did not converge"),
            std::string::npos);
    }

    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_local_cells());
         ++cell_lid)
    {
        EXPECT_DOUBLE_EQ(
            temperature.local_value(cell_lid),
            accepted_temperature[static_cast<size_t>(cell_lid)]);
    }
}

/** @brief Verifies that a rejected physical-temperature solve preserves accepted state. */
TEST(PhysicalEquationsTest,
     TemperaturePhysicalRejectionPreservesAliasedAcceptedField)
{
    auto mesh = make_2x2x2_mesh();
    FieldType temperature(mesh, 0.0, "temperature");
    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_owned_cells());
         ++cell_lid)
    {
        const auto index = static_cast<double>(cell_lid + 1);
        temperature.set_value(
            cell_lid, 1.0 + index * index);
    }
    temperature.sync_ghosts();
    const auto accepted_temperature = local_values(temperature);

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.thermal_conductivity = 1.0;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, time_options);
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 1;
    linear_options.tolerance = 1.0e-14;
    auto zero_power =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };

    try
    {
        equation.advance_physical(
            temperature,
            zero_fluxes,
            1.0,
            material,
            temperature,
            zero_power,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
            linear_options);
        FAIL() << "Expected the under-iterated temperature solve to fail.";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_NE(
            std::string(error.what()).find(
                "physical transport solve did not converge"),
            std::string::npos);
    }

    for (MeshType::local_ordinal_type cell_lid = 0;
         cell_lid < static_cast<MeshType::local_ordinal_type>(
                        mesh->num_local_cells());
         ++cell_lid)
    {
        EXPECT_DOUBLE_EQ(
            temperature.local_value(cell_lid),
            accepted_temperature[static_cast<size_t>(cell_lid)]);
    }
}

/**
 * @brief A failed reused variable-coefficient assembly is discarded before
 *        the next physical-temperature advance.
 */
TEST(PhysicalEquationsTest,
     PhysicalTemperatureRecoversAfterCachedAssemblyThrows)
{
    auto mesh = make_single_hex_mesh();
    constexpr double initial_temperature = 275.0;
    FieldType temperature(
        mesh, initial_temperature, "recovery_temperature");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);

    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.density = 4.0;
    model_options.reference_density = 4.0;
    model_options.specific_heat_capacity = 5.0;
    model_options.thermal_conductivity = 1.0;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, time_options);
    auto zero_power =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };
    auto throwing_power =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        throw std::runtime_error("intentional assembly failure");
    };

    equation.advance_physical(
        temperature,
        zero_fluxes,
        0.1,
        material,
        temperature,
        zero_power,
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);

    EXPECT_THROW(
        equation.advance_physical(
            temperature,
            zero_fluxes,
            0.1,
            material,
            temperature,
            throwing_power,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit),
        std::runtime_error);

    const auto statistics = equation.advance_physical(
        temperature,
        zero_fluxes,
        0.1,
        material,
        temperature,
        zero_power,
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);
    EXPECT_TRUE(statistics.converged);
    EXPECT_NEAR(temperature.value(0), initial_temperature, 1.0e-12);
}

/** @brief Verifies conversion of a Neumann gradient into prescribed wall heat flux. */
TEST(PhysicalEquationsTest,
     PhysicalTemperatureNeumannGradientAddsPrescribedWallHeatFlux)
{
    auto mesh = make_single_hex_mesh();
    constexpr double initial_temperature = 300.0;
    constexpr double density = 1000.0;
    constexpr double heat_capacity = 4200.0;
    constexpr double conductivity = 0.588;
    constexpr double wall_heat_flux = 25.0;
    constexpr double time_step = 1.0;

    FieldType temperature(mesh, initial_temperature, "temperature");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["zmax"] = {
        SimpleFluid::BoundaryConditionType::Neumann,
        wall_heat_flux / conductivity};
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);

    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.reference_density = density;
    model_options.density = density;
    model_options.specific_heat_capacity = heat_capacity;
    model_options.thermal_conductivity = conductivity;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, time_options);
    auto zero_power =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };

    equation.advance_physical(
        temperature,
        zero_fluxes,
        time_step,
        material,
        temperature,
        zero_power,
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);

    double heated_area = 0.0;
    for (const auto& [batch_id, boundary_batch] : mesh->boundary_batches())
    {
        if (mesh->boundary_batch_name(batch_id) != "zmax")
        {
            continue;
        }
        for (const auto face_lid : boundary_batch.face_lids)
        {
            if (mesh->is_owned_face(face_lid))
            {
                heated_area += mesh->face_area(face_lid);
            }
        }
    }
    ASSERT_GT(heated_area, 0.0);
    const auto expected_temperature =
        initial_temperature
      + time_step * wall_heat_flux * heated_area
      / (density * heat_capacity * mesh->cell_volume(0));
    EXPECT_NEAR(temperature.value(0), expected_temperature, 1.0e-12);
}

/** @brief Verifies use and validation of a spatial conductivity override. */
TEST(PhysicalEquationsTest,
     PhysicalTemperatureUsesValidatedConductivityOverride)
{
    auto mesh = make_single_hex_mesh();
    FieldType molecular_temperature(mesh, 0.0, "molecular_temperature");
    FieldType effective_temperature(mesh, 0.0, "effective_temperature");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    SimpleFluid::TemperatureDiffusionEquation<Pack> molecular_equation(
        mesh, bcs);
    SimpleFluid::TemperatureDiffusionEquation<Pack> effective_equation(
        mesh, bcs);

    SimpleFluid::TimeStepperOptions time_options;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.thermal_conductivity = 0.0;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, time_options);
    FieldType effective_conductivity(mesh, 1.0, "effective_conductivity");
    auto zero_power =
        [](MeshType::local_ordinal_type) -> Pack::scalar_type
    {
        return 0.0;
    };

    molecular_equation.advance_physical(
        molecular_temperature,
        zero_fluxes,
        0.1,
        material,
        molecular_temperature,
        zero_power,
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit);
    effective_equation.advance_physical(
        effective_temperature,
        zero_fluxes,
        0.1,
        material,
        effective_temperature,
        zero_power,
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
        {},
        &effective_conductivity);

    EXPECT_DOUBLE_EQ(molecular_temperature.value(0), 0.0);
    EXPECT_GT(effective_temperature.value(0), 0.0);
    EXPECT_LT(effective_temperature.value(0), 1.0);

    FieldType negative_conductivity(
        mesh, -1.0, "negative_effective_conductivity");
    EXPECT_THROW(
        effective_equation.advance_physical(
            effective_temperature,
            zero_fluxes,
            0.1,
            material,
            effective_temperature,
            zero_power,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
            {},
            &negative_conductivity),
        std::invalid_argument);

    auto other_mesh = make_single_hex_mesh();
    FieldType wrong_mesh_conductivity(
        other_mesh, 1.0, "wrong_mesh_conductivity");
    EXPECT_THROW(
        effective_equation.advance_physical(
            effective_temperature,
            zero_fluxes,
            0.1,
            material,
            effective_temperature,
            zero_power,
            SimpleFluid::FVM::NonOrthogonalTreatment::Implicit,
            {},
            &wrong_mesh_conductivity),
        std::invalid_argument);
}

/** @brief Verifies pressure projection against an identity linear system. */
TEST(PhysicalEquationsTest, PressureProjectionSolvesIdentitySystem)
{
    auto mesh = make_single_hex_mesh();
    FieldType pressure(mesh, "pressure");
    pressure.set_value(0, 7.0);
    pressure.sync_ghosts();

    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;

    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh, options);
    equation.solve(pressure);

    EXPECT_NEAR(pressure.value(0), 0.0, 1.0e-12);
}

/** @brief Verifies that pressure projection selects MueLu by default. */
TEST(PhysicalEquationsTest, PressureProjectionUsesMueLuByDefault)
{
    auto mesh = make_2x2x2_mesh();
    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh);

    EXPECT_EQ(
        equation.linear_solver_options().preconditioner,
        SimpleFluid::LinearPreconditioner::MueLu);
}

/** @brief Verifies MueLu reuse until pressure-projection matrix reconstruction. */
TEST(PhysicalEquationsTest,
     PressureProjectionReusesMueLuUntilMatrixIsRebuilt)
{
    auto mesh = make_2x2x2_mesh();
    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(
        mesh, SimpleFluid::vec3{1.0, 0.0, 0.0}, "velocity");
    SimpleFluid::BoundaryConditionSet bcs;
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);

    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    options.preconditioner = SimpleFluid::LinearPreconditioner::MueLu;
    options.reuse_preconditioner = true;
    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh, options);
    const auto setup_count = [&]()
    {
        return SimpleFluid::detail::
            PressureProjectionEquationTestAccess<Pack>::
                preconditioner_setup_count(equation);
    };

    equation.project(pressure, 0.1, 1.0, cache, velocity);
    EXPECT_EQ(setup_count(), 1U);
    equation.project(pressure, 0.1, 1.0, cache, velocity);
    EXPECT_EQ(setup_count(), 1U);

    equation.rebuild_matrix();
    equation.project(pressure, 0.1, 1.0, cache, velocity);
    EXPECT_EQ(setup_count(), 2U);
}

/**
 * @brief Verify an adjacent PISO corrector reuses the preceding final flux
 *        without changing pressure, velocity, fluxes, or linear iterations.
 */
TEST(PhysicalEquationsTest,
     PressureProjectionCachedPredictorMatchesReconstruction)
{
    auto mesh = make_2x2x2_mesh();
    FieldType reconstructed_pressure(mesh, "reconstructed_pressure");
    FieldType reused_pressure(mesh, "reused_pressure");
    FieldType reconstructed_correction(mesh, "reconstructed_correction");
    FieldType reused_correction(mesh, "reused_correction");
    VectorFieldType reconstructed_velocity(
        mesh, "reconstructed_velocity");
    VectorFieldType reused_velocity(mesh, "reused_velocity");

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto center = mesh->cell_centroid(cell_lid);
        const auto pressure_value =
            2.0 * center.x - 0.5 * center.y + 0.25 * center.z;
        const SimpleFluid::vec3 velocity_value{
            center.x, -0.5 * center.y, 0.25 * center.z};
        reconstructed_pressure.set_value(cell_lid, pressure_value);
        reused_pressure.set_value(cell_lid, pressure_value);
        reconstructed_velocity.set_value(cell_lid, velocity_value);
        reused_velocity.set_value(cell_lid, velocity_value);
    }
    mesh->sync_periodic_boundaries(reconstructed_pressure);
    mesh->sync_periodic_boundaries(reused_pressure);
    mesh->sync_periodic_boundaries(reconstructed_velocity);
    mesh->sync_periodic_boundaries(reused_velocity);

    SimpleFluid::BoundaryConditionSet bcs;
    const auto velocity_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    options.preconditioner = SimpleFluid::LinearPreconditioner::None;
    SimpleFluid::PressureProjectionEquation<Pack> reconstructed_equation(
        mesh, options, bcs.pressure);
    SimpleFluid::PressureProjectionEquation<Pack> reused_equation(
        mesh, options, bcs.pressure);

    constexpr double time_step = 0.1;
    constexpr double reference_density = 1000.0;
    reconstructed_equation.project(
        reconstructed_pressure,
        reconstructed_correction,
        time_step,
        reference_density,
        velocity_cache,
        reconstructed_velocity);
    reused_equation.project(
        reused_pressure,
        reused_correction,
        time_step,
        reference_density,
        velocity_cache,
        reused_velocity);
    const auto first_rhs_norm =
        SimpleFluid::detail::PressureProjectionEquationTestAccess<Pack>::
            rhs_norm_reference(reused_equation);
    ASSERT_GT(first_rhs_norm, 0.0);

    const auto reconstructed_result =
        reconstructed_equation.project(
            reconstructed_pressure,
            reconstructed_correction,
            time_step,
            reference_density,
            velocity_cache,
            reconstructed_velocity);
    const auto reused_result =
        SimpleFluid::detail::PressureProjectionEquationTestAccess<Pack>::
            project_reusing_cached_predictor(
                reused_equation,
                reused_pressure,
                reused_correction,
                time_step,
                reference_density,
                velocity_cache,
                reused_velocity);

    EXPECT_TRUE(reconstructed_result.linear_solve.converged);
    EXPECT_TRUE(reused_result.linear_solve.converged);
    EXPECT_DOUBLE_EQ(
        SimpleFluid::detail::PressureProjectionEquationTestAccess<Pack>::
            rhs_norm_reference(reused_equation),
        first_rhs_norm);
    EXPECT_EQ(
        SimpleFluid::detail::PressureProjectionEquationTestAccess<Pack>::
            predictor_flux_reuse_count(reused_equation),
        1U);
    EXPECT_EQ(
        reconstructed_result.linear_solve.iterations,
        reused_result.linear_solve.iterations);
    EXPECT_DOUBLE_EQ(
        reconstructed_result.pressure_correction,
        reused_result.pressure_correction);
    EXPECT_DOUBLE_EQ(
        reconstructed_result.continuity,
        reused_result.continuity);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        EXPECT_DOUBLE_EQ(
            reconstructed_pressure.value(cell_lid),
            reused_pressure.value(cell_lid));
        EXPECT_DOUBLE_EQ(
            reconstructed_correction.value(cell_lid),
            reused_correction.value(cell_lid));
        const auto reconstructed_value =
            reconstructed_velocity.value(cell_lid);
        const auto reused_value = reused_velocity.value(cell_lid);
        EXPECT_DOUBLE_EQ(reconstructed_value.x, reused_value.x);
        EXPECT_DOUBLE_EQ(reconstructed_value.y, reused_value.y);
        EXPECT_DOUBLE_EQ(reconstructed_value.z, reused_value.z);
    }

    const auto& reconstructed_fluxes =
        reconstructed_equation.corrected_face_fluxes();
    const auto& reused_fluxes = reused_equation.corrected_face_fluxes();
    for (size_t face = 0; face < mesh->num_faces(); ++face)
    {
        const auto face_lid =
            static_cast<MeshType::local_ordinal_type>(face);
        if (!reconstructed_fluxes.is_owned_face(face_lid))
        {
            continue;
        }
        EXPECT_DOUBLE_EQ(
            reconstructed_fluxes.value(face_lid),
            reused_fluxes.value(face_lid));
    }
}

/**
 * @brief Verifies an external source term is incorporated into the Poisson RHS during pressure projection.
 */
TEST(PhysicalEquationsTest, PressureProjectionAddsSourceTermToPoissonRhs)
{
    auto db = SimpleFluid::test::make_box_database(2, 1, 1, 0.5);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(db);
    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(mesh, SimpleFluid::vec3{}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh);
    auto source =
        [](MeshType::local_ordinal_type cell_lid) -> Pack::scalar_type
    {
        return cell_lid == 0 ? 0.0 : 4.0;
    };

    constexpr double reference_density = 1000.0;
    equation.project(
        pressure,
        1.0,
        reference_density,
        cache,
        velocity,
        source);

    EXPECT_NEAR(pressure.value(0), 0.0, 1.0e-12);
    EXPECT_NEAR(pressure.value(1), reference_density, 1.0e-8);
}

/** @brief Verifies that a pressure Dirichlet boundary replaces the gauge constraint. */
TEST(PhysicalEquationsTest,
     PressureProjectionDirichletBoundaryReplacesGaugeConstraint)
{
    auto db = SimpleFluid::test::make_box_database(2, 1, 1, 0.5);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(db);
    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(mesh, SimpleFluid::vec3{}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.pressure["xmax"] = {
        SimpleFluid::BoundaryConditionType::Dirichlet, 37.0};
    const auto velocity_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;
    options.preconditioner = SimpleFluid::LinearPreconditioner::None;
    SimpleFluid::PressureProjectionEquation<Pack> equation(
        mesh, options, bcs.pressure);
    auto source =
        [](MeshType::local_ordinal_type cell_lid) -> Pack::scalar_type
    {
        return cell_lid == 0 ? 4.0 : 0.0;
    };

    constexpr double reference_density = 1000.0;
    equation.project(
        pressure,
        1.0,
        reference_density,
        velocity_cache,
        velocity,
        source);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto expected =
            mesh->cell_centroid(cell_lid).x < 0.5
          ? 1.5 * reference_density
          : 0.5 * reference_density;
        EXPECT_NEAR(pressure.value(cell_lid), expected, 1.0e-8);
    }
}

/**
 * @brief Ensures pressure projection reduces the divergence of the velocity flux field.
 */
TEST(PhysicalEquationsTest, PressureProjectionReducesFluxDivergence)
{
    auto mesh = make_2x2x2_mesh();
    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 0.0, 0.0}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    auto divergence_norm = [&]()
    {
        const auto cache =
            SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
                mesh, bcs);
        SimpleFluid::FaceField<Pack> fluxes(mesh, "face_flux");
        SimpleFluid::FVM::face_fluxes(velocity, cache, fluxes);
        const auto divergence =
            SimpleFluid::FVM::cell_divergence_from_fluxes<Pack>(
                *mesh, fluxes);

        Pack::scalar_type norm = 0.0;
        for (const auto value : divergence)
        {
            norm += value * value;
        }
        return std::sqrt(norm);
    };

    const auto before = divergence_norm();

    SimpleFluid::LinearSolverOptions options;
    options.tolerance = 1.0e-12;

    SimpleFluid::PressureProjectionEquation<Pack> equation(mesh, options);
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    equation.project(pressure, 0.1, 1.0, cache, velocity);

    const auto after = divergence_norm();
    EXPECT_LT(after, before);
}

/**
 * @brief Verifies that explicit thermal diffusion in 1D produces the
 *        correct linear steady-state profile T(x) = 1 - x/L.
 *
 * A 10×1×1 box mesh is insulated on all faces except xmin (T = 1) and
 * xmax (T = 0).  After a sufficient number of explicit time steps the
 * numerical solution should converge to the linear analytical profile.
 */
TEST(PhysicalEquationsTest, Steady1DDiffusionMatchesLinearProfile)
{
    // 5 cells along X, domain [0, 1].  Smaller mesh converges faster.
    constexpr double domain_length = 1.0;
    constexpr int n_cells = 5;
    auto db = SimpleFluid::test::make_box_database(
        n_cells, 1, 1, domain_length / n_cells);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(db);
    FieldType temperature(mesh, "temperature");

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] =
        {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] =
        {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    bcs.temperature["ymin"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["ymax"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["zmin"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["zmax"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        temperature.set_owned_value(
            static_cast<MeshType::local_ordinal_type>(owned), 0.0);
    }
    temperature.sync_ghosts();

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);

    constexpr double diffusivity = 1.0;
    constexpr double time_step = 1.0e-3;
    constexpr double steady_update_tolerance = 1.0e-12;
    constexpr int max_steps = 10000;
    const auto steps_taken = advance_explicit_diffusion_until_converged(
        equation, temperature, time_step, diffusivity,
        steady_update_tolerance, max_steps);
    EXPECT_LT(steps_taken, max_steps);

    auto exact = [](SimpleFluid::vec3<> pos)
    {
        return 1.0 - pos.x / domain_length;
    };

    const auto l2 = SimpleFluid::l2_error(temperature, exact);
    const auto linf = SimpleFluid::linf_error(temperature, exact);

    EXPECT_LT(l2, 5.0e-2);
    EXPECT_LT(linf, 1.0e-1);
}

/**
 * @brief Sanity check that a single explicit diffusion step changes
 *        the temperature at a Dirichlet-adjacent cell.
 */
TEST(PhysicalEquationsTest, ExplicitDiffusionUpdatesDirichletCell)
{
    auto db = SimpleFluid::test::make_box_database(3, 1, 1, 0.1);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(db);
    FieldType temperature(mesh, "temperature");

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] =
        {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["xmax"] =
        {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        temperature.set_owned_value(
            static_cast<MeshType::local_ordinal_type>(owned), 0.0);
    }
    temperature.sync_ghosts();

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    const auto old_t = local_values(temperature);
    equation.advance_explicit(old_t, 0.1, 1.0, temperature);

    // After one step, the cell next to the hot wall must have warmed up.
    EXPECT_GT(temperature.value(0), 0.0);
}

/**
 * @brief Verifies that explicit diffusion preserves a uniform scalar
 *        field (zero-gradient everywhere → zero Laplacian).
 *
 * A uniform field should remain unchanged under pure diffusion because
 * the Laplacian of a constant is identically zero.
 */
TEST(PhysicalEquationsTest, ExplicitDiffusionPreservesUniformField)
{
    auto mesh = make_2x2x2_mesh();
    FieldType temperature(mesh, "temperature");

    // Set uniform value 0.5 everywhere.
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        temperature.set_value(lid, 0.5);
    }
    temperature.sync_ghosts();

    // Insulated on all sides — zero-flux Neumann everywhere.
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.temperature[name] =
            {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    }

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);

    const auto old_t = local_values(temperature);
    equation.advance_explicit(old_t, 0.1, 1.0, temperature);

    // Uniform field must remain unchanged.
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        EXPECT_NEAR(temperature.value(lid), 0.5, 1.0e-12);
    }
}

/** @brief Verifies physical-equation advances on skewed triangular-prism cells. */
TEST(PhysicalEquationsTest, SkewedTriangularPrismCellsAdvancePhysicalEquations)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    ASSERT_GE(mesh->num_owned_cells(), 3u * 3u * 3u);
    EXPECT_EQ(mesh->boundary_batches().size(), 6u);

    bool saw_triangular_face = false;
    size_t boundary_cells = 0;
    size_t interior_cells = 0;
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        EXPECT_EQ(mesh->cell(lid).type, SimpleFluid::MeshUtils::CellType::TRIPRISM);
        EXPECT_GT(mesh->cell_volume(lid), 0.0);
        if (cell_has_exterior_face(*mesh, lid))
        {
            ++boundary_cells;
        }
        else
        {
            ++interior_cells;
        }
    }
    for (MeshType::local_ordinal_type fid = 0;
         fid < static_cast<MeshType::local_ordinal_type>(mesh->num_faces());
         ++fid)
    {
        EXPECT_TRUE(std::isfinite(mesh->face_area(fid)));
        EXPECT_GT(mesh->face_area(fid), 0.0);
        saw_triangular_face =
            saw_triangular_face
         || mesh->face(fid).type == SimpleFluid::MeshUtils::FaceType::TRIANGLE;
    }
    EXPECT_TRUE(saw_triangular_face);
    EXPECT_GT(boundary_cells, 0u);
    EXPECT_GT(interior_cells, 0u);

    FieldType temperature(mesh, "temperature");
    VectorFieldType velocity(mesh, "velocity");
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto center = mesh->cell_centroid(lid);
        temperature.set_value(lid, 1.0 + center.x - 0.25 * center.y);
        velocity.set_value(lid, {center.x, -0.5 * center.y, 0.25 * center.z});
    }
    temperature.sync_ghosts();
    velocity.sync_ghosts();

    const auto old_temperature = local_values(temperature);
    const auto old_temperature_integral = weighted_sum(temperature);

    SimpleFluid::BoundaryConditionSet bcs;
    SimpleFluid::TemperatureDiffusionEquation<Pack> temperature_equation(
        mesh, bcs);
    temperature_equation.advance_explicit(old_temperature, 0.01, 0.2,
                                          temperature);

    EXPECT_NEAR(weighted_sum(temperature), old_temperature_integral, 1.0e-10);
    bool temperature_changed = false;
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        EXPECT_TRUE(std::isfinite(temperature.value(lid)));
        temperature_changed =
            temperature_changed
         || std::abs(temperature.value(lid)
                   - old_temperature[static_cast<size_t>(lid)]) > 1.0e-14;
    }
    EXPECT_TRUE(temperature_changed);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.02;
    options.kinematic_viscosity = 0.1;
    options.thermal_expansion = 0.0;
    options.reference_temperature = 0.5;

    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    std::vector<VectorFieldType::vec_type> old_velocity(mesh->num_owned_cells());
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        old_velocity[static_cast<size_t>(lid)] = velocity.value(lid);
    }

    SimpleFluid::BoussinesqMomentumEquation<Pack> momentum_equation(mesh);
    momentum_equation.advance_velocity(velocity,
                                       zero_fluxes,
                                       temperature,
                                       cache,
                                       options,
                                       velocity);

    bool velocity_changed = false;
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto v = velocity.value(lid);
        EXPECT_TRUE(std::isfinite(v.x));
        EXPECT_TRUE(std::isfinite(v.y));
        EXPECT_TRUE(std::isfinite(v.z));
        const auto old_v = old_velocity[static_cast<size_t>(lid)];
        velocity_changed =
            velocity_changed
         || std::abs(v.x - old_v.x) > 1.0e-14
         || std::abs(v.y - old_v.y) > 1.0e-14
         || std::abs(v.z - old_v.z) > 1.0e-14;
    }
    EXPECT_TRUE(velocity_changed);
}

/**
 * @brief Verifies one explicit diffusion step against the analytical
 *        Laplacian of T(x) = x^2 on interior cells.
 */
TEST(PhysicalEquationsTest, ExplicitDiffusionMatchesQuadraticLaplacian)
{
    constexpr double domain_length = 1.0;
    constexpr int n_cells = 5;
    constexpr double cell_width = domain_length / n_cells;
    constexpr double diffusivity = 0.25;
    constexpr double time_step = 1.0e-2;

    auto db = SimpleFluid::test::make_box_database(
        n_cells, 1, 1, cell_width);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(db);
    FieldType temperature(mesh, "temperature");

    auto exact = [](SimpleFluid::vec3<> pos)
    {
        return pos.x * pos.x;
    };

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        temperature.set_value(lid, exact(mesh->cell_centroid(lid)));
    }
    temperature.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["xmin"] =
        {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};
    bcs.temperature["xmax"] =
        {SimpleFluid::BoundaryConditionType::Dirichlet,
         domain_length * domain_length};
    bcs.temperature["ymin"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["ymax"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["zmin"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};
    bcs.temperature["zmax"] =
        {SimpleFluid::BoundaryConditionType::Neumann, 0.0};

    const auto old_temperature = local_values(temperature);
    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
    equation.advance_explicit(old_temperature, time_step, diffusivity,
                              temperature);

    bool checked_interior = false;
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto& center = mesh->cell_centroid(lid);
        if (center.x <= cell_width || center.x >= domain_length - cell_width)
        {
            continue;
        }

        checked_interior = true;
        const auto expected =
            old_temperature[static_cast<size_t>(lid)]
          + time_step * diffusivity * 2.0;
        EXPECT_NEAR(temperature.value(lid), expected, 1.0e-12);
    }

    EXPECT_TRUE(checked_interior);
}

/**
 * @brief Verifies that semi-implicit momentum diffusion with no
 *        convection and no buoyancy preserves a zero velocity field
 *        and produces finite results for a non-zero field.
 *
 * A 4×1×1 box mesh with a sinusoidal velocity in X is advanced one step
 * with the Boussinesq momentum equation.  The result must be finite and
 * the velocity must change under viscous diffusion.
 */
TEST(PhysicalEquationsTest, MomentumDiffusionAdvancesVelocityField)
{
    constexpr int n_cells = 4;
    auto db = SimpleFluid::test::make_box_database(
        n_cells, 1, 1, 1.0 / n_cells);
    auto mesh = SimpleFluid::test::build_mesh<Pack>(db);
    FieldType temperature(mesh, "temperature");
    VectorFieldType velocity(mesh, "velocity");

    // Set temperature uniform to eliminate buoyancy effects.
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        temperature.set_value(lid, 0.5);
        const auto x = mesh->cell_centroid(lid).x;
        velocity.set_value(lid, {std::sin(M_PI * x), 0.0, 0.0});
    }
    temperature.sync_ghosts();
    velocity.sync_ghosts();

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.01;
    options.kinematic_viscosity = 0.1;
    options.thermal_expansion = 0.0;  // disable buoyancy
    options.reference_temperature = 0.5;

    // No-slip velocity BCs on all faces.
    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] =
            {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }

    SimpleFluid::BoussinesqMomentumEquation<Pack> equation(mesh);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    equation.advance_velocity(velocity,
                              zero_fluxes,
                              temperature,
                              cache,
                              options,
                              velocity);

    // All components must be finite.
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto v = velocity.value(lid);
        EXPECT_TRUE(std::isfinite(v.x)) << "Non-finite v.x at cell " << lid;
        EXPECT_TRUE(std::isfinite(v.y)) << "Non-finite v.y at cell " << lid;
        EXPECT_TRUE(std::isfinite(v.z)) << "Non-finite v.z at cell " << lid;
    }
}

/** @brief Verifies uniform momentum at a homogeneous-Neumann outlet. */
TEST(PhysicalEquationsTest,
     PhysicalMomentumPreservesUniformVelocityAtHomogeneousNeumannOutlet)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(4, 1, 1, 0.25));
    FieldType temperature(mesh, 0.5, "temperature");
    const VectorFieldType::vec_type uniform_velocity{0.75, -0.5, 0.25};
    VectorFieldType velocity(mesh, uniform_velocity, "velocity");
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0, "zero_fluxes");

    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {
             "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {
            SimpleFluid::BoundaryConditionType::Neumann, {}};
    }
    const auto boundary_cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, boundaries);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.thermal_expansion = 0.0;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.dynamic_viscosity = 2.0;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, options);
    auto zero_source =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {};
    };
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::BoussinesqMomentumEquation<Pack> equation(mesh);
    equation.advance_velocity_physical(
        velocity,
        zero_fluxes,
        temperature,
        boundary_cache,
        options,
        material,
        1.0,
        false,
        velocity,
        zero_source,
        linear_options);

    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto actual = velocity.value(cell_lid);
        EXPECT_NEAR(actual.x, uniform_velocity.x, 1.0e-11);
        EXPECT_NEAR(actual.y, uniform_velocity.y, 1.0e-11);
        EXPECT_NEAR(actual.z, uniform_velocity.z, 1.0e-11);
    }
}

/** @brief Verifies use and validation of a spatial dynamic-viscosity override. */
TEST(PhysicalEquationsTest,
     PhysicalMomentumUsesValidatedDynamicViscosityOverride)
{
    constexpr int n_cells = 4;
    auto mesh = SimpleFluid::test::build_mesh<Pack>(
        SimpleFluid::test::make_box_database(
            n_cells, 1, 1, 1.0 / n_cells));
    FieldType temperature(mesh, 0.5, "temperature");
    VectorFieldType molecular_velocity(mesh, "molecular_velocity");
    VectorFieldType effective_velocity(mesh, "effective_velocity");
    std::vector<VectorFieldType::vec_type> initial_velocity(
        mesh->num_owned_cells());
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        const auto value = VectorFieldType::vec_type{
            std::sin(M_PI * mesh->cell_centroid(cell_lid).x), 0.0, 0.0};
        initial_velocity[owned] = value;
        molecular_velocity.set_owned_value(cell_lid, value);
        effective_velocity.set_owned_value(cell_lid, value);
    }
    molecular_velocity.sync_ghosts();
    effective_velocity.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {
             "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] = {
            SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);

    SimpleFluid::TimeStepperOptions options;
    options.time_step = 0.1;
    options.thermal_expansion = 0.0;
    SimpleFluid::BoussinesqModelOptions model_options;
    model_options.dynamic_viscosity = 0.0;
    SimpleFluid::MaterialPropertyFields<Pack> material(
        mesh, model_options, options);
    FieldType effective_viscosity(mesh, 1.0, "effective_viscosity");
    auto zero_source =
        [](MeshType::local_ordinal_type) -> VectorFieldType::vec_type
    {
        return {};
    };
    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::BoussinesqMomentumEquation<Pack> molecular_equation(mesh);
    molecular_equation.advance_velocity_physical(
        molecular_velocity,
        zero_fluxes,
        temperature,
        cache,
        options,
        material,
        1.0,
        false,
        molecular_velocity,
        zero_source,
        linear_options);
    SimpleFluid::BoussinesqMomentumEquation<Pack> effective_equation(mesh);
    effective_equation.advance_velocity_physical(
        effective_velocity,
        zero_fluxes,
        temperature,
        cache,
        options,
        material,
        1.0,
        false,
        effective_velocity,
        zero_source,
        linear_options,
        &effective_viscosity);

    double maximum_override_change = 0.0;
    for (size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<MeshType::local_ordinal_type>(owned);
        EXPECT_NEAR(
            molecular_velocity.value(cell_lid).x,
            initial_velocity[owned].x,
            1.0e-11);
        maximum_override_change = std::max(
            maximum_override_change,
            std::abs(
                effective_velocity.value(cell_lid).x
              - initial_velocity[owned].x));
    }
    EXPECT_GT(maximum_override_change, 1.0e-6);

    FieldType negative_viscosity(mesh, -1.0, "negative_viscosity");
    EXPECT_THROW(
        effective_equation.advance_velocity_physical(
            effective_velocity,
            zero_fluxes,
            temperature,
            cache,
            options,
            material,
            1.0,
            false,
            effective_velocity,
            zero_source,
            linear_options,
            &negative_viscosity),
        std::invalid_argument);
}

/** @brief Verifies non-orthogonal momentum diffusion on a skewed mesh. */
TEST(PhysicalEquationsTest, MomentumDiffusionHonorsNonOrthogonalTreatmentOnSkewedMesh)
{
    auto mesh = SimpleFluid::test::make_skewed_prism_mesh<Pack>();
    FieldType temperature(mesh, "temperature");
    VectorFieldType orthogonal_velocity(mesh, "orthogonal_velocity");
    VectorFieldType implicit_velocity(mesh, "implicit_velocity");

    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto center = mesh->cell_centroid(lid);
        const SimpleFluid::vec3<> value{
            0.2 + center.x * center.y,
            -0.1 + center.y * center.z,
            0.3 + center.x * center.z};
        temperature.set_value(lid, 0.5);
        orthogonal_velocity.set_value(lid, value);
        implicit_velocity.set_value(lid, value);
    }
    temperature.sync_ghosts();
    orthogonal_velocity.sync_ghosts();
    implicit_velocity.sync_ghosts();

    SimpleFluid::BoundaryConditionSet bcs;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        bcs.velocity[name] =
            {SimpleFluid::BoundaryConditionType::NoSlip, {}};
    }
    const auto cache =
        SimpleFluid::FVM::cache_velocity_boundary_conditions<Pack>(
            mesh, bcs);
    SimpleFluid::FaceField<Pack> zero_fluxes(mesh, 0.0);

    SimpleFluid::TimeStepperOptions orthogonal_options;
    orthogonal_options.time_step = 0.05;
    orthogonal_options.kinematic_viscosity = 0.25;
    orthogonal_options.thermal_expansion = 0.0;
    orthogonal_options.reference_temperature = 0.5;
    orthogonal_options.non_orthogonal_treatment =
        SimpleFluid::FVM::NonOrthogonalTreatment::Explicit;
    orthogonal_options.n_non_orthogonal_correctors = 0;

    auto implicit_options = orthogonal_options;
    implicit_options.non_orthogonal_treatment =
        SimpleFluid::FVM::NonOrthogonalTreatment::Implicit;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.tolerance = 1.0e-12;
    linear_options.max_iterations = 500;

    SimpleFluid::BoussinesqMomentumEquation<Pack> equation(mesh);
    equation.advance_velocity(orthogonal_velocity,
                              zero_fluxes,
                              temperature,
                              cache,
                              orthogonal_options,
                              orthogonal_velocity,
                              linear_options);
    equation.advance_velocity(implicit_velocity,
                              zero_fluxes,
                              temperature,
                              cache,
                              implicit_options,
                              implicit_velocity,
                              linear_options);

    double max_difference = 0.0;
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
         ++lid)
    {
        const auto orthogonal_value = orthogonal_velocity.value(lid);
        const auto implicit_value = implicit_velocity.value(lid);
        EXPECT_TRUE(std::isfinite(orthogonal_value.x));
        EXPECT_TRUE(std::isfinite(orthogonal_value.y));
        EXPECT_TRUE(std::isfinite(orthogonal_value.z));
        EXPECT_TRUE(std::isfinite(implicit_value.x));
        EXPECT_TRUE(std::isfinite(implicit_value.y));
        EXPECT_TRUE(std::isfinite(implicit_value.z));

        max_difference = std::max(
            max_difference,
            std::abs(orthogonal_value.x - implicit_value.x));
        max_difference = std::max(
            max_difference,
            std::abs(orthogonal_value.y - implicit_value.y));
        max_difference = std::max(
            max_difference,
            std::abs(orthogonal_value.z - implicit_value.z));
    }

    EXPECT_GT(max_difference, 1.0e-10);
}
