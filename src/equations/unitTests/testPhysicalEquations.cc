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
#include "FVM/FvmOperators.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/ErrorNorms.hh"
#include "utils/testing_environment.hh"

#include <cmath>
#include <memory>
#include <vector>

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
        values[static_cast<std::size_t>(lid)] = field.local_value(lid);
    }

    return values;
}

std::vector<MeshType::Vec3> local_vector_values(const VectorFieldType& field)
{
    std::vector<MeshType::Vec3> values(field.num_local_cells());
    for (MeshType::local_ordinal_type lid = 0;
         lid < static_cast<MeshType::local_ordinal_type>(field.num_local_cells());
         ++lid)
    {
        values[static_cast<std::size_t>(lid)] = field.local_value(lid);
    }

    return values;
}

} // namespace

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
    equation.advance_velocity(local_vector_values(velocity),
                              zero_fluxes,
                              temperature,
                              bcs,
                              options,
                              velocity);

    EXPECT_NEAR(velocity.value(0).x, 1.0, 1.0e-10);
    EXPECT_NEAR(velocity.value(0).y, 2.0, 1.0e-10);
    EXPECT_NEAR(velocity.value(0).z, 3.0, 1.0e-10);
}

/**
 * @brief Runs semi-implicit temperature advection in X, Y, and Z directions and verifies changes.
 */
TEST(PhysicalEquationsTest, TemperatureSemiImplicitAdvectionRunsInEachDirection)
{
    auto mesh = make_2x2x2_mesh();

    for (std::size_t component = 0; component < 3; ++component)
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
        const auto fluxes = SimpleFluid::FvmOperators::face_fluxes(
            *mesh, velocity);

        SimpleFluid::BoundaryConditionSet bcs;
        SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);
        equation.advance_semi_implicit(old_temperature, fluxes, 0.1, 0.0,
                                       temperature);

        bool changed = false;
        for (MeshType::local_ordinal_type lid = 0;
             lid < static_cast<MeshType::local_ordinal_type>(mesh->num_owned_cells());
             ++lid)
        {
            EXPECT_TRUE(std::isfinite(temperature.value(lid)));
            changed = changed
                   || std::abs(temperature.value(lid)
                             - old_temperature[static_cast<std::size_t>(lid)])
                      > 1.0e-12;
        }
        EXPECT_TRUE(changed);
    }
}

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

TEST(PhysicalEquationsTest, PressureProjectionReducesFluxDivergence)
{
    auto mesh = make_2x2x2_mesh();
    FieldType pressure(mesh, "pressure");
    VectorFieldType velocity(mesh, SimpleFluid::vec3{1.0, 0.0, 0.0}, "velocity");

    SimpleFluid::BoundaryConditionSet bcs;
    auto divergence_norm = [&]()
    {
        const auto fluxes = SimpleFluid::FvmOperators::face_fluxes(
            *mesh, velocity, bcs);
        const auto divergence =
            SimpleFluid::FvmOperators::cell_divergence_from_fluxes<Pack>(
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
    equation.project(pressure, 0.1, bcs, velocity);

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

    for (std::size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
    {
        temperature.set_owned_value(
            static_cast<MeshType::local_ordinal_type>(owned), 0.0);
    }
    temperature.sync_ghosts();

    SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);

    constexpr double diffusivity = 1.0;
    constexpr double time_step = 1.0e-3;
    constexpr int steps = 100000;

    for (int step = 0; step < steps; ++step)
    {
        const auto old_temperature = local_values(temperature);
        equation.advance_explicit(old_temperature, time_step, diffusivity,
                                  temperature);
    }

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

    for (std::size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
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


