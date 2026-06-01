/**
 * @file testAnalyticalConvergence.cc
 * @brief Slow analytical convergence tests for physical equations.
 */

#include <gtest/gtest.h>

#include "equations/BoundaryConditions.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "fields/CellField.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/ErrorNorms.hh"
#include "utils/testing_environment.hh"

#include <array>
#include <cmath>
#include <vector>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using MeshType = SimpleFluid::Mesh<Pack>;
using FieldType = SimpleFluid::CellField<Pack>;

using utils_test::KokkosEnvironment;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

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
              - old_temperature[static_cast<std::size_t>(lid)]);
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

SimpleFluid::BoundaryConditionSet one_dimensional_diffusion_bcs()
{
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

    return bcs;
}

} // namespace

/**
 * @brief Slow steady-state diffusion study for a known linear solution.
 */
TEST(AnalyticalConvergenceTest, DiffusionErrorDecreasesWithMeshRefinement)
{
    constexpr double domain_length = 1.0;
    constexpr double diffusivity = 1.0;
    constexpr double time_step = 1.0e-3;
    constexpr double steady_update_tolerance = 1.0e-13;
    constexpr int max_steps = 50000;

    const std::array<int, 3> resolutions = {4, 8, 16};
    std::array<double, 3> l2_errors{};
    std::array<int, 3> steps_taken{};

    auto exact = [domain_length](SimpleFluid::vec3<> pos)
    {
        return 1.0 - pos.x / domain_length;
    };

    for (std::size_t i = 0; i < resolutions.size(); ++i)
    {
        const auto n = resolutions[i];
        const double h = domain_length / static_cast<double>(n);
        auto db = SimpleFluid::test::make_box_database(n, 1, 1, h);
        auto mesh = SimpleFluid::test::build_mesh<Pack>(db);
        FieldType temperature(mesh, "temperature");

        for (std::size_t owned = 0; owned < mesh->num_owned_cells(); ++owned)
        {
            temperature.set_owned_value(
                static_cast<MeshType::local_ordinal_type>(owned), 0.0);
        }
        temperature.sync_ghosts();

        const auto bcs = one_dimensional_diffusion_bcs();
        SimpleFluid::TemperatureDiffusionEquation<Pack> equation(mesh, bcs);

        steps_taken[i] = advance_explicit_diffusion_until_converged(
            equation, temperature, time_step, diffusivity,
            steady_update_tolerance, max_steps);

        l2_errors[i] = SimpleFluid::l2_error(temperature, exact);
    }

    for (std::size_t i = 0; i < resolutions.size(); ++i)
    {
        EXPECT_LT(steps_taken[i], max_steps)
            << "Resolution " << resolutions[i]
            << " cells did not converge before the iteration limit";
        EXPECT_LT(l2_errors[i], 1.0e-10)
            << "Resolution " << resolutions[i] << " cells: L2 error too large";
    }
}
