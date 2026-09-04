/**
 * @file testIncompressibleIsothermalSolver.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Tests for the transient incompressible isothermal solver.
 * @version 0.1
 * @date 2026-08-20
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "solvers/IncompressibleIsothermalSolver.hh"

#include <gtest/gtest.h>

#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using Solver = SimpleFluid::IncompressibleIsothermalSolver<Pack>;

testing::Environment* const kokkos_environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

static_assert(std::is_base_of_v<SimpleFluid::FluidSolver<Pack>, Solver>);

/** @brief Expose the Problem registry for field-ownership checks. */
class InspectableSolver : public Solver
{
public:
    using Solver::Solver;

    bool contains_object(const std::string& name) const { return this->d_problem.contains(name); }
};

/** @brief Build slip velocity conditions on every Cartesian box wall. */
SimpleFluid::BoundaryConditionSet slip_box_boundaries()
{
    SimpleFluid::BoundaryConditionSet boundaries;
    for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
    {
        boundaries.velocity[name] = {SimpleFluid::BoundaryConditionType::Slip, {}};
    }
    return boundaries;
}

/** @brief Return stable one-step controls for a pressure-velocity algorithm. */
SimpleFluid::TimeStepperOptions stable_time_options(SimpleFluid::PressureVelocityCoupling coupling)
{
    SimpleFluid::TimeStepperOptions options;
    options.time_step = 1.0e-3;
    options.steps = 1;
    options.kinematic_viscosity = 1.0e-2;
    options.non_orthogonal_treatment = SimpleFluid::FVM::NonOrthogonalTreatment::Explicit;
    options.pressure_velocity_coupling = coupling;
    options.n_pressure_correctors = 1;
    return options;
}

/** @brief Return a small standard-k-epsilon configuration. */
SimpleFluid::TurbulenceModelOptions standard_k_epsilon_options()
{
    SimpleFluid::TurbulenceModelOptions options;
    options.model = SimpleFluid::TurbulenceModelType::StandardKEpsilon;
    options.initial_turbulent_kinetic_energy = 0.1;
    options.initial_dissipation_rate = 0.009;
    options.min_turbulent_kinetic_energy = 1.0e-10;
    options.min_dissipation_rate = 1.0e-10;
    return options;
}

/** @brief Return tighter controls for the small coupled test systems. */
SimpleFluid::LinearSolverOptions focused_linear_options()
{
    SimpleFluid::LinearSolverOptions options;
    options.max_iterations = 300;
    options.tolerance = 1.0e-11;
    return options;
}

/** @brief Initialize a finite checkerboard shear field. */
void initialize_shear(Solver& solver)
{
    const auto& mesh = solver.velocity().mesh();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        const auto center = mesh.cell_centroid(cell_lid);
        const auto velocity_y = center.x < 1.0 ? (center.y < 1.0 ? 0.2 : -0.2) : (center.y < 1.0 ? -0.2 : 0.2);
        solver.velocity().set_owned_value(cell_lid, {0.0, velocity_y, 0.0});
    }
    solver.velocity().sync_ghosts();
}

/** @brief Assert positive finite turbulence and effective-viscosity fields. */
void expect_positive_turbulence_fields(
    const Solver::turbulence_model_type& turbulence, const Solver::material_type& material, double reference_density)
{
    const auto* epsilon = turbulence.dissipation_rate();
    ASSERT_NE(epsilon, nullptr);
    ASSERT_EQ(turbulence.specific_dissipation_rate(), nullptr);

    const auto& k = turbulence.turbulent_kinetic_energy();
    const auto& nu_t = turbulence.turbulent_kinematic_viscosity();
    const auto& mu_eff = turbulence.effective_dynamic_viscosity();
    for (size_t owned = 0; owned < k.num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<Pack::local_ordinal_type>(owned);
        EXPECT_TRUE(std::isfinite(k.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(epsilon->value(cell_lid)));
        EXPECT_TRUE(std::isfinite(nu_t.value(cell_lid)));
        EXPECT_TRUE(std::isfinite(mu_eff.value(cell_lid)));
        EXPECT_GT(k.value(cell_lid), 0.0);
        EXPECT_GT(epsilon->value(cell_lid), 0.0);
        EXPECT_GT(nu_t.value(cell_lid), 0.0);
        EXPECT_NEAR(mu_eff.value(cell_lid),
            material.dynamic_viscosity.value(cell_lid) + reference_density * nu_t.value(cell_lid), 1.0e-12);
    }
}

/** @brief Create a native two-by-two-by-two Cartesian mesh handle. */
SimpleFluid::SP<const SimpleFluid::MeshHandle<Pack>> make_native_cartesian_mesh()
{
    auto cartesian = std::make_shared<SimpleFluid::Meshes::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}});
    return std::make_shared<SimpleFluid::MeshHandle<Pack>>(std::move(cartesian));
}

/** @brief Delete a temporary output file when a test scope exits. */
class TemporaryFile
{
public:
    explicit TemporaryFile(std::string stem)
        : d_path(std::filesystem::temp_directory_path() /
                 (std::move(stem) + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                     ".vtu"))
    {
    }

    ~TemporaryFile()
    {
        std::error_code error;
        std::filesystem::remove(d_path, error);
    }

    const std::filesystem::path& path() const noexcept { return d_path; }

private:
    std::filesystem::path d_path;
};

} // namespace

/** @brief The isothermal driver owns no temperature field or temperature solve. */
TEST(IncompressibleIsothermalSolverTest, DoesNotRegisterOrSolveTemperatureAndRestoresLaminarMode)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    InspectableSolver solver(mesh, slip_box_boundaries(),
        stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO), focused_linear_options(), 2.0);

    EXPECT_FALSE(solver.contains_object("temperature"));
    EXPECT_EQ(solver.find_turbulence_model(), nullptr);

    auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
    EXPECT_EQ(solver.find_turbulence_model(), &turbulence);
    EXPECT_EQ(std::as_const(solver).find_turbulence_model(), &turbulence);
    EXPECT_TRUE(solver.remove_turbulence_model());
    EXPECT_EQ(solver.find_turbulence_model(), nullptr);
    EXPECT_FALSE(solver.remove_turbulence_model());

    ASSERT_NO_THROW(solver.step());
    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_DOUBLE_EQ(solver.last_step_statistics().temperature, 0.0);
}

/** @brief RANS advances through both segregated and monolithic legacy paths. */
TEST(IncompressibleIsothermalSolverTest, AdvancesStandardKEpsilonThroughPisoAndCoupledKrylov)
{
    constexpr double reference_density = 2.0;
    for (const auto coupling :
        {SimpleFluid::PressureVelocityCoupling::PISO, SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        SCOPED_TRACE("coupling=" + std::to_string(static_cast<int>(coupling)));
        auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
        Solver solver(
            mesh, slip_box_boundaries(), stable_time_options(coupling), focused_linear_options(), reference_density);
        initialize_shear(solver);

        auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
        ASSERT_NO_THROW(solver.step());

        EXPECT_EQ(solver.step_index(), 1);
        EXPECT_TRUE(solver.last_step_statistics().converged);
        EXPECT_GE(solver.last_step_statistics().linear_solves, 3);
        EXPECT_DOUBLE_EQ(solver.last_step_statistics().temperature, 0.0);
        expect_positive_turbulence_fields(turbulence, std::as_const(solver).material_properties(), reference_density);
    }
}

/** @brief A genuinely native MeshHandle executes isothermal RANS directly. */
TEST(IncompressibleIsothermalSolverTest, AdvancesStandardKEpsilonOnNativeMeshHandle)
{
    constexpr double reference_density = 2.0;
    auto mesh = make_native_cartesian_mesh();
    ASSERT_FALSE(mesh->legacy_mesh());

    Solver solver(mesh, slip_box_boundaries(), stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO),
        focused_linear_options(), reference_density);
    initialize_shear(solver);
    auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());

    ASSERT_NO_THROW(solver.step());
    EXPECT_EQ(solver.step_index(), 1);
    EXPECT_TRUE(solver.last_step_statistics().converged);
    EXPECT_DOUBLE_EQ(solver.last_step_statistics().temperature, 0.0);
    expect_positive_turbulence_fields(turbulence, std::as_const(solver).material_properties(), reference_density);
}

/** @brief Optional turbulence output excludes every thermal field. */
TEST(IncompressibleIsothermalSolverTest, WritesTurbulenceFieldsWithoutTemperature)
{
    auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_single_hex_database());
    Solver solver(mesh, slip_box_boundaries(), stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO),
        focused_linear_options(), 1.0);
    solver.configure_turbulence(standard_k_epsilon_options());

    TemporaryFile output("SimpleFluid_isothermal_solution");
    SimpleFluid::SolutionOutputOptions output_options;
    output_options.include_turbulence_fields = true;
    solver.write_solution_vtu(output.path().string(), output_options);

    std::ifstream input(output.path(), std::ios::binary);
    ASSERT_TRUE(input.good());
    const std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    EXPECT_NE(contents.find("Name=\"pressure\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"velocity\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"k\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"epsilon\""), std::string::npos);
    EXPECT_NE(contents.find("Name=\"nu_t\""), std::string::npos);
    EXPECT_EQ(contents.find("Name=\"temperature\""), std::string::npos);
}
