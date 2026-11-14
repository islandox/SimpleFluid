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

#include "FVM/TransportSystem.hh"
#include "FVM/TransportSystemFactory.hh"
#include "solvers/IncompressibleIsothermalSolver.hh"

#include "solvers/unitTests/CoupledBackendTestSupport.hh"
#include <gtest/gtest.h>
#include <Teuchos_CommHelpers.hpp>

#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/unitTests/test_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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
        for (const auto selection : SimpleFluid::test::backends_for(coupling))
        {
            SCOPED_TRACE(SimpleFluid::test::backend_name(selection));
            SCOPED_TRACE("coupling=" + std::to_string(static_cast<int>(coupling)));
            auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
            Solver solver(mesh, slip_box_boundaries(),
                SimpleFluid::test::with_coupled_backend(stable_time_options(coupling), selection),
                focused_linear_options(), reference_density);
            initialize_shear(solver);

            auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
            ASSERT_NO_THROW(solver.step());

            EXPECT_EQ(solver.step_index(), 1);
            EXPECT_TRUE(solver.last_step_statistics().converged);
            EXPECT_GE(solver.last_step_statistics().linear_solves, 3);
            EXPECT_DOUBLE_EQ(solver.last_step_statistics().temperature, 0.0);
            expect_positive_turbulence_fields(
                turbulence, std::as_const(solver).material_properties(), reference_density);
        }
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
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::Step).calls, 1u);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::PressureVelocity).calls, 1u);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::Turbulence).calls, 1u);
    EXPECT_EQ(solver.solver_phase_timing(SimpleFluid::SolverPhase::Temperature).calls, 0u);
    EXPECT_GE(solver.solver_phase_timing(SimpleFluid::SolverPhase::Step).seconds,
        solver.solver_phase_timing(SimpleFluid::SolverPhase::Turbulence).seconds);
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

/** Modified pressure absorbs isotropic turbulence in both coupling paths. */
TEST(IncompressibleIsothermalSolverTest, ModifiedPressureAbsorbsNonuniformK)
{
    for (const auto coupling : {SimpleFluid::PressureVelocityCoupling::PISO,
                               SimpleFluid::PressureVelocityCoupling::CoupledKrylov})
    {
        double mechanical_pressure_range = 0.0;
        for (const bool modified : {false, true})
        {
            auto mesh = SimpleFluid::test::build_mesh<Pack>(SimpleFluid::test::make_2x2x2_database());
            auto controls = stable_time_options(coupling);
            controls.isothermal_modified_pressure = modified;
            Solver solver(mesh, slip_box_boundaries(), controls, focused_linear_options(), 2.0);
            auto& turbulence = solver.configure_turbulence(standard_k_epsilon_options());
            Solver::turbulence_model_type::field_type k(solver.velocity().mesh_ptr(), 0.1, "restart_nonuniform_k");
            for (size_t i = 0; i < k.num_owned_cells(); ++i)
            {
                const auto lid = static_cast<Pack::local_ordinal_type>(i);
                k.set_owned_value(lid, 0.1 + 0.02 * k.mesh().cell_centroid(lid).x);
            }
            k.sync_ghosts();
            turbulence.restore_transported_state(k, *turbulence.dissipation_rate(),
                turbulence.turbulent_kinematic_viscosity(), solver.velocity(),
                std::as_const(solver).material_properties(), 2.0);
            ASSERT_NO_THROW(solver.step());
            ASSERT_NO_THROW(solver.step());
            double minimum = std::numeric_limits<double>::infinity();
            double maximum = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < k.num_owned_cells(); ++i)
            {
                const double p = solver.pressure().value(static_cast<Pack::local_ordinal_type>(i));
                minimum = std::min(minimum, p);
                maximum = std::max(maximum, p);
            }
            // A partition may be constant locally while the mechanical
            // pressure varies between ranks. Check the complete field range.
            const auto comm = k.mesh().owned_cell_map()->getComm();
            double global_minimum{}, global_maximum{};
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &minimum, &global_minimum);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &maximum, &global_maximum);
            minimum = global_minimum;
            maximum = global_maximum;
            if (modified)
            {
                EXPECT_LT(maximum - minimum, 1.0e-10);
                EXPECT_GT(mechanical_pressure_range, 1.0e-8);
            }
            else mechanical_pressure_range = maximum - minimum;
        }
    }
}

/** ZeroGradient retains tangential transpose traction and removes normal gradients. */
TEST(IncompressibleIsothermalSolverTest, ZeroGradientOutletStressMatchesManufacturedShear)
{
    auto mesh = make_native_cartesian_mesh();
    Solver::velocity_field_type velocity(mesh, Solver::vec_type{}, "outlet_test_U");
    Solver::field_type viscosity(mesh, 1.0, "outlet_test_mu");
    Solver::face_flux_field_type flux(mesh, 0.0, "outlet_test_phi");
    for (const bool normal_profile : {false, true})
    {
        for (size_t i=0; i<velocity.num_owned_cells(); ++i)
        {
            const auto lid=static_cast<Pack::local_ordinal_type>(i);
            const auto centre=mesh->cell_centroid(lid);
            velocity.set_owned_value(lid, {normal_profile ? centre.x : centre.y, 0.0, 0.0});
        }
        velocity.sync_ghosts();
        auto is_x_face = [&](int batch, size_t face)
        {
            const auto lid=mesh->boundary_face_batch(batch).face_lids[face];
            const auto centre=mesh->face_centroid(lid);
            return centre.x == 0.0 || centre.x == 2.0;
        };
        auto boundary = [&](int batch, size_t face)
        {
            const auto lid=mesh->boundary_face_batch(batch).face_lids[face];
            const auto centre=mesh->face_centroid(lid);
            return Solver::vec_type{normal_profile ? centre.x : centre.y, 0.0, 0.0};
        };
        auto diffusion = [&](int batch, size_t face) { return !is_x_face(batch,face); };
        auto zero = [](Pack::local_ordinal_type) { return Solver::vec_type{}; };
        auto assemble = [&](bool enabled)
        {
            SimpleFluid::FVM::BoundaryFaceSelector selector;
            if (enabled) selector=is_x_face;
            using Factory = SimpleFluid::FVM::TransportSystemFactory<Pack, SimpleFluid::MeshHandle<Pack>>;
            return Factory::assemble(Factory::MomentumTerms{{velocity, 0.1}, flux, {viscosity, 1.0}, zero},
                Factory::MomentumBoundary{boundary, diffusion, nullptr, selector}, Factory::MomentumOptions{});
        };
        const auto traction_free=assemble(false);
        const auto zero_gradient=assemble(true);
        const auto before=traction_free.rhs->getLocalViewHost(Tpetra::Access::ReadOnly);
        const auto after=zero_gradient.rhs->getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t i=0; i<velocity.num_owned_cells(); ++i)
        {
            const auto centre=mesh->cell_centroid(static_cast<Pack::local_ordinal_type>(i));
            EXPECT_NEAR(after(i,0)-before(i,0),0.0,1e-12);
            EXPECT_NEAR(after(i,1)-before(i,1),normal_profile ? 0.0 : (centre.x<1.0 ? -1.0 : 1.0),1e-12);
            EXPECT_NEAR(after(i,2)-before(i,2),0.0,1e-12);
        }
    }
}

TEST(IncompressibleIsothermalSolverTest, IndependentDiagnosticsDetectPressureAndPreserveAcceptedState)
{
    auto mesh=make_native_cartesian_mesh();
    auto time_options=stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO);
    Solver solver(mesh,slip_box_boundaries(),time_options,focused_linear_options(),2.0);
    SimpleFluid::SolverDiagnosticsOptions options;
    options.include_cell_data=true;options.include_face_data=true;
    const auto equilibrium=solver.evaluate_diagnostics(options);
    EXPECT_NEAR(equilibrium.momentum_vector_rms,0.0,1e-14);
    EXPECT_FALSE(equilibrium.stored_flux_available);
    for(size_t i=0;i<mesh->num_owned_cells();++i) {
        const auto lid=static_cast<Pack::local_ordinal_type>(i);
        solver.pressure().set_owned_value(lid,mesh->cell_centroid(lid).x);
    }
    solver.pressure().sync_ghosts();
    std::vector<double> before;
    for(size_t i=0;i<mesh->num_owned_cells();++i) before.push_back(solver.pressure().value(static_cast<Pack::local_ordinal_type>(i)));
    const auto report=solver.evaluate_diagnostics(options);
    const auto repeated=solver.evaluate_diagnostics(options);
    EXPECT_GT(report.momentum_vector_rms,1e-3);
    EXPECT_DOUBLE_EQ(report.momentum_vector_rms,repeated.momentum_vector_rms);
    EXPECT_EQ(solver.step_index(),0);EXPECT_DOUBLE_EQ(solver.time(),0.0);
    for(size_t i=0;i<before.size();++i) EXPECT_DOUBLE_EQ(before[i],solver.pressure().value(static_cast<Pack::local_ordinal_type>(i)));
    ASSERT_EQ(report.reconstructed_face_flux.size(),mesh->num_owned_faces());
    for(size_t i=0;i<report.reconstructed_face_flux.size();++i)
        EXPECT_NEAR(report.reconstructed_face_flux[i],report.velocity_face_flux[i]+report.pressure_face_flux[i],1e-14);
    options.pressure_coefficient_time_step=0.0;
    const auto zero_coefficient=solver.evaluate_diagnostics(options);
    for(auto value:zero_coefficient.pressure_face_flux) EXPECT_DOUBLE_EQ(value,0.0);
    EXPECT_EQ(solver.step_index(),0);
}

TEST(IncompressibleIsothermalSolverTest, DiagnosticsHistoryPublishedForAcceptedSteps)
{
    auto mesh=make_native_cartesian_mesh();
    Solver solver(mesh,slip_box_boundaries(),stable_time_options(SimpleFluid::PressureVelocityCoupling::PISO),focused_linear_options());
    solver.enable_diagnostic_history();
    EXPECT_FALSE(solver.evaluate_diagnostics().last_pressure_update_l2.has_value());
    solver.step();
    const auto report=solver.evaluate_diagnostics();
    EXPECT_TRUE(report.stored_flux_available);
    ASSERT_EQ(report.pressure_update_l2_history.size(),1u);
    ASSERT_TRUE(report.accumulated_pressure_update_l2.has_value());
    EXPECT_DOUBLE_EQ(*report.accumulated_pressure_update_l2,0.0);
    EXPECT_EQ(solver.evaluate_diagnostics().pressure_update_l2_history.size(),1u);
}

TEST(IncompressibleIsothermalSolverTest, DiagnosticsDoNotChangeNextTurbulentStep)
{
    for(const auto coupling:{SimpleFluid::PressureVelocityCoupling::PISO,SimpleFluid::PressureVelocityCoupling::CoupledKrylov}) {
        auto mesh=make_native_cartesian_mesh();
        Solver observed(mesh,slip_box_boundaries(),stable_time_options(coupling),focused_linear_options());
        Solver control(mesh,slip_box_boundaries(),stable_time_options(coupling),focused_linear_options());
        initialize_shear(observed);initialize_shear(control);
        observed.configure_turbulence(standard_k_epsilon_options());control.configure_turbulence(standard_k_epsilon_options());
        ASSERT_NO_THROW(observed.evaluate_diagnostics());
        ASSERT_NO_THROW(observed.evaluate_diagnostics());
        observed.step();control.step();
        for(size_t i=0;i<mesh->num_owned_cells();++i) {
            const auto c=static_cast<Pack::local_ordinal_type>(i);
            EXPECT_DOUBLE_EQ(observed.pressure().value(c),control.pressure().value(c));
            const auto a=observed.velocity().value(c),b=control.velocity().value(c);
            EXPECT_DOUBLE_EQ(a.x,b.x);EXPECT_DOUBLE_EQ(a.y,b.y);EXPECT_DOUBLE_EQ(a.z,b.z);
            EXPECT_DOUBLE_EQ(observed.find_turbulence_model()->turbulent_kinetic_energy().value(c),control.find_turbulence_model()->turbulent_kinetic_energy().value(c));
        }
    }
}
