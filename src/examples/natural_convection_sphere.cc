/**
 * @file natural_convection_sphere.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Runnable bottom-heated sphere Boussinesq smoke example.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */

#if defined(SIMPLEFLUID_USE_CXX_MODULES)
#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif
import SimpleFluid.Examples;
#else
#include "examples/ExampleRunner.hh"
#include "trilinos_wrapper/Tpetra.hh"
#endif

/**
 * @brief Entry point for the natural-convection sphere example.
 *
 * Configures a 3D spherified-cube mesh, sets Dirichlet temperature on
 * lower_surface (hot) and upper_surface (cold), and runs a short Boussinesq
 * simulation.
 *
 * @param argc Argument count (passed to Tpetra for MPI init).
 * @param argv Argument vector (passed to Tpetra for MPI init).
 * @return Exit code (always 0).
 */
int main(int argc, char** argv)
{
    Tpetra::ScopeGuard tpetra_scope(&argc, &argv);

    auto db = SimpleFluid::make_example_database();
    db->set("dimension", 3);
    db->set("mesh_size", SimpleFluid::real_t{1.0});
    db->set("domain_type",
            static_cast<int>(SimpleFluid::MeshFactory::DomainType::SPHERE));
    db->set("radius", SimpleFluid::real_t{1.0});
    db->set("domain_exterior_face_types",
            SimpleFluid::ArrString{"lower_surface", "upper_surface"});

    SimpleFluid::BoundaryConditionSet bcs;
    bcs.temperature["lower_surface"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 1.0};
    bcs.temperature["upper_surface"] = {SimpleFluid::BoundaryConditionType::Dirichlet, 0.0};

    SimpleFluid::TimeStepperOptions time_options;
    time_options.time_step = 1.0e-2;
    time_options.steps = 5;
    time_options.thermal_diffusivity = 1.0e-2;
    time_options.kinematic_viscosity = 1.0e-2;
    time_options.reference_temperature = 0.5;

    SimpleFluid::LinearSolverOptions linear_options;
    linear_options.max_iterations = 50;
    linear_options.tolerance = 1.0e-12;

    SimpleFluid::run_boussinesq_example<>(
        db, bcs, time_options, linear_options,
        [](auto& solver) { solver.initialize_bottom_hot_top_cold(1.0, 0.0); },
        "natural_convection_sphere.vtu");

    return 0;
}
