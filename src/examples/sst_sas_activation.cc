/** @file sst_sas_activation.cc @brief Small transient SAS source-activation fixture, not turbulence validation. */
#include "solvers/IncompressibleIsothermalSolver.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include <Tpetra_Core.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard scope(&argc, &argv);
    using namespace SimpleFluid;
    using Pack = DefaultTpetraTypes;
    try
    {
        constexpr size_t n = 6;
        ArrReal edges(n + 1);
        for (size_t i = 0; i <= n; ++i) edges[i] = real_t(i)/n;
        SP<const MeshHandle<Pack>> mesh = std::make_shared<MeshHandle<Pack>>(
            std::make_shared<Meshes::OrthogonalCartesian3D>(Vec3D<ArrReal>{{edges, edges, edges}}));
        BoundaryConditionSet boundaries;
        for (auto name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
            boundaries.velocity[name] = {BoundaryConditionType::NoSlip, {}};
        TimeStepperOptions time;
        time.time_step = 1e-3; time.kinematic_viscosity = 1e-3;
        time.non_orthogonal_treatment = FVM::NonOrthogonalTreatment::Explicit;
        time.n_pressure_correctors = 2;
        LinearSolverOptions linear; linear.tolerance = 1e-11;
        IncompressibleIsothermalSolver<Pack> solver(mesh, boundaries, time, linear);
        Database config;
        config.set("turbulence_model", std::string("SSTKOmegaSAS"));
        config.set("initial_turbulent_kinetic_energy", .2);
        config.set("initial_specific_dissipation_rate", 2.);
        config.set("wall_distance", .25); // Controlled uniform source fixture; not a wall-resolved case.
        config.set("turbulence_sas_diagnostics", true);
        auto& model = solver.configure_turbulence(config);
        const auto pi = std::numbers::pi;
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        {
            const auto p = mesh->cell_centroid(i);
            // Curl of a smooth stream function; zero velocity on every physical wall.
            const double envelope = std::sin(pi*p.z)*std::sin(pi*p.z);
            solver.velocity().set_owned_value(i, {
                std::sin(pi*p.x)*std::sin(pi*p.x)*std::sin(2*pi*p.y)*envelope,
                -std::sin(2*pi*p.x)*std::sin(pi*p.y)*std::sin(pi*p.y)*envelope, 0});
        }
        solver.velocity().sync_ghosts();
        for (int step = 0; step < 2; ++step) solver.step();
        const auto& stats = model.sas_statistics();
        if (!stats.valid || stats.max_source <= 1e-6)
            throw std::runtime_error("SAS activation fixture did not produce a nonzero source.");
        double local_omega = 0, omega = 0;
        for (size_t i = 0; i < mesh->num_owned_cells(); ++i)
        {
            const auto k = model.turbulent_kinetic_energy().value(i);
            const auto w = model.specific_dissipation_rate()->value(i);
            if (!std::isfinite(k) || !std::isfinite(w) || k <= 0 || w <= 0)
                throw std::runtime_error("Invalid turbulence state in activation fixture.");
            local_omega += mesh->cell_volume(i)*w;
        }
        const auto comm = mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_omega, &omega);
        if (argc > 1)
        {
            SolutionOutputOptions output; output.include_turbulence_fields = true;
            solver.write_parallel_solution_vtu(argv[1], output);
        }
        if (comm->getRank() == 0)
            std::cout << std::setprecision(17) << "sas_result " << stats.active_cell_fraction << ' '
                      << stats.cap_active_cell_fraction << ' ' << stats.min_source << ' '
                      << stats.max_source << ' ' << stats.volume_mean_source << ' '
                      << stats.volume_integrated_source << ' ' << omega << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
