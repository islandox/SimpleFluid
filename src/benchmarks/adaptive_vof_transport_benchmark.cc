/** @file adaptive_vof_transport_benchmark.cc
 * @brief Opt-in full low-Mach transport timing with conservation diagnostics.
 */
#include "solvers/AdaptiveVoFSolver.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <sys/resource.h>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    using namespace SimpleFluid;
    using LO = DefaultTpetraTypes::local_ordinal_type;
    const auto comm = Tpetra::getDefaultComm();
    const std::size_t b = argc > 1 ? std::stoul(argv[1]) : 4;
    const std::size_t populations = argc > 2 ? std::stoul(argv[2]) : 2;
    const std::size_t repetitions = argc > 3 ? std::stoul(argv[3]) : 3;
    if (b < 2 || b % 2 || !repetitions)
        return 2;
    Meshes::AMRGrid grid;
    grid.root_blocks = {4, 1, 1};
    grid.block_cells = {b, b, b};
    grid.extent = {4, 1, 1};
    grid.periodic = {true, true, true};
    const auto hierarchy = std::make_shared<const Meshes::CartesianAMR>(grid);
    VoFMaterial material;
    material.liquid_viscosity = material.gas_viscosity = material.surface_tension = 0;
    AdaptiveVoFSolver flow(hierarchy, material);
    VoF::LowMachOptions options;
    VoF::LinearCompressibleLiquid liquid;
    liquid.bounds = {250, 400, 10000, 1000000};
    liquid.reference_temperature = 300;
    liquid.reference_pressure = 100000;
    liquid.reference_density = 1000;
    liquid.isobaric_heat_capacity = 4000;
    liquid.thermal_expansion = 2e-4;
    liquid.compressibility = 5e-9;
    liquid.reference_internal_energy = 100000;
    options.liquid = VoF::make_linear_compressible_liquid(liquid);
    options.hydrogen = {0.002, 20, 300, 5000, 0, 0};
    options.inert = {0.028, 25, 300, 7500, 0, 0};
    options.volume_relative_tolerance = 2e-11;
    const auto n = flow.mesh()->num_owned_cells();
    VoF::LowMachInitialState initial;
    initial.thermodynamic_pressure = 100000;
    initial.temperature.assign(n, 300);
    initial.resolved_fraction.resize(n);
    initial.resolved_hydrogen_mole_fraction.assign(n, 0.5);
    initial.velocity.assign(n, {0.1, 0, 0});
    for (std::size_t c = 0; c < n; ++c)
    {
        const auto x = flow.mesh()->cell_centroid(static_cast<LO>(c));
        initial.resolved_fraction[c] = 0.2 + 0.1 * std::sin(std::numbers::pi * x.x / 2);
    }
    const double bubble_moles = 100000 * 4 * std::numbers::pi / 3 * 1e-12 / (options.gas_constant * 300);
    for (std::size_t p = 0; p < populations; ++p)
    {
        VoF::DispersedPopulation population;
        population.name = "population_" + std::to_string(p);
        for (std::size_t c = 0; c < n; ++c)
        {
            const double count = 1000 * (1 - initial.resolved_fraction[c]) * flow.mesh()->cell_volume(static_cast<LO>(c));
            population.number.push_back(count);
            population.hydrogen_moles.push_back(count * bubble_moles);
        }
        initial.dispersed.push_back(std::move(population));
    }
    flow.initialize_low_mach(options, initial);
    comm->barrier();
    const auto start = std::chrono::steady_clock::now();
    int correctors = 0, pressure_iterations = 0;
    for (std::size_t i = 0; i < repetitions; ++i)
    {
        const auto trial = flow.preview_low_mach_step(0.001, {});
        correctors += trial.diagnostics().correctors;
        flow.commit_low_mach_trial(trial);
        pressure_iterations += flow.diagnostics().pressure_iterations;
    }
    const double local = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double elapsed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &elapsed);
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    double rss = usage.ru_maxrss, peak = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &rss, &peak);
    const auto& d = flow.low_mach_diagnostics();
    if (comm->getRank() == 0)
        std::cout << "block_axis,cells,populations,ranks,steps,dt,total_s,correctors,pressure_iterations,rss_kib,mass_error,occupancy_error,continuity_error\n"
                  << std::setprecision(17) << b << ',' << 4*b*b*b << ',' << populations << ',' << comm->getSize()
                  << ',' << repetitions << ",0.001," << elapsed << ',' << correctors << ',' << pressure_iterations
                  << ',' << peak << ',' << d.mass_error << ',' << d.maximum_occupancy_error << ','
                  << d.maximum_continuity_residual << '\n';
}
