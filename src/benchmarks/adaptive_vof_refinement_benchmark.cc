/** @file adaptive_vof_refinement_benchmark.cc
 * @brief Opt-in sparse-interface refinement policy scaling, using the original public API.
 */
#include "FVM/AdaptiveVoF.hh"
#include "benchmarks/BenchmarkSupport.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    using namespace SimpleFluid;
    using namespace VoF;
    using Clock = std::chrono::steady_clock;
    using LO = DefaultTpetraTypes::local_ordinal_type;
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 1)
        return 2;
    const std::size_t roots_x = argc > 1 ? std::stoul(argv[1]) : 64;
    const std::size_t repetitions = argc > 2 ? std::stoul(argv[2]) : 5;
    if (roots_x < 4 || repetitions == 0)
        return 2;
    SimpleFluid::Meshes::AMR::AMRGrid grid;
    grid.root_blocks = {roots_x, 2, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {0.5 * roots_x, 1.0, 0.5};
    grid.periodic = {true, true, true};
    grid.max_level = 1;
    const auto hierarchy = std::make_shared<SimpleFluid::Meshes::AMR::CartesianAMR>(grid);
    const auto start = Clock::now();
    AdaptiveVoF geometry(hierarchy);
    const auto constructed = Clock::now();
    std::vector<double> fraction(geometry.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < fraction.size(); ++c)
    {
        const auto box = hierarchy->cell_box(geometry.mesh()->cell_global_id(static_cast<LO>(c)));
        const double front = 0.25 * grid.extent[0] + 0.1;
        fraction[c] = std::clamp((front - box.lower[0]) / (box.upper[0] - box.lower[0]), 0.0, 1.0);
    }
    RefinementOptions options;
    options.curvature_resolution = 0.0;
    const std::vector<double> velocity(geometry.mesh()->num_faces(), 0.0);
    std::vector<std::size_t> selected;
    for (std::size_t n = 0; n < repetitions; ++n)
        selected = geometry.refinement_plan(fraction, velocity, options).refine;
    const auto completed = Clock::now();
    std::uint64_t checksum = 0;
    for (const auto b : selected)
        checksum = checksum * 1099511628211ULL + b + 1;
    const auto memory = Benchmark::sample_process_memory();
    std::cout << "block_axis,leaves,global_cells,ranks,repetitions,construct_s,refine_s,selected_count,selected_"
                 "checksum,peak_rss_kib\n"
              << std::setprecision(17) << "2," << hierarchy->blocks().size() << ',' << fraction.size() << ",1,"
              << repetitions << ',' << std::chrono::duration<double>(constructed - start).count() << ','
              << std::chrono::duration<double>(completed - constructed).count() << ',' << selected.size() << ','
              << checksum << ',' << memory.peak_resident_kib << '\n';
    if (argc > 3)
    {
        for (const auto b : selected)
            std::cout << b << '\n';
    }
}
