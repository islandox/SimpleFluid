/** @file adaptive_vof_geometry_benchmark.cc
 * @brief Opt-in corrected-baseline geometry timings; no performance acceptance threshold.
 */
#include "FVM/AdaptiveVoF.hh"
#include "benchmarks/BenchmarkSupport.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    using namespace SimpleFluid;
    using namespace VoF;
    using Clock = std::chrono::steady_clock;
    using LO = DefaultTpetraTypes::local_ordinal_type;
    const auto comm = Tpetra::getDefaultComm();
    const std::size_t block_cells = argc > 1 ? std::stoul(argv[1]) : 4;
    const std::size_t repetitions = argc > 2 ? std::stoul(argv[2]) : 3;
    if (block_cells < 2 || block_cells % 2 || repetitions == 0)
        return 2;
    Meshes::AMRGrid grid;
    grid.root_blocks = {4, 2, 1};
    grid.block_cells = {block_cells, block_cells, block_cells};
    grid.extent = {2.0, 1.0, 0.5};
    grid.periodic = {true, true, true};
    grid.max_level = 1;
    const auto hierarchy = std::make_shared<Meshes::CartesianAMR>(grid);
    const auto start = Clock::now();
    AdaptiveVoF geometry(hierarchy);
    const auto constructed = Clock::now();
    // Keep all instances alive: this measures sharing rather than allocator reuse.
    std::vector<std::unique_ptr<AdaptiveVoF>> retained;
    for (std::size_t n = 0; n < 3; ++n)
        retained.push_back(std::make_unique<AdaptiveVoF>(hierarchy));
    const auto shared = Clock::now();
    std::vector<double> f(geometry.mesh()->num_owned_cells());
    for (std::size_t c = 0; c < f.size(); ++c)
    {
        const auto box = hierarchy->cell_box(geometry.mesh()->cell_global_id(static_cast<LO>(c)));
        const double x = 0.5 * (box.lower[0] + box.upper[0]);
        const double y = 0.5 * (box.lower[1] + box.upper[1]);
        f[c] = std::clamp(
            0.5 + 0.7 * std::sin(2.0 * std::numbers::pi * x) * std::cos(2.0 * std::numbers::pi * y), 0.0, 1.0);
    }
    std::ofstream dump;
    if (argc > 3)
        dump.open(std::string(argv[3]) + ".rank" + std::to_string(comm->getRank()) + ".csv");
    dump << std::setprecision(17);
    double checksum = 0.0;
    for (std::size_t n = 0; n < repetitions; ++n)
    {
        const auto planes = geometry.reconstruct(f);
        for (std::size_t c = 0; c < planes.size(); ++c)
        {
            const auto& p = planes[c];
            checksum += p.offset + p.normal[0] + 2.0 * p.normal[1] + 3.0 * p.normal[2];
            if (n == 0 && dump)
                dump << "plane," << geometry.mesh()->cell_global_id(static_cast<LO>(c)) << ',' << p.normal[0] << ','
                     << p.normal[1] << ',' << p.normal[2] << ',' << p.offset << '\n';
        }
    }
    const auto reconstructed = Clock::now();
    for (std::size_t n = 0; n < repetitions; ++n)
    {
        const auto curvature = geometry.curvature(f);
        for (std::size_t c = 0; c < curvature.size(); ++c)
        {
            checksum += curvature[c];
            if (n == 0 && dump)
                dump << "curvature," << geometry.mesh()->cell_global_id(static_cast<LO>(c)) << ',' << curvature[c]
                     << '\n';
        }
    }
    const auto curved = Clock::now();
    RefinementOptions options;
    options.curvature_resolution = 0.0;
    const std::vector<double> velocity(geometry.mesh()->num_faces(), 0.0);
    for (std::size_t n = 0; n < repetitions; ++n)
        for (const auto b : geometry.refinement_plan(f, velocity, options).refine)
            checksum += static_cast<double>(b + 1);
    const auto refined = Clock::now();
    const auto global_max = [&](double local)
    {
        double result = 0.0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &result);
        return result;
    };
    const auto seconds = [&](auto a, auto b) { return global_max(std::chrono::duration<double>(b - a).count()); };
    const auto construct_s = seconds(start, constructed), share_s = seconds(constructed, shared),
               reconstruct_s = seconds(shared, reconstructed), curvature_s = seconds(reconstructed, curved),
               refine_s = seconds(curved, refined);
    double global_checksum = 0.0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &checksum, &global_checksum);
    const auto memory = Benchmark::sample_process_memory();
    const auto rss = global_max(static_cast<double>(memory.peak_resident_kib));
    const auto storage = geometry.geometry_storage();
    const auto payload = global_max(static_cast<double>(storage.sample_payload_bytes));
    const auto virtual_cells = global_max(static_cast<double>(storage.virtual_cells));
    const auto direct_cells = global_max(static_cast<double>(storage.direct_cells));
    const auto explicit_intersections = global_max(static_cast<double>(storage.explicit_intersections));
    if (comm->getRank() == 0)
    {
        std::cout << "block_axis,root_blocks,global_cells,populations,ranks,repetitions,construct_s,retain_three_s,"
                     "reconstruct_s,curvature_s,refine_s,max_rank_peak_rss_kib,checksum,max_rank_virtual_cells,max_"
                     "rank_direct_cells,max_rank_explicit_intersections,max_rank_sample_payload_estimate_bytes\n"
                  << std::setprecision(17) << block_cells << ",8," << 8 * block_cells * block_cells * block_cells
                  << ",0," << comm->getSize() << ',' << repetitions << ',' << construct_s << ',' << share_s << ','
                  << reconstruct_s << ',' << curvature_s << ',' << refine_s << ',' << rss << ',' << global_checksum
                  << ',' << virtual_cells << ',' << direct_cells << ',' << explicit_intersections << ',' << payload
                  << '\n';
    }
}
