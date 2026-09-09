/** @file coupled_operator_memory.cc @brief Opt-in, one-backend-per-process coupled benchmark. */
#include "equations/IncompressibleMomentumEquation.hh"
#include "equations/TimeStepperOptions.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "solvers/CoupledPressureVelocitySolver.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    using namespace SimpleFluid;
    using Pack = DefaultTpetraTypes;
    using Handle = MeshHandle<Pack>;
    using Solver = CoupledPressureVelocitySolver<Pack, Handle>;
    const auto comm = Tpetra::getDefaultComm();
    try
    {
        if (argc != 3 && argc != 4)
            throw std::invalid_argument("usage: coupled_operator_memory assembled|block_composite cells_per_axis "
                                        "[cached_products|streamed_products]");
        TimeStepperOptions options;
        options.coupled_operator_backend = coupled_operator_backend_from_string(argv[1]);
        if (argc == 4)
            options.coupled_workspace_policy = coupled_workspace_policy_from_string(argv[3]);
        const int n = std::stoi(argv[2]);
        if (n < 2 || n > 100)
            throw std::invalid_argument("cells_per_axis must be in [2,100]");
        options.time_step = .01;
        options.non_orthogonal_treatment = FVM::NonOrthogonalTreatment::Explicit;
        Vec3D<ArrReal> edges;
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k <= n; ++k)
                edges[c].push_back(double(k) / n);
        SP<const Handle> mesh = std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(edges));
        VectorCellFieldStored<Pack> u(mesh, vec3<double>{.1, .2, -.1}, "u");
        ScalarCellFieldStored<Pack> p(mesh, 0., "p");
        ScalarFaceFieldStored<Pack> flux(mesh, 0., "phi");
        BoundaryConditionSet boundaries;
        for (const auto* name : {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"})
            boundaries.velocity[name] = {BoundaryConditionType::NoSlip, {}};
        const auto boundary = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
        IncompressibleMomentumEquation<Pack, Handle> equation(mesh);
        Solver solver(mesh);
        LinearSolverOptions linear;
        linear.tolerance = 1e-9;
        linear.max_iterations = 400; // same restart / basis budget in every process
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto seconds = [&](auto start) { return std::chrono::duration<double>(now() - start).count(); };
        if (comm->getRank() == 0)
            std::cout << "{\"type\":\"metadata\",\"commit\":\"" << SIMPLEFLUID_GIT_COMMIT
                      << "\",\"dirty\":" << SIMPLEFLUID_GIT_DIRTY << ",\"compiler\":\"" << SIMPLEFLUID_COMPILER
                      << "\",\"build\":\"" << SIMPLEFLUID_BUILD_TYPE << "\",\"backend_requested\":\"" << argv[1]
                      << "\",\"backend_effective\":\"" << to_string(options.coupled_operator_backend)
                      << "\",\"workspace\":\"" << to_string(options.coupled_workspace_policy)
                      << "\",\"ranks\":" << comm->getSize()
                      << ",\"global_cells\":" << mesh->owned_cell_map()->getGlobalNumElements()
                      << ",\"scalar_unknowns\":" << 4 * mesh->owned_cell_map()->getGlobalNumElements()
                      << ",\"tolerance\":" << linear.tolerance
                      << ",\"max_iterations\":" << linear.max_iterations
                      << ",\"block_size\":4,\"num_blocks\":"
                      << std::min<size_t>(linear.max_iterations / 4, mesh->owned_cell_map()->getGlobalNumElements())
                      << ",\"preconditioner\":\"Ifpack2_Jacobi_MueLu_full_reuse\""
                      << ",\"scalar_bytes\":" << sizeof(Pack::scalar_type)
                      << ",\"local_index_bytes\":" << sizeof(Pack::local_ordinal_type)
                      << ",\"global_index_bytes\":" << sizeof(Pack::global_ordinal_type)
                      << ",\"offset_bytes\":" << sizeof(Pack::size_type) << ",\"kokkos\":\""
                      << Pack::execution_space::name()
                      << "\",\"unsupported_metrics\":[\"map_import_bytes\",\"preconditioner_internal_bytes\",\"krylov_"
                         "bytes\",\"allocator_capacity\",\"device_rss\",\"separate_preconditioner_setup_checkpoint\"]}\n";
        auto checkpoint = [&](const char* stage, double elapsed, int iterations, double residual, double continuity)
        {
            comm->barrier();
            const auto storage = solver.storage_statistics();
            rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            long resident_pages = 0, virtual_pages = 0;
            std::ifstream("/proc/self/statm") >> virtual_pages >> resident_pages;
            const long long rss = resident_pages * sysconf(_SC_PAGESIZE), peak = usage.ru_maxrss * 1024LL;
            long long maximum_peak = 0, simultaneous_sum = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &peak, &maximum_peak);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &rss, &simultaneous_sum);
            // Rank-ordered JSON lines, no vector/matrix gathering.
            for (int rank = 0; rank < comm->getSize(); ++rank)
            {
                if (rank == comm->getRank())
                    std::cout << std::setprecision(12) << "{\"type\":\"checkpoint\",\"stage\":\"" << stage
                              << "\",\"rank\":" << rank << ",\"owned_cells\":" << mesh->num_owned_cells()
                              << ",\"seconds\":" << elapsed << ",\"iterations\":" << iterations
                              << ",\"true_relative_residual\":" << residual << ",\"continuity_l2\":" << continuity
                              << ",\"rss_bytes\":" << rss << ",\"peak_rss_bytes\":" << peak
                              << ",\"maximum_rank_peak_rss_bytes\":" << maximum_peak
                              << ",\"simultaneous_sum_rss_bytes\":" << simultaneous_sum
                              << ",\"live_matrices\":" << storage.live_matrices
                              << ",\"graph_view_bytes\":" << storage.graph_bytes
                              << ",\"value_view_bytes\":" << storage.value_bytes
                              << ",\"composite_scratch_payload_bytes\":" << storage.composite_scratch_bytes
                              << ",\"coupled_matrix_builds\":" << solver.cache_statistics().coupled_matrix_builds
                              << "}\n"
                              << std::flush;
                comm->barrier();
            }
        };
        checkpoint("mesh_fields", 0, 0, 0, 0);
        auto start = now();
        auto system = solver.assemble(equation, u, p, flux, boundary, boundaries, options);
        checkpoint("operator_setup", seconds(start), 0, 0, 0);
        Pack::multi_vector_type x(system.map, 1), y(system.map, 1);
        x.putScalar(1.0);
        system.linear_operator->apply(x, y); // consistently warm application workspace
        start = now();
        for (int k = 0; k < 20; ++k)
            system.linear_operator->apply(x, y);
        checkpoint("apply_average", seconds(start) / 20, 0, 0, 0);
        auto solve = [&](const char* label)
        {
            start = now();
            const auto result = solver.solve(system, u, p, linear);
            const auto elapsed = seconds(start);
            Pack::vector_type solution(system.map), residual(system.map);
            for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
            {
                const auto v = u.value(static_cast<Pack::local_ordinal_type>(cell));
                for (int c = 0; c < 3; ++c)
                    solution.replaceLocalValue(4 * cell + c, v.component(c));
                solution.replaceLocalValue(
                    4 * cell + 3, p.value(static_cast<Pack::local_ordinal_type>(cell)) / system.reference_density);
            }
            system.linear_operator->apply(solution, residual);
            residual.update(-1., *system.rhs, 1.);
            checkpoint(label, elapsed, result.iterations, residual.norm2() / std::max(1e-300, system.rhs->norm2()),
                result.continuity.l2);
            if (!result.converged)
                throw std::runtime_error("benchmark solve did not converge");
        };
        solve("first_solve_including_preconditioner");
        for (int update = 0; update < 3; ++update)
        {
            system = {}; // production ownership: no externally retained previous system
            options.time_step *= 1.1;
            for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
                u.set_owned_value(static_cast<Pack::local_ordinal_type>(cell), {.1, .2, -.1});
            p.owned_data().putScalar(0.0);
            start = now();
            system = solver.assemble(equation, u, p, flux, boundary, boundaries, options);
            checkpoint("numeric_update_setup", seconds(start), 0, 0, 0);
            solve("updated_solve");
        }
        system = {};
        solver.clear_cache();
        checkpoint("cleared", 0, 0, 0, 0);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
