/** @file region_scaling_benchmark.cc @brief Independent N/R/P mesh execution diagnostics. */
#include "benchmarks/BenchmarkSupport.hh"
#include "FVM/Operators.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Clock = std::chrono::steady_clock;

ArrReal axis(size_t first, size_t last, size_t n)
{
    ArrReal result;
    result.reserve(last - first + 1);
    for (size_t i = first; i <= last; ++i) result.push_back(real_t(i) / n);
    return result;
}

SP<MultiRegionMesh> composite(size_t n, size_t count)
{
    std::vector<MultiRegionMesh::Region> regions;
    std::vector<MultiRegionMesh::Interface> interfaces;
    const auto yz = axis(0, n, n);
    for (size_t r = 0; r < count; ++r)
    {
        regions.push_back(cartesian_region("region_" + std::to_string(r),
            {{axis(r * n / count, (r + 1) * n / count, n), yz, yz}}));
        if (r) interfaces.emplace_back(StructuredPatchInterface{{r - 1, 1}, {r, 0}});
    }
    return std::make_shared<MultiRegionMesh>(std::move(regions), std::move(interfaces));
}

// Construct explicit nodes and hexahedra directly. No composite is retained and
// no compatibility-materialized structured object substitutes for this baseline.
SP<UnstructuredMesh> flattened(size_t n)
{
    using U = UnstructuredMesh;
    Arr<MeshUtils::Vec3> nodes;
    Arr<U::CellDefinition> cells;
    Arr<U::BoundaryFaceDefinition> boundaries;
    nodes.reserve((n + 1) * (n + 1) * (n + 1));
    cells.reserve(n * n * n);
    boundaries.reserve(6 * n * n);
    auto node = [n](size_t i, size_t j, size_t k) -> U::NodeID
    { return (k * (n + 1) + j) * (n + 1) + i; };
    for (size_t k = 0; k <= n; ++k)
        for (size_t j = 0; j <= n; ++j)
            for (size_t i = 0; i <= n; ++i)
                nodes.push_back({real_t(i) / n, real_t(j) / n, real_t(k) / n});
    for (size_t k = 0; k < n; ++k)
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < n; ++i)
            {
                const std::array<U::NodeID, 8> v{node(i,j,k), node(i+1,j,k), node(i+1,j+1,k), node(i,j+1,k),
                    node(i,j,k+1), node(i+1,j,k+1), node(i+1,j+1,k+1), node(i,j+1,k+1)};
                cells.push_back({U::CellType::HEXAHEDRON, {v.begin(), v.end()}});
                if (i == 0) boundaries.push_back({{v[0],v[3],v[7],v[4]},0,"xmin"});
                if (i + 1 == n) boundaries.push_back({{v[1],v[5],v[6],v[2]},1,"xmax"});
                if (j == 0) boundaries.push_back({{v[0],v[4],v[5],v[1]},2,"ymin"});
                if (j + 1 == n) boundaries.push_back({{v[3],v[2],v[6],v[7]},3,"ymax"});
                if (k == 0) boundaries.push_back({{v[0],v[1],v[2],v[3]},4,"zmin"});
                if (k + 1 == n) boundaries.push_back({{v[4],v[7],v[6],v[5]},5,"zmax"});
            }
    return std::make_shared<U>(nodes, cells, boundaries);
}

real_t analytic(MeshUtils::Vec3 p) { return 1 + p.x + 2 * p.y - .5 * p.z; }

size_t positive_integer(const char* value)
{
    const std::string text(value);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("axis cells, regions and repeats must be positive integers");
    const auto result = std::stoull(text);
    if (!result) throw std::invalid_argument("axis cells, regions and repeats must be positive integers");
    return result;
}
}

int main(int argc, char** argv)
{
    Tpetra::ScopeGuard guard(&argc, &argv);
    const auto comm = Tpetra::getDefaultComm();
    try
    {
        if (argc != 6)
            throw std::invalid_argument("usage: region_scaling_benchmark native|composite|materialized|flattened "
                "axis_cells regions traversal_and_apply_repeats output.vtu");
        const std::string mode = argv[1];
        const size_t n = positive_integer(argv[2]), regions = positive_integer(argv[3]), repeats = positive_integer(argv[4]);
        if (mode != "native" && mode != "composite" && mode != "materialized" && mode != "flattened")
            throw std::invalid_argument("unknown mesh baseline");
        if (n < 2 || n > 512 || regions > n || n % regions || repeats > 100000)
            throw std::invalid_argument("require 2 <= axis_cells <= 512, regions divides axis_cells, repeats <= 100000");
        if (mode == "flattened" && comm->getSize() != 1)
        {
            if (!comm->getRank())
                std::cout << "{\"type\":\"unsupported\",\"mode\":\"flattened\",\"ranks\":" << comm->getSize()
                    << ",\"reason\":\"explicit unstructured MPI requires a separately partitioned reference\"}\n";
            return 77;
        }
        if (!comm->getRank())
            std::cout << "{\"type\":\"metadata\",\"commit\":\"" << SIMPLEFLUID_GIT_COMMIT
                << "\",\"dirty\":" << SIMPLEFLUID_GIT_DIRTY << ",\"build\":\"" << SIMPLEFLUID_BUILD_TYPE
                << "\",\"compiler\":\"" << SIMPLEFLUID_COMPILER << "\",\"mode\":\"" << mode
                << "\",\"axis_cells\":" << n << ",\"global_cells\":" << n*n*n << ",\"regions_requested\":" << regions
                << ",\"representation_regions\":" << ((mode == "composite" || mode == "materialized") ? regions : 1)
                << ",\"ranks\":" << comm->getSize() << ",\"repeats\":" << repeats
                << ",\"solver\":\"cg\",\"preconditioner\":\"jacobi\",\"tolerance\":1e-10,\"max_iterations\":1000}\n";

        auto begin = [&] { comm->barrier(); return Clock::now(); };
        auto elapsed = [](auto start) { return std::chrono::duration<double>(Clock::now() - start).count(); };
        SP<Handle> mesh;
        auto start = begin();
        if (mode == "native")
        {
            const auto a = axis(0, n, n);
            mesh = std::make_shared<Handle>(std::make_shared<OrthogonalCartesian3D>(Vec3D<ArrReal>{{a,a,a}}));
        }
        else if (mode == "flattened") mesh = std::make_shared<Handle>(flattened(n));
        else
        {
            mesh = std::make_shared<Handle>(composite(n, regions));
            if (mode == "materialized") { mesh->materialize_cell_faces(); static_cast<void>(mesh->indexer()); }
        }
        const double construction = elapsed(start);
        const bool initial_indexer = mesh->has_materialized_indexer();

        auto checkpoint = [&](const char* stage, double seconds)
        {
            const auto storage = mesh->storage_report();
            const auto memory = Benchmark::sample_process_memory();
            const double local[] = {seconds, mesh->num_owned_cells() ?
                double(mesh->num_local_cells() - mesh->num_owned_cells()) / mesh->num_owned_cells() : 0.};
            double maximum[2]{};
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 2, local, maximum);
            const long long values[] = {memory.peak_resident_kib * 1024,
                static_cast<long long>(storage.measured_bytes()), static_cast<long long>(storage.topology),
                static_cast<long long>(storage.geometry), static_cast<long long>(storage.interfaces),
                static_cast<long long>(storage.indexing), static_cast<long long>(storage.compatibility),
                static_cast<long long>(storage.boundaries), static_cast<long long>(storage.objects),
                static_cast<long long>(storage.indexing_estimate), static_cast<long long>(storage.required_maps_estimate),
                static_cast<long long>(mesh->num_owned_cells()),
                static_cast<long long>(mesh->num_local_cells() - mesh->num_owned_cells())};
            long long maxima[13]{}, sums[13]{};
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 13, values, maxima);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 13, values, sums);
            if (!comm->getRank())
            {
                const char* names[] = {"peak_rss_bytes", "measured_mesh_bytes", "topology_bytes", "geometry_bytes",
                    "interfaces_bytes", "indexing_bytes", "compatibility_bytes", "boundary_bytes", "object_bytes",
                    "index_tree_estimate_bytes", "map_id_payload_estimate_bytes", "owned_cells", "ghost_cells"};
                std::cout << std::setprecision(15) << "{\"type\":\"checkpoint\",\"stage\":\"" << stage
                    << "\",\"seconds_max_rank\":" << maximum[0] << ",\"ghost_owned_ratio_max_rank\":" << maximum[1];
                for (size_t i = 0; i < 13; ++i)
                    std::cout << ",\"" << names[i] << "_max_rank\":" << maxima[i]
                        << ",\"" << names[i] << "_sum_ranks\":" << sums[i];
                std::cout << "}\n" << std::flush;
            }
        };
        checkpoint("construction", construction);

        real_t checksum = 0, native_checksum = 0;
        {
            start = begin();
            const auto execution = mesh->acquire_execution_view();
            checkpoint("execution_view_setup", elapsed(start));
            start = begin();
            for (size_t repetition = 0; repetition < repeats; ++repetition)
                for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
                {
                    checksum += mesh->cell_volume(c) * (1 + mesh->cell_centroid(c).norm());
                    auto faces = [&](const auto& range)
                    { for (const auto f : range) checksum += mesh->face_area(f); };
                    if (mode == "materialized") faces(mesh->materialized_faces(c)); else faces(mesh->faces(c));
                }
            checkpoint("traversal_average", elapsed(start) / repeats);
            if (mode == "composite" || mode == "materialized")
            {
                // The current composite partition is a contiguous canonical
                // interval. Translate it once per region, without global cells
                // or a face-to-native lookup table. This is a provider traversal
                // diagnostic, not a replacement for canonical FVM assembly.
                const size_t count = mesh->num_owned_cells();
                const size_t first = count ? mesh->owned_cell_map()->getGlobalElement(0) : 0;
                const size_t last = count ? mesh->owned_cell_map()->getGlobalElement(count - 1) + 1 : 0;
                if (last - first != count)
                    throw std::logic_error("native region diagnostic requires contiguous composite ownership");
                start = begin();
                for (size_t repetition = 0; repetition < repeats; ++repetition)
                    execution.visit_regions([&](size_t, size_t offset, const auto& region)
                    {
                        const size_t a = std::max(first, offset), b = std::min(last, offset + region.layout().cells);
                        for (size_t global = a; global < b; ++global)
                        {
                            const size_t c = global - offset;
                            const auto& geometry = region.geometry();
                            native_checksum += geometry.cell_volume(c) * (1 + geometry.cell_centroid(c).norm());
                            for (const auto f : region.topology().cell_faces(c))
                                native_checksum += geometry.face_area(f);
                        }
                    });
                checkpoint("native_region_traversal_average", elapsed(start) / repeats);
            }
        }
        real_t global_checksum = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &checksum, &global_checksum);
        real_t global_native_checksum = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &native_checksum, &global_native_checksum);
        if ((mode == "composite" || mode == "materialized")
            && std::abs(global_checksum - global_native_checksum) > 1e-10 * std::abs(global_checksum))
            throw std::runtime_error("native and canonical Cartesian traversal checksums disagree");

        auto condition = [&](int b, size_t i)
        { return BoundaryCondition{BoundaryConditionType::Dirichlet,
            analytic(mesh->face_centroid(mesh->boundary_face_batch(b).face_lids[i]))}; };
        start = begin();
        const auto system = FVM::diffusion_system<Pack>(*mesh, 1., condition, [](int) { return 0.; });
        checkpoint("assembly", elapsed(start));

        Pack::vector_type x(mesh->owned_cell_map(), true), y(mesh->owned_cell_map(), true);
        {
            const auto execution = mesh->acquire_execution_view();
            for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
                x.replaceLocalValue(c, analytic(mesh->cell_centroid(c)));
        }
        system.matrix->apply(x, y); // warm application/import workspace equally
        start = begin();
        for (size_t i = 0; i < repeats; ++i) system.matrix->apply(x, y);
        checkpoint("application_average", elapsed(start) / repeats);

        BelosLinearSolver<Pack> solver;
        LinearSolverOptions options;
        options.backend = LinearSolverBackend::Cg;
        options.preconditioner = LinearPreconditioner::Jacobi;
        options.tolerance = 1e-10;
        options.max_iterations = 1000;
        x.putScalar(0.);
        start = begin();
        const auto solve = solver.solve_with_statistics(system.matrix, *system.rhs, x, options);
        checkpoint("solve_including_preconditioner", elapsed(start));
        system.matrix->apply(x, y);
        y.update(-1., *system.rhs, 1.);
        const double residual = y.norm2() / std::max(1e-300, system.rhs->norm2());
        const auto solution = x.getData();
        real_t local_error = 0, error = 0;
        {
            const auto execution = mesh->acquire_execution_view();
            for (size_t c = 0; c < mesh->num_owned_cells(); ++c)
                local_error = std::max(local_error, std::abs(solution[c] - analytic(mesh->cell_centroid(c))));
        }
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_error, &error);
        if (!solve.converged || !std::isfinite(residual) || residual > 1e-9 || error > 1e-7)
            throw std::runtime_error("benchmark solve failed convergence, true-residual or analytic solution gate");

        start = begin();
        mesh->export_vtu(argv[5]);
        checkpoint("output", elapsed(start));
        const int expanded = (mode == "native" || mode == "composite")
            && (mesh->connectivity_storage_bytes() || mesh->has_materialized_indexer() != initial_indexer);
        int any_expanded = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &expanded, &any_expanded);
        if (any_expanded) throw std::runtime_error("compact benchmark materialized compatibility storage");
        if (!comm->getRank())
            std::cout << std::setprecision(15) << "{\"type\":\"result\",\"converged\":true,\"iterations\":"
                << solve.iterations << ",\"true_relative_residual\":" << residual << ",\"solution_max_error\":"
                << error << ",\"traversal_checksum\":" << global_checksum / repeats
                << ",\"native_region_traversal_checksum\":" << global_native_checksum / repeats << "}\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "rank " << comm->getRank() << ": " << error.what() << '\n';
        return 1;
    }
}
