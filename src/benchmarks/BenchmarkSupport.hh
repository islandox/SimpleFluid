/**
 * @file BenchmarkSupport.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Benchmark infrastructure: process memory sampling, CSV records, and CsvWriter.
 * @version 0.1
 * @date 2026-06-12
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace SimpleFluid::Benchmark
{

/**
 * @brief Snapshot of process memory usage from /proc/self/status.
 *
 * Values are in KiB; `resident_kib` is current usage and
 * `peak_resident_kib` is the process high-water mark.
 */
struct ProcessMemory
{
    long long resident_kib = 0;
    long long peak_resident_kib = 0;
};

ProcessMemory sample_process_memory();
std::string utc_timestamp();
std::string host_name();
std::string csv_escape(std::string_view value);

/**
 * @brief Benchmark record containing timing, convergence, and resource metrics.
 *
 * All fields are populated from a solver run and written as a CSV row.
 *
 * Field suffixes encode units: `_seconds`, `_kib`, and `_deg` denote seconds,
 * KiB, and degrees. `rss_max_rank_kib`/`rss_sum_ranks_kib` are respectively
 * the maximum and sum across ranks; the `peak_rss_` variants aggregate each
 * process high-water mark. `linear_tolerance` is relative.
 */
struct Record
{
    std::string timestamp;
    std::string run_id;
    std::string git_commit;
    std::string git_dirty;
    std::string build_type;
    std::string compiler;
    std::string hostname;
    std::string benchmark_case;
    std::string preset;
    long long mesh_nx = 0;
    long long mesh_ny = 0;
    long long mesh_nz = 0;
    long long global_cells = 0;
    int mpi_ranks = 1;
    double shear = 0.0;
    double mean_nonorthogonality_deg = 0.0;
    double max_nonorthogonality_deg = 0.0;
    std::string treatment;
    std::string coupling;
    std::string preconditioner;
    double linear_tolerance = 0.0;
    int max_linear_iterations = 0;
    int nonorthogonal_correctors = 0;
    int pressure_correctors = 0;
    int outer_correctors = 0;
    int repetition = 0;
    int nonlinear_iterations = 0;
    int linear_solves = 0;
    int krylov_iterations = 0;
    double setup_seconds = 0.0;
    double solve_seconds = 0.0;
    double total_seconds = 0.0;
    long long rss_max_rank_kib = 0;
    long long rss_sum_ranks_kib = 0;
    long long peak_rss_max_rank_kib = 0;
    long long peak_rss_sum_ranks_kib = 0;
    double momentum_residual = 0.0;
    double pressure_residual = 0.0;
    double temperature_residual = 0.0;
    double continuity_residual = 0.0;
    double l2_error = 0.0;
    double linf_error = 0.0;
    double achieved_tolerance = 0.0;
    bool converged = false;
};

/**
 * @brief Writes benchmark records to a CSV file with schema validation.
 *
 * On construction, the writer validates that any existing CSV has a compatible header.
 * Records are appended one row at a time.
 */
class CsvWriter
{
public:
    explicit CsvWriter(std::filesystem::path path);

    void append(const Record& record);

    static std::string header();

private:
    std::filesystem::path d_path;
};

} // namespace SimpleFluid::Benchmark
