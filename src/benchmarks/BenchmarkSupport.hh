#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace SimpleFluid::Benchmark
{

struct ProcessMemory
{
    long long resident_kib = 0;
    long long peak_resident_kib = 0;
};

ProcessMemory sample_process_memory();
std::string utc_timestamp();
std::string host_name();
std::string csv_escape(std::string_view value);

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
