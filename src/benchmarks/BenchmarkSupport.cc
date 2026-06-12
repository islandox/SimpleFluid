/**
 * @file BenchmarkSupport.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Implementations of benchmark support utilities: memory, CSV, timestamps.
 * @version 0.1
 * @date 2026-06-12
 *
 * @copyright Copyright (c) 2026
 *
 */
#include "benchmarks/BenchmarkSupport.hh"

#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace SimpleFluid::Benchmark
{

namespace
{

/**
 * @brief Read a memory value in KiB from /proc/self/status.
 *
 * @param key Status file key (e.g., "VmRSS:").
 * @return Value in KiB, or 0 if not found.
 */
long long status_value_kib(std::string_view key)
{
    std::ifstream status("/proc/self/status");
    std::string name;
    long long value = 0;
    std::string unit;
    while (status >> name >> value >> unit)
    {
        if (name == key)
        {
            return value;
        }
        status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}

/**
 * @brief Append a value to an output stream (generic overload).
 *
 * @tparam Value Type of value to write.
 * @param output Output stream.
 * @param value Value to append.
 */
template<class Value>
void append_value(std::ostream& output, const Value& value)
{
    output << value;
}

void append_value(std::ostream& output, const std::string& value)
{
    output << csv_escape(value);
}

} // namespace

/**
 * @brief Sample current and peak resident memory usage of the process.
 *
 * Reads from /proc/self/status, falling back to getrusage.
 *
 * @return ProcessMemory with resident_kib and peak_resident_kib populated.
 */
ProcessMemory sample_process_memory()
{
    ProcessMemory result{
        status_value_kib("VmRSS:"),
        status_value_kib("VmHWM:")};

    if (result.peak_resident_kib == 0)
    {
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0)
        {
#if defined(__APPLE__)
            result.peak_resident_kib = usage.ru_maxrss / 1024;
#else
            result.peak_resident_kib = usage.ru_maxrss;
#endif
        }
    }
    if (result.resident_kib == 0)
    {
        result.resident_kib = result.peak_resident_kib;
    }
    return result;
}

/**
 * @brief Return the current UTC timestamp as an ISO 8601 string.
 *
 * @return Timestamp formatted as YYYY-MM-DDTHH:MM:SSZ.
 */
std::string utc_timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&time, &utc);

    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

/**
 * @brief Return the hostname of the current machine.
 *
 * @return Hostname string, or "unknown" on failure.
 */
std::string host_name()
{
    char buffer[256]{};
    if (gethostname(buffer, sizeof(buffer) - 1) != 0)
    {
        return "unknown";
    }
    return buffer;
}

/**
 * @brief Escape a string value for inclusion in a CSV field.
 *
 * Wraps the value in quotes and doubles embedded quotes if the value
 * contains commas, quotes, newlines, or carriage returns.
 *
 * @param value String to escape.
 * @return CSV-safe escaped string.
 */
std::string csv_escape(std::string_view value)
{
    const bool needs_quotes =
        value.find_first_of(",\"\r\n") != std::string_view::npos;
    if (!needs_quotes)
    {
        return std::string(value);
    }

    std::string escaped{"\""};
    for (const auto character : value)
    {
        if (character == '"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += character;
        }
    }
    escaped += '"';
    return escaped;
}

/**
 * @brief Return the CSV header line matching the Record schema.
 *
 * @return Comma-separated header string.
 */
std::string CsvWriter::header()
{
    return
        "timestamp,run_id,git_commit,git_dirty,build_type,compiler,hostname,"
        "benchmark_case,preset,mesh_nx,mesh_ny,mesh_nz,global_cells,mpi_ranks,"
        "shear,mean_nonorthogonality_deg,max_nonorthogonality_deg,treatment,"
        "coupling,preconditioner,linear_tolerance,max_linear_iterations,"
        "nonorthogonal_correctors,pressure_correctors,outer_correctors,"
        "repetition,nonlinear_iterations,linear_solves,krylov_iterations,"
        "setup_seconds,solve_seconds,total_seconds,rss_max_rank_kib,"
        "rss_sum_ranks_kib,peak_rss_max_rank_kib,peak_rss_sum_ranks_kib,"
        "momentum_residual,pressure_residual,temperature_residual,"
        "continuity_residual,l2_error,linf_error,achieved_tolerance,converged";
}

/**
 * @brief Construct a CsvWriter, creating the output directory and validating any existing header.
 *
 * @param path Path to the output CSV file.
 * @throws std::runtime_error if an existing CSV has an incompatible header or the file cannot be created.
 */
CsvWriter::CsvWriter(std::filesystem::path path)
    : d_path(std::move(path))
{
    if (d_path.has_parent_path())
    {
        std::filesystem::create_directories(d_path.parent_path());
    }

    if (std::filesystem::exists(d_path)
        && std::filesystem::file_size(d_path) > 0)
    {
        std::ifstream input(d_path);
        std::string existing_header;
        std::getline(input, existing_header);
        if (existing_header != header())
        {
            throw std::runtime_error(
                "Benchmark CSV header is incompatible with the current schema.");
        }
        return;
    }

    std::ofstream output(d_path);
    if (!output)
    {
        throw std::runtime_error("Could not create benchmark CSV output.");
    }
    output << header() << '\n';
}

/**
 * @brief Append a benchmark record as a new CSV row.
 *
 * @param record Benchmark record to write.
 * @throws std::runtime_error if the output file cannot be opened for appending.
 */
void CsvWriter::append(const Record& record)
{
    std::ofstream output(d_path, std::ios::app);
    if (!output)
    {
        throw std::runtime_error("Could not append benchmark CSV output.");
    }
    output << std::setprecision(17);

    bool first = true;
    auto append =
        [&](const auto& value)
    {
        if (!first)
        {
            output << ',';
        }
        first = false;
        append_value(output, value);
    };

    append(record.timestamp);
    append(record.run_id);
    append(record.git_commit);
    append(record.git_dirty);
    append(record.build_type);
    append(record.compiler);
    append(record.hostname);
    append(record.benchmark_case);
    append(record.preset);
    append(record.mesh_nx);
    append(record.mesh_ny);
    append(record.mesh_nz);
    append(record.global_cells);
    append(record.mpi_ranks);
    append(record.shear);
    append(record.mean_nonorthogonality_deg);
    append(record.max_nonorthogonality_deg);
    append(record.treatment);
    append(record.coupling);
    append(record.preconditioner);
    append(record.linear_tolerance);
    append(record.max_linear_iterations);
    append(record.nonorthogonal_correctors);
    append(record.pressure_correctors);
    append(record.outer_correctors);
    append(record.repetition);
    append(record.nonlinear_iterations);
    append(record.linear_solves);
    append(record.krylov_iterations);
    append(record.setup_seconds);
    append(record.solve_seconds);
    append(record.total_seconds);
    append(record.rss_max_rank_kib);
    append(record.rss_sum_ranks_kib);
    append(record.peak_rss_max_rank_kib);
    append(record.peak_rss_sum_ranks_kib);
    append(record.momentum_residual);
    append(record.pressure_residual);
    append(record.temperature_residual);
    append(record.continuity_residual);
    append(record.l2_error);
    append(record.linf_error);
    append(record.achieved_tolerance);
    append(record.converged ? 1 : 0);
    output << '\n';
}

} // namespace SimpleFluid::Benchmark
