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

#include <algorithm>
#include <chrono>
#include <cmath>
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

/**
 * @brief Append a CSV-escaped string to an output stream.
 *
 * @param output Output stream receiving the field.
 * @param value String field to escape and append.
 */
void append_value(std::ostream& output, const std::string& value)
{
    output << csv_escape(value);
}

/** @brief Parse one RFC 4180-style CSV row. */
std::vector<std::string> parse_csv_row(std::string_view row)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < row.size(); ++i)
    {
        const auto character = row[i];
        if (character == '"')
        {
            if (quoted && i + 1 < row.size() && row[i + 1] == '"')
            {
                field += '"';
                ++i;
            }
            else
            {
                quoted = !quoted;
            }
        }
        else if (character == ',' && !quoted)
        {
            fields.push_back(std::move(field));
            field.clear();
        }
        else
        {
            field += character;
        }
    }
    if (quoted)
    {
        throw std::runtime_error("Unterminated quoted CSV field.");
    }
    fields.push_back(std::move(field));
    return fields;
}

/** @brief Parse a non-negative integer baseline field. */
long long parse_baseline_integer(
    std::string_view value,
    std::string_view name,
    size_t line_number)
{
    size_t consumed = 0;
    long long parsed = 0;
    bool parsed_successfully = true;
    try
    {
        parsed = std::stoll(std::string(value), &consumed);
    }
    catch (const std::exception&)
    {
        parsed_successfully = false;
    }
    if (!parsed_successfully || consumed != value.size() || parsed < 0)
    {
        throw std::runtime_error(
            "Invalid " + std::string(name) + " on baseline line "
            + std::to_string(line_number) + ".");
    }
    return parsed;
}

/** @brief Parse a finite non-negative floating-point baseline field. */
double parse_baseline_real(
    std::string_view value,
    std::string_view name,
    size_t line_number)
{
    size_t consumed = 0;
    double parsed = 0.0;
    bool parsed_successfully = true;
    try
    {
        parsed = std::stod(std::string(value), &consumed);
    }
    catch (const std::exception&)
    {
        parsed_successfully = false;
    }
    if (!parsed_successfully || consumed != value.size()
        || !std::isfinite(parsed) || parsed < 0.0)
    {
        throw std::runtime_error(
            "Invalid " + std::string(name) + " on baseline line "
            + std::to_string(line_number) + ".");
    }
    return parsed;
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

/** @brief Return the regression-baseline CSV schema. */
std::string RegressionGate::header()
{
    return
        "benchmark_case,preset,mesh_nx,mesh_ny,mesh_nz,mpi_ranks,shear,"
        "treatment,coupling,preconditioner,max_nonlinear_iterations,"
        "max_linear_solves,max_krylov_iterations,max_total_seconds";
}

/**
 * @brief Load and validate a checked-in benchmark baseline.
 */
RegressionGate::RegressionGate(
    std::filesystem::path baseline_path,
    int expected_samples_per_case)
    : d_baseline_path(std::move(baseline_path))
    , d_expected_samples_per_case(expected_samples_per_case)
{
    if (d_expected_samples_per_case <= 0)
    {
        throw std::invalid_argument(
            "Benchmark regression gate requires at least one measured sample.");
    }

    std::ifstream input(d_baseline_path);
    if (!input)
    {
        throw std::runtime_error(
            "Could not open benchmark baseline: "
            + d_baseline_path.string());
    }

    std::string baseline_header;
    std::getline(input, baseline_header);
    if (!baseline_header.empty() && baseline_header.back() == '\r')
    {
        baseline_header.pop_back();
    }
    if (baseline_header != header())
    {
        throw std::runtime_error(
            "Benchmark baseline has an incompatible header: "
            + d_baseline_path.string());
    }

    std::string line;
    size_t line_number = 1;
    while (std::getline(input, line))
    {
        ++line_number;
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.empty())
        {
            continue;
        }

        const auto fields = parse_csv_row(line);
        if (fields.size() != 14)
        {
            throw std::runtime_error(
                "Benchmark baseline line " + std::to_string(line_number)
                + " has " + std::to_string(fields.size())
                + " fields; expected 14.");
        }

        Entry entry;
        entry.benchmark_case = fields[0];
        entry.preset = fields[1];
        entry.mesh_nx = parse_baseline_integer(
            fields[2], "mesh_nx", line_number);
        entry.mesh_ny = parse_baseline_integer(
            fields[3], "mesh_ny", line_number);
        entry.mesh_nz = parse_baseline_integer(
            fields[4], "mesh_nz", line_number);
        const auto mpi_ranks = parse_baseline_integer(
            fields[5], "mpi_ranks", line_number);
        entry.shear = parse_baseline_real(
            fields[6], "shear", line_number);
        entry.treatment = fields[7];
        entry.coupling = fields[8];
        entry.preconditioner = fields[9];
        const auto nonlinear_iterations = parse_baseline_integer(
            fields[10], "max_nonlinear_iterations", line_number);
        const auto linear_solves = parse_baseline_integer(
            fields[11], "max_linear_solves", line_number);
        const auto krylov_iterations = parse_baseline_integer(
            fields[12], "max_krylov_iterations", line_number);
        entry.max_total_seconds = parse_baseline_real(
            fields[13], "max_total_seconds", line_number);

        if (entry.mesh_nx == 0 || entry.mesh_ny == 0 || entry.mesh_nz == 0
            || mpi_ranks == 0
            || mpi_ranks > std::numeric_limits<int>::max()
            || nonlinear_iterations > std::numeric_limits<int>::max()
            || linear_solves > std::numeric_limits<int>::max()
            || krylov_iterations > std::numeric_limits<int>::max()
            || entry.max_total_seconds == 0.0)
        {
            throw std::runtime_error(
                "Benchmark baseline line " + std::to_string(line_number)
                + " contains an out-of-range limit or dimension.");
        }
        entry.mpi_ranks = static_cast<int>(mpi_ranks);
        entry.max_nonlinear_iterations =
            static_cast<int>(nonlinear_iterations);
        entry.max_linear_solves = static_cast<int>(linear_solves);
        entry.max_krylov_iterations =
            static_cast<int>(krylov_iterations);

        const auto duplicate = std::find_if(
            d_entries.begin(), d_entries.end(),
            [&](const Entry& existing)
            {
                return existing.benchmark_case == entry.benchmark_case
                    && existing.preset == entry.preset
                    && existing.mesh_nx == entry.mesh_nx
                    && existing.mesh_ny == entry.mesh_ny
                    && existing.mesh_nz == entry.mesh_nz
                    && existing.mpi_ranks == entry.mpi_ranks
                    && std::abs(existing.shear - entry.shear) <= 1.0e-12
                    && existing.treatment == entry.treatment
                    && existing.coupling == entry.coupling
                    && existing.preconditioner == entry.preconditioner;
            });
        if (duplicate != d_entries.end())
        {
            throw std::runtime_error(
                "Benchmark baseline contains a duplicate configuration on line "
                + std::to_string(line_number) + ".");
        }
        d_entries.push_back(std::move(entry));
    }

    if (d_entries.empty())
    {
        throw std::runtime_error(
            "Benchmark baseline contains no configurations: "
            + d_baseline_path.string());
    }
}

/**
 * @brief Check one measured record against its matching baseline row.
 */
void RegressionGate::check(const Record& record)
{
    const auto match = std::find_if(
        d_entries.begin(), d_entries.end(),
        [&](const Entry& entry)
        {
            return entry.benchmark_case == record.benchmark_case
                && entry.preset == record.preset
                && entry.mesh_nx == record.mesh_nx
                && entry.mesh_ny == record.mesh_ny
                && entry.mesh_nz == record.mesh_nz
                && entry.mpi_ranks == record.mpi_ranks
                && std::abs(entry.shear - record.shear) <= 1.0e-12
                && entry.treatment == record.treatment
                && entry.coupling == record.coupling
                && entry.preconditioner == record.preconditioner;
        });
    if (match == d_entries.end())
    {
        throw std::runtime_error(
            "Benchmark regression baseline has no entry for "
            + record.benchmark_case + "/" + record.treatment + "/"
            + record.coupling + " at shear=" + std::to_string(record.shear)
            + " with " + std::to_string(record.mpi_ranks) + " MPI rank(s).");
    }
    if (match->observed_samples >= d_expected_samples_per_case)
    {
        throw std::runtime_error(
            "Benchmark regression received too many samples for "
            + record.benchmark_case + "/" + record.treatment + "/"
            + record.coupling + ".");
    }
    if (!record.converged)
    {
        throw std::runtime_error(
            "Benchmark regression solver did not converge for "
            + record.benchmark_case + "/" + record.treatment + "/"
            + record.coupling + ".");
    }
    if (!std::isfinite(record.total_seconds) || record.total_seconds < 0.0)
    {
        throw std::runtime_error(
            "Benchmark regression produced an invalid elapsed time.");
    }

    const auto check_ceiling =
        [&](const auto actual, const auto maximum, std::string_view metric)
    {
        if (actual > maximum)
        {
            std::ostringstream message;
            message << "Benchmark regression exceeded " << metric
                    << " for " << record.benchmark_case << '/'
                    << record.treatment << '/' << record.coupling
                    << ": measured " << actual
                    << ", limit " << maximum << '.';
            throw std::runtime_error(message.str());
        }
    };
    check_ceiling(
        record.nonlinear_iterations,
        match->max_nonlinear_iterations,
        "nonlinear-iteration ceiling");
    check_ceiling(
        record.linear_solves,
        match->max_linear_solves,
        "linear-solve ceiling");
    check_ceiling(
        record.krylov_iterations,
        match->max_krylov_iterations,
        "Krylov-iteration ceiling");
    check_ceiling(
        record.total_seconds,
        match->max_total_seconds,
        "elapsed-time ceiling (seconds)");
    ++match->observed_samples;
}

/**
 * @brief Verify that every baseline configuration produced every sample.
 */
void RegressionGate::verify_complete() const
{
    for (const auto& entry : d_entries)
    {
        if (entry.observed_samples != d_expected_samples_per_case)
        {
            throw std::runtime_error(
                "Benchmark regression expected "
                + std::to_string(d_expected_samples_per_case)
                + " sample(s) for " + entry.benchmark_case + "/"
                + entry.treatment + "/" + entry.coupling
                + " but observed "
                + std::to_string(entry.observed_samples) + ".");
        }
    }
}

} // namespace SimpleFluid::Benchmark
