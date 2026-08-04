/**
 * @file testBenchmarkSupport.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Unit tests for benchmark support utilities: CSV, memory, and escaping.
 * @version 0.1
 * @date 2026-06-12
 *
 * @copyright Copyright (c) 2026
 *
 */
#include <gtest/gtest.h>
#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#else
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#endif

#if defined(SIMPLEFLUID_USE_CXX_MODULES)
import SimpleFluid.BenchmarkSupport;
#else
#include "benchmarks/BenchmarkSupport.hh"
#endif

namespace
{

std::filesystem::path write_regression_baseline(
    std::string_view filename)
{
    const auto path =
        std::filesystem::temp_directory_path() / filename;
    std::ofstream output(path);
    output << SimpleFluid::Benchmark::RegressionGate::header() << '\n'
           << "diffusion_nonorthogonal,debug-small,6,6,6,1,0.4,"
              "implicit,none,none,0,1,40,10\n";
    return path;
}

SimpleFluid::Benchmark::Record regression_record()
{
    SimpleFluid::Benchmark::Record record;
    record.benchmark_case = "diffusion_nonorthogonal";
    record.preset = "debug-small";
    record.mesh_nx = 6;
    record.mesh_ny = 6;
    record.mesh_nz = 6;
    record.mpi_ranks = 1;
    record.shear = 0.4;
    record.treatment = "implicit";
    record.coupling = "none";
    record.preconditioner = "none";
    record.linear_solves = 1;
    record.krylov_iterations = 31;
    record.total_seconds = 0.1;
    record.converged = true;
    return record;
}

} // namespace

/** @brief Verifies CSV escaping preserves plain text and quotes special content. */
TEST(BenchmarkSupportTest, EscapesCsvText)
{
    EXPECT_EQ(SimpleFluid::Benchmark::csv_escape("plain"), "plain");
    EXPECT_EQ(SimpleFluid::Benchmark::csv_escape("a,b"), "\"a,b\"");
    EXPECT_EQ(SimpleFluid::Benchmark::csv_escape("a\"b"), "\"a\"\"b\"");
}

/**
 * @brief Verifies CsvWriter appends compatible records and the output has the correct row count and escaped values.
 */
TEST(BenchmarkSupportTest, WritesAndAppendsCompatibleCsv)
{
    const auto path =
        std::filesystem::temp_directory_path()
        / "simplefluid_benchmark_support.csv";
    std::filesystem::remove(path);

    SimpleFluid::Benchmark::Record record;
    record.timestamp = "2026-06-12T00:00:00Z";
    record.run_id = "test,run";
    record.converged = true;

    SimpleFluid::Benchmark::CsvWriter(path).append(record);
    SimpleFluid::Benchmark::CsvWriter(path).append(record);

    std::ifstream input(path);
    const std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    EXPECT_EQ(
        static_cast<size_t>(
            std::count(contents.begin(), contents.end(), '\n')),
        3U);
    EXPECT_NE(contents.find("\"test,run\""), std::string::npos);
    std::filesystem::remove(path);
}

/** @brief Verify an existing CSV with an incompatible schema is rejected. */
TEST(BenchmarkSupportTest, RejectsIncompatibleCsvHeader)
{
    const auto path =
        std::filesystem::temp_directory_path()
        / "simplefluid_benchmark_bad_header.csv";
    {
        std::ofstream output(path);
        output << "old,header\n";
    }

    EXPECT_THROW(
        (void)SimpleFluid::Benchmark::CsvWriter{path},
        std::runtime_error);
    std::filesystem::remove(path);
}

/** @brief Verifies process memory sampling reports resident and peak usage. */
TEST(BenchmarkSupportTest, SamplesProcessMemory)
{
    const auto memory =
        SimpleFluid::Benchmark::sample_process_memory();
    EXPECT_GT(memory.resident_kib, 0);
    EXPECT_GT(memory.peak_resident_kib, 0);
}

/** @brief Verifies all repeated samples are checked and accounted for. */
TEST(BenchmarkSupportTest, RegressionGateAcceptsCompleteRepeatedSamples)
{
    const auto path = write_regression_baseline(
        "simplefluid_benchmark_regression_complete.csv");
    SimpleFluid::Benchmark::RegressionGate gate(path, 2);
    const auto record = regression_record();

    EXPECT_NO_THROW(gate.check(record));
    EXPECT_NO_THROW(gate.check(record));
    EXPECT_NO_THROW(gate.verify_complete());
    std::filesystem::remove(path);
}

/** @brief Verifies solver-work regressions fail the baseline gate. */
TEST(BenchmarkSupportTest, RegressionGateRejectsExceededCeiling)
{
    const auto path = write_regression_baseline(
        "simplefluid_benchmark_regression_ceiling.csv");
    SimpleFluid::Benchmark::RegressionGate gate(path, 1);
    auto record = regression_record();
    record.krylov_iterations = 41;

    EXPECT_THROW(gate.check(record), std::runtime_error);
    std::filesystem::remove(path);
}

/** @brief Verifies a measured configuration must exist in the baseline. */
TEST(BenchmarkSupportTest, RegressionGateRejectsMissingConfiguration)
{
    const auto path = write_regression_baseline(
        "simplefluid_benchmark_regression_configuration.csv");
    SimpleFluid::Benchmark::RegressionGate gate(path, 1);
    auto record = regression_record();
    record.treatment = "explicit";

    EXPECT_THROW(gate.check(record), std::runtime_error);
    std::filesystem::remove(path);
}

/** @brief Verifies missing measured configurations cannot silently pass. */
TEST(BenchmarkSupportTest, RegressionGateRejectsMissingSamples)
{
    const auto path = write_regression_baseline(
        "simplefluid_benchmark_regression_missing.csv");
    SimpleFluid::Benchmark::RegressionGate gate(path, 2);
    gate.check(regression_record());

    EXPECT_THROW(gate.verify_complete(), std::runtime_error);
    std::filesystem::remove(path);
}

/** @brief Verifies duplicate baseline keys are rejected as ambiguous. */
TEST(BenchmarkSupportTest, RegressionGateRejectsDuplicateConfiguration)
{
    const auto path =
        std::filesystem::temp_directory_path()
        / "simplefluid_benchmark_regression_duplicate.csv";
    std::ofstream output(path);
    output << SimpleFluid::Benchmark::RegressionGate::header() << '\n'
           << "diffusion_nonorthogonal,debug-small,6,6,6,1,0.4,"
              "implicit,none,none,0,1,40,10\n"
           << "diffusion_nonorthogonal,debug-small,6,6,6,1,0.4,"
              "implicit,none,none,0,1,50,20\n";
    output.close();

    EXPECT_THROW(
        (SimpleFluid::Benchmark::RegressionGate(path, 1)),
        std::runtime_error);
    std::filesystem::remove(path);
}
