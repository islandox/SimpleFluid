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

#include "benchmarks/BenchmarkSupport.hh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>

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

TEST(BenchmarkSupportTest, SamplesProcessMemory)
{
    const auto memory =
        SimpleFluid::Benchmark::sample_process_memory();
    EXPECT_GT(memory.resident_kib, 0);
    EXPECT_GT(memory.peak_resident_kib, 0);
}
