module;

#include "benchmarks/BenchmarkSupport.hh"

export module SimpleFluid.BenchmarkSupport;

#if defined(SIMPLEFLUID_USE_STD_MODULE)
import std;
#endif

export namespace SimpleFluid::Benchmark
{
using ::SimpleFluid::Benchmark::ProcessMemory;
using ::SimpleFluid::Benchmark::sample_process_memory;
using ::SimpleFluid::Benchmark::utc_timestamp;
using ::SimpleFluid::Benchmark::host_name;
using ::SimpleFluid::Benchmark::csv_escape;
using ::SimpleFluid::Benchmark::Record;
using ::SimpleFluid::Benchmark::CsvWriter;
using ::SimpleFluid::Benchmark::RegressionGate;
}
