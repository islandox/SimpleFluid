/** @file VerificationParallel.hh
 * @brief Shared MPI diagnostics and per-rank output for verification drivers.
 */
#pragma once

#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid::Verification
{

class ParallelContext
{
public:
    explicit ParallelContext(Teuchos::RCP<const Teuchos::Comm<int>> communicator)
        : d_comm(std::move(communicator))
    {
        if (d_comm.is_null())
            throw std::invalid_argument("Verification requires a communicator.");
    }
    int rank() const noexcept { return d_comm->getRank(); }
    int size() const noexcept { return d_comm->getSize(); }
    bool is_root() const noexcept { return rank() == 0; }
    template<class T> T sum(T value) const { return reduce(value, Teuchos::REDUCE_SUM); }
    template<class T> T min(T value) const { return reduce(value, Teuchos::REDUCE_MIN); }
    template<class T> T max(T value) const { return reduce(value, Teuchos::REDUCE_MAX); }
    bool all(bool value) const { return min(value ? 1 : 0) != 0; }
    void require(bool value, const std::string& message) const
    {
        if (!all(value))
            throw std::runtime_error(message);
    }
    void barrier() const { d_comm->barrier(); }
    std::filesystem::path output_directory(const std::filesystem::path& root) const
    {
        const auto local = size() == 1 ? root : root / ("rank" + std::to_string(rank()));
        std::filesystem::create_directories(local);
        return local;
    }
    void write_timing(const std::filesystem::path& local_output, double wall, double cpu) const
    {
        std::ofstream stream;
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        stream.open(local_output / "timing.json");
        stream << std::setprecision(17) << "{\"rank\":" << rank() << ",\"ranks\":" << size()
               << ",\"loop_wall_s\":" << wall << ",\"loop_cpu_s\":" << cpu << "}\n";
    }
private:
    template<class T> T reduce(T value, Teuchos::EReductionType operation) const
    {
        T result{};
        Teuchos::reduceAll(*d_comm, operation, 1, &value, &result);
        return result;
    }
    Teuchos::RCP<const Teuchos::Comm<int>> d_comm;
};

/** Loop wall and process CPU time, independent of launcher/setup duration. */
class LoopTimer
{
public:
    double wall_seconds() const
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - d_wall).count();
    }
    double cpu_seconds() const
    {
        return static_cast<double>(std::clock() - d_cpu) / CLOCKS_PER_SEC;
    }
private:
    std::chrono::steady_clock::time_point d_wall = std::chrono::steady_clock::now();
    std::clock_t d_cpu = std::clock();
};
} // namespace SimpleFluid::Verification
