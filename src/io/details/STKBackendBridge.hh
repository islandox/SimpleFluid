/** @file STKBackendBridge.hh
 * @brief Private I/O access across the single static-Trilinos owner boundary.
 */
#pragma once
#include <stdexcept>
#include <string>

#if defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_STK_IO_BRIDGE_EXPORT __attribute__((visibility("default")))
#else
#define SIMPLEFLUID_STK_IO_BRIDGE_EXPORT
#endif
extern "C" SIMPLEFLUID_STK_IO_BRIDGE_EXPORT void simplefluid_trilinos_stk_io_set_string_property(
    void* broker, const char* name, const char* value);

namespace SimpleFluid::detail
{
/** The standard Ioss basename.N.rank convention, without an Ioss symbol dependency. */
inline std::string stk_decomposed_filename(const std::string& filename, int rank, int ranks)
{
    if (rank < 0 || ranks <= 0 || rank >= ranks)
        throw std::invalid_argument("Invalid STK input file rank.");
    if (ranks == 1)
        return filename;
    const auto count = std::to_string(ranks);
    const auto process = std::to_string(rank);
    return filename + "." + count + "." + std::string(count.size() - process.size(), '0') + process;
}
} // namespace SimpleFluid::detail
