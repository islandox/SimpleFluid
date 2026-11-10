/**
 * @file FVM/details/DiagnosticCapture.hh
 * @brief Centralised opt-in configuration for the isolation capture families.
 *
 * Every capture writer in the FVM and solver layers asks this header whether
 * its family is enabled and where to write. The environment is read once, and
 * the per-family predicates below are the only gates the writers consult.
 * Nothing is written when the predicates are false.
 */
#pragma once

#include "SimpleFluidExport.hh"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace SimpleFluid::diagnostics
{
/** Environment names, kept here so no caller repeats a literal. */
inline constexpr std::string_view transpose_stress_prefix_env = "SIMPLEFLUID_TRANSPOSE_STRESS_DUMP";
inline constexpr std::string_view pressure_gradient_prefix_env = "SIMPLEFLUID_GRADIENT_CAPTURE";
inline constexpr std::string_view pressure_gradient_gids_env = "SIMPLEFLUID_GRADIENT_CAPTURE_GIDS";

/** One immutable snapshot of the capture switches, taken from the environment once. */
struct CaptureConfiguration
{
    std::string transpose_stress_prefix;
    std::string pressure_gradient_prefix;
    std::vector<std::uint64_t> pressure_gradient_gids;

    bool transpose_stress_enabled() const noexcept { return !transpose_stress_prefix.empty(); }

    bool pressure_gradient_eval_enabled() const noexcept
    {
        return !pressure_gradient_prefix.empty() && !pressure_gradient_gids.empty();
    }

    bool targets_pressure_gradient_cell(std::uint64_t gid) const noexcept
    {
        for (const auto target : pressure_gradient_gids)
            if (target == gid) return true;
        return false;
    }
};

namespace capture_detail
{
inline std::string read_prefix(std::string_view name)
{
    const char* value = std::getenv(std::string(name).c_str());
    return value == nullptr ? std::string{} : std::string(value);
}

inline std::vector<std::uint64_t> read_gids(std::string_view name)
{
    std::vector<std::uint64_t> gids;
    const auto text = read_prefix(name);
    std::size_t begin = 0;
    while (begin < text.size())
    {
        const auto end = text.find(',', begin);
        const auto item = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!item.empty()) gids.push_back(std::stoull(item));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return gids;
}

inline CaptureConfiguration load()
{
    CaptureConfiguration configuration;
    configuration.transpose_stress_prefix = read_prefix(transpose_stress_prefix_env);
    configuration.pressure_gradient_prefix = read_prefix(pressure_gradient_prefix_env);
    configuration.pressure_gradient_gids = read_gids(pressure_gradient_gids_env);
    return configuration;
}
}

/** The single named source of every capture switch. */
inline const CaptureConfiguration& capture_configuration()
{
    static const CaptureConfiguration configuration = capture_detail::load();
    return configuration;
}

/**
 * Depth counter for the in-process pressure-gradient build-capture window.
 * Defined in the FVM library so the solver and FVM shared objects share one
 * instance (the project builds with hidden visibility).
 */
SIMPLEFLUID_FVM_EXPORT std::atomic<int>& pressure_gradient_build_depth() noexcept;

/** Single gate for the transpose-stress and pre-stress source captures. */
inline bool transpose_stress_capture_enabled() noexcept
{
    return capture_configuration().transpose_stress_enabled();
}

/** Single gate for the frozen-operand pressure-gradient evaluation capture. */
inline bool pressure_gradient_eval_capture_enabled() noexcept
{
    return capture_configuration().pressure_gradient_eval_enabled();
}

/** Single gate for the cached weight-construction capture. */
inline bool pressure_gradient_build_capture_enabled() noexcept
{
    return capture_configuration().pressure_gradient_eval_enabled()
        && pressure_gradient_build_depth().load(std::memory_order_relaxed) > 0;
}

/** Explicit arming, for callers whose window is wider than one C++ scope. */
inline void arm_pressure_gradient_build_capture() noexcept
{
    pressure_gradient_build_depth().fetch_add(1, std::memory_order_relaxed);
}

inline void disarm_pressure_gradient_build_capture() noexcept
{
    pressure_gradient_build_depth().fetch_sub(1, std::memory_order_relaxed);
}

} // namespace SimpleFluid::diagnostics
