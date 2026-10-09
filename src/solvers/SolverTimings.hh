#pragma once

#include <Teuchos_Time.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace SimpleFluid
{
/** Stable logical flow-chart stages; parent scopes include child work. */
enum class SolverPhase : std::size_t
{
    Step,
    PressureVelocity,
    Momentum,
    PressureProjection,
    PhysicalModels,
    Turbulence,
    PreTemperatureModels,
    Temperature,
    PostTemperatureModels,
    Gas,
    FreeSurface,
    ALE,
    ALETrial,
    ALEGeometry,
    ALERestore,
    GeometryRefresh,
    GasRestore,
    Count
};

struct SolverPhaseTiming
{
    const char* name;
    std::uint64_t calls = 0;
    double seconds = 0;
    unsigned depth = 0; ///< Logical flow-chart depth, not a runtime stack depth.
};

/** Rank-local completed-scope diagnostics. Not physical/checkpoint state.
 * Uses Teuchos::Time; performs no MPI synchronization.
 * Single-threaded, like the owning solver. Snapshot excludes active scopes.
 */
class SolverTimings
{
public:
    SolverTimings() = default;
    SolverTimings(const SolverTimings&) = delete;
    SolverTimings& operator=(const SolverTimings&) = delete;
    SolverTimings(SolverTimings&&) = delete;
    SolverTimings& operator=(SolverTimings&&) = delete;

    using snapshot_type = std::array<SolverPhaseTiming, static_cast<std::size_t>(SolverPhase::Count)>;
    class Scope
    {
    public:
        Scope(SolverTimings& owner, SolverPhase phase)
            : d_owner(owner), d_phase(phase),
              d_timer(owner.d_entries.at(static_cast<std::size_t>(phase)).name, true) { ++d_owner.d_active; }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() noexcept
        {
            auto& entry = d_owner.d_entries[static_cast<std::size_t>(d_phase)];
            entry.seconds += d_timer.stop();
            ++entry.calls;
            --d_owner.d_active;
        }
    private:
        SolverTimings& d_owner;
        SolverPhase d_phase;
        Teuchos::Time d_timer;
    };
    [[nodiscard]] Scope scope(SolverPhase phase) { return Scope(*this, phase); }
    snapshot_type snapshot() const noexcept { return d_entries; }
    SolverPhaseTiming query(SolverPhase phase) const { return d_entries.at(static_cast<std::size_t>(phase)); }
    void reset()
    {
        if (d_active != 0) throw std::logic_error("Cannot reset solver timings with active scopes.");
        for (auto& entry : d_entries) { entry.calls = 0; entry.seconds = 0; }
    }
private:
    snapshot_type d_entries{{
        {"sf.step", 0, 0, 0},
        {"sf.pressure_velocity", 0, 0, 1},
        {"sf.momentum", 0, 0, 2},
        {"sf.pressure_project", 0, 0, 2},
        {"sf.physical_models", 0, 0, 1},
        {"sf.turbulence", 0, 0, 1},
        {"sf.pre_temp_models", 0, 0, 1},
        {"sf.temperature", 0, 0, 1},
        {"sf.post_temp_models", 0, 0, 1},
        {"sf.gas", 0, 0, 2},
        {"sf.free_surface", 0, 0, 1},
        {"sf.ale", 0, 0, 1},
        {"sf.ale_trial", 0, 0, 2},
        {"sf.ale_geometry", 0, 0, 3},
        {"sf.ale_restore", 0, 0, 3},
        {"sf.geometry_refresh", 0, 0, 4},
        {"sf.gas_restore", 0, 0, 4},
    }};
    unsigned d_active = 0;
};
}
