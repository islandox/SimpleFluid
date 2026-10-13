#pragma once

#include "FVM/BoundaryCache.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/MeshHandle.hh"
#include <string>
#include <optional>

namespace SimpleFluid
{
/** Typed thermal transaction for solver-integrated planar ALE.
 * Preview must advance from the begin_substep baseline, never from a previous
 * candidate. Flux snapshots are immutable outward liquid heat fluxes [W/m^2]
 * on the current candidate mesh. Residual is a dimensionless, nonnegative
 * global maximum of errors divided by participant tolerances; <=1 converges.
 * All methods must be collective-safe on the mesh communicator.
 * State includes every external thermal inventory and trial ledger. Restore
 * must also restore a live substep baseline when capturing after begin_substep.
 * Participant configuration is immutable while installed in the solver.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class ThermalStepParticipant
{
public:
    using scalar_type = typename Pack::scalar_type;
    using mesh_type = MeshHandle<Pack>;
    using field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_cell_type;
    using boundary_cache_type = FVM::MeshBoundaryCache<Pack, mesh_type>;
    struct State { virtual ~State() = default; };
    virtual ~ThermalStepParticipant() = default;
    virtual std::string configuration_signature() const = 0;
    virtual SP<const State> capture_state() const = 0;
    virtual void restore_state(const State& state) = 0;
    virtual void begin_substep(scalar_type time, scalar_type time_step) = 0;
    virtual SP<const boundary_cache_type> preview_trial(
        SP<const mesh_type> mesh, const field_type& new_time_temperature_guess) = 0;
    virtual scalar_type residual(const field_type& solved_temperature) const = 0;
    /** Cavity owner temperature: accepted while idle, candidate after preview.
     * Required only with HeadspaceTemperatureMode::External.
     */
    virtual std::optional<scalar_type> trial_cavity_temperature() const { return std::nullopt; }
    virtual void validate_commit() const = 0;
    virtual void commit() noexcept = 0;
};
}
