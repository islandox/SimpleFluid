/**
 * @file VolumeContinuityModel.hh
 * @brief Conservative material-volume and bubble-slip continuity ledger.
 */

#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/ALEControlVolumeState.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/VolumeContinuityTarget.hh"
#include "fields/FieldStored.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/** @brief Global balance diagnostics for one previewed material-volume ledger step.
 *
 * Volume entries are in m^3 and rates/residuals are in m^3/s. These values
 * report the source, carrier transport, bubble slip, and pool closure used to
 * build a candidate continuity target.
 */
template<class Scalar> struct VolumeSourceDiagnostics
{
    Scalar old_material_volume = {};           ///< [m^3]
    Scalar new_material_volume = {};           ///< [m^3]
    Scalar global_material_source = {};        ///< [m^3/s]
    Scalar global_carrier_transport = {};      ///< Relative boundary transport [m^3/s]
    Scalar global_bubble_slip_divergence = {}; ///< [m^3/s]
    Scalar bubble_escape_volume_rate = {};     ///< [m^3/s]
    Scalar other_outflow_volume_rate = {};     ///< [m^3/s]
    Scalar source_pool_closure_residual = {};  ///< [m^3/s]
    Scalar normalized_source_pool_closure_residual = {};
    Scalar maximum_target_change = {}; ///< [m^3/s]
};

/**
 * @brief Builds the liquid-carrier target from extensive material-volume change.
 *
 * For each cell the ledger evaluates
 *
 * `Q_material = (W_new-W_old)/dt + div(Phi_carrier,material) + div(Phi_b,slip)`
 *
 * where `W` is authoritative liquid plus raw submerged-bubble material volume
 * and `Phi_carrier,material` is the exact implicit transport flux when the
 * owning equations publish it. The fallback is `phi_rel*C_new`, using the
 * trial post-transport/pre-source fraction so local kinetics is not advected
 * retroactively. The pressure target is then
 * `Q_l = Q_material - div(phi_b,slip)`. Internal face contributions cancel
 * pairwise; moving-surface escape remains an explicit outward boundary term.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = MeshHandle<Pack>> class SIMPLEFLUID_SOLVERS_EXPORT VolumeContinuityModel
{
public:
    using mesh_type = MeshType;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_cell_type;
    using face_flux_field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_face_type;
    using target_type = VolumeContinuityTarget<Pack, mesh_type>;
    using diagnostics_type = VolumeSourceDiagnostics<scalar_type>;

    /** @brief Borrowed step data consumed collectively by a non-mutating preview.
     *
     * Material-volume and carrier-fraction spans follow local cell order,
     * including ghost cells; `previous_target` alone follows owned-cell order.
     * Optional exact carrier and bubble-slip fluxes override their documented
     * fallbacks, and the ALE state supplies current and trial geometry.
     */
    struct Inputs
    {
        const FVM::ALEControlVolumeState& ale;
        /** Extensive liquid plus raw bubble volume in local-cell order [m^3]. */
        std::span<const scalar_type> old_material_volume;
        /** Extensive liquid plus raw bubble volume in local-cell order [m^3]. */
        std::span<const scalar_type> new_material_volume;
        /**
         * Trial-new material fraction used by the implicit carrier flux.
         * Empty derives W_new/V_new; operator-split models should provide the
         * post-transport/pre-source fraction explicitly.
         */
        std::span<const scalar_type> carrier_material_fraction;
        const face_flux_field_type& carrier_relative_flux; ///< [m^3/s]
        /** Exact implicit carrier material-volume flux when available [m^3/s]. */
        const face_flux_field_type* carrier_material_volume_flux = nullptr;
        /** Bubble material-volume slip flux, owner-oriented [m^3/s]. */
        const face_flux_field_type* bubble_slip_volume_flux = nullptr;
        scalar_type old_pool_volume = {};             ///< [m^3]
        scalar_type new_pool_volume = {};             ///< [m^3]
        scalar_type bubble_escape_volume_rate = {};   ///< [m^3/s]
        scalar_type other_outflow_volume_rate = {};   ///< [m^3/s]
        std::span<const scalar_type> previous_target; ///< Optional owned order [m^3/s]
    };

    /** @brief Candidate target and diagnostics that can be committed once by this model.
     *
     * A later preview or snapshot restore invalidates this trial. Its exposed
     * fields are read-only views of the candidate and do not mutate accepted
     * ledger state.
     */
    class Trial
    {
    public:
        const target_type& target() const noexcept { return d_target; }
        const diagnostics_type& diagnostics() const noexcept { return d_diagnostics; }
        std::span<const scalar_type> material_source() const noexcept { return d_material_source; }
        std::span<const scalar_type> bubble_slip_contribution() const noexcept { return d_slip_contribution; }

    private:
        friend class VolumeContinuityModel;
        Trial(const VolumeContinuityModel* owner, std::uint64_t nonce, target_type target,
            std::vector<scalar_type> material_source, std::vector<scalar_type> slip_contribution,
            diagnostics_type diagnostics) noexcept
            : d_owner(owner), d_nonce(nonce), d_target(std::move(target)),
              d_material_source(std::move(material_source)), d_slip_contribution(std::move(slip_contribution)),
              d_diagnostics(diagnostics)
        {
        }

        const VolumeContinuityModel* d_owner = nullptr;
        std::uint64_t d_nonce = 0;
        target_type d_target;
        std::vector<scalar_type> d_material_source;
        std::vector<scalar_type> d_slip_contribution;
        diagnostics_type d_diagnostics;
    };

    /** Opaque copy of every mutable accepted ledger value. */
    class StateSnapshot
    {
    public:
        StateSnapshot(StateSnapshot&&) noexcept = default;
        StateSnapshot& operator=(StateSnapshot&&) noexcept = default;
        StateSnapshot(const StateSnapshot&) = default;
        StateSnapshot& operator=(const StateSnapshot&) = default;
        ~StateSnapshot() = default;

    private:
        friend class VolumeContinuityModel;
        StateSnapshot() = default;

        const VolumeContinuityModel* d_owner = nullptr;
        std::vector<scalar_type> d_material_source;
        std::vector<scalar_type> d_slip_contribution;
        std::vector<scalar_type> d_continuity_target;
        diagnostics_type d_diagnostics;
        std::uint64_t d_generation = 0;
        std::uint64_t d_trial_nonce = 0;
    };

    VolumeContinuityModel(SP<const mesh_type> mesh, scalar_type volume_absolute_tolerance = 1.0e-12,
        scalar_type relative_tolerance = 1.0e-10)
        : d_mesh(std::move(mesh)), d_material_source(d_mesh, "volumeSourceRate"),
          d_slip_contribution(d_mesh, "bubbleSlipVolumeRate"), d_continuity_target(d_mesh, "continuityTarget"),
          d_volume_absolute_tolerance(volume_absolute_tolerance), d_relative_tolerance(relative_tolerance)
    {
        if (!d_mesh || !std::isfinite(d_volume_absolute_tolerance) || d_volume_absolute_tolerance < scalar_type{} ||
            !std::isfinite(d_relative_tolerance) || d_relative_tolerance < scalar_type{})
        {
            throw std::invalid_argument("VolumeContinuityModel requires a mesh, a finite non-negative absolute volume "
                                        "tolerance, and a finite non-negative relative tolerance.");
        }
    }

    /**
     * @brief Build a non-mutating ledger candidate and invalidate older trials.
     *
     * The public target generation identifies pressure-coupling data. The
     * private trial nonce instead identifies the one preview that may commit;
     * it advances after each successful preview without changing accepted
     * fields or diagnostics.
     */
    [[nodiscard]] Trial preview(const Inputs& inputs, std::uint64_t generation) const;

    void commit(const Trial& trial);

    /** Capture every mutable accepted field, diagnostic, and generation. */
    [[nodiscard]] StateSnapshot snapshot() const;

    /**
     * Restore accepted ledger state while invalidating trials from both sides
     * of the rollback boundary.
     */
    void restore(const StateSnapshot& snapshot);

    const field_type& material_source_field() const noexcept { return d_material_source; }
    const field_type& slip_contribution_field() const noexcept { return d_slip_contribution; }
    const field_type& continuity_target_field() const noexcept { return d_continuity_target; }
    const diagnostics_type& diagnostics() const noexcept { return d_diagnostics; }
    std::uint64_t generation() const noexcept { return d_generation; }

private:
    [[nodiscard]] static std::uint64_t invalidated_counter(std::uint64_t current, std::uint64_t saved, const char* name)
    {
        const auto newest = std::max(current, saved);
        if (newest == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error("VolumeContinuityModel " + std::string(name) + " exhausted.");
        }
        return newest + 1;
    }

    SP<const mesh_type> d_mesh;
    field_type d_material_source;
    field_type d_slip_contribution;
    field_type d_continuity_target;
    scalar_type d_volume_absolute_tolerance = {}; ///< [m^3]
    scalar_type d_relative_tolerance = {};
    diagnostics_type d_diagnostics;
    std::uint64_t d_generation = 0;
    mutable std::uint64_t d_trial_nonce = 0;
};
extern template class VolumeContinuityModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;


} // namespace SimpleFluid
