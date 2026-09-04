/**
 * @file VolumeContinuityModel.hh
 * @brief Conservative material-volume and bubble-slip continuity ledger.
 */

#pragma once

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
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = MeshHandle<Pack>> class VolumeContinuityModel
{
public:
    using mesh_type = MeshType;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_cell_type;
    using face_flux_field_type = typename MeshFieldTraits<Pack, mesh_type>::scalar_face_type;
    using target_type = VolumeContinuityTarget<Pack, mesh_type>;
    using diagnostics_type = VolumeSourceDiagnostics<scalar_type>;

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
    [[nodiscard]] Trial preview(const Inputs& inputs, std::uint64_t generation) const
    {
        inputs.ale.validate(*d_mesh);
        collective_detail::collective_local_validation(*d_mesh, "VolumeContinuityModel input validation",
            [&]
            {
                if (inputs.carrier_relative_flux.mesh_ptr().get() != d_mesh.get() ||
                    (inputs.carrier_material_volume_flux != nullptr &&
                        inputs.carrier_material_volume_flux->mesh_ptr().get() != d_mesh.get()) ||
                    (inputs.bubble_slip_volume_flux != nullptr &&
                        inputs.bubble_slip_volume_flux->mesh_ptr().get() != d_mesh.get()))
                {
                    throw std::invalid_argument("VolumeContinuityModel flux fields must use its exact mesh handle.");
                }
                if (inputs.old_material_volume.size() != d_mesh->num_local_cells() ||
                    inputs.new_material_volume.size() != d_mesh->num_local_cells() ||
                    (!inputs.carrier_material_fraction.empty() &&
                        inputs.carrier_material_fraction.size() != d_mesh->num_local_cells()) ||
                    (!inputs.previous_target.empty() && inputs.previous_target.size() != d_mesh->num_owned_cells()))
                {
                    throw std::invalid_argument("VolumeContinuityModel material volumes must use local-cell order and "
                                                "previous targets owned order.");
                }
                const scalar_type scalar_inputs[]{inputs.old_pool_volume, inputs.new_pool_volume,
                    inputs.bubble_escape_volume_rate, inputs.other_outflow_volume_rate};
                for (const auto value : scalar_inputs)
                {
                    if (!std::isfinite(value) || value < scalar_type{})
                    {
                        throw std::invalid_argument(
                            "VolumeContinuityModel requires finite non-negative pool volumes and outflow rates.");
                    }
                }
            });

        collective_detail::require_uniform_value(*d_mesh, static_cast<int>(!inputs.carrier_material_fraction.empty()),
            "VolumeContinuityModel carrier-material-fraction selection");
        collective_detail::require_uniform_value(*d_mesh,
            static_cast<int>(inputs.carrier_material_volume_flux != nullptr),
            "VolumeContinuityModel exact carrier-flux selection");
        collective_detail::require_uniform_value(*d_mesh, static_cast<int>(inputs.bubble_slip_volume_flux != nullptr),
            "VolumeContinuityModel bubble-slip-flux selection");
        collective_detail::require_uniform_value(*d_mesh, static_cast<int>(!inputs.previous_target.empty()),
            "VolumeContinuityModel previous-target selection");
        collective_detail::require_uniform_value(
            *d_mesh, inputs.old_pool_volume, "VolumeContinuityModel old pool volume");
        collective_detail::require_uniform_value(
            *d_mesh, inputs.new_pool_volume, "VolumeContinuityModel new pool volume");
        collective_detail::require_uniform_value(
            *d_mesh, inputs.bubble_escape_volume_rate, "VolumeContinuityModel bubble escape volume rate");
        collective_detail::require_uniform_value(
            *d_mesh, inputs.other_outflow_volume_rate, "VolumeContinuityModel other outflow volume rate");
        collective_detail::require_uniform_value(
            *d_mesh, d_volume_absolute_tolerance, "VolumeContinuityModel absolute volume-closure tolerance");
        collective_detail::require_uniform_value(
            *d_mesh, d_relative_tolerance, "VolumeContinuityModel relative closure tolerance");
        collective_detail::require_uniform_value(
            *d_mesh, static_cast<unsigned long long>(generation), "VolumeContinuityModel target generation");
        const auto old_volumes = inputs.ale.old_cell_volumes();
        const auto new_volumes = inputs.ale.new_cell_volumes();
        const auto time_step = static_cast<scalar_type>(inputs.ale.time_step());
        std::vector<scalar_type> material_source(d_mesh->num_owned_cells());
        std::vector<scalar_type> slip_contribution(d_mesh->num_owned_cells());
        std::vector<scalar_type> target_values(d_mesh->num_owned_cells());

        int local_invalid = 0;
        scalar_type local_old_material{};
        scalar_type local_new_material{};
        scalar_type local_material_source{};
        scalar_type local_carrier_transport{};
        scalar_type local_slip_divergence{};
        scalar_type local_maximum_target_change{};
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto old_material = inputs.old_material_volume[owned];
            const auto new_material = inputs.new_material_volume[owned];
            local_invalid = local_invalid || !std::isfinite(old_material) || old_material < scalar_type{} ||
                            !std::isfinite(new_material) || new_material < scalar_type{};
            scalar_type carrier_divergence{};
            scalar_type slip_divergence{};
            for (const auto face_lid : d_mesh->faces(cell_lid))
            {
                const auto owner_flux = inputs.carrier_relative_flux.local_value(face_lid);
                const auto outward_flux = d_mesh->owner_cell(face_lid) == cell_lid ? owner_flux : -owner_flux;
                if (inputs.carrier_material_volume_flux != nullptr)
                {
                    const auto owner_material_flux = inputs.carrier_material_volume_flux->local_value(face_lid);
                    carrier_divergence +=
                        d_mesh->owner_cell(face_lid) == cell_lid ? owner_material_flux : -owner_material_flux;
                }
                else
                {
                    auto upwind = cell_lid;
                    if (outward_flux < scalar_type{} && d_mesh->is_interior_face(face_lid))
                    {
                        upwind = d_mesh->opposite_or_periodic_neighbor_cell(face_lid, cell_lid);
                    }
                    // The supported transports are backward-Euler implicit
                    // in their advected state. Remove that same trial-new
                    // carrier transport here; accepted-old W/V fabricates a
                    // source whenever a nonuniform inventory advects.
                    const auto concentration =
                        inputs.carrier_material_fraction.empty()
                            ? inputs.new_material_volume[static_cast<size_t>(upwind)] /
                                  static_cast<scalar_type>(new_volumes[static_cast<size_t>(upwind)])
                            : inputs.carrier_material_fraction[static_cast<size_t>(upwind)];
                    carrier_divergence += outward_flux * concentration;
                }
                if (inputs.bubble_slip_volume_flux != nullptr)
                {
                    const auto owner_slip = inputs.bubble_slip_volume_flux->local_value(face_lid);
                    slip_divergence += d_mesh->owner_cell(face_lid) == cell_lid ? owner_slip : -owner_slip;
                }
            }
            const auto volume_rate = (new_material - old_material) / time_step;
            material_source[owned] = volume_rate + carrier_divergence + slip_divergence;
            slip_contribution[owned] = -slip_divergence;
            target_values[owned] = material_source[owned] + slip_contribution[owned];
            local_invalid =
                local_invalid || !std::isfinite(material_source[owned]) || !std::isfinite(target_values[owned]);
            local_old_material += old_material;
            local_new_material += new_material;
            local_material_source += material_source[owned];
            local_carrier_transport += carrier_divergence;
            local_slip_divergence += slip_divergence;
            if (!inputs.previous_target.empty())
            {
                local_maximum_target_change = std::max(
                    local_maximum_target_change, std::abs(target_values[owned] - inputs.previous_target[owned]));
            }
        }
        for (size_t local = 0; local < d_mesh->num_local_cells(); ++local)
        {
            const auto carrier_fraction =
                inputs.carrier_material_fraction.empty()
                    ? inputs.new_material_volume[local] / static_cast<scalar_type>(new_volumes[local])
                    : inputs.carrier_material_fraction[local];
            local_invalid = local_invalid || !std::isfinite(carrier_fraction) || carrier_fraction < scalar_type{};
        }

        int any_invalid = 0;
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
        if (any_invalid != 0)
        {
            throw std::invalid_argument(
                "VolumeContinuityModel requires finite non-negative volumes and finite flux/source values.");
        }

        const scalar_type local_totals[]{local_old_material, local_new_material, local_material_source,
            local_carrier_transport, local_slip_divergence};
        scalar_type global_totals[5]{};
        scalar_type maximum_target_change{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 5, local_totals, global_totals);
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_maximum_target_change,
            &maximum_target_change);

        diagnostics_type diagnostics;
        diagnostics.old_material_volume = global_totals[0];
        diagnostics.new_material_volume = global_totals[1];
        diagnostics.global_material_source = global_totals[2];
        diagnostics.global_carrier_transport = global_totals[3];
        diagnostics.global_bubble_slip_divergence = global_totals[4];
        diagnostics.bubble_escape_volume_rate = inputs.bubble_escape_volume_rate;
        diagnostics.other_outflow_volume_rate = inputs.other_outflow_volume_rate;
        diagnostics.source_pool_closure_residual =
            diagnostics.global_material_source -
            ((inputs.new_pool_volume - inputs.old_pool_volume) / time_step + inputs.bubble_escape_volume_rate +
                inputs.other_outflow_volume_rate);
        const auto scale = std::max({scalar_type{1}, std::abs(diagnostics.global_material_source),
            std::abs((inputs.new_pool_volume - inputs.old_pool_volume) / time_step),
            std::abs(inputs.bubble_escape_volume_rate), std::abs(inputs.other_outflow_volume_rate)});
        diagnostics.normalized_source_pool_closure_residual = diagnostics.source_pool_closure_residual / scale;
        diagnostics.maximum_target_change = maximum_target_change;
        const auto tolerance = d_volume_absolute_tolerance / time_step + d_relative_tolerance * scale;
        if (std::abs(diagnostics.source_pool_closure_residual) > tolerance)
        {
            std::ostringstream message;
            message << std::scientific << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                    << "Volume-source to pool-volume closure exceeded its physical tolerance: residual="
                    << diagnostics.source_pool_closure_residual << " m^3/s, tolerance=" << tolerance
                    << " m^3/s, old material=" << diagnostics.old_material_volume
                    << " m^3, new material=" << diagnostics.new_material_volume
                    << " m^3, carrier transport=" << diagnostics.global_carrier_transport
                    << " m^3/s, bubble slip divergence=" << diagnostics.global_bubble_slip_divergence
                    << " m^3/s, old pool=" << inputs.old_pool_volume << " m^3, new pool=" << inputs.new_pool_volume
                    << " m^3.";
            throw std::runtime_error(message.str());
        }

        auto target = target_type(d_mesh, std::move(target_values), generation);
        target.validate(*d_mesh, "Volume continuity model target");
        const auto trial_nonce = invalidated_counter(d_trial_nonce, d_trial_nonce, "trial nonce");
        d_trial_nonce = trial_nonce;
        return Trial(this, trial_nonce, std::move(target), std::move(material_source), std::move(slip_contribution),
            diagnostics);
    }

    void commit(const Trial& trial)
    {
        const int local_foreign = trial.d_owner != this ? 1 : 0;
        int any_foreign = 0;
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_foreign, &any_foreign);
        if (any_foreign != 0)
        {
            throw std::invalid_argument("VolumeContinuityModel trial belongs to another model.");
        }
        const int local_stale = trial.d_nonce != d_trial_nonce ? 1 : 0;
        int any_stale = 0;
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_stale, &any_stale);
        if (any_stale != 0)
        {
            throw std::logic_error("VolumeContinuityModel trial is stale or already committed.");
        }
        trial.d_target.validate(*d_mesh, "Volume continuity model trial");
        const auto next_trial_nonce = invalidated_counter(d_trial_nonce, d_trial_nonce, "trial nonce");
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            d_material_source.set_owned_value(cell, trial.d_material_source[owned]);
            d_slip_contribution.set_owned_value(cell, trial.d_slip_contribution[owned]);
            d_continuity_target.set_owned_value(cell, trial.d_target.integrated_rate(cell));
        }
        d_material_source.sync_ghosts();
        d_slip_contribution.sync_ghosts();
        d_continuity_target.sync_ghosts();
        d_diagnostics = trial.d_diagnostics;
        d_generation = trial.d_target.generation();
        d_trial_nonce = next_trial_nonce;
    }

    /** Capture every mutable accepted field, diagnostic, and generation. */
    [[nodiscard]] StateSnapshot snapshot() const
    {
        StateSnapshot result;
        result.d_owner = this;
        result.d_material_source.resize(d_mesh->num_owned_cells());
        result.d_slip_contribution.resize(d_mesh->num_owned_cells());
        result.d_continuity_target.resize(d_mesh->num_owned_cells());
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            result.d_material_source[owned] = d_material_source.value(cell);
            result.d_slip_contribution[owned] = d_slip_contribution.value(cell);
            result.d_continuity_target[owned] = d_continuity_target.value(cell);
        }
        result.d_diagnostics = d_diagnostics;
        result.d_generation = d_generation;
        result.d_trial_nonce = d_trial_nonce;
        return result;
    }

    /**
     * Restore accepted ledger state while invalidating trials from both sides
     * of the rollback boundary.
     */
    void restore(const StateSnapshot& snapshot)
    {
        const int local_foreign = snapshot.d_owner != this ? 1 : 0;
        int any_foreign = 0;
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_foreign, &any_foreign);
        if (any_foreign != 0)
        {
            throw std::invalid_argument("VolumeContinuityModel snapshot belongs to another model.");
        }
        const int local_incompatible = snapshot.d_material_source.size() != d_mesh->num_owned_cells() ||
                                               snapshot.d_slip_contribution.size() != d_mesh->num_owned_cells() ||
                                               snapshot.d_continuity_target.size() != d_mesh->num_owned_cells()
                                           ? 1
                                           : 0;
        int any_incompatible = 0;
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_incompatible, &any_incompatible);
        if (any_incompatible != 0)
        {
            throw std::invalid_argument("VolumeContinuityModel snapshot has incompatible field "
                                        "storage.");
        }

        const auto saved_generation = static_cast<unsigned long long>(snapshot.d_generation);
        const auto saved_nonce = static_cast<unsigned long long>(snapshot.d_trial_nonce);
        unsigned long long minimum_generation = 0;
        unsigned long long maximum_generation = 0;
        unsigned long long minimum_nonce = 0;
        unsigned long long maximum_nonce = 0;
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &saved_generation, &minimum_generation);
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &saved_generation, &maximum_generation);
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &saved_nonce, &minimum_nonce);
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &saved_nonce, &maximum_nonce);
        if (minimum_generation != maximum_generation || minimum_nonce != maximum_nonce)
        {
            throw std::invalid_argument("VolumeContinuityModel snapshot generation must agree on "
                                        "every rank.");
        }

        const auto next_trial_nonce = invalidated_counter(d_trial_nonce, snapshot.d_trial_nonce, "trial nonce");
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            d_material_source.set_owned_value(cell, snapshot.d_material_source[owned]);
            d_slip_contribution.set_owned_value(cell, snapshot.d_slip_contribution[owned]);
            d_continuity_target.set_owned_value(cell, snapshot.d_continuity_target[owned]);
        }
        d_material_source.sync_ghosts();
        d_slip_contribution.sync_ghosts();
        d_continuity_target.sync_ghosts();
        d_diagnostics = snapshot.d_diagnostics;
        d_generation = snapshot.d_generation;
        d_trial_nonce = next_trial_nonce;
    }

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

} // namespace SimpleFluid
