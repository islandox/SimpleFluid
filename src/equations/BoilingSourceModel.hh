/**
 * @file BoilingSourceModel.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Explicit bulk and wall boiling source model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "dataclass/Database.hh"
#include "dataclass/DatabaseOptionReader.hh"
#include "equations/BoussinesqModel.hh"
#include "equations/ScalarVoidFractionModel.hh"
#include "fields/MeshFieldTraits.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Globally reduced boiling mass and scalar-void balance for one step.
 *
 * Mass quantities have units kg and volume quantities have units m^3.  The
 * accepted and condensed values are the conservative inputs for a liquid-mass
 * inventory; rejected vapor is diagnostic demand that caused neither phase
 * change nor latent-energy removal.
 *
 * @tparam Scalar Floating-point scalar type used by the solver fields.
 */
template<class Scalar> struct BoilingPhaseChangeDiagnostics
{
    Scalar requested_evaporation_mass = {};
    Scalar accepted_evaporation_mass = {};
    Scalar rejected_vapor_mass = {};
    Scalar rejected_void_volume = {};
    Scalar scalar_void_collapse_volume = {};
    Scalar nonsteam_collapse_volume = {};
    Scalar condensed_liquid_mass = {};
    Scalar condensation_latent_energy_release = {};
    Scalar submerged_steam_mass = {};
    Scalar submerged_steam_volume = {};
    Scalar cumulative_accepted_evaporation_mass = {};
    Scalar cumulative_condensed_liquid_mass = {};
    Scalar mass_balance_residual = {};
    Scalar void_balance_residual = {};
    Scalar latent_energy_balance_residual = {};
};

/**
 * @brief Runtime controls for explicit bulk and wall boiling sources.
 */
struct BoilingSourceOptions
{
    bool enable_bulk_boiling = false;
    bool enable_wall_boiling = false;
    real_t saturation_temperature = 373.15;
    real_t boiling_activation_delta_t = 0.0;
    real_t boiling_time_scale = 1.0;
    real_t latent_heat = 2.256e6;
    real_t gas_density = 1.0; ///< Density used to convert vapor mass to void.
    real_t wall_evaporation_fraction = 0.0; ///< Fraction of wall heat used for evaporation.
    real_t wall_heat_flux = 0.0;
    ArrString wall_boiling_patches;
};

/**
 * @brief Validate boiling options and throw on invalid active parameters.
 *
 * @param options Candidate boiling configuration.
 */
inline void validate_boiling_source_options(
    const BoilingSourceOptions& options)
{
    auto require_finite = [](real_t value, const std::string& label)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(label + " must be finite.");
        }
    };

    require_finite(
        options.saturation_temperature, "saturation temperature");
    require_finite(
        options.boiling_activation_delta_t,
        "boiling activation temperature");
    require_finite(options.boiling_time_scale, "boiling time scale");
    require_finite(options.latent_heat, "latent heat");
    require_finite(options.gas_density, "gas density");
    require_finite(
        options.wall_evaporation_fraction,
        "wall evaporation fraction");
    require_finite(options.wall_heat_flux, "wall heat flux");

    if (options.enable_bulk_boiling
        && options.boiling_time_scale <= 0.0)
    {
        throw std::invalid_argument(
            "Bulk boiling requires a positive boiling time scale.");
    }
    if ((options.enable_bulk_boiling || options.enable_wall_boiling)
        && (options.latent_heat <= 0.0 || options.gas_density <= 0.0))
    {
        throw std::invalid_argument(
            "Boiling requires positive latent heat and gas density.");
    }
    if (options.wall_evaporation_fraction < 0.0
        || options.wall_evaporation_fraction > 1.0)
    {
        throw std::invalid_argument(
            "Wall evaporation fraction must be in [0, 1].");
    }
    if (options.enable_wall_boiling && options.wall_heat_flux < 0.0)
    {
        throw std::invalid_argument(
            "Wall boiling requires a non-negative wall heat flux.");
    }
}

/**
 * @brief Parse boiling-source options from a flat database.
 *
 * @param database Database containing optional boiling keys.
 * @return Validated boiling-source options.
 */
inline BoilingSourceOptions boiling_source_options_from_database(
    const Database& database)
{
    BoilingSourceOptions options;
    const detail::DatabaseOptionReader reader(
        database, "Boiling source model");
    options.enable_bulk_boiling = reader.value_or<bool>(
        "enable_bulk_boiling", options.enable_bulk_boiling);
    options.enable_wall_boiling = reader.value_or<bool>(
        "enable_wall_boiling", options.enable_wall_boiling);
    options.saturation_temperature = reader.value_or<real_t>(
        "saturation_temperature",
        options.saturation_temperature);
    options.boiling_activation_delta_t =
        reader.value_or<real_t>(
            "boiling_activation_delta_t",
            options.boiling_activation_delta_t);
    options.boiling_time_scale = reader.value_or<real_t>(
        "boiling_time_scale", options.boiling_time_scale);
    options.latent_heat = reader.value_or<real_t>(
        "latent_heat", options.latent_heat);
    options.gas_density = reader.value_or<real_t>(
        "gas_density", options.gas_density);
    options.wall_evaporation_fraction =
        reader.value_or<real_t>(
            "wall_evaporation_fraction",
            options.wall_evaporation_fraction);
    options.wall_heat_flux = reader.value_or<real_t>(
        "wall_heat_flux", options.wall_heat_flux);
    options.wall_boiling_patches = reader.value_or<ArrString>(
        "wall_boiling_patches",
        options.wall_boiling_patches);
    validate_boiling_source_options(options);
    return options;
}

/**
 * @brief Computes explicit void-fraction and latent-heat source fields.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes,
         class MeshType = Mesh<Pack>>
class BoilingSourceModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using void_model_type = ScalarVoidFractionModel<Pack, mesh_type>;
    using diagnostics_type = BoilingPhaseChangeDiagnostics<scalar_type>;

    /**
     * @brief Construct a boiling model on a mesh with optional configuration.
     */
    BoilingSourceModel(SP<const mesh_type> mesh, BoilingSourceOptions options = {})
        : d_mesh(std::move(mesh)), d_options(std::move(options)), d_source_alpha_boil(d_mesh, "S_alpha_boil"),
          d_latent_heat_sink(d_mesh, "latentHeatSink"),
          d_condensation_latent_heat_release(d_mesh, "condensationLatentHeatRelease"),
          d_condensation_mass_rate(d_mesh, "condensationMassRate"),
          d_phase_change_mass_rate(d_mesh, "phaseChangeMassRate"),
          d_rejected_vapor_mass_rate(d_mesh, "rejectedVaporMassRate")
    {
        if (!d_mesh)
        {
            throw std::invalid_argument(
                "BoilingSourceModel requires a non-null mesh.");
        }
        configure(d_options);
    }

    /**
     * @brief Replace the boiling configuration and clear source fields.
     */
    void configure(const BoilingSourceOptions& options)
    {
        validate_boiling_source_options(options);
        d_options = options;
        d_source_alpha_boil.put_scalar(0.0);
        d_latent_heat_sink.put_scalar(0.0);
        d_condensation_latent_heat_release.put_scalar(0.0);
        d_condensation_mass_rate.put_scalar(0.0);
        d_phase_change_mass_rate.put_scalar(0.0);
        d_rejected_vapor_mass_rate.put_scalar(0.0);
        d_submerged_steam_mass = {};
        d_cumulative_accepted_evaporation_mass = {};
        d_cumulative_condensed_liquid_mass = {};
        d_last_phase_change_diagnostics = {};
        d_pending_time_step = {};
        d_pending_void_volume_before = {};
        d_pending_reserved_void_volume = {};
        d_pending_accepted_void_volume = {};
        d_pending_collapse_volume = {};
        d_pending_submerged_steam_mass_before = {};
        d_pending_collapse_rate.clear();
        d_condensation_release_scratch.clear();
        d_completion_pending = false;
        register_output_fields();
    }

    /**
     * @brief Return the active boiling configuration.
     */
    const BoilingSourceOptions& options() const noexcept
    {
        return d_options;
    }

    /**
     * @brief True when either bulk or wall boiling source terms are enabled.
     */
    bool enabled() const noexcept
    {
        return d_options.enable_bulk_boiling
            || d_options.enable_wall_boiling;
    }

    /**
     * @brief Volumetric void-fraction source generated by boiling.
     */
    const field_type& source_alpha_boil() const noexcept
    {
        return d_source_alpha_boil;
    }

    /**
     * @brief Positive volumetric latent-heat sink before sign conversion.
     */
    const field_type& latent_heat_sink() const noexcept
    {
        return d_latent_heat_sink;
    }

    /**
     * @brief Latent heat released by accepted steam condensation, W/m^3.
     *
     * The boiling model owns only a lumped submerged-steam inventory.  The
     * globally accepted pre-step steam share is therefore distributed over
     * cells in proportion to their bounded scalar-void collapse.  This is a
     * global-conservative spatial approximation, not a local steam fraction.
     */
    const field_type& condensation_latent_heat_release() const noexcept { return d_condensation_latent_heat_release; }

    /**
     * @brief Accepted condensation mass source in kg/(m^3 s).
     *
     * This is the local conservative liquid-mass return corresponding exactly
     * to `condensationLatentHeatRelease / latent_heat`.
     */
    const field_type& condensation_mass_rate() const noexcept { return d_condensation_mass_rate; }

    /**
     * @brief Accepted evaporation mass source in kg/(m^3 s).
     *
     * This is exactly the accepted `latentHeatSink / latent_heat`, and is the
     * local conservative phase-change source for a liquid-mass budget.
     */
    const field_type& phase_change_mass_rate() const noexcept { return d_phase_change_mass_rate; }

    /**
     * @brief Requested vapor mass rate rejected by the void cap, kg/(m^3 s).
     *
     * Rejected demand is reported explicitly but does not remove liquid mass
     * or latent energy because no vapor was admitted to scalar void.
     */
    const field_type& rejected_vapor_mass_rate() const noexcept { return d_rejected_vapor_mass_rate; }

    /** @brief Global accepted evaporation mass for the current step, kg. */
    scalar_type accepted_evaporation_mass_this_step() const noexcept
    {
        return d_last_phase_change_diagnostics.accepted_evaporation_mass;
    }

    /** @brief Global rejected requested vapor mass for the current step, kg. */
    scalar_type rejected_vapor_mass_this_step() const noexcept
    {
        return d_last_phase_change_diagnostics.rejected_vapor_mass;
    }

    /** @brief Global condensate returned to liquid in the current step, kg. */
    scalar_type condensed_liquid_mass_this_step() const noexcept
    {
        return d_last_phase_change_diagnostics.condensed_liquid_mass;
    }

    /** @brief Global submerged steam mass owned by this model, kg. */
    scalar_type global_submerged_steam_mass() const noexcept
    {
        return d_last_phase_change_diagnostics.submerged_steam_mass;
    }

    /**
     * @brief Global submerged steam volume at configured gas density, m^3.
     */
    scalar_type global_submerged_steam_volume() const noexcept
    {
        return d_last_phase_change_diagnostics.submerged_steam_volume;
    }

    /** @brief Latest globally reduced phase-change diagnostics. */
    const diagnostics_type& last_phase_change_diagnostics() const noexcept { return d_last_phase_change_diagnostics; }

    /**
     * @brief True after source acceptance and before steam inventory closure.
     */
    bool phase_change_completion_pending() const noexcept { return d_completion_pending; }

    /**
     * @brief Temperature-equation source contribution for one cell.
     * @return Net phase-change power density, W/m^3.
     */
    scalar_type temperature_source(local_ordinal_type cell_lid) const
    {
        return -d_latent_heat_sink.value(cell_lid) + d_condensation_latent_heat_release.value(cell_lid);
    }

    /**
     * @brief Fields that can be published to solution output.
     */
    const std::map<std::string, const field_type*>& output_fields() const
        noexcept
    {
        return d_output_fields;
    }

    /**
     * @brief Atomically recompute boiling admitted by energy and void bounds.
     *
     * @param time_step Positive explicit timestep.
     * @param temperature Current cell temperature.
     * @param material Current material properties.
     * @param void_model Authoritative scalar gas state and bounds.
     * @param reserved_alpha_source Non-boiling source already reserving void.
     */
    void update(
        scalar_type time_step,
        const field_type& temperature,
        const material_type& material,
        const void_model_type& void_model,
        const field_type* reserved_alpha_source = nullptr)
    {
        const auto enabled_range = global_min_max(enabled() ? 1 : 0);
        if (enabled_range.first != enabled_range.second)
        {
            throw std::logic_error("Boiling enabled state must match on every mesh rank.");
        }
        const int local_invalid_inputs = !std::isfinite(time_step) || time_step <= scalar_type{} ||
                                         &temperature.mesh() != d_mesh.get() ||
                                         &material.density.mesh() != d_mesh.get();
        if (global_max(local_invalid_inputs) != 0)
        {
            throw std::invalid_argument("Boiling source update requires a positive finite timestep "
                                        "and fields on the model mesh on every rank.");
        }

        if (enabled_range.second == 0)
        {
            clear_step_fields();
            reset_step_diagnostics();
            d_completion_pending = false;
            return;
        }
        const auto pending_range = global_min_max(d_completion_pending ? 1 : 0);
        if (pending_range.first != pending_range.second)
        {
            throw std::logic_error("Boiling completion state must match on every mesh rank.");
        }
        if (pending_range.second != 0)
        {
            throw std::logic_error("Boiling source update requires completion of the prior "
                                   "scalar-void step.");
        }

        clear_step_fields();
        reset_step_diagnostics();
        d_completion_pending = false;
        try
        {
            validate_void_inputs(void_model, reserved_alpha_source);
            if (d_options.enable_bulk_boiling)
            {
                add_bulk_boiling(time_step, temperature, material);
            }
            if (d_options.enable_wall_boiling)
            {
                add_wall_boiling();
            }

            limit_to_available_void(
                time_step,
                void_model,
                reserved_alpha_source);
        }
        catch (...)
        {
            clear_step_fields();
            reset_step_diagnostics();
            d_completion_pending = false;
            throw;
        }
    }

    /**
     * @brief Close the accepted steam inventory after scalar void is updated.
     *
     * Call exactly once after `ScalarVoidFractionModel::update_explicit()` for
     * the sources produced by the preceding update().  The hook converts only
     * the scalar model's configured collapse to condensate, limited by steam
     * that was already tracked at the beginning of the step.  Vapor accepted
     * during this step cannot immediately satisfy an old-void collapse.  Any
     * remaining collapse belongs to non-steam scalar gas and is reported
     * separately.  No gas law is introduced: mass/volume conversion uses the
     * already configured constant `gas_density`.
     *
     * Because steam ownership is lumped rather than cellwise, the accepted
     * pre-step steam share is spread proportionally over every cell's bounded
     * collapse.  The resulting latent-heat release is globally conservative
     * but is only a documented spatial approximation.
     *
     * @param time_step Timestep used by both preceding updates, s.
     * @param void_model Authoritative scalar void state after its update.
     * @param track_phase_inventory Enable the new steam/condensate/latent-return
     *        ownership path. The default false preserves legacy boiling behavior
     *        when no free-surface liquid inventory consumes it.
     */
    void complete_void_fraction_update(
        scalar_type time_step, const void_model_type& void_model, bool track_phase_inventory = false)
    {
        const auto enabled_range = global_min_max(enabled() ? 1 : 0);
        if (enabled_range.first != enabled_range.second)
        {
            throw std::logic_error("Boiling enabled state must match on every mesh rank.");
        }
        if (enabled_range.second == 0)
        {
            return;
        }
        const auto pending_range = global_min_max(d_completion_pending ? 1 : 0);
        if (pending_range.first != pending_range.second)
        {
            throw std::logic_error("Boiling completion state must match on every mesh rank.");
        }
        if (pending_range.second == 0)
        {
            throw std::logic_error("Boiling completion requires a preceding source update.");
        }
        const auto tracking_range = global_min_max(track_phase_inventory ? 1 : 0);
        if (tracking_range.first != tracking_range.second)
        {
            throw std::logic_error("Boiling phase-inventory coupling selection must match on every mesh rank.");
        }
        const int local_invalid_control =
            !std::isfinite(time_step) || time_step <= scalar_type{} || time_step != d_pending_time_step ||
            &void_model.alpha_g().mesh() != d_mesh.get() || d_pending_collapse_rate.size() != d_mesh->num_owned_cells();
        if (global_max(local_invalid_control) != 0)
        {
            throw std::invalid_argument("Boiling completion requires matching mesh, timestep, and "
                                        "pending collapse state on every rank.");
        }

        scalar_type local_void_after{};
        int local_invalid_cell = 0;
        const auto& void_options = void_model.options();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto alpha = void_model.alpha_g().value(cell_lid);
            const auto volume = static_cast<scalar_type>(d_mesh->cell_volume(cell_lid));
            if (!std::isfinite(alpha) || alpha < void_options.alpha_min || alpha > void_options.alpha_max ||
                !std::isfinite(volume) || volume <= scalar_type{})
            {
                local_invalid_cell = 1;
                continue;
            }
            local_void_after += alpha * volume;
        }
        if (global_max(local_invalid_cell) != 0)
        {
            throw std::runtime_error("Boiling completion received invalid scalar void or cell "
                                     "volume on at least one rank.");
        }
        const auto void_after = global_sum(local_void_after);
        if (!track_phase_inventory)
        {
            auto diagnostics = d_last_phase_change_diagnostics;
            diagnostics.scalar_void_collapse_volume = d_pending_collapse_volume;
            diagnostics.nonsteam_collapse_volume = d_pending_collapse_volume;
            diagnostics.condensed_liquid_mass = scalar_type{};
            diagnostics.condensation_latent_energy_release = scalar_type{};
            diagnostics.submerged_steam_mass = d_submerged_steam_mass;
            diagnostics.submerged_steam_volume = d_submerged_steam_mass / d_options.gas_density;
            diagnostics.cumulative_accepted_evaporation_mass = d_cumulative_accepted_evaporation_mass;
            diagnostics.cumulative_condensed_liquid_mass = d_cumulative_condensed_liquid_mass;
            diagnostics.mass_balance_residual = scalar_type{};
            diagnostics.void_balance_residual =
                void_after - (d_pending_void_volume_before + d_pending_reserved_void_volume +
                                 d_pending_accepted_void_volume - d_pending_collapse_volume);
            diagnostics.latent_energy_balance_residual = scalar_type{};
            const auto void_scale = std::abs(void_after) + std::abs(d_pending_void_volume_before) +
                                    std::abs(d_pending_reserved_void_volume) +
                                    std::abs(d_pending_accepted_void_volume) + std::abs(d_pending_collapse_volume);
            if (!within_roundoff(diagnostics.void_balance_residual, void_scale))
            {
                throw std::runtime_error("Boiling completion failed its scalar-void closure tolerance.");
            }
            d_condensation_latent_heat_release.put_scalar(0.0);
            d_last_phase_change_diagnostics = std::move(diagnostics);
            d_completion_pending = false;
            return;
        }
        const auto accepted_mass = d_last_phase_change_diagnostics.accepted_evaporation_mass;
        const auto collapse_mass = d_pending_collapse_volume * d_options.gas_density;
        const auto condensed_mass = std::min(collapse_mass, d_pending_submerged_steam_mass_before);
        const auto submerged_steam_mass = d_pending_submerged_steam_mass_before - condensed_mass + accepted_mass;
        const auto condensation_share = collapse_mass > scalar_type{} ? condensed_mass / collapse_mass : scalar_type{};

        scalar_type local_condensation_energy{};
        local_invalid_cell = 0;
        d_condensation_release_scratch.resize(d_mesh->num_owned_cells());
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto volume = static_cast<scalar_type>(d_mesh->cell_volume(cell_lid));
            const auto release =
                d_pending_collapse_rate[owned] * d_options.gas_density * d_options.latent_heat * condensation_share;
            d_condensation_release_scratch[owned] = release;
            if (!std::isfinite(release) || release < scalar_type{} || !std::isfinite(volume) || volume <= scalar_type{})
            {
                local_invalid_cell = 1;
                continue;
            }
            local_condensation_energy += release * volume * time_step;
        }
        if (global_max(local_invalid_cell) != 0)
        {
            throw std::runtime_error("Boiling condensation release is invalid on at least one "
                                     "rank.");
        }
        const auto condensation_energy = global_sum(local_condensation_energy);

        auto diagnostics = d_last_phase_change_diagnostics;
        diagnostics.scalar_void_collapse_volume = d_pending_collapse_volume;
        diagnostics.nonsteam_collapse_volume =
            std::max(d_pending_collapse_volume - condensed_mass / d_options.gas_density, scalar_type{});
        diagnostics.condensed_liquid_mass = condensed_mass;
        diagnostics.condensation_latent_energy_release = condensation_energy;
        diagnostics.submerged_steam_mass = submerged_steam_mass;
        diagnostics.submerged_steam_volume = submerged_steam_mass / d_options.gas_density;
        diagnostics.cumulative_accepted_evaporation_mass = d_cumulative_accepted_evaporation_mass + accepted_mass;
        diagnostics.cumulative_condensed_liquid_mass = d_cumulative_condensed_liquid_mass + condensed_mass;
        diagnostics.mass_balance_residual = diagnostics.submerged_steam_mass + diagnostics.condensed_liquid_mass -
                                            d_pending_submerged_steam_mass_before -
                                            diagnostics.accepted_evaporation_mass;
        diagnostics.void_balance_residual =
            void_after - (d_pending_void_volume_before + d_pending_reserved_void_volume +
                             d_pending_accepted_void_volume - d_pending_collapse_volume);
        diagnostics.latent_energy_balance_residual =
            diagnostics.condensation_latent_energy_release - diagnostics.condensed_liquid_mass * d_options.latent_heat;

        const auto mass_scale =
            std::abs(diagnostics.submerged_steam_mass) + std::abs(diagnostics.condensed_liquid_mass) +
            std::abs(d_pending_submerged_steam_mass_before) + std::abs(diagnostics.accepted_evaporation_mass);
        const auto void_scale = std::abs(void_after) + std::abs(d_pending_void_volume_before) +
                                std::abs(d_pending_reserved_void_volume) + std::abs(d_pending_accepted_void_volume) +
                                std::abs(d_pending_collapse_volume);
        const auto energy_scale = std::abs(diagnostics.condensation_latent_energy_release) +
                                  std::abs(diagnostics.condensed_liquid_mass * d_options.latent_heat);
        if (!within_roundoff(diagnostics.mass_balance_residual, mass_scale) ||
            !within_roundoff(diagnostics.void_balance_residual, void_scale) ||
            !within_roundoff(diagnostics.latent_energy_balance_residual, energy_scale))
        {
            throw std::runtime_error("Boiling completion failed its mass, void, or latent-energy "
                                     "closure tolerance.");
        }

        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto condensation_mass_rate = d_condensation_release_scratch[owned] / d_options.latent_heat;
            d_condensation_mass_rate.set_owned_value(static_cast<local_ordinal_type>(owned), condensation_mass_rate);
            d_condensation_latent_heat_release.set_owned_value(
                static_cast<local_ordinal_type>(owned), d_condensation_release_scratch[owned]);
        }
        d_condensation_mass_rate.sync_ghosts();
        d_condensation_latent_heat_release.sync_ghosts();
        d_cumulative_accepted_evaporation_mass = diagnostics.cumulative_accepted_evaporation_mass;
        d_cumulative_condensed_liquid_mass = diagnostics.cumulative_condensed_liquid_mass;
        d_submerged_steam_mass = diagnostics.submerged_steam_mass;
        d_last_phase_change_diagnostics = std::move(diagnostics);
        d_completion_pending = false;
    }

private:
    /** @brief Clear all per-step source and energy-transfer fields. */
    void clear_step_fields()
    {
        d_source_alpha_boil.put_scalar(0.0);
        d_latent_heat_sink.put_scalar(0.0);
        d_condensation_latent_heat_release.put_scalar(0.0);
        d_condensation_mass_rate.put_scalar(0.0);
        d_phase_change_mass_rate.put_scalar(0.0);
        d_rejected_vapor_mass_rate.put_scalar(0.0);
        std::fill(d_pending_collapse_rate.begin(), d_pending_collapse_rate.end(), scalar_type{});
    }

    /** @brief Clear per-step values while retaining accepted inventory. */
    void reset_step_diagnostics()
    {
        d_last_phase_change_diagnostics = {};
        d_last_phase_change_diagnostics.submerged_steam_mass = d_submerged_steam_mass;
        d_last_phase_change_diagnostics.submerged_steam_volume =
            d_options.gas_density > scalar_type{} ? d_submerged_steam_mass / d_options.gas_density : scalar_type{};
        d_last_phase_change_diagnostics.cumulative_accepted_evaporation_mass = d_cumulative_accepted_evaporation_mass;
        d_last_phase_change_diagnostics.cumulative_condensed_liquid_mass = d_cumulative_condensed_liquid_mass;
    }

    /** @brief Sum a rank-local scalar and replicate it on every rank. */
    scalar_type global_sum(scalar_type local_value) const
    {
        scalar_type global_value{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_value, &global_value);
        return global_value;
    }

    /** @brief Return the communicator-wide maximum of an error flag. */
    int global_max(int local_value) const
    {
        int global_value{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_value, &global_value);
        return global_value;
    }

    /** @brief Return communicator-wide minimum and maximum of an integer. */
    std::pair<int, int> global_min_max(int local_value) const
    {
        int minimum{};
        int maximum{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_value, &minimum);
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_value, &maximum);
        return {minimum, maximum};
    }

    /** @brief Test a globally replicated closure against roundoff scale. */
    static bool within_roundoff(scalar_type residual, scalar_type scale) noexcept
    {
        const auto tolerance =
            scalar_type{1024} * std::numeric_limits<scalar_type>::epsilon() * std::max(scalar_type{1}, scale);
        return std::isfinite(residual) && std::abs(residual) <= tolerance;
    }

    void validate_void_inputs(const void_model_type& void_model, const field_type* reserved_alpha_source) const
    {
        const auto& alpha_g = void_model.alpha_g();
        const int local_mesh_mismatch =
            &alpha_g.mesh() != d_mesh.get() ||
            (reserved_alpha_source != nullptr && &reserved_alpha_source->mesh() != d_mesh.get());
        if (global_max(local_mesh_mismatch) != 0)
        {
            throw std::invalid_argument("Boiling void limiter fields must be on the model mesh on "
                                        "every rank.");
        }
        const auto& void_options = void_model.options();
        int local_invalid_state = 0;
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto alpha = alpha_g.value(cell_lid);
            const auto reserved_rate =
                reserved_alpha_source == nullptr ? scalar_type{} : reserved_alpha_source->value(cell_lid);
            if (!std::isfinite(alpha) || alpha < void_options.alpha_min || alpha > void_options.alpha_max ||
                !std::isfinite(reserved_rate) || reserved_rate < scalar_type{})
            {
                local_invalid_state = 1;
            }
        }
        if (global_max(local_invalid_state) != 0)
        {
            throw std::invalid_argument("Boiling void limiter received invalid canonical state on "
                                        "at least one rank.");
        }
    }

    /**
     * @brief Limit requested boiling to void capacity left by other sources.
     *
     * The source and latent sink are reduced together so every unit of
     * removed latent energy corresponds to vapor admitted by the scalar
     * void-fraction update.
     */
    void limit_to_available_void(
        scalar_type time_step, const void_model_type& void_model, const field_type* reserved_alpha_source)
    {
        const auto& alpha_g = void_model.alpha_g();
        const auto& void_options = void_model.options();
        scalar_type local_requested_mass{};
        scalar_type local_accepted_mass{};
        scalar_type local_rejected_mass{};
        scalar_type local_rejected_void{};
        scalar_type local_void_before{};
        scalar_type local_reserved_void{};
        scalar_type local_accepted_void{};
        scalar_type local_collapse_void{};
        int local_invalid_state = 0;
        d_pending_collapse_rate.assign(d_mesh->num_owned_cells(), scalar_type{});
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto alpha = alpha_g.value(cell_lid);
            const auto reserved_rate =
                reserved_alpha_source == nullptr ? scalar_type{} : reserved_alpha_source->value(cell_lid);
            const auto requested_rate = d_source_alpha_boil.value(cell_lid);
            const auto volume = static_cast<scalar_type>(d_mesh->cell_volume(cell_lid));
            if (!std::isfinite(alpha) || alpha < void_options.alpha_min || alpha > void_options.alpha_max ||
                !std::isfinite(reserved_rate) || reserved_rate < scalar_type{} || !std::isfinite(requested_rate) ||
                requested_rate < scalar_type{} || !std::isfinite(volume) || volume <= scalar_type{})
            {
                local_invalid_state = 1;
                continue;
            }

            const auto collapse_rate = void_model.bounded_collapse_rate(cell_lid, time_step);
            const auto available_rate =
                std::max((void_options.alpha_max - alpha) / time_step - reserved_rate + collapse_rate, scalar_type{});
            const auto accepted_rate = std::min(requested_rate, available_rate);
            const auto accepted_latent_heat_sink = accepted_rate * d_options.gas_density * d_options.latent_heat;
            const auto accepted_mass_rate = accepted_latent_heat_sink / d_options.latent_heat;
            const auto rejected_void_rate = std::max(requested_rate - accepted_rate, scalar_type{});
            const auto rejected_mass_rate = rejected_void_rate * d_options.gas_density;
            const auto requested_mass_rate = accepted_mass_rate + rejected_mass_rate;
            if (!std::isfinite(collapse_rate) || collapse_rate < scalar_type{} || !std::isfinite(available_rate) ||
                !std::isfinite(accepted_rate) || accepted_rate < scalar_type{} ||
                !std::isfinite(accepted_latent_heat_sink) || !std::isfinite(accepted_mass_rate) ||
                !std::isfinite(rejected_void_rate) || !std::isfinite(rejected_mass_rate) ||
                !std::isfinite(requested_mass_rate))
            {
                local_invalid_state = 1;
                continue;
            }
            d_source_alpha_boil.set_owned_value(cell_lid, accepted_rate);
            d_latent_heat_sink.set_owned_value(cell_lid, accepted_latent_heat_sink);
            d_phase_change_mass_rate.set_owned_value(cell_lid, accepted_mass_rate);
            d_rejected_vapor_mass_rate.set_owned_value(cell_lid, rejected_mass_rate);
            d_pending_collapse_rate[owned] = collapse_rate;

            local_requested_mass += requested_mass_rate * volume * time_step;
            local_accepted_mass += accepted_mass_rate * volume * time_step;
            local_rejected_mass += rejected_mass_rate * volume * time_step;
            local_rejected_void += rejected_void_rate * volume * time_step;
            local_void_before += alpha * volume;
            local_reserved_void += reserved_rate * volume * time_step;
            local_accepted_void += accepted_rate * volume * time_step;
            local_collapse_void += collapse_rate * volume * time_step;
        }
        if (!std::isfinite(local_requested_mass) || !std::isfinite(local_accepted_mass) ||
            !std::isfinite(local_rejected_mass) || !std::isfinite(local_rejected_void) ||
            !std::isfinite(local_void_before) || !std::isfinite(local_reserved_void) ||
            !std::isfinite(local_accepted_void) || !std::isfinite(local_collapse_void))
        {
            local_invalid_state = 1;
        }
        if (global_max(local_invalid_state) != 0)
        {
            throw std::runtime_error("Boiling void limiter received invalid cell state on at "
                                     "least one rank.");
        }
        d_source_alpha_boil.sync_ghosts();
        d_latent_heat_sink.sync_ghosts();
        d_phase_change_mass_rate.sync_ghosts();
        d_rejected_vapor_mass_rate.sync_ghosts();

        scalar_type local_values[]{local_requested_mass, local_accepted_mass, local_rejected_mass, local_rejected_void,
            local_void_before, local_reserved_void, local_accepted_void, local_collapse_void};
        scalar_type global_values[8]{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 8, local_values, global_values);
        for (const auto value : global_values)
        {
            if (!std::isfinite(value) || value < scalar_type{})
            {
                throw std::runtime_error("Boiling global phase-change reduction is invalid.");
            }
        }

        auto& diagnostics = d_last_phase_change_diagnostics;
        diagnostics.requested_evaporation_mass = global_values[0];
        diagnostics.accepted_evaporation_mass = global_values[1];
        diagnostics.rejected_vapor_mass = global_values[2];
        diagnostics.rejected_void_volume = global_values[3];
        d_pending_time_step = time_step;
        d_pending_void_volume_before = global_values[4];
        d_pending_reserved_void_volume = global_values[5];
        d_pending_accepted_void_volume = global_values[6];
        d_pending_collapse_volume = global_values[7];
        d_pending_submerged_steam_mass_before = d_submerged_steam_mass;
        d_completion_pending = true;
    }

    void register_output_fields()
    {
        d_output_fields = {{"S_alpha_boil", &d_source_alpha_boil}, {"latentHeatSink", &d_latent_heat_sink},
            {"condensationLatentHeatRelease", &d_condensation_latent_heat_release},
            {"condensationMassRate", &d_condensation_mass_rate}, {"phaseChangeMassRate", &d_phase_change_mass_rate},
            {"rejectedVaporMassRate", &d_rejected_vapor_mass_rate}};
    }

    void add_bulk_boiling(scalar_type time_step, const field_type& temperature, const material_type& material)
    {
        const auto threshold = d_options.saturation_temperature + d_options.boiling_activation_delta_t;
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto temperature_value = temperature.value(cell_lid);
            if (temperature_value <= threshold)
            {
                continue;
            }
            const auto superheat = std::max(temperature_value - d_options.saturation_temperature, scalar_type{});
            const auto sensible_energy =
                material.density.value(cell_lid) * material.specific_heat_capacity.value(cell_lid) * superheat;
            const auto available_energy_rate = sensible_energy / std::max(d_options.boiling_time_scale, time_step);
            const auto mass_rate = available_energy_rate / d_options.latent_heat;
            d_source_alpha_boil.sum_into_value(cell_lid, mass_rate / d_options.gas_density);
            d_latent_heat_sink.sum_into_value(cell_lid, mass_rate * d_options.latent_heat);
        }
    }

    void add_wall_boiling()
    {
        if (d_options.wall_boiling_patches.empty()
            || d_options.wall_heat_flux == scalar_type{}
            || d_options.wall_evaporation_fraction == scalar_type{})
        {
            return;
        }

        for (const auto& [batch_id, batch] : d_mesh->boundary_batches())
        {
            const auto& name = d_mesh->boundary_batch_name(batch_id);
            if (std::find(
                    d_options.wall_boiling_patches.begin(),
                    d_options.wall_boiling_patches.end(),
                    name)
                == d_options.wall_boiling_patches.end())
            {
                continue;
            }
            for (const auto face_lid : batch.face_lids)
            {
                const auto owner = d_mesh->owner_cell(face_lid);
                if (!d_mesh->is_owned_cell(owner))
                {
                    continue;
                }
                const auto volumetric_mass_rate =
                    d_options.wall_evaporation_fraction
                  * d_options.wall_heat_flux
                  / d_options.latent_heat
                  * d_mesh->face_area(face_lid)
                  / d_mesh->cell_volume(owner);
                d_source_alpha_boil.sum_into_value(
                    owner,
                    volumetric_mass_rate / d_options.gas_density);
                d_latent_heat_sink.sum_into_value(
                    owner,
                    volumetric_mass_rate * d_options.latent_heat);
            }
        }
    }

    SP<const mesh_type> d_mesh;
    BoilingSourceOptions d_options;
    field_type d_source_alpha_boil;
    field_type d_latent_heat_sink;
    field_type d_condensation_latent_heat_release;
    field_type d_condensation_mass_rate;
    field_type d_phase_change_mass_rate;
    field_type d_rejected_vapor_mass_rate;
    scalar_type d_submerged_steam_mass = {};
    scalar_type d_cumulative_accepted_evaporation_mass = {};
    scalar_type d_cumulative_condensed_liquid_mass = {};
    scalar_type d_pending_time_step = {};
    scalar_type d_pending_void_volume_before = {};
    scalar_type d_pending_reserved_void_volume = {};
    scalar_type d_pending_accepted_void_volume = {};
    scalar_type d_pending_collapse_volume = {};
    scalar_type d_pending_submerged_steam_mass_before = {};
    std::vector<scalar_type> d_pending_collapse_rate;
    std::vector<scalar_type> d_condensation_release_scratch;
    bool d_completion_pending = false;
    diagnostics_type d_last_phase_change_diagnostics;
    std::map<std::string, const field_type*> d_output_fields;
};

} // namespace SimpleFluid
