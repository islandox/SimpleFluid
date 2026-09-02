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

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace SimpleFluid
{

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

    /**
     * @brief Construct a boiling model on a mesh with optional configuration.
     */
    BoilingSourceModel(
        SP<const mesh_type> mesh,
        BoilingSourceOptions options = {})
        : d_mesh(std::move(mesh)),
          d_options(std::move(options)),
          d_source_alpha_boil(d_mesh, "S_alpha_boil"),
          d_latent_heat_sink(d_mesh, "latentHeatSink")
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
     * @brief Temperature-equation source contribution for one cell.
     * @return Negative latent-heat source for the temperature equation.
     */
    scalar_type temperature_source(local_ordinal_type cell_lid) const
    {
        return -d_latent_heat_sink.value(cell_lid);
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
        if (!std::isfinite(time_step) || time_step <= scalar_type{})
        {
            throw std::invalid_argument(
                "Boiling source update requires a positive finite time step.");
        }
        if (&temperature.mesh() != d_mesh.get()
            || &material.density.mesh() != d_mesh.get())
        {
            throw std::invalid_argument(
                "BoilingSourceModel fields must be on the model mesh.");
        }

        if (!enabled())
        {
            d_source_alpha_boil.put_scalar(0.0);
            d_latent_heat_sink.put_scalar(0.0);
            return;
        }

        validate_void_inputs(
            void_model, reserved_alpha_source);
        d_source_alpha_boil.put_scalar(0.0);
        d_latent_heat_sink.put_scalar(0.0);
        try
        {
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
            d_source_alpha_boil.put_scalar(0.0);
            d_latent_heat_sink.put_scalar(0.0);
            throw;
        }
    }

private:
    void validate_void_inputs(
        const void_model_type& void_model,
        const field_type* reserved_alpha_source) const
    {
        const auto& alpha_g = void_model.alpha_g();
        if (&alpha_g.mesh() != d_mesh.get()
            || (reserved_alpha_source != nullptr
                && &reserved_alpha_source->mesh() != d_mesh.get()))
        {
            throw std::invalid_argument(
                "Boiling void limiter fields must be on the model mesh.");
        }
        const auto& void_options = void_model.options();
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto alpha = alpha_g.value(cell_lid);
            const auto reserved_rate = reserved_alpha_source == nullptr
                                     ? scalar_type{}
                                     : reserved_alpha_source->value(cell_lid);
            if (!std::isfinite(alpha)
                || alpha < void_options.alpha_min
                || alpha > void_options.alpha_max
                || !std::isfinite(reserved_rate)
                || reserved_rate < scalar_type{})
            {
                throw std::invalid_argument(
                    "Boiling void limiter received invalid canonical state.");
            }
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
        scalar_type time_step,
        const void_model_type& void_model,
        const field_type* reserved_alpha_source)
    {
        const auto& alpha_g = void_model.alpha_g();
        const auto& void_options = void_model.options();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto alpha = alpha_g.value(cell_lid);
            const auto reserved_rate = reserved_alpha_source == nullptr
                                     ? scalar_type{}
                                     : reserved_alpha_source->value(cell_lid);
            const auto requested_rate =
                d_source_alpha_boil.value(cell_lid);
            if (!std::isfinite(alpha)
                || alpha < void_options.alpha_min
                || alpha > void_options.alpha_max
                || !std::isfinite(reserved_rate)
                || !std::isfinite(requested_rate)
                || requested_rate < scalar_type{})
            {
                throw std::runtime_error(
                    "Boiling void limiter received invalid cell state.");
            }

            const auto collapse_rate =
                void_model.bounded_collapse_rate(cell_lid, time_step);
            const auto available_rate = std::max(
                (void_options.alpha_max - alpha) / time_step
              - reserved_rate
              + collapse_rate,
                scalar_type{});
            const auto accepted_rate =
                std::min(requested_rate, available_rate);
            d_source_alpha_boil.set_owned_value(
                cell_lid, accepted_rate);
            d_latent_heat_sink.set_owned_value(
                cell_lid,
                accepted_rate
              * d_options.gas_density
              * d_options.latent_heat);
        }
        d_source_alpha_boil.sync_ghosts();
        d_latent_heat_sink.sync_ghosts();
    }

    void register_output_fields()
    {
        d_output_fields = {
            {"S_alpha_boil", &d_source_alpha_boil},
            {"latentHeatSink", &d_latent_heat_sink}};
    }

    void add_bulk_boiling(
        scalar_type time_step,
        const field_type& temperature,
        const material_type& material)
    {
        const auto threshold =
            d_options.saturation_temperature
          + d_options.boiling_activation_delta_t;
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto temperature_value =
                temperature.value(cell_lid);
            if (temperature_value <= threshold)
            {
                continue;
            }
            const auto superheat = std::max(
                temperature_value - d_options.saturation_temperature,
                scalar_type{});
            const auto sensible_energy =
                material.density.value(cell_lid)
              * material.specific_heat_capacity.value(cell_lid)
              * superheat;
            const auto available_energy_rate =
                sensible_energy
              / std::max(d_options.boiling_time_scale, time_step);
            const auto mass_rate =
                available_energy_rate / d_options.latent_heat;
            d_source_alpha_boil.sum_into_value(
                cell_lid, mass_rate / d_options.gas_density);
            d_latent_heat_sink.sum_into_value(
                cell_lid, mass_rate * d_options.latent_heat);
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
    std::map<std::string, const field_type*> d_output_fields;
};

} // namespace SimpleFluid
