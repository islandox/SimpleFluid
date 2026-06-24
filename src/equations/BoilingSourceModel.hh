/**
 * @file BoilingSourceModel.hh
 * @brief Explicit bulk and wall boiling source model.
 */
#pragma once

#include "equations/BoussinesqModel.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace SimpleFluid
{

struct BoilingSourceOptions
{
    bool enable_bulk_boiling = false;
    bool enable_wall_boiling = false;
    real_t saturation_temperature = 373.15;
    real_t boiling_activation_delta_t = 0.0;
    real_t boiling_time_scale = 1.0;
    real_t latent_heat = 2.256e6;
    real_t gas_density = 1.0;
    real_t wall_evaporation_fraction = 0.0;
    real_t wall_heat_flux = 0.0;
    ArrString wall_boiling_patches;
};

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
}

inline BoilingSourceOptions boiling_source_options_from_database(
    const Database& database)
{
    BoilingSourceOptions options;
    options.enable_bulk_boiling = detail::database_value_or<bool>(
        database, "enable_bulk_boiling", options.enable_bulk_boiling);
    options.enable_wall_boiling = detail::database_value_or<bool>(
        database, "enable_wall_boiling", options.enable_wall_boiling);
    options.saturation_temperature = detail::database_value_or<real_t>(
        database,
        "saturation_temperature",
        options.saturation_temperature);
    options.boiling_activation_delta_t =
        detail::database_value_or<real_t>(
            database,
            "boiling_activation_delta_t",
            options.boiling_activation_delta_t);
    options.boiling_time_scale = detail::database_value_or<real_t>(
        database, "boiling_time_scale", options.boiling_time_scale);
    options.latent_heat = detail::database_value_or<real_t>(
        database, "latent_heat", options.latent_heat);
    options.gas_density = detail::database_value_or<real_t>(
        database, "gas_density", options.gas_density);
    options.wall_evaporation_fraction =
        detail::database_value_or<real_t>(
            database,
            "wall_evaporation_fraction",
            options.wall_evaporation_fraction);
    options.wall_heat_flux = detail::database_value_or<real_t>(
        database, "wall_heat_flux", options.wall_heat_flux);
    options.wall_boiling_patches = detail::database_value_or<ArrString>(
        database,
        "wall_boiling_patches",
        options.wall_boiling_patches);
    validate_boiling_source_options(options);
    return options;
}

template<TpetraTypePack Pack = DefaultTpetraTypes>
class BoilingSourceModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using material_type = MaterialPropertyFields<Pack>;

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

    void configure(const BoilingSourceOptions& options)
    {
        validate_boiling_source_options(options);
        d_options = options;
        d_source_alpha_boil.put_scalar(0.0);
        d_latent_heat_sink.put_scalar(0.0);
        register_output_fields();
    }

    const BoilingSourceOptions& options() const noexcept
    {
        return d_options;
    }

    bool enabled() const noexcept
    {
        return d_options.enable_bulk_boiling
            || d_options.enable_wall_boiling;
    }

    const field_type& source_alpha_boil() const noexcept
    {
        return d_source_alpha_boil;
    }

    const field_type& latent_heat_sink() const noexcept
    {
        return d_latent_heat_sink;
    }

    scalar_type temperature_source(local_ordinal_type cell_lid) const
    {
        return -d_latent_heat_sink.value(cell_lid);
    }

    const std::map<std::string, const field_type*>& output_fields() const
        noexcept
    {
        return d_output_fields;
    }

    void update(
        const field_type& temperature,
        const material_type& material)
    {
        if (&temperature.mesh() != d_mesh.get()
            || &material.density.mesh() != d_mesh.get())
        {
            throw std::invalid_argument(
                "BoilingSourceModel fields must be on the model mesh.");
        }

        d_source_alpha_boil.put_scalar(0.0);
        d_latent_heat_sink.put_scalar(0.0);
        if (!enabled())
        {
            return;
        }

        if (d_options.enable_bulk_boiling)
        {
            add_bulk_boiling(temperature, material);
        }
        if (d_options.enable_wall_boiling)
        {
            add_wall_boiling();
        }

        d_source_alpha_boil.sync_ghosts();
        d_latent_heat_sink.sync_ghosts();
    }

private:
    void register_output_fields()
    {
        d_output_fields = {
            {"S_alpha_boil", &d_source_alpha_boil},
            {"latentHeatSink", &d_latent_heat_sink}};
    }

    void add_bulk_boiling(
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
            const auto available_energy_rate =
                material.density.value(cell_lid)
              * material.specific_heat_capacity.value(cell_lid)
              * superheat / d_options.boiling_time_scale;
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
