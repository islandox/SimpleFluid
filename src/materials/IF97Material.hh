/** @file IF97Material.hh
 * @brief Adapters from optional IF97 water to existing material fields.
 */
#pragma once

#include "equations/BoussinesqModel.hh"
#include "materials/IF97Water.hh"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace SimpleFluid
{

/** Uniform physical coefficients at a liquid-water reference state in SI. */
inline BoussinesqModelOptions if97_liquid_model_options(double temperature, double absolute_pressure)
{
    const auto water = IF97Water::liquid(temperature, absolute_pressure);
    BoussinesqModelOptions options;
    options.reference_density = options.density = water.density;
    options.specific_heat_capacity = water.specific_heat_capacity;
    options.dynamic_viscosity = water.dynamic_viscosity;
    options.thermal_conductivity = water.thermal_conductivity;
    return options;
}

/**
 * Make a pure-liquid material updater at a prescribed absolute pressure [Pa].
 * Temperature comes from the context in Kelvin. Gauge pressure is deliberately
 * unused: this is an isobaric material closure for the Boussinesq solver.
 * Install before free-surface initialization and do not combine with a
 * MaterialFeedbackModel that would overwrite density/viscosity. If buoyancy
 * uses these densities, set model_options.density_feedback_enabled = true.
 * The callback performs no communication; MaterialPropertyFields::update()
 * owns collective validation and ghost synchronization. All local states
 * are evaluated before publishing any values on this rank.
 * The current planarALE solver rejects user callbacks, including this one.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>>
auto make_if97_liquid_material_updater(double absolute_pressure) ->
    typename MaterialPropertyFields<Pack, MeshType>::updater_type
{
    // Validate the pressure independently of any mesh/temperature state.
    if (!std::isfinite(absolute_pressure) || absolute_pressure <= 0.0)
        throw std::invalid_argument("IF97 material absolute pressure must be finite and positive [Pa].");
    if (absolute_pressure < 611.213 || absolute_pressure > 1.0e8)
        throw std::out_of_range("IF97 material absolute pressure must be in 611.213..1e8 Pa.");
    return [absolute_pressure](
               const BoussinesqUpdateContext<Pack, MeshType>& context, MaterialPropertyFields<Pack, MeshType>& material)
    {
        if (&context.mesh != &context.temperature.mesh() || &context.mesh != &material.density.mesh() ||
            &context.mesh != &material.specific_heat_capacity.mesh() ||
            &context.mesh != &material.dynamic_viscosity.mesh() ||
            &context.mesh != &material.thermal_conductivity.mesh())
            throw std::invalid_argument("IF97 material updater received fields on different meshes.");
        std::vector<IF97Water::Properties> values;
        values.reserve(context.mesh.num_owned_cells());
        for (size_t owned = 0; owned < context.mesh.num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<typename Pack::local_ordinal_type>(owned);
            values.push_back(IF97Water::liquid(context.temperature.value(cell), absolute_pressure));
        }
        for (size_t owned = 0; owned < values.size(); ++owned)
        {
            const auto cell = static_cast<typename Pack::local_ordinal_type>(owned);
            const auto& water = values[owned];
            material.density.set_owned_value(cell, water.density);
            material.specific_heat_capacity.set_owned_value(cell, water.specific_heat_capacity);
            material.dynamic_viscosity.set_owned_value(cell, water.dynamic_viscosity);
            material.thermal_conductivity.set_owned_value(cell, water.thermal_conductivity);
        }
    };
}

} // namespace SimpleFluid
