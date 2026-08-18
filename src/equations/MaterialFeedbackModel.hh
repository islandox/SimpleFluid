/**
 * @file MaterialFeedbackModel.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Temperature- and void-dependent material feedback model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoussinesqModel.hh"
#include "fields/MeshFieldTraits.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace SimpleFluid
{

/**
 * @brief Density update modes for thermal and void feedback.
 */
enum class DensityFeedbackMode
{
    Constant,                  ///< Preserve the reference density.
    BoussinesqTemperatureOnly, ///< Apply temperature-dependent liquid density.
    BoussinesqVoid,            ///< Blend Boussinesq liquid and gas densities.
    Mixture                    ///< Blend constant liquid and gas densities.
};

/**
 * @brief Dynamic-viscosity update modes.
 */
enum class ViscosityFeedbackMode
{
    Constant ///< Preserve the reference dynamic viscosity.
};

/**
 * @brief Runtime controls for material-property feedback.
 */
struct MaterialFeedbackOptions
{
    DensityFeedbackMode density_mode = DensityFeedbackMode::Constant;
    ViscosityFeedbackMode viscosity_mode = ViscosityFeedbackMode::Constant;
    real_t reference_density = 1.0;
    real_t liquid_density = 1.0; ///< Constant liquid density for mixture mode.
    real_t gas_density = 1.0; ///< Gas density used by void-aware modes.
    real_t reference_temperature = 0.0;
    real_t thermal_expansion = 0.0;
    real_t reference_dynamic_viscosity = 1.0;
    real_t min_density = 1.0e-12; ///< Positive density floor.
    real_t min_viscosity = 0.0; ///< Non-negative viscosity floor.
};

/**
 * @brief Parse a database density-feedback mode string.
 *
 * @param value Mode name from user input.
 * @return Parsed density-feedback mode.
 */
inline DensityFeedbackMode parse_density_feedback_mode(
    const std::string& value)
{
    if (value == "constant")
        return DensityFeedbackMode::Constant;
    if (value == "boussinesqTemperatureOnly")
        return DensityFeedbackMode::BoussinesqTemperatureOnly;
    if (value == "boussinesqVoid")
        return DensityFeedbackMode::BoussinesqVoid;
    if (value == "mixture")
        return DensityFeedbackMode::Mixture;
    throw std::invalid_argument(
        "Unknown density feedback model '" + value + "'.");
}

/**
 * @brief Parse a database viscosity-feedback mode string.
 *
 * @param value Mode name from user input.
 * @return Parsed viscosity-feedback mode.
 */
inline ViscosityFeedbackMode parse_viscosity_feedback_mode(
    const std::string& value)
{
    if (value == "constant")
        return ViscosityFeedbackMode::Constant;
    throw std::invalid_argument(
        "Unknown viscosity feedback model '" + value + "'.");
}

/**
 * @brief Validate material-feedback constants and floors.
 *
 * @param options Candidate feedback configuration.
 */
inline void validate_material_feedback_options(
    const MaterialFeedbackOptions& options)
{
    const real_t values[] = {
        options.reference_density,
        options.liquid_density,
        options.gas_density,
        options.reference_temperature,
        options.thermal_expansion,
        options.reference_dynamic_viscosity,
        options.min_density,
        options.min_viscosity};
    for (const auto value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(
                "Material feedback options must be finite.");
        }
    }
    if (options.reference_density <= 0.0
        || options.liquid_density <= 0.0
        || options.gas_density <= 0.0
        || options.min_density <= 0.0)
    {
        throw std::invalid_argument(
            "Material feedback density values must be positive.");
    }
    if (options.reference_dynamic_viscosity < 0.0
        || options.min_viscosity < 0.0)
    {
        throw std::invalid_argument(
            "Material feedback viscosity values cannot be negative.");
    }
}

/**
 * @brief Parse material-feedback options from model, time, and database state.
 *
 * @param database Database containing optional feedback keys.
 * @param model_options Boussinesq material defaults.
 * @param time_options Time-stepper values used for Boussinesq density.
 * @return Validated material-feedback options.
 */
inline MaterialFeedbackOptions material_feedback_options_from_database(
    const Database& database,
    const BoussinesqModelOptions& model_options,
    const TimeStepperOptions& time_options)
{
    MaterialFeedbackOptions options;
    options.reference_density = model_options.reference_density;
    options.liquid_density = model_options.density;
    options.gas_density = detail::database_value_or<real_t>(
        database, "gas_density", options.gas_density);
    options.reference_temperature = time_options.reference_temperature;
    options.thermal_expansion = time_options.thermal_expansion;
    options.reference_dynamic_viscosity =
        model_options.dynamic_viscosity.value_or(
            model_options.reference_density
          * time_options.kinematic_viscosity);
    options.density_mode = parse_density_feedback_mode(
        detail::database_value_or<std::string>(
            database, "density_feedback_model", "constant"));
    options.viscosity_mode = parse_viscosity_feedback_mode(
        detail::database_value_or<std::string>(
            database, "viscosity_feedback_model", "constant"));
    options.min_density = detail::database_value_or<real_t>(
        database, "min_density", options.min_density);
    options.min_viscosity = detail::database_value_or<real_t>(
        database, "min_viscosity", options.min_viscosity);
    validate_material_feedback_options(options);
    return options;
}

/**
 * @brief Updates material fields from temperature and optional void fraction.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes,
         class MeshType = Mesh<Pack>>
class MaterialFeedbackModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using context_type = BoussinesqUpdateContext<Pack, mesh_type>;

    /**
     * @brief Construct a material-feedback model on a mesh.
     */
    MaterialFeedbackModel(
        SP<const mesh_type> mesh,
        MaterialFeedbackOptions options = {})
        : d_mesh(std::move(mesh)),
          d_options(std::move(options)),
          d_density_feedback(
              d_mesh, d_options.reference_density, "rhoFeedback"),
          d_viscosity_feedback(
              d_mesh,
              d_options.reference_dynamic_viscosity,
              "muFeedback")
    {
        if (!d_mesh)
        {
            throw std::invalid_argument(
                "MaterialFeedbackModel requires a non-null mesh.");
        }
        configure(d_options);
    }

    /**
     * @brief Replace feedback options and reset feedback fields.
     */
    void configure(const MaterialFeedbackOptions& options)
    {
        validate_material_feedback_options(options);
        d_options = options;
        d_density_feedback.put_scalar(d_options.reference_density);
        d_viscosity_feedback.put_scalar(
            d_options.reference_dynamic_viscosity);
        register_output_fields();
    }

    /**
     * @brief Return the active material-feedback options.
     */
    const MaterialFeedbackOptions& options() const noexcept
    {
        return d_options;
    }

    /**
     * @brief True when density is computed from a non-constant mode.
     */
    bool density_feedback_enabled() const noexcept
    {
        return d_options.density_mode != DensityFeedbackMode::Constant;
    }

    /**
     * @brief Density feedback field written during the last apply call.
     */
    const field_type& density_feedback() const noexcept
    {
        return d_density_feedback;
    }

    /**
     * @brief Dynamic-viscosity feedback field written during the last apply call.
     */
    const field_type& viscosity_feedback() const noexcept
    {
        return d_viscosity_feedback;
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
     * @brief Apply feedback to the solver material-property fields.
     */
    void apply(
        const context_type& context,
        const field_type* alpha_g,
        material_type& material)
    {
        if (&context.mesh != d_mesh.get()
            || &material.density.mesh() != d_mesh.get())
        {
            throw std::invalid_argument(
                "MaterialFeedbackModel received fields on the wrong mesh.");
        }
        if (alpha_g != nullptr && &alpha_g->mesh() != d_mesh.get())
        {
            throw std::invalid_argument(
                "MaterialFeedbackModel received alpha_g on the wrong mesh.");
        }

        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto alpha =
                alpha_g == nullptr ? scalar_type{} : alpha_g->value(cell_lid);
            const auto density =
                std::max(density_value(context, cell_lid, alpha),
                         d_options.min_density);
            const auto viscosity =
                std::max(viscosity_value(), d_options.min_viscosity);
            d_density_feedback.set_owned_value(cell_lid, density);
            d_viscosity_feedback.set_owned_value(cell_lid, viscosity);
            material.density.set_owned_value(cell_lid, density);
            material.dynamic_viscosity.set_owned_value(cell_lid, viscosity);
        }
        d_density_feedback.sync_ghosts();
        d_viscosity_feedback.sync_ghosts();
        material.validate_and_sync();
    }

private:
    void register_output_fields()
    {
        d_output_fields = {
            {"rhoFeedback", &d_density_feedback},
            {"muFeedback", &d_viscosity_feedback}};
    }

    scalar_type density_value(
        const context_type& context,
        local_ordinal_type cell_lid,
        scalar_type alpha) const
    {
        switch (d_options.density_mode)
        {
            case DensityFeedbackMode::Constant:
                return d_options.reference_density;
            case DensityFeedbackMode::BoussinesqTemperatureOnly:
                return boussinesq_liquid_density(
                    context.temperature.value(cell_lid));
            case DensityFeedbackMode::BoussinesqVoid:
            {
                const auto liquid =
                    boussinesq_liquid_density(
                        context.temperature.value(cell_lid));
                return liquid * (1.0 - alpha)
                     + d_options.gas_density * alpha;
            }
            case DensityFeedbackMode::Mixture:
                return d_options.liquid_density * (1.0 - alpha)
                     + d_options.gas_density * alpha;
        }
        return d_options.reference_density;
    }

    scalar_type boussinesq_liquid_density(
        scalar_type temperature) const
    {
        return d_options.reference_density
             * (1.0 - d_options.thermal_expansion
                      * (temperature
                         - d_options.reference_temperature));
    }

    scalar_type viscosity_value() const
    {
        return d_options.reference_dynamic_viscosity;
    }

    SP<const mesh_type> d_mesh;
    MaterialFeedbackOptions d_options;
    field_type d_density_feedback;
    field_type d_viscosity_feedback;
    std::map<std::string, const field_type*> d_output_fields;
};

} // namespace SimpleFluid
