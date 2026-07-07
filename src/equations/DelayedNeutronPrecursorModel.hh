/**
 * @file DelayedNeutronPrecursorModel.hh
 * @brief Liquid-phase delayed-neutron precursor scalar model.
 */
#pragma once

#include "equations/BoussinesqModel.hh"

#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Runtime controls for delayed-neutron precursor groups.
 */
struct DelayedNeutronPrecursorOptions
{
    int group_count = 0;
    ArrReal decay_constants;
    ArrReal initial_concentrations;
    ArrReal source_terms;
    ArrReal power_yields;
    real_t effective_diffusivity = 0.0;
};

/**
 * @brief Validate precursor group counts, vectors, and transport constants.
 *
 * @param options Candidate precursor configuration.
 */
inline void validate_delayed_neutron_precursor_options(
    const DelayedNeutronPrecursorOptions& options)
{
    if (options.group_count < 0)
    {
        throw std::invalid_argument(
            "Precursor group count cannot be negative.");
    }
    const auto count = static_cast<size_t>(options.group_count);
    auto check_size =
        [count](const ArrReal& values, const std::string& label)
    {
        if (!values.empty() && values.size() != count)
        {
            throw std::invalid_argument(
                label + " must be empty or match precursor_group_count.");
        }
        for (const auto value : values)
        {
            if (!std::isfinite(value))
            {
                throw std::invalid_argument(
                    label + " values must be finite.");
            }
        }
    };
    check_size(options.decay_constants, "precursor_decay_constants");
    check_size(
        options.initial_concentrations,
        "precursor_initial_concentrations");
    check_size(options.source_terms, "precursor_source_terms");
    check_size(options.power_yields, "precursor_power_yields");
    for (const auto value : options.decay_constants)
    {
        if (value < 0.0)
        {
            throw std::invalid_argument(
                "Precursor decay constants cannot be negative.");
        }
    }
    for (const auto value : options.initial_concentrations)
    {
        if (value < 0.0)
        {
            throw std::invalid_argument(
                "Precursor initial concentrations cannot be negative.");
        }
    }
    if (!std::isfinite(options.effective_diffusivity)
        || options.effective_diffusivity < 0.0)
    {
        throw std::invalid_argument(
            "Precursor effective diffusivity must be finite and non-negative.");
    }
}

/**
 * @brief Parse delayed-neutron precursor options from a flat database.
 *
 * @param database Database containing optional precursor keys.
 * @return Validated precursor options.
 */
inline DelayedNeutronPrecursorOptions
delayed_neutron_precursor_options_from_database(
    const Database& database)
{
    DelayedNeutronPrecursorOptions options;
    options.group_count = detail::database_value_or<int>(
        database, "precursor_group_count", options.group_count);
    options.decay_constants = detail::database_value_or<ArrReal>(
        database, "precursor_decay_constants", options.decay_constants);
    options.initial_concentrations = detail::database_value_or<ArrReal>(
        database,
        "precursor_initial_concentrations",
        options.initial_concentrations);
    options.source_terms = detail::database_value_or<ArrReal>(
        database, "precursor_source_terms", options.source_terms);
    options.power_yields = detail::database_value_or<ArrReal>(
        database, "precursor_power_yields", options.power_yields);
    options.effective_diffusivity = detail::database_value_or<real_t>(
        database,
        "precursor_effective_diffusivity",
        options.effective_diffusivity);
    validate_delayed_neutron_precursor_options(options);
    return options;
}

/**
 * @brief Liquid-phase explicit delayed-neutron precursor source model.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class DelayedNeutronPrecursorModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;

    /**
     * @brief Construct a precursor model on a mesh.
     */
    DelayedNeutronPrecursorModel(
        SP<const mesh_type> mesh,
        DelayedNeutronPrecursorOptions options = {})
        : d_mesh(std::move(mesh))
    {
        if (!d_mesh)
        {
            throw std::invalid_argument(
                "DelayedNeutronPrecursorModel requires a non-null mesh.");
        }
        configure(options);
    }

    /**
     * @brief Replace precursor options and rebuild group fields.
     */
    void configure(const DelayedNeutronPrecursorOptions& options)
    {
        validate_delayed_neutron_precursor_options(options);
        d_options = options;
        d_fields.clear();
        d_sources.clear();
        d_output_fields.clear();
        const auto count = static_cast<size_t>(d_options.group_count);
        d_decay_constants = complete_vector(
            d_options.decay_constants, count, 0.0);
        d_source_terms = complete_vector(
            d_options.source_terms, count, 0.0);
        d_power_yields = complete_vector(
            d_options.power_yields, count, 0.0);
        const auto initial = complete_vector(
            d_options.initial_concentrations, count, 0.0);

        for (size_t group = 0; group < count; ++group)
        {
            const auto suffix = std::to_string(group + 1);
            auto field = std::make_unique<field_type>(
                d_mesh, initial[group], "C_" + suffix);
            auto source = std::make_unique<field_type>(
                d_mesh, 0.0, "S_C_" + suffix);
            d_output_fields.emplace(field->name(), field.get());
            d_output_fields.emplace(source->name(), source.get());
            d_fields.push_back(std::move(field));
            d_sources.push_back(std::move(source));
        }
    }

    /**
     * @brief True when at least one precursor group is active.
     */
    bool enabled() const noexcept
    {
        return !d_fields.empty();
    }

    /**
     * @brief Number of configured precursor groups.
     */
    size_t group_count() const noexcept { return d_fields.size(); }

    /**
     * @brief Concentration field for a zero-based precursor group index.
     */
    const field_type& concentration(size_t group) const
    {
        return *d_fields.at(group);
    }

    /**
     * @brief Source field for a zero-based precursor group index.
     */
    const field_type& source(size_t group) const
    {
        return *d_sources.at(group);
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
     * @brief Advance all precursor groups by one explicit source/decay step.
     */
    void advance(
        scalar_type time_step,
        const field_type& alpha_l,
        const field_type* fission_power_density)
    {
        if (!enabled())
        {
            return;
        }
        if (!std::isfinite(time_step) || time_step <= 0.0)
        {
            throw std::invalid_argument(
                "Precursor update requires a positive finite time step.");
        }
        if (&alpha_l.mesh() != d_mesh.get()
            || (fission_power_density != nullptr
                && &fission_power_density->mesh() != d_mesh.get()))
        {
            throw std::invalid_argument(
                "Precursor fields must be on the model mesh.");
        }

        for (size_t group = 0; group < d_fields.size(); ++group)
        {
            auto& field = *d_fields[group];
            auto& source_field = *d_sources[group];
            const auto lambda = d_decay_constants[group];
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto liquid_fraction =
                    std::max(alpha_l.value(cell_lid), scalar_type{1.0e-14});
                const auto source =
                    d_source_terms[group]
                  + (fission_power_density
                         ? d_power_yields[group]
                         * fission_power_density->value(cell_lid)
                         : scalar_type{});
                const auto old_value = field.value(cell_lid);
                scalar_type new_value{};
                if (lambda > scalar_type{})
                {
                    const auto decay = std::exp(-lambda * time_step);
                    new_value =
                        old_value * decay
                      + source / (liquid_fraction * lambda)
                      * (1.0 - decay);
                }
                else
                {
                    new_value =
                        old_value
                      + time_step * source / liquid_fraction;
                }
                new_value = std::max(new_value, scalar_type{});
                field.set_owned_value(cell_lid, new_value);
                source_field.set_owned_value(cell_lid, source);
            }
            field.sync_ghosts();
            source_field.sync_ghosts();
        }
    }

private:
    static ArrReal complete_vector(
        const ArrReal& values,
        size_t count,
        real_t fallback)
    {
        if (values.empty())
        {
            return ArrReal(count, fallback);
        }
        return values;
    }

    SP<const mesh_type> d_mesh;
    DelayedNeutronPrecursorOptions d_options;
    ArrReal d_decay_constants;
    ArrReal d_source_terms;
    ArrReal d_power_yields;
    std::vector<std::unique_ptr<field_type>> d_fields;
    std::vector<std::unique_ptr<field_type>> d_sources;
    std::map<std::string, const field_type*> d_output_fields;
};

} // namespace SimpleFluid
