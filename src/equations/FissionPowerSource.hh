/**
 * @file FissionPowerSource.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Prescribed fission power-density profiles for thermal coupling.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoussinesqModel.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Supported prescribed fission power-density profiles.
 */
enum class FissionPowerProfile
{
    Disabled,
    Constant, ///< Apply a spatially uniform power density.
    Gaussian  ///< Apply a Gaussian profile normalized to total power.
};

/**
 * @brief Parse a fission power profile name.
 */
inline FissionPowerProfile
fission_power_profile_from_string(std::string_view value)
{
    std::string normalized(value);
    std::ranges::transform(
        normalized,
        normalized.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });

    if (normalized == "disabled")
        return FissionPowerProfile::Disabled;
    if (normalized == "constant")
        return FissionPowerProfile::Constant;
    if (normalized == "gaussian")
        return FissionPowerProfile::Gaussian;

    throw std::invalid_argument(
        "Unknown fission power mode; expected disabled, constant, or gaussian.");
}

/**
 * @brief Runtime controls for the prescribed fission power source.
 */
struct FissionPowerSourceOptions
{
    FissionPowerProfile profile = FissionPowerProfile::Disabled;
    real_t power_density = 0.0; ///< Uniform power density for the constant profile.
    real_t total_power = 0.0; ///< Integrated power for the Gaussian profile.
    vec3<real_t> center{};
    vec3<real_t> standard_deviation{1.0, 1.0, 1.0}; ///< Gaussian widths by axis.
};

namespace detail
{

/**
 * @brief Read a required, typed fission-source database value.
 * @tparam T Requested database value type.
 * @param database Source database.
 * @param key Required option name.
 * @return The decoded option value.
 * @throws std::invalid_argument If the option is absent or has the wrong type.
 */
template<class T>
T fission_database_value(
    const Database& database,
    const std::string& key)
{
    if (!database.contains(key))
    {
        throw std::invalid_argument(
            "Missing required fission power option '" + key + "'.");
    }
    try
    {
        return database.get<T>(key);
    }
    catch (const std::out_of_range&)
    {
        throw std::invalid_argument(
            "Fission power option '" + key + "' has the wrong type.");
    }
}

/**
 * @brief Validate a non-negative fission-source scalar.
 * @param value Value to validate.
 * @param name Option name used in diagnostics.
 * @throws std::invalid_argument If @p value is negative or not finite.
 */
inline void validate_non_negative_fission_value(
    real_t value,
    const std::string& name)
{
    if (!std::isfinite(value) || value < 0.0)
    {
        throw std::invalid_argument(
            "Fission power option '" + name
            + "' must be finite and non-negative.");
    }
}

/**
 * @brief Read a finite three-component fission-source option.
 * @param database Source database.
 * @param key Required array option name.
 * @return The option converted to a three-component vector.
 * @throws std::invalid_argument If the option is absent, malformed, or non-finite.
 */
inline vec3<real_t> fission_vec3(
    const Database& database,
    const std::string& key)
{
    const auto values =
        fission_database_value<ArrReal>(database, key);
    if (values.size() != 3)
    {
        throw std::invalid_argument(
            "Fission power option '" + key
            + "' must contain exactly three values.");
    }
    for (const auto value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(
                "Fission power option '" + key
                + "' must contain finite values.");
        }
    }
    return {values[0], values[1], values[2]};
}

/**
 * @brief Validate prescribed fission-power options for the selected profile.
 * @param options Options to validate.
 * @throws std::invalid_argument If an active profile contains invalid parameters.
 */
inline void validate_fission_power_options(
    const FissionPowerSourceOptions& options)
{
    switch (options.profile)
    {
        case FissionPowerProfile::Disabled:
            return;
        case FissionPowerProfile::Constant:
            validate_non_negative_fission_value(
                options.power_density, "fission_power_density");
            return;
        case FissionPowerProfile::Gaussian:
            validate_non_negative_fission_value(
                options.total_power, "fission_total_power");
            for (size_t component = 0; component < 3; ++component)
            {
                if (!std::isfinite(options.center.component(component)))
                {
                    throw std::invalid_argument(
                        "Fission Gaussian center must be finite.");
                }
                if (!std::isfinite(
                        options.standard_deviation.component(component))
                    || options.standard_deviation.component(component) <= 0.0)
                {
                    throw std::invalid_argument(
                        "Fission Gaussian standard deviations must be finite "
                        "and positive.");
                }
            }
            return;
    }
}

} // namespace detail

/**
 * @brief Parse fission power-source options from a flat database.
 */
inline FissionPowerSourceOptions
fission_power_source_options_from_database(const Database& database)
{
    FissionPowerSourceOptions options;
    if (!database.contains("fission_power_mode"))
    {
        return options;
    }

    options.profile = fission_power_profile_from_string(
        detail::fission_database_value<std::string>(
            database, "fission_power_mode"));
    switch (options.profile)
    {
        case FissionPowerProfile::Disabled:
            break;
        case FissionPowerProfile::Constant:
            options.power_density =
                detail::fission_database_value<real_t>(
                    database, "fission_power_density");
            break;
        case FissionPowerProfile::Gaussian:
            options.total_power =
                detail::fission_database_value<real_t>(
                    database, "fission_total_power");
            options.center =
                detail::fission_vec3(database, "fission_center");
            options.standard_deviation =
                detail::fission_vec3(
                    database, "fission_standard_deviation");
            break;
    }

    detail::validate_fission_power_options(options);
    return options;
}

/**
 * @brief Specialized non-negative heat source named qdot_fission.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class FissionPowerSource
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using field_type = CellField<Pack>;
    using context_type = BoussinesqUpdateContext<Pack>;
    using multiplier_type =
        std::function<scalar_type(const context_type&)>;

    static constexpr std::string_view field_name = "qdot_fission";

    /**
     * @brief Construct and reserve the qdot_fission temperature source.
     */
    FissionPowerSource(
        SP<const Mesh<Pack>> mesh,
        TemperatureSourceRegistry<Pack>& registry)
        : d_mesh(require_mesh(std::move(mesh))),
          d_base_profile(d_mesh, scalar_type{}, "qdot_fission_base"),
          d_registry(&registry),
          d_source(&registry.add_reserved(
              std::string(field_name), scalar_type{}))
    {
        try
        {
            d_source->set_updater(
                [this](const context_type& context, field_type& field)
                {
                    apply_time_multiplier(context, field);
                });
        }
        catch (...)
        {
            d_registry->remove_reserved(std::string(field_name));
            throw;
        }
    }

    ~FissionPowerSource()
    {
        if (d_registry != nullptr)
        {
            d_registry->remove_reserved(std::string(field_name));
        }
    }

    FissionPowerSource(const FissionPowerSource&) = delete;
    FissionPowerSource& operator=(const FissionPowerSource&) = delete;
    FissionPowerSource(FissionPowerSource&&) = delete;
    FissionPowerSource& operator=(FissionPowerSource&&) = delete;

    /**
     * @brief Configure from a disabled, constant, or Gaussian profile.
     */
    void configure(const FissionPowerSourceOptions& options)
    {
        detail::validate_fission_power_options(options);
        switch (options.profile)
        {
            case FissionPowerProfile::Disabled:
                initialize_constant(0.0);
                break;
            case FissionPowerProfile::Constant:
                initialize_constant(options.power_density);
                break;
            case FissionPowerProfile::Gaussian:
                initialize_gaussian(
                    options.total_power,
                    options.center,
                    options.standard_deviation);
                break;
        }
    }

    /**
     * @brief Set a uniform non-negative power density.
     */
    void initialize_constant(scalar_type power_density)
    {
        require_non_negative(power_density, "power density");
        d_base_profile.put_scalar(power_density);
        apply_base_profile(scalar_type{1});
    }

    /**
     * @brief Normalize a Gaussian shape to a distributed total power.
     */
    void initialize_gaussian(
        scalar_type total_power,
        const vec3<scalar_type>& center,
        const vec3<scalar_type>& standard_deviation)
    {
        require_non_negative(total_power, "total power");
        for (size_t component = 0; component < 3; ++component)
        {
            if (!std::isfinite(center.component(component)))
            {
                throw std::invalid_argument(
                    "Fission Gaussian center must be finite.");
            }
            const auto width =
                standard_deviation.component(component);
            if (!std::isfinite(width) || width <= scalar_type{})
            {
                throw std::invalid_argument(
                    "Fission Gaussian standard deviations must be finite "
                    "and positive.");
            }
        }

        scalar_type local_integral{};
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto centroid = d_mesh->cell_centroid(cell_lid);
            scalar_type exponent{};
            for (size_t component = 0; component < 3; ++component)
            {
                const auto scaled =
                    (centroid.component(component)
                     - center.component(component))
                  / standard_deviation.component(component);
                exponent += scaled * scaled;
            }
            local_integral +=
                std::exp(-scalar_type{0.5} * exponent)
              * d_mesh->cell_volume(cell_lid);
        }
        const auto profile_integral = global_sum(local_integral);
        if (total_power > scalar_type{}
            && (!std::isfinite(profile_integral)
                || profile_integral <= scalar_type{}))
        {
            throw std::invalid_argument(
                "Positive fission total power requires a positive profile "
                "integral.");
        }
        const auto scale =
            total_power == scalar_type{}
                ? scalar_type{}
                : checked_scale(total_power, profile_integral);

        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto centroid = d_mesh->cell_centroid(cell_lid);
            scalar_type exponent{};
            for (size_t component = 0; component < 3; ++component)
            {
                const auto scaled =
                    (centroid.component(component)
                     - center.component(component))
                  / standard_deviation.component(component);
                exponent += scaled * scaled;
            }
            const auto value =
                scale * std::exp(-scalar_type{0.5} * exponent);
            require_non_negative(value, "Gaussian field value");
            d_base_profile.set_owned_value(cell_lid, value);
        }
        d_base_profile.sync_ghosts();
        apply_base_profile(scalar_type{1});
    }

    /**
     * @brief Copy a non-negative power-density field into the source.
     */
    void initialize_from_power_density(const field_type& power_density)
    {
        require_same_mesh(power_density);
        copy_non_negative_field(power_density);
        apply_base_profile(scalar_type{1});
    }

    /**
     * @brief Normalize a non-negative shape field to a total power.
     */
    void initialize_from_shape(
        const field_type& shape,
        scalar_type total_power)
    {
        require_same_mesh(shape);
        require_non_negative(total_power, "total power");
        validate_non_negative_field(shape);
        if (total_power == scalar_type{})
        {
            initialize_constant(scalar_type{});
            return;
        }

        const auto profile_integral = integrate(shape);
        if (!std::isfinite(profile_integral)
            || profile_integral <= scalar_type{})
        {
            throw std::invalid_argument(
                "Positive fission total power requires a positive profile "
                "integral.");
        }
        copy_scaled_field(
            shape, checked_scale(total_power, profile_integral));
        apply_base_profile(scalar_type{1});
    }

    /**
     * @brief Install a non-negative time multiplier for the source field.
     */
    void set_time_multiplier(multiplier_type multiplier)
    {
        if (!multiplier)
        {
            throw std::invalid_argument(
                "Fission power time multiplier must be callable.");
        }
        d_time_multiplier = std::move(multiplier);
    }

    /**
     * @brief Remove any installed time multiplier.
     */
    void clear_time_multiplier() noexcept
    {
        d_time_multiplier = {};
    }

    /**
     * @brief Distributed integral of the currently applied power density.
     */
    scalar_type integrated_power() const
    {
        return integrate(d_source->field());
    }

    /**
     * @brief Applied qdot_fission field after the current time multiplier.
     */
    const field_type& field() const noexcept
    {
        return d_source->field();
    }

    /**
     * @brief Stored base profile before any time multiplier is applied.
     */
    const field_type& base_profile() const noexcept
    {
        return d_base_profile;
    }

private:
    static SP<const Mesh<Pack>> require_mesh(
        SP<const Mesh<Pack>> mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "FissionPowerSource requires a non-null mesh.");
        }
        return mesh;
    }

    static void require_non_negative(
        scalar_type value,
        const std::string& label)
    {
        if (!std::isfinite(value) || value < scalar_type{})
        {
            throw std::invalid_argument(
                "Fission " + label
                + " must be finite and non-negative.");
        }
    }

    void require_same_mesh(const field_type& field) const
    {
        if (&field.mesh() != d_mesh.get())
        {
            throw std::invalid_argument(
                "Fission power field must use the solver mesh.");
        }
    }

    void copy_non_negative_field(const field_type& field)
    {
        validate_non_negative_field(field);
        copy_scaled_field(field, scalar_type{1});
    }

    void validate_non_negative_field(const field_type& field) const
    {
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            require_non_negative(
                field.value(cell_lid), "field value");
        }
    }

    void copy_scaled_field(
        const field_type& field,
        scalar_type scale)
    {
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto value = field.value(cell_lid) * scale;
            require_non_negative(value, "normalized field value");
            d_base_profile.set_owned_value(cell_lid, value);
        }
        d_base_profile.sync_ghosts();
    }

    /** @brief Compute the globally reduced volume integral of a cell field. */
    scalar_type integrate(const field_type& field) const
    {
        scalar_type local_power{};
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            local_power +=
                field.value(cell_lid)
              * d_mesh->cell_volume(cell_lid);
        }

        return global_sum(local_power);
    }

    /** @brief Sum a rank-local scalar and replicate the result on every rank. */
    scalar_type global_sum(scalar_type local_value) const
    {
        scalar_type global_value{};
        const auto comm = d_mesh->owned_cell_map()->getComm();
        Teuchos::reduceAll(
            *comm,
            Teuchos::REDUCE_SUM,
            1,
            &local_value,
            &global_value);
        return global_value;
    }

    static scalar_type checked_scale(
        scalar_type total_power,
        scalar_type profile_integral)
    {
        const auto scale = total_power / profile_integral;
        if (!std::isfinite(scale))
        {
            throw std::invalid_argument(
                "Fission profile normalization produced a non-finite scale.");
        }
        return scale;
    }

    void apply_base_profile(scalar_type multiplier)
    {
        auto& applied = d_source->field();
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            applied.set_owned_value(
                cell_lid,
                d_base_profile.value(cell_lid) * multiplier);
        }
        applied.sync_ghosts();
    }

    void apply_time_multiplier(
        const context_type& context,
        field_type& field)
    {
        scalar_type multiplier = scalar_type{1};
        if (d_time_multiplier)
        {
            multiplier = d_time_multiplier(context);
        }
        require_non_negative(multiplier, "time multiplier");
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            field.set_owned_value(
                cell_lid,
                d_base_profile.value(cell_lid) * multiplier);
        }
    }

    SP<const Mesh<Pack>> d_mesh;
    field_type d_base_profile;
    TemperatureSourceRegistry<Pack>* d_registry; ///< Non-owning source registry.
    VolumetricScalarSource<Pack>* d_source; ///< Non-owning registry entry.
    multiplier_type d_time_multiplier;
};

} // namespace SimpleFluid
