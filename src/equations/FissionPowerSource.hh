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

#include "SimpleFluidExport.hh"

#include "dataclass/Database.hh"
#include "dataclass/DatabaseOptionReader.hh"
#include "equations/BoussinesqModel.hh"
#include "fields/MeshFieldTraits.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <vector>
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
    const DatabaseOptionReader& reader,
    const std::string& key)
{
    const auto values = reader.required<ArrReal>(key);
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
    const detail::DatabaseOptionReader reader(
        database, "Fission power model");
    if (!reader.contains("fission_power_mode"))
    {
        return options;
    }

    options.profile = fission_power_profile_from_string(
        reader.required<std::string>("fission_power_mode"));
    switch (options.profile)
    {
        case FissionPowerProfile::Disabled:
            break;
        case FissionPowerProfile::Constant:
            options.power_density =
                reader.required<real_t>("fission_power_density");
            break;
        case FissionPowerProfile::Gaussian:
            options.total_power =
                reader.required<real_t>("fission_total_power");
            options.center =
                detail::fission_vec3(reader, "fission_center");
            options.standard_deviation =
                detail::fission_vec3(
                    reader, "fission_standard_deviation");
            break;
    }

    detail::validate_fission_power_options(options);
    return options;
}

/**
 * @brief Specialized non-negative heat source named qdot_fission.
 */
template<TpetraTypePack Pack, class MeshType>
class SIMPLEFLUID_EQUATIONS_EXPORT FissionPowerSource
{
public:
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using field_type = typename field_traits::scalar_cell_type;
    using context_type = BoussinesqUpdateContext<Pack, mesh_type>;
    using source_registry_type = TemperatureSourceRegistry<Pack, mesh_type>;
    using source_type = VolumetricScalarSource<Pack, mesh_type>;
    using multiplier_type =
        std::function<scalar_type(const context_type&)>;

    static constexpr std::string_view field_name = "qdot_fission";

    /**
     * @brief Construct and reserve the qdot_fission temperature source.
     */
    FissionPowerSource(
        SP<const mesh_type> mesh,
        source_registry_type& registry)
        : d_mesh(require_mesh(std::move(mesh))),
          d_base_profile(d_mesh, scalar_type{}, "qdot_fission_base"),
          d_registry(&registry),
          d_source(&registry.add_reserved(
              std::string(field_name), scalar_type{}))
    {
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
    void configure(const FissionPowerSourceOptions& options);

    /**
     * @brief Set a uniform non-negative power density.
     */
    void initialize_constant(scalar_type power_density);

    /**
     * @brief Normalize a Gaussian shape to a distributed total power.
     */
    void initialize_gaussian(
        scalar_type total_power,
        const vec3<scalar_type>& center,
        const vec3<scalar_type>& standard_deviation);

    /**
     * @brief Copy a non-negative power-density field into the source.
     */
    void initialize_from_power_density(const field_type& power_density);

    /**
     * @brief Normalize a non-negative shape field to a total power.
     */
    void initialize_from_shape(
        const field_type& shape,
        scalar_type total_power);

    /** Opaque source state; callbacks are deliberately excluded. */
    class StateSnapshot
    {
        friend class FissionPowerSource;
        const FissionPowerSource* owner = nullptr;
        std::vector<scalar_type> base, applied, interval_energy;
        scalar_type interval_start = {}, interval_duration = {};
        bool enabled = true;
    };

    /**
     * Set extensive joules per owned cell over [start, start + duration].
     * Stable cell IDs follow the moving material mesh. The interval is replayable:
     * refresh_interval_energy() never consumes energy or advances a ledger.
     * Collective; values must be finite/nonnegative, duration positive, and no
     * user time multiplier may be installed. Time endpoints and substeps must
     * advance representably, with clock rounding within 32 epsilon times the
     * interval duration; underresolved time intervals fail before source mutation.
     */
    void set_interval_energy(std::span<const scalar_type> owned_energy,
        scalar_type start, scalar_type duration);
    /** Recompute W/m^3 using current geometry; validate the requested subinterval. */
    void refresh_interval_energy(scalar_type time, scalar_type time_step);
    bool has_interval_energy() const noexcept { return d_interval_duration > scalar_type{}; }
    scalar_type interval_duration() const noexcept { return d_interval_duration; }
    scalar_type interval_end_time() const noexcept { return d_interval_start + d_interval_duration; }
    StateSnapshot snapshot() const;
    void restore(const StateSnapshot& snapshot);

    /**
     * @brief Install a non-negative time multiplier for the source field.
     * @note The callback must perform rank-local work only; callback failures
     *       are propagated collectively by the source registry update.
     */
    void set_time_multiplier(multiplier_type multiplier);

    /**
     * @brief Remove any installed time multiplier.
     */
    void clear_time_multiplier() noexcept
    {
        if (!d_time_multiplier)
        {
            return;
        }
        d_time_multiplier = {};
        d_source->clear_updater_after_next_update();
    }

    /**
     * @brief Distributed integral of the currently applied power density.
     */
    scalar_type integrated_power() const;

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
    /** @brief Require one scalar configuration value on every mesh rank. */
    SIMPLEFLUID_EQUATIONS_LOCAL
    void require_uniform_value(
        scalar_type local_value,
        const std::string& label) const;

    /** @brief Require one communicator-wide prescribed profile. */
    SIMPLEFLUID_EQUATIONS_LOCAL
    void require_uniform_configuration(
        const FissionPowerSourceOptions& options) const;

    static SP<const mesh_type> require_mesh(
        SP<const mesh_type> mesh)
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

    SIMPLEFLUID_EQUATIONS_LOCAL
    void copy_non_negative_field(const field_type& field);

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

    SIMPLEFLUID_EQUATIONS_LOCAL
    void copy_scaled_field(
        const field_type& field,
        scalar_type scale);

    /** @brief Compute the globally reduced volume integral of a cell field. */
    SIMPLEFLUID_EQUATIONS_LOCAL
    scalar_type integrate(const field_type& field) const;

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

    SIMPLEFLUID_EQUATIONS_LOCAL
    void apply_base_profile(scalar_type multiplier);

    SIMPLEFLUID_EQUATIONS_LOCAL
    void apply_time_multiplier(
        const context_type& context,
        field_type& field);

    SP<const mesh_type> d_mesh;
    field_type d_base_profile;
    source_registry_type* d_registry; ///< Non-owning source registry.
    source_type* d_source; ///< Non-owning registry entry.
    multiplier_type d_time_multiplier;
    std::vector<scalar_type> d_interval_energy;
    scalar_type d_interval_start = {};
    scalar_type d_interval_duration = {};
};
extern template class FissionPowerSource<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class FissionPowerSource<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;


} // namespace SimpleFluid
