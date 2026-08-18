/**
 * @file BoussinesqModel.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Physical material fields and volumetric heat-source infrastructure.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "dataclass/Database.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/CellField.hh"
#include "fields/MeshFieldTraits.hh"
#include "fields/VectorCellField.hh"

#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Forward declaration of the specialized fission heat source.
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>> class FissionPowerSource;

/**
 * @brief Uniform initial values for the physical Boussinesq material fields.
 *
 * Units are kg/m^3, J/(kg K), Pa s, and W/(m K), respectively.
 */
struct BoussinesqModelOptions
{
    real_t reference_density = 1.0; ///< Reference density used by momentum scaling.
    real_t density = 1.0;
    real_t specific_heat_capacity = 1.0;
    std::optional<real_t> dynamic_viscosity;
    std::optional<real_t> thermal_conductivity;
    bool density_feedback_enabled = false; ///< Whether density already includes buoyancy feedback.
    ArrString temperature_source_names;
    ArrReal temperature_source_power_densities;

    /** @brief Derive physical material defaults from legacy kinematic inputs. */
    static BoussinesqModelOptions legacy_defaults(const TimeStepperOptions& time_options)
    {
        BoussinesqModelOptions options;
        options.dynamic_viscosity = options.reference_density * time_options.kinematic_viscosity;
        options.thermal_conductivity =
            options.reference_density * options.specific_heat_capacity * time_options.thermal_diffusivity;
        return options;
    }
};

namespace detail
{

/**
 * @brief Read an optional Boussinesq database value.
 * @tparam T Requested database value type.
 * @param database Source database.
 * @param key Option key.
 * @param fallback Value used when @p key is absent.
 * @return Parsed value or @p fallback.
 * @throws std::invalid_argument if the stored value has the wrong type.
 */
template<class T> T database_value_or(const Database& database, const std::string& key, T fallback)
{
    if (!database.contains(key))
    {
        return fallback;
    }

    try
    {
        return database.get<T>(key);
    }
    catch (const std::out_of_range&)
    {
        throw std::invalid_argument("Boussinesq model option '" + key + "' has the wrong type.");
    }
}

/**
 * @brief Require a finite model option.
 * @param value Value to validate.
 * @param name Option name used in diagnostics.
 * @throws std::invalid_argument if @p value is not finite.
 */
inline void require_finite(real_t value, const std::string& name)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument("Boussinesq model option '" + name + "' must be finite.");
    }
}

/**
 * @brief Validate material defaults and configured temperature sources.
 * @param options Boussinesq material and source options.
 * @param time_options Time-stepper values used for legacy defaults.
 * @throws std::invalid_argument if an option is physically inconsistent.
 */
inline void validate_model_options(const BoussinesqModelOptions& options, const TimeStepperOptions& time_options)
{
    require_finite(options.reference_density, "reference_density");
    require_finite(options.density, "density");
    require_finite(options.specific_heat_capacity, "specific_heat_capacity");
    if (options.reference_density <= 0.0)
    {
        throw std::invalid_argument("Boussinesq model option 'reference_density' must be positive.");
    }
    if (options.density <= 0.0)
    {
        throw std::invalid_argument("Boussinesq model option 'density' must be positive.");
    }
    if (options.specific_heat_capacity <= 0.0)
    {
        throw std::invalid_argument("Boussinesq model option 'specific_heat_capacity' must be positive.");
    }

    const auto dynamic_viscosity =
        options.dynamic_viscosity.value_or(options.reference_density * time_options.kinematic_viscosity);
    const auto thermal_conductivity = options.thermal_conductivity.value_or(
        options.reference_density * options.specific_heat_capacity * time_options.thermal_diffusivity);
    require_finite(dynamic_viscosity, "dynamic_viscosity");
    require_finite(thermal_conductivity, "thermal_conductivity");
    if (dynamic_viscosity < 0.0)
    {
        throw std::invalid_argument("Boussinesq model option 'dynamic_viscosity' cannot be negative.");
    }
    if (thermal_conductivity < 0.0)
    {
        throw std::invalid_argument("Boussinesq model option 'thermal_conductivity' cannot be negative.");
    }

    if (options.temperature_source_names.size() != options.temperature_source_power_densities.size())
    {
        throw std::invalid_argument("Boussinesq model temperature source names and power densities "
                                    "must have the same length.");
    }

    std::set<std::string> names;
    for (size_t index = 0; index < options.temperature_source_names.size(); ++index)
    {
        const auto& name = options.temperature_source_names[index];
        if (name.empty())
        {
            throw std::invalid_argument("Boussinesq model temperature source names cannot be empty.");
        }
        if (!names.insert(name).second)
        {
            throw std::invalid_argument("Duplicate Boussinesq temperature source name '" + name + "'.");
        }
        require_finite(options.temperature_source_power_densities[index], "temperature_source_power_densities");
    }
}

} // namespace detail

/**
 * @brief Parse and validate physical Boussinesq options from a flat database.
 * @param database Source configuration database.
 * @param time_options Time-stepper values used for legacy defaults.
 * @return Validated Boussinesq model options.
 * @throws std::invalid_argument if an option is ill-typed or invalid.
 */
inline BoussinesqModelOptions boussinesq_model_options_from_database(
    const Database& database, const TimeStepperOptions& time_options)
{
    auto options = BoussinesqModelOptions::legacy_defaults(time_options);
    options.reference_density =
        detail::database_value_or<real_t>(database, "reference_density", options.reference_density);
    options.density = detail::database_value_or<real_t>(database, "density", options.reference_density);
    options.specific_heat_capacity =
        detail::database_value_or<real_t>(database, "specific_heat_capacity", options.specific_heat_capacity);
    if (database.contains("dynamic_viscosity"))
    {
        options.dynamic_viscosity = detail::database_value_or<real_t>(database, "dynamic_viscosity", 0.0);
    }
    else
    {
        options.dynamic_viscosity = options.reference_density * time_options.kinematic_viscosity;
    }
    if (database.contains("thermal_conductivity"))
    {
        options.thermal_conductivity = detail::database_value_or<real_t>(database, "thermal_conductivity", 0.0);
    }
    else
    {
        options.thermal_conductivity =
            options.reference_density * options.specific_heat_capacity * time_options.thermal_diffusivity;
    }
    options.density_feedback_enabled = detail::database_value_or<bool>(database, "density_feedback_enabled", false);
    options.temperature_source_names = detail::database_value_or<ArrString>(database, "temperature_source_names", {});
    options.temperature_source_power_densities =
        detail::database_value_or<ArrReal>(database, "temperature_source_power_densities", {});

    detail::validate_model_options(options, time_options);
    return options;
}

/**
 * @brief Forward declaration of updateable material-property fields.
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>> struct MaterialPropertyFields;

template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>> class TemperatureSourceRegistry;

/**
 * @brief Read-only physical state passed to material and source updaters.
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack, class MeshType = Mesh<Pack>> struct BoussinesqUpdateContext
{
    using scalar_type = typename Pack::scalar_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using velocity_field_type = typename field_traits::vector_cell_type;

    scalar_type time{};
    int step_index{};
    const mesh_type& mesh;
    const field_type& temperature;
    const field_type& pressure; ///< Gauge pressure in Pa.
    const velocity_field_type& velocity;
};

/**
 * @brief Initialize a scalar cell field from a centroid-based provider.
 * @tparam Pack Tpetra type pack used by the field.
 * @tparam Provider Callable returning a value from a cell centroid.
 * @tparam Validator Callable validating each initialized value.
 * @param[out] field Field to initialize and synchronize.
 * @param provider Centroid-based value provider.
 * @param validator Per-value validator.
 * @param label Field label used in diagnostics.
 * @throws std::invalid_argument if a provider result is not finite.
 */
template<class Field, class Provider, class Validator>
    requires std::invocable<Provider, const typename Field::mesh_type::Vec3&>
void initialize_cell_field(Field& field, Provider&& provider, Validator&& validator, const std::string& label)
{
    const auto& mesh = field.mesh();
    collective_detail::collective_local_validation(mesh, label + " initialization",
        [&]
        {
            for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<typename Field::local_ordinal_type>(owned);
                const auto value = std::invoke(provider, mesh.cell_centroid(cell_lid));
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument(label + " initializer produced a non-finite value.");
                }
                std::invoke(validator, value, label);
                field.set_owned_value(cell_lid, value);
            }
        });
    field.sync_ghosts();
}

/**
 * @brief Initialize a scalar cell field to one validated value.
 * @tparam Pack Tpetra type pack used by the field.
 * @tparam Validator Callable validating the initial value.
 * @param[out] field Field to initialize and synchronize.
 * @param value Uniform initial value.
 * @param validator Per-value validator.
 * @param label Field label used in diagnostics.
 * @throws std::invalid_argument if @p value is not finite.
 */
template<class Field, class Validator>
void initialize_cell_field(
    Field& field, typename Field::scalar_type value, Validator&& validator, const std::string& label)
{
    initialize_cell_field(field, [value](const auto&) { return value; }, std::forward<Validator>(validator), label);
}

/**
 * @brief Named physical heat-source field in W/m^3.
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>> class VolumetricScalarSource
{
public:
    using scalar_type = typename Pack::scalar_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using context_type = BoussinesqUpdateContext<Pack, mesh_type>;
    using updater_type = std::function<void(const context_type&, field_type&)>;

    VolumetricScalarSource(SP<const mesh_type> mesh, std::string name, scalar_type initial_value = {})
        : d_name(require_name(std::move(name))), d_field(std::move(mesh), initial_value, d_name)
    {
        validate();
    }

    const std::string& name() const noexcept { return d_name; }
    field_type& field() noexcept { return d_field; }
    const field_type& field() const noexcept { return d_field; }
    bool enabled() const noexcept { return d_enabled; }
    void set_enabled(bool enabled) noexcept { d_enabled = enabled; }

    /**
     * @brief Install rank-local source update work.
     * @note The callback must not perform communication; failures are reduced
     *       collectively after it returns or throws.
     */
    void set_updater(updater_type updater)
    {
        if (!updater)
        {
            throw std::invalid_argument("VolumetricScalarSource updater must be callable.");
        }
        d_updater = std::move(updater);
        d_clear_updater_after_update = false;
    }

    void clear_updater() noexcept
    {
        d_updater = {};
        d_clear_updater_after_update = false;
    }

    /** @brief Refresh and validate this source directly on every rank. */
    void update(const context_type& context)
    {
        collective_detail::collective_local_validation(d_field.mesh(), "Temperature source '" + d_name + "' update",
            [&]
            {
                if (d_updater)
                {
                    d_updater(context, d_field);
                }
                validate();
            });
        sync_after_update();
    }

    template<class Provider> void initialize(Provider&& provider)
    {
        initialize_cell_field(
            d_field, std::forward<Provider>(provider), [](scalar_type, const std::string&) {},
            "Temperature source '" + d_name + "'");
    }

    void initialize(scalar_type value)
    {
        initialize_cell_field(
            d_field, value, [](scalar_type, const std::string&) {}, "Temperature source '" + d_name + "'");
    }

    void validate() const
    {
        for (size_t owned = 0; owned < d_field.num_owned_cells(); ++owned)
        {
            const auto value = d_field.value(static_cast<typename Pack::local_ordinal_type>(owned));
            if (!std::isfinite(value))
            {
                throw std::invalid_argument("Temperature source '" + d_name + "' contains a non-finite value.");
            }
        }
    }

private:
    template<TpetraTypePack, class> friend class TemperatureSourceRegistry;

    template<TpetraTypePack, class> friend class FissionPowerSource;

    bool requires_dynamic_update() const noexcept { return d_enabled && static_cast<bool>(d_updater); }

    void update_local(const context_type& context)
    {
        d_updater(context, d_field);
        validate();
    }

    void sync_after_update()
    {
        d_field.sync_ghosts();
        if (d_clear_updater_after_update)
        {
            clear_updater();
        }
    }

    void clear_updater_after_next_update() noexcept { d_clear_updater_after_update = static_cast<bool>(d_updater); }

    static std::string require_name(std::string name)
    {
        if (name.empty())
        {
            throw std::invalid_argument("VolumetricScalarSource requires a non-empty name.");
        }
        return name;
    }

    std::string d_name;
    field_type d_field;
    bool d_enabled = true;
    updater_type d_updater;
    bool d_clear_updater_after_update = false;
};

/**
 * @brief Deterministic registry of named temperature heat-source fields.
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack, class MeshType> class TemperatureSourceRegistry
{
public:
    using mesh_type = MeshType;
    using source_type = VolumetricScalarSource<Pack, mesh_type>;
    using scalar_type = typename Pack::scalar_type;
    using context_type = BoussinesqUpdateContext<Pack, mesh_type>;

    explicit TemperatureSourceRegistry(SP<const mesh_type> mesh) : d_mesh(std::move(mesh))
    {
        if (!d_mesh)
        {
            throw std::invalid_argument("TemperatureSourceRegistry requires a non-null mesh.");
        }
    }

    /** @note Invoke source-registry mutations consistently on every mesh rank. */
    source_type& add(std::string name, scalar_type initial_value = {})
    {
        if (is_reserved_name(name))
        {
            throw std::invalid_argument(
                "Temperature source name '" + name + "' collides with a solution or material field.");
        }
        if (d_sources.contains(name))
        {
            throw std::invalid_argument("Temperature source '" + name + "' already exists.");
        }

        auto source = std::make_unique<source_type>(d_mesh, name, initial_value);
        auto [iter, inserted] = d_sources.emplace(std::move(name), std::move(source));
        (void) inserted;
        d_schema_dirty = true;
        return *iter->second;
    }

    /** @note Invoke source-registry mutations consistently on every mesh rank. */
    bool remove(const std::string& name)
    {
        if (name == "qdot_fission")
        {
            throw std::invalid_argument("Temperature source name '" + name + "' is reserved for a specialized model.");
        }
        const bool removed = d_sources.erase(name) > 0;
        d_schema_dirty = d_schema_dirty || removed;
        return removed;
    }

    source_type* find(const std::string& name) noexcept
    {
        const auto iter = d_sources.find(name);
        return iter == d_sources.end() ? nullptr : iter->second.get();
    }

    const source_type* find(const std::string& name) const noexcept
    {
        const auto iter = d_sources.find(name);
        return iter == d_sources.end() ? nullptr : iter->second.get();
    }

    source_type& at(const std::string& name)
    {
        auto* result = find(name);
        if (!result)
        {
            throw std::out_of_range("Temperature source '" + name + "' does not exist.");
        }
        return *result;
    }

    const source_type& at(const std::string& name) const
    {
        const auto* result = find(name);
        if (!result)
        {
            throw std::out_of_range("Temperature source '" + name + "' does not exist.");
        }
        return *result;
    }

    const auto& entries() const noexcept { return d_sources; }

    /**
     * @brief Refresh enabled dynamic sources with one collective validation.
     *
     * Disabled and updater-free fields remain unchanged and require no ghost
     * import. Source count, ordered names, and active state must agree on every
     * mesh rank.
     */
    void update(const context_type& context)
    {
        validate_collective_schema();

        const auto globally_active = collective_detail::collective_local_validation_batch(
            *d_mesh, "Temperature source registry update", d_sources,
            [](const auto& entry) { return entry.second->requires_dynamic_update(); },
            [&](auto& entry) { entry.second->update_local(context); }, d_update_validation_scratch);

        size_t source_index = 0;
        for (auto& [name, source] : d_sources)
        {
            (void) name;
            if (globally_active[source_index])
            {
                source->sync_after_update();
            }
            ++source_index;
        }
    }

    scalar_type total_power_density(typename Pack::local_ordinal_type cell_lid) const
    {
        scalar_type total{};
        for (const auto& [name, source] : d_sources)
        {
            (void) name;
            if (source->enabled())
            {
                total += source->field().value(cell_lid);
            }
        }
        return total;
    }

private:
    template<TpetraTypePack, class> friend class FissionPowerSource;

    source_type& add_reserved(std::string name, scalar_type initial_value = {})
    {
        if (!is_reserved_name(name))
        {
            throw std::invalid_argument("Specialized temperature source name '" + name + "' is not reserved.");
        }
        if (d_sources.contains(name))
        {
            throw std::invalid_argument("Temperature source '" + name + "' already exists.");
        }

        auto source = std::make_unique<source_type>(d_mesh, name, initial_value);
        auto [iter, inserted] = d_sources.emplace(std::move(name), std::move(source));
        (void) inserted;
        d_schema_dirty = true;
        return *iter->second;
    }

    void remove_reserved(const std::string& name) noexcept
    {
        d_schema_dirty = d_sources.erase(name) > 0 || d_schema_dirty;
    }

    /** @brief Fail coherently unless every rank has the same ordered names. */
    void validate_collective_schema()
    {
        const auto communicator = d_mesh->owned_cell_map()->getComm();
        const int local_schema_dirty = d_schema_dirty ? 1 : 0;
        int any_schema_dirty = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_schema_dirty, &any_schema_dirty);
        if (any_schema_dirty == 0)
        {
            return;
        }

        const int local_count_is_valid =
            d_sources.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) ? 1 : 0;
        int global_count_is_valid = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_count_is_valid, &global_count_is_valid);
        if (global_count_is_valid == 0)
        {
            throw std::invalid_argument("Temperature source registry has too many entries for "
                                        "collective validation.");
        }

        const auto local_count = static_cast<int>(d_sources.size());
        int minimum_count = 0;
        int maximum_count = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_count, &minimum_count);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_count, &maximum_count);
        if (minimum_count != maximum_count)
        {
            throw std::invalid_argument("Temperature source registry requires the same source "
                                        "count on every rank.");
        }

        // Length-prefix every map key so boundaries and embedded bytes are
        // compared exactly rather than through a collision-prone hash.
        std::vector<int> local_name_encoding;
        for (const auto& [name, source] : d_sources)
        {
            (void) source;
            const auto name_size = static_cast<std::uint64_t>(name.size());
            for (int shift = 56; shift >= 0; shift -= 8)
            {
                local_name_encoding.push_back(static_cast<int>((name_size >> shift) & 0xffU));
            }
            for (const unsigned char byte : name)
            {
                local_name_encoding.push_back(static_cast<int>(byte));
            }
        }

        const int local_name_size_is_valid =
            local_name_encoding.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) ? 1 : 0;
        int global_name_size_is_valid = 0;
        Teuchos::reduceAll(
            *communicator, Teuchos::REDUCE_MIN, 1, &local_name_size_is_valid, &global_name_size_is_valid);
        if (global_name_size_is_valid == 0)
        {
            throw std::invalid_argument("Temperature source registry names are too large for "
                                        "collective validation.");
        }

        const auto local_name_size = static_cast<int>(local_name_encoding.size());
        int minimum_name_size = 0;
        int maximum_name_size = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_name_size, &minimum_name_size);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_name_size, &maximum_name_size);
        if (minimum_name_size != maximum_name_size)
        {
            throw std::invalid_argument("Temperature source registry requires identical ordered "
                                        "source names on every rank.");
        }
        if (local_name_encoding.empty())
        {
            d_schema_dirty = false;
            return;
        }

        std::vector<int> minimum_name_encoding(local_name_encoding.size(), 0);
        std::vector<int> maximum_name_encoding(local_name_encoding.size(), 0);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, local_name_size, local_name_encoding.data(),
            minimum_name_encoding.data());
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, local_name_size, local_name_encoding.data(),
            maximum_name_encoding.data());
        if (minimum_name_encoding != maximum_name_encoding)
        {
            throw std::invalid_argument("Temperature source registry requires identical ordered "
                                        "source names on every rank.");
        }
        d_schema_dirty = false;
    }

    static bool is_reserved_name(const std::string& name)
    {
        static const std::set<std::string> reserved{"temperature", "pressure", "velocity", "density",
            "specific_heat_capacity", "dynamic_viscosity", "thermal_conductivity", "k", "epsilon", "omega", "nu_t",
            "mu_eff", "lambda_eff", "wall_distance", "wall_y_plus", "buoyancy_production", "qdot_fission"};
        return reserved.contains(name);
    }

    SP<const mesh_type> d_mesh;
    std::map<std::string, std::unique_ptr<source_type>> d_sources;
    bool d_schema_dirty = true;
    collective_detail::CollectiveValidationBatchScratch d_update_validation_scratch;
};

/**
 * @brief Updateable physical material-property fields.
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack, class MeshType> struct MaterialPropertyFields
{
    using scalar_type = typename Pack::scalar_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using context_type = BoussinesqUpdateContext<Pack, mesh_type>;
    using updater_type = std::function<void(const context_type&, MaterialPropertyFields&)>;

    MaterialPropertyFields(
        SP<const mesh_type> mesh, const BoussinesqModelOptions& options, const TimeStepperOptions& time_options)
        : density(mesh, options.density, "density"),
          specific_heat_capacity(mesh, options.specific_heat_capacity, "specific_heat_capacity"),
          dynamic_viscosity(mesh,
              options.dynamic_viscosity.value_or(options.reference_density * time_options.kinematic_viscosity),
              "dynamic_viscosity"),
          thermal_conductivity(std::move(mesh),
              options.thermal_conductivity.value_or(
                  options.reference_density * options.specific_heat_capacity * time_options.thermal_diffusivity),
              "thermal_conductivity")
    {
        validate_and_sync();
    }

    field_type density;
    field_type specific_heat_capacity;
    field_type dynamic_viscosity;
    field_type thermal_conductivity;

    template<class Provider> void initialize_density(Provider&& provider)
    {
        initialize_positive(density, std::forward<Provider>(provider), "density");
    }

    void initialize_density(scalar_type value)
    {
        initialize_positive(density, [value](const auto&) { return value; }, "density");
    }

    template<class Provider> void initialize_specific_heat_capacity(Provider&& provider)
    {
        initialize_positive(specific_heat_capacity, std::forward<Provider>(provider), "specific_heat_capacity");
    }

    void initialize_specific_heat_capacity(scalar_type value)
    {
        initialize_positive(specific_heat_capacity, [value](const auto&) { return value; }, "specific_heat_capacity");
    }

    template<class Provider> void initialize_dynamic_viscosity(Provider&& provider)
    {
        initialize_non_negative(dynamic_viscosity, std::forward<Provider>(provider), "dynamic_viscosity");
    }

    void initialize_dynamic_viscosity(scalar_type value)
    {
        initialize_non_negative(dynamic_viscosity, [value](const auto&) { return value; }, "dynamic_viscosity");
    }

    template<class Provider> void initialize_thermal_conductivity(Provider&& provider)
    {
        initialize_non_negative(thermal_conductivity, std::forward<Provider>(provider), "thermal_conductivity");
    }

    void initialize_thermal_conductivity(scalar_type value)
    {
        initialize_non_negative(thermal_conductivity, [value](const auto&) { return value; }, "thermal_conductivity");
    }

    /**
     * @brief Install rank-local material-property update work.
     * @note The callback must not perform communication; failures are reduced
     *       collectively after it returns or throws.
     */
    void set_updater(updater_type value)
    {
        if (!value)
        {
            throw std::invalid_argument("Material property updater must be callable.");
        }
        updater = std::move(value);
    }

    void clear_updater() noexcept { updater = {}; }

    /**
     * @brief Run a dynamic updater, validate bounds, and synchronize ghosts.
     *
     * Static fields are already validated and synchronized by construction or
     * explicit initialization, so this is a no-op without an updater.
     */
    void update(const context_type& context)
    {
        if (!updater)
        {
            return;
        }
        collective_detail::collective_local_validation(density.mesh(), "Material property update",
            [&]
            {
                if (updater)
                {
                    updater(context, *this);
                }
                validate_local();
            });
        sync_ghosts();
    }

    /** @brief Validate physical bounds and synchronize all material fields. */
    void validate_and_sync()
    {
        collective_detail::collective_local_validation(
            density.mesh(), "Material property validation", [&] { validate_local(); });
        sync_ghosts();
    }

private:
    void validate_local() const
    {
        validate_positive(density, "density");
        validate_positive(specific_heat_capacity, "specific_heat_capacity");
        validate_non_negative(dynamic_viscosity, "dynamic_viscosity");
        validate_non_negative(thermal_conductivity, "thermal_conductivity");
    }

    void sync_ghosts()
    {
        density.sync_ghosts();
        specific_heat_capacity.sync_ghosts();
        dynamic_viscosity.sync_ghosts();
        thermal_conductivity.sync_ghosts();
    }

    template<class Provider>
    static void initialize_positive(field_type& field, Provider&& provider, const std::string& name)
    {
        initialize_cell_field(
            field, std::forward<Provider>(provider),
            [](scalar_type value, const std::string& label)
            {
                if (value <= scalar_type{})
                {
                    throw std::invalid_argument("Material field '" + label + "' must be positive.");
                }
            },
            name);
    }

    template<class Provider>
    static void initialize_non_negative(field_type& field, Provider&& provider, const std::string& name)
    {
        initialize_cell_field(
            field, std::forward<Provider>(provider),
            [](scalar_type value, const std::string& label)
            {
                if (value < scalar_type{})
                {
                    throw std::invalid_argument("Material field '" + label + "' cannot be negative.");
                }
            },
            name);
    }

    static void validate_positive(const field_type& field, const std::string& name)
    {
        for (size_t owned = 0; owned < field.num_owned_cells(); ++owned)
        {
            const auto value = field.value(static_cast<typename Pack::local_ordinal_type>(owned));
            if (!std::isfinite(value) || value <= scalar_type{})
            {
                throw std::invalid_argument("Material field '" + name + "' must contain finite positive values.");
            }
        }
    }

    static void validate_non_negative(const field_type& field, const std::string& name)
    {
        for (size_t owned = 0; owned < field.num_owned_cells(); ++owned)
        {
            const auto value = field.value(static_cast<typename Pack::local_ordinal_type>(owned));
            if (!std::isfinite(value) || value < scalar_type{})
            {
                throw std::invalid_argument("Material field '" + name + "' must contain finite non-negative values.");
            }
        }
    }

    updater_type updater;
};

/** @brief Selects optional field groups written with the primary solution. */
struct SolutionOutputOptions
{
    bool include_sources = false;
    bool include_material_properties = false;
    bool include_radiolytic_gas_fields = false;
    bool include_precursor_fields = false;
    bool include_turbulence_fields = false;
};

} // namespace SimpleFluid
