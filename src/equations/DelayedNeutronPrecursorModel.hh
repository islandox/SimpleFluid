/**
 * @file DelayedNeutronPrecursorModel.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Liquid-phase delayed-neutron precursor scalar model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoussinesqModel.hh"
#include "FVM/TransportSystem.hh"
#include "fields/MeshFieldTraits.hh"
#include "solvers/BelosLinearSolver.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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
    ArrReal power_yields; ///< Production per unit fission power density.
    real_t effective_diffusivity = 0.0; ///< Shared liquid-phase diffusivity.
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
 * @brief Conservative liquid-phase delayed-neutron precursor model.
 *
 * The persistent state is the mixture-volume inventory @f$\alpha_l C_i@f$.
 * Source and decay are integrated exactly over each step, then optional
 * diffusion solves for @f$C_i@f$ with coefficient
 * @f$\alpha_l D_i^{eff}@f$ and homogeneous zero-flux boundaries.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes,
         class MeshType = Mesh<Pack>>
class DelayedNeutronPrecursorModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;

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
        d_transport_geometry_cache.emplace(*d_mesh);
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
        d_inventories.clear();
        d_sources.clear();
        d_output_fields.clear();
        d_inventory_initialized = false;
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
            auto inventory = std::make_unique<field_type>(
                d_mesh, 0.0, "alpha_l_C_" + suffix);
            auto source = std::make_unique<field_type>(
                d_mesh, 0.0, "S_C_" + suffix);
            d_output_fields.emplace(field->name(), field.get());
            d_output_fields.emplace(source->name(), source.get());
            d_fields.push_back(std::move(field));
            d_inventories.push_back(std::move(inventory));
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
     * @brief Conserved liquid inventory @f$\alpha_l C_i@f$ for a group.
     */
    const field_type& liquid_inventory(size_t group) const
    {
        return *d_inventories.at(group);
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
     * @brief Initialize conserved inventories from the current liquid fraction.
     *
     * This is called by the coupled solver when the model is configured so
     * that configured concentrations correspond to the initial void state.
     * Direct model users may omit it; the first advance initializes lazily.
     */
    void initialize_inventory(const field_type& alpha_l)
    {
        validate_liquid_fraction_field(alpha_l);
        for (size_t group = 0; group < d_fields.size(); ++group)
        {
            auto& field = *d_fields[group];
            auto& inventory = *d_inventories[group];
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto liquid_fraction =
                    checked_liquid_fraction(alpha_l.value(cell_lid));
                inventory.set_owned_value(
                    cell_lid,
                    liquid_fraction * field.value(cell_lid));
            }
            inventory.sync_ghosts();
        }
        d_inventory_initialized = true;
    }

    /**
     * @brief Advance conserved precursor inventories by one reaction/diffusion step.
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
        validate_liquid_fraction_field(alpha_l);
        if (!d_inventory_initialized)
        {
            initialize_inventory(alpha_l);
        }

        for (size_t group = 0; group < d_fields.size(); ++group)
        {
            auto& inventory = *d_inventories[group];
            auto& source_field = *d_sources[group];
            const auto lambda = d_decay_constants[group];
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto source =
                    d_source_terms[group]
                  + (fission_power_density
                         ? d_power_yields[group]
                         * fission_power_density->value(cell_lid)
                         : scalar_type{});
                if (!std::isfinite(source))
                {
                    throw std::invalid_argument(
                        "Precursor source values must be finite.");
                }
                const auto old_inventory = inventory.value(cell_lid);
                scalar_type new_inventory{};
                if (lambda > scalar_type{})
                {
                    const auto decay = std::exp(-lambda * time_step);
                    const auto reacted_fraction =
                        -std::expm1(-lambda * time_step);
                    new_inventory =
                        old_inventory * decay
                      + source / lambda * reacted_fraction;
                }
                else
                {
                    new_inventory =
                        old_inventory + time_step * source;
                }
                new_inventory =
                    std::max(new_inventory, scalar_type{});
                inventory.set_owned_value(cell_lid, new_inventory);
                source_field.set_owned_value(cell_lid, source);
            }
            inventory.sync_ghosts();
            source_field.sync_ghosts();
        }

        if (d_options.effective_diffusivity > scalar_type{})
        {
            diffuse(time_step, alpha_l);
        }
        else
        {
            reconstruct_concentrations(alpha_l);
        }
    }

private:
    /** @brief Require a finite liquid fraction in the interval (0, 1]. */
    static scalar_type checked_liquid_fraction(scalar_type value)
    {
        if (!std::isfinite(value)
            || value <= scalar_type{}
            || value > scalar_type{1})
        {
            throw std::invalid_argument(
                "Precursor liquid fractions must be finite and in (0, 1].");
        }
        return value;
    }

    void validate_liquid_fraction_field(const field_type& alpha_l) const
    {
        if (&alpha_l.mesh() != d_mesh.get())
        {
            throw std::invalid_argument(
                "Precursor liquid fraction must be on the model mesh.");
        }
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            checked_liquid_fraction(alpha_l.value(
                static_cast<local_ordinal_type>(owned)));
        }
    }

    void reconstruct_concentrations(const field_type& alpha_l)
    {
        for (size_t group = 0; group < d_fields.size(); ++group)
        {
            auto& field = *d_fields[group];
            const auto& inventory = *d_inventories[group];
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto liquid_fraction =
                    checked_liquid_fraction(alpha_l.value(cell_lid));
                field.set_owned_value(
                    cell_lid,
                    inventory.value(cell_lid) / liquid_fraction);
            }
            field.sync_ghosts();
        }
    }

    /** @brief Diffuse group concentrations while preserving liquid inventory. */
    void diffuse(
        scalar_type time_step,
        const field_type& alpha_l)
    {
        face_flux_field_type zero_flux(
            d_mesh, 0.0, "precursor_zero_face_flux");
        field_type storage_weight(
            d_mesh, 1.0, "precursor_storage_weight");
        field_type advection_weight(
            d_mesh, 1.0, "precursor_advection_weight");
        field_type diffusion_weight(
            d_mesh, 0.0, "precursor_diffusion_weight");
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto liquid_fraction =
                checked_liquid_fraction(alpha_l.value(cell_lid));
            storage_weight.set_owned_value(cell_lid, liquid_fraction);
            advection_weight.set_owned_value(cell_lid, liquid_fraction);
            diffusion_weight.set_owned_value(
                cell_lid,
                liquid_fraction * d_options.effective_diffusivity);
        }
        storage_weight.sync_ghosts();
        advection_weight.sync_ghosts();
        diffusion_weight.sync_ghosts();

        auto boundary_condition =
            [](int, size_t)
        {
            return BoundaryCondition{
                BoundaryConditionType::Neumann, 0.0};
        };
        auto boundary_value =
            [](int, size_t) -> scalar_type
        {
            return scalar_type{};
        };
        auto zero_source =
            [](local_ordinal_type) -> scalar_type
        {
            return scalar_type{};
        };

        for (size_t group = 0; group < d_fields.size(); ++group)
        {
            auto& field = *d_fields[group];
            auto& inventory = *d_inventories[group];
            field_type old_concentration(
                d_mesh, "precursor_old_concentration");
            scalar_type concentration_scale{};
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                old_concentration.set_owned_value(
                    cell_lid,
                    inventory.value(cell_lid)
                  / storage_weight.value(cell_lid));
                concentration_scale = std::max(
                    concentration_scale,
                    std::abs(old_concentration.value(cell_lid)));
            }
            old_concentration.sync_ghosts();

            auto system = FVM::weighted_scalar_transport_system<Pack>(
                old_concentration,
                zero_flux,
                time_step,
                storage_weight,
                advection_weight,
                diffusion_weight,
                boundary_condition,
                boundary_value,
                zero_source,
                FVM::NonOrthogonalTreatment::Explicit,
                &old_concentration,
                Teuchos::null,
                {},
                {},
                nullptr,
                &*d_transport_geometry_cache);
            field_type solution(d_mesh, "precursor_diffusion_solution");
            const auto statistics =
                d_transport_solver.solve_with_statistics(
                    system.matrix,
                    *system.rhs,
                    solution.owned_data(),
                    LinearSolverOptions{});
            if (!statistics.converged)
            {
                throw std::runtime_error(
                    "Precursor diffusion solve did not converge.");
            }

            const auto negative_tolerance =
                scalar_type{100}
              * std::numeric_limits<scalar_type>::epsilon()
              * std::max(concentration_scale, scalar_type{1});
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                auto concentration = solution.value(cell_lid);
                if (!std::isfinite(concentration))
                {
                    throw std::runtime_error(
                        "Precursor diffusion produced a non-finite value.");
                }
                if (concentration < -negative_tolerance)
                {
                    throw std::runtime_error(
                        "Precursor diffusion produced a negative value.");
                }
                if (concentration < scalar_type{})
                {
                    concentration = scalar_type{};
                }
                field.set_owned_value(cell_lid, concentration);
                inventory.set_owned_value(
                    cell_lid,
                    storage_weight.value(cell_lid) * concentration);
            }
            field.sync_ghosts();
            inventory.sync_ghosts();
        }
    }

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
    std::optional<FVM::TransportGeometryCache<mesh_type>>
        d_transport_geometry_cache;
    DelayedNeutronPrecursorOptions d_options;
    ArrReal d_decay_constants;
    ArrReal d_source_terms;
    ArrReal d_power_yields;
    std::vector<std::unique_ptr<field_type>> d_fields;
    std::vector<std::unique_ptr<field_type>> d_inventories;
    std::vector<std::unique_ptr<field_type>> d_sources;
    std::map<std::string, const field_type*> d_output_fields;
    BelosLinearSolver<Pack> d_transport_solver;
    bool d_inventory_initialized = false;
};

} // namespace SimpleFluid
