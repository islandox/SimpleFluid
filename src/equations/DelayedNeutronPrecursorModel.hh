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

#include "SimpleFluidExport.hh"

#include "FVM/TransportSystem.hh"
#include "dataclass/Database.hh"
#include "dataclass/DatabaseOptionReader.hh"
#include "dataclass/typedefs.hh"
#include "fields/MeshFieldTraits.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <array>
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
 * @brief Globally reduced inventory balance for one precursor group and step.
 *
 * Every quantity is an integrated inventory rather than a rate.  A positive
 * boundary_outflow is inventory that left the domain during the step.
 */
template<class Scalar> struct PrecursorInventoryDiagnostics
{
    Scalar inventory_before{};
    Scalar source_added{};
    Scalar decay_removed{};
    Scalar positivity_adjustment{};
    Scalar boundary_outflow{};
    Scalar inventory_after{};
    Scalar balance_error{};
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
    for (const auto value : options.source_terms)
    {
        if (value < 0.0)
        {
            throw std::invalid_argument(
                "Precursor source terms cannot be negative.");
        }
    }
    for (const auto value : options.power_yields)
    {
        if (value < 0.0)
        {
            throw std::invalid_argument(
                "Precursor power yields cannot be negative.");
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
    const detail::DatabaseOptionReader reader(
        database, "Delayed-neutron precursor model");
    options.group_count = reader.value_or<int>(
        "precursor_group_count", options.group_count);
    options.decay_constants = reader.value_or<ArrReal>(
        "precursor_decay_constants", options.decay_constants);
    options.initial_concentrations = reader.value_or<ArrReal>(
        "precursor_initial_concentrations",
        options.initial_concentrations);
    options.source_terms = reader.value_or<ArrReal>(
        "precursor_source_terms", options.source_terms);
    options.power_yields = reader.value_or<ArrReal>(
        "precursor_power_yields", options.power_yields);
    options.effective_diffusivity = reader.value_or<real_t>(
        "precursor_effective_diffusivity",
        options.effective_diffusivity);
    validate_delayed_neutron_precursor_options(options);
    return options;
}

/**
 * @brief Conservative liquid-phase delayed-neutron precursor model.
 *
 * The persistent state is the mixture-volume inventory @f$\alpha_l C_i@f$.
 * Source and decay are integrated exactly over each step, then conservative
 * liquid advection and optional diffusion solve for @f$C_i@f$ with storage
 * and advection weight @f$\alpha_l@f$, diffusion coefficient
 * @f$\alpha_l D_i^{eff}@f$, and homogeneous zero-flux diffusion boundaries.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes,
         class MeshType = Mesh<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT DelayedNeutronPrecursorModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;
    using diagnostics_type = PrecursorInventoryDiagnostics<scalar_type>;

    /**
     * @brief Construct a precursor model on a mesh.
     * @note Construction validates options collectively on the mesh
     *       communicator.
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
     * @note Invoke collectively on every mesh rank.
     */
    void configure(const DelayedNeutronPrecursorOptions& options);

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
     * @brief Last globally reduced inventory balance for a precursor group.
     *
     * The returned values are replicated on every rank after each advance.
     */
    const diagnostics_type& last_inventory_diagnostics(size_t group) const
    {
        return d_last_inventory_diagnostics.at(group);
    }

    /**
     * @brief Initialize conserved inventories from the current liquid fraction.
     *
     * This is called by the coupled solver when the model is configured so
     * that configured concentrations correspond to the initial void state.
     * Direct model users may omit it; the first advance initializes lazily.
     */
    void initialize_inventory(const field_type& alpha_l);

    /**
     * @brief Advance reaction and diffusion without an advecting face flux.
     *
     * This compatibility overload retains the original public entry point.
     */
    void advance(
        scalar_type time_step,
        const field_type& alpha_l,
        const field_type* fission_power_density)
    {
        advance(time_step, alpha_l, fission_power_density, nullptr);
    }

    /**
     * @brief Advance conserved precursor inventories by one reaction/transport step.
     *
     * @param liquid_face_flux Optional owner-oriented volumetric liquid flux.
     *        When absent, the transport solve contains diffusion only.
     */
    void advance(
        scalar_type time_step,
        const field_type& alpha_l,
        const field_type* fission_power_density,
        const face_flux_field_type* liquid_face_flux);

private:
    /** @brief Validate options collectively before group-dependent allocation. */
    SIMPLEFLUID_EQUATIONS_LOCAL
    void validate_collective_configuration(
        const DelayedNeutronPrecursorOptions& options) const;

    /** @brief Require every rank to select the same collective update path. */
    SIMPLEFLUID_EQUATIONS_LOCAL
    void validate_collective_advance_selection(
        scalar_type time_step,
        const field_type* fission_power_density,
        const face_flux_field_type* liquid_face_flux) const;

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

    SIMPLEFLUID_EQUATIONS_LOCAL
    void validate_liquid_fraction_field(const field_type& alpha_l) const;

    SIMPLEFLUID_EQUATIONS_LOCAL
    void reconstruct_concentrations(const field_type& alpha_l);

    /** @brief Advect/diffuse concentrations while preserving liquid inventory. */
    SIMPLEFLUID_EQUATIONS_LOCAL
    void transport(
        scalar_type time_step,
        const field_type& alpha_l,
        const face_flux_field_type* liquid_face_flux);

    /**
     * @brief Finish the replicated step balance after transport.
     * @param reaction_inventory_reference Direct post-reaction inventories
     *        before positivity clipping.
     */
    SIMPLEFLUID_EQUATIONS_LOCAL
    void update_inventory_diagnostics(
        scalar_type time_step,
        const field_type& alpha_l,
        const face_flux_field_type* liquid_face_flux,
        const std::vector<scalar_type>& reaction_inventory_reference);

    /** @brief Sum a rank-local scalar and replicate it on every rank. */
    scalar_type global_sum(scalar_type local_value) const
    {
        scalar_type result{};
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(),
            Teuchos::REDUCE_SUM,
            1,
            &local_value,
            &result);
        return result;
    }

    /** @brief Replicate the maximum of a rank-local integer flag. */
    int global_max(int local_value) const
    {
        int result{};
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(),
            Teuchos::REDUCE_MAX,
            1,
            &local_value,
            &result);
        return result;
    }

    /** @brief Replicate the maximum of a rank-local scalar. */
    scalar_type global_maximum(scalar_type local_value) const
    {
        scalar_type result{};
        Teuchos::reduceAll(
            *d_mesh->owned_cell_map()->getComm(),
            Teuchos::REDUCE_MAX,
            1,
            &local_value,
            &result);
        return result;
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
    ArrReal d_initial_concentrations;
    ArrReal d_source_terms;
    ArrReal d_power_yields;
    std::vector<std::unique_ptr<field_type>> d_fields;
    std::vector<std::unique_ptr<field_type>> d_inventories;
    std::vector<std::unique_ptr<field_type>> d_sources;
    std::vector<diagnostics_type> d_last_inventory_diagnostics;
    std::map<std::string, const field_type*> d_output_fields;
    BelosLinearSolver<Pack> d_transport_solver;
    bool d_inventory_initialized = false;
};
extern template class DelayedNeutronPrecursorModel<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class DelayedNeutronPrecursorModel<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;


} // namespace SimpleFluid
