/**
 * @file ScalarSourceDecayTransport.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Liquid-phase scalar source-decay scalar model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/TransportSystem.hh"
#include "equations/CollectiveValidation.hh"
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
 * @brief Runtime controls for scalar source-decay groups.
 */
struct ScalarSourceDecayOptions
{
    int group_count = 0;
    ArrReal decay_constants;
    ArrReal initial_concentrations;
    ArrReal source_terms; ///< Scalar amount/(bulk m3 s), independently supplied for each group.
    real_t effective_diffusivity = 0.0; ///< Shared liquid-phase diffusivity.
};

/**
 * @brief Globally reduced inventory balance for one scalar_group group and step.
 *
 * Every quantity is an integrated inventory rather than a rate.  A positive
 * boundary_outflow is inventory that left the domain during the step.
 */
template<class Scalar> struct ScalarInventoryDiagnostics
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
 * @brief Validate scalar_group group counts, vectors, and transport constants.
 *
 * @param options Candidate scalar_group configuration.
 */
inline void validate_scalar_source_decay_options(
    const ScalarSourceDecayOptions& options)
{
    if (options.group_count < 0)
    {
        throw std::invalid_argument(
            "Scalar group group count cannot be negative.");
    }
    const auto count = static_cast<size_t>(options.group_count);
    auto check_size =
        [count](const ArrReal& values, const std::string& label)
    {
        if (!values.empty() && values.size() != count)
        {
            throw std::invalid_argument(
                label + " must be empty or match scalar_group_count.");
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
    check_size(options.decay_constants, "scalar_group_decay_constants");
    check_size(
        options.initial_concentrations,
        "scalar_group_initial_concentrations");
    check_size(options.source_terms, "scalar_group_source_terms");
    for (const auto value : options.decay_constants)
    {
        if (value < 0.0)
        {
            throw std::invalid_argument(
                "Scalar group decay constants cannot be negative.");
        }
    }
    for (const auto value : options.initial_concentrations)
    {
        if (value < 0.0)
        {
            throw std::invalid_argument(
                "Scalar group initial concentrations cannot be negative.");
        }
    }
    for (const auto value : options.source_terms)
    {
        if (value < 0.0)
        {
            throw std::invalid_argument(
                "Scalar group source terms cannot be negative.");
        }
    }
    if (!std::isfinite(options.effective_diffusivity)
        || options.effective_diffusivity < 0.0)
    {
        throw std::invalid_argument(
            "Scalar group effective diffusivity must be finite and non-negative.");
    }
}

/**
 * @brief Parse scalar source-decay options from a flat database.
 *
 * @param database Database containing optional scalar_group keys.
 * @return Validated scalar_group options.
 */
inline ScalarSourceDecayOptions
scalar_source_decay_options_from_database(
    const Database& database)
{
    for (const auto* key : {
             "precursor_group_count",
             "precursor_decay_constants",
             "precursor_initial_concentrations",
             "precursor_source_terms",
             "precursor_power_yields",
             "precursor_effective_diffusivity"})
    {
        if (database.contains(key))
        {
            throw std::invalid_argument(
                std::string(key) + " is retired; configure scalar_group_* inventory sources and transport explicitly.");
        }
    }

    ScalarSourceDecayOptions options;
    const detail::DatabaseOptionReader reader(
        database, "Scalar source-decay model");
    options.group_count = reader.value_or<int>(
        "scalar_group_count", options.group_count);
    options.decay_constants = reader.value_or<ArrReal>(
        "scalar_group_decay_constants", options.decay_constants);
    options.initial_concentrations = reader.value_or<ArrReal>(
        "scalar_group_initial_concentrations",
        options.initial_concentrations);
    options.source_terms = reader.value_or<ArrReal>(
        "scalar_group_source_terms", options.source_terms);
    options.effective_diffusivity = reader.value_or<real_t>(
        "scalar_group_effective_diffusivity",
        options.effective_diffusivity);
    validate_scalar_source_decay_options(options);
    return options;
}

/**
 * @brief Conservative liquid-phase scalar source-decay transport.
 *
 * The persistent state is the mixture-volume inventory @f$\alpha_l C_i@f$.
 * Source and decay are integrated exactly over each step, then conservative
 * liquid advection and optional diffusion solve for @f$C_i@f$ with storage
 * and advection weight @f$\alpha_l@f$, diffusion coefficient
 * @f$\alpha_l D_i^{eff}@f$, and homogeneous zero-flux diffusion boundaries.
 *
 * Amount units are declared by the caller and must be identical in concentration,
 * bulk-volume sources, and the optional spatial source field. This utility does
 * not define species stoichiometry, thermal coupling, or a reaction mechanism.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes,
         class MeshType = Mesh<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT ScalarSourceDecayTransport
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;
    using diagnostics_type = ScalarInventoryDiagnostics<scalar_type>;

    /** Complete accepted scalar_group state, tied to one configuration and geometry. */
    class StateSnapshot
    {
        friend class ScalarSourceDecayTransport;
        const ScalarSourceDecayTransport* owner = nullptr;
        std::shared_ptr<const int> configuration;
        std::uint64_t epoch{};
        std::array<std::vector<std::vector<scalar_type>>,3> values;
        std::vector<diagnostics_type> diagnostics;
        bool inventory_initialized{};
    };

    StateSnapshot snapshot() const
    {
        StateSnapshot saved;
        saved.owner=this; saved.configuration=d_snapshot_configuration;
        saved.epoch=mesh_geometry_epoch(*d_mesh);
        const std::array groups{&d_fields,&d_inventories,&d_sources};
        for(size_t kind=0;kind<groups.size();++kind)
            for(const auto& field:*groups[kind])
            {
                auto& values=saved.values[kind].emplace_back();
                for(size_t i=0;i<d_mesh->num_owned_cells();++i) values.push_back(field->value(i));
            }
        saved.diagnostics=d_last_inventory_diagnostics;
        saved.inventory_initialized=d_inventory_initialized;
        return saved;
    }

    void restore(const StateSnapshot& saved)
    {
        collective_detail::collective_local_validation(*d_mesh,"Scalar group snapshot restore",[&]
        {
            if(saved.owner!=this || saved.configuration!=d_snapshot_configuration || saved.epoch!=mesh_geometry_epoch(*d_mesh))
                throw std::invalid_argument("Scalar group snapshot is foreign or stale.");
        });
        const std::array groups{&d_fields,&d_inventories,&d_sources};
        for(size_t kind=0;kind<groups.size();++kind)
            for(size_t group=0;group<groups[kind]->size();++group)
            {
                auto& field=*groups[kind]->at(group);
                for(size_t i=0;i<d_mesh->num_owned_cells();++i) field.set_owned_value(i,saved.values[kind][group][i]);
                field.sync_ghosts();
            }
        d_last_inventory_diagnostics=saved.diagnostics;
        d_inventory_initialized=saved.inventory_initialized;
        d_transport_solver.reset();
    }

    /**
     * @brief Construct a scalar_group model on a mesh.
     * @note Construction validates options collectively on the mesh
     *       communicator.
     */
    ScalarSourceDecayTransport(
        SP<const mesh_type> mesh,
        ScalarSourceDecayOptions options = {})
        : d_mesh(std::move(mesh))
    {
        if (!d_mesh)
        {
            throw std::invalid_argument(
                "ScalarSourceDecayTransport requires a non-null mesh.");
        }
        d_transport_geometry_cache.emplace(*d_mesh);
        configure(options);
    }

    /**
     * @brief Replace scalar_group options and rebuild group fields.
     * @note Invoke collectively on every mesh rank.
     */
    void configure(const ScalarSourceDecayOptions& options);

    /**
     * @brief True when at least one scalar_group group is active.
     */
    bool enabled() const noexcept
    {
        return !d_fields.empty();
    }

    /**
     * @brief Number of configured scalar_group groups.
     */
    size_t group_count() const noexcept { return d_fields.size(); }

    /**
     * @brief Concentration field for a zero-based scalar_group group index.
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
     * @brief Source field for a zero-based scalar_group group index.
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
     * @brief Last globally reduced inventory balance for a scalar_group group.
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
     * Call before transport to bind concentrations to the initial liquid fraction.
     * The first advance initializes lazily when this call is omitted.
     */
    void initialize_inventory(const field_type& alpha_l);

    /**
     * @brief Advance reaction and diffusion without an advecting face flux.
     *
     * The optional source is a scalar inventory rate, independent of thermal input.
     */
    void advance(
        scalar_type time_step,
        const field_type& alpha_l,
        const field_type* additional_source)
    {
        advance(time_step, alpha_l, additional_source, nullptr);
    }

    /**
     * @brief Advance conserved scalar_group inventories by one reaction/transport step.
     *
     * @param time_step Positive physical timestep in seconds.
     * @param alpha_l Current liquid volume fraction.
     * @param additional_source Optional nonnegative scalar inventory source per bulk volume per second,
     *        added directly to every configured scalar group; no heat/yield conversion.
     * @param liquid_face_flux Optional owner-oriented volumetric liquid flux.
     *        When absent, the transport solve contains diffusion only.
     */
    void advance(
        scalar_type time_step,
        const field_type& alpha_l,
        const field_type* additional_source,
        const face_flux_field_type* liquid_face_flux);

private:
    /** @brief Validate options collectively before group-dependent allocation. */
    SIMPLEFLUID_EQUATIONS_LOCAL
    void validate_collective_configuration(
        const ScalarSourceDecayOptions& options) const;

    /** @brief Require every rank to select the same collective update path. */
    SIMPLEFLUID_EQUATIONS_LOCAL
    void validate_collective_advance_selection(
        scalar_type time_step,
        const field_type* additional_source,
        const face_flux_field_type* liquid_face_flux) const;

    /** @brief Require a finite liquid fraction in the interval (0, 1]. */
    static scalar_type checked_liquid_fraction(scalar_type value)
    {
        if (!std::isfinite(value)
            || value <= scalar_type{}
            || value > scalar_type{1})
        {
            throw std::invalid_argument(
                "Scalar group liquid fractions must be finite and in (0, 1].");
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
     * @param time_step Physical timestep in seconds.
     * @param alpha_l Liquid volume fraction used for inventory conversion.
     * @param liquid_face_flux Optional owner-oriented liquid flux used by transport.
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

    std::shared_ptr<const int> d_snapshot_configuration;
    SP<const mesh_type> d_mesh;
    std::optional<FVM::TransportGeometryCache<mesh_type>>
        d_transport_geometry_cache;
    ScalarSourceDecayOptions d_options;
    ArrReal d_decay_constants;
    ArrReal d_initial_concentrations;
    ArrReal d_source_terms;
    std::vector<std::unique_ptr<field_type>> d_fields;
    std::vector<std::unique_ptr<field_type>> d_inventories;
    std::vector<std::unique_ptr<field_type>> d_sources;
    std::vector<diagnostics_type> d_last_inventory_diagnostics;
    std::map<std::string, const field_type*> d_output_fields;
    BelosLinearSolver<Pack> d_transport_solver;
    bool d_inventory_initialized = false;
};
extern template class ScalarSourceDecayTransport<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class ScalarSourceDecayTransport<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;


} // namespace SimpleFluid
