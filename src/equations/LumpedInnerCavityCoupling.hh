/** @file LumpedInnerCavityCoupling.hh
 * @brief Collective field sampling and conservative thermal boundary refresh.
 */
#pragma once

#include "FVM/BoundaryCache.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/LumpedInnerCavityModel.hh"
#include "parallel/CommunicatorCompatibility.hh"

#include <Teuchos_Comm.hpp>
#include <algorithm>
#include <limits>
#include <set>
#include <string_view>

namespace SimpleFluid
{

enum class InnerCavitySampling
{
    Facewise,
    AreaAverage ///< One isothermal surface per batch; transfers distributed by face area.
};

enum class InnerCavityEnergyTransfer
{
    Heat,       ///< Sensible + latent + radiation; liquid enthalpy is accounted for separately.
    TotalEnergy ///< Includes transported liquid enthalpy using the cavity's caloric reference.
};

/** Collective setup convenience; ranks without local faces may have no batch entry. */
template<class Mesh> int inner_cavity_batch_id(const Mesh& mesh, std::string_view name)
{
    int local_min = std::numeric_limits<int>::max(), local_max = std::numeric_limits<int>::min();
    collective_detail::collective_local_validation(mesh, "Cavity batch-name resolution",
        [&]
        {
            for (const auto& [id, batch] : mesh.boundary_batches())
                if (mesh.boundary_batch_name(id) == name)
                {
                    local_min = std::min(local_min, id);
                    local_max = std::max(local_max, id);
                }
        });
    int minimum = 0, maximum = 0;
    const auto comm = mesh.owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_min, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_max, &maximum);
    if (minimum != maximum)
        throw std::invalid_argument("Unknown or ambiguous cavity boundary name.");
    return minimum;
}

/** A batch owns one mesh/communicator, boundary ID, closure and live field reader.
 * Face entries contain only sampled face data. The temperature field must
 * outlive the batch; copying or moving a batch preserves the field binding.
 */
class InnerCavityFieldBatch
{
public:
    struct FaceEntry
    {
        global_index_t face;
        real_t area, temperature;
        std::size_t slot;
    };

    /** Face fields supply interface temperatures; cell fields use owned owner-cell
     * values without reconstruction. Each read samples current geometry and values.
     * Supports scalar legacy fields and FieldStored on mapped/subdomain meshes.
     */
    template<class Field>
    static InnerCavityFieldBatch from_field(Field& temperature, int batch_id, InnerCavitySurface properties,
        InnerCavitySampling sampling = InnerCavitySampling::Facewise)
    {
        static_assert(std::convertible_to<decltype(temperature.value(0)), real_t>);
        static_assert(requires { temperature.num_owned_faces(); } || requires { temperature.num_owned_cells(); });
        InnerCavityFieldBatch batch;
        batch.d_mesh = temperature.mesh_ptr();
        batch.d_comm = temperature.mesh().owned_cell_map()->getComm();
        batch.d_batch_id = batch_id;
        batch.d_properties = std::move(properties);
        batch.d_sampling = sampling;
        // The reader captures only the field. Shared context is supplied by
        // the batch at invocation, so it is not copied into the closure.
        batch.d_read = [&temperature](const void* expected_mesh, int batch_id)
        {
            std::vector<FaceEntry> values;
            const auto& mesh = temperature.mesh();
            if (&mesh != expected_mesh)
                throw std::invalid_argument("Cavity field binding changed meshes; rebuild the batch.");
            const auto found = mesh.boundary_batches().find(batch_id);
            if (found == mesh.boundary_batches().end())
                return values; // Valid on an empty owning rank.
            const auto view = temperature.owned_read_view();
            const auto& faces = found->second.face_lids;
            values.reserve(faces.size());
            for (std::size_t slot = 0; slot < faces.size(); ++slot)
            {
                const auto face = faces[slot];
                if (!mesh.is_owned_face(face))
                    continue;
                const auto owner = mesh.owner_cell(face);
                if (!mesh.is_boundary_face(face) || !mesh.is_owned_cell(owner))
                    throw std::invalid_argument("Cavity field batch requires owned physical boundary faces.");
                real_t value;
                if constexpr (requires { temperature.num_owned_faces(); })
                    value = view(temperature.owned_row(face), 0);
                else
                    value = view(owner, 0);
                values.push_back({mesh.face_global_id(face), mesh.face_area(face), value, slot});
            }
            return values;
        };
        return batch;
    }

    /** Collective convenience overload: resolve a name exactly once at binding. */
    template<class Field>
    static InnerCavityFieldBatch from_field(Field& temperature, std::string_view patch, InnerCavitySurface properties,
        InnerCavitySampling sampling = InnerCavitySampling::Facewise)
    {
        return from_field(
            temperature, inner_cavity_batch_id(temperature.mesh(), patch), std::move(properties), sampling);
    }

private:
    friend class InnerCavityFieldInput;
    std::shared_ptr<const void> d_mesh;
    Teuchos::RCP<const Teuchos::Comm<int>> d_comm;
    int d_batch_id = -1;
    InnerCavitySurface d_properties;
    InnerCavitySampling d_sampling = InnerCavitySampling::Facewise;
    std::function<std::vector<FaceEntry>(const void*, int)> d_read;
};

/**
 * @brief Field batches on identical or congruent ordered communicator groups.
 *
 * All operations are collective, including on ranks without boundary faces.
 * Batches may reference different mesh/field types. IDs, modes and closure
 * configuration are validated collectively once. Sampling is fresh on every
 * operation. Global surface order is batch order, then ascending global face ID
 * for Facewise batches; AreaAverage batches contribute one surface each.
 * View factors use that global order. The dense enclosure is replicated, so
 * facewise radiation is intended for small boundary sets.
 */
class SIMPLEFLUID_EQUATIONS_EXPORT InnerCavityFieldInput
{
public:
    InnerCavityFieldInput(Teuchos::RCP<const Teuchos::Comm<int>> comm, std::vector<InnerCavityFieldBatch> batches);

    LumpedInnerCavityExchange evaluate(const LumpedInnerCavityModel& model, std::span<const real_t> factors = {}) const;
    LumpedInnerCavityStep preview(
        const LumpedInnerCavityModel& model, real_t dt, std::span<const real_t> factors = {}) const;
    LumpedInnerCavityStep advance(LumpedInnerCavityModel& model, real_t dt, std::span<const real_t> factors = {}) const;

private:
    struct Snapshot
    {
        std::vector<cavity_detail::SurfaceSample> surfaces;
        std::vector<InnerCavityBoundaryExchange> boundaries;
        std::vector<std::vector<std::size_t>> indices;
    };
    struct BatchPlan
    {
        std::vector<global_index_t> local_ids;
        std::vector<global_index_t> global_ids;
        std::vector<int> rank_counts;
        std::vector<std::size_t> gathered_to_sorted;
        std::vector<std::size_t> local_to_sorted;
        bool ready = false;
    };
    void configure() const;
    void rebuild_plan(std::size_t batch, const std::vector<InnerCavityFieldBatch::FaceEntry>& local) const;
    Snapshot sample() const;
    void attach(Snapshot&& snapshot, LumpedInnerCavityExchange& exchange) const;
    void validate_model(const LumpedInnerCavityModel& model, real_t dt, std::span<const real_t> factors) const;
    void validate_result(const LumpedInnerCavityStep& step) const;

    Teuchos::RCP<const Teuchos::Comm<int>> d_comm;
    std::vector<InnerCavityFieldBatch> d_batches;
    mutable bool d_configured = false;
    mutable std::vector<BatchPlan> d_plans;
};

template<class Batch>
    requires std::same_as<Batch, InnerCavityFieldInput>
LumpedInnerCavityExchange LumpedInnerCavityModel::evaluate(const Batch& batch, std::span<const real_t> factors) const
{
    return batch.evaluate(*this, factors);
}
template<class Batch>
    requires std::same_as<Batch, InnerCavityFieldInput>
LumpedInnerCavityStep LumpedInnerCavityModel::preview(
    real_t dt, const Batch& batch, std::span<const real_t> factors) const
{
    return batch.preview(*this, dt, factors);
}
template<class Batch>
    requires std::same_as<Batch, InnerCavityFieldInput>
LumpedInnerCavityStep LumpedInnerCavityModel::advance(real_t dt, const Batch& batch, std::span<const real_t> factors)
{
    return batch.advance(*this, dt, factors);
}

namespace cavity_detail
{
/** One staged cache copy. The caller publishes or moves it after validation. */
template<class Cache>
std::shared_ptr<Cache> refreshed_heat_flux(
    const Cache& old, const LumpedInnerCavityExchange& exchange, InnerCavityEnergyTransfer transfer)
{
    if (!old.mesh)
        throw std::invalid_argument("Cavity boundary refresh requires a mesh-bound cache.");
    const auto& mesh = *old.mesh;
    collective_detail::require_uniform_value(mesh, static_cast<int>(transfer), "Cavity boundary energy transfer");
    std::shared_ptr<Cache> candidate;
    collective_detail::collective_local_validation(mesh, "Cavity boundary refresh",
        [&]
        {
            if (transfer != InnerCavityEnergyTransfer::Heat && transfer != InnerCavityEnergyTransfer::TotalEnergy)
                throw std::invalid_argument("Unknown cavity energy transfer selection.");
            candidate = std::make_shared<Cache>(old);
            bool matched = false;
            std::set<int> batches;
            for (const auto& boundary : exchange.boundaries)
            {
                if (boundary.mesh.get() != &mesh)
                    continue;
                matched = true;
                if (!batches.insert(boundary.batch_id).second)
                    throw std::invalid_argument("Duplicate cavity batch in refresh.");
                const auto found = mesh.boundary_batches().find(boundary.batch_id);
                if (found == mesh.boundary_batches().end())
                {
                    if (!boundary.faces.empty())
                        throw std::invalid_argument("Cavity refresh refers to a missing local batch.");
                    candidate->value.erase(boundary.batch_id);
                    continue;
                }
                const auto& faces = found->second.face_lids;
                auto& values = candidate->value[boundary.batch_id];
                values.assign(faces.size(), 0.0);
                std::vector<bool> seen(faces.size(), false);
                for (const auto& face : boundary.faces)
                {
                    const auto slot = face.batch_slot;
                    if (slot >= faces.size() || seen[slot] || !mesh.is_owned_face(faces[slot]) ||
                        mesh.face_global_id(faces[slot]) != face.face_global_id)
                        throw std::invalid_argument("Cavity refresh has duplicate or stale face slots.");
                    seen[slot] = true;
                    const auto area = mesh.face_area(faces[slot]);
                    if (!std::isfinite(area) || area <= 0.0 || !std::isfinite(face.area) ||
                        std::abs(area - face.area) > 1e-12 * area)
                        throw std::invalid_argument("Cavity refresh has stale or invalid face geometry.");
                    const auto rate = transfer == InnerCavityEnergyTransfer::Heat ? face.exchange.heat_rate
                                                                                  : face.exchange.energy_rate;
                    values[slot] = rate / area;
                    if (!std::isfinite(values[slot]))
                        throw std::invalid_argument("Cavity refresh has a non-finite heat flux.");
                }
                for (std::size_t slot = 0; slot < faces.size(); ++slot)
                    if (mesh.is_owned_face(faces[slot]) && !seen[slot])
                        throw std::invalid_argument("Cavity refresh is missing an owned boundary face.");
            }
            if (!matched)
                throw std::invalid_argument("Cavity exchange does not refer to the boundary cache mesh.");
        });
    return candidate;
}
} // namespace cavity_detail

/** Refresh a mutable cache, preserving unrelated batches and transactional failure. */
template<class Cache>
void refresh_inner_cavity_boundary_conditions(Cache& heat_flux, const LumpedInnerCavityExchange& exchange,
    InnerCavityEnergyTransfer transfer = InnerCavityEnergyTransfer::Heat)
{
    auto candidate = cavity_detail::refreshed_heat_flux(heat_flux, exchange, transfer);
    heat_flux.value.swap(candidate->value);
}

/** Preferred coupled refresh: one staged snapshot shared by caller and equation.
 * Initialize heat_flux with an empty mesh-bound cache. Published snapshots are
 * immutable, so an earlier shared pointer is a safe boundary checkpoint.
 */
template<class Equation, class Cache>
void refresh_inner_cavity_boundary_conditions(Equation& equation, std::shared_ptr<const Cache>& heat_flux,
    const LumpedInnerCavityExchange& exchange, InnerCavityEnergyTransfer transfer = InnerCavityEnergyTransfer::Heat)
{
    if (!heat_flux)
        throw std::invalid_argument("Cavity refresh requires an initial mesh-bound heat-flux snapshot.");
    std::shared_ptr<const Cache> candidate = cavity_detail::refreshed_heat_flux(*heat_flux, exchange, transfer);
    equation.set_boundary_heat_flux_snapshot(candidate);
    heat_flux.swap(candidate);
}

/** Mutable-cache convenience: separate caller/equation ownership requires one extra copy. */
template<class Equation, class Cache>
void refresh_inner_cavity_boundary_conditions(Equation& equation, Cache& heat_flux,
    const LumpedInnerCavityExchange& exchange, InnerCavityEnergyTransfer transfer = InnerCavityEnergyTransfer::Heat)
{
    auto candidate = cavity_detail::refreshed_heat_flux(heat_flux, exchange, transfer);
    equation.set_boundary_heat_flux(*candidate);
    heat_flux.value.swap(candidate->value);
}

} // namespace SimpleFluid
