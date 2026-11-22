/** @file FVM/CarrierTransportPlan.hh
 * @brief Physical carrier vs numerical operator-group planning and MultiVector pack/unpack buffers.
 */
#pragma once

#include "FVM/FaceCoefficientInterpolation.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/ScalarTransportDiscretization.hh"
#include "dataclass/TpetraTypes.hh"
#include "dataclass/typedefs.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshConcepts.hh"

#include <Teuchos_RCP.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimpleFluid::FVM
{

/** @brief Physical carrier family owning a transported column. */
enum class CarrierKind
{
    Liquid,
    BulkGas,
    Population
};

/** @brief Stable identity of a physical carrier (phase or named dispersed population). */
struct CarrierId
{
    CarrierKind kind = CarrierKind::Liquid;
    std::string identity = {};

    bool operator==(const CarrierId&) const = default;
    auto operator<=>(const CarrierId&) const = default;
};

/** @brief Physical meaning of a scalar transport column. */
enum class QuantityKind
{
    SpeciesMoles,
    PopulationNumber,
    InternalEnergy
};

/** @brief Per-column linear/physical acceptance contract. */
struct TransportAcceptancePolicy
{
    real_t relative_tolerance = 1.0e-10;
    real_t absolute_tolerance = {};
    real_t rhs_norm_floor = {};
    bool allow_negative = false;

    bool operator==(const TransportAcceptancePolicy&) const = default;
    auto operator<=>(const TransportAcceptancePolicy&) const = default;
};

/** @brief Explicit semantic metadata for one transported column. */
struct TransportColumn
{
    CarrierId carrier = {};
    QuantityKind quantity = QuantityKind::SpeciesMoles;
    std::optional<std::uint32_t> species_id = {};
    std::optional<std::string> compartment_id = {};
    TransportAcceptancePolicy acceptance_policy = {};
    std::size_t source_index = 0;

    bool operator==(const TransportColumn&) const = default;
};

/** @brief Explicit field identity + mutation epoch proof for matrix-sharing keys. */
struct FieldVersionKey
{
    const void* field_identity = nullptr;
    std::uint64_t version_epoch = 0;

    bool operator==(const FieldVersionKey&) const = default;
    auto operator<=>(const FieldVersionKey&) const = default;
};

/** @brief Explicit matrix-affecting boundary condition & coefficient-cache key. */
struct BoundaryOperatorKey
{
    std::uint64_t condition_epoch = 0;
    const void* coefficient_cache_identity = nullptr;
    std::uint64_t coefficient_cache_epoch = 0;

    bool operator==(const BoundaryOperatorKey&) const = default;
    auto operator<=>(const BoundaryOperatorKey&) const = default;
};

/** @brief Explicit fixed-cell mask key affecting matrix rows. */
struct OperatorConstraintKey
{
    bool has_fixed_rows = false;
    std::uint64_t fixed_rows_epoch = 0;

    bool operator==(const OperatorConstraintKey&) const = default;
    auto operator<=>(const OperatorConstraintKey&) const = default;
};

/**
 * @brief Exact discrete-matrix identity key for an OperatorGroup.
 *
 * Columns may share a single assembled matrix A_g if and only if their entire
 * TransportOperatorKey matches identically. Floating-point approximations and
 * carrier-name guessing are forbidden.
 */
struct TransportOperatorKey
{
    const void* mesh_identity = nullptr;
    std::uint64_t geometry_epoch = 0;
    bool has_ale = false;
    std::uint64_t ale_old_geometry_epoch = 0;
    std::uint64_t ale_new_geometry_epoch = 0;
    real_t time_step = {};
    ScalarTransportDiscretization discretization = {};
    NonOrthogonalTreatment non_orthogonal_treatment = NonOrthogonalTreatment::Explicit;
    FaceCoefficientInterpolation coefficient_interpolation = FaceCoefficientInterpolation::Harmonic;
    FieldVersionKey storage_weight = {};
    FieldVersionKey advection_flux = {};
    FieldVersionKey advection_weight = {};
    FieldVersionKey diffusivity = {};
    std::optional<FieldVersionKey> implicit_sink = {};
    std::optional<FieldVersionKey> implicit_correction = {};
    BoundaryOperatorKey matrix_boundary = {};
    OperatorConstraintKey constraints = {};
    std::optional<CarrierId> carrier_scope = {};

    bool operator==(const TransportOperatorKey&) const = default;
    auto operator<=>(const TransportOperatorKey&) const = default;
};

/** @brief Input column + operator identity pair passed to the planner. */
struct PlannedColumnEntry
{
    TransportColumn column = {};
    TransportOperatorKey operator_key = {};
};

/** @brief Physical grouping of all columns belonging to one carrier. */
struct CarrierGroup
{
    CarrierId carrier = {};
    std::vector<std::size_t> plan_column_indices = {};
};

/** @brief Numerical grouping of columns sharing the exact same discrete matrix A_g. */
struct OperatorGroup
{
    std::size_t group_index = 0;
    TransportOperatorKey operator_key = {};
    std::vector<std::size_t> plan_column_indices = {};
};

/** @brief Controls for building a CarrierTransportPlan. */
struct CarrierTransportPlanOptions
{
    /** Default false: never guess cross-carrier matrix sharing. */
    bool allow_cross_carrier_sharing = false;
};

/** @brief Deterministic separation of physical carriers and numerical operator groups. */
struct CarrierTransportPlan
{
    std::vector<TransportColumn> columns = {};
    std::vector<CarrierGroup> carrier_groups = {};
    std::vector<OperatorGroup> operator_groups = {};
};

/** @brief Build a deterministic CarrierTransportPlan from explicit column + operator keys. */
inline CarrierTransportPlan build_carrier_transport_plan(
    std::span<const PlannedColumnEntry> entries,
    const CarrierTransportPlanOptions& options = {})
{
    std::vector<PlannedColumnEntry> normalized;
    normalized.reserve(entries.size());

    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        PlannedColumnEntry item = entries[i];
        if (item.operator_key.mesh_identity == nullptr)
        {
            throw std::invalid_argument(
                "CarrierTransportPlan requires a non-null mesh_identity in TransportOperatorKey.");
        }
        if (!std::isfinite(item.operator_key.time_step)
            || item.operator_key.time_step <= real_t{})
        {
            throw std::invalid_argument(
                "CarrierTransportPlan requires a finite positive time_step.");
        }
        if (!std::isfinite(item.column.acceptance_policy.relative_tolerance)
            || item.column.acceptance_policy.relative_tolerance <= real_t{})
        {
            throw std::invalid_argument(
                "CarrierTransportPlan requires positive finite column relative_tolerance.");
        }
        if (!std::isfinite(item.column.acceptance_policy.absolute_tolerance)
            || item.column.acceptance_policy.absolute_tolerance < real_t{})
        {
            throw std::invalid_argument(
                "CarrierTransportPlan requires non-negative finite column absolute_tolerance.");
        }
        if (!std::isfinite(item.column.acceptance_policy.rhs_norm_floor)
            || item.column.acceptance_policy.rhs_norm_floor < real_t{})
        {
            throw std::invalid_argument(
                "CarrierTransportPlan requires non-negative finite column rhs_norm_floor.");
        }
        if (item.column.quantity == QuantityKind::InternalEnergy)
        {
            // InternalEnergy is signed and must never carry a non-negativity gate.
            item.column.acceptance_policy.allow_negative = true;
        }
        if (!options.allow_cross_carrier_sharing)
        {
            item.operator_key.carrier_scope = item.column.carrier;
        }
        normalized.push_back(std::move(item));
    }

    auto column_order = [](const PlannedColumnEntry& a, const PlannedColumnEntry& b)
    {
        if (a.column.carrier != b.column.carrier)
            return a.column.carrier < b.column.carrier;
        if (a.column.quantity != b.column.quantity)
            return a.column.quantity < b.column.quantity;
        if (a.column.species_id != b.column.species_id)
            return a.column.species_id < b.column.species_id;
        if (a.column.compartment_id != b.column.compartment_id)
            return a.column.compartment_id < b.column.compartment_id;
        if (a.operator_key != b.operator_key)
            return a.operator_key < b.operator_key;
        return a.column.source_index < b.column.source_index;
    };

    std::stable_sort(normalized.begin(), normalized.end(), column_order);

    CarrierTransportPlan plan;
    plan.columns.reserve(normalized.size());

    std::map<CarrierId, std::size_t> carrier_index;
    std::map<TransportOperatorKey, std::size_t> operator_index;

    for (std::size_t plan_col = 0; plan_col < normalized.size(); ++plan_col)
    {
        const auto& entry = normalized[plan_col];
        plan.columns.push_back(entry.column);

        auto [c_it, c_inserted] = carrier_index.emplace(entry.column.carrier, plan.carrier_groups.size());
        if (c_inserted)
        {
            plan.carrier_groups.push_back(CarrierGroup{entry.column.carrier, {}});
        }
        plan.carrier_groups[c_it->second].plan_column_indices.push_back(plan_col);

        auto [o_it, o_inserted] = operator_index.emplace(entry.operator_key, plan.operator_groups.size());
        if (o_inserted)
        {
            plan.operator_groups.push_back(OperatorGroup{plan.operator_groups.size(), entry.operator_key, {}});
        }
        plan.operator_groups[o_it->second].plan_column_indices.push_back(plan_col);
    }

    return plan;
}

/** @brief Pre-allocated MultiVector scratch buffer for deterministic pack/unpack around solves. */
template<TpetraTypePack Pack>
class CarrierMultiVectorBuffer
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using map_type = typename Pack::map_type;
    using multi_vector_type = typename Pack::multi_vector_type;

    multi_vector_type& ensure(
        const Teuchos::RCP<const map_type>& map,
        std::size_t num_columns)
    {
        if (map.is_null())
        {
            throw std::invalid_argument("CarrierMultiVectorBuffer requires a non-null map.");
        }
        if (num_columns == 0)
        {
            throw std::invalid_argument("CarrierMultiVectorBuffer requires at least one column.");
        }
        if (d_storage.is_null()
            || d_storage->getNumVectors() != num_columns
            || !d_storage->getMap()->isSameAs(*map))
        {
            d_storage = Teuchos::rcp(new multi_vector_type(map, num_columns, false));
        }
        return *d_storage;
    }

    Teuchos::RCP<multi_vector_type> storage_rcp() const noexcept { return d_storage; }

    template<SimpleFluid::MeshType MeshType>
    double pack_owned_fields(
        const Teuchos::RCP<const map_type>& map,
        std::span<const ScalarCellFieldStored<Pack, MeshType>* const> fields,
        std::span<const scalar_type> cell_inverse_scale = {})
    {
        const auto start = std::chrono::steady_clock::now();
        auto& mv = ensure(map, fields.size());
        const std::size_t local_rows = mv.getLocalLength();
        if (!cell_inverse_scale.empty() && cell_inverse_scale.size() != local_rows)
        {
            throw std::invalid_argument("CarrierMultiVectorBuffer scale span must match owned cell count.");
        }
        for (std::size_t col = 0; col < fields.size(); ++col)
        {
            if (fields[col] == nullptr)
            {
                throw std::invalid_argument("CarrierMultiVectorBuffer received null source field.");
            }
            const auto src = fields[col]->owned_read_view();
            auto dst = mv.getDataNonConst(col);
            for (std::size_t row = 0; row < local_rows; ++row)
            {
                dst[row] = cell_inverse_scale.empty()
                    ? src(row, 0)
                    : src(row, 0) * cell_inverse_scale[row];
            }
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    double pack_owned_spans(
        const Teuchos::RCP<const map_type>& map,
        std::span<const std::span<const scalar_type>> columns,
        std::span<const scalar_type> cell_inverse_scale = {})
    {
        const auto start = std::chrono::steady_clock::now();
        auto& mv = ensure(map, columns.size());
        const std::size_t local_rows = mv.getLocalLength();
        if (!cell_inverse_scale.empty() && cell_inverse_scale.size() != local_rows)
        {
            throw std::invalid_argument("CarrierMultiVectorBuffer scale span must match owned cell count.");
        }
        for (std::size_t col = 0; col < columns.size(); ++col)
        {
            if (columns[col].size() != local_rows)
            {
                throw std::invalid_argument("CarrierMultiVectorBuffer source span size mismatch.");
            }
            auto dst = mv.getDataNonConst(col);
            for (std::size_t row = 0; row < local_rows; ++row)
            {
                dst[row] = cell_inverse_scale.empty()
                    ? columns[col][row]
                    : columns[col][row] * cell_inverse_scale[row];
            }
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    template<SimpleFluid::MeshType MeshType>
    double unpack_owned_fields(
        std::span<ScalarCellFieldStored<Pack, MeshType>* const> fields,
        std::span<const scalar_type> cell_scale = {}) const
    {
        const auto start = std::chrono::steady_clock::now();
        if (d_storage.is_null() || d_storage->getNumVectors() != fields.size())
        {
            throw std::invalid_argument("CarrierMultiVectorBuffer unpack column count mismatch.");
        }
        const std::size_t local_rows = d_storage->getLocalLength();
        if (!cell_scale.empty() && cell_scale.size() != local_rows)
        {
            throw std::invalid_argument("CarrierMultiVectorBuffer scale span must match owned cell count.");
        }
        for (std::size_t col = 0; col < fields.size(); ++col)
        {
            if (fields[col] == nullptr)
            {
                throw std::invalid_argument("CarrierMultiVectorBuffer received null destination field.");
            }
            const auto src = d_storage->getData(col);
            const auto dst = fields[col]->owned_write_view();
            for (std::size_t row = 0; row < local_rows; ++row)
            {
                dst(row, 0) = cell_scale.empty() ? src[row] : src[row] * cell_scale[row];
            }
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    double unpack_owned_spans(
        std::span<const std::span<scalar_type>> columns,
        std::span<const scalar_type> cell_scale = {}) const
    {
        const auto start = std::chrono::steady_clock::now();
        if (d_storage.is_null() || d_storage->getNumVectors() != columns.size())
        {
            throw std::invalid_argument("CarrierMultiVectorBuffer unpack column count mismatch.");
        }
        const std::size_t local_rows = d_storage->getLocalLength();
        if (!cell_scale.empty() && cell_scale.size() != local_rows)
        {
            throw std::invalid_argument("CarrierMultiVectorBuffer scale span must match owned cell count.");
        }
        for (std::size_t col = 0; col < columns.size(); ++col)
        {
            if (columns[col].size() != local_rows)
            {
                throw std::invalid_argument("CarrierMultiVectorBuffer destination span size mismatch.");
            }
            const auto src = d_storage->getData(col);
            for (std::size_t row = 0; row < local_rows; ++row)
            {
                columns[col][row] = cell_scale.empty() ? src[row] : src[row] * cell_scale[row];
            }
        }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

private:
    Teuchos::RCP<multi_vector_type> d_storage;
};

} // namespace SimpleFluid::FVM
