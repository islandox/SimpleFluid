/**
 * @file FeedbackMap.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Utilities for thermal-hydraulic feedback mapping.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/CellField.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid::FeedbackMap
{

/**
 * @brief Coarse feedback cell described by rank-local owned fine-cell IDs.
 *
 * In a distributed mesh, every rank supplies the same ordered coarse-cell
 * definitions while `cell_lids` contains only the portion owned by that
 * rank.  An empty local portion is valid when another rank contributes to the
 * same coarse cell.
 *
 * @tparam Pack Tpetra type pack used by the mapped field.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
struct FeedbackCell
{
    std::string name;
    std::vector<typename Pack::local_ordinal_type> cell_lids;
};

namespace detail
{

/** @brief Scalar cell field interface shared by legacy and mapped storage. */
template<class Field, class Pack>
concept ScalarFeedbackField =
    std::same_as<typename Field::scalar_type,
                 typename Pack::scalar_type>
 && requires(const Field& field,
             typename Pack::local_ordinal_type cell_lid)
{
    { field.mesh().owned_cell_map() };
    { field.mesh().num_owned_cells() } -> std::convertible_to<size_t>;
    { field.mesh().num_local_cells() } -> std::convertible_to<size_t>;
    { field.mesh().is_owned_cell(cell_lid) } -> std::convertible_to<bool>;
    { field.mesh().cell_volume(cell_lid) }
        -> std::convertible_to<typename Pack::scalar_type>;
    { field.value(cell_lid) }
        -> std::convertible_to<typename Pack::scalar_type>;
};

/** @brief Mutable subset needed to import an owned scalar field. */
template<class Field, class Pack>
concept MutableScalarFeedbackField =
    ScalarFeedbackField<Field, Pack>
 && std::same_as<typename Field::local_ordinal_type,
                 typename Pack::local_ordinal_type>
 && requires(Field& field,
             typename Pack::local_ordinal_type cell_lid,
             typename Pack::scalar_type value)
{
    field.set_owned_value(cell_lid, value);
    field.sync_ghosts();
};

/** @brief Pack-independent mutable scalar field interface used by imports. */
template<class Field>
concept DeducibleMutableScalarFeedbackField =
    requires(Field& field,
             typename Field::local_ordinal_type cell_lid,
             typename Field::scalar_type value)
{
    { field.mesh().owned_cell_map() };
    { field.mesh().num_owned_cells() } -> std::convertible_to<size_t>;
    field.set_owned_value(cell_lid, value);
    field.sync_ghosts();
};

/**
 * @brief Reject rank-dependent ordered name sequences before field mapping.
 *
 * Every rank executes the same fixed set of reductions before an exception is
 * raised, so a registry mismatch cannot turn later per-field collectives into
 * a deadlock or silently cross-map differently named fields.
 */
template<class Comm>
void validate_collective_names(
    const Comm& comm,
    const std::vector<std::string>& names,
    std::string_view consistency_requirement)
{
    int local_count_is_valid =
        names.size()
                <= static_cast<size_t>(std::numeric_limits<int>::max())
            ? 1
            : 0;
    int global_count_is_valid = 0;
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_count_is_valid,
        &global_count_is_valid);
    if (global_count_is_valid == 0)
    {
        throw std::invalid_argument(
            "Names for " + std::string(consistency_requirement)
            + " exceed the collective reduction limit.");
    }

    const auto local_count = static_cast<int>(names.size());
    int minimum_count = 0;
    int maximum_count = 0;
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_count,
        &minimum_count);
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MAX,
        1,
        &local_count,
        &maximum_count);
    if (minimum_count != maximum_count)
    {
        throw std::invalid_argument(
            std::string(consistency_requirement) + " on every rank.");
    }
    if (names.empty())
    {
        return;
    }

    std::vector<int> encoding;
    for (const auto& name : names)
    {
        const auto name_size = static_cast<std::uint64_t>(name.size());
        for (int shift = 56; shift >= 0; shift -= 8)
        {
            encoding.push_back(
                static_cast<int>((name_size >> shift) & 0xffU));
        }
        for (const unsigned char byte : name)
        {
            encoding.push_back(static_cast<int>(byte));
        }
    }

    int local_size_is_valid =
        encoding.size()
                <= static_cast<size_t>(std::numeric_limits<int>::max())
            ? 1
            : 0;
    int global_size_is_valid = 0;
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_size_is_valid,
        &global_size_is_valid);
    if (global_size_is_valid == 0)
    {
        throw std::invalid_argument(
            "Names for " + std::string(consistency_requirement)
            + " are too large for a collective check.");
    }

    const auto local_size = static_cast<int>(encoding.size());
    int minimum_size = 0;
    int maximum_size = 0;
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_size,
        &minimum_size);
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MAX,
        1,
        &local_size,
        &maximum_size);
    if (minimum_size != maximum_size)
    {
        throw std::invalid_argument(
            std::string(consistency_requirement) + " on every rank.");
    }

    std::vector<int> minimum_encoding(encoding.size(), 0);
    std::vector<int> maximum_encoding(encoding.size(), 0);
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MIN,
        local_size,
        encoding.data(),
        minimum_encoding.data());
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MAX,
        local_size,
        encoding.data(),
        maximum_encoding.data());
    if (minimum_encoding != maximum_encoding)
    {
        throw std::invalid_argument(
            std::string(consistency_requirement) + " on every rank.");
    }
}

/** @brief Require one size-valued control to match across all ranks. */
template<class Comm>
size_t validate_collective_size(
    const Comm& comm,
    size_t value,
    std::string_view description)
{
    int local_is_valid =
        value <= static_cast<size_t>(std::numeric_limits<int>::max())
            ? 1
            : 0;
    int global_is_valid = 0;
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_is_valid,
        &global_is_valid);
    if (global_is_valid == 0)
    {
        throw std::invalid_argument(
            std::string(description)
            + " exceeds the collective reduction limit.");
    }

    const auto local_value = static_cast<int>(value);
    int minimum_value = 0;
    int maximum_value = 0;
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_value,
        &minimum_value);
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MAX,
        1,
        &local_value,
        &maximum_value);
    if (minimum_value != maximum_value)
    {
        throw std::invalid_argument(
            std::string(description) + " must match on every rank.");
    }
    return static_cast<size_t>(minimum_value);
}

/** @brief Convert a rank-local callback failure into an all-rank failure. */
template<class Comm>
void throw_if_collective_callback_failed(
    const Comm& comm,
    bool local_failed,
    std::string_view callback_name)
{
    int local_failure = local_failed ? 1 : 0;
    int global_failure = 0;
    Teuchos::reduceAll(
        comm,
        Teuchos::REDUCE_MAX,
        1,
        &local_failure,
        &global_failure);
    if (global_failure != 0)
    {
        throw std::runtime_error(
            "Outer coupling " + std::string(callback_name)
            + " failed on at least one rank.");
    }
}

/** @brief Shared implementation for explicit- and deduced-pack imports. */
template<DeducibleMutableScalarFeedbackField Field>
void import_power_density(
    Field& target,
    const std::vector<typename Field::scalar_type>& owned_values)
{
    using scalar_type = typename Field::scalar_type;
    using local_ordinal_type = typename Field::local_ordinal_type;
    std::array<int, 2> local_validity{
        owned_values.size() == target.mesh().num_owned_cells() ? 1 : 0,
        1};
    if (local_validity[0] != 0)
    {
        for (const auto value : owned_values)
        {
            if (!std::isfinite(value)
                || value < static_cast<scalar_type>(0))
            {
                local_validity[1] = 0;
                break;
            }
        }
    }

    std::array<int, 2> global_validity{};
    const auto comm = target.mesh().owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        static_cast<int>(local_validity.size()),
        local_validity.data(),
        global_validity.data());
    if (global_validity[0] == 0)
    {
        throw std::invalid_argument(
            "Imported power-density field size must match owned cells.");
    }
    if (global_validity[1] == 0)
    {
        throw std::invalid_argument(
            "Imported power-density values must be finite and "
            "non-negative.");
    }

    for (size_t owned = 0; owned < owned_values.size(); ++owned)
    {
        target.set_owned_value(
            static_cast<local_ordinal_type>(owned),
            owned_values[owned]);
    }
    target.sync_ghosts();
}

} // namespace detail

/**
 * @brief Compute volume-weighted averages over coarse feedback cells.
 *
 * @tparam Pack Tpetra type pack used by the input field.
 * @tparam Field Legacy CellField or native scalar cell FieldStored type.
 * @param field Fine-mesh scalar field to average.
 * @param feedback_cells Coarse cells and their rank-local owned fine-cell
 *        LIDs. The number and ordering of coarse cells must match on every
 *        rank in the field communicator.
 * @return One globally reduced averaged value per feedback cell, replicated
 *         on every rank.
 * @note This operation is collective over the field communicator.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class Field>
    requires detail::ScalarFeedbackField<Field, Pack>
std::vector<typename Pack::scalar_type>
volume_weighted_average(
    const Field& field,
    const std::vector<FeedbackCell<Pack>>& feedback_cells)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    const auto& mesh = field.mesh();
    const auto comm = mesh.owned_cell_map()->getComm();

    std::vector<std::string> feedback_cell_names;
    feedback_cell_names.reserve(feedback_cells.size());
    for (const auto& feedback_cell : feedback_cells)
    {
        feedback_cell_names.push_back(feedback_cell.name);
    }
    detail::validate_collective_names(
        *comm,
        feedback_cell_names,
        "Feedback mapping requires identical coarse-cell names and "
        "ordering");
    if (feedback_cells.empty())
    {
        return {};
    }

    const auto local_count = static_cast<int>(feedback_cells.size());

    int local_has_invalid_lid = 0;
    for (const auto& feedback_cell : feedback_cells)
    {
        for (const auto cell_lid : feedback_cell.cell_lids)
        {
            bool valid_lid = true;
            if constexpr (std::is_signed_v<local_ordinal_type>)
            {
                valid_lid = cell_lid >= local_ordinal_type{};
            }
            if (valid_lid)
            {
                valid_lid =
                    static_cast<size_t>(cell_lid) < mesh.num_local_cells();
            }
            if (valid_lid)
            {
                valid_lid = mesh.is_owned_cell(cell_lid);
            }
            if (!valid_lid)
            {
                local_has_invalid_lid = 1;
                break;
            }
        }
        if (local_has_invalid_lid != 0)
        {
            break;
        }
    }

    int global_has_invalid_lid = 0;
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MAX,
        1,
        &local_has_invalid_lid,
        &global_has_invalid_lid);
    if (global_has_invalid_lid != 0)
    {
        throw std::invalid_argument(
            "Feedback mapping expects each rank to list only its owned "
            "cell LIDs.");
    }

    std::vector<scalar_type> local_weighted_sums(
        feedback_cells.size(), scalar_type{});
    std::vector<scalar_type> local_volume_sums(
        feedback_cells.size(), scalar_type{});

    std::array<int, 2> local_input_validity{1, 1};
    for (const auto& feedback_cell : feedback_cells)
    {
        for (const auto cell_lid : feedback_cell.cell_lids)
        {
            const auto value = field.value(cell_lid);
            const auto volume = mesh.cell_volume(cell_lid);
            if (!std::isfinite(value))
            {
                local_input_validity[0] = 0;
            }
            if (!std::isfinite(volume) || volume <= scalar_type{})
            {
                local_input_validity[1] = 0;
            }
        }
    }
    std::array<int, 2> global_input_validity{};
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        static_cast<int>(local_input_validity.size()),
        local_input_validity.data(),
        global_input_validity.data());
    if (global_input_validity[0] == 0)
    {
        throw std::invalid_argument(
            "Feedback mapping requires finite field values.");
    }
    if (global_input_validity[1] == 0)
    {
        throw std::invalid_argument(
            "Feedback mapping requires finite positive cell volumes.");
    }

    for (size_t feedback_id = 0;
         feedback_id < feedback_cells.size();
         ++feedback_id)
    {
        const auto& feedback_cell = feedback_cells[feedback_id];
        for (const auto cell_lid : feedback_cell.cell_lids)
        {
            const auto volume = mesh.cell_volume(cell_lid);
            local_weighted_sums[feedback_id] +=
                field.value(cell_lid) * volume;
            local_volume_sums[feedback_id] += volume;
        }
    }

    std::vector<scalar_type> global_weighted_sums(
        feedback_cells.size(), scalar_type{});
    std::vector<scalar_type> global_volume_sums(
        feedback_cells.size(), scalar_type{});
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_SUM,
        local_count,
        local_weighted_sums.data(),
        global_weighted_sums.data());
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_SUM,
        local_count,
        local_volume_sums.data(),
        global_volume_sums.data());

    std::vector<scalar_type> result;
    result.reserve(feedback_cells.size());
    for (size_t feedback_id = 0;
         feedback_id < feedback_cells.size();
         ++feedback_id)
    {
        if (!std::isfinite(global_volume_sums[feedback_id])
            || global_volume_sums[feedback_id] <= scalar_type{})
        {
            throw std::invalid_argument(
                "Feedback cell '" + feedback_cells[feedback_id].name
                + "' requires a finite positive mapped volume.");
        }
        if (!std::isfinite(global_weighted_sums[feedback_id]))
        {
            throw std::invalid_argument(
                "Feedback cell '" + feedback_cells[feedback_id].name
                + "' has a non-finite weighted field sum.");
        }
        const auto average =
            global_weighted_sums[feedback_id]
            / global_volume_sums[feedback_id];
        if (!std::isfinite(average))
        {
            throw std::invalid_argument(
                "Feedback cell '" + feedback_cells[feedback_id].name
                + "' has a non-finite mapped field value.");
        }
        result.push_back(average);
    }
    return result;
}

/**
 * @brief Import one owned power-density value per target cell.
 *
 * @tparam Field Legacy CellField or native scalar cell FieldStored type.
 * @param target Field receiving the imported values.
 * @param owned_values Values ordered by owned cell local index.
 */
template<detail::DeducibleMutableScalarFeedbackField Field>
void import_power_density(
    Field& target,
    const std::vector<typename Field::scalar_type>& owned_values)
{
    detail::import_power_density(target, owned_values);
}

/** @brief Backward-compatible overload for explicit pack template calls. */
template<TpetraTypePack Pack, class Field>
    requires detail::MutableScalarFeedbackField<Field, Pack>
void import_power_density(
    Field& target,
    const std::vector<typename Pack::scalar_type>& owned_values)
{
    detail::import_power_density(target, owned_values);
}

/** @brief Canonical external names used by the multiphysics feedback map. */
inline constexpr std::string_view liquid_temperature_name = "T_liquid";
inline constexpr std::string_view gas_fraction_name = "alpha_g";
inline constexpr std::string_view density_feedback_name = "rhoFeedback";

/**
 * @brief Return the canonical one-based delayed-precursor field name.
 * @param one_based_group Delayed-neutron group number, starting at one.
 */
inline std::string precursor_name(size_t one_based_group)
{
    if (one_based_group == 0)
    {
        throw std::invalid_argument(
            "Feedback precursor group numbers start at one.");
    }
    return "C_" + std::to_string(one_based_group);
}

/**
 * @brief Deterministic in-memory export of coarse feedback values.
 *
 * Coarse-cell names retain the input mapping order. Fields use an ordered map,
 * giving every rank and callback the same lexicographic field order.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class MappedFeedbackSnapshot
{
public:
    using scalar_type = typename Pack::scalar_type;
    using field_map_type =
        std::map<std::string, std::vector<scalar_type>>;

    MappedFeedbackSnapshot(
        size_t sequence_index,
        std::vector<std::string> feedback_cell_names,
        field_map_type fields)
        : d_sequence_index(sequence_index),
          d_feedback_cell_names(std::move(feedback_cell_names)),
          d_fields(std::move(fields))
    {
        for (const auto& [name, values] : d_fields)
        {
            if (values.size() != d_feedback_cell_names.size())
            {
                throw std::invalid_argument(
                    "Mapped feedback field '" + name
                    + "' must have one value per feedback cell.");
            }
            for (const auto value : values)
            {
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument(
                        "Mapped feedback field '" + name
                        + "' contains a non-finite value.");
                }
            }
        }
    }

    size_t sequence_index() const noexcept { return d_sequence_index; }

    const std::vector<std::string>& feedback_cell_names() const noexcept
    {
        return d_feedback_cell_names;
    }

    const field_map_type& fields() const noexcept { return d_fields; }

    const std::vector<scalar_type>& field(std::string_view name) const
    {
        const auto iterator = d_fields.find(std::string(name));
        if (iterator == d_fields.end())
        {
            throw std::out_of_range(
                "Mapped feedback snapshot has no field named '"
                + std::string(name) + "'.");
        }
        return iterator->second;
    }

private:
    size_t d_sequence_index = 0;
    std::vector<std::string> d_feedback_cell_names;
    field_map_type d_fields;
};

/**
 * @brief Type-erased registry of scalar fields exported to neutronics.
 *
 * Registration accepts either legacy CellField or native FieldStored scalar
 * cell fields. Registered fields are referenced, not owned, and therefore must
 * outlive the registry and every snapshot operation.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class FeedbackFieldRegistry
{
public:
    using scalar_type = typename Pack::scalar_type;
    using comm_type = typename Pack::comm_type;
    using feedback_cell_type = FeedbackCell<Pack>;
    using snapshot_type = MappedFeedbackSnapshot<Pack>;
    using mapper_type = std::function<std::vector<scalar_type>(
        const std::vector<feedback_cell_type>&)>;

    /** @brief Bind the registry to one fine mesh and its communicator. */
    template<class MeshType>
    explicit FeedbackFieldRegistry(const MeshType& mesh)
        requires requires(const MeshType& candidate)
        {
            { candidate.owned_cell_map() };
        }
        : d_mesh_identity(std::addressof(mesh)),
          d_comm(mesh.owned_cell_map()->getComm())
    {
        if (d_comm.is_null())
        {
            throw std::invalid_argument(
                "Feedback field registry requires a mesh communicator.");
        }
    }

    /** @brief Register a scalar field under an explicit external name. */
    template<class Field>
        requires detail::ScalarFeedbackField<Field, Pack>
    void register_field(std::string name, const Field& field)
    {
        if (name.empty())
        {
            throw std::invalid_argument(
                "Feedback field names cannot be empty.");
        }
        if (std::addressof(field.mesh()) != d_mesh_identity)
        {
            throw std::invalid_argument(
                "Feedback fields must use the registry mesh.");
        }
        const auto* field_pointer = std::addressof(field);
        const auto [iterator, inserted] = d_mappers.emplace(
            std::move(name),
            [field_pointer](const auto& feedback_cells)
            {
                return volume_weighted_average<Pack>(
                    *field_pointer, feedback_cells);
            });
        (void)iterator;
        if (!inserted)
        {
            throw std::invalid_argument(
                "Feedback field names must be unique.");
        }
    }

    template<class Field>
        requires detail::ScalarFeedbackField<Field, Pack>
    void register_liquid_temperature(const Field& field)
    {
        register_field(std::string(liquid_temperature_name), field);
    }

    template<class Field>
        requires detail::ScalarFeedbackField<Field, Pack>
    void register_gas_fraction(const Field& field)
    {
        register_field(std::string(gas_fraction_name), field);
    }

    template<class Field>
        requires detail::ScalarFeedbackField<Field, Pack>
    void register_density_feedback(const Field& field)
    {
        register_field(std::string(density_feedback_name), field);
    }

    template<class Field>
        requires detail::ScalarFeedbackField<Field, Pack>
    void register_precursor_group(
        size_t one_based_group,
        const Field& field)
    {
        register_field(precursor_name(one_based_group), field);
    }

    bool contains(std::string_view name) const
    {
        return d_mappers.contains(std::string(name));
    }

    std::vector<std::string> field_names() const
    {
        std::vector<std::string> names;
        names.reserve(d_mappers.size());
        for (const auto& [name, mapper] : d_mappers)
        {
            (void)mapper;
            names.push_back(name);
        }
        return names;
    }

    /** @brief Require the standard TH fields and requested precursor groups. */
    void require_standard_fields(size_t precursor_group_count = 0) const
    {
        const auto collective_precursor_group_count =
            detail::validate_collective_size(
                *d_comm,
                precursor_group_count,
                "Feedback precursor group count");
        const auto names = field_names();
        detail::validate_collective_names(
            *d_comm,
            names,
            "Feedback field registry requires identical field names and "
            "ordering");

        std::vector<std::string> missing;
        for (const auto name : {liquid_temperature_name,
                                gas_fraction_name,
                                density_feedback_name})
        {
            if (!contains(name))
            {
                missing.emplace_back(name);
            }
        }
        for (size_t group = 1;
             group <= collective_precursor_group_count;
             ++group)
        {
            const auto name = precursor_name(group);
            if (!contains(name))
            {
                missing.push_back(name);
            }
        }
        if (!missing.empty())
        {
            std::string message =
                "Feedback registry is missing required field(s): ";
            for (size_t index = 0; index < missing.size(); ++index)
            {
                if (index != 0)
                {
                    message += ", ";
                }
                message += missing[index];
            }
            throw std::invalid_argument(message);
        }
    }

    /**
     * @brief Collectively map every registered field into one memory snapshot.
     */
    snapshot_type export_snapshot(
        const std::vector<feedback_cell_type>& feedback_cells,
        size_t sequence_index = 0) const
    {
        const auto collective_sequence_index =
            detail::validate_collective_size(
                *d_comm,
                sequence_index,
                "Feedback snapshot sequence index");
        const auto names = field_names();
        detail::validate_collective_names(
            *d_comm,
            names,
            "Feedback field registry requires identical field names and "
            "ordering");

        typename snapshot_type::field_map_type fields;
        for (const auto& [name, mapper] : d_mappers)
        {
            fields.emplace(name, mapper(feedback_cells));
        }

        std::vector<std::string> cell_names;
        cell_names.reserve(feedback_cells.size());
        for (const auto& cell : feedback_cells)
        {
            cell_names.push_back(cell.name);
        }
        return snapshot_type(
            collective_sequence_index,
            std::move(cell_names),
            std::move(fields));
    }

    template<class MeshType>
    bool uses_mesh(const MeshType& mesh) const noexcept
    {
        return std::addressof(mesh) == d_mesh_identity;
    }

    /** @brief Communicator shared by every collective registry operation. */
    const comm_type& communicator() const noexcept
    {
        return *d_comm;
    }

private:
    const void* d_mesh_identity = nullptr;
    Teuchos::RCP<const comm_type> d_comm;
    std::map<std::string, mapper_type> d_mappers;
};

/** @brief Controls the deterministic placeholder outer iteration. */
struct PlaceholderOuterCouplingOptions
{
    size_t outer_iterations = 1;
    size_t thermal_hydraulic_subcycles = 1;
    size_t precursor_group_count = 0;
};

/** @brief Observable record from one placeholder outer iteration. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
struct PlaceholderOuterCouplingRecord
{
    size_t iteration = 0;
    std::vector<typename Pack::scalar_type> power_applied;
    MappedFeedbackSnapshot<Pack> feedback;
    std::vector<typename Pack::scalar_type> power_returned;
};

/**
 * @brief Minimal callback-driven thermal-hydraulic/neutronics coupling loop.
 *
 * The driver imports initial owned-cell fission power, advances the caller's
 * TH callback for a fixed number of subcycles, exports a coarse feedback
 * snapshot, calls a placeholder neutronics update, imports its returned power,
 * and repeats. It deliberately owns neither solver so a production transport
 * interface can replace either callback without changing the exchange order.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class PlaceholderOuterCouplingDriver
{
public:
    using scalar_type = typename Pack::scalar_type;
    using registry_type = FeedbackFieldRegistry<Pack>;
    using feedback_cell_type = FeedbackCell<Pack>;
    using record_type = PlaceholderOuterCouplingRecord<Pack>;

    PlaceholderOuterCouplingDriver(
        const registry_type& registry,
        std::vector<feedback_cell_type> feedback_cells,
        PlaceholderOuterCouplingOptions options = {})
        : d_registry(registry),
          d_feedback_cells(std::move(feedback_cells)),
          d_options(options)
    {
        const auto outer_iterations = detail::validate_collective_size(
            d_registry.communicator(),
            d_options.outer_iterations,
            "Outer coupling iteration count");
        const auto thermal_hydraulic_subcycles =
            detail::validate_collective_size(
                d_registry.communicator(),
                d_options.thermal_hydraulic_subcycles,
                "Outer coupling TH subcycle count");
        const auto precursor_group_count =
            detail::validate_collective_size(
                d_registry.communicator(),
                d_options.precursor_group_count,
                "Outer coupling precursor group count");
        if (outer_iterations == 0)
        {
            throw std::invalid_argument(
                "Outer coupling requires at least one iteration.");
        }
        if (thermal_hydraulic_subcycles == 0)
        {
            throw std::invalid_argument(
                "Outer coupling requires at least one TH subcycle.");
        }
        d_registry.require_standard_fields(precursor_group_count);
    }

    /**
     * @brief Execute the placeholder coupling exchange.
     *
     * @param power_density Mutable fine-mesh qdot_fission field.
     * @param initial_owned_power Initial values ordered by owned cell LID.
     * @param advance_thermal_hydraulics Callable `(iteration, subcycle)`.
     * @param update_neutronics Callable `(snapshot)` returning the next owned
     *        qdot_fission vector.
     * @warning A callback must not allow an exception to escape while peer
     *          ranks remain inside a callback-owned collective. The driver
     *          synchronizes exceptions from rank-local callback work after
     *          each callback returns or throws.
     */
    template<class PowerField,
             class ThermalHydraulicsAdvance,
             class NeutronicsUpdate>
        requires detail::MutableScalarFeedbackField<PowerField, Pack>
    std::vector<record_type> run(
        PowerField& power_density,
        std::vector<scalar_type> initial_owned_power,
        ThermalHydraulicsAdvance&& advance_thermal_hydraulics,
        NeutronicsUpdate&& update_neutronics) const
    {
        int local_mesh_matches =
            d_registry.uses_mesh(power_density.mesh()) ? 1 : 0;
        int global_mesh_matches = 0;
        Teuchos::reduceAll(
            d_registry.communicator(),
            Teuchos::REDUCE_MIN,
            1,
            &local_mesh_matches,
            &global_mesh_matches);
        if (global_mesh_matches == 0)
        {
            throw std::invalid_argument(
                "Outer-coupling power and feedback fields must share a mesh.");
        }

        // Validate the fixed map before power import or TH callbacks can
        // partially advance the coupled state.
        (void)d_registry.export_snapshot(d_feedback_cells);
        import_power_density<Pack>(
            power_density, initial_owned_power);
        auto current_power = std::move(initial_owned_power);

        std::vector<record_type> records;
        records.reserve(d_options.outer_iterations);
        for (size_t iteration = 0;
             iteration < d_options.outer_iterations;
             ++iteration)
        {
            for (size_t subcycle = 0;
                 subcycle < d_options.thermal_hydraulic_subcycles;
                 ++subcycle)
            {
                bool local_callback_failed = false;
                try
                {
                    std::invoke(
                        advance_thermal_hydraulics,
                        iteration,
                        subcycle);
                }
                catch (...)
                {
                    local_callback_failed = true;
                }
                detail::throw_if_collective_callback_failed(
                    d_registry.communicator(),
                    local_callback_failed,
                    "thermal-hydraulic callback");
            }

            auto feedback = d_registry.export_snapshot(
                d_feedback_cells, iteration);
            std::optional<std::vector<scalar_type>> next_power_result;
            bool local_callback_failed = false;
            try
            {
                next_power_result.emplace(std::invoke(
                    update_neutronics,
                    std::as_const(feedback)));
            }
            catch (...)
            {
                local_callback_failed = true;
            }
            detail::throw_if_collective_callback_failed(
                d_registry.communicator(),
                local_callback_failed,
                "neutronics callback");
            auto next_power = std::move(*next_power_result);
            import_power_density<Pack>(power_density, next_power);

            records.push_back(record_type{
                iteration,
                current_power,
                std::move(feedback),
                next_power});
            current_power = std::move(next_power);
        }
        return records;
    }

private:
    const registry_type& d_registry;
    std::vector<feedback_cell_type> d_feedback_cells;
    PlaceholderOuterCouplingOptions d_options;
};

} // namespace SimpleFluid::FeedbackMap
