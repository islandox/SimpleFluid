/**
 * @file FeedbackMap.hh
 * @brief Utilities for thermal-hydraulic feedback mapping.
 */
#pragma once

#include "fields/CellField.hh"

#include <Teuchos_CommHelpers.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
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

/**
 * @brief Compute volume-weighted averages over coarse feedback cells.
 *
 * @tparam Pack Tpetra type pack used by the input field.
 * @param field Fine-mesh scalar field to average.
 * @param feedback_cells Coarse cells and their rank-local owned fine-cell
 *        LIDs. The number and ordering of coarse cells must match on every
 *        rank in the field communicator.
 * @return One globally reduced averaged value per feedback cell, replicated
 *         on every rank.
 * @note This operation is collective over the field communicator.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
std::vector<typename Pack::scalar_type>
volume_weighted_average(
    const CellField<Pack>& field,
    const std::vector<FeedbackCell<Pack>>& feedback_cells)
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    const auto& mesh = field.mesh();
    const auto comm = mesh.owned_cell_map()->getComm();

    int local_count_is_valid =
        feedback_cells.size()
                <= static_cast<size_t>(std::numeric_limits<int>::max())
            ? 1
            : 0;
    int global_count_is_valid = 0;
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_count_is_valid,
        &global_count_is_valid);
    if (global_count_is_valid == 0)
    {
        throw std::invalid_argument(
            "Feedback mapping has too many coarse cells for a collective "
            "reduction.");
    }

    const auto local_count = static_cast<int>(feedback_cells.size());
    int minimum_count = 0;
    int maximum_count = 0;
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_count,
        &minimum_count);
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MAX,
        1,
        &local_count,
        &maximum_count);
    if (minimum_count != maximum_count)
    {
        throw std::invalid_argument(
            "Feedback mapping requires the same ordered coarse cells on "
            "every rank.");
    }
    if (feedback_cells.empty())
    {
        return {};
    }

    // Encode both name boundaries and contents so equal-length vectors with
    // different names or ordering cannot silently cross-mix coarse regions.
    std::vector<int> local_name_encoding;
    for (const auto& feedback_cell : feedback_cells)
    {
        const auto name_size =
            static_cast<std::uint64_t>(feedback_cell.name.size());
        for (int shift = 56; shift >= 0; shift -= 8)
        {
            local_name_encoding.push_back(
                static_cast<int>((name_size >> shift) & 0xffU));
        }
        for (const unsigned char byte : feedback_cell.name)
        {
            local_name_encoding.push_back(static_cast<int>(byte));
        }
    }

    int local_name_size_is_valid =
        local_name_encoding.size()
                <= static_cast<size_t>(std::numeric_limits<int>::max())
            ? 1
            : 0;
    int global_name_size_is_valid = 0;
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_name_size_is_valid,
        &global_name_size_is_valid);
    if (global_name_size_is_valid == 0)
    {
        throw std::invalid_argument(
            "Feedback coarse-cell names are too large for a collective "
            "consistency check.");
    }

    const auto local_name_size =
        static_cast<int>(local_name_encoding.size());
    int minimum_name_size = 0;
    int maximum_name_size = 0;
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        1,
        &local_name_size,
        &minimum_name_size);
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MAX,
        1,
        &local_name_size,
        &maximum_name_size);
    if (minimum_name_size != maximum_name_size)
    {
        throw std::invalid_argument(
            "Feedback mapping requires identical coarse-cell names and "
            "ordering on every rank.");
    }

    std::vector<int> minimum_name_encoding(
        local_name_encoding.size(), 0);
    std::vector<int> maximum_name_encoding(
        local_name_encoding.size(), 0);
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MIN,
        local_name_size,
        local_name_encoding.data(),
        minimum_name_encoding.data());
    Teuchos::reduceAll(
        *comm,
        Teuchos::REDUCE_MAX,
        local_name_size,
        local_name_encoding.data(),
        maximum_name_encoding.data());
    if (minimum_name_encoding != maximum_name_encoding)
    {
        throw std::invalid_argument(
            "Feedback mapping requires identical coarse-cell names and "
            "ordering on every rank.");
    }

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
        if (global_volume_sums[feedback_id] <= scalar_type{})
        {
            throw std::invalid_argument(
                "Feedback cell '" + feedback_cells[feedback_id].name
                + "' has zero mapped volume.");
        }
        result.push_back(
            global_weighted_sums[feedback_id]
            / global_volume_sums[feedback_id]);
    }
    return result;
}

/**
 * @brief Import one owned power-density value per target cell.
 *
 * @tparam Pack Tpetra type pack used by the target field.
 * @param target Field receiving the imported values.
 * @param owned_values Values ordered by owned cell local index.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
void import_power_density(
    CellField<Pack>& target,
    const std::vector<typename Pack::scalar_type>& owned_values)
{
    if (owned_values.size() != target.mesh().num_owned_cells())
    {
        throw std::invalid_argument(
            "Imported power-density field size must match owned cells.");
    }
    for (size_t owned = 0; owned < owned_values.size(); ++owned)
    {
        const auto value = owned_values[owned];
        if (!std::isfinite(value)
            || value < static_cast<typename Pack::scalar_type>(0))
        {
            throw std::invalid_argument(
                "Imported power-density values must be finite and "
                "non-negative.");
        }
        target.set_owned_value(
            static_cast<typename Pack::local_ordinal_type>(owned),
            value);
    }
    target.sync_ghosts();
}

} // namespace SimpleFluid::FeedbackMap
