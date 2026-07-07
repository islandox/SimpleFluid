/**
 * @file FeedbackMap.hh
 * @brief Utilities for thermal-hydraulic feedback mapping.
 */
#pragma once

#include "fields/CellField.hh"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace SimpleFluid::FeedbackMap
{

/**
 * @brief Coarse feedback cell described by owned fine-cell local IDs.
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
 * @param feedback_cells Coarse cells and their owned fine-cell LIDs.
 * @return One averaged value per feedback cell.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
std::vector<typename Pack::scalar_type>
volume_weighted_average(
    const CellField<Pack>& field,
    const std::vector<FeedbackCell<Pack>>& feedback_cells)
{
    using scalar_type = typename Pack::scalar_type;
    std::vector<scalar_type> result;
    result.reserve(feedback_cells.size());
    const auto& mesh = field.mesh();

    for (const auto& feedback_cell : feedback_cells)
    {
        scalar_type weighted_sum{};
        scalar_type volume_sum{};
        for (const auto cell_lid : feedback_cell.cell_lids)
        {
            if (!mesh.is_owned_cell(cell_lid))
            {
                throw std::invalid_argument(
                    "Feedback mapping currently expects owned cell LIDs.");
            }
            const auto volume = mesh.cell_volume(cell_lid);
            weighted_sum += field.value(cell_lid) * volume;
            volume_sum += volume;
        }
        if (volume_sum <= scalar_type{})
        {
            throw std::invalid_argument(
                "Feedback cell '" + feedback_cell.name
                + "' has zero mapped volume.");
        }
        result.push_back(weighted_sum / volume_sum);
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
