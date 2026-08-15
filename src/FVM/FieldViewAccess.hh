/**
 * @file FVM/FieldViewAccess.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Small field-view reconstruction helpers shared by FVM implementations.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"

namespace SimpleFluid::FVM::detail
{

/** @brief Reconstruct one three-component value from a cached field view. */
template<TpetraTypePack Pack, class View>
inline auto vector_view_value(const View& values, typename Pack::local_ordinal_type cell_lid) ->
    typename VectorCellField<Pack>::vec_type
{
    return {values(cell_lid, 0), values(cell_lid, 1), values(cell_lid, 2)};
}

/** @brief Reconstruct one row-major 3x3 value from a cached field view. */
template<TpetraTypePack Pack, class View>
inline auto tensor_view_value(const View& values, typename Pack::local_ordinal_type cell_lid) ->
    typename TensorCellField<Pack>::tensor_type
{
    return {typename VectorCellField<Pack>::vec_type{values(cell_lid, 0), values(cell_lid, 1), values(cell_lid, 2)},
        typename VectorCellField<Pack>::vec_type{values(cell_lid, 3), values(cell_lid, 4), values(cell_lid, 5)},
        typename VectorCellField<Pack>::vec_type{values(cell_lid, 6), values(cell_lid, 7), values(cell_lid, 8)}};
}

} // namespace SimpleFluid::FVM::detail
