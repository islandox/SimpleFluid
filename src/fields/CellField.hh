/** @file CellField.hh
 * @brief Scalar cell storage on the native runtime mesh.
 */
#pragma once

#include "fields/FieldStored.hh"

namespace SimpleFluid
{

/** @brief Scalar cell storage on the native runtime mesh. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
using CellField = ScalarCellFieldStored<Pack>;

} // namespace SimpleFluid
