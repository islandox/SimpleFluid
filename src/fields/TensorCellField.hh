/** @file TensorCellField.hh
 * @brief Row-major nine-component cell storage on the native runtime mesh.
 */
#pragma once

#include "fields/FieldStored.hh"

namespace SimpleFluid
{

/** @brief Row-major nine-component cell storage on the native runtime mesh. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
using TensorCellField = TensorCellFieldStored<Pack>;

} // namespace SimpleFluid
