/** @file VectorCellField.hh
 * @brief Three-component cell storage on the native runtime mesh.
 */
#pragma once

#include "fields/FieldStored.hh"

namespace SimpleFluid
{

/** @brief Three-component cell storage on the native runtime mesh. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
using VectorCellField = VectorCellFieldStored<Pack>;

} // namespace SimpleFluid
