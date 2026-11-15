/** @file VectorFaceField.hh
 * @brief Three-component face storage on the native runtime mesh.
 */
#pragma once

#include "fields/FieldStored.hh"

namespace SimpleFluid
{

/** @brief Three-component face storage on the native runtime mesh. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
using VectorFaceField = VectorFaceFieldStored<Pack>;

} // namespace SimpleFluid
