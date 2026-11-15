/** @file BoundaryFaceField.hh
 * @brief Scalar boundary-face storage on the native runtime mesh.
 */
#pragma once

#include "fields/FieldStored.hh"

namespace SimpleFluid
{

/** @brief Scalar boundary-face storage on the native runtime mesh. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
using BoundaryFaceField = ScalarBoundaryFaceFieldStored<Pack>;

} // namespace SimpleFluid
