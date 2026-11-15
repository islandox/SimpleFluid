/** @file FaceField.hh
 * @brief Scalar owned and overlap face storage on the native runtime mesh.
 */
#pragma once

#include "fields/FieldStored.hh"

namespace SimpleFluid
{

/** @brief Scalar owned and overlap face storage on the native runtime mesh. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
using FaceField = ScalarFaceFieldStored<Pack>;

} // namespace SimpleFluid
