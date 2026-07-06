/**
 * @file FieldInstantiations.cc
 * @brief Explicit template instantiations for default SimpleFluid fields.
 */

#include "fields/BoundaryFaceField.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"

namespace SimpleFluid
{

template class CellField<DefaultTpetraTypes>;

template class VectorCellField<DefaultTpetraTypes>;

template class TensorCellField<DefaultTpetraTypes>;

template class FaceField<DefaultTpetraTypes>;

template class VectorFaceField<DefaultTpetraTypes>;

template class BoundaryFaceField<DefaultTpetraTypes>;

} // namespace SimpleFluid
