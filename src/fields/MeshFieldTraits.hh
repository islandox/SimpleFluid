/**
 * @file MeshFieldTraits.hh
 * @brief Select field storage for legacy and mapped mesh interfaces.
 */

#pragma once

#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/FieldStored.hh"
#include "fields/TensorCellField.hh"
#include "fields/VectorCellField.hh"
#include "fields/VectorFaceField.hh"
#include "geometry/Mesh.hh"

namespace SimpleFluid
{

/**
 * @brief Field family associated with an FVM mesh interface.
 *
 * Mapped CRTP meshes and MeshHandle use FieldStored.  The specialization for
 * the legacy STK Mesh preserves the established field API for callers that
 * construct equations directly on that backend.
 */
template<TpetraTypePack Pack, class MeshType> struct MeshFieldTraits
{
    using scalar_cell_type = ScalarCellFieldStored<Pack, MeshType>;
    using vector_cell_type = VectorCellFieldStored<Pack, MeshType>;
    using tensor_cell_type = TensorCellFieldStored<Pack, MeshType>;
    using scalar_face_type = ScalarFaceFieldStored<Pack, MeshType>;
    using vector_face_type = VectorFaceFieldStored<Pack, MeshType>;
};

template<TpetraTypePack Pack> struct MeshFieldTraits<Pack, Mesh<Pack>>
{
    using scalar_cell_type = CellField<Pack>;
    using vector_cell_type = VectorCellField<Pack>;
    using tensor_cell_type = TensorCellField<Pack>;
    using scalar_face_type = FaceField<Pack>;
    using vector_face_type = VectorFaceField<Pack>;
};

} // namespace SimpleFluid
