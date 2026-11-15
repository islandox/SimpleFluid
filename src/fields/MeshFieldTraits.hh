/** @file MeshFieldTraits.hh
 * @brief Select native stored-field descriptors for a mesh interface.
 */
#pragma once

#include "fields/FieldStored.hh"
#include "geometry/MeshConcepts.hh"

namespace SimpleFluid
{

/** @brief Native field family associated with an FVM mesh interface. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType> struct MeshFieldTraits
{
    using scalar_cell_type = ScalarCellFieldStored<Pack, MeshType>;
    using vector_cell_type = VectorCellFieldStored<Pack, MeshType>;
    using tensor_cell_type = TensorCellFieldStored<Pack, MeshType>;
    using scalar_face_type = ScalarFaceFieldStored<Pack, MeshType>;
    using vector_face_type = VectorFaceFieldStored<Pack, MeshType>;
};

} // namespace SimpleFluid
