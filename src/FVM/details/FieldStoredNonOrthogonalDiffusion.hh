/**
 * @file FVM/details/FieldStoredNonOrthogonalDiffusion.hh
 * @brief Non-orthogonal steady diffusion for mapped meshes and FieldStored.
 */

#pragma once

#include "SimpleFluidExport.hh"

#include "FVM/AssemblyCallbacks.hh"
#include "FVM/DiffusionSystem.hh"
#include "FVM/NonOrthogonalTreatment.hh"
#include "FVM/details/NonOrthogonalSelection.hh"
#include "fields/FieldStored.hh"

#include <string_view>

namespace SimpleFluid::FVM::detail
{

template<class Scalar> using StoredDiffusionWeights = NonOrthogonalWeights<Scalar>;

/** Collectively validate mapped non-orthogonal diffusion inputs. */
template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_LOCAL StoredDiffusionWeights<typename Pack::scalar_type> validate_stored_diffusion_selection(
    const MeshType& mesh, typename Pack::scalar_type diffusivity, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field, std::string_view context);

/** Evaluate interior least-squares gradients and synchronize overlap values. */
template<TpetraTypePack Pack, class MeshType, class Stencils>
SIMPLEFLUID_LOCAL void evaluate_stored_diffusion_gradients(const ScalarCellFieldStored<Pack, MeshType>& field,
    const Stencils& stencils, VectorCellFieldStored<Pack, MeshType>& gradients);

/** Add the lagged tangential diffusion contribution to a mapped RHS. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition>
SIMPLEFLUID_LOCAL void add_stored_explicit_diffusion_correction(
    const ScalarCellFieldStored<Pack, MeshType>& correction_field, typename Pack::scalar_type diffusivity,
    BoundaryCondition boundary_condition, typename Pack::vector_type& rhs,
    typename Pack::scalar_type correction_weight);

/** Assemble the implicit fraction of mapped steady non-orthogonal diffusion. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class Source>
SIMPLEFLUID_LOCAL DiffusionSystem<Pack> stored_implicit_diffusion_system(const MeshType& mesh,
    typename Pack::scalar_type diffusivity, BoundaryCondition boundary_condition, Source right_hand_source,
    typename Pack::scalar_type implicit_weight,
    const ScalarCellFieldStored<Pack, MeshType>* partition_correction_field);

/** Assemble mapped steady diffusion for a selected treatment. */
template<TpetraTypePack Pack, class MeshType, class BoundaryCondition, class Source>
SIMPLEFLUID_LOCAL DiffusionSystem<Pack> stored_non_orthogonal_diffusion_system(const MeshType& mesh,
    typename Pack::scalar_type diffusivity, BoundaryCondition boundary_condition, Source right_hand_source,
    NonOrthogonalTreatment treatment, const ScalarCellFieldStored<Pack, MeshType>* correction_field);

} // namespace SimpleFluid::FVM::detail
