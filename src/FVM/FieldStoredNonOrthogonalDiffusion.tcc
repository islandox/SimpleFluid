/** @brief Compiled mapped nonorthogonal diffusion and custom-provider opt-in. */
#pragma once

#include "FVM/NonOrthogonalCorrection.hh"
#include "FVM/details/FieldStoredNonOrthogonalDiffusion.tcc"

namespace SimpleFluid::FVM
{

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT DiffusionSystem<Pack> non_orthogonal_diffusion_system(const MeshType& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field)
{
    return detail::stored_non_orthogonal_diffusion_system<Pack>(
        mesh, diffusivity, std::move(boundary_condition), std::move(right_hand_source), treatment, correction_field);
}

template<TpetraTypePack Pack, class MeshType>
SIMPLEFLUID_FVM_EXPORT DiffusionSystem<Pack> implicit_non_orthogonal_diffusion_system(const MeshType& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, typename Pack::scalar_type implicit_weight,
    const ScalarCellFieldStored<Pack, MeshType>* partition_correction_field)
{
    return detail::stored_implicit_diffusion_system<Pack>(mesh, diffusivity, std::move(boundary_condition),
        std::move(right_hand_source), implicit_weight, partition_correction_field);
}

} // namespace SimpleFluid::FVM
