/** @brief Compiled mapped diffusion provider for SolidSubdomain. */
#include "FVM/FieldStoredNonOrthogonalDiffusion.tcc"
#include "geometry/SolidSubdomain.hh"

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using MeshType = SolidSubdomain<Pack>;
} // namespace

template DiffusionSystem<Pack> non_orthogonal_diffusion_system<Pack, MeshType>(const MeshType& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, NonOrthogonalTreatment treatment,
    const ScalarCellFieldStored<Pack, MeshType>* correction_field);

template DiffusionSystem<Pack> implicit_non_orthogonal_diffusion_system<Pack, MeshType>(const MeshType& mesh,
    typename Pack::scalar_type diffusivity, ScalarBoundaryConditionProvider<Pack> boundary_condition,
    ScalarCellValueProvider<Pack> right_hand_source, typename Pack::scalar_type implicit_weight,
    const ScalarCellFieldStored<Pack, MeshType>* partition_correction_field);

} // namespace SimpleFluid::FVM
