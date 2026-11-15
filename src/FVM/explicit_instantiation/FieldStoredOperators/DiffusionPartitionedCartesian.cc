/** @brief Compiled mapped diffusion provider for PartitionedCartesian. */
#include "FVM/FieldStoredNonOrthogonalDiffusion.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

// Only these public static-mesh specializations override hidden argument
// visibility. GNU attributes on instantiations are accepted by GCC and Clang;
// standard [[gnu::visibility]] attributes are not valid there in Clang.
#if defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_STATIC_OPERATOR_EXPORT __attribute__((visibility("default")))
#else
#define SIMPLEFLUID_STATIC_OPERATOR_EXPORT
#endif

namespace SimpleFluid::FVM
{
namespace
{
using Pack = DefaultTpetraTypes;
using Mesh = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>;
} // namespace

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT DiffusionSystem<Pack> non_orthogonal_diffusion_system<Pack, Mesh>(
    const Mesh& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    NonOrthogonalTreatment treatment, const ScalarCellFieldStored<Pack, Mesh>* correction_field);

template SIMPLEFLUID_STATIC_OPERATOR_EXPORT DiffusionSystem<Pack>
implicit_non_orthogonal_diffusion_system<Pack, Mesh>(const Mesh& mesh, typename Pack::scalar_type diffusivity,
    ScalarBoundaryConditionProvider<Pack> boundary_condition, ScalarCellValueProvider<Pack> right_hand_source,
    typename Pack::scalar_type implicit_weight,
    const ScalarCellFieldStored<Pack, Mesh>* partition_correction_field);

} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_OPERATOR_EXPORT
