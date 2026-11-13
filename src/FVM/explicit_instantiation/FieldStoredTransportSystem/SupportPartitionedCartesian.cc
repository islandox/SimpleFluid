/** @brief Compiled ALE and gradient support for static Cartesian transport. */
#include "FVM/ALEControlVolumeState.tcc"
#include "FVM/CellGradientCache.tcc"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"

// This static mesh type is intentionally hidden. Export only these public
// function specializations; standard attributes on explicit instantiations
// are rejected by Clang, so use its supported GNU attribute spelling here.
#if defined(__GNUC__) || defined(__clang__)
#define SIMPLEFLUID_STATIC_TRANSPORT_EXPORT __attribute__((visibility("default")))
#else
#define SIMPLEFLUID_STATIC_TRANSPORT_EXPORT
#endif

namespace SimpleFluid::FVM
{
using PartitionedCartesian = Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>;
template SIMPLEFLUID_STATIC_TRANSPORT_EXPORT void ALEControlVolumeState::validate(const PartitionedCartesian&) const;
#if defined(__GNUC__) || defined(__clang__)
template class __attribute__((visibility("default"))) CellGradientCache<DefaultTpetraTypes, PartitionedCartesian>;
#else
template class CellGradientCache<DefaultTpetraTypes, PartitionedCartesian>;
#endif
} // namespace SimpleFluid::FVM

#undef SIMPLEFLUID_STATIC_TRANSPORT_EXPORT
