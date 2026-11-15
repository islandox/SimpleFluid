/**
 * @file EquationForward.hh
 * @brief Visibility-consistent forward declarations for compiled equations.
 */
#pragma once
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN


#include "geometry/MeshConcepts.hh"
#include "SimpleFluidExport.hh"
#include "dataclass/TpetraTypes.hh"
#include "geometry/Mesh.hh"

namespace SimpleFluid
{

template<TpetraTypePack Pack = DefaultTpetraTypes, SimpleFluid::MeshType MeshType = Mesh<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT BoussinesqMomentumEquation;

template<TpetraTypePack Pack = DefaultTpetraTypes, SimpleFluid::MeshType MeshType = Mesh<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT IncompressibleMomentumEquation;

} // namespace SimpleFluid

SIMPLEFLUID_LEGACY_API_END
