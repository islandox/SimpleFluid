/**
 * @file EquationForward.hh
 * @brief Visibility-consistent forward declarations for compiled equations.
 */
#pragma once
#include "utils/export_helper.hh"

#include "geometry/MeshConcepts.hh"
#include "dataclass/TpetraTypes.hh"
#include "geometry/MeshHandle.hh"

namespace SimpleFluid
{

template<TpetraTypePack Pack = DefaultTpetraTypes, SimpleFluid::MeshType MeshType = MeshHandle<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT BoussinesqMomentumEquation;

template<TpetraTypePack Pack = DefaultTpetraTypes, SimpleFluid::MeshType MeshType = MeshHandle<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT IncompressibleMomentumEquation;

} // namespace SimpleFluid
