/** @file VariableDensityMomentumEquation.hh
 * @brief Conservative density-weighted momentum transport with canonical face mass flux.
 */
#pragma once

#include "FVM/TransportSystem.hh"
#include "SimpleFluidExport.hh"
#include "equations/CollectiveValidation.hh"
#include "equations/EquationValidation.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"

namespace SimpleFluid
{
/** Advance rho*u using accepted/trial density and a caller-owned conservative mass flux.
 * Boundary callbacks supply one scalar velocity component. Force is in N/m3;
 * pressure, gravity and non-Laplacian deviatoric-stress terms belong to the caller.
 * The mass flux is kg/s, including ALE relative transport, and is not multiplied
 * by density again. Density and viscosity inputs must have synchronized ghosts.
 * All ranks participate, including empty owners. Only successful solves publish velocity.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = MeshHandle<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT VariableDensityMomentumEquation
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using field_type = typename MeshFieldTraits<Pack, MeshType>::scalar_cell_type;
    using velocity_field_type = typename MeshFieldTraits<Pack, MeshType>::vector_cell_type;
    using face_field_type = typename MeshFieldTraits<Pack, MeshType>::scalar_face_type;
    using vector_type = typename velocity_field_type::vec_type;
    using boundary_type = std::function<BoundaryCondition(int, size_t, size_t)>;
    using boundary_value_type = std::function<scalar_type(int, size_t, size_t)>;
    using force_type = std::function<vector_type(local_ordinal_type)>;

    explicit VariableDensityMomentumEquation(SP<const MeshType> mesh);
    void refresh_geometry();
    LinearSolveSummary advance(const velocity_field_type& old_velocity, const field_type& old_density,
        const field_type& density, const face_field_type& mass_flux, const field_type& dynamic_viscosity,
        scalar_type time_step, velocity_field_type& velocity, force_type force = {}, boundary_type boundary = {},
        boundary_value_type boundary_value = {}, const LinearSolverOptions& linear = {},
        FVM::NonOrthogonalTreatment treatment = FVM::NonOrthogonalTreatment::Implicit,
        const FVM::ALEControlVolumeState* ale = nullptr);

private:
    SP<const MeshType> d_mesh;
    field_type d_old_component, d_component, d_unit;
    velocity_field_type d_candidate, d_force;
    FVM::TransportGeometryCache<MeshType> d_geometry;
    BelosLinearSolver<Pack> d_solver;
};
extern template class VariableDensityMomentumEquation<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class VariableDensityMomentumEquation<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;
} // namespace SimpleFluid
