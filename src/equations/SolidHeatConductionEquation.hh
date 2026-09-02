/**
 * @file SolidHeatConductionEquation.hh
 * @brief Transient finite-volume heat conduction on a solid subdomain.
 */
#pragma once

#include "FVM/BoundaryCache.hh"
#include "FVM/Operators.hh"
#include "SimpleFluidExport.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/TemperatureDiffusionEquation.hh"
#include "fields/MeshFieldTraits.hh"
#include "geometry/SolidSubdomain.hh"
#include "solvers/BelosLinearSolver.hh"

#include <functional>
#include <memory>

namespace SimpleFluid
{

/**
 * @brief Solve transient heat conduction in a solid region.
 *
 * The equation advances
 * @f[
 *   \rho c_p \frac{\partial T}{\partial t}
 *       = \nabla \cdot (k \nabla T) + \dot q
 * @f]
 * on @p MeshType.  No advective face flux is accepted by this API: the
 * implementation supplies an identically zero flux field to the shared
 * physical-temperature transport path.
 *
 * The caller owns accepted temperature and material-property storage.  A
 * converged candidate is published to the output field only after the linear
 * solve succeeds.
 *
 * @tparam Pack Tpetra type pack used for distributed storage.
 * @tparam MeshType Solid mesh view; defaults to SolidSubdomain.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = SolidSubdomain<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT SolidHeatConductionEquation
{
public:
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using boundary_cache_type = FVM::MeshBoundaryCache<Pack, mesh_type>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using source_type = std::function<scalar_type(local_ordinal_type)>;

    /** @brief Construct solid conduction on a mesh with temperature boundary conditions. */
    SolidHeatConductionEquation(SP<const mesh_type> mesh, const BoundaryConditionSet& boundary_conditions);

    /** @brief Replace temperature boundary data for subsequent advances. */
    void set_boundary_conditions(const BoundaryConditionSet& boundary_conditions);

    /** @brief Advance conduction without volumetric heat generation. */
    LinearSolveStatistics advance(const field_type& old_temperature, scalar_type time_step,
        const material_type& material, field_type& temperature,
        FVM::NonOrthogonalTreatment treatment = FVM::NonOrthogonalTreatment::Implicit,
        const LinearSolverOptions& linear_options = {}, const field_type* thermal_conductivity_override = nullptr,
        const boundary_cache_type* boundary_thermal_conductivity = nullptr,
        FVM::FaceCoefficientInterpolation coefficient_interpolation =
            FVM::FaceCoefficientInterpolation::Harmonic) const;

    /** @brief Advance conduction with volumetric power density in W/m^3. */
    LinearSolveStatistics advance(const field_type& old_temperature, scalar_type time_step,
        const material_type& material, field_type& temperature, const source_type& power_density,
        FVM::NonOrthogonalTreatment treatment = FVM::NonOrthogonalTreatment::Implicit,
        const LinearSolverOptions& linear_options = {}, const field_type* thermal_conductivity_override = nullptr,
        const boundary_cache_type* boundary_thermal_conductivity = nullptr,
        FVM::FaceCoefficientInterpolation coefficient_interpolation =
            FVM::FaceCoefficientInterpolation::Harmonic) const;

private:
    SP<const mesh_type> d_mesh;
    face_flux_field_type d_zero_face_flux;
    std::unique_ptr<TemperatureDiffusionEquation<Pack, mesh_type>> d_temperature_equation;
};

extern template class SolidHeatConductionEquation<DefaultTpetraTypes, SolidSubdomain<DefaultTpetraTypes>>;

} // namespace SimpleFluid
