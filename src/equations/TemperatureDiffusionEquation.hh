/**
 * @file TemperatureDiffusionEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Finite-volume temperature diffusion and convection equation.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "FVM/BoundaryCache.hh"
#include "FVM/Operators.hh"
#include "SimpleFluidExport.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/BoussinesqModel.hh"
#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/MeshFieldTraits.hh"
#include "solvers/BelosLinearSolver.hh"

#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Finite-volume heat equation for cell-centered temperature.
 *
 * The class owns the boundary-condition lookup needed by the equation while
 * the caller owns field storage and time integration order.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes, class MeshType = Mesh<Pack>>
class SIMPLEFLUID_EQUATIONS_EXPORT TemperatureDiffusionEquation
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

    TemperatureDiffusionEquation(SP<const mesh_type> mesh, const BoundaryConditionSet& boundary_conditions);

    void refresh_boundary_cache();

    void advance_explicit(const std::vector<scalar_type>& old_temperature, scalar_type time_step,
        scalar_type thermal_diffusivity, field_type& temperature) const;

    void advance_explicit(const std::vector<scalar_type>& old_temperature, scalar_type time_step,
        scalar_type thermal_diffusivity, field_type& temperature, const source_type& right_hand_source) const;

    LinearSolveStatistics advance_semi_implicit(const field_type& old_temperature,
        const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
        field_type& temperature, const LinearSolverOptions& linear_options = {}) const;

    LinearSolveStatistics advance_semi_implicit(const field_type& old_temperature,
        const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
        field_type& temperature, const source_type& right_hand_source,
        const LinearSolverOptions& linear_options = {}) const;

    /** @brief Advance with an explicitly selected non-orthogonal treatment. */
    LinearSolveStatistics advance_semi_implicit(const field_type& old_temperature,
        const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
        field_type& temperature, FVM::NonOrthogonalTreatment treatment,
        const LinearSolverOptions& linear_options = {}) const;

    /** @brief Advance sourced transport with selectable non-orthogonal diffusion. */
    LinearSolveStatistics advance_semi_implicit(const field_type& old_temperature,
        const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
        field_type& temperature, const source_type& right_hand_source, FVM::NonOrthogonalTreatment treatment,
        const LinearSolverOptions& linear_options = {}) const;

    /**
     * @brief Advance conservative temperature with rho-cp storage,
     *        conductivity diffusion, and volumetric power density.
     */
    LinearSolveStatistics advance_physical(const field_type& old_temperature, const face_flux_field_type& face_fluxes,
        scalar_type time_step, const material_type& material, field_type& temperature, const source_type& power_density,
        FVM::NonOrthogonalTreatment treatment, const LinearSolverOptions& linear_options = {},
        const field_type* thermal_conductivity_override = nullptr,
        const boundary_cache_type* boundary_thermal_conductivity = nullptr,
        FVM::FaceCoefficientInterpolation coefficient_interpolation =
            FVM::FaceCoefficientInterpolation::Harmonic) const;

private:
    SIMPLEFLUID_EQUATIONS_LOCAL
    LinearSolveStatistics advance_semi_implicit_impl(const field_type& old_temperature,
        const face_flux_field_type& face_fluxes, scalar_type time_step, scalar_type thermal_diffusivity,
        field_type& temperature, const source_type& right_hand_source,
        std::optional<FVM::NonOrthogonalTreatment> treatment, const LinearSolverOptions& linear_options) const;

    SP<const mesh_type> d_mesh;
    FVM::TransportGeometryCache<mesh_type> d_transport_geometry_cache;
    mutable field_type d_candidate_temperature;
    boundary_cache_type d_face_boundary_temperature;
    SP<BoundaryConditionMap> d_boundary_condition;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_transport_matrix;
    mutable bool d_cached_transport_graph_supports_non_orthogonal_correction = false;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_physical_transport_matrix;
    mutable bool d_cached_physical_graph_supports_non_orthogonal_correction = false;
    mutable BelosLinearSolver<Pack> d_linear_solver;
};

extern template class TemperatureDiffusionEquation<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class TemperatureDiffusionEquation<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid
