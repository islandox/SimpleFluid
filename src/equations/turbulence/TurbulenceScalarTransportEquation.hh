/**
 * @file TurbulenceScalarTransportEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Positive scalar transport used by two-equation turbulence models.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "FVM/TransportSystem.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "solvers/BelosLinearSolver.hh"

#include <functional>
#include <optional>

namespace SimpleFluid
{

/**
 * @brief Dynamic boundary and adjacent-cell data for a turbulence scalar.
 *
 * A provider returns std::nullopt when a face or cell keeps the equation's
 * configured behavior. This lets a wall treatment override only selected
 * wall batches while inlet, outlet, and symmetry data continue to come from
 * the ordinary boundary-condition map.
 * @tparam Pack Tpetra type pack used for field and boundary-cache storage.
 */
template <TpetraTypePack Pack = DefaultTpetraTypes>
struct TurbulenceScalarBoundaryOverrides
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using boundary_condition_provider_type =
        std::function<std::optional<BoundaryCondition>(int, size_t)>;
    using boundary_value_provider_type =
        std::function<std::optional<scalar_type>(int, size_t)>;
    using fixed_cell_value_provider_type =
        std::function<std::optional<scalar_type>(local_ordinal_type)>;

    boundary_condition_provider_type boundary_condition;
    boundary_value_provider_type boundary_value;
    fixed_cell_value_provider_type fixed_cell_value;

    /** Sparse per-face diffusivity; omitted faces use the owner-cell value. */
    const FVM::BoundaryCache<Pack>* boundary_diffusivity = nullptr;

    /** Permit an overridden Dirichlet face to be exactly zero (wall k). */
    bool allow_zero_dirichlet = false;
};

/**
 * @brief Semi-implicit transport equation for a positive turbulence scalar.
 *
 * The equation advances a density-divided scalar with unit storage and
 * advection weights, variable cell diffusivity, an explicit source, and a
 * linearized implicit sink. The accepted field is replaced only after a
 * converged candidate has been validated and floored.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template <TpetraTypePack Pack = DefaultTpetraTypes> class TurbulenceScalarTransportEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using face_field_type = FaceField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_provider_type = std::function<scalar_type(local_ordinal_type)>;
    using boundary_overrides_type = TurbulenceScalarBoundaryOverrides<Pack>;

    /**
     * @brief Construct an equation on @p mesh with scalar boundary data.
     *
     * Missing boundary names use homogeneous Neumann conditions.
     */
    explicit TurbulenceScalarTransportEquation(SP<const mesh_type> mesh,
                                               BoundaryConditionMap boundary_conditions = {});

    /**
     * @brief Advance one accepted scalar state through a transport solve.
     *
     * The explicit source and implicit sink have units of scalar/time and
     * 1/time, respectively. Both must be finite and non-negative in every
     * owned cell. The solved candidate is clamped to @p positive_floor before
     * it atomically replaces @p state.
     *
     * @return Linear-solver convergence statistics.
     * @throws std::invalid_argument for invalid inputs.
     * @throws std::runtime_error if the solve fails or returns non-finite data.
     */
    LinearSolveStatistics
    advance(const field_type& old_state, const face_field_type& face_fluxes, scalar_type time_step,
            const field_type& effective_diffusivity, field_type& state,
            const scalar_provider_type& explicit_source, const scalar_provider_type& implicit_sink,
            scalar_type positive_floor,
            FVM::NonOrthogonalTreatment treatment = FVM::NonOrthogonalTreatment::Explicit,
            const LinearSolverOptions& linear_options = {},
            const boundary_overrides_type* boundary_overrides = nullptr) const;

    /** @brief Return the configured scalar boundary conditions. */
    const BoundaryConditionMap& boundary_conditions() const noexcept
    {
        return d_boundary_conditions;
    }

private:
    SP<const mesh_type> d_mesh;
    FVM::TransportGeometryCache<mesh_type> d_transport_geometry_cache;
    BoundaryConditionMap d_boundary_conditions;
    field_type d_unit_weight;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_transport_matrix;
    mutable bool d_cached_graph_supports_non_orthogonal_correction = false;
    mutable BelosLinearSolver<Pack> d_linear_solver;
};

} // namespace SimpleFluid
