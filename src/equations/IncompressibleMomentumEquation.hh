/**
 * @file IncompressibleMomentumEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Incompressible velocity-transport assembly and solution.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "equations/EquationForward.hh"
#include "equations/EquationValidation.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "fields/MeshFieldTraits.hh"
#include "fields/VectorCellField.hh"
#include "FVM/BoundaryCache.hh"
#include "FVM/Operators.hh"
#include "solvers/BelosLinearSolver.hh"

#include <Teuchos_RCP.hpp>

#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Momentum update for incompressible three-component velocity fields.
 *
 * This class owns generic velocity transport assembly, boundary treatment,
 * non-orthogonal correction sweeps, and linear solves. Physics-specific
 * momentum equations provide their acceleration through the source callback.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack, class MeshType>
class SIMPLEFLUID_EQUATIONS_EXPORT IncompressibleMomentumEquation
{
public:
    using mesh_type = MeshType;
    using field_traits = MeshFieldTraits<Pack, mesh_type>;
    using field_type = typename field_traits::scalar_cell_type;
    using velocity_field_type = typename field_traits::vector_cell_type;
    using face_flux_field_type = typename field_traits::scalar_face_type;
    using velocity_boundary_cache_type = std::conditional_t<
        std::same_as<mesh_type, Mesh<Pack>>,
        FVM::VelocityBoundaryCache<Pack>,
        FVM::FieldStoredVelocityBoundaryCache<Pack, mesh_type>>;
    using boundary_cache_type = FVM::MeshBoundaryCache<Pack, mesh_type>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using source_type =
        std::function<typename velocity_field_type::vec_type(local_ordinal_type)>;
    using system_type = FVM::VectorTransportSystem<Pack>;

    explicit IncompressibleMomentumEquation(SP<const mesh_type> mesh);
    virtual ~IncompressibleMomentumEquation() = default;

    const mesh_type& mesh() const noexcept { return *d_mesh; }
    SP<const mesh_type> mesh_ptr() const noexcept { return d_mesh; }

    LinearSolveSummary advance_velocity(
        const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const LinearSolverOptions& linear_options = {}) const;

    LinearSolveSummary advance_velocity(
        const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options,
        velocity_field_type& velocity,
        const source_type& right_hand_source,
        const LinearSolverOptions& linear_options = {}) const;

    system_type assemble_system(
        const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const velocity_field_type* correction_field = nullptr) const;

    system_type assemble_system(
        const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const source_type& right_hand_source,
        const velocity_field_type* correction_field = nullptr) const;

    /** @brief Advance using dynamic viscosity divided by reference density. */
    LinearSolveSummary advance_velocity_physical(
        const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const field_type& dynamic_viscosity,
        scalar_type reference_density,
        velocity_field_type& velocity,
        const source_type& acceleration_source,
        const LinearSolverOptions& linear_options = {},
        const boundary_cache_type* boundary_dynamic_viscosity = nullptr) const;

    /** @brief Assemble using dynamic viscosity divided by reference density. */
    system_type assemble_physical_system(
        const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const field_type& dynamic_viscosity,
        scalar_type reference_density,
        const source_type& acceleration_source,
        const velocity_field_type* correction_field = nullptr,
        const boundary_cache_type* boundary_dynamic_viscosity = nullptr) const;

private:
    SIMPLEFLUID_EQUATIONS_LOCAL
    void validate_transport_inputs(
        const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes,
        const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options,
        const velocity_field_type* correction_field) const;

    SP<const mesh_type> d_mesh;
    FVM::TransportGeometryCache<mesh_type> d_transport_geometry_cache;
    mutable velocity_field_type d_candidate_velocity;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_transport_matrix;
    mutable bool
        d_cached_graph_supports_non_orthogonal_correction = false;
    mutable Teuchos::RCP<typename Pack::matrix_type>
        d_cached_physical_transport_matrix;
    mutable bool
        d_cached_physical_graph_supports_non_orthogonal_correction = false;
    mutable BelosLinearSolver<Pack> d_linear_solver;
};

extern template class
    IncompressibleMomentumEquation<DefaultTpetraTypes,
                                   Mesh<DefaultTpetraTypes>>;
extern template class
    IncompressibleMomentumEquation<DefaultTpetraTypes,
                                   MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid
