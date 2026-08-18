/**
 * @file BoussinesqMomentumEquation.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Boussinesq buoyancy specialization of incompressible momentum.
 * @version 0.1
 * @date 2026-05-28
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoussinesqModel.hh"
#include "equations/IncompressibleMomentumEquation.hh"

namespace SimpleFluid
{

/**
 * @brief Boussinesq momentum update for coupled three-component velocity fields.
 *
 * The solver stores velocity as a three-column MultiVector-backed field.
 * This equation class advances all velocity components in a single linear solve.
 *
 * @tparam Pack Tpetra type pack used for field storage.
 */
template<TpetraTypePack Pack, class MeshType>
class SIMPLEFLUID_EQUATIONS_EXPORT BoussinesqMomentumEquation : public IncompressibleMomentumEquation<Pack, MeshType>
{
public:
    using base_type = IncompressibleMomentumEquation<Pack, MeshType>;
    using mesh_type = typename base_type::mesh_type;
    using field_type = typename base_type::field_type;
    using velocity_field_type = typename base_type::velocity_field_type;
    using face_flux_field_type = typename base_type::face_flux_field_type;
    using velocity_boundary_cache_type = typename base_type::velocity_boundary_cache_type;
    using boundary_cache_type = typename base_type::boundary_cache_type;
    using material_type = MaterialPropertyFields<Pack, mesh_type>;
    using scalar_type = typename base_type::scalar_type;
    using local_ordinal_type = typename base_type::local_ordinal_type;
    using source_type = typename base_type::source_type;
    using system_type = typename base_type::system_type;

    using base_type::advance_velocity;
    using base_type::advance_velocity_physical;
    using base_type::assemble_physical_system;
    using base_type::assemble_system;

    explicit BoussinesqMomentumEquation(SP<const mesh_type> mesh);

    LinearSolveSummary advance_velocity(const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes, const field_type& temperature,
        const velocity_boundary_cache_type& velocity_boundary_cache, const TimeStepperOptions& options,
        velocity_field_type& velocity, const LinearSolverOptions& linear_options = {}) const;

    LinearSolveSummary advance_velocity(const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes, const field_type& temperature,
        const velocity_boundary_cache_type& velocity_boundary_cache, const TimeStepperOptions& options,
        velocity_field_type& velocity, const source_type& right_hand_source,
        const LinearSolverOptions& linear_options = {}) const;

    system_type assemble_system(const velocity_field_type& old_velocity, const face_flux_field_type& face_fluxes,
        const field_type& temperature, const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options, const velocity_field_type* correction_field = nullptr) const;

    system_type assemble_system(const velocity_field_type& old_velocity, const face_flux_field_type& face_fluxes,
        const field_type& temperature, const velocity_boundary_cache_type& velocity_boundary_cache,
        const TimeStepperOptions& options, const source_type& right_hand_source,
        const velocity_field_type* correction_field = nullptr) const;

    /**
     * @brief Advance with physical viscosity and material-density buoyancy
     *        when density feedback is enabled.
     */
    LinearSolveSummary advance_velocity_physical(const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes, const field_type& temperature,
        const velocity_boundary_cache_type& velocity_boundary_cache, const TimeStepperOptions& options,
        const material_type& material, scalar_type reference_density, bool density_feedback_enabled,
        velocity_field_type& velocity, const source_type& right_hand_source,
        const LinearSolverOptions& linear_options = {}, const field_type* dynamic_viscosity_override = nullptr,
        const boundary_cache_type* boundary_dynamic_viscosity = nullptr) const;

    /**
     * @brief Assemble with physical viscosity and material-density buoyancy
     *        when density feedback is enabled.
     */
    system_type assemble_physical_system(const velocity_field_type& old_velocity,
        const face_flux_field_type& face_fluxes, const field_type& temperature,
        const velocity_boundary_cache_type& velocity_boundary_cache, const TimeStepperOptions& options,
        const material_type& material, scalar_type reference_density, bool density_feedback_enabled,
        const source_type& right_hand_source, const velocity_field_type* correction_field = nullptr,
        const field_type* dynamic_viscosity_override = nullptr,
        const boundary_cache_type* boundary_dynamic_viscosity = nullptr) const;

private:
    SIMPLEFLUID_EQUATIONS_LOCAL
    const field_type& select_dynamic_viscosity(
        const material_type& material, const field_type* dynamic_viscosity_override) const;
};

extern template class BoussinesqMomentumEquation<DefaultTpetraTypes, Mesh<DefaultTpetraTypes>>;
extern template class BoussinesqMomentumEquation<DefaultTpetraTypes, MeshHandle<DefaultTpetraTypes>>;

} // namespace SimpleFluid
