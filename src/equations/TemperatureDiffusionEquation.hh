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

#include "equations/BoundaryConditions.hh"
#include "equations/EquationValidation.hh"
#include "fields/CellField.hh"
#include "fields/FaceField.hh"
#include "FVM/Operators.hh"
#include "FVM/BoundaryCache.hh"
#include "solvers/BelosLinearSolver.hh"

#include <cstddef>
#include <cmath>
#include <functional>
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
template<TpetraTypePack Pack = DefaultTpetraTypes>
class TemperatureDiffusionEquation
{
public:
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using source_type = std::function<scalar_type(local_ordinal_type)>;

    TemperatureDiffusionEquation(SP<const mesh_type> mesh,
                                 const BoundaryConditionSet& boundary_conditions);

    void refresh_boundary_cache();

    void advance_explicit(const std::vector<scalar_type>& old_temperature,
                          scalar_type time_step,
                          scalar_type thermal_diffusivity,
                          field_type& temperature) const;

    void advance_explicit(const std::vector<scalar_type>& old_temperature,
                          scalar_type time_step,
                          scalar_type thermal_diffusivity,
                          field_type& temperature,
                          const source_type& right_hand_source) const;

    void advance_semi_implicit(
        const field_type& old_temperature,
        const FaceField<Pack>& face_fluxes,
        scalar_type time_step,
        scalar_type thermal_diffusivity,
        field_type& temperature,
        const LinearSolverOptions& linear_options = {}) const;

    void advance_semi_implicit(
        const field_type& old_temperature,
        const FaceField<Pack>& face_fluxes,
        scalar_type time_step,
        scalar_type thermal_diffusivity,
        field_type& temperature,
        const source_type& right_hand_source,
        const LinearSolverOptions& linear_options = {}) const;

private:
    SP<const mesh_type> d_mesh;
    BoundaryCache<Pack> d_face_boundary_temperature;
    SP<BoundaryConditionMap>  d_boundary_condition;
    mutable Teuchos::RCP<typename Pack::matrix_type> d_cached_transport_matrix;
};

} // namespace SimpleFluid
