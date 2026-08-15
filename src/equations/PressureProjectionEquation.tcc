/**
 * @file PressureProjectionEquation.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Out-of-line template method implementations for PressureProjectionEquation.
 * @version 0.1
 * @date 2026-06-01
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "PressureProjectionEquation.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>

namespace SimpleFluid
{

/**
 * @brief Construct a PressureProjectionEquation on the given mesh.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @param linear_options Linear solver configuration.
 * @param pressure_boundary_conditions Physical pressure boundary data. The
 *        projection applies the corresponding homogeneous correction types.
 * @throws std::invalid_argument if @p mesh is null.
 * @throws std::runtime_error if the mesh has no owned-cell map.
 */
template<TpetraTypePack Pack>
PressureProjectionEquation<Pack>::PressureProjectionEquation(
    SP<const mesh_type> mesh,
    LinearSolverOptions linear_options,
    BoundaryConditionMap pressure_boundary_conditions,
    FVM::CellGradientScheme gradient_scheme)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "PressureProjectionEquation")),
      d_linear_options(linear_options),
      d_pressure_boundary_conditions(
          std::move(pressure_boundary_conditions)),
      d_pressure_correction_boundary_conditions(
          d_pressure_boundary_conditions),
      d_gradient_scheme(gradient_scheme),
      d_cached_face_fluxes(d_mesh, "pressure_projection_face_flux"),
      d_face_flux_workspace(d_mesh)
{
    require_owned_cell_map(d_mesh);
    for (auto& [name, condition] :
         d_pressure_correction_boundary_conditions)
    {
        static_cast<void>(name);
        condition.value = 0.0;
    }
}

/**
 * @brief Require and return the mesh owned-cell map.
 *
 * @tparam Pack Tpetra type pack.
 * @param mesh Shared pointer to the computational mesh.
 * @return The mesh's owned-cell map.
 * @throws std::runtime_error if the mesh has no owned-cell map.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::require_owned_cell_map(
    const SP<const mesh_type>& mesh) -> Teuchos::RCP<const map_type>
{
    auto map = mesh->owned_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error(
            "PressureProjectionEquation requires an assembled mesh with an owned-cell map.");
    }
    if (map->getGlobalNumElements() == 0)
    {
        throw std::runtime_error(
            "PressureProjectionEquation requires at least one global cell.");
    }

    return map;
}

/**
 * @brief (Re)build the cached pressure-Poisson matrix.
 *
 * Uses a physical Dirichlet pressure face to remove the nullspace. For an
 * all-Neumann system, the globally smallest owned-row ID remains the gauge.
 *
 * @tparam Pack Tpetra type pack.
 */
template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::rebuild_matrix() const
{
    const auto owned_map = require_owned_cell_map(d_mesh);
    int local_has_dirichlet = 0;
    for (const auto& [batch_id, boundary_batch] :
         d_mesh->boundary_batches())
    {
        const auto condition_iter =
            d_pressure_boundary_conditions.find(
                d_mesh->boundary_batch_name(batch_id));
        if (condition_iter == d_pressure_boundary_conditions.end()
            || condition_iter->second.type
               != BoundaryConditionType::Dirichlet)
        {
            continue;
        }
        for (const auto face_lid : boundary_batch.face_lids)
        {
            if (d_mesh->is_owned_face(face_lid)
                && d_mesh->is_boundary_face(face_lid))
            {
                local_has_dirichlet = 1;
                break;
            }
        }
        if (local_has_dirichlet != 0)
        {
            break;
        }
    }
    int global_has_dirichlet = 0;
    Teuchos::reduceAll(
        *owned_map->getComm(),
        Teuchos::REDUCE_MAX,
        1,
        &local_has_dirichlet,
        &global_has_dirichlet);
    d_pressure_gauge_gid =
        global_has_dirichlet != 0
      ? std::optional<typename Pack::global_ordinal_type>{}
      : std::optional<typename Pack::global_ordinal_type>{
            owned_map->getMinAllGlobalIndex()};
    auto boundary_condition =
        [&](int batch_id, size_t)
    {
        const auto iter =
            d_pressure_correction_boundary_conditions.find(
                d_mesh->boundary_batch_name(batch_id));
        return iter == d_pressure_correction_boundary_conditions.end()
             ? BoundaryCondition{}
             : iter->second;
    };
    d_cached_pressure_matrix =
        FVM::pressure_poisson_matrix<Pack>(
            *d_mesh, d_pressure_gauge_gid, boundary_condition);
}

/**
 * @brief Solve for the pressure field (initialise with zero and sync
 *        periodic boundaries).
 *
 * @tparam Pack Tpetra type pack.
 * @param[out] pressure Pressure field to initialise.
 * @throws std::invalid_argument if the pressure field mesh does not match.
 */
template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::solve(field_type& pressure)
{
    EquationValidation::require_mesh_match(*d_mesh, pressure,
                                           "PressureProjectionEquation");

    pressure.owned_data().putScalar(0.0);
    d_mesh->sync_periodic_boundaries(pressure);
}

/**
 * @brief Perform the pressure projection step with a zero source term.
 *
 * @tparam Pack Tpetra type pack.
 * @param[in,out] pressure Pressure field (updated on output).
 * @param time_step Time-step size.
 * @param reference_density Density used to normalize pressure internally.
 * @param velocity_boundary_cache Cached velocity boundary conditions.
 * @param[in,out] velocity Velocity field corrected by the pressure
 *        gradient on output.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::project(
    field_type& pressure,
    scalar_type time_step,
    scalar_type reference_density,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    velocity_field_type& velocity) -> ProjectionResult
{
    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    return project(
        pressure,
        time_step,
        reference_density,
        velocity_boundary_cache,
        velocity,
        zero_source);
}

/**
 * @brief Perform a pressure projection against accumulated physical pressure.
 *
 * @tparam Pack Tpetra type pack.
 * @param[in,out] pressure Accumulated physical gauge pressure in Pa.
 * @param[out] pressure_correction Physical pressure correction in Pa.
 * @param time_step Time-step size.
 * @param reference_density Density used to normalize pressure internally.
 * @param velocity_boundary_cache Cached velocity boundary conditions.
 * @param[in,out] velocity Velocity field corrected by the pressure update.
 * @return Projection statistics and the norm of the physical correction.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::project(
    field_type& pressure,
    field_type& pressure_correction,
    scalar_type time_step,
    scalar_type reference_density,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    velocity_field_type& velocity) -> ProjectionResult
{
    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    return project_impl(
        pressure_correction,
        time_step,
        reference_density,
        velocity_boundary_cache,
        velocity,
        zero_source,
        &pressure,
        false);
}

/**
 * @brief Reuse the preceding correction's final flux as the next PISO
 *        predictor.
 *
 * Consecutive pressure correctors do not change pressure or velocity between
 * calls. The preceding final Rhie--Chow reconstruction is therefore exactly
 * the next predictor flux, including its cached pressure gradient.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::project_reusing_cached_predictor(
    field_type& pressure,
    field_type& pressure_correction,
    scalar_type time_step,
    scalar_type reference_density,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    velocity_field_type& velocity) -> ProjectionResult
{
    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    return project_impl(
        pressure_correction,
        time_step,
        reference_density,
        velocity_boundary_cache,
        velocity,
        zero_source,
        &pressure,
        true);
}

/**
 * @brief Perform the pressure projection step: compute face velocities,
 *        assemble the pressure-Poisson RHS, solve for pressure, and
 *        correct the velocity field.
 *
 * @tparam Pack Tpetra type pack.
 * @param[in,out] pressure Pressure field (solved on output).
 * @param time_step Time-step size (must be positive).
 * @param reference_density Positive density used to convert the internally
 *        normalized correction to Pa.
 * @param velocity_boundary_cache Cached velocity boundary conditions.
 * @param[in,out] velocity Velocity field corrected by the pressure
 *        gradient on output.
 * @param right_hand_source Per-cell scalar source provider.
 * @throws std::invalid_argument on mesh mismatch or non-positive time
 *         step.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::project(
    field_type& pressure,
    scalar_type time_step,
    scalar_type reference_density,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    velocity_field_type& velocity,
    const source_type& right_hand_source) -> ProjectionResult
{
    return project_impl(
        pressure,
        time_step,
        reference_density,
        velocity_boundary_cache,
        velocity,
        right_hand_source,
        nullptr,
        false);
}

/**
 * @brief Shared pressure-correction implementation.
 *
 * When @p accumulated_pressure is non-null, the Rhie-Chow predictor uses the
 * current physical pressure and the final flux is reconstructed from the
 * updated total pressure.  This makes the Poisson RHS, corrected velocity,
 * stored pressure, and transport face flux one consistent discrete state.
 *
 * @tparam Pack Tpetra type pack.
 * @param[out] pressure_correction Correction field, normalized internally.
 * @param time_step Positive time-step size.
 * @param reference_density Positive pressure-normalization density.
 * @param velocity_boundary_cache Cached velocity boundary conditions.
 * @param[in,out] velocity Velocity field corrected by the pressure update.
 * @param right_hand_source Per-cell scalar source provider.
 * @param[in,out] accumulated_pressure Optional physical pressure in Pa.
 * @param reuse_cached_predictor_flux Whether to reuse the final flux from the
 *        immediately preceding accumulated-pressure projection.
 * @return Projection statistics.
 */
template<TpetraTypePack Pack>
auto PressureProjectionEquation<Pack>::project_impl(
    field_type& pressure_correction,
    scalar_type time_step,
    scalar_type reference_density,
    const FVM::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    velocity_field_type& velocity,
    const source_type& right_hand_source,
    field_type* accumulated_pressure,
    bool reuse_cached_predictor_flux) -> ProjectionResult
{
    EquationValidation::require_mesh_match(*d_mesh, pressure_correction,
                                           "PressureProjectionEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity,
                                           "PressureProjectionEquation");
    if (accumulated_pressure != nullptr)
    {
        EquationValidation::require_mesh_match(
            *d_mesh, *accumulated_pressure, "PressureProjectionEquation");
        if (accumulated_pressure == &pressure_correction)
        {
            throw std::invalid_argument(
                "PressureProjectionEquation requires distinct accumulated "
                "and correction pressure fields.");
        }
    }
    if (time_step <= 0.0)
    {
        throw std::invalid_argument("PressureProjectionEquation requires a positive time step.");
    }
    if (!std::isfinite(reference_density)
        || reference_density <= scalar_type{})
    {
        throw std::invalid_argument(
            "PressureProjectionEquation requires a finite positive "
            "reference density.");
    }
    if (reuse_cached_predictor_flux
        && (accumulated_pressure == nullptr
            || !d_cached_predictor_flux_valid))
    {
        throw std::logic_error(
            "PressureProjectionEquation cannot reuse a predictor flux "
            "without an adjacent accumulated-pressure projection.");
    }

    // Any failure after this point invalidates reuse. A successful
    // accumulated-pressure projection publishes a new final flux below.
    d_cached_predictor_flux_valid = false;
    pressure_correction.owned_data().putScalar(0.0);
    d_mesh->sync_periodic_boundaries(pressure_correction);
    if (!reuse_cached_predictor_flux
        && accumulated_pressure != nullptr)
    {
        FVM::pressure_weighted_face_fluxes(
            velocity,
            *accumulated_pressure,
            time_step / reference_density,
            velocity_boundary_cache,
            d_pressure_boundary_conditions,
            d_face_flux_workspace,
            d_cached_face_fluxes,
            d_gradient_scheme);
    }
    else if (!reuse_cached_predictor_flux)
    {
        FVM::pressure_weighted_face_fluxes(
            velocity,
            pressure_correction,
            time_step,
            velocity_boundary_cache,
            d_pressure_correction_boundary_conditions,
            d_face_flux_workspace,
            d_cached_face_fluxes,
            d_gradient_scheme);
    }
    const auto owned_map = require_owned_cell_map(d_mesh);
    if (d_cached_pressure_matrix.is_null())
    {
        rebuild_matrix();
    }
    if (d_cached_rhs.is_null())
    {
        d_cached_rhs = Teuchos::rcp(
            new typename Pack::vector_type(d_mesh->owned_cell_map(), true));
    }
    else
    {
        d_cached_rhs->putScalar(0.0);
    }

    {
        const auto predictor_flux_values =
            d_cached_face_fluxes.owned_read_view();
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto row_gid =
                owned_map->getGlobalElement(cell_lid);
            const auto is_pressure_gauge =
                d_pressure_gauge_gid
                && row_gid == *d_pressure_gauge_gid;
            const auto rhs_value =
                is_pressure_gauge
              ? scalar_type{}
              : -FVM::cell_flux_balance<Pack>(
                    *d_mesh,
                    d_cached_face_fluxes,
                    predictor_flux_values,
                    cell_lid)
                    / time_step
                + d_mesh->cell_volume(cell_lid)
                    * right_hand_source(cell_lid);
            d_cached_rhs->replaceLocalValue(
                cell_lid, rhs_value);
        }
    }

    Teuchos::RCP<const typename Pack::matrix_type> const_matrix =
        d_cached_pressure_matrix;
    const LinearResidualScaling residual_scaling{
        reuse_cached_predictor_flux
            ? d_rhs_norm_reference
            : real_t{}};
    const auto linear_statistics =
        d_linear_solver.solve_with_statistics(
            const_matrix, *d_cached_rhs, pressure_correction.owned_data(),
            d_linear_options, residual_scaling);
    if (!linear_statistics.converged)
    {
        throw std::runtime_error("PressureProjectionEquation projection solve did not converge.");
    }
    if (!reuse_cached_predictor_flux)
    {
        d_rhs_norm_reference = linear_statistics.rhs_norm;
    }
    d_mesh->sync_periodic_boundaries(pressure_correction);

    scalar_type pressure_norm_squared = {};
    {
        const auto correction_values =
            pressure_correction.owned_read_view();
        const auto& volumes =
            d_mesh->host_views().cell_geometry.volume;
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto correction =
                correction_values(cell_lid, 0);
            pressure_norm_squared +=
                correction * correction * volumes[owned];
        }
    }

    FVM::cell_gradient(
        pressure_correction,
        d_pressure_correction_boundary_conditions,
        d_face_flux_workspace.pressure_gradient(),
        d_face_flux_workspace.gradient_cache(),
        d_gradient_scheme);
    velocity.owned_data().update(
        -time_step,
        d_face_flux_workspace.pressure_gradient().owned_data(),
        1.0);

    d_mesh->sync_periodic_boundaries(velocity);

    if (accumulated_pressure != nullptr)
    {
        pressure_correction.owned_data().scale(reference_density);
        d_mesh->sync_periodic_boundaries(pressure_correction);
        accumulated_pressure->owned_data().update(
            scalar_type{1},
            pressure_correction.owned_data(),
            scalar_type{1});
        d_mesh->sync_periodic_boundaries(*accumulated_pressure);

        FVM::pressure_weighted_face_fluxes(
            velocity,
            *accumulated_pressure,
            time_step / reference_density,
            velocity_boundary_cache,
            d_pressure_boundary_conditions,
            d_face_flux_workspace,
            d_cached_face_fluxes,
            d_gradient_scheme);
    }
    else
    {
        FVM::pressure_weighted_face_fluxes(
            velocity,
            pressure_correction,
            d_face_flux_workspace.pressure_gradient(),
            time_step,
            velocity_boundary_cache,
            d_pressure_correction_boundary_conditions,
            d_face_flux_workspace,
            d_cached_face_fluxes);
        pressure_correction.owned_data().scale(reference_density);
        d_mesh->sync_periodic_boundaries(pressure_correction);
    }

    scalar_type continuity_norm_squared = {};
    {
        const auto corrected_flux_values =
            d_cached_face_fluxes.owned_read_view();
        for (size_t owned = 0;
             owned < d_mesh->num_owned_cells();
             ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto balance =
                FVM::cell_flux_balance<Pack>(
                    *d_mesh,
                    d_cached_face_fluxes,
                    corrected_flux_values,
                    cell_lid);
            continuity_norm_squared += balance * balance;
        }
    }

    const std::array<scalar_type, 2> local_norms_squared{
        pressure_norm_squared,
        continuity_norm_squared};
    std::array<scalar_type, 2> global_norms_squared{};
    Teuchos::reduceAll(
        *owned_map->getComm(),
        Teuchos::REDUCE_SUM,
        2,
        local_norms_squared.data(),
        global_norms_squared.data());

    if (reuse_cached_predictor_flux)
    {
        ++d_cached_predictor_flux_reuse_count;
    }
    if (accumulated_pressure != nullptr)
    {
        d_cached_predictor_flux_valid = true;
    }

    using std::sqrt;
    return {
        reference_density * sqrt(global_norms_squared[0]),
        sqrt(global_norms_squared[1]),
        linear_statistics};
}

} // namespace SimpleFluid
