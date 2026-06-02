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

namespace SimpleFluid
{

template<TpetraTypePack Pack>
PressureProjectionEquation<Pack>::PressureProjectionEquation(
    SP<const mesh_type> mesh,
    LinearSolverOptions linear_options)
    : d_mesh(EquationValidation::require_non_null_mesh(
          std::move(mesh), "PressureProjectionEquation")),
      d_linear_options(linear_options),
      d_cached_face_velocity(d_mesh, "pressure_projection_face_velocity"),
      d_cached_face_fluxes(d_mesh, "pressure_projection_face_flux")
{
    require_owned_cell_map(d_mesh);
}

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

    return map;
}

template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::rebuild_matrix() const
{
    if (d_mesh->num_owned_cells() == 0)
    {
        d_cached_pressure_matrix = Teuchos::null;
        return;
    }

    const auto gauge_gid = d_mesh->owned_cell_global_ids().front();
    d_cached_pressure_matrix =
        FvmOperators::pressure_poisson_matrix<Pack>(*d_mesh, gauge_gid);
}

template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::solve(field_type& pressure)
{
    EquationValidation::require_mesh_match(*d_mesh, pressure,
                                           "PressureProjectionEquation");

    pressure.owned_data().putScalar(0.0);
    d_mesh->sync_periodic_boundaries(pressure);
}

template<TpetraTypePack Pack>
void PressureProjectionEquation<Pack>::project(
    field_type& pressure,
    scalar_type time_step,
    const FvmOperators::VelocityBoundaryCache<Pack>& velocity_boundary_cache,
    velocity_field_type& velocity)
{
    EquationValidation::require_mesh_match(*d_mesh, pressure,
                                           "PressureProjectionEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity,
                                           "PressureProjectionEquation");
    if (time_step <= 0.0)
    {
        throw std::invalid_argument("PressureProjectionEquation requires a positive time step.");
    }
    if (d_mesh->num_owned_cells() == 0)
    {
        return;
    }

    FvmOperators::face_velocities(velocity, velocity_boundary_cache,
                                  d_cached_face_velocity);
    FvmOperators::normal_face_fluxes(d_cached_face_velocity,
                                     d_cached_face_fluxes);
    const auto gauge_gid = d_mesh->owned_cell_global_ids().front();
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

    for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const auto row_gid = d_mesh->cell_global_id(cell_lid);
        const auto rhs_value = row_gid == gauge_gid
                             ? scalar_type{}
                             : -FvmOperators::cell_flux_balance<Pack>(
                                   *d_mesh, d_cached_face_fluxes, cell_lid)
                               / time_step;
        d_cached_rhs->replaceLocalValue(cell_lid, rhs_value);
    }

    pressure.owned_data().putScalar(0.0);
    Teuchos::RCP<const typename Pack::matrix_type> const_matrix =
        d_cached_pressure_matrix;
    if (!solve_linear_system<Pack>(const_matrix, *d_cached_rhs, pressure.owned_data(),
                                   d_linear_options))
    {
        throw std::runtime_error("PressureProjectionEquation projection solve did not converge.");
    }
    d_mesh->sync_periodic_boundaries(pressure);

    FvmOperators::cell_gradient(pressure, d_cached_gradients);

    // Correct velocity using Tpetra::MultiVector::update: V = V - dt * grad(p)
    typename Pack::multi_vector_type gradient_mv(d_mesh->owned_cell_map(),
                                                  velocity_field_type::num_components,
                                                  true);
    for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<typename Pack::local_ordinal_type>(owned);
        const auto& gradient = d_cached_gradients[owned];
        for (std::size_t comp = 0; comp < velocity_field_type::num_components; ++comp)
        {
            gradient_mv.replaceLocalValue(cell_lid, comp,
                                           FvmOperators::detail::component_value(gradient, comp));
        }
    }
    velocity.owned_data().update(-time_step, gradient_mv, 1.0);

    d_mesh->sync_periodic_boundaries(velocity);
}

} // namespace SimpleFluid
