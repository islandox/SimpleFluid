/** @file ScalarVoidFractionModel.tcc @brief Compiled template implementations. */
#pragma once

#include "equations/ScalarVoidFractionModel.hh"

namespace SimpleFluid
{

template<TpetraTypePack Pack, class MeshType>
auto ScalarVoidFractionModel<Pack, MeshType>::bounded_collapse_rate(
    local_ordinal_type cell_lid,
    scalar_type time_step) const -> scalar_type
{
    if (!std::isfinite(time_step) || time_step <= scalar_type{})
    {
        throw std::invalid_argument(
            "Scalar void collapse requires a positive finite timestep.");
    }
    if (!std::isfinite(d_options.alpha_collapse_time))
    {
        return scalar_type{};
    }
    const auto alpha = d_alpha_g.value(cell_lid);
    const auto available_alpha = std::max(
        alpha - d_options.alpha_min, scalar_type{});
    return std::min(
        alpha / d_options.alpha_collapse_time,
        available_alpha / time_step);
}

template<TpetraTypePack Pack, class MeshType>
auto ScalarVoidFractionModel<Pack, MeshType>::snapshot() const -> StateSnapshot
{
    StateSnapshot result;
    result.d_owner = this;
    result.d_alpha_g.resize(d_mesh->num_owned_cells());
    result.d_alpha_l.resize(d_mesh->num_owned_cells());
    result.d_source.resize(d_mesh->num_owned_cells());
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        result.d_alpha_g[owned] = d_alpha_g.value(cell);
        result.d_alpha_l[owned] = d_alpha_l.value(cell);
        result.d_source[owned] = d_source_alpha_total.value(cell);
    }
    return result;
}

template<TpetraTypePack Pack, class MeshType>
void ScalarVoidFractionModel<Pack, MeshType>::restore(const StateSnapshot& snapshot)
{
    const int local_invalid = snapshot.d_owner != this ||
                              snapshot.d_alpha_g.size() != d_mesh->num_owned_cells() ||
                              snapshot.d_alpha_l.size() != d_mesh->num_owned_cells() ||
                              snapshot.d_source.size() != d_mesh->num_owned_cells();
    int any_invalid = 0;
    Teuchos::reduceAll(
        *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
    {
        throw std::invalid_argument("ScalarVoidFractionModel snapshot is foreign or incompatible.");
    }
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        d_alpha_g.set_owned_value(cell, snapshot.d_alpha_g[owned]);
        d_alpha_l.set_owned_value(cell, snapshot.d_alpha_l[owned]);
        d_source_alpha_total.set_owned_value(cell, snapshot.d_source[owned]);
    }
    sync_fields();
    refresh_geometry();
}

template<TpetraTypePack Pack, class MeshType>
void ScalarVoidFractionModel<Pack, MeshType>::initialize_from(const field_type& alpha_g)
{
    check_source(&alpha_g, "alpha_g");
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto alpha = std::clamp(
            alpha_g.value(cell_lid),
            d_options.alpha_min,
            d_options.alpha_max);
        d_alpha_g.set_owned_value(cell_lid, alpha);
        d_alpha_l.set_owned_value(cell_lid, scalar_type{1} - alpha);
        d_source_alpha_total.set_owned_value(
            cell_lid, scalar_type{});
    }
    sync_fields();
}

template<TpetraTypePack Pack, class MeshType>
void ScalarVoidFractionModel<Pack, MeshType>::update_explicit(
    scalar_type time_step,
    const field_type* source_alpha_rad,
    const field_type* source_alpha_boil)
{
    if (!std::isfinite(time_step) || time_step <= 0.0)
    {
        throw std::invalid_argument(
            "Scalar void update requires a positive finite time step.");
    }
    check_source(source_alpha_rad, "radiolytic alpha source");
    check_source(source_alpha_boil, "boiling alpha source");

    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto old_alpha = d_alpha_g.value(cell_lid);
        scalar_type raw_rate{};
        if (source_alpha_rad)
            raw_rate += source_alpha_rad->value(cell_lid);
        if (source_alpha_boil)
            raw_rate += source_alpha_boil->value(cell_lid);
        raw_rate -= bounded_collapse_rate(cell_lid, time_step);
        const auto new_alpha = std::clamp(
            old_alpha + time_step * raw_rate,
            d_options.alpha_min,
            d_options.alpha_max);
        d_alpha_g.set_owned_value(cell_lid, new_alpha);
        d_alpha_l.set_owned_value(cell_lid, 1.0 - new_alpha);
        d_source_alpha_total.set_owned_value(
            cell_lid, (new_alpha - old_alpha) / time_step);
    }
    diffuse(time_step);
    sync_fields();
}

template<TpetraTypePack Pack, class MeshType>
void ScalarVoidFractionModel<Pack, MeshType>::mirror(
    const field_type& alpha_g,
    scalar_type time_step)
{
    if (!std::isfinite(time_step) || time_step <= scalar_type{})
    {
        throw std::invalid_argument(
            "Scalar void mirror requires a positive finite timestep.");
    }
    check_source(&alpha_g, "alpha_g");
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto old_alpha = d_alpha_g.value(cell_lid);
        const auto alpha = std::clamp(
            alpha_g.value(cell_lid),
            d_options.alpha_min,
            d_options.alpha_max);
        d_alpha_g.set_owned_value(cell_lid, alpha);
        d_alpha_l.set_owned_value(cell_lid, scalar_type{1} - alpha);
        d_source_alpha_total.set_owned_value(
            cell_lid, (alpha - old_alpha) / time_step);
    }
    sync_fields();
}

template<TpetraTypePack Pack, class MeshType>
void ScalarVoidFractionModel<Pack, MeshType>::diffuse(scalar_type time_step)
{
    if (d_options.alpha_diffusivity == scalar_type{})
    {
        return;
    }

    d_alpha_g.sync_ghosts();
    face_flux_field_type zero_flux(
        d_mesh, scalar_type{}, "alpha_zero_face_flux");
    field_type unit_weight(
        d_mesh, scalar_type{1}, "alpha_unit_weight");
    field_type diffusivity(
        d_mesh,
        static_cast<scalar_type>(d_options.alpha_diffusivity),
        "alpha_diffusivity");
    auto zero_neumann =
        [](int, size_t)
    {
        return BoundaryCondition{
            BoundaryConditionType::Neumann, scalar_type{}};
    };
    auto zero_boundary_value =
        [](int, size_t) -> scalar_type
    {
        return scalar_type{};
    };
    auto zero_source =
        [](local_ordinal_type) -> scalar_type
    {
        return scalar_type{};
    };

    auto system = FVM::weighted_scalar_transport_system<Pack>(
        FVM::MeshWeightedScalarTransportRequest<Pack, mesh_type>{
            .old_values = d_alpha_g,
            .face_fluxes = zero_flux,
            .time_step = time_step,
            .storage_weight = unit_weight,
            .advection_weight = unit_weight,
            .diffusivity = diffusivity,
            .boundary_condition = zero_neumann,
            .boundary_value = zero_boundary_value,
            .source = zero_source,
            .treatment = FVM::NonOrthogonalTreatment::Explicit,
            .geometry_cache = &*d_transport_geometry_cache});
    field_type solution(d_mesh, "alpha_diffusion_solution");
    const auto statistics =
        d_diffusion_solver.solve_with_statistics(
            system.matrix,
            *system.rhs,
            solution.owned_data(),
            LinearSolverOptions{});
    if (!statistics.converged)
    {
        throw std::runtime_error(
            "Scalar void diffusion solve did not converge.");
    }

    const auto bound_scale = std::max(
        {scalar_type{1},
         std::abs(static_cast<scalar_type>(d_options.alpha_min)),
         std::abs(static_cast<scalar_type>(d_options.alpha_max))});
    const auto bound_tolerance = std::max(
        scalar_type{1000}
          * std::numeric_limits<scalar_type>::epsilon()
          * bound_scale,
        scalar_type{10}
          * static_cast<scalar_type>(LinearSolverOptions{}.tolerance)
          * bound_scale);
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto solved_alpha = solution.value(cell_lid);
        if (!std::isfinite(solved_alpha))
        {
            throw std::runtime_error(
                "Scalar void diffusion produced a non-finite value.");
        }
        if (solved_alpha
                < static_cast<scalar_type>(d_options.alpha_min)
                  - bound_tolerance
            || solved_alpha
                > static_cast<scalar_type>(d_options.alpha_max)
                  + bound_tolerance)
        {
            throw std::runtime_error(
                "Scalar void diffusion violated configured bounds.");
        }
        const auto alpha = std::clamp(
            solved_alpha,
            static_cast<scalar_type>(d_options.alpha_min),
            static_cast<scalar_type>(d_options.alpha_max));
        d_alpha_g.set_owned_value(cell_lid, alpha);
        d_alpha_l.set_owned_value(
            cell_lid, scalar_type{1} - alpha);
    }
}

} // namespace SimpleFluid
