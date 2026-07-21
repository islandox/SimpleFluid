/**
 * @file TurbulenceScalarTransportEquation.tcc
 * @brief Template implementation of positive turbulence scalar transport.
 */

#include "TurbulenceCollectiveValidation.hh"
#include "TurbulenceScalarTransportEquation.hh"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

template <TpetraTypePack Pack>
TurbulenceScalarTransportEquation<Pack>::TurbulenceScalarTransportEquation(
    SP<const mesh_type> mesh, BoundaryConditionMap boundary_conditions)
    : d_mesh(EquationValidation::require_non_null_mesh(std::move(mesh),
                                                       "TurbulenceScalarTransportEquation")),
      d_boundary_conditions(std::move(boundary_conditions)),
      d_unit_weight(d_mesh, scalar_type{1}, "turbulence_scalar_unit_weight")
{
}

template <TpetraTypePack Pack>
auto TurbulenceScalarTransportEquation<Pack>::advance(
    const field_type& old_state, const face_field_type& face_fluxes, scalar_type time_step,
    const field_type& effective_diffusivity, field_type& state,
    const scalar_provider_type& explicit_source, const scalar_provider_type& implicit_sink,
    scalar_type positive_floor, FVM::NonOrthogonalTreatment treatment,
    const LinearSolverOptions& linear_options,
    const boundary_overrides_type* boundary_overrides) const -> LinearSolveStatistics
{
    constexpr const char* class_name = "TurbulenceScalarTransportEquation";
    struct PreparedBoundaryData
    {
        BoundaryCondition condition{};
        scalar_type value{};
        bool active = false;
    };

    std::vector<scalar_type> source_values(d_mesh->num_owned_cells());
    std::vector<scalar_type> sink_values(d_mesh->num_owned_cells());
    std::vector<std::optional<scalar_type>> fixed_cell_values(
        d_mesh->num_owned_cells());
    std::vector<PreparedBoundaryData> prepared_boundaries(
        d_mesh->num_faces());
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence scalar transport input validation",
        [&]
        {
            EquationValidation::require_mesh_match(*d_mesh, old_state, class_name);
            EquationValidation::require_mesh_match(*d_mesh, face_fluxes, class_name);
            EquationValidation::require_mesh_match(*d_mesh, effective_diffusivity, class_name);
            EquationValidation::require_mesh_match(*d_mesh, state, class_name);

            if (!std::isfinite(time_step) || time_step <= scalar_type{})
            {
                throw std::invalid_argument("TurbulenceScalarTransportEquation requires a positive "
                                            "finite time step.");
            }
            if (!std::isfinite(positive_floor) || positive_floor <= scalar_type{})
            {
                throw std::invalid_argument("TurbulenceScalarTransportEquation requires a positive "
                                            "finite scalar floor.");
            }
            if (!explicit_source || !implicit_sink)
            {
                throw std::invalid_argument(
                    "TurbulenceScalarTransportEquation requires explicit-source "
                    "and implicit-sink providers.");
            }

            for (const auto& [name, condition] : d_boundary_conditions)
            {
                if (!std::isfinite(condition.value) || !std::isfinite(condition.robin_coefficient))
                {
                    throw std::invalid_argument("Turbulence scalar boundary '" + name +
                                                "' contains non-finite data.");
                }
                if (condition.type != BoundaryConditionType::Dirichlet &&
                    condition.type != BoundaryConditionType::Neumann)
                {
                    throw std::invalid_argument("Turbulence scalar boundary '" + name +
                                                "' must be Dirichlet or Neumann.");
                }
            }

            for (const auto face_lid : face_fluxes.owned_face_ids())
            {
                if (!std::isfinite(face_fluxes.value(face_lid)))
                {
                    throw std::invalid_argument("TurbulenceScalarTransportEquation requires finite "
                                                "face fluxes.");
                }
            }

            for (size_t local = 0; local < d_mesh->num_local_cells(); ++local)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(local);
                const auto old_value = old_state.local_value(cell_lid);
                if (!std::isfinite(old_value) || old_value < positive_floor)
                {
                    throw std::invalid_argument(
                        "TurbulenceScalarTransportEquation requires accepted "
                        "state values at or above the positive floor.");
                }
                const auto diffusivity = effective_diffusivity.local_value(cell_lid);
                if (!std::isfinite(diffusivity) || diffusivity < scalar_type{})
                {
                    throw std::invalid_argument("TurbulenceScalarTransportEquation requires finite "
                                                "non-negative diffusivity.");
                }
            }

            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(owned);
                const auto source = explicit_source(cell_lid);
                const auto sink = implicit_sink(cell_lid);
                if (!std::isfinite(source) || source < scalar_type{})
                {
                    throw std::invalid_argument(
                        "TurbulenceScalarTransportEquation requires a finite "
                        "non-negative explicit source.");
                }
                if (!std::isfinite(sink) || sink < scalar_type{})
                {
                    throw std::invalid_argument(
                        "TurbulenceScalarTransportEquation requires a finite "
                        "non-negative implicit sink.");
                }
                source_values[owned] = source;
                sink_values[owned] = sink;

                if (boundary_overrides != nullptr &&
                    boundary_overrides->fixed_cell_value)
                {
                    auto fixed_value =
                        boundary_overrides->fixed_cell_value(cell_lid);
                    if (fixed_value.has_value() &&
                        (!std::isfinite(*fixed_value) ||
                         *fixed_value < positive_floor))
                    {
                        throw std::invalid_argument(
                            "TurbulenceScalarTransportEquation fixed-cell "
                            "values must be finite and at or above the "
                            "positive floor.");
                    }
                    fixed_cell_values[owned] = fixed_value;
                }
            }

            for (const auto& [batch_id, batch] :
                 d_mesh->boundary_batches())
            {
                const auto& name =
                    d_mesh->boundary_batch_name(batch_id);
                const auto configured = d_boundary_conditions.find(name);
                const auto configured_condition =
                    configured == d_boundary_conditions.end()
                        ? BoundaryCondition{}
                        : configured->second;

                for (size_t in_batch_id = 0;
                     in_batch_id < batch.face_lids.size();
                     ++in_batch_id)
                {
                    const auto face_lid = batch.face_lids[in_batch_id];
                    if (!d_mesh->is_boundary_face(face_lid))
                    {
                        continue;
                    }

                    auto condition = configured_condition;
                    bool condition_overridden = false;
                    if (boundary_overrides != nullptr &&
                        boundary_overrides->boundary_condition)
                    {
                        auto override_condition =
                            boundary_overrides->boundary_condition(
                                batch_id, in_batch_id);
                        if (override_condition.has_value())
                        {
                            condition = *override_condition;
                            condition_overridden = true;
                        }
                    }
                    if (!std::isfinite(condition.value) ||
                        !std::isfinite(condition.robin_coefficient))
                    {
                        throw std::invalid_argument(
                            "Turbulence scalar boundary '" + name +
                            "' contains non-finite dynamic data.");
                    }
                    if (condition.type !=
                            BoundaryConditionType::Dirichlet &&
                        condition.type != BoundaryConditionType::Neumann)
                    {
                        throw std::invalid_argument(
                            "Turbulence scalar boundary '" + name +
                            "' must be Dirichlet or Neumann.");
                    }

                    std::optional<scalar_type> override_value;
                    if (boundary_overrides != nullptr &&
                        boundary_overrides->boundary_value)
                    {
                        override_value =
                            boundary_overrides->boundary_value(
                                batch_id, in_batch_id);
                    }

                    scalar_type value{};
                    if (override_value.has_value())
                    {
                        value = *override_value;
                    }
                    else if (condition.type ==
                             BoundaryConditionType::Dirichlet)
                    {
                        value = condition.value;
                    }
                    else
                    {
                        const auto owner =
                            d_mesh->owner_cell(face_lid);
                        value = old_state.local_value(owner) +
                                condition.value * static_cast<scalar_type>(
                                    FVM::detail::boundary_normal_distance(
                                        *d_mesh, face_lid, owner));
                    }
                    if (!std::isfinite(value))
                    {
                        throw std::invalid_argument(
                            "Turbulence scalar boundary '" + name +
                            "' produced a non-finite face value.");
                    }
                    if (condition.type ==
                            BoundaryConditionType::Dirichlet &&
                        value < positive_floor)
                    {
                        const bool zero_override_allowed =
                            value == scalar_type{} &&
                            boundary_overrides != nullptr &&
                            boundary_overrides->allow_zero_dirichlet &&
                            (condition_overridden ||
                             override_value.has_value());
                        if (!zero_override_allowed)
                        {
                            throw std::invalid_argument(
                                "Turbulence scalar Dirichlet boundary '" +
                                name +
                                "' lies below the positive floor.");
                        }
                    }

                    auto& prepared = prepared_boundaries.at(
                        static_cast<size_t>(face_lid));
                    prepared.condition = condition;
                    prepared.value = value;
                    prepared.active = true;
                }
            }
        });

    bool all_diffusivities_positive = true;
    for (size_t local = 0; local < d_mesh->num_local_cells(); ++local)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(local);
        all_diffusivities_positive = all_diffusivities_positive &&
                                     effective_diffusivity.local_value(cell_lid) > scalar_type{};
    }

    auto boundary_condition = [&](int batch_id, size_t in_batch_id)
    {
        const auto face_lid =
            d_mesh->boundary_batches().at(batch_id).face_lids.at(
                in_batch_id);
        const auto& prepared = prepared_boundaries.at(
            static_cast<size_t>(face_lid));
        return prepared.active ? prepared.condition
                               : BoundaryCondition{};
    };
    auto boundary_value = [&](int batch_id, size_t in_batch_id) -> scalar_type
    {
        const auto face_lid =
            d_mesh->boundary_batches().at(batch_id).face_lids.at(
                in_batch_id);
        return prepared_boundaries.at(
            static_cast<size_t>(face_lid)).value;
    };
    auto source = [&](local_ordinal_type cell_lid) -> scalar_type
    { return source_values[static_cast<size_t>(cell_lid)]; };
    scalar_provider_type sink = [&](local_ordinal_type cell_lid) -> scalar_type
    { return sink_values[static_cast<size_t>(cell_lid)]; };
    typename boundary_overrides_type::fixed_cell_value_provider_type
        fixed_cell_value =
            [&](local_ordinal_type cell_lid)
                -> std::optional<scalar_type>
    {
        return fixed_cell_values.at(static_cast<size_t>(cell_lid));
    };
    const auto* correction_field =
        treatment == FVM::NonOrthogonalTreatment::Implicit ? nullptr : &old_state;
    const auto requires_non_orthogonal_graph = treatment != FVM::NonOrthogonalTreatment::Explicit;
    if (requires_non_orthogonal_graph && !d_cached_graph_supports_non_orthogonal_correction)
    {
        d_cached_transport_matrix = Teuchos::null;
    }

    auto system = FVM::weighted_scalar_transport_system<Pack>(
        old_state, face_fluxes, time_step, d_unit_weight, d_unit_weight, effective_diffusivity,
        boundary_condition, boundary_value, source, treatment, correction_field,
        d_cached_transport_matrix, std::move(sink),
        std::move(fixed_cell_value),
        boundary_overrides != nullptr
            ? boundary_overrides->boundary_diffusivity
            : nullptr);
    d_cached_transport_matrix = system.matrix;
    if (requires_non_orthogonal_graph && all_diffusivities_positive)
    {
        d_cached_graph_supports_non_orthogonal_correction = true;
    }

    field_type candidate(d_mesh, "turbulence_scalar_candidate");
    Teuchos::RCP<const typename Pack::matrix_type> matrix = system.matrix;
    const auto statistics = d_linear_solver.solve_with_statistics(
        matrix, *system.rhs, candidate.owned_data(), linear_options);
    turbulence_detail::collective_local_validation(
        *d_mesh, "Turbulence scalar transport candidate validation",
        [&]
        {
            if (!statistics.converged)
            {
                throw std::runtime_error("Turbulence scalar transport solve did not converge.");
            }
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell_lid = static_cast<local_ordinal_type>(owned);
                const auto value = candidate.value(cell_lid);
                if (!std::isfinite(value))
                {
                    throw std::runtime_error("Turbulence scalar transport produced a non-finite "
                                             "value.");
                }
                if (value < positive_floor)
                {
                    candidate.set_owned_value(cell_lid, positive_floor);
                }
            }
        });

    state.owned_data().update(scalar_type{1}, candidate.owned_data(), scalar_type{0});
    d_mesh->sync_periodic_boundaries(state);
    return statistics;
}

} // namespace SimpleFluid
