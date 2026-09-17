/** @file LiquidMassInventory.tcc @brief Compiled template implementations. */
#pragma once

#include "solvers/PlanarFreeSurfaceModel.hh"

#include <iomanip>
#include <sstream>

namespace SimpleFluid
{

template<TpetraTypePack Pack, class MeshType>
auto LiquidMassInventory<Pack, MeshType>::previewPhaseChange(
    scalar_type evaporated_mass, scalar_type condensed_mass) const -> PhaseChangePreview
{
    requireInitialized();
    if (d_options.mode != LiquidVolumeMode::GlobalConstantMass)
    {
        throw std::logic_error("previewPhaseChange is available only for globalConstantMass mode.");
    }
    const int local_invalid = !std::isfinite(evaporated_mass) || evaporated_mass < scalar_type{} ||
                                      !std::isfinite(condensed_mass) || condensed_mass < scalar_type{}
                                  ? 1
                                  : 0;
    int any_invalid = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
    {
        throw std::invalid_argument("Phase-change masses must be finite and non-negative.");
    }
    requireReplicated(evaporated_mass, "evaporated mass");
    requireReplicated(condensed_mass, "condensed mass");
    requireReplicated(static_cast<scalar_type>(d_phase_change_generation), "phase-change generation");

    auto preview = d_diagnostics;
    const auto available = preview.total_mass + condensed_mass;
    auto accepted_evaporation = evaporated_mass;
    auto deficit = scalar_type{};
    if (evaporated_mass > available)
    {
        deficit = evaporated_mass - available;
        if (d_options.depletion_policy == FreeSurfaceRangePolicy::Error)
        {
            throw std::out_of_range("Liquid evaporation exceeds the available liquid mass.");
        }
        accepted_evaporation = available;
    }

    preview.total_mass = available - accepted_evaporation;
    preview.cumulative_evaporated_mass += accepted_evaporation;
    preview.cumulative_condensed_mass += condensed_mass;
    preview.dryout_mass_deficit += deficit;
    preview.liquid_volume = preview.total_mass * preview.mass_weighted_specific_volume;
    preview.step_mass_balance_residual =
        d_diagnostics.total_mass + condensed_mass - accepted_evaporation - preview.total_mass;
    preview.normalized_step_mass_balance_residual =
        preview.step_mass_balance_residual /
        std::max(scalar_type{1}, std::abs(d_diagnostics.total_mass) + condensed_mass);
    updateMassBalance(preview);
    return PhaseChangePreview(this, d_phase_change_generation, std::move(preview), false, 0);
}

template<TpetraTypePack Pack, class MeshType>
auto LiquidMassInventory<Pack, MeshType>::previewCellwiseAdvance(scalar_type time_step,
    const face_flux_field_type& liquid_face_flux, const field_type* evaporation_mass_rate,
    const field_type* condensation_mass_rate, const LinearSolverOptions& linear_options,
    const FVM::ALEControlVolumeState* ale) -> PhaseChangePreview
{
    requireInitialized();
    const auto communicator = d_mesh->owned_cell_map()->getComm();

    const int backend = static_cast<int>(linear_options.backend);
    const int preconditioner = static_cast<int>(linear_options.preconditioner);
    const std::array<int, 13> local_control{static_cast<int>(d_options.mode),
        d_options.depletion_policy == FreeSurfaceRangePolicy::Error ? 1 : 0,
        evaporation_mass_rate != nullptr ? 1 : 0, condensation_mass_rate != nullptr ? 1 : 0,
        std::isfinite(time_step) && time_step > scalar_type{} ? 1 : 0,
        std::isfinite(linear_options.tolerance) && linear_options.tolerance > 0.0 ? 1 : 0,
        linear_options.max_iterations > 0 ? 1 : 0, linear_options.max_iterations, linear_options.verbosity, backend,
        preconditioner, linear_options.reuse_preconditioner ? 1 : 0, ale != nullptr ? 1 : 0};
    std::array<int, 13> minimum_control{};
    std::array<int, 13> maximum_control{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(local_control.size()),
        local_control.data(), minimum_control.data());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_control.size()),
        local_control.data(), maximum_control.data());
    if (minimum_control != maximum_control)
    {
        throw std::invalid_argument("Cellwise liquid-mass mode, optional sources, timestep validity, and linear "
                                    "solver controls must match on every rank.");
    }
    if (local_control[0] != static_cast<int>(LiquidVolumeMode::CellMassInventory))
    {
        throw std::logic_error("previewCellwiseAdvance requires cellMassInventory mode.");
    }
    if (local_control[1] == 0)
    {
        throw std::logic_error("cellMassInventory requires error depletion policy because source-side phase "
                               "acceptance cannot be conservatively clamped here.");
    }
    if (local_control[4] == 0 || linear_options.backend == LinearSolverBackend::Cg ||
        linear_options.preconditioner == LinearPreconditioner::DIC)
    {
        throw std::invalid_argument("Cellwise liquid-mass advance requires a positive finite timestep and "
                                    "nonsymmetric-compatible linear solver controls.");
    }
    // Enum values and numeric validity already agree across ranks above.
    // Share backend validation so newly supported transport policies work
    // here without maintaining a second list of enum limits.
    BelosLinearSolver<Pack>::validate_options(linear_options);
    const std::array<scalar_type, 2> local_scalar_control{
        time_step, static_cast<scalar_type>(linear_options.tolerance)};
    std::array<scalar_type, 2> minimum_scalar_control{};
    std::array<scalar_type, 2> maximum_scalar_control{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(local_scalar_control.size()),
        local_scalar_control.data(), minimum_scalar_control.data());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_scalar_control.size()),
        local_scalar_control.data(), maximum_scalar_control.data());
    if (minimum_scalar_control != maximum_scalar_control)
    {
        throw std::invalid_argument(
            "Cellwise liquid-mass timestep and linear tolerance must match exactly on every rank.");
    }
    if (ale != nullptr)
    {
        ale->validate(*d_mesh, static_cast<real_t>(time_step));
    }

    const int local_mesh_mismatch =
        &liquid_face_flux.mesh() != d_mesh.get() ||
        (evaporation_mass_rate != nullptr && &evaporation_mass_rate->mesh() != d_mesh.get()) ||
        (condensation_mass_rate != nullptr && &condensation_mass_rate->mesh() != d_mesh.get());
    int any_mesh_mismatch = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_mesh_mismatch, &any_mesh_mismatch);
    if (any_mesh_mismatch != 0)
    {
        throw std::invalid_argument("Cellwise liquid-mass flux and phase-change fields must use the inventory "
                                    "mesh on every rank.");
    }

    int local_invalid_value = 0;
    int local_boundary_flux = 0;
    int local_depleted_cell = 0;
    for (const auto face_lid : liquid_face_flux.owned_face_ids())
    {
        const auto flux = liquid_face_flux.value(face_lid);
        local_invalid_value = local_invalid_value || !std::isfinite(flux);
        local_boundary_flux = local_boundary_flux || (d_mesh->is_boundary_face(face_lid) && flux != scalar_type{});
    }
    {
        const auto old_mass = d_cell_mass_inventory.owned_read_view();
        const auto evaporation_values = evaporation_mass_rate
                ? evaporation_mass_rate->owned_read_view() : decltype(old_mass){};
        const auto condensation_values = condensation_mass_rate
                ? condensation_mass_rate->owned_read_view() : decltype(old_mass){};
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell = static_cast<local_ordinal_type>(owned);
            const auto mass_density = old_mass(owned, 0);
            const auto evaporation =
                evaporation_mass_rate == nullptr ? scalar_type{} : evaporation_values(owned, 0);
            const auto condensation =
                condensation_mass_rate == nullptr ? scalar_type{} : condensation_values(owned, 0);
            const auto new_volume = ale == nullptr ? static_cast<scalar_type>(d_mesh->cell_volume(cell))
                                                   : static_cast<scalar_type>(ale->new_cell_volumes()[owned]);
            const auto old_volume =
                ale == nullptr ? new_volume : static_cast<scalar_type>(ale->old_cell_volumes()[owned]);
            local_invalid_value = local_invalid_value || !std::isfinite(mass_density) || mass_density < scalar_type{} ||
                                  !std::isfinite(evaporation) || evaporation < scalar_type{} ||
                                  !std::isfinite(condensation) || condensation < scalar_type{} ||
                                  !std::isfinite(new_volume) || new_volume <= scalar_type{} ||
                                  !std::isfinite(old_volume) || old_volume <= scalar_type{};
            local_depleted_cell =
                local_depleted_cell ||
                mass_density * old_volume + time_step * (condensation - evaporation) * new_volume < scalar_type{};
        }
    }
    const std::array<int, 3> local_validation{local_invalid_value, local_boundary_flux, local_depleted_cell};
    std::array<int, 3> global_validation{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_validation.size()),
        local_validation.data(), global_validation.data());
    if (global_validation[0] != 0)
    {
        throw std::invalid_argument("Cellwise liquid mass, phase-change rates, cell volumes, and face fluxes must "
                                    "be finite and non-negative where required.");
    }
    if (global_validation[1] != 0)
    {
        throw std::invalid_argument("cellMassInventory currently requires zero physical-boundary liquid flux; "
                                    "inlet liquid-mass composition is not configured.");
    }
    if (global_validation[2] != 0)
    {
        throw std::out_of_range("Cellwise evaporation exceeds locally available liquid mass before transport.");
    }

    // Invalidate every older cellwise token before changing shared trial
    // storage, including when the following assembly or solve fails.
    d_cellwise_trial_nonce =
        invalidatedCounter(d_cellwise_trial_nonce, d_cellwise_trial_nonce, "cellwise trial nonce");
    d_trial_cell_mass_inventory.owned_data().update(
        scalar_type{1}, d_cell_mass_inventory.owned_data(), scalar_type{});
    d_trial_cell_mass_inventory.sync_ghosts();

    auto zero_neumann = [](int, size_t)
    { return BoundaryCondition{BoundaryConditionType::Neumann, scalar_type{}}; };
    auto zero_boundary_value = [](int, size_t) -> scalar_type { return scalar_type{}; };
    auto phase_change_source = [&](local_ordinal_type cell) -> scalar_type
    {
        const auto evaporation =
            evaporation_mass_rate == nullptr ? scalar_type{} : evaporation_mass_rate->value(cell);
        const auto condensation =
            condensation_mass_rate == nullptr ? scalar_type{} : condensation_mass_rate->value(cell);
        return condensation - evaporation;
    };
    auto system = FVM::weighted_scalar_transport_system<Pack>(
        FVM::MeshWeightedScalarTransportRequest<Pack, mesh_type>{.old_values = d_cell_mass_inventory,
            .face_fluxes = liquid_face_flux,
            .time_step = time_step,
            .storage_weight = d_unit_transport_weight,
            .advection_weight = d_unit_transport_weight,
            .diffusivity = d_zero_diffusivity,
            .boundary_condition = zero_neumann,
            .boundary_value = zero_boundary_value,
            .source = phase_change_source,
            .treatment = FVM::NonOrthogonalTreatment::Explicit,
            .cached_matrix = d_transport_matrix,
            .ale = ale});
    d_transport_matrix = system.matrix;
    auto transport_linear_options = linear_options;
    // The cached graph is refilled with new transient/flux/source values;
    // a preconditioner prepared for its previous numeric values is stale.
    transport_linear_options.reuse_preconditioner = false;
    auto statistics = d_transport_solver.solve_with_statistics(
        system.matrix, *system.rhs, d_trial_cell_mass_inventory.owned_data(), transport_linear_options);
    const int local_solve_failure = statistics.converged ? 0 : 1;
    int any_solve_failure = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_solve_failure, &any_solve_failure);
    if (any_solve_failure != 0)
    {
        throw std::runtime_error("Cellwise liquid-mass transport solve did not converge on every rank.");
    }
    // Refine the same conservative equation before publishing a trial. A
    // per-step error below the physical gate can otherwise accumulate until
    // the fixed initial-mass gate fails over many weakly heated substeps.
    const auto measure_trial = [&]()
    {
        detail::CompensatedSum<> local_mass_before{};
        detail::CompensatedSum<> local_evaporated{};
        detail::CompensatedSum<> local_condensed{};
        detail::CompensatedSum<> local_mass_after{};
        detail::CompensatedSum<> local_liquid_volume{};
        int invalid_trial_values = 0;
        {
            const auto old_mass = d_cell_mass_inventory.owned_read_view();
            const auto trial_mass = d_trial_cell_mass_inventory.owned_read_view();
            const auto pure_density = d_pure_liquid_density.owned_read_view();
            const auto evaporation_values = evaporation_mass_rate
                    ? evaporation_mass_rate->owned_read_view() : decltype(old_mass){};
            const auto condensation_values = condensation_mass_rate
                    ? condensation_mass_rate->owned_read_view() : decltype(old_mass){};
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const auto new_volume = ale == nullptr ? static_cast<scalar_type>(d_mesh->cell_volume(cell))
                                                       : static_cast<scalar_type>(ale->new_cell_volumes()[owned]);
                const auto old_volume =
                    ale == nullptr ? new_volume : static_cast<scalar_type>(ale->old_cell_volumes()[owned]);
                const auto mass_before = old_mass(owned, 0);
                const auto mass_after = trial_mass(owned, 0);
                const auto density = pure_density(owned, 0);
                const auto evaporation =
                    evaporation_mass_rate == nullptr ? scalar_type{} : evaporation_values(owned, 0);
                const auto condensation =
                    condensation_mass_rate == nullptr ? scalar_type{} : condensation_values(owned, 0);
                invalid_trial_values = invalid_trial_values || !std::isfinite(mass_after) || mass_after < scalar_type{} ||
                                      !std::isfinite(density) || density <= scalar_type{};
                local_mass_before += static_cast<long double>(mass_before) * old_volume;
                local_evaporated += static_cast<long double>(evaporation) * new_volume * time_step;
                local_condensed += static_cast<long double>(condensation) * new_volume * time_step;
                local_mass_after += static_cast<long double>(mass_after) * new_volume;
                local_liquid_volume += static_cast<long double>(mass_after) / density * new_volume;
            }
        }
        int any_invalid_trial = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &invalid_trial_values, &any_invalid_trial);
        if (any_invalid_trial != 0)
        {
            throw std::runtime_error("Cellwise liquid-mass transport produced a non-finite or negative inventory.");
        }
        const std::array<scalar_type, 5> local_totals{static_cast<scalar_type>(local_mass_before.value()),
            static_cast<scalar_type>(local_evaporated.value()), static_cast<scalar_type>(local_condensed.value()),
            static_cast<scalar_type>(local_mass_after.value()), static_cast<scalar_type>(local_liquid_volume.value())};
        std::array<scalar_type, 5> global_totals{};
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, static_cast<int>(local_totals.size()),
            local_totals.data(), global_totals.data());
        for (const auto value : global_totals)
        {
            if (!std::isfinite(value) || value < scalar_type{})
            {
                throw std::runtime_error("Cellwise liquid-mass global reduction is invalid.");
            }
        }
        return global_totals;
    };
    const auto mass_defect = [](const auto& totals)
    {
        return static_cast<scalar_type>(static_cast<long double>(totals[0])
            + totals[2] - totals[1] - totals[3]);
    };
    const auto roundoff_tolerance = [](const auto& totals)
    {
        const auto scale = std::max(totals[0] + totals[1] + totals[2], totals[3]);
        return scalar_type{64} * std::max(std::numeric_limits<scalar_type>::epsilon() * scale,
            std::numeric_limits<scalar_type>::denorm_min());
    };
    auto global_totals = measure_trial();
    if (std::abs(mass_defect(global_totals)) > roundoff_tolerance(global_totals))
    {
        typename Pack::vector_type residual(system.rhs->getMap(), false);
        typename Pack::vector_type correction(d_mesh->owned_cell_map(), false);
        const auto form_residual = [&]
        {
            residual.update(scalar_type{1}, *system.rhs, scalar_type{});
            system.matrix->apply(d_trial_cell_mass_inventory.owned_data(), residual,
                Teuchos::NO_TRANS, scalar_type{-1}, scalar_type{1});
        };
        auto refinement_options = transport_linear_options;
        refinement_options.reuse_preconditioner = true;
        constexpr int maximum_mass_refinements = 2;
        for (int refinement = 0; refinement < maximum_mass_refinements &&
             std::abs(mass_defect(global_totals)) > roundoff_tolerance(global_totals); ++refinement)
        {
            form_residual();
            const auto refined = d_transport_solver.solve_from_zero_with_statistics(
                system.matrix, residual, correction, refinement_options);
            statistics.iterations += refined.iterations;
            const int local_refinement_failure = refined.converged ? 0 : 1;
            int any_refinement_failure = 0;
            Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1,
                &local_refinement_failure, &any_refinement_failure);
            if (any_refinement_failure != 0)
                throw std::runtime_error("Cellwise liquid-mass residual correction did not converge.");
            d_trial_cell_mass_inventory.owned_data().update(scalar_type{1}, correction, scalar_type{1});
            global_totals = measure_trial();
        }
        // Report the final residual of the original equation, not the much
        // smaller correction RHS. Iterations include all bounded corrections.
        form_residual();
        const auto residual_norm = static_cast<real_t>(residual.norm2());
        statistics.achieved_tolerance = statistics.rhs_norm > real_t{}
            ? residual_norm / statistics.rhs_norm : residual_norm;
        statistics.converged = std::isfinite(statistics.achieved_tolerance) &&
            statistics.achieved_tolerance <= transport_linear_options.tolerance;
        if (!statistics.converged)
            throw std::runtime_error("Cellwise liquid-mass residual correction failed the original equation tolerance.");
    }
    if (std::abs(mass_defect(global_totals)) > roundoff_tolerance(global_totals))
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                << "Cellwise liquid-mass transport could not resolve its balance after residual correction: residual="
                << mass_defect(global_totals) << " kg, roundoff=" << roundoff_tolerance(global_totals)
                << " kg, strict physical tolerance="
                << massClosureTolerance(global_totals[0] + global_totals[1] + global_totals[2])
                << " kg, achieved linear relative residual=" << statistics.achieved_tolerance << '.';
        throw std::runtime_error(message.str());
    }

    auto preview = d_diagnostics;
    preview.total_mass = global_totals[3];
    preview.cumulative_evaporated_mass += global_totals[1];
    preview.cumulative_condensed_mass += global_totals[2];
    preview.liquid_volume = global_totals[4];
    preview.mass_weighted_specific_volume =
        preview.total_mass > scalar_type{} ? preview.liquid_volume / preview.total_mass : scalar_type{};
    preview.step_mass_balance_residual = mass_defect(global_totals);
    const auto step_scale = std::max(scalar_type{1}, global_totals[0] + global_totals[1] + global_totals[2]);
    preview.normalized_step_mass_balance_residual = preview.step_mass_balance_residual / step_scale;
    const auto tolerance = massClosureTolerance(step_scale);
    if (std::abs(preview.step_mass_balance_residual) > tolerance)
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                << "Cellwise liquid-mass step closure exceeded its strict physical tolerance: residual="
                << preview.step_mass_balance_residual << " kg, tolerance=" << tolerance
                << " kg, achieved linear relative residual=" << statistics.achieved_tolerance << '.';
        throw std::runtime_error(message.str());
    }
    updateMassBalance(preview);
    // Publish overlap values only after refinement and both unchanged physical
    // gates pass; accepted inventory remains untouched until commitPhaseChange.
    d_trial_cell_mass_inventory.sync_ghosts();
    return PhaseChangePreview(
        this, d_phase_change_generation, std::move(preview), true, d_cellwise_trial_nonce, statistics);
}

template<TpetraTypePack Pack, class MeshType>
void LiquidMassInventory<Pack, MeshType>::commitPhaseChange(const PhaseChangePreview& preview)
{
    const auto cellwise_mode = d_options.mode == LiquidVolumeMode::CellMassInventory;
    const int local_invalid = preview.d_owner != this || preview.d_generation != d_phase_change_generation ||
                                      preview.d_cellwise != cellwise_mode ||
                                      (cellwise_mode && preview.d_trial_nonce != d_cellwise_trial_nonce)
                                  ? 1
                                  : 0;
    int any_invalid = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
    {
        throw std::logic_error(
            "LiquidMassInventory phase-change preview is stale, foreign, or inconsistent across ranks.");
    }
    const auto next_generation =
        invalidatedCounter(d_phase_change_generation, d_phase_change_generation, "phase-change generation");
    if (cellwise_mode)
    {
        d_cell_mass_inventory.owned_data().update(
            scalar_type{1}, d_trial_cell_mass_inventory.owned_data(), scalar_type{});
        d_cell_mass_inventory.sync_ghosts();
    }
    d_diagnostics = preview.d_diagnostics;
    if (!cellwise_mode)
    {
        updateGlobalCellMassField();
    }
    d_phase_change_generation = next_generation;
}

template<TpetraTypePack Pack, class MeshType>
auto LiquidMassInventory<Pack, MeshType>::trialCellMassInventory(const PhaseChangePreview& preview) const -> const field_type&
{
    const auto cellwise_mode = d_options.mode == LiquidVolumeMode::CellMassInventory;
    const int local_invalid = !d_initialized || preview.d_owner != this ||
                              preview.d_generation != d_phase_change_generation || !preview.d_cellwise ||
                              !cellwise_mode || preview.d_trial_nonce != d_cellwise_trial_nonce;
    int any_invalid = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
    {
        throw std::logic_error("LiquidMassInventory trial field preview is stale, foreign, non-cellwise, or "
                               "inconsistent across ranks.");
    }
    return d_trial_cell_mass_inventory;
}

template<TpetraTypePack Pack, class MeshType>
auto LiquidMassInventory<Pack, MeshType>::snapshot() const -> StateSnapshot
{
    StateSnapshot result;
    result.d_owner = this;
    result.d_pure_density.resize(d_mesh->num_owned_cells());
    result.d_cell_mass.resize(d_mesh->num_owned_cells());
    result.d_trial_cell_mass.resize(d_mesh->num_owned_cells());
    {
        const auto density = d_pure_liquid_density.owned_read_view();
        const auto mass = d_cell_mass_inventory.owned_read_view();
        const auto trial_mass = d_trial_cell_mass_inventory.owned_read_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            result.d_pure_density[owned] = density(owned, 0);
            result.d_cell_mass[owned] = mass(owned, 0);
            result.d_trial_cell_mass[owned] = trial_mass(owned, 0);
        }
    }
    result.d_diagnostics = d_diagnostics;
    result.d_phase_change_generation = d_phase_change_generation;
    result.d_cellwise_trial_nonce = d_cellwise_trial_nonce;
    result.d_initialized = d_initialized;
    return result;
}

template<TpetraTypePack Pack, class MeshType>
void LiquidMassInventory<Pack, MeshType>::restore(const StateSnapshot& snapshot)
{
    const int local_invalid = snapshot.d_owner != this ||
                              snapshot.d_pure_density.size() != d_mesh->num_owned_cells() ||
                              snapshot.d_cell_mass.size() != d_mesh->num_owned_cells() ||
                              snapshot.d_trial_cell_mass.size() != d_mesh->num_owned_cells();
    int any_invalid = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
    {
        throw std::invalid_argument("LiquidMassInventory snapshot is foreign or has incompatible storage.");
    }
    const auto next_phase_change_generation = invalidatedCounter(
        d_phase_change_generation, snapshot.d_phase_change_generation, "phase-change generation");
    const auto next_cellwise_trial_nonce =
        invalidatedCounter(d_cellwise_trial_nonce, snapshot.d_cellwise_trial_nonce, "cellwise trial nonce");
    {
        auto density = d_pure_liquid_density.owned_write_view();
        auto mass = d_cell_mass_inventory.owned_write_view();
        auto trial_mass = d_trial_cell_mass_inventory.owned_write_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            density(owned, 0) = snapshot.d_pure_density[owned];
            mass(owned, 0) = snapshot.d_cell_mass[owned];
            trial_mass(owned, 0) = snapshot.d_trial_cell_mass[owned];
        }
    }
    d_pure_liquid_density.sync_ghosts();
    d_cell_mass_inventory.sync_ghosts();
    d_trial_cell_mass_inventory.sync_ghosts();
    d_diagnostics = snapshot.d_diagnostics;
    d_phase_change_generation = next_phase_change_generation;
    d_cellwise_trial_nonce = next_cellwise_trial_nonce;
    d_initialized = snapshot.d_initialized;
    invalidateGeometry();
}

template<TpetraTypePack Pack, class MeshType>
void LiquidMassInventory<Pack, MeshType>::updateVolumeFromStoredDensity()
{
    if (d_options.mode == LiquidVolumeMode::CellMassInventory)
    {
        detail::CompensatedSum<> local_liquid_volume{};
        {
            const auto mass = d_cell_mass_inventory.owned_read_view();
            const auto pure_density = d_pure_liquid_density.owned_read_view();
            for (size_t owned = 0; owned < d_reference_mass_fraction.size(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const auto mass_density = mass(owned, 0);
                const auto density = pure_density(owned, 0);
                const auto volume = static_cast<scalar_type>(d_mesh->cell_volume(cell));
                local_liquid_volume += mass_density / density * volume;
            }
        }
        d_diagnostics.liquid_volume = globalSum(static_cast<scalar_type>(local_liquid_volume.value()));
        d_diagnostics.mass_weighted_specific_volume = d_diagnostics.total_mass > scalar_type{}
                                                          ? d_diagnostics.liquid_volume / d_diagnostics.total_mass
                                                          : scalar_type{};
        return;
    }
    detail::CompensatedSum<> local_specific_volume = {};
    {
        const auto pure_density = d_pure_liquid_density.owned_read_view();
        for (size_t owned = 0; owned < d_reference_mass_fraction.size(); ++owned)
        {
            const auto density = pure_density(owned, 0);
            local_specific_volume += d_reference_mass_fraction[owned] / density;
        }
    }
    d_diagnostics.mass_weighted_specific_volume = globalSum(static_cast<scalar_type>(local_specific_volume.value()));
    d_diagnostics.liquid_volume = d_diagnostics.total_mass * d_diagnostics.mass_weighted_specific_volume;
}

template<TpetraTypePack Pack, class MeshType>
void LiquidMassInventory<Pack, MeshType>::updateMassBalance(diagnostics_type& diagnostics)
{
    diagnostics.mass_balance_residual = diagnostics.initial_mass + diagnostics.cumulative_condensed_mass -
                                        diagnostics.cumulative_evaporated_mass - diagnostics.total_mass;
    const auto scale =
        std::max(scalar_type{1}, std::abs(diagnostics.initial_mass) + diagnostics.cumulative_condensed_mass);
    diagnostics.normalized_mass_balance_residual = diagnostics.mass_balance_residual / scale;
    const auto tolerance = massClosureTolerance(scale);
    if (std::abs(diagnostics.mass_balance_residual) > tolerance)
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                << "LiquidMassInventory mass closure exceeded its accepted tolerance: residual="
                << diagnostics.mass_balance_residual << " kg, tolerance=" << tolerance
                << " kg, initial mass=" << diagnostics.initial_mass
                << " kg, condensed=" << diagnostics.cumulative_condensed_mass
                << " kg, evaporated=" << diagnostics.cumulative_evaporated_mass
                << " kg, current mass=" << diagnostics.total_mass << " kg.";
        throw std::logic_error(message.str());
    }
}


template<TpetraTypePack Pack, class MeshType>
void LiquidMassInventory<Pack, MeshType>::initialize_impl(scalar_type initial_liquid_volume, const std::function<scalar_type(local_ordinal_type)>& pure_liquid_density)
{
    const int local_initialized = d_initialized ? 1 : 0;
    int minimum_initialized = 0;
    int maximum_initialized = 0;
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_initialized, &minimum_initialized);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_initialized, &maximum_initialized);
    if (minimum_initialized != maximum_initialized)
    {
        throw std::logic_error("LiquidMassInventory initialized state must match on every rank.");
    }
    if (maximum_initialized != 0)
    {
        throw std::logic_error("LiquidMassInventory is already initialized.");
    }
    const auto next_phase_change_generation =
        invalidatedCounter(d_phase_change_generation, d_phase_change_generation, "phase-change generation");
    const auto next_cellwise_trial_nonce =
        invalidatedCounter(d_cellwise_trial_nonce, d_cellwise_trial_nonce, "cellwise trial nonce");

    const int local_invalid_volume =
        !std::isfinite(initial_liquid_volume) || initial_liquid_volume < scalar_type{} ? 1 : 0;
    int any_invalid_volume = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid_volume, &any_invalid_volume);
    if (any_invalid_volume != 0)
    {
        throw std::invalid_argument("Initial liquid volume must be finite and non-negative.");
    }
    requireReplicated(initial_liquid_volume, "initial liquid volume");
    const auto valid_mode = d_options.mode == LiquidVolumeMode::GlobalConstantMass ||
                            d_options.mode == LiquidVolumeMode::CellMassInventory;
    const int local_invalid_options =
        !valid_mode ||
                (d_options.mode == LiquidVolumeMode::CellMassInventory &&
                    d_options.depletion_policy != FreeSurfaceRangePolicy::Error) ||
                (d_options.initial_liquid_mass && (!std::isfinite(*d_options.initial_liquid_mass) ||
                                                      *d_options.initial_liquid_mass < scalar_type{}))
            ? 1
            : 0;
    int any_invalid_options = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_invalid_options, &any_invalid_options);
    if (any_invalid_options != 0)
    {
        throw std::invalid_argument("LiquidMassInventory requires a supported mode, a finite non-negative "
                                    "initial mass, and error depletion policy for cellMassInventory.");
    }
    const std::array<scalar_type, 4> local_options{static_cast<scalar_type>(d_options.mode),
        static_cast<scalar_type>(d_options.depletion_policy),
        d_options.initial_liquid_mass ? scalar_type{1} : scalar_type{},
        static_cast<scalar_type>(d_options.initial_liquid_mass.value_or(real_t{}))};
    std::array<scalar_type, 4> minimum_options{};
    std::array<scalar_type, 4> maximum_options{};
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, static_cast<int>(local_options.size()),
        local_options.data(), minimum_options.data());
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, static_cast<int>(local_options.size()),
        local_options.data(), maximum_options.data());
    if (minimum_options != maximum_options)
    {
        throw std::invalid_argument("LiquidMassInventory options must match on every rank.");
    }

    const auto count = d_mesh->num_owned_cells();
    d_density_scratch.resize(count);
    int local_invalid = 0;
    for (size_t owned = 0; owned < count; ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        try
        {
            d_density_scratch[owned] = pure_liquid_density(cell);
        }
        catch (...)
        {
            local_invalid = 1;
            d_density_scratch[owned] = scalar_type{};
        }
        if (!std::isfinite(d_density_scratch[owned]) || d_density_scratch[owned] <= scalar_type{})
        {
            local_invalid = 1;
        }
    }
    requireCollectivelyValidDensity(local_invalid);

    // Compensate local sums before rounding to the MPI scalar; long double
    // alone provides no extra precision on Apple Arm64.
    // Large cell counts otherwise inject a spurious volume source into ALE.
    detail::CompensatedSum<> local_reference_mass = {};
    detail::CompensatedSum<> local_mesh_volume = {};
    for (size_t owned = 0; owned < count; ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        const auto cell_volume = static_cast<scalar_type>(d_mesh->cell_volume(cell));
        if (!std::isfinite(cell_volume) || cell_volume <= scalar_type{})
        {
            local_invalid = 1;
            continue;
        }
        local_reference_mass += d_density_scratch[owned] * cell_volume;
        local_mesh_volume += cell_volume;
    }
    requireCollectivelyValidDensity(local_invalid);
    const auto reference_mass = globalSum(static_cast<scalar_type>(local_reference_mass.value()));
    const auto mesh_volume = globalSum(static_cast<scalar_type>(local_mesh_volume.value()));
    if (!(reference_mass > scalar_type{}) || !(mesh_volume > scalar_type{}))
    {
        throw std::invalid_argument("LiquidMassInventory requires positive global mesh volume and reference mass.");
    }

    d_reference_mass_fraction.resize(count);
    for (size_t owned = 0; owned < count; ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        d_reference_mass_fraction[owned] =
            d_density_scratch[owned] * static_cast<scalar_type>(d_mesh->cell_volume(cell)) / reference_mass;
        d_pure_liquid_density.set_value(cell, d_density_scratch[owned]);
    }
    d_pure_liquid_density.sync_ghosts();

    d_diagnostics.initial_mass = d_options.initial_liquid_mass
                                     ? static_cast<scalar_type>(*d_options.initial_liquid_mass)
                                     : initial_liquid_volume * reference_mass / mesh_volume;
    d_diagnostics.total_mass = d_diagnostics.initial_mass;
    for (size_t owned = 0; owned < count; ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        const auto cell_volume = static_cast<scalar_type>(d_mesh->cell_volume(cell));
        const auto mass_density = d_diagnostics.initial_mass * d_reference_mass_fraction[owned] / cell_volume;
        d_cell_mass_inventory.set_owned_value(cell, mass_density);
        d_trial_cell_mass_inventory.set_owned_value(cell, mass_density);
    }
    d_cell_mass_inventory.sync_ghosts();
    d_trial_cell_mass_inventory.sync_ghosts();
    d_initialized = true;
    d_phase_change_generation = next_phase_change_generation;
    d_cellwise_trial_nonce = next_cellwise_trial_nonce;
    updateVolumeFromStoredDensity();
    updateMassBalance();
}

template<TpetraTypePack Pack, class MeshType>
void LiquidMassInventory<Pack, MeshType>::update_pure_liquid_density_impl(const std::function<scalar_type(local_ordinal_type)>& pure_liquid_density)
{
    requireInitialized();
    const auto count = d_mesh->num_owned_cells();
    if (d_density_scratch.size() != count)
    {
        throw std::logic_error("LiquidMassInventory mesh ownership changed after initialization.");
    }
    int local_invalid = 0;
    int local_changed = 0;
    for (size_t owned = 0; owned < count; ++owned)
    {
        const auto cell = static_cast<local_ordinal_type>(owned);
        try
        {
            d_density_scratch[owned] = pure_liquid_density(cell);
        }
        catch (...)
        {
            local_invalid = 1;
            d_density_scratch[owned] = scalar_type{};
        }
        if (!std::isfinite(d_density_scratch[owned]) || d_density_scratch[owned] <= scalar_type{})
        {
            local_invalid = 1;
        }
        else if (d_density_scratch[owned] != d_pure_liquid_density.value(cell))
        {
            local_changed = 1;
        }
    }
    requireCollectivelyValidDensity(local_invalid);
    int any_changed = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_changed, &any_changed);
    for (size_t owned = 0; owned < count; ++owned)
    {
        d_pure_liquid_density.set_value(static_cast<local_ordinal_type>(owned), d_density_scratch[owned]);
    }
    d_pure_liquid_density.sync_ghosts();
    updateVolumeFromStoredDensity();
    if (any_changed != 0)
    {
        // A preview contains a liquid volume evaluated with the density
        // accepted at its creation.  Changing that density is an accepted
        // state mutation and must make every older preview uncommittable.
        d_phase_change_generation =
            invalidatedCounter(d_phase_change_generation, d_phase_change_generation, "phase-change generation");
    }
}

} // namespace SimpleFluid
