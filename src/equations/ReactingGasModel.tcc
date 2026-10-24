/**
 * @file ReactingGasModel.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for the runtime reacting gas model.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "equations/BoussFiniteVapor.hh"
#include "equations/BubblePopulationKinetics.hh"
#include "equations/ReactingGasModel.hh"

#include "FVM/CellOperators.hh"
#include "FVM/TransportSystem.hh"
#include "geometry/GeometryExecutionGuard.hh"
#include "utils/CompensatedSum.hh"

#include <Teuchos_CommHelpers.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace SimpleFluid
{

/**
 * @brief Require a mesh before constructing model-owned fields.
 * @tparam Pack Tpetra type pack used by the model.
 * @param mesh Candidate computational mesh.
 * @return The validated mesh pointer.
 * @throws std::invalid_argument if @p mesh is null.
 */
template<TpetraTypePack Pack, class MeshType>
SP<const typename ReactingGasModel<Pack, MeshType>::mesh_type>
ReactingGasModel<Pack, MeshType>::require_mesh(SP<const mesh_type> mesh)
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "ReactingGasModel requires a non-null mesh.");
    }
    return mesh;
}

template<TpetraTypePack Pack, class MeshType>
ReactingGasModel<Pack, MeshType>::ReactingGasModel(SP<const mesh_type> mesh, ReactingGasOptions options)
    : d_mesh(require_mesh(std::move(mesh))), d_transport_geometry_cache(*d_mesh),
      d_options(require_options(d_mesh, std::move(options))),
      d_species_thermal_source(d_mesh, "species_thermal_source"),
      d_species_material_source(d_mesh, "species_material_volume_source"),
      d_species_diffusivity(d_mesh, "species_diffusivity_scratch"),
      d_species_liquid_mass(d_mesh, "species_liquid_mass_density"),
      d_species_liquid_density(d_mesh, "species_pure_liquid_density"), d_alpha_g(d_mesh, "alpha_g"),
      d_alpha_l(d_mesh, "alpha_l"), d_source_alpha_reaction(d_mesh, "S_alpha_reaction"),
      d_absolute_pressure(d_mesh, "p_abs"), d_previous_temperature(d_mesh, "reacting_previous_temperature"),
      d_previous_alpha_g(d_mesh, "reacting_previous_alpha_g"), d_micro_number(d_mesh, "N_micro"),
      d_micro_moles(d_mesh, "M_micro"), d_large_number(d_mesh, "N_large"), d_large_moles(d_mesh, "M_large"),
      d_micro_radius(d_mesh, "r_micro"), d_large_radius(d_mesh, "r_large"), d_alpha_g_micro(d_mesh, "alpha_g_micro"),
      d_alpha_g_large(d_mesh, "alpha_g_large"), d_alpha_g_raw(d_mesh, "alpha_g_raw"),
      d_transport_alpha_g_raw(d_mesh, "alpha_g_transport_raw"), d_alpha_g_excess(d_mesh, "alpha_g_excess"),
      d_escape_number_rate(d_mesh, "bubble_escape_number_rate"),
      d_transport_bubble_slip_volume_flux(d_mesh, "bubble_transport_slip_volume_flux"),
      d_transport_bubble_carrier_volume_flux(d_mesh, "bubble_transport_carrier_volume_flux"),
      d_transport_liquid_material_volume_flux(d_mesh, "liquid_transport_material_volume_flux")
{
    d_transport_linear_options.tolerance = d_options.transport_solver_tolerance;

    register_output_fields();
    initialize_fields();
    configure_species_fields();
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::configure(const ReactingGasOptions& options)
{
    d_options = require_options(d_mesh, options);
    d_transport_workspace.reset();
    d_transport_linear_options = {};
    d_transport_linear_options.tolerance = d_options.transport_solver_tolerance;
    d_transport_solver.reset();
    initialize_fields();
    configure_species_fields();
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::set_transport_linear_solver_options(LinearSolverOptions options)
{
    constexpr auto context = "Reacting transport linear solver options";
    collective_detail::collective_local_validation(*d_mesh, context, [&]
    {
        BelosLinearSolver<Pack>::validate_options(options);
        if (options.backend == LinearSolverBackend::Cg)
        {
            throw std::invalid_argument(
                "Reacting transport requires GMRES or BiCGStab for its nonsymmetric systems.");
        }
    });
    collective_detail::require_uniform_value(
        *d_mesh, static_cast<int>(options.backend), context);
    collective_detail::require_uniform_value(
        *d_mesh, static_cast<int>(options.preconditioner), context);
    collective_detail::require_uniform_value(*d_mesh, options.max_iterations, context);
    collective_detail::require_uniform_value(*d_mesh, options.tolerance, context);
    collective_detail::require_uniform_value(*d_mesh, options.verbosity, context);
    collective_detail::require_uniform_value(
        *d_mesh, static_cast<int>(options.reuse_preconditioner), context);

    d_transport_solver.reset();
    d_transport_linear_options = std::move(options);
    d_options.transport_solver_tolerance = d_transport_linear_options.tolerance;
}

template<TpetraTypePack Pack, class MeshType> void ReactingGasModel<Pack, MeshType>::register_output_fields()
{
    d_output_fields = {{"alpha_g", &d_alpha_g}, {"alpha_l", &d_alpha_l}, {"S_alpha_reaction", &d_source_alpha_reaction},
        {"p_abs", &d_absolute_pressure}, {"N_micro", &d_micro_number}, {"M_micro", &d_micro_moles},
        {"N_large", &d_large_number}, {"M_large", &d_large_moles}, {"r_micro", &d_micro_radius},
        {"r_large", &d_large_radius}, {"alpha_g_micro", &d_alpha_g_micro}, {"alpha_g_large", &d_alpha_g_large},
        {"alpha_g_raw", &d_alpha_g_raw}, {"alpha_g_excess", &d_alpha_g_excess}};
}

template<TpetraTypePack Pack, class MeshType> void ReactingGasModel<Pack, MeshType>::initialize_fields()
{
    d_history_initialized = false;
    d_initial_state_initialized = false;
    d_absolute_pressure_offset = d_options.reference_pressure;
    d_last_statistics = {};
    for (auto* field : mutable_state_fields())
        field->put_scalar(0);
    d_alpha_g.put_scalar(d_options.alpha_min);
    d_alpha_l.put_scalar(1 - d_options.alpha_min);
    d_alpha_g_raw.put_scalar(d_options.alpha_min);
    d_transport_alpha_g_raw.put_scalar(d_options.alpha_min);
    d_previous_alpha_g.put_scalar(d_options.alpha_min);
    d_absolute_pressure.put_scalar(d_absolute_pressure_offset);
    d_transport_bubble_slip_volume_flux.put_scalar(0);
    d_transport_bubble_carrier_volume_flux.put_scalar(0);
    d_transport_liquid_material_volume_flux.put_scalar(0);
    d_liquid_transport_volume_receipt = {};
    sync_all_fields();
    if constexpr (requires { d_transport_bubble_slip_volume_flux.sync_ghosts(); })
    {
        d_transport_bubble_slip_volume_flux.sync_ghosts();
        d_transport_bubble_carrier_volume_flux.sync_ghosts();
        d_transport_liquid_material_volume_flux.sync_ghosts();
    }
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::initialize_state(scalar_type time, const field_type& temperature,
    const field_type& gauge_pressure, const velocity_field_type& velocity, const material_type& material, bool force)
{
    if (!enabled())
        return;

    StateSnapshot prior;
    collective_detail::collective_local_validation(*d_mesh, "Reacting complete initialization snapshot",
        [&] { prior = snapshot(); });
    try
    {
        reconstruct_absolute_pressure(time, gauge_pressure);
        if (!d_initial_state_initialized || (force && d_options.initial_species_state))
        {
            std::vector<Reacting::CellSpeciesState> states;
            std::vector<std::vector<double>> numbers;
            collective_detail::collective_local_validation(*d_mesh, "Reacting typed initialization callback",
                [&]
                {
                    if (!d_options.initial_species_state)
                        throw std::logic_error("Supply initial_species_state or initialize_species before deriving typed fields.");
                    const auto& catalog = d_options.reaction_mechanism->catalog();
                    for (std::size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
                    {
                        const auto cell = static_cast<local_ordinal_type>(owned);
                        auto environment =
                            species_environment(cell, time, temperature.value(cell), d_mesh->cell_volume(cell));
                        environment.liquidVolume = 0.0; // Preserve the initial callback's existing volume contract.
                        states.push_back(d_options.initial_species_state(environment,
                            static_cast<std::int64_t>(d_mesh->cell_global_id(cell))));
                        const Reacting::ReactionContext context{catalog, states.back(), environment};
                        numbers.push_back(d_options.initial_population_numbers ? d_options.initial_population_numbers(context)
                            : std::vector<double>(d_options.population_ids.size(), 0.0));
                    }
                });
            initialize_species(states, numbers);
        }
        reconstruct_species_derived(temperature);
        sync_all_fields();
        return;
    }
    catch (...)
    {
        restore_for_initialization_rollback(prior);
        throw;
    }
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::global_sum(scalar_type local_value) const -> scalar_type
{
    scalar_type global_value{};
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_SUM,
        1,
        &local_value,
        &global_value);
    return global_value;
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::global_min(scalar_type local_value) const -> scalar_type
{
    scalar_type global_value{};
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        1,
        &local_value,
        &global_value);
    return global_value;
}

template<TpetraTypePack Pack, class MeshType> int ReactingGasModel<Pack, MeshType>::global_max(int local_value) const
{
    int global_value{};
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_value,
        &global_value);
    return global_value;
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::global_integral(
    const field_type& field, std::span<const real_t> cell_volumes) const -> scalar_type
{
    if (!cell_volumes.empty() && cell_volumes.size() != d_mesh->num_local_cells())
    {
        throw std::invalid_argument("Reacting integral volume span must use mesh-local cell order.");
    }
    detail::CompensatedSum<> local_integral;
    {
        const auto values = field.owned_read_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto volume = cell_volumes.empty()
                                    ? static_cast<scalar_type>(d_mesh->cell_volume(cell_lid))
                                    : static_cast<scalar_type>(cell_volumes[owned]);
            local_integral += static_cast<long double>(values(owned, 0)) * volume;
        }
    }
    return global_sum(static_cast<scalar_type>(local_integral.value()));
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::global_submerged_bubble_volume() const -> scalar_type
{
    return global_integral(d_alpha_g_raw);
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::global_unrepresented_bubble_volume() const -> scalar_type
{
    return global_integral(d_alpha_g_excess);
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::minimum_valid_absolute_pressure_offset() const -> scalar_type
{
    if (d_options.pressure_mode != ReactingPressureMode::Constant
        && d_options.pressure_mode != ReactingPressureMode::Reconstructed)
    {
        throw std::logic_error(
            "A coupled pressure-offset domain is available only for constant "
            "or reconstructed reacting pressure mode.");
    }

    auto local_minimum_gauge =
        std::numeric_limits<scalar_type>::infinity();
    int local_invalid_pressure = 0;
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto gauge_variation =
            d_absolute_pressure.value(cell_lid)
          - d_absolute_pressure_offset;
        if (!std::isfinite(gauge_variation))
        {
            local_invalid_pressure = 1;
            continue;
        }
        local_minimum_gauge =
            std::min(local_minimum_gauge, gauge_variation);
    }
    if (global_max(local_invalid_pressure) != 0)
    {
        throw std::runtime_error(
            "The accepted reacting pressure field contains a non-finite "
            "gauge variation.");
    }

    const auto minimum_gauge = global_min(local_minimum_gauge);
    if (!std::isfinite(minimum_gauge))
    {
        throw std::runtime_error(
            "The reacting pressure-offset domain requires at least one "
            "globally owned cell.");
    }
    const auto minimum_offset = std::max(
        static_cast<scalar_type>(d_options.minimum_absolute_pressure),
        static_cast<scalar_type>(d_options.minimum_absolute_pressure)
          - minimum_gauge);
    if (!std::isfinite(minimum_offset))
    {
        throw std::runtime_error(
            "The accepted gauge-pressure variation does not admit a finite "
            "absolute-pressure offset.");
    }
    return minimum_offset;
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::evaluate_submerged_bubble_volume(
    scalar_type candidate_absolute_pressure_offset) const -> scalar_type
{
    return evaluate_species_bubble_volume(candidate_absolute_pressure_offset);
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::set_absolute_pressure_offset(scalar_type pressure_offset)
{
    if (d_options.pressure_mode != ReactingPressureMode::Constant &&
        d_options.pressure_mode != ReactingPressureMode::Reconstructed)
    {
        throw std::logic_error("External pressure offsets require constant or reconstructed "
                               "reacting pressure mode.");
    }
    const int local_invalid_candidate = !std::isfinite(pressure_offset);
    if (global_max(local_invalid_candidate) != 0)
    {
        throw std::invalid_argument("Absolute-pressure offset must be finite on every rank.");
    }
    scalar_type minimum_candidate{};
    scalar_type maximum_candidate{};
    Teuchos::reduceAll(
        *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &pressure_offset, &minimum_candidate);
    Teuchos::reduceAll(
        *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &pressure_offset, &maximum_candidate);
    if (minimum_candidate != maximum_candidate)
    {
        throw std::invalid_argument("Absolute-pressure offset must match exactly on every rank.");
    }
    const auto minimum_offset = minimum_valid_absolute_pressure_offset();
    const int local_outside_domain = pressure_offset < minimum_offset;
    if (global_max(local_outside_domain) != 0)
    {
        throw std::invalid_argument("Absolute-pressure offset is below the collective minimum allowed "
                                    "by the configured pressure floor and accepted gauge-pressure "
                                    "variation.");
    }

    if (d_initial_state_initialized)
    {
        set_species_absolute_pressure_offset(pressure_offset);
        return;
    }
    const auto shift = pressure_offset - d_absolute_pressure_offset;
    for (std::size_t row = 0; row < d_mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<local_ordinal_type>(row);
        d_absolute_pressure.set_owned_value(cell, d_absolute_pressure.value(cell) + shift);
    }
    d_absolute_pressure_offset = pressure_offset;
    d_absolute_pressure.sync_ghosts();
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::prescribed_pressure(scalar_type time) const -> scalar_type
{
    const auto& times = d_options.pressure_history_times;
    const auto& values = d_options.pressure_history_values;
    if (times.empty())
        return d_options.reference_pressure;
    if (time <= times.front())
        return values.front();
    if (time >= times.back())
        return values.back();

    const auto upper =
        std::upper_bound(times.begin(), times.end(), time);
    const auto high =
        static_cast<size_t>(std::distance(times.begin(), upper));
    const auto low = high - 1;
    const auto fraction =
        (time - times[low]) / (times[high] - times[low]);
    return values[low]
         + fraction * (values[high] - values[low]);
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::reconstruct_absolute_pressure(scalar_type time, const field_type& gauge_pressure)
{
    if (d_options.pressure_mode == ReactingPressureMode::Constant ||
        d_options.pressure_mode == ReactingPressureMode::PrescribedHistory)
    {
        d_absolute_pressure.put_scalar(
            d_options.pressure_mode
                    == ReactingPressureMode::Constant
                ? d_absolute_pressure_offset
                : prescribed_pressure(time));
        d_absolute_pressure.sync_ghosts();
        return;
    }

    scalar_type local_pressure_volume{};
    scalar_type local_volume{};
    {
        const auto gauge_values = gauge_pressure.owned_read_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto volume = d_mesh->cell_volume(cell_lid);
            local_pressure_volume += gauge_values(owned, 0) * volume;
            local_volume += volume;
        }
    }
    const auto volume = global_sum(local_volume);
    const auto mean_pressure =
        volume > 0.0
            ? global_sum(local_pressure_volume) / volume
            : 0.0;

    {
        const auto gauge_values = gauge_pressure.owned_read_view();
        const auto pressure_values = d_absolute_pressure.owned_write_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            auto pressure = d_absolute_pressure_offset + gauge_values(owned, 0) - mean_pressure;
            if (pressure < d_options.minimum_absolute_pressure)
            {
                ++d_last_statistics.pressure_floor_cells;
                pressure = d_options.minimum_absolute_pressure;
            }
            pressure_values(owned, 0) = pressure;
        }
    }
    d_absolute_pressure.sync_ghosts();
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::transport_scalar(field_type& field, scalar_type time_step,
    const face_flux_field_type& liquid_face_flux, field_type* slip_velocity, scalar_type diffusivity, bool diffuse,
    bool liquid_weighted, field_type& escape_rate, const FVM::ALEControlVolumeState* ale, Dimension slip_axis,
    size_t operator_slot, bool reuse_population_operator, const field_type* species_diffusivity,
    face_flux_field_type* canonical_molar_packet)
{
    const auto started = std::chrono::steady_clock::now();
    const auto elapsed = [](auto begin)
    { return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(); };
    // Four operator roles share five work lanes: solution, population
    // number/material, the auxiliary scalar role, and stationary Resolved.
    const size_t field_index =
        operator_slot == 1 ? 1 + size_t(reuse_population_operator) : operator_slot + (operator_slot > 1 ? 1 : 0);
    auto& work = d_last_statistics.transport_work.at(field_index);

    if (&escape_rate.mesh() != d_mesh.get())
    {
        throw std::invalid_argument(
            "Reacting escape rate is on the wrong mesh.");
    }
    const auto slip_component = static_cast<int>(slip_axis);
    if (slip_component < static_cast<int>(Dimension::X) || slip_component > static_cast<int>(Dimension::Z))
    {
        throw std::invalid_argument("Reacting bubble slip axis must be X, Y, or Z.");
    }
    if (ale != nullptr)
    {
        ale->validate(*d_mesh, static_cast<real_t>(time_step));
    }

    auto is_free_surface = [&](local_ordinal_type face_lid)
    {
        if (!d_mesh->is_boundary_face(face_lid))
            return false;
        const auto& name = d_mesh->boundary_batch_name(
            d_mesh->boundary_id(face_lid));
        return std::ranges::find(
                   d_options.free_surface_patches, name)
            != d_options.free_surface_patches.end();
    };

    auto& workspace = *d_transport_workspace;
    auto& transport_flux = workspace.flux;
    auto& old_values = workspace.old_values;
    auto& storage_weight = workspace.storage;
    auto& diffusion_weight = workspace.diffusion;
    auto& solution = workspace.solution;
    auto& system = workspace.systems.at(operator_slot);
    // If the number equation was skipped, a populated moles equation must
    // assemble current coefficients instead of reusing a previous timestep.
    reuse_population_operator = reuse_population_operator && workspace.operator_ready.at(operator_slot);
    const auto assembly_started = std::chrono::steady_clock::now();
    if (!reuse_population_operator)
    {
        {
            const auto liquid_flux_values = liquid_face_flux.owned_read_view();
            const auto slip_values = slip_velocity ? slip_velocity->local_read_view()
                                                  : decltype(field.local_read_view()){};
            const auto flux_values = transport_flux.owned_write_view();
            for (const auto face_lid : transport_flux.owned_face_ids())
            {
                auto flux = liquid_face_flux.is_owned_face(face_lid)
                    ? liquid_flux_values(liquid_face_flux.owned_row(face_lid), 0)
                    : scalar_type{};
                if (slip_velocity)
                {
                    const auto owner = d_mesh->owner_cell(face_lid);
                    auto face_slip = slip_values(owner, 0);
                    if (!d_mesh->is_boundary_face(face_lid))
                    {
                        const auto neighbor =
                            d_mesh->opposite_or_periodic_neighbor_cell(face_lid, owner);
                        face_slip = 0.5 * (face_slip + slip_values(neighbor, 0));
                    }
                    flux += face_slip * d_mesh->face_area_vector(face_lid).component(
                                            static_cast<size_t>(slip_component));
                }
                if (d_mesh->is_boundary_face(face_lid))
                {
                    flux = is_free_surface(face_lid)
                        ? std::max(flux, scalar_type{})
                        : scalar_type{};
                }
                flux_values(transport_flux.owned_row(face_lid), 0) = flux;
            }
        }
        if constexpr (requires { transport_flux.sync_ghosts(); })
        {
            transport_flux.sync_ghosts();
        }

        storage_weight.put_scalar(1.0);
        diffusion_weight.put_scalar(0.0);
        const auto assemble_weights = [&]
        {
            const auto liquid_values = d_alpha_l.owned_read_view();
            const auto explicit_diffusivity =
                species_diffusivity ? species_diffusivity->owned_read_view() : decltype(field.owned_read_view()){};
            const auto field_values = field.owned_read_view();
            const auto old = old_values.owned_write_view();
            const auto storage = storage_weight.owned_write_view();
            const auto diffusion = diffusion_weight.owned_write_view();
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto raw_liquid_fraction = liquid_values(owned, 0);
                if ((!std::isfinite(raw_liquid_fraction) || raw_liquid_fraction < 0.0 ||
                        (raw_liquid_fraction == 0.0 && liquid_weighted && field_values(owned, 0) != 0.0)))
                    throw std::invalid_argument("Species transport needs a finite physical liquid carrier.");
                const auto liquid_fraction = raw_liquid_fraction == 0 ? scalar_type{1} : raw_liquid_fraction;
                old(owned, 0) = liquid_weighted ? field_values(owned, 0) / liquid_fraction : field_values(owned, 0);
                if (liquid_weighted)
                    storage(owned, 0) = liquid_fraction;
                const auto cell_diffusivity = species_diffusivity ? explicit_diffusivity(owned, 0) : diffusivity;
                diffusion(owned, 0) = diffuse ? raw_liquid_fraction * cell_diffusivity : 0.0;
            }
        };
        if (diffuse)
            collective_detail::collective_local_validation(*d_mesh, "Reacting diffusivity evaluation", assemble_weights);
        else
            assemble_weights();
        old_values.sync_ghosts();
        if (liquid_weighted)
            storage_weight.sync_ghosts();
        if (diffuse)
            diffusion_weight.sync_ghosts();

        auto boundary_condition =
            [](int, size_t)
        {
            return BoundaryCondition{
                BoundaryConditionType::Neumann, 0.0};
        };
        auto boundary_value =
            [](int, size_t) -> scalar_type { return 0.0; };
        auto source =
            [](local_ordinal_type) -> scalar_type { return 0.0; };
        system = FVM::weighted_scalar_transport_system<Pack>(
            FVM::MeshWeightedScalarTransportRequest<Pack, mesh_type>{
                .old_values = old_values,
                .face_fluxes = transport_flux,
                .time_step = time_step,
                .storage_weight = storage_weight,
                // Storage and advection use the same frozen liquid fraction.
                .advection_weight = storage_weight,
                .diffusivity = diffusion_weight,
                .boundary_condition = boundary_condition,
                .boundary_value = boundary_value,
                .source = source,
                .treatment = FVM::NonOrthogonalTreatment::Hybrid,
                .correction_field = &old_values,
                .cached_matrix = system.matrix,
                .geometry_cache = &d_transport_geometry_cache,
                .ale = ale,
                .symbolic_plan = &workspace.symbolic_plans.at(operator_slot),
                .cached_rhs = system.rhs});
        workspace.operator_ready.at(operator_slot) = true;
        ++work.assemblies;
    }
    else
    {
        // The immediately preceding number solve assembled this population's
        // frozen BE/upwind operator: unit storage/advection, zero diffusion,
        // homogeneous Neumann boundaries and no source. Only its RHS changes.
        // Preserve the assembly multiplication order and accepted-old ALE volume.
        if (liquid_weighted || diffuse || operator_slot == 0 || system.matrix.is_null())
            throw std::logic_error("Invalid reacting population operator reuse.");
        const auto values = field.owned_read_view();
        const auto rhs = system.rhs->getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto volume = ale == nullptr
                ? static_cast<scalar_type>(d_mesh->cell_volume(static_cast<local_ordinal_type>(owned)))
                : static_cast<scalar_type>(ale->old_cell_volumes()[owned]);
            rhs(owned, 0) = volume / time_step * values(owned, 0);
        }
    }

    work.assembly_seconds += elapsed(assembly_started);
    const auto solve_started = std::chrono::steady_clock::now();
    // Keep the original zero initial guess. The shared matrix is numerically
    // refreshed each stage; only its adjacent number/moles solves share a factor.
    auto solve_options = d_transport_linear_options;
    solve_options.reuse_preconditioner = reuse_population_operator;
    const auto solve_statistics = d_transport_solver.solve_from_zero_with_statistics(
        system.matrix, *system.rhs, solution.owned_data(), solve_options);
    d_last_statistics.transport_linear.add(solve_statistics);
    ++work.solves;
    work.iterations += solve_statistics.iterations;
    work.solve_seconds += elapsed(solve_started);
    if (!solve_statistics.converged)
    {
        throw std::runtime_error(
            "Reacting weighted transport solve did not converge.");
    }

    // A configured RHS-relative linear tolerance can leave an inventory
    // defect larger than the transport ledger's existing 64-epsilon gate.
    // Check the physical balance before publishing fields or escape rates;
    // refine the same equation rather than rescaling its conserved inventory.
    struct TransportBalance
    {
        scalar_type before, after, outflow, residual, tolerance;
        bool satisfied() const { return std::abs(residual) <= tolerance; }
    };
    const auto transport_balance = [&]() -> TransportBalance
    {
        std::array<detail::CompensatedSum<>, 3> accumulated;
        int invalid_candidate = 0;
        size_t first_invalid_row = d_mesh->num_owned_cells();
        {
            const auto previous = field.owned_read_view();
            const auto candidate = solution.owned_read_view();
            const auto storage = storage_weight.owned_read_view();
            const auto advection = storage_weight.local_read_view();
            const auto flux = transport_flux.owned_read_view();
            // Match publication's nonnegative inventory gate. An invalid
            // candidate is rejected, never repaired for the balance sum.
            const auto published_value = [&](size_t owned)
            {
                auto value = candidate(owned, 0);
                if (liquid_weighted)
                    value *= storage(owned, 0);
                if (!std::isfinite(value) || value < scalar_type{})
                {
                    invalid_candidate = 1;
                    if (first_invalid_row == d_mesh->num_owned_cells())
                        first_invalid_row = owned;
                    return scalar_type{};
                }
                return value;
            };
            for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
            {
                const auto cell = static_cast<local_ordinal_type>(owned);
                const auto new_volume = ale == nullptr
                    ? static_cast<scalar_type>(d_mesh->cell_volume(cell))
                    : static_cast<scalar_type>(ale->new_cell_volumes()[owned]);
                const auto old_volume = ale == nullptr ? new_volume
                    : static_cast<scalar_type>(ale->old_cell_volumes()[owned]);
                accumulated[0] += static_cast<long double>(previous(owned, 0)) * old_volume;
                accumulated[1] += static_cast<long double>(published_value(owned)) * new_volume;
            }
            for (const auto face : transport_flux.owned_face_ids())
            {
                if (!is_free_surface(face))
                    continue;
                const auto owner = d_mesh->owner_cell(face);
                const auto value = published_value(static_cast<size_t>(owner));
                const auto primary = liquid_weighted ? value / storage(owner, 0) : value;
                const auto boundary_rate = std::max(
                    flux(transport_flux.owned_row(face), 0), scalar_type{})
                    * advection(owner, 0) * primary;
                accumulated[2] += static_cast<long double>(boundary_rate) * time_step;
            }
        }
        const std::array<scalar_type, 4> local{
            static_cast<scalar_type>(accumulated[0].value()),
            static_cast<scalar_type>(accumulated[1].value()),
            static_cast<scalar_type>(accumulated[2].value()),
            static_cast<scalar_type>(invalid_candidate)};
        std::array<scalar_type, 4> global{};
        Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM,
            static_cast<int>(local.size()), local.data(), global.data());
        if (global[3] != scalar_type{} || !std::isfinite(global[0]) ||
            !std::isfinite(global[1]) || !std::isfinite(global[2]))
        {
            const auto comm = d_mesh->owned_cell_map()->getComm();
            const int local_bad_rank =
                first_invalid_row < d_mesh->num_owned_cells() ? comm->getRank() : std::numeric_limits<int>::max();
            int first_bad_rank = std::numeric_limits<int>::max();
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_bad_rank, &first_bad_rank);
            // All ranks enter the same error fence, including empty owners.
            // Only the first bad owner formats its original row; peers reject.
            collective_detail::collective_local_validation(*d_mesh, "Reacting transport invalid candidate inventory",
                [&]
                {
                    if (comm->getRank() != first_bad_rank)
                        return;
                    const auto cell = static_cast<local_ordinal_type>(first_invalid_row);
                    const auto row_gid = system.matrix->getRowMap()->getGlobalElement(cell);
                    typename Pack::matrix_type::local_inds_host_view_type columns;
                    typename Pack::matrix_type::values_host_view_type coefficients;
                    system.matrix->getLocalRowView(cell, columns, coefficients);
                    scalar_type diagonal{}, off_min = std::numeric_limits<scalar_type>::infinity(),
                                            off_max = -std::numeric_limits<scalar_type>::infinity();
                    size_t off_count = 0, positive_off = 0, nonfinite_coefficients = 0;
                    for (size_t entry = 0; entry < columns.extent(0); ++entry)
                    {
                        const auto value = coefficients(entry);
                        nonfinite_coefficients += !std::isfinite(value);
                        if (system.matrix->getColMap()->getGlobalElement(columns(entry)) == row_gid)
                            diagonal += value;
                        else
                        {
                            ++off_count;
                            positive_off += value > 0;
                            off_min = std::min(off_min, value);
                            off_max = std::max(off_max, value);
                        }
                    }
                    if (off_count == 0)
                        off_min = off_max = 0;
                    std::string_view field_name = "unregistered";
                    for (const auto& [name, pointer] : d_output_fields)
                        if (pointer == &field)
                        {
                            field_name = name;
                            break;
                        }
                    auto geometry_gid = d_mesh->cell_global_id(cell);
                    if constexpr (requires { d_mesh->cell_geometry_global_id(cell); })
                        geometry_gid = d_mesh->cell_geometry_global_id(cell);
                    const auto raw = solution.owned_read_view()(first_invalid_row, 0);
                    const auto weight = storage_weight.owned_read_view()(first_invalid_row, 0);
                    std::ostringstream message;
                    message << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                            << "Reacting transport produced an invalid candidate inventory: field=" << field_name
                            << " operator_slot=" << operator_slot << " work_lane=" << field_index
                            << " rank=" << first_bad_rank << " cell_lid=" << first_invalid_row
                            << " cell_gid=" << row_gid << " geometry_gid=" << geometry_gid
                            << " old=" << field.owned_read_view()(first_invalid_row, 0) << " raw_candidate=" << raw
                            << " published_candidate=" << (liquid_weighted ? raw * weight : raw)
                            << " storage=" << weight
                            << " rhs=" << system.rhs->getLocalViewHost(Tpetra::Access::ReadOnly)(first_invalid_row, 0)
                            << " diagonal=" << diagonal << " offdiag_min=" << off_min << " offdiag_max=" << off_max
                            << " positive_offdiagonals=" << positive_off
                            << " nonfinite_coefficients=" << nonfinite_coefficients
                            << " solver=" << to_string(solve_options.backend) << '/'
                            << to_string(solve_options.preconditioner) << " iterations=" << solve_statistics.iterations
                            << " rhs_norm=" << solve_statistics.rhs_norm
                            << " achieved_tolerance=" << solve_statistics.achieved_tolerance
                            << " requested_tolerance=" << solve_options.tolerance
                            << " converged=" << solve_statistics.converged;
                    throw std::runtime_error(message.str());
                });
            throw std::runtime_error("Reacting transport produced an invalid candidate inventory.");
        }
        return {global[0], global[1], global[2],
            (global[1] - global[0]) + global[2],
            // Below the normal range, epsilon * inventory can underflow to
            // zero. Bound this refinement check by 64 representable ULPs;
            // the downstream escape and donor gates remain unchanged.
            scalar_type{64} * std::max(
                std::numeric_limits<scalar_type>::epsilon()
                    * std::max(std::abs(global[0]), std::abs(global[1])),
                std::numeric_limits<scalar_type>::denorm_min())};
    };
    auto balance = transport_balance();
    constexpr int maximum_inventory_refinements = 2;
    for (int refinement = 0;
         refinement < maximum_inventory_refinements && !balance.satisfied(); ++refinement)
    {
        const auto refinement_started = std::chrono::steady_clock::now();
        typename Pack::vector_type residual(system.rhs->getMap(), false);
        typename Pack::vector_type correction(solution.owned_data().getMap(), false);
        residual.update(scalar_type{1}, *system.rhs, scalar_type{});
        system.matrix->apply(solution.owned_data(), residual, Teuchos::NO_TRANS,
            scalar_type{-1}, scalar_type{1});
        auto refinement_options = solve_options;
        refinement_options.reuse_preconditioner = true;
        const auto refinement_statistics = d_transport_solver.solve_from_zero_with_statistics(
            system.matrix, residual, correction, refinement_options);
        d_last_statistics.transport_linear.add(refinement_statistics);
        ++work.solves;
        work.iterations += refinement_statistics.iterations;
        work.solve_seconds += elapsed(refinement_started);
        if (!refinement_statistics.converged)
            throw std::runtime_error("Reacting inventory residual correction did not converge.");
        solution.owned_data().update(scalar_type{1}, correction, scalar_type{1});
        balance = transport_balance();
    }
    if (!balance.satisfied())
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(std::numeric_limits<scalar_type>::max_digits10)
                << "Reacting transport inventory closure failed after residual correction: before="
                << balance.before << ", after=" << balance.after << ", outflow=" << balance.outflow
                << ", residual=" << balance.residual << ", roundoff=" << balance.tolerance << '.';
        throw std::runtime_error(message.str());
    }

    {
        const auto solution_values = solution.owned_read_view();
        const auto storage = storage_weight.owned_read_view();
        const auto field_values = field.owned_write_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            auto transported = solution_values(owned, 0);
            if (liquid_weighted)
                transported *= storage(owned, 0);
            if (!std::isfinite(transported) || transported < 0.0)
            {
                throw std::runtime_error("Reacting transport cannot publish a negative or nonfinite inventory.");
            }
            field_values(owned, 0) = transported;
        }
    }
    field.sync_ghosts();

    if (canonical_molar_packet)
    {
        if constexpr (requires { solution.sync_ghosts(); })
            solution.sync_ghosts();
        collective_detail::collective_local_validation(*d_mesh, "Canonical transported molar face packet",
            [&]
            {
                if (&canonical_molar_packet->mesh() != d_mesh.get())
                    throw std::invalid_argument("Canonical transport packet has a foreign mesh.");
                for (const auto face : canonical_molar_packet->owned_face_ids())
                {
                    const auto flux = transport_flux.value(face); // identical boundary gating and orientation
                    const auto owner = d_mesh->owner_cell(face);
                    const auto donor = flux >= scalar_type{} || d_mesh->is_boundary_face(face)
                                           ? owner
                                           : d_mesh->opposite_or_periodic_neighbor_cell(face, owner);
                    const auto primary = solution.local_value(donor);
                    const auto weight = storage_weight.local_value(donor);
                    // Same left-associated coefficient/product as the scalar boundary
                    // ledger and upwind operator; do not reconstruct from rounded alpha*C.
                    const auto stored = flux * weight * primary;
                    if (!std::isfinite(stored) || (flux != 0 && weight != 0 && primary != 0 && stored == 0) ||
                        primary < 0 || weight < 0)
                        throw std::runtime_error("Canonical transported molar packet is invalid.");
                    canonical_molar_packet->owned_write_view()(canonical_molar_packet->owned_row(face), 0) = stored;
                }
            });
        if constexpr (requires { canonical_molar_packet->sync_ghosts(); })
            canonical_molar_packet->sync_ghosts();
    }

    const auto flux_values = transport_flux.owned_read_view();
    const auto field_values = field.owned_read_view();
    const auto storage_values = storage_weight.owned_read_view();
    const auto advection_values = storage_weight.local_read_view();
    const auto escape_values = escape_rate.owned_write_view();
    for (const auto face_lid : transport_flux.owned_face_ids())
    {
        if (!is_free_surface(face_lid))
        {
            continue;
        }
        const auto flux =
            std::max(flux_values(transport_flux.owned_row(face_lid), 0), scalar_type{});
        const auto owner = d_mesh->owner_cell(face_lid);
        const auto primary_value = liquid_weighted
            ? field_values(owner, 0) / storage_values(owner, 0)
            : field_values(owner, 0);
        const auto boundary_rate =
            flux
          * advection_values(owner, 0)
          * primary_value;
        const auto volume = ale == nullptr
                                ? static_cast<scalar_type>(d_mesh->cell_volume(owner))
                                : static_cast<scalar_type>(
                                      ale->new_cell_volumes()[static_cast<size_t>(owner)]);
        escape_values(owner, 0) += boundary_rate / volume;
    }
    work.total_seconds += elapsed(started);
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::advance(scalar_type time, scalar_type time_step, const field_type& temperature,
    const field_type& gauge_pressure, const velocity_field_type& velocity, const face_flux_field_type& liquid_face_flux,
    const material_type& material, const FVM::ALEControlVolumeState* ale, Dimension slip_axis)
{
    if (!enabled())
    {
        d_source_alpha_reaction.put_scalar(0);
        d_species_thermal_source.put_scalar(0);
        return;
    }
    advance_species_fields(time, time_step, temperature, gauge_pressure, velocity, liquid_face_flux, material, ale,
        slip_axis, d_active_bouss_wall_heat);
}

template<TpetraTypePack Pack, class MeshType> void ReactingGasModel<Pack, MeshType>::sync_all_fields()
{
    const auto fields = mutable_state_fields();
    if (d_state_sync_owned.is_null() || d_state_sync_owned->getNumVectors() != fields.size())
    {
        d_state_sync_owned = Teuchos::rcp(new typename Pack::multi_vector_type(
            fields.front()->owned_data().getMap(), fields.size(), false));
        d_state_sync_overlap = Teuchos::rcp(new typename Pack::multi_vector_type(
            fields.front()->overlap_data().getMap(), fields.size(), false));
        d_state_sync_import = Teuchos::rcp(new typename Pack::import_type(
            d_state_sync_owned->getMap(), d_state_sync_overlap->getMap()));
    }
    {
        const auto packed = d_state_sync_owned->getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t column = 0; column < fields.size(); ++column)
        {
            const auto values = fields[column]->owned_read_view();
            for (size_t row = 0; row < values.extent(0); ++row)
                packed(row, column) = values(row, 0);
        }
    }
    d_state_sync_overlap->doImport(*d_state_sync_owned, *d_state_sync_import, Tpetra::REPLACE);
    {
        const auto packed = d_state_sync_overlap->getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t column = 0; column < fields.size(); ++column)
        {
            const auto values = fields[column]->overlap_data().getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (size_t row = 0; row < values.extent(0); ++row)
                values(row, 0) = packed(row, column);
        }
    }
}

template<TpetraTypePack Pack, class MeshType>
std::vector<typename ReactingGasModel<Pack, MeshType>::field_type*>
ReactingGasModel<Pack, MeshType>::mutable_state_fields()
{
    std::vector<field_type*> result{&d_alpha_g, &d_alpha_l, &d_source_alpha_reaction, &d_absolute_pressure,
        &d_previous_temperature, &d_previous_alpha_g, &d_micro_number, &d_micro_moles, &d_large_number, &d_large_moles,
        &d_micro_radius, &d_large_radius, &d_alpha_g_micro, &d_alpha_g_large, &d_alpha_g_raw, &d_transport_alpha_g_raw,
        &d_alpha_g_excess, &d_escape_number_rate};
    append_species_fields(result);
    return result;
}

template<TpetraTypePack Pack, class MeshType>
std::vector<const typename ReactingGasModel<Pack, MeshType>::field_type*>
ReactingGasModel<Pack, MeshType>::state_fields() const
{
    std::vector<const field_type*> result{&d_alpha_g, &d_alpha_l, &d_source_alpha_reaction, &d_absolute_pressure,
        &d_previous_temperature, &d_previous_alpha_g, &d_micro_number, &d_micro_moles, &d_large_number, &d_large_moles,
        &d_micro_radius, &d_large_radius, &d_alpha_g_micro, &d_alpha_g_large, &d_alpha_g_raw, &d_transport_alpha_g_raw,
        &d_alpha_g_excess, &d_escape_number_rate};
    append_species_fields(result);
    return result;
}

template<TpetraTypePack Pack, class MeshType>
auto ReactingGasModel<Pack, MeshType>::snapshot() const -> StateSnapshot
{
    StateSnapshot result;
    result.d_owner = this;
    result.d_species_generation = d_species_generation;
    result.d_source_configuration_generation = d_source_configuration_generation;
    result.d_species_fingerprint = species_enabled() ? d_options.reaction_mechanism->catalog().fingerprint() : 0;
    if constexpr (requires { d_mesh->geometry_epoch(); })
        result.d_geometry_epoch = d_mesh->geometry_epoch();
    result.d_species_sources = d_species_source_ledgers;
    result.d_species_populations = d_species_population_ledgers;
    result.d_species_escapes = d_species_escape_ledgers;
    result.d_species_history = d_species_history;
    result.d_bouss_receipts = d_bouss_receipts;
    result.d_species_event_receipt = d_species_event_receipt;
    result.d_population_escape_receipt = d_population_escape_receipt;
    for (const auto* field : state_fields())
    {
        auto& values = result.d_fields.emplace_back();
        values.resize(d_mesh->num_owned_cells());
        const auto field_values = field->owned_read_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            values[owned] = field_values(owned, 0);
        }
    }
    result.d_transport_slip_face_ids =
        d_transport_bubble_slip_volume_flux.owned_face_ids();
    result.d_transport_slip_face_values.reserve(
        result.d_transport_slip_face_ids.size());
    result.d_transport_carrier_face_values.reserve(
        result.d_transport_slip_face_ids.size());
    result.d_transport_liquid_face_values.reserve(result.d_transport_slip_face_ids.size());
    for (const auto face_lid : result.d_transport_slip_face_ids)
    {
        result.d_transport_slip_face_values.push_back(
            d_transport_bubble_slip_volume_flux.value(face_lid));
        result.d_transport_carrier_face_values.push_back(
            d_transport_bubble_carrier_volume_flux.value(face_lid));
        result.d_transport_liquid_face_values.push_back(d_transport_liquid_material_volume_flux.value(face_lid));
    }
    result.d_liquid_transport_volume_receipt = d_liquid_transport_volume_receipt;
    result.d_statistics = d_last_statistics;
    result.d_history_initialized = d_history_initialized;
    result.d_initial_state_initialized = d_initial_state_initialized;
    result.d_absolute_pressure_offset = d_absolute_pressure_offset;
    return result;
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::restore(const StateSnapshot& snapshot)
{
    restore_state(snapshot, true);
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::restore_for_ale_replay(const StateSnapshot& snapshot)
{
    restore_state(snapshot, false);
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::restore_for_initialization_rollback(const StateSnapshot& snapshot)
{
    restore_state(snapshot, true, true);
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::restore_state(
    const StateSnapshot& snapshot, bool refresh_metrics, bool restore_initialization_authority)
{
    const auto fields = mutable_state_fields();
    // Public/ALE snapshots bind initialization authority. Only private failed-init/step rollback may restore it.
    int local_invalid = snapshot.d_owner != this ||
                        snapshot.d_source_configuration_generation != d_source_configuration_generation ||
                        (!restore_initialization_authority && snapshot.d_species_generation != d_species_generation) ||
                        snapshot.d_species_generation > d_species_generation_allocator ||
                        snapshot.d_species_fingerprint !=
                            (species_enabled() ? d_options.reaction_mechanism->catalog().fingerprint() : 0) ||
                        snapshot.d_fields.size() != fields.size() ||
                        snapshot.d_transport_slip_face_ids != d_transport_bubble_slip_volume_flux.owned_face_ids() ||
                        snapshot.d_transport_slip_face_values.size() != snapshot.d_transport_slip_face_ids.size() ||
                        snapshot.d_transport_carrier_face_values.size() != snapshot.d_transport_slip_face_ids.size() ||
                        snapshot.d_transport_liquid_face_values.size() != snapshot.d_transport_slip_face_ids.size();
    if (!local_invalid)
    {
        for (const auto& values : snapshot.d_fields)
        {
            local_invalid = local_invalid || values.size() != d_mesh->num_owned_cells();
        }
    }
    int any_invalid = 0;
    Teuchos::reduceAll(
        *d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
    {
        throw std::invalid_argument("ReactingGasModel snapshot is foreign or has incompatible storage: owner=" +
            std::to_string(snapshot.d_owner == this) + ", configuration=" +
            std::to_string(snapshot.d_source_configuration_generation) + "/" +
            std::to_string(d_source_configuration_generation) + ", initialization=" +
            std::to_string(snapshot.d_species_generation) + "/" + std::to_string(d_species_generation) +
            ", fields=" + std::to_string(snapshot.d_fields.size()) + "/" + std::to_string(fields.size()) +
            ", face_layout=" +
            std::to_string(snapshot.d_transport_slip_face_ids == d_transport_bubble_slip_volume_flux.owned_face_ids()) +
            ".");
    }
    for (size_t field_index = 0; field_index < fields.size(); ++field_index)
    {
        const auto values = fields[field_index]->owned_write_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            values(owned, 0) = snapshot.d_fields[field_index][owned];
        }
    }
    for (size_t face = 0; face < snapshot.d_transport_slip_face_ids.size(); ++face)
    {
        if constexpr (requires {
                          d_transport_bubble_slip_volume_flux.set_owned_value(
                              snapshot.d_transport_slip_face_ids[face],
                              snapshot.d_transport_slip_face_values[face]);
                      })
        {
            d_transport_bubble_slip_volume_flux.set_owned_value(
                snapshot.d_transport_slip_face_ids[face],
                snapshot.d_transport_slip_face_values[face]);
            d_transport_bubble_carrier_volume_flux.set_owned_value(
                snapshot.d_transport_slip_face_ids[face],
                snapshot.d_transport_carrier_face_values[face]);
            d_transport_liquid_material_volume_flux.set_owned_value(
                snapshot.d_transport_slip_face_ids[face], snapshot.d_transport_liquid_face_values[face]);
        }
        else
        {
            d_transport_bubble_slip_volume_flux.set_value(
                snapshot.d_transport_slip_face_ids[face],
                snapshot.d_transport_slip_face_values[face]);
            d_transport_bubble_carrier_volume_flux.set_value(
                snapshot.d_transport_slip_face_ids[face],
                snapshot.d_transport_carrier_face_values[face]);
            d_transport_liquid_material_volume_flux.set_value(
                snapshot.d_transport_slip_face_ids[face], snapshot.d_transport_liquid_face_values[face]);
        }
    }
    d_liquid_transport_volume_receipt = snapshot.d_liquid_transport_volume_receipt;
    d_last_statistics = snapshot.d_statistics;
    d_species_source_ledgers = snapshot.d_species_sources;
    d_species_population_ledgers = snapshot.d_species_populations;
    d_species_escape_ledgers = snapshot.d_species_escapes;
    d_species_history = snapshot.d_species_history;
    d_bouss_receipts = snapshot.d_bouss_receipts;
    d_species_event_receipt = snapshot.d_species_event_receipt;
    d_population_escape_receipt = snapshot.d_population_escape_receipt;
    d_history_initialized = snapshot.d_history_initialized;
    d_initial_state_initialized = snapshot.d_initial_state_initialized;
    if (restore_initialization_authority)
        d_species_generation = snapshot.d_species_generation; // The allocator deliberately never rewinds.
    d_absolute_pressure_offset = snapshot.d_absolute_pressure_offset;
    sync_all_fields();
    if constexpr (requires { d_transport_bubble_slip_volume_flux.sync_ghosts(); })
    {
        d_transport_bubble_slip_volume_flux.sync_ghosts();
        d_transport_bubble_carrier_volume_flux.sync_ghosts();
        d_transport_liquid_material_volume_flux.sync_ghosts();
    }
    if (refresh_metrics) refresh_geometry();
    else invalidate_transport_operator();
}

template<TpetraTypePack Pack, class MeshType> void ReactingGasModel<Pack, MeshType>::refresh_geometry()
{
    refresh_geometry({});
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::refresh_geometry(
    typename FVM::TransportGeometryCache<mesh_type>::shared_geometry_type geometry)
{
    d_transport_geometry_cache.refresh(std::move(geometry));
    invalidate_transport_operator();
}

template<TpetraTypePack Pack, class MeshType>
void ReactingGasModel<Pack, MeshType>::invalidate_transport_operator()
{
    d_transport_solver.notify_operator_values_changed();
    if (d_transport_workspace)
    {
        // Fixed-topology ALE keeps maps and graph storage valid. Assembly
        // validates the required columns and overwrites all numerical values.
        // Number/moles may share an operator only within the next stage.
        d_transport_workspace->operator_ready.fill(false);
    }
}

} // namespace SimpleFluid

#include "equations/ReactingGasSpecies.ipp"
