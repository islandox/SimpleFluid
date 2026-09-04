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
template<TpetraTypePack Pack, class MeshType>
PressureProjectionEquation<Pack, MeshType>::PressureProjectionEquation(SP<const mesh_type> mesh,
    LinearSolverOptions linear_options, BoundaryConditionMap pressure_boundary_conditions,
    FVM::CellGradientScheme gradient_scheme)
    : d_mesh(EquationValidation::require_non_null_mesh(std::move(mesh), "PressureProjectionEquation")),
      d_linear_options(linear_options), d_pressure_boundary_conditions(std::move(pressure_boundary_conditions)),
      d_pressure_correction_boundary_conditions(d_pressure_boundary_conditions), d_gradient_scheme(gradient_scheme),
      d_cached_face_fluxes(d_mesh, "pressure_projection_face_flux"), d_face_flux_workspace(d_mesh)
{
    require_owned_cell_map(d_mesh);
    for (auto& [name, condition] : d_pressure_correction_boundary_conditions)
    {
        static_cast<void>(name);
        condition.value = 0.0;
    }
}

/** @brief Replace the pressure linear-solver policy. */
template<TpetraTypePack Pack, class MeshType>
void PressureProjectionEquation<Pack, MeshType>::set_linear_solver_options(LinearSolverOptions options)
{
    d_linear_options = std::move(options);
}

/** @brief Return the pressure linear-solver policy. */
template<TpetraTypePack Pack, class MeshType>
const LinearSolverOptions& PressureProjectionEquation<Pack, MeshType>::linear_solver_options() const noexcept
{
    return d_linear_options;
}

/** @brief Refresh geometry-dependent state after a fixed-topology motion. */
template<TpetraTypePack Pack, class MeshType> void PressureProjectionEquation<Pack, MeshType>::refresh_geometry()
{
    d_cached_pressure_matrix = Teuchos::null;
    d_cached_rhs = Teuchos::null;
    d_pressure_gauge_gid.reset();
    d_cached_predictor_flux_valid = false;
    d_rhs_norm_reference = {};
    d_cached_target_generation = 0;
    d_cached_target_geometry_epoch = 0;
    d_cached_predictor_time_step = {};
    d_cached_predictor_reference_density = {};
    d_cached_fixed_boundary_flux_generation = 0;
    d_linear_solver.reset();
    d_face_flux_workspace.refresh_geometry();
}

/** @brief Configure exact fluxes on selected physical-pressure boundaries. */
template<TpetraTypePack Pack, class MeshType>
void PressureProjectionEquation<Pack, MeshType>::set_fixed_boundary_flux_provider(
    std::vector<std::string> boundary_names, fixed_boundary_flux_provider_type provider, std::uint64_t generation)
{
    if (!provider || boundary_names.empty())
    {
        throw std::invalid_argument("PressureProjectionEquation fixed boundary fluxes require a "
                                    "provider and at least one boundary name.");
    }
    std::sort(boundary_names.begin(), boundary_names.end());
    if (std::adjacent_find(boundary_names.begin(), boundary_names.end()) != boundary_names.end())
    {
        throw std::invalid_argument("PressureProjectionEquation fixed boundary flux names must be unique.");
    }
    for (const auto& name : boundary_names)
    {
        const auto condition = d_pressure_boundary_conditions.find(name);
        if (condition == d_pressure_boundary_conditions.end() ||
            condition->second.type != BoundaryConditionType::Dirichlet)
        {
            throw std::invalid_argument("PressureProjectionEquation fixed-flux boundary '" + name +
                                        "' requires a physical Dirichlet pressure condition.");
        }
    }

    d_fixed_boundary_flux_names = std::move(boundary_names);
    d_fixed_boundary_flux_provider = std::move(provider);
    d_fixed_boundary_flux_generation = generation;
    d_pressure_correction_boundary_conditions = d_pressure_boundary_conditions;
    for (auto& [name, condition] : d_pressure_correction_boundary_conditions)
    {
        condition.value = scalar_type{};
        if (std::binary_search(d_fixed_boundary_flux_names.begin(), d_fixed_boundary_flux_names.end(), name))
        {
            condition.type = BoundaryConditionType::Neumann;
        }
    }
    d_cached_pressure_matrix = Teuchos::null;
    d_cached_rhs = Teuchos::null;
    d_pressure_gauge_gid.reset();
    d_cached_predictor_flux_valid = false;
    d_rhs_norm_reference = {};
    d_linear_solver.reset();
}

/** @brief Clear exact boundary fluxes and restore physical correction types. */
template<TpetraTypePack Pack, class MeshType>
void PressureProjectionEquation<Pack, MeshType>::clear_fixed_boundary_flux_provider()
{
    d_fixed_boundary_flux_names.clear();
    d_fixed_boundary_flux_provider = {};
    d_fixed_boundary_flux_generation = 0;
    d_pressure_correction_boundary_conditions = d_pressure_boundary_conditions;
    for (auto& [name, condition] : d_pressure_correction_boundary_conditions)
    {
        static_cast<void>(name);
        condition.value = scalar_type{};
    }
    d_cached_pressure_matrix = Teuchos::null;
    d_cached_rhs = Teuchos::null;
    d_pressure_gauge_gid.reset();
    d_cached_predictor_flux_valid = false;
    d_rhs_norm_reference = {};
    d_linear_solver.reset();
}

/** @brief Relax only the compatibility check for exact fixed-flux faces. */
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::pressure_flux_boundary_cache(
    const velocity_boundary_cache_type& boundary_cache) const -> velocity_boundary_cache_type
{
    auto result = boundary_cache;
    for (const auto& name : d_fixed_boundary_flux_names)
    {
        result.type_by_name[name] = BoundaryConditionType::Neumann;
    }
    return result;
}

/** @brief Validate fixed-flux configuration collectively before evaluation. */
template<TpetraTypePack Pack, class MeshType>
void PressureProjectionEquation<Pack, MeshType>::validate_fixed_boundary_flux_provider() const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const int local_count = static_cast<int>(d_fixed_boundary_flux_names.size());
    int minimum_count = 0;
    int maximum_count = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_count, &minimum_count);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_count, &maximum_count);

    auto hash_names = [](const std::vector<std::string>& names)
    {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const auto& name : names)
        {
            for (const unsigned char character : name)
            {
                hash ^= character;
                hash *= 1099511628211ULL;
            }
            hash ^= 0xffU;
            hash *= 1099511628211ULL;
        }
        return hash;
    };
    const auto local_hash = static_cast<unsigned long long>(hash_names(d_fixed_boundary_flux_names));
    const auto local_generation = static_cast<unsigned long long>(d_fixed_boundary_flux_generation);
    unsigned long long minimum_hash = 0;
    unsigned long long maximum_hash = 0;
    unsigned long long minimum_generation = 0;
    unsigned long long maximum_generation = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_hash, &minimum_hash);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_hash, &maximum_hash);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MIN, 1, &local_generation, &minimum_generation);
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_generation, &maximum_generation);

    const int local_provider_state =
        static_cast<bool>(d_fixed_boundary_flux_provider) == !d_fixed_boundary_flux_names.empty() ? 0 : 1;
    int any_provider_error = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_provider_state, &any_provider_error);
    if (minimum_count != maximum_count || minimum_hash != maximum_hash || minimum_generation != maximum_generation ||
        any_provider_error != 0)
    {
        throw std::invalid_argument("PressureProjectionEquation fixed boundary flux names, provider "
                                    "state, and generation must match on every mesh rank.");
    }

    for (const auto& name : d_fixed_boundary_flux_names)
    {
        int local_faces = 0;
        int local_condition_error = 0;
        const auto pressure_condition = d_pressure_boundary_conditions.find(name);
        const auto correction_condition = d_pressure_correction_boundary_conditions.find(name);
        local_condition_error = pressure_condition == d_pressure_boundary_conditions.end() ||
                                pressure_condition->second.type != BoundaryConditionType::Dirichlet ||
                                correction_condition == d_pressure_correction_boundary_conditions.end() ||
                                correction_condition->second.type != BoundaryConditionType::Neumann;
        for (const auto& [batch_id, batch] : d_mesh->boundary_batches())
        {
            if (d_mesh->boundary_batch_name(batch_id) != name)
            {
                continue;
            }
            for (const auto face_lid : batch.face_lids)
            {
                local_faces += d_mesh->is_owned_face(face_lid) ? 1 : 0;
            }
        }
        int global_faces = 0;
        int any_condition_error = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_condition_error, &any_condition_error);
        if (global_faces == 0 || any_condition_error != 0)
        {
            throw std::invalid_argument("PressureProjectionEquation fixed-flux boundary '" + name +
                                        "' must exist globally with Dirichlet physical pressure "
                                        "and Neumann pressure correction.");
        }
    }
}

/** @brief Overwrite selected owned boundary faces after flux reconstruction. */
template<TpetraTypePack Pack, class MeshType>
void PressureProjectionEquation<Pack, MeshType>::apply_fixed_boundary_fluxes(face_flux_field_type& fluxes) const
{
    if (d_fixed_boundary_flux_names.empty())
    {
        return;
    }
    struct Value
    {
        local_ordinal_type face_lid;
        scalar_type flux;
    };
    std::vector<Value> values;
    std::exception_ptr local_error;
    try
    {
        for (const auto& [batch_id, batch] : d_mesh->boundary_batches())
        {
            const auto& name = d_mesh->boundary_batch_name(batch_id);
            if (!std::binary_search(d_fixed_boundary_flux_names.begin(), d_fixed_boundary_flux_names.end(), name))
            {
                continue;
            }
            for (size_t in_batch = 0; in_batch < batch.face_lids.size(); ++in_batch)
            {
                const auto face_lid = batch.face_lids[in_batch];
                if (!d_mesh->is_owned_face(face_lid))
                {
                    continue;
                }
                const auto value = d_fixed_boundary_flux_provider(batch_id, in_batch, face_lid);
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument("Fixed boundary volume flux must be finite [m^3/s].");
                }
                values.push_back({face_lid, value});
            }
        }
    }
    catch (...)
    {
        local_error = std::current_exception();
    }
    const int local_failed = local_error ? 1 : 0;
    int any_failed = 0;
    Teuchos::reduceAll(*d_mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_failed, &any_failed);
    if (any_failed != 0)
    {
        if (local_error)
        {
            std::rethrow_exception(local_error);
        }
        throw std::runtime_error("Fixed boundary volume-flux provider failed on another mesh rank.");
    }
    for (const auto& value : values)
    {
        fluxes.set_value(value.face_lid, value.flux);
    }
    if constexpr (requires { fluxes.sync_ghosts(); })
    {
        fluxes.sync_ghosts();
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
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::require_owned_cell_map(const SP<const mesh_type>& mesh)
    -> Teuchos::RCP<const map_type>
{
    auto map = mesh->owned_cell_map();
    if (map == Teuchos::null)
    {
        throw std::runtime_error("PressureProjectionEquation requires an assembled mesh with an owned-cell map.");
    }
    if (map->getGlobalNumElements() == 0)
    {
        throw std::runtime_error("PressureProjectionEquation requires at least one global cell.");
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
template<TpetraTypePack Pack, class MeshType> void PressureProjectionEquation<Pack, MeshType>::rebuild_matrix() const
{
    const auto owned_map = require_owned_cell_map(d_mesh);
    int local_has_dirichlet = 0;
    for (const auto& [batch_id, boundary_batch] : d_mesh->boundary_batches())
    {
        const auto condition_iter =
            d_pressure_correction_boundary_conditions.find(d_mesh->boundary_batch_name(batch_id));
        if (condition_iter == d_pressure_correction_boundary_conditions.end() ||
            condition_iter->second.type != BoundaryConditionType::Dirichlet)
        {
            continue;
        }
        for (const auto face_lid : boundary_batch.face_lids)
        {
            if (d_mesh->is_owned_face(face_lid) && d_mesh->is_boundary_face(face_lid))
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
    Teuchos::reduceAll(*owned_map->getComm(), Teuchos::REDUCE_MAX, 1, &local_has_dirichlet, &global_has_dirichlet);
    d_pressure_gauge_gid = global_has_dirichlet != 0
                               ? std::optional<typename Pack::global_ordinal_type>{}
                               : std::optional<typename Pack::global_ordinal_type>{owned_map->getMinAllGlobalIndex()};
    auto boundary_condition = [&](int batch_id, size_t)
    {
        const auto iter = d_pressure_correction_boundary_conditions.find(d_mesh->boundary_batch_name(batch_id));
        return iter == d_pressure_correction_boundary_conditions.end() ? BoundaryCondition{} : iter->second;
    };
    d_cached_pressure_matrix = FVM::pressure_poisson_matrix<Pack>(*d_mesh, d_pressure_gauge_gid, boundary_condition);
}

/**
 * @brief Solve for the pressure field (initialise with zero and sync
 *        periodic boundaries).
 *
 * @tparam Pack Tpetra type pack.
 * @param[out] pressure Pressure field to initialise.
 * @throws std::invalid_argument if the pressure field mesh does not match.
 */
template<TpetraTypePack Pack, class MeshType>
void PressureProjectionEquation<Pack, MeshType>::solve(field_type& pressure)
{
    EquationValidation::require_mesh_match(*d_mesh, pressure, "PressureProjectionEquation");

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
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::project(field_type& pressure, scalar_type time_step,
    scalar_type reference_density, const velocity_boundary_cache_type& velocity_boundary_cache,
    velocity_field_type& velocity) -> ProjectionResult
{
    const continuity_target_type zero_target(d_mesh);
    return project_impl(
        pressure, time_step, reference_density, velocity_boundary_cache, velocity, zero_target, nullptr, false, false);
}

/** @brief Project against an integrated per-cell volume target. */
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::project(field_type& pressure, scalar_type time_step,
    scalar_type reference_density, const velocity_boundary_cache_type& velocity_boundary_cache,
    velocity_field_type& velocity, const continuity_target_type& continuity_target) -> ProjectionResult
{
    return project_impl(pressure, time_step, reference_density, velocity_boundary_cache, velocity, continuity_target,
        nullptr, false, true);
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
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::project(field_type& pressure, field_type& pressure_correction,
    scalar_type time_step, scalar_type reference_density, const velocity_boundary_cache_type& velocity_boundary_cache,
    velocity_field_type& velocity) -> ProjectionResult
{
    const continuity_target_type zero_target(d_mesh);
    return project_impl(pressure_correction, time_step, reference_density, velocity_boundary_cache, velocity,
        zero_target, &pressure, false, false);
}

/** @brief Accumulated-pressure projection to an integrated volume target. */
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::project(field_type& pressure, field_type& pressure_correction,
    scalar_type time_step, scalar_type reference_density, const velocity_boundary_cache_type& velocity_boundary_cache,
    velocity_field_type& velocity, const continuity_target_type& continuity_target) -> ProjectionResult
{
    return project_impl(pressure_correction, time_step, reference_density, velocity_boundary_cache, velocity,
        continuity_target, &pressure, false, true);
}

/**
 * @brief Reuse the preceding correction's final flux as the next PISO
 *        predictor.
 *
 * Consecutive pressure correctors do not change pressure or velocity between
 * calls. The preceding final Rhie--Chow reconstruction is therefore exactly
 * the next predictor flux, including its cached pressure gradient.
 */
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::project_reusing_cached_predictor(field_type& pressure,
    field_type& pressure_correction, scalar_type time_step, scalar_type reference_density,
    const velocity_boundary_cache_type& velocity_boundary_cache, velocity_field_type& velocity) -> ProjectionResult
{
    const continuity_target_type zero_target(d_mesh);
    return project_impl(pressure_correction, time_step, reference_density, velocity_boundary_cache, velocity,
        zero_target, &pressure, true, false);
}

/** @brief Reuse a predictor for the same target generation and geometry. */
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::project_reusing_cached_predictor(field_type& pressure,
    field_type& pressure_correction, scalar_type time_step, scalar_type reference_density,
    const velocity_boundary_cache_type& velocity_boundary_cache, velocity_field_type& velocity,
    const continuity_target_type& continuity_target) -> ProjectionResult
{
    return project_impl(pressure_correction, time_step, reference_density, velocity_boundary_cache, velocity,
        continuity_target, &pressure, true, true);
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
 * @param right_hand_source Legacy per-cell Poisson source provider.
 * @throws std::invalid_argument on mesh mismatch or non-positive time
 *         step.
 */
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::project(field_type& pressure, scalar_type time_step,
    scalar_type reference_density, const velocity_boundary_cache_type& velocity_boundary_cache,
    velocity_field_type& velocity, const source_type& right_hand_source) -> ProjectionResult
{
    if (!std::isfinite(time_step) || time_step <= scalar_type{})
    {
        throw std::invalid_argument("PressureProjectionEquation requires a finite positive time step.");
    }
    // Legacy source semantics assembled V*S beside -div(phi)/dt. Preserve
    // that algebra exactly by converting to Q_V = dt*V*S [m^3/s].
    auto continuity_target = continuity_target_type::from_integrated_rate_provider(
        d_mesh, [&](local_ordinal_type cell_lid)
        { return time_step * static_cast<scalar_type>(d_mesh->cell_volume(cell_lid)) * right_hand_source(cell_lid); },
        0, "Pressure projection compatibility source");
    return project_impl(pressure, time_step, reference_density, velocity_boundary_cache, velocity, continuity_target,
        nullptr, false, false);
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
 * @param continuity_target Integrated per-cell volume target [m^3/s].
 * @param[in,out] accumulated_pressure Optional physical pressure in Pa.
 * @param reuse_cached_predictor_flux Whether to reuse the final flux from the
 *        immediately preceding accumulated-pressure projection.
 * @return Projection statistics.
 */
template<TpetraTypePack Pack, class MeshType>
auto PressureProjectionEquation<Pack, MeshType>::project_impl(field_type& pressure_correction, scalar_type time_step,
    scalar_type reference_density, const velocity_boundary_cache_type& velocity_boundary_cache,
    velocity_field_type& velocity, const continuity_target_type& continuity_target, field_type* accumulated_pressure,
    bool reuse_cached_predictor_flux, bool enforce_global_compatibility) -> ProjectionResult
{
    EquationValidation::require_mesh_match(*d_mesh, pressure_correction, "PressureProjectionEquation");
    EquationValidation::require_mesh_match(*d_mesh, velocity, "PressureProjectionEquation");
    if (accumulated_pressure != nullptr)
    {
        EquationValidation::require_mesh_match(*d_mesh, *accumulated_pressure, "PressureProjectionEquation");
        if (accumulated_pressure == &pressure_correction)
        {
            throw std::invalid_argument("PressureProjectionEquation requires distinct accumulated "
                                        "and correction pressure fields.");
        }
    }
    if (!std::isfinite(time_step) || time_step <= 0.0)
    {
        throw std::invalid_argument("PressureProjectionEquation requires a positive time step.");
    }
    if (!std::isfinite(reference_density) || reference_density <= scalar_type{})
    {
        throw std::invalid_argument("PressureProjectionEquation requires a finite positive "
                                    "reference density.");
    }
    continuity_target.validate(*d_mesh, "PressureProjectionEquation continuity target");
    validate_fixed_boundary_flux_provider();
    auto pressure_flux_boundary_cache = this->pressure_flux_boundary_cache(velocity_boundary_cache);
    const auto current_geometry_epoch = mesh_geometry_epoch(*d_mesh);
    if (reuse_cached_predictor_flux &&
        (accumulated_pressure == nullptr || !d_cached_predictor_flux_valid ||
            d_cached_target_generation != continuity_target.generation() ||
            d_cached_target_geometry_epoch != continuity_target.geometry_epoch() ||
            d_cached_target_geometry_epoch != current_geometry_epoch || d_cached_predictor_time_step != time_step ||
            d_cached_predictor_reference_density != reference_density ||
            d_cached_fixed_boundary_flux_generation != d_fixed_boundary_flux_generation))
    {
        d_cached_predictor_flux_valid = false;
        throw std::logic_error("PressureProjectionEquation cannot reuse a predictor flux "
                               "unless the adjacent accumulated-pressure projection has the "
                               "same target generation, geometry epoch, timestep, reference "
                               "density, and fixed-boundary-flux generation.");
    }

    // Any failure after this point invalidates reuse. A successful
    // accumulated-pressure projection publishes a new final flux below.
    d_cached_predictor_flux_valid = false;
    pressure_correction.owned_data().putScalar(0.0);
    d_mesh->sync_periodic_boundaries(pressure_correction);
    if (!reuse_cached_predictor_flux && accumulated_pressure != nullptr)
    {
        FVM::pressure_weighted_face_fluxes(velocity, *accumulated_pressure, time_step / reference_density,
            pressure_flux_boundary_cache, d_pressure_boundary_conditions, d_face_flux_workspace, d_cached_face_fluxes,
            d_gradient_scheme);
        apply_fixed_boundary_fluxes(d_cached_face_fluxes);
    }
    else if (!reuse_cached_predictor_flux)
    {
        FVM::pressure_weighted_face_fluxes(velocity, pressure_correction, time_step, pressure_flux_boundary_cache,
            d_pressure_correction_boundary_conditions, d_face_flux_workspace, d_cached_face_fluxes, d_gradient_scheme);
        apply_fixed_boundary_fluxes(d_cached_face_fluxes);
    }
    const auto owned_map = require_owned_cell_map(d_mesh);
    if (d_cached_pressure_matrix.is_null())
    {
        rebuild_matrix();
    }
    if (d_cached_rhs.is_null())
    {
        d_cached_rhs = Teuchos::rcp(new typename Pack::vector_type(d_mesh->owned_cell_map(), true));
    }
    else
    {
        d_cached_rhs->putScalar(0.0);
    }

    {
        const auto predictor_flux_values = d_cached_face_fluxes.owned_read_view();
        scalar_type local_compatibility_residual{};
        scalar_type local_compatibility_scale{};
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto balance =
                FVM::cell_flux_balance<Pack>(*d_mesh, d_cached_face_fluxes, predictor_flux_values, cell_lid);
            const auto target = continuity_target.integrated_rate(cell_lid);
            local_compatibility_residual += balance - target;
            local_compatibility_scale += std::abs(balance) + std::abs(target);
        }
        const std::array<scalar_type, 2> local_values{local_compatibility_residual, local_compatibility_scale};
        std::array<scalar_type, 2> global_values{};
        Teuchos::reduceAll(*owned_map->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(local_values.size()),
            local_values.data(), global_values.data());
        const auto tolerance = scalar_type{1.0e-12} + scalar_type{1.0e-10} * global_values[1];
        if (enforce_global_compatibility && d_pressure_gauge_gid && std::abs(global_values[0]) > tolerance)
        {
            d_cached_predictor_flux_valid = false;
            throw std::invalid_argument("PressureProjectionEquation all-Neumann/fixed-flux "
                                        "boundaries are globally incompatible with the integrated "
                                        "continuity target.");
        }
    }

    {
        const auto predictor_flux_values = d_cached_face_fluxes.owned_read_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto row_gid = owned_map->getGlobalElement(cell_lid);
            const auto is_pressure_gauge = d_pressure_gauge_gid && row_gid == *d_pressure_gauge_gid;
            const auto rhs_value =
                is_pressure_gauge
                    ? scalar_type{}
                    : (-FVM::cell_flux_balance<Pack>(*d_mesh, d_cached_face_fluxes, predictor_flux_values, cell_lid) +
                          continuity_target.integrated_rate(cell_lid)) /
                          time_step;
            d_cached_rhs->replaceLocalValue(cell_lid, rhs_value);
        }
    }

    Teuchos::RCP<const typename Pack::matrix_type> const_matrix = d_cached_pressure_matrix;
    const LinearResidualScaling residual_scaling{reuse_cached_predictor_flux ? d_rhs_norm_reference : real_t{}};
    const auto linear_statistics = d_linear_solver.solve_with_statistics(
        const_matrix, *d_cached_rhs, pressure_correction.owned_data(), d_linear_options, residual_scaling);
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
        const auto correction_values = pressure_correction.owned_read_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto correction = correction_values(cell_lid, 0);
            pressure_norm_squared += correction * correction * static_cast<scalar_type>(d_mesh->cell_volume(cell_lid));
        }
    }

    FVM::cell_gradient(pressure_correction, d_pressure_correction_boundary_conditions,
        d_face_flux_workspace.pressure_gradient(), d_face_flux_workspace.gradient_cache(), d_gradient_scheme);
    velocity.owned_data().update(-time_step, d_face_flux_workspace.pressure_gradient().owned_data(), 1.0);

    d_mesh->sync_periodic_boundaries(velocity);

    if (accumulated_pressure != nullptr)
    {
        pressure_correction.owned_data().scale(reference_density);
        d_mesh->sync_periodic_boundaries(pressure_correction);
        accumulated_pressure->owned_data().update(scalar_type{1}, pressure_correction.owned_data(), scalar_type{1});
        d_mesh->sync_periodic_boundaries(*accumulated_pressure);

        FVM::pressure_weighted_face_fluxes(velocity, *accumulated_pressure, time_step / reference_density,
            pressure_flux_boundary_cache, d_pressure_boundary_conditions, d_face_flux_workspace, d_cached_face_fluxes,
            d_gradient_scheme);
        apply_fixed_boundary_fluxes(d_cached_face_fluxes);
    }
    else
    {
        FVM::pressure_weighted_face_fluxes(velocity, pressure_correction, d_face_flux_workspace.pressure_gradient(),
            time_step, pressure_flux_boundary_cache, d_pressure_correction_boundary_conditions, d_face_flux_workspace,
            d_cached_face_fluxes);
        apply_fixed_boundary_fluxes(d_cached_face_fluxes);
        pressure_correction.owned_data().scale(reference_density);
        d_mesh->sync_periodic_boundaries(pressure_correction);
    }

    scalar_type continuity_norm_squared = {};
    scalar_type continuity_normalization_squared = {};
    scalar_type continuity_maximum = {};
    {
        const auto corrected_flux_values = d_cached_face_fluxes.owned_read_view();
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid = static_cast<local_ordinal_type>(owned);
            const auto balance =
                FVM::cell_flux_balance<Pack>(*d_mesh, d_cached_face_fluxes, corrected_flux_values, cell_lid);
            const auto target = continuity_target.integrated_rate(cell_lid);
            const auto residual = balance - target;
            continuity_norm_squared += residual * residual;
            continuity_normalization_squared += std::max(balance * balance, target * target);
            continuity_maximum = std::max(continuity_maximum, std::abs(residual));
        }
    }

    const std::array<scalar_type, 3> local_norms_squared{
        pressure_norm_squared, continuity_norm_squared, continuity_normalization_squared};
    std::array<scalar_type, 3> global_norms_squared{};
    Teuchos::reduceAll(*owned_map->getComm(), Teuchos::REDUCE_SUM, static_cast<int>(local_norms_squared.size()),
        local_norms_squared.data(), global_norms_squared.data());
    scalar_type global_continuity_maximum{};
    Teuchos::reduceAll(*owned_map->getComm(), Teuchos::REDUCE_MAX, 1, &continuity_maximum, &global_continuity_maximum);

    if (reuse_cached_predictor_flux)
    {
        ++d_cached_predictor_flux_reuse_count;
    }
    if (accumulated_pressure != nullptr)
    {
        d_cached_predictor_flux_valid = true;
        d_cached_target_generation = continuity_target.generation();
        d_cached_target_geometry_epoch = continuity_target.geometry_epoch();
        d_cached_predictor_time_step = time_step;
        d_cached_predictor_reference_density = reference_density;
        d_cached_fixed_boundary_flux_generation = d_fixed_boundary_flux_generation;
    }

    using std::sqrt;
    const auto continuity_l2 = sqrt(global_norms_squared[1]);
    const auto normalization = sqrt(global_norms_squared[2]);
    return {reference_density * sqrt(global_norms_squared[0]), continuity_l2, linear_statistics,
        {continuity_l2, global_continuity_maximum,
            normalization > scalar_type{} ? continuity_l2 / normalization : scalar_type{}, normalization}};
}

} // namespace SimpleFluid
