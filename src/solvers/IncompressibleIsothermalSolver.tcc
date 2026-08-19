/**
 * @file IncompressibleIsothermalSolver.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for IncompressibleIsothermalSolver.
 * @version 0.1
 * @date 2026-08-20
 *
 * @copyright Copyright (c) 2026
 *
 */

#include "IncompressibleIsothermalSolver.hh"

#include <Teuchos_CommHelpers.hpp>

#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/** @brief Wrap a legacy mesh without reconstructing it. */
template<TpetraTypePack Pack>
IncompressibleIsothermalSolver<Pack>::IncompressibleIsothermalSolver(SP<const legacy_mesh_type> mesh,
    BoundaryConditionSet boundary_conditions, TimeStepperOptions time_options, LinearSolverOptions linear_options,
    scalar_type reference_density)
    : IncompressibleIsothermalSolver(std::make_shared<mesh_type>(require_mesh(std::move(mesh))),
          std::move(boundary_conditions), time_options, linear_options, reference_density)
{
}

/** @brief Initialize constant molecular properties and optional-RANS storage. */
template<TpetraTypePack Pack>
IncompressibleIsothermalSolver<Pack>::IncompressibleIsothermalSolver(SP<const mesh_type> mesh,
    BoundaryConditionSet boundary_conditions, TimeStepperOptions time_options, LinearSolverOptions linear_options,
    scalar_type reference_density)
    : base_type(std::move(mesh), std::move(boundary_conditions), time_options, linear_options),
      d_reference_density(reference_density)
{
    if (!std::isfinite(d_reference_density) || d_reference_density <= scalar_type{})
    {
        throw std::invalid_argument("IncompressibleIsothermalSolver requires a finite positive "
                                    "reference density.");
    }

    BoussinesqModelOptions material_options;
    material_options.reference_density = static_cast<real_t>(d_reference_density);
    material_options.density = static_cast<real_t>(d_reference_density);
    material_options.specific_heat_capacity = 1.0;
    material_options.dynamic_viscosity =
        static_cast<real_t>(d_reference_density) * d_problem.time_options().kinematic_viscosity;
    material_options.thermal_conductivity = 0.0;

    d_problem.template emplace_object<material_type>(
        "material_properties", d_mesh, material_options, d_problem.time_options());
    d_problem.template emplace_object<turbulence_model_type>(
        "turbulence_model", d_mesh, d_problem.boundary_conditions());

    // FluidSolver's legacy objects use CellField storage. RANS is natively
    // MeshHandle/FieldStored-based, so retain the established public fields and
    // add only the canonical handle-backed workspaces needed by its momentum
    // hook. The supplied legacy mesh remains the MeshHandle's exact backend.
    if (uses_legacy_backend())
    {
        d_problem.template emplace_object<momentum_equation_type>("isothermal_momentum_equation", d_mesh);
        d_problem.template emplace_object<canonical_velocity_boundary_cache_type>("isothermal_velocity_boundary_cache",
            FVM::cache_velocity_boundary_conditions<Pack>(d_mesh, d_problem.boundary_conditions()));
        d_problem.template emplace_object<canonical_face_flux_workspace_type>(
            "isothermal_pressure_face_flux_workspace", d_mesh);
        d_problem.template emplace_object<canonical_coupled_solver_type>(
            "isothermal_coupled_pressure_velocity_solver", d_mesh);
    }
}

/** @brief Return the handle-backed momentum equation on either backend. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::isothermal_momentum_equation() -> momentum_equation_type&
{
    if (!uses_legacy_backend())
    {
        return native_momentum_equation();
    }
    return d_problem.template object<momentum_equation_type>("isothermal_momentum_equation");
}

/** @brief Return the handle-backed velocity boundary cache. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::isothermal_velocity_boundary_cache()
    -> canonical_velocity_boundary_cache_type&
{
    if (!uses_legacy_backend())
    {
        return native_velocity_boundary_cache();
    }
    return d_problem.template object<canonical_velocity_boundary_cache_type>("isothermal_velocity_boundary_cache");
}

/** @brief Return the handle-backed pressure-flux workspace. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::isothermal_pressure_face_flux_workspace()
    -> canonical_face_flux_workspace_type&
{
    if (!uses_legacy_backend())
    {
        return native_pressure_face_flux_workspace();
    }
    return d_problem.template object<canonical_face_flux_workspace_type>("isothermal_pressure_face_flux_workspace");
}

/** @brief Return the handle-backed coupled assembler. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::isothermal_coupled_pressure_velocity_solver()
    -> canonical_coupled_solver_type&
{
    if (!uses_legacy_backend())
    {
        return native_coupled_pressure_velocity_solver();
    }
    return d_problem.template object<canonical_coupled_solver_type>("isothermal_coupled_pressure_velocity_solver");
}

/** @brief Return mutable molecular material fields. */
template<TpetraTypePack Pack> auto IncompressibleIsothermalSolver<Pack>::stored_material_properties() -> material_type&
{
    return d_problem.template object<material_type>("material_properties");
}

/** @brief Return immutable molecular material fields. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::stored_material_properties() const -> const material_type&
{
    return d_problem.template object<material_type>("material_properties");
}

/** @brief Expose mutable molecular material fields. */
template<TpetraTypePack Pack> auto IncompressibleIsothermalSolver<Pack>::material_properties() -> material_type&
{
    return stored_material_properties();
}

/** @brief Expose immutable molecular material fields. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::material_properties() const -> const material_type&
{
    return stored_material_properties();
}

/** @brief Return mutable Problem-owned turbulence storage. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::stored_turbulence_model() -> turbulence_model_type&
{
    return d_problem.template object<turbulence_model_type>("turbulence_model");
}

/** @brief Return immutable Problem-owned turbulence storage. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::stored_turbulence_model() const -> const turbulence_model_type&
{
    return d_problem.template object<turbulence_model_type>("turbulence_model");
}

/** @brief Configure an isothermal turbulence closure. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::configure_turbulence(const TurbulenceModelOptions& options)
    -> turbulence_model_type&
{
    if (options.buoyancy_model != TurbulenceBuoyancyModel::None)
    {
        throw std::invalid_argument("IncompressibleIsothermalSolver does not support turbulence "
                                    "buoyancy production.");
    }
    auto& model = stored_turbulence_model();
    model.configure(options, stored_material_properties(), d_reference_density);
    return model;
}

/** @brief Parse and configure an isothermal turbulence closure. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::configure_turbulence(const Database& database) -> turbulence_model_type&
{
    return configure_turbulence(turbulence_model_options_from_database(database));
}

/** @brief Disable the active turbulence model. */
template<TpetraTypePack Pack> bool IncompressibleIsothermalSolver<Pack>::remove_turbulence_model() noexcept
{
    return stored_turbulence_model().disable();
}

/** @brief Return the mutable active turbulence model. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::find_turbulence_model() noexcept -> turbulence_model_type*
{
    auto& model = stored_turbulence_model();
    return model.enabled() ? &model : nullptr;
}

/** @brief Return the immutable active turbulence model. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::find_turbulence_model() const noexcept -> const turbulence_model_type*
{
    const auto& model = stored_turbulence_model();
    return model.enabled() ? &model : nullptr;
}

/** @brief Advance momentum using molecular or effective dynamic viscosity. */
template<TpetraTypePack Pack> auto IncompressibleIsothermalSolver<Pack>::advance_momentum() -> LinearSolveSummary
{
    auto& pressure_workspace = isothermal_pressure_face_flux_workspace();
    FVM::cell_gradient(pressure(), d_problem.boundary_conditions().pressure, predictor_pressure_gradient(),
        pressure_workspace.gradient_cache(), d_problem.time_options().pressure_gradient_scheme);

    const auto inverse_reference_density = scalar_type{1} / d_reference_density;
    const auto pressure_gradient_values = predictor_pressure_gradient().owned_read_view();
    const auto* turbulence = find_turbulence_model();
    using gradient_view_type = decltype(predictor_pressure_gradient().owned_read_view());
    std::optional<gradient_view_type> turbulent_gradient_values;
    if (turbulence != nullptr)
    {
        turbulent_gradient_values.emplace(turbulence->turbulent_kinetic_energy_gradient().owned_read_view());
    }

    auto acceleration_source = [&](local_ordinal_type cell_lid) -> vec_type
    {
        vec_type acceleration{pressure_gradient_values(cell_lid, 0) * (-inverse_reference_density),
            pressure_gradient_values(cell_lid, 1) * (-inverse_reference_density),
            pressure_gradient_values(cell_lid, 2) * (-inverse_reference_density)};
        if (turbulent_gradient_values)
        {
            constexpr scalar_type turbulent_pressure_factor{-2.0 / 3.0};
            acceleration.x += (*turbulent_gradient_values)(cell_lid, 0) * turbulent_pressure_factor;
            acceleration.y += (*turbulent_gradient_values)(cell_lid, 1) * turbulent_pressure_factor;
            acceleration.z += (*turbulent_gradient_values)(cell_lid, 2) * turbulent_pressure_factor;
        }
        return acceleration;
    };

    const auto& dynamic_viscosity = turbulence != nullptr ? turbulence->effective_dynamic_viscosity()
                                                          : stored_material_properties().dynamic_viscosity;
    return isothermal_momentum_equation().advance_velocity_physical(velocity(), old_face_fluxes(),
        isothermal_velocity_boundary_cache(), d_problem.time_options(), dynamic_viscosity, d_reference_density,
        velocity(), acceleration_source, d_problem.linear_options(),
        turbulence != nullptr ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr);
}

/** @brief Assemble the coupled isothermal velocity-pressure system. */
template<TpetraTypePack Pack>
auto IncompressibleIsothermalSolver<Pack>::assemble_coupled_system() -> coupled_system_type
{
    const auto* turbulence = find_turbulence_model();
    const auto& dynamic_viscosity = turbulence != nullptr ? turbulence->effective_dynamic_viscosity()
                                                          : stored_material_properties().dynamic_viscosity;
    return isothermal_coupled_pressure_velocity_solver().assemble(isothermal_momentum_equation(), velocity(),
        pressure(), old_face_fluxes(), isothermal_velocity_boundary_cache(), d_problem.boundary_conditions(),
        d_problem.time_options(), d_reference_density, &dynamic_viscosity,
        turbulence != nullptr ? &turbulence->turbulent_kinetic_energy_gradient() : nullptr,
        turbulence != nullptr ? turbulence->effective_dynamic_viscosity_boundary_cache() : nullptr);
}

/** @brief Advance pressure, velocity, and optional turbulence one step. */
template<TpetraTypePack Pack> void IncompressibleIsothermalSolver<Pack>::step()
{
    begin_step();
    if (auto* turbulence = find_turbulence_model())
    {
        turbulence->refresh_effective_properties(stored_material_properties(), d_reference_density);
    }

    solve_pressure_velocity_coupling();
    if (auto* turbulence = find_turbulence_model())
    {
        const auto statistics = turbulence->advance(velocity(), projected_face_fluxes(),
            isothermal_velocity_boundary_cache(), d_problem.time_options().time_step, stored_material_properties(),
            d_reference_density, d_problem.time_options().non_orthogonal_treatment, d_problem.linear_options());
        d_last_step_statistics.add(statistics);
    }
    finish_step();
}

/** @brief Build a writer containing requested isothermal solution fields. */
template<TpetraTypePack Pack>
VTUWriter IncompressibleIsothermalSolver<Pack>::solution_writer(const SolutionOutputOptions& output_options) const
{
    auto writer = fluid_solution_writer();
    if (output_options.include_material_properties)
    {
        const auto& material = stored_material_properties();
        writer.add_scalar_cell_data("density", collect_scalar_field(material.density));
        writer.add_scalar_cell_data("dynamic_viscosity", collect_scalar_field(material.dynamic_viscosity));
    }
    if (output_options.include_turbulence_fields)
    {
        if (const auto* turbulence = find_turbulence_model())
        {
            for (const auto& [name, field] : turbulence->output_fields())
            {
                writer.add_scalar_cell_data(name, collect_scalar_field(*field));
            }
        }
    }
    return writer;
}

/** @brief Write one isothermal VTU solution piece. */
template<TpetraTypePack Pack>
void IncompressibleIsothermalSolver<Pack>::write_solution_vtu(
    const std::string& filename, const SolutionOutputOptions& output_options) const
{
    solution_writer(output_options).write(filename, VTUWriter::Encoding::AppendedBinary);
}

/** @brief Write collision-free pieces and a rank-zero PVTU index. */
template<TpetraTypePack Pack>
void IncompressibleIsothermalSolver<Pack>::write_parallel_solution_vtu(
    const std::string& filename, const SolutionOutputOptions& output_options) const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const auto rank = communicator->getRank();
    const auto rank_count = communicator->getSize();
    if (rank_count > 1)
    {
        const auto option_mask = static_cast<int>(output_options.include_sources) |
                                 (static_cast<int>(output_options.include_material_properties) << 1) |
                                 (static_cast<int>(output_options.include_radiolytic_gas_fields) << 2) |
                                 (static_cast<int>(output_options.include_precursor_fields) << 3) |
                                 (static_cast<int>(output_options.include_turbulence_fields) << 4);
        std::array<long long, 2> root_arguments{rank == 0 ? static_cast<long long>(filename.size()) : 0LL,
            rank == 0 ? static_cast<long long>(option_mask) : 0LL};
        Teuchos::broadcast(*communicator, 0, static_cast<int>(root_arguments.size()), root_arguments.data());
        if (root_arguments[0] < 0 || root_arguments[0] > std::numeric_limits<int>::max())
        {
            throw std::invalid_argument("Parallel VTU filename is too long for MPI broadcast.");
        }

        std::string root_filename(static_cast<size_t>(root_arguments[0]), '\0');
        if (rank == 0)
        {
            root_filename = filename;
        }
        if (!root_filename.empty())
        {
            Teuchos::broadcast(*communicator, 0, static_cast<int>(root_filename.size()), root_filename.data());
        }

        const int local_arguments_mismatch =
            filename != root_filename || option_mask != static_cast<int>(root_arguments[1]);
        int any_arguments_mismatch = 0;
        Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_arguments_mismatch, &any_arguments_mismatch);
        if (any_arguments_mismatch != 0)
        {
            throw std::invalid_argument("Parallel VTU filename and output options must agree on "
                                        "every rank.");
        }
    }

    const auto piece_filename = VTUWriter::rank_piece_filename(filename, rank, rank_count);
    if (rank_count <= 1)
    {
        solution_writer(output_options).write(piece_filename, VTUWriter::Encoding::AppendedBinary);
        return;
    }

    std::optional<VTUWriter> writer;
    std::string local_schema_key;
    std::exception_ptr preparation_error;
    try
    {
        writer.emplace(solution_writer(output_options));
        local_schema_key = writer->cell_data_schema_key();
    }
    catch (...)
    {
        preparation_error = std::current_exception();
    }
    const int local_preparation_failed = preparation_error ? 1 : 0;
    int any_preparation_failed = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_preparation_failed, &any_preparation_failed);
    if (any_preparation_failed != 0)
    {
        if (preparation_error)
        {
            std::rethrow_exception(preparation_error);
        }
        throw std::runtime_error("VTU output preparation failed on another MPI rank.");
    }

    const int local_schema_size_is_valid =
        local_schema_key.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) ? 1 : 0;
    int global_schema_size_is_valid = 0;
    Teuchos::reduceAll(
        *communicator, Teuchos::REDUCE_MIN, 1, &local_schema_size_is_valid, &global_schema_size_is_valid);
    if (global_schema_size_is_valid == 0)
    {
        throw std::invalid_argument("Parallel VTU CellData schema is too large for MPI broadcast.");
    }

    int root_schema_size = rank == 0 ? static_cast<int>(local_schema_key.size()) : 0;
    Teuchos::broadcast(*communicator, 0, 1, &root_schema_size);
    std::string root_schema_key(static_cast<size_t>(root_schema_size), '\0');
    if (rank == 0)
    {
        root_schema_key = local_schema_key;
    }
    if (!root_schema_key.empty())
    {
        Teuchos::broadcast(*communicator, 0, root_schema_size, root_schema_key.data());
    }
    const int local_schema_mismatch = local_schema_key == root_schema_key ? 0 : 1;
    int any_schema_mismatch = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_schema_mismatch, &any_schema_mismatch);
    if (any_schema_mismatch != 0)
    {
        throw std::invalid_argument("Parallel VTU CellData names, types, and component counts must "
                                    "agree on every rank.");
    }

    std::exception_ptr piece_error;
    try
    {
        writer->write(piece_filename, VTUWriter::Encoding::AppendedBinary);
    }
    catch (...)
    {
        piece_error = std::current_exception();
    }
    const int local_piece_failed = piece_error ? 1 : 0;
    int any_piece_failed = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_piece_failed, &any_piece_failed);
    if (any_piece_failed != 0)
    {
        if (piece_error)
        {
            std::rethrow_exception(piece_error);
        }
        throw std::runtime_error("A VTU piece failed to write on another MPI rank.");
    }

    std::exception_ptr index_error;
    if (rank == 0)
    {
        try
        {
            std::vector<std::string> piece_filenames;
            piece_filenames.reserve(static_cast<size_t>(rank_count));
            for (int piece_rank = 0; piece_rank < rank_count; ++piece_rank)
            {
                piece_filenames.push_back(VTUWriter::rank_piece_filename(filename, piece_rank, rank_count));
            }
            writer->write_parallel_index(VTUWriter::parallel_index_filename(filename), piece_filenames);
        }
        catch (...)
        {
            index_error = std::current_exception();
        }
    }
    const int local_index_failed = index_error ? 1 : 0;
    int any_index_failed = 0;
    Teuchos::reduceAll(*communicator, Teuchos::REDUCE_MAX, 1, &local_index_failed, &any_index_failed);
    if (any_index_failed != 0)
    {
        if (index_error)
        {
            std::rethrow_exception(index_error);
        }
        throw std::runtime_error("The PVTU index failed to write on another MPI rank.");
    }
}

} // namespace SimpleFluid
