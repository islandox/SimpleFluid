/**
 * @file FluidSolver.tcc
 * @brief Template implementations for FluidSolver.
 */

#include "FluidSolver.hh"
#include "geometry/MeshFactory.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SimpleFluid
{

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::require_mesh(SP<const mesh_type> mesh)
    -> SP<const mesh_type>
{
    return EquationValidation::require_non_null_mesh(
        std::move(mesh), "FluidSolver");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::require_legacy_mesh(
    const SP<const MeshHandle<Pack>>& mesh) -> SP<const mesh_type>
{
    if (!mesh)
    {
        throw std::invalid_argument(
            "FluidSolver requires a non-null mesh handle.");
    }
    auto legacy = mesh->legacy_mesh();
    if (legacy)
    {
        return legacy;
    }

    const auto* cartesian =
        std::get_if<typename MeshHandle<Pack>::CartesianPtr>(
            &mesh->variant());
    if (!cartesian)
    {
        throw std::invalid_argument(
            "FluidSolver currently supports STK and Cartesian mesh handles.");
    }
    if (mesh->num_owned_cells() != (*cartesian)->num_cells())
    {
        throw std::invalid_argument(
            "FluidSolver Cartesian compatibility currently requires a "
            "serial mesh.");
    }

    auto database = std::make_shared<Database>();
    database->set("dimension", 3);
    database->set("mesh_size", real_t{1.0});
    database->set(
        "domain_type",
        static_cast<int>(MeshFactory::DomainType::BOX));
    database->set("X", ArrReal((*cartesian)->cell_edges()[0]));
    database->set("Y", ArrReal((*cartesian)->cell_edges()[1]));
    database->set("Z", ArrReal((*cartesian)->cell_edges()[2]));
    database->set(
        "domain_exterior_face_types",
        ArrString{
            "xmin", "xmax", "ymin", "ymax", "zmin", "zmax"});
    return MeshFactory(database).template build<Pack>();
}

template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<const mesh_type> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : FluidSolver(
          std::make_shared<MeshHandle<Pack>>(
              require_mesh(std::move(mesh))),
          std::move(boundary_conditions),
          time_options,
          linear_options)
{
}

template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options)
    : FluidSolver(
          std::move(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          true)
{
}

template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    DeferredMomentumEquationTag)
    : FluidSolver(
          std::move(mesh),
          std::move(boundary_conditions),
          time_options,
          linear_options,
          false)
{
}

template<TpetraTypePack Pack>
FluidSolver<Pack>::FluidSolver(
    SP<const MeshHandle<Pack>> mesh,
    BoundaryConditionSet boundary_conditions,
    TimeStepperOptions time_options,
    LinearSolverOptions linear_options,
    bool register_momentum_equation)
    : d_mesh(require_legacy_mesh(mesh)),
      d_problem(std::make_shared<MeshHandle<Pack>>(d_mesh),
                std::move(boundary_conditions),
                time_options,
                linear_options)
{
    if (d_problem.time_options().time_step <= 0.0)
    {
        throw std::invalid_argument(
            "FluidSolver requires a positive time step.");
    }

    d_problem.template emplace_object<FVM::VelocityBoundaryCache<Pack>>(
        "velocity_boundary_cache",
        FVM::cache_velocity_boundary_conditions<Pack>(
            d_mesh, d_problem.boundary_conditions()));
    if (register_momentum_equation)
    {
        d_problem.template emplace_object<
            IncompressibleMomentumEquation<Pack>>(
                "momentum_equation", d_mesh);
    }
    auto pressure_linear_options = d_problem.linear_options();
    pressure_linear_options.preconditioner =
        LinearPreconditioner::MueLu;
    d_problem.template emplace_object<PressureProjectionEquation<Pack>>(
        "pressure_projection", d_mesh, pressure_linear_options);
    d_problem.template emplace_object<CoupledPressureVelocitySolver<Pack>>(
        "coupled_pressure_velocity_solver", d_mesh);
    d_problem.template emplace_object<field_type>(
        "pressure", d_mesh, "pressure");
    d_problem.template emplace_object<field_type>(
        "pressure_correction", d_mesh, "pressure_correction");
    d_problem.template emplace_object<velocity_field_type>(
        "velocity", d_mesh, "velocity");
    d_problem.template emplace_object<velocity_field_type>(
        "pressure_velocity_predictor",
        d_mesh,
        "pressure_velocity_predictor");
    d_problem.template emplace_object<face_flux_field_type>(
        "old_face_flux", d_mesh, "old_face_flux");
    d_problem.template emplace_object<face_flux_field_type>(
        "projected_face_flux", d_mesh, "projected_face_flux");
    d_problem.template emplace_object<residual_type>(
        "pressure_velocity_residuals");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure() const noexcept -> const field_type&
{
    return d_problem.template object<field_type>("pressure");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity() const noexcept
    -> const velocity_field_type&
{
    return d_problem.template object<velocity_field_type>("velocity");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure() noexcept -> field_type&
{
    return d_problem.template object<field_type>("pressure");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity() noexcept -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>("velocity");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::last_pressure_velocity_residuals() const noexcept
    -> const residual_type&
{
    return pressure_velocity_residuals();
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::momentum_equation()
    -> IncompressibleMomentumEquation<Pack>&
{
    return d_problem.template object<
        IncompressibleMomentumEquation<Pack>>("momentum_equation");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_projection()
    -> PressureProjectionEquation<Pack>&
{
    return d_problem.template object<
        PressureProjectionEquation<Pack>>("pressure_projection");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::coupled_pressure_velocity_solver()
    -> CoupledPressureVelocitySolver<Pack>&
{
    return d_problem.template object<
        CoupledPressureVelocitySolver<Pack>>(
            "coupled_pressure_velocity_solver");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity_boundary_cache()
    -> FVM::VelocityBoundaryCache<Pack>&
{
    return d_problem.template object<
        FVM::VelocityBoundaryCache<Pack>>("velocity_boundary_cache");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_correction() -> field_type&
{
    return d_problem.template object<field_type>("pressure_correction");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::predictor_velocity() -> velocity_field_type&
{
    return d_problem.template object<velocity_field_type>(
        "pressure_velocity_predictor");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::old_face_fluxes() -> face_flux_field_type&
{
    return d_problem.template object<face_flux_field_type>("old_face_flux");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::projected_face_fluxes() -> face_flux_field_type&
{
    return d_problem.template object<face_flux_field_type>(
        "projected_face_flux");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_velocity_residuals() -> residual_type&
{
    return d_problem.template object<residual_type>(
        "pressure_velocity_residuals");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::pressure_velocity_residuals() const
    -> const residual_type&
{
    return d_problem.template object<residual_type>(
        "pressure_velocity_residuals");
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::velocity_update_norm(
    const velocity_field_type& before,
    const velocity_field_type& after) const -> scalar_type
{
    EquationValidation::require_mesh_match(
        *d_mesh, before, "FluidSolver");
    EquationValidation::require_mesh_match(
        *d_mesh, after, "FluidSolver");

    scalar_type norm_squared = {};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto delta = after.value(cell_lid) - before.value(cell_lid);
        norm_squared += delta.dot(delta) * d_mesh->cell_volume(cell_lid);
    }

    using std::sqrt;
    return sqrt(norm_squared);
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::advance_momentum() -> LinearSolveSummary
{
    return momentum_equation().advance_velocity(
        velocity(),
        old_face_fluxes(),
        velocity_boundary_cache(),
        d_problem.time_options(),
        velocity(),
        d_problem.linear_options());
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::run_momentum_predictor() -> LinearSolveSummary
{
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        predictor_velocity().set_value(
            cell_lid, velocity().value(cell_lid));
    }
    d_mesh->sync_periodic_boundaries(predictor_velocity());

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(), d_problem.time_options().time_step,
        velocity_boundary_cache(), old_face_fluxes());
    const auto linear_summary = advance_momentum();
    pressure_velocity_residuals().momentum =
        velocity_update_norm(predictor_velocity(), velocity());
    return linear_summary;
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::run_pressure_correction()
    -> typename PressureProjectionEquation<Pack>::ProjectionResult
{
    const auto result =
        pressure_projection().project(
            pressure_correction(),
            d_problem.time_options().time_step,
            velocity_boundary_cache(),
            velocity());
    pressure().owned_data().update(
        1.0, pressure_correction().owned_data(), 1.0);
    d_mesh->sync_periodic_boundaries(pressure());
    return result;
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::assemble_coupled_system()
    -> coupled_system_type
{
    return coupled_pressure_velocity_solver().assemble(
        momentum_equation(),
        velocity(),
        pressure(),
        old_face_fluxes(),
        velocity_boundary_cache(),
        d_problem.boundary_conditions(),
        d_problem.time_options());
}

template<TpetraTypePack Pack>
void FluidSolver<Pack>::solve_coupled_krylov()
{
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        predictor_velocity().set_value(
            cell_lid, velocity().value(cell_lid));
    }
    d_mesh->sync_periodic_boundaries(predictor_velocity());

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(), d_problem.time_options().time_step,
        velocity_boundary_cache(), old_face_fluxes());
    const auto system = assemble_coupled_system();
    const auto result =
        coupled_pressure_velocity_solver().solve(
            system,
            velocity(),
            pressure(),
            d_problem.linear_options());
    if (!result.converged)
    {
        throw std::runtime_error(
            "FluidSolver coupled Krylov solve did not converge.");
    }

    pressure_velocity_residuals().momentum =
        velocity_update_norm(predictor_velocity(), velocity());
    pressure_velocity_residuals().pressure =
        result.achieved_tolerance;
    pressure_velocity_residuals().achieved_tolerance =
        result.achieved_tolerance;
    pressure_velocity_residuals().linear_iterations =
        result.iterations;
    d_last_step_statistics.nonlinear_iterations = 1;
    d_last_step_statistics.add(LinearSolveStatistics{
        result.converged,
        result.iterations,
        result.achieved_tolerance});

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(), d_problem.time_options().time_step,
        velocity_boundary_cache(), projected_face_fluxes());
    scalar_type continuity_norm_squared = {};
    for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
    {
        const auto cell_lid = static_cast<local_ordinal_type>(owned);
        const auto balance =
            FVM::cell_flux_balance<Pack>(
                *d_mesh, projected_face_fluxes(), cell_lid);
        continuity_norm_squared += balance * balance;
    }
    using std::sqrt;
    pressure_velocity_residuals().continuity =
        sqrt(continuity_norm_squared);
}

template<TpetraTypePack Pack>
void FluidSolver<Pack>::solve_pressure_velocity_coupling()
{
    pressure_velocity_residuals() = {};
    if (d_problem.time_options().pressure_velocity_coupling
        == PressureVelocityCoupling::CoupledKrylov)
    {
        solve_coupled_krylov();
        return;
    }

    if (d_problem.time_options().n_pressure_correctors < 1)
    {
        throw std::invalid_argument(
            "FluidSolver requires at least one pressure corrector.");
    }
    if (d_problem.time_options().n_outer_correctors < 1)
    {
        throw std::invalid_argument(
            "FluidSolver requires at least one outer corrector.");
    }

    const auto pressure_corrections =
        d_problem.time_options().pressure_velocity_coupling
                == PressureVelocityCoupling::SIMPLE
          ? 1
          : d_problem.time_options().n_pressure_correctors;
    const auto outer_corrections =
        d_problem.time_options().pressure_velocity_coupling
                == PressureVelocityCoupling::PIMPLE
          ? d_problem.time_options().n_outer_correctors
          : 1;

    for (int outer = 0; outer < outer_corrections; ++outer)
    {
        ++d_last_step_statistics.nonlinear_iterations;
        d_last_step_statistics.add(run_momentum_predictor());

        typename PressureProjectionEquation<Pack>::ProjectionResult result;
        for (int corrector = 0;
             corrector < pressure_corrections;
             ++corrector)
        {
            result = run_pressure_correction();
            d_last_step_statistics.add(result.linear_solve);
        }

        pressure_velocity_residuals().pressure =
            result.pressure_correction;
        pressure_velocity_residuals().continuity =
            result.continuity;
    }
    pressure_velocity_residuals().linear_iterations =
        d_last_step_statistics.krylov_iterations;
    pressure_velocity_residuals().achieved_tolerance =
        d_last_step_statistics.achieved_tolerance;

    FVM::pressure_weighted_face_fluxes(
        velocity(), pressure(), d_problem.time_options().time_step,
        velocity_boundary_cache(), projected_face_fluxes());
}

template<TpetraTypePack Pack>
void FluidSolver<Pack>::begin_step()
{
    d_last_step_statistics = {};
    if (d_step_index == 0)
    {
        d_mesh->sync_periodic_boundaries(velocity());
    }
}

template<TpetraTypePack Pack>
void FluidSolver<Pack>::finish_step()
{
    d_last_step_statistics.momentum =
        pressure_velocity_residuals().momentum;
    d_last_step_statistics.pressure =
        pressure_velocity_residuals().pressure;
    d_last_step_statistics.continuity =
        pressure_velocity_residuals().continuity;
    d_mesh->sync_periodic_boundaries(velocity());
    d_time += d_problem.time_options().time_step;
    ++d_step_index;
}

template<TpetraTypePack Pack>
void FluidSolver<Pack>::step()
{
    begin_step();
    solve_pressure_velocity_coupling();
    finish_step();
}

template<TpetraTypePack Pack>
void FluidSolver<Pack>::run(int steps)
{
    if (steps < 0)
    {
        throw std::invalid_argument(
            "FluidSolver::run steps cannot be negative.");
    }

    for (int step_id = 0; step_id < steps; ++step_id)
    {
        step();
    }
}

template<TpetraTypePack Pack>
auto FluidSolver<Pack>::collect_scalar_field(
    const field_type& field) const -> VTUWriter::ScalarData
{
    VTUWriter::ScalarData values;
    values.reserve(d_mesh->num_local_cells());
    for (size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        values.push_back(static_cast<real_t>(
            field.local_value(
                static_cast<local_ordinal_type>(lid))));
    }
    return values;
}

template<TpetraTypePack Pack>
VTUWriter FluidSolver<Pack>::fluid_solution_writer() const
{
    std::unordered_map<global_ordinal_type, global_index_t> node_lid;
    VTUWriter::VectorData node_coords;
    VTUWriter::Int64Data cell_node_offsets;
    VTUWriter::Int64Data cell_node_ids;
    VTUWriter::UInt8Data cell_types;

    auto append_node = [&](global_ordinal_type node_gid) -> global_index_t
    {
        const auto iter = node_lid.find(node_gid);
        if (iter != node_lid.end())
        {
            return iter->second;
        }

        const auto lid = static_cast<global_index_t>(node_coords.size());
        node_lid.emplace(node_gid, lid);
        node_coords.push_back(d_mesh->node_coord(node_gid));
        return lid;
    };

    for (size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        const auto& cell_info =
            d_mesh->cell(static_cast<local_ordinal_type>(lid));
        for (const auto node_gid : cell_info.node_gids)
        {
            cell_node_ids.push_back(append_node(node_gid));
        }
        cell_node_offsets.push_back(
            static_cast<global_index_t>(cell_node_ids.size()));
        cell_types.push_back(static_cast<std::uint8_t>(
            MeshUtils::vtu_cell_type_code(cell_info.type)));
    }

    VTUWriter::VectorData velocity_values;
    velocity_values.reserve(d_mesh->num_local_cells());
    for (size_t lid = 0; lid < d_mesh->num_local_cells(); ++lid)
    {
        velocity_values.push_back(
            velocity().local_value(
                static_cast<local_ordinal_type>(lid)));
    }

    VTUWriter writer;
    writer.set_points(std::move(node_coords));
    writer.set_cells(
        std::move(cell_node_ids),
        std::move(cell_node_offsets),
        std::move(cell_types));
    writer.add_scalar_cell_data(
        "pressure", collect_scalar_field(pressure()));
    writer.add_vector_cell_data(
        "velocity", std::move(velocity_values));
    return writer;
}

template<TpetraTypePack Pack>
void FluidSolver<Pack>::write_solution_vtu(
    const std::string& filename) const
{
    fluid_solution_writer().write(filename);
}

} // namespace SimpleFluid
