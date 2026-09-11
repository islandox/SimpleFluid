#include "solvers/CoupledNonlinearProblem.hh"

#include "FVM/CellOperators.hh"
#include "FVM/FaceFlux.hh"
#include "equations/IncompressibleMomentumEquation.hh"
#include "geometry/GeometryEpoch.hh"
#include "solvers/CoupledPressureVelocitySolver.hh"
#include "solvers/NoxNonlinearSolver.hh"

#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SimpleFluid
{
namespace
{
using Problem = CoupledNonlinearProblem;
using Pack = Problem::pack_type;
using MeshType = Problem::mesh_type;
using Velocity = Problem::velocity_field_type;
using Pressure = Problem::field_type;
using Flux = Problem::face_flux_field_type;
using Vector = Pack::vector_type;
using MultiVector = Pack::multi_vector_type;
using Operator = Pack::operator_type;
using LO = Pack::local_ordinal_type;
using Vec = vec3<double>;
using NativeVelocityBoundaryCache = FVM::FieldStoredVelocityBoundaryCache<Pack, MeshType>;
using FluxWorkspace = FVM::FieldStoredPressureWeightedFaceFluxWorkspace<Pack, MeshType>;
using CoupledSolver = CoupledPressureVelocitySolver<Pack, MeshType>;
using MomentumEquation = IncompressibleMomentumEquation<Pack, MeshType>;

void collective_require(const MeshType& mesh, bool condition, const char* message)
{
    const int local_invalid = !condition;
    int any_invalid = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    if (any_invalid != 0)
        throw std::invalid_argument(message);
}

SP<const MeshType> require_mesh(SP<const MeshType> mesh)
{
    if (!mesh || mesh->owned_cell_map().is_null())
        throw std::invalid_argument("CoupledNonlinearProblem requires a mapped native mesh.");
    return mesh;
}

void require_boundary_consistency(const MeshType& mesh, const BoundaryConditionSet& boundaries)
{
    // Like the native fixed-flux configuration, named boundary configuration
    // is replicated on the mesh communicator, even on ranks without a patch.
    std::uint64_t hash = 1469598103934665603ULL;
    const auto byte = [&](unsigned char value) { hash = (hash ^ value) * 1099511628211ULL; };
    const auto word = [&](std::uint64_t value)
    {
        for (int i = 0; i < 8; ++i)
        {
            byte(static_cast<unsigned char>(value & 0xff));
            value >>= 8;
        }
    };
    const auto name = [&](const std::string& value)
    {
        word(value.size());
        for (unsigned char character : value)
            byte(character);
    };
    const auto append = [&](const auto& map, const auto& value)
    {
        std::vector<std::string> names;
        for (const auto& entry : map)
            names.push_back(entry.first);
        std::sort(names.begin(), names.end());
        word(names.size());
        for (const auto& key : names)
        {
            name(key);
            const auto& condition = map.at(key);
            word(static_cast<std::uint64_t>(condition.type));
            value(condition);
        }
    };
    append(boundaries.velocity,
        [&](const auto& condition)
        {
            word(std::bit_cast<std::uint64_t>(static_cast<double>(condition.value.x)));
            word(std::bit_cast<std::uint64_t>(static_cast<double>(condition.value.y)));
            word(std::bit_cast<std::uint64_t>(static_cast<double>(condition.value.z)));
        });
    append(boundaries.pressure,
        [&](const auto& condition) { word(std::bit_cast<std::uint64_t>(static_cast<double>(condition.value))); });
    const auto signature = static_cast<unsigned long long>(hash);
    unsigned long long minimum = 0, maximum = 0;
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &signature, &minimum);
    Teuchos::reduceAll(*mesh.owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &signature, &maximum);
    collective_require(mesh, minimum == maximum,
        "Coupled nonlinear velocity and pressure boundary configuration must match on every mesh rank.");
}

bool finite_vector(const Vector& vector)
{
    bool valid = true;
    {
        const auto values = vector.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t row = 0; row < vector.getLocalLength(); ++row)
            valid = valid && std::isfinite(values(row, 0));
    }
    const int local_invalid = !valid;
    int any_invalid = 0;
    Teuchos::reduceAll(*vector.getMap()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
    return any_invalid == 0;
}

void unpack(const Vector& state, double density, Velocity& velocity, Pressure& pressure)
{
    {
        const auto values = state.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto u = velocity.owned_write_view();
        auto p = pressure.owned_write_view();
        for (size_t cell = 0; cell < velocity.num_owned_cells(); ++cell)
        {
            for (size_t component = 0; component < 3; ++component)
                u(cell, component) = values(4 * cell + component, 0);
            p(cell, 0) = density * values(4 * cell + 3, 0);
        }
    }
    velocity.sync_ghosts();
    pressure.sync_ghosts();
}

struct RowFace
{
    LO face = 0;
    LO neighbor = 0;
    double orientation = 1;
    bool interior = false;
    Vec boundary_value{};
};

struct FrozenGeometry
{
    SP<const MeshType> mesh;
    std::vector<std::vector<RowFace>> rows;
    BoundaryConditionSet homogeneous_boundaries;
    NativeVelocityBoundaryCache homogeneous_velocity;
    double density;
    double time_step;
    FVM::CellGradientScheme gradient_scheme;
    std::uint64_t epoch;

    FrozenGeometry(SP<const MeshType> mesh_, const BoundaryConditionSet& boundaries, const TimeStepperOptions& options,
        double density_)
        : mesh(std::move(mesh_)), homogeneous_boundaries(boundaries), homogeneous_velocity(mesh), density(density_),
          time_step(options.time_step), gradient_scheme(options.pressure_gradient_scheme),
          epoch(mesh_geometry_epoch(*mesh))
    {
        for (auto& [name, condition] : homogeneous_boundaries.velocity)
            condition.value = {};
        for (auto& [name, condition] : homogeneous_boundaries.pressure)
            condition.value = 0;
        homogeneous_velocity = FVM::cache_velocity_boundary_conditions<Pack>(mesh, homogeneous_boundaries);
        const auto locations = FVM::detail::boundary_face_locations(*mesh);
        rows.resize(mesh->num_owned_cells());
        for (size_t cell = 0; cell < rows.size(); ++cell)
        {
            const auto cell_lid = static_cast<LO>(cell);
            for (const auto face : mesh->faces(cell_lid))
            {
                RowFace entry;
                entry.face = face;
                entry.orientation = mesh->owner_cell(face) == cell_lid ? 1.0 : -1.0;
                entry.interior = mesh->is_interior_face(face);
                if (entry.interior)
                    entry.neighbor = mesh->opposite_or_periodic_neighbor_cell(face, cell_lid);
                else if (mesh->is_boundary_face(face))
                {
                    const auto& location = locations.at(static_cast<size_t>(face));
                    const auto& condition = boundaries.velocity.at(mesh->boundary_batch_name(location.batch_id));
                    if (condition.type == BoundaryConditionType::Dirichlet)
                        entry.boundary_value = condition.value;
                }
                rows[cell].push_back(entry);
            }
        }
    }
};

/** Keep the native numeric generation frozen when only its inverse is held. */
class RetainedPreconditioner final : public Operator
{
public:
    RetainedPreconditioner(Teuchos::RCP<const Operator> inverse, Teuchos::RCP<const Operator> generation)
        : d_inverse(std::move(inverse)), d_generation(std::move(generation))
    {
    }

    Teuchos::RCP<const Pack::map_type> getDomainMap() const override { return d_inverse->getDomainMap(); }
    Teuchos::RCP<const Pack::map_type> getRangeMap() const override { return d_inverse->getRangeMap(); }
    bool hasTransposeApply() const override { return d_inverse->hasTransposeApply(); }
    void apply(const MultiVector& input, MultiVector& output, Teuchos::ETransp mode = Teuchos::NO_TRANS,
        double alpha = 1.0, double beta = 0.0) const override
    {
        d_inverse->apply(input, output, mode, alpha, beta);
    }

private:
    Teuchos::RCP<const Operator> d_inverse;
    Teuchos::RCP<const Operator> d_generation;
};

/** B + d(phi) u_upwind, retaining the base iterate and operator generation. */
class AnalyticCoupledOperator final : public Operator
{
public:
    AnalyticCoupledOperator(std::shared_ptr<const FrozenGeometry> geometry, CoupledSolver::system_type base,
        const Velocity& velocity, const Flux& flux)
        : d_geometry(std::move(geometry)), d_base(std::move(base)),
          d_direction_velocity(d_geometry->mesh, "nonlinear_direction_velocity"),
          d_direction_pressure(d_geometry->mesh, "nonlinear_direction_pressure"),
          d_direction_flux(d_geometry->mesh, "nonlinear_direction_flux"), d_flux_workspace(d_geometry->mesh)
    {
        const auto u = velocity.local_read_view();
        const auto phi = flux.local_read_view();
        d_donors.resize(d_geometry->rows.size());
        for (size_t cell = 0; cell < d_geometry->rows.size(); ++cell)
            for (const auto& face : d_geometry->rows[cell])
            {
                const auto outward = face.orientation * phi(face.face, 0);
                if (outward < 0 && !face.interior)
                    d_donors[cell].push_back(face.boundary_value);
                else
                {
                    // At exactly zero interior flux use the canonical owner
                    // on both rows. This selects one conservative generalized
                    // derivative at the nonsmooth upwind switch.
                    const bool own_donor =
                        face.interior && phi(face.face, 0) == 0 ? face.orientation > 0 : outward >= 0;
                    const auto donor = own_donor ? static_cast<LO>(cell) : face.neighbor;
                    d_donors[cell].push_back({u(donor, 0), u(donor, 1), u(donor, 2)});
                }
            }
    }

    Teuchos::RCP<const Pack::map_type> getDomainMap() const override { return d_base.map; }
    Teuchos::RCP<const Pack::map_type> getRangeMap() const override { return d_base.map; }
    bool hasTransposeApply() const override { return false; }

    void apply(const MultiVector& input, MultiVector& output, Teuchos::ETransp mode = Teuchos::NO_TRANS,
        double alpha = 1.0, double beta = 0.0) const override
    {
        collective_require(*d_geometry->mesh, mode == Teuchos::NO_TRANS,
            "Analytic coupled Newton action does not support transpose apply.");
        const bool valid_input_map = input.getMap()->isSameAs(*d_base.map);
        const bool valid_output_map = output.getMap()->isSameAs(*d_base.map);
        collective_require(*d_geometry->mesh,
            valid_input_map && valid_output_map && input.getNumVectors() == output.getNumVectors(),
            "Analytic coupled Newton action received incompatible maps or columns.");
        collective_require(*d_geometry->mesh, mesh_geometry_epoch(*d_geometry->mesh) == d_geometry->epoch,
            "Analytic coupled Newton action belongs to an obsolete geometry epoch.");
        if (alpha == 0)
        {
            if (beta == 0)
                output.putScalar(0);
            else
                output.scale(beta);
            return;
        }
        if (d_result.is_null() || d_result->getNumVectors() != input.getNumVectors())
            d_result = Teuchos::rcp(new MultiVector(d_base.map, input.getNumVectors(), true));
        d_base.linear_operator->apply(input, *d_result);
        for (size_t column = 0; column < input.getNumVectors(); ++column)
        {
            unpack(*input.getVector(column), d_geometry->density, d_direction_velocity, d_direction_pressure);
            // The reconstruction is affine in (u,p). Homogeneous boundary data
            // remove its constant exactly, yielding analytic directional flux.
            FVM::pressure_weighted_face_fluxes(d_direction_velocity, d_direction_pressure,
                d_geometry->time_step / d_geometry->density, d_geometry->homogeneous_velocity,
                d_geometry->homogeneous_boundaries.pressure, d_flux_workspace, d_direction_flux,
                d_geometry->gradient_scheme);
            const auto delta_phi = d_direction_flux.local_read_view();
            const auto result_column = d_result->getVectorNonConst(column);
            auto result = result_column->getLocalViewHost(Tpetra::Access::ReadWrite);
            for (size_t cell = 0; cell < d_geometry->rows.size(); ++cell)
                for (size_t face_index = 0; face_index < d_geometry->rows[cell].size(); ++face_index)
                {
                    const auto& face = d_geometry->rows[cell][face_index];
                    const double delta_outward = face.orientation * delta_phi(face.face, 0);
                    for (size_t component = 0; component < 3; ++component)
                        result(4 * cell + component, 0) +=
                            delta_outward * d_donors[cell][face_index].component(component);
                }
        }
        // Complete every input read before touching output, including aliases.
        output.update(alpha, *d_result, beta);
    }

private:
    std::shared_ptr<const FrozenGeometry> d_geometry;
    CoupledSolver::system_type d_base;
    std::vector<std::vector<Vec>> d_donors;
    mutable Velocity d_direction_velocity;
    mutable Pressure d_direction_pressure;
    mutable Flux d_direction_flux;
    mutable FluxWorkspace d_flux_workspace;
    mutable Teuchos::RCP<MultiVector> d_result;
};
} // namespace

struct CoupledNonlinearProblem::Impl
{
    SP<const mesh_type> mesh;
    BoundaryConditionSet boundaries;
    TimeStepperOptions time;
    NonlinearSolverOptions nonlinear;
    double density;
    double initial_pressure_gauge = 0;
    continuity_target_type target;
    Velocity accepted_velocity;
    Pressure initial_pressure;
    Velocity trial_velocity;
    Pressure trial_pressure;
    Flux trial_flux;
    FluxWorkspace flux_workspace;
    NativeVelocityBoundaryCache velocity_boundaries;
    MomentumEquation dynamic_equation;
    CoupledSolver dynamic_solver;
    CoupledSolver::system_type static_system;
    std::shared_ptr<FrozenGeometry> geometry;
    Teuchos::RCP<Vector> state_scale;
    Teuchos::RCP<Vector> residual_scale;
    VolumeContinuityResiduals<double> last_continuity;
    CoupledNonlinearProblemStatistics counters;

    Impl(SP<const mesh_type> mesh_, const Velocity& accepted, const Pressure& pressure,
        const BoundaryConditionSet& boundaries_, const TimeStepperOptions& time_,
        const NonlinearSolverOptions& nonlinear_, double density_, const continuity_target_type* target_)
        : mesh(require_mesh(std::move(mesh_))), boundaries(boundaries_), time(time_), nonlinear(nonlinear_),
          density(density_), target(target_ ? *target_ : continuity_target_type(mesh)),
          accepted_velocity(mesh, "nonlinear_accepted_velocity"), initial_pressure(mesh, "nonlinear_initial_pressure"),
          trial_velocity(mesh, "nonlinear_trial_velocity"), trial_pressure(mesh, "nonlinear_trial_pressure"),
          trial_flux(mesh, "nonlinear_trial_flux"), flux_workspace(mesh), velocity_boundaries(mesh),
          dynamic_equation(mesh), dynamic_solver(mesh)
    {
        collective_require(*mesh, accepted.mesh_ptr().get() == mesh.get() && pressure.mesh_ptr().get() == mesh.get(),
            "CoupledNonlinearProblem requires accepted fields on its exact mesh.");
        const std::array<double, 9> parameters{density, time.time_step, time.kinematic_viscosity,
            nonlinear.velocity_scale, nonlinear.pressure_scale, nonlinear.momentum_residual_scale,
            nonlinear.continuity_residual_scale, nonlinear.absolute_tolerance, nonlinear.continuity_tolerance};
        bool valid = true;
        for (size_t i = 0; i < parameters.size(); ++i)
            valid =
                valid && std::isfinite(parameters[i]) && (i == 2 || i == 7 ? parameters[i] >= 0 : parameters[i] > 0);
        std::array<double, 9> minimum{}, maximum{};
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, static_cast<int>(parameters.size()),
            parameters.data(), minimum.data());
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, static_cast<int>(parameters.size()),
            parameters.data(), maximum.data());
        collective_require(*mesh, valid && minimum == maximum,
            "CoupledNonlinearProblem requires finite, positive, rank-consistent reference scales and timestep, "
            "and non-negative kinematic viscosity.");
        const int linearization = static_cast<int>(nonlinear.linearization);
        int minimum_linearization = 0, maximum_linearization = 0;
        Teuchos::reduceAll(
            *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &linearization, &minimum_linearization);
        Teuchos::reduceAll(
            *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &linearization, &maximum_linearization);
        collective_require(*mesh,
            minimum_linearization == maximum_linearization &&
                (nonlinear.linearization == CoupledLinearization::Picard ||
                    nonlinear.linearization == CoupledLinearization::AnalyticNewton),
            "CoupledNonlinearProblem requires a supported, rank-consistent linearization.");
        require_boundary_consistency(*mesh, boundaries);

        // Periodicity is mesh topology. Remove only explicit periodic markers;
        // a physical face carrying one is rejected by the local checks below.
        const auto locations = FVM::detail::boundary_face_locations(*mesh);
        bool valid_boundaries = true;
        bool orthogonal = true;
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            const auto lid = static_cast<LO>(cell);
            const auto volume = mesh->cell_volume(lid);
            orthogonal = orthogonal && std::isfinite(volume) && volume > 0;
            for (const auto face : mesh->faces(lid))
            {
                const bool interior = mesh->is_interior_face(face);
                const auto direction = interior ? mesh->cell_center_vector(face, lid)
                                                : mesh->face_centroid(face) - mesh->cell_centroid(lid);
                const auto area = mesh->face_area_vector_outward(face, lid);
                const double area_squared = area.dot(area);
                const double distance_squared = direction.dot(direction);
                const double projection = area.dot(direction);
                const auto tangent = distance_squared > 0 ? area - direction * (projection / distance_squared) : area;
                orthogonal = orthogonal && std::isfinite(area_squared) && std::isfinite(distance_squared) &&
                             area_squared > 0 && distance_squared > 0 && projection > 0 &&
                             tangent.dot(tangent) <= 1.0e-24 * area_squared;
                if (interior || !mesh->is_boundary_face(face))
                    continue;
                const auto location = locations.at(static_cast<size_t>(face));
                if (!location.active)
                {
                    valid_boundaries = false;
                    continue;
                }
                const auto& name = mesh->boundary_batch_name(location.batch_id);
                const auto velocity = boundaries.velocity.find(name);
                const auto p = boundaries.pressure.find(name);
                // Native Slip projection has exactly zero normal velocity on
                // coordinate planes. It therefore contributes neither an
                // advective boundary term nor a trial-dependent continuity
                // source; native component diffusion also excludes Slip.
                // Require exact Cartesian normals rather than tolerances so
                // those cancellations hold for every trial and direction.
                const auto normal = mesh->face_normal_outward(face, lid);
                const bool axis_aligned =
                    (std::abs(normal.x) == 1. && normal.y == 0. && normal.z == 0.) ||
                    (normal.x == 0. && std::abs(normal.y) == 1. && normal.z == 0.) ||
                    (normal.x == 0. && normal.y == 0. && std::abs(normal.z) == 1.);
                valid_boundaries = valid_boundaries && velocity != boundaries.velocity.end() &&
                                   (velocity->second.type == BoundaryConditionType::Dirichlet ||
                                       velocity->second.type == BoundaryConditionType::NoSlip ||
                                       (velocity->second.type == BoundaryConditionType::Slip && axis_aligned)) &&
                                   (p == boundaries.pressure.end() || p->second.type == BoundaryConditionType::Neumann);
            }
        }
        for (const auto& [name, condition] : boundaries.velocity)
            valid_boundaries = valid_boundaries && std::isfinite(condition.value.x) &&
                               std::isfinite(condition.value.y) && std::isfinite(condition.value.z) &&
                               (condition.type == BoundaryConditionType::Dirichlet ||
                                   condition.type == BoundaryConditionType::NoSlip ||
                                   condition.type == BoundaryConditionType::Slip ||
                                   condition.type == BoundaryConditionType::Periodic);
        for (const auto& [name, condition] : boundaries.pressure)
            valid_boundaries =
                valid_boundaries && std::isfinite(condition.value) &&
                (condition.type == BoundaryConditionType::Neumann || condition.type == BoundaryConditionType::Periodic);
        collective_require(
            *mesh, orthogonal, "CoupledNonlinearProblem currently requires fixed orthogonal finite-volume geometry.");
        collective_require(*mesh, valid_boundaries,
            "CoupledNonlinearProblem currently supports prescribed Dirichlet/NoSlip or axis-aligned Slip velocity "
            "and Neumann pressure on physical boundaries, plus mesh periodic interfaces.");
        std::erase_if(boundaries.velocity,
            [](const auto& entry) { return entry.second.type == BoundaryConditionType::Periodic; });
        std::erase_if(boundaries.pressure,
            [](const auto& entry) { return entry.second.type == BoundaryConditionType::Periodic; });
        target.validate(*mesh, "Coupled nonlinear continuity target");
        accepted_velocity.owned_data().update(1.0, accepted.owned_data(), 0.0);
        initial_pressure.owned_data().update(1.0, pressure.owned_data(), 0.0);
        accepted_velocity.sync_ghosts();
        initial_pressure.sync_ghosts();
        velocity_boundaries = FVM::cache_velocity_boundary_conditions<Pack>(mesh, boundaries);
        geometry = std::make_shared<FrozenGeometry>(mesh, boundaries, time, density);
        // Zero convection gives the immutable affine BE/diffusion/pressure
        // residual. Only this setup and linearize() assemble Schur products.
        trial_flux.put_value(0.0);
        MomentumEquation static_equation(mesh);
        CoupledSolver static_solver(mesh);
        static_system = static_solver.assemble(static_equation, accepted_velocity, initial_pressure, trial_flux,
            velocity_boundaries, boundaries, time, target, density);
        if (static_system.pressure_gauge_gid)
        {
            const auto gauge = mesh->owned_cell_map()->getLocalElement(*static_system.pressure_gauge_gid);
            const double local_gauge =
                gauge == Teuchos::OrdinalTraits<LO>::invalid() ? 0 : initial_pressure.value(gauge);
            Teuchos::reduceAll(
                *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &local_gauge, &initial_pressure_gauge);
        }
        // Keep only the residual action and RHS. Composite actions retain their
        // necessary blocks internally; assembled actions retain just their CRS.
        static_system.matrix = Teuchos::null;
        static_system.overlap_map = Teuchos::null;
        static_system.momentum = Teuchos::null;
        static_system.gradient = {};
        static_system.divergence = {};
        static_system.pressure_stabilization = Teuchos::null;
        static_system.schur = Teuchos::null;
        static_system.numeric_lease.reset();
        static_solver.clear_cache();
        state_scale = Teuchos::rcp(new Vector(static_system.map, true));
        residual_scale = Teuchos::rcp(new Vector(static_system.map, true));
        {
            auto state_values = state_scale->getLocalViewHost(Tpetra::Access::OverwriteAll);
            auto residual_values = residual_scale->getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
            {
                const auto lid = static_cast<LO>(cell);
                const auto volume = mesh->cell_volume(lid);
                for (size_t component = 0; component < 3; ++component)
                {
                    state_values(4 * cell + component, 0) = nonlinear.velocity_scale;
                    residual_values(4 * cell + component, 0) = 1 / (volume * nonlinear.momentum_residual_scale);
                }
                state_values(4 * cell + 3, 0) = nonlinear.pressure_scale;
                const bool gauge = static_system.pressure_gauge_gid &&
                                   *static_system.pressure_gauge_gid == mesh->owned_cell_map()->getGlobalElement(lid);
                residual_values(4 * cell + 3, 0) =
                    gauge ? 1 / nonlinear.pressure_scale : 1 / (volume * nonlinear.continuity_residual_scale);
            }
        }
        if (!finite_vector(*pack_initial()))
            throw std::invalid_argument("CoupledNonlinearProblem requires finite accepted velocity and pressure.");
    }

    Teuchos::RCP<Vector> pack_initial() const
    {
        auto state = Teuchos::rcp(new Vector(static_system.map, true));
        const auto u = accepted_velocity.owned_read_view();
        const auto p = initial_pressure.owned_read_view();
        auto values = state->getLocalViewHost(Tpetra::Access::OverwriteAll);
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            for (size_t component = 0; component < 3; ++component)
                values(4 * cell + component, 0) = u(cell, component);
            // Remove the irrelevant all-Neumann offset before division, so
            // pressure differences survive large physical reference levels.
            // This only prepares the initial guess; the snapshot and residual
            // equations, including the gauge row, are unchanged.
            values(4 * cell + 3, 0) = (p(cell, 0) - initial_pressure_gauge) / density;
        }
        return state;
    }

    bool trial(const Vector& state)
    {
        collective_require(
            *mesh, state.getMap()->isSameAs(*static_system.map), "Coupled nonlinear state has an incompatible map.");
        collective_require(*mesh, mesh_geometry_epoch(*mesh) == geometry->epoch,
            "Coupled nonlinear timestep context belongs to an obsolete geometry epoch.");
        if (!finite_vector(state))
            return false;
        unpack(state, density, trial_velocity, trial_pressure);
        // A finite normalized pressure may overflow when converted to Pa.
        if (!finite_vector(trial_pressure.owned_data()))
            return false;
        FVM::pressure_weighted_face_fluxes(trial_velocity, trial_pressure, time.time_step / density,
            velocity_boundaries, boundaries.pressure, flux_workspace, trial_flux, time.pressure_gradient_scheme);
        return finite_vector(trial_flux.owned_data());
    }

    void measure_continuity()
    {
        const auto flux = trial_flux.local_read_view();
        std::array<double, 2> local{};
        double local_maximum = 0;
        for (size_t cell = 0; cell < mesh->num_owned_cells(); ++cell)
        {
            const auto lid = static_cast<LO>(cell);
            const auto balance = FVM::cell_flux_balance<Pack>(*mesh, trial_flux, flux, lid);
            const auto desired = target.integrated_rate(lid);
            const auto residual = balance - desired;
            local[0] += residual * residual;
            local[1] += std::max(balance * balance, desired * desired);
            local_maximum = std::max(local_maximum, std::abs(residual));
        }
        std::array<double, 2> global{};
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local.data(), global.data());
        Teuchos::reduceAll(
            *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_maximum, &last_continuity.maximum);
        last_continuity.l2 = std::sqrt(global[0]);
        last_continuity.normalization = std::sqrt(global[1]);
        last_continuity.normalized_l2 =
            last_continuity.normalization > 0 ? last_continuity.l2 / last_continuity.normalization : last_continuity.l2;
    }

    bool residual(const Vector& state, Vector& residual)
    {
        ++counters.residual_evaluations;
        collective_require(*mesh, residual.getMap()->isSameAs(*static_system.map),
            "Coupled nonlinear residual has an incompatible map.");
        if (!trial(state))
            return false;
        static_system.linear_operator->apply(state, residual);
        residual.update(-1, *static_system.rhs, 1);
        {
            const auto u = trial_velocity.local_read_view();
            const auto phi = trial_flux.local_read_view();
            auto values = residual.getLocalViewHost(Tpetra::Access::ReadWrite);
            for (size_t cell = 0; cell < geometry->rows.size(); ++cell)
                for (const auto& face : geometry->rows[cell])
                {
                    const double outward = face.orientation * phi(face.face, 0);
                    Vec donor;
                    if (outward < 0 && !face.interior)
                        donor = face.boundary_value;
                    else
                    {
                        const auto upwind = outward >= 0 ? static_cast<LO>(cell) : face.neighbor;
                        donor = {u(upwind, 0), u(upwind, 1), u(upwind, 2)};
                    }
                    for (size_t component = 0; component < 3; ++component)
                        values(4 * cell + component, 0) += outward * donor.component(component);
                }
        }
        if (!finite_vector(residual))
            return false;
        measure_continuity();
        return std::isfinite(last_continuity.l2) && std::isfinite(last_continuity.maximum);
    }

    NonlinearLinearization linearize(const Vector& state)
    {
        if (!trial(state))
            throw std::invalid_argument("Cannot linearize a non-finite coupled nonlinear state.");
        ++counters.linearizations;
        auto system = dynamic_solver.assemble(dynamic_equation, accepted_velocity, initial_pressure, trial_flux,
            velocity_boundaries, boundaries, time, target, density);
        NonlinearLinearization result;
        result.right_preconditioner = Teuchos::rcp(
            new RetainedPreconditioner(dynamic_solver.right_preconditioner(system), system.linear_operator));
        if (nonlinear.linearization == CoupledLinearization::AnalyticNewton)
            result.jacobian =
                Teuchos::rcp(new AnalyticCoupledOperator(geometry, std::move(system), trial_velocity, trial_flux));
        else
            result.jacobian = system.linear_operator;
        return result;
    }

    bool convergence_gate(const Vector& state)
    {
        if (!trial(state))
            return false;
        measure_continuity();
        bool valid = std::isfinite(last_continuity.maximum) && std::isfinite(last_continuity.l2) &&
                     last_continuity.maximum <= nonlinear.continuity_tolerance;
        if (static_system.pressure_gauge_gid)
        {
            const auto gauge = mesh->owned_cell_map()->getLocalElement(*static_system.pressure_gauge_gid);
            if (gauge != Teuchos::OrdinalTraits<LO>::invalid())
            {
                const auto values = state.getLocalViewHost(Tpetra::Access::ReadOnly);
                valid = valid &&
                        std::abs(values(4 * gauge + 3, 0)) / nonlinear.pressure_scale <= nonlinear.absolute_tolerance;
            }
        }
        const int local_invalid = !valid;
        int any_invalid = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_invalid, &any_invalid);
        return any_invalid == 0;
    }
};

CoupledNonlinearProblem::CoupledNonlinearProblem(SP<const mesh_type> mesh, const velocity_field_type& accepted_velocity,
    const field_type& physical_pressure, const BoundaryConditionSet& boundaries, const TimeStepperOptions& time_options,
    const NonlinearSolverOptions& nonlinear_options, double reference_density,
    const continuity_target_type* continuity_target)
    : d_impl(std::make_shared<Impl>(std::move(mesh), accepted_velocity, physical_pressure, boundaries, time_options,
          nonlinear_options, reference_density, continuity_target))
{
}

CoupledNonlinearProblem::~CoupledNonlinearProblem() = default;

NonlinearCallbacks CoupledNonlinearProblem::callbacks()
{
    NonlinearCallbacks result;
    result.map = d_impl->static_system.map;
    result.state_scale = d_impl->state_scale;
    result.residual_scale = d_impl->residual_scale;
    result.residual = [context = d_impl](const Vector& x, Vector& f) { return context->residual(x, f); };
    result.linearize = [context = d_impl](const Vector& x) { return context->linearize(x); };
    result.convergence_gate = [context = d_impl](const Vector& x) { return context->convergence_gate(x); };
    return result;
}

Teuchos::RCP<CoupledNonlinearProblem::vector_type> CoupledNonlinearProblem::pack_initial() const
{
    return d_impl->pack_initial();
}

VolumeContinuityResiduals<double> CoupledNonlinearProblem::continuity() const
{
    return d_impl->last_continuity;
}
CoupledNonlinearProblemStatistics CoupledNonlinearProblem::statistics() const
{
    return d_impl->counters;
}

void CoupledNonlinearProblem::commit(
    const vector_type& state, velocity_field_type& velocity, field_type& physical_pressure, face_flux_field_type& flux)
{
    collective_require(*d_impl->mesh,
        velocity.mesh_ptr().get() == d_impl->mesh.get() && physical_pressure.mesh_ptr().get() == d_impl->mesh.get() &&
            flux.mesh_ptr().get() == d_impl->mesh.get(),
        "Coupled nonlinear commit requires destination fields on the context mesh.");
    if (!d_impl->trial(state))
        throw std::invalid_argument("Cannot commit a non-finite coupled nonlinear state.");
    d_impl->measure_continuity();
    collective_require(*d_impl->mesh,
        std::isfinite(d_impl->last_continuity.l2) && std::isfinite(d_impl->last_continuity.maximum),
        "Cannot commit a non-finite coupled nonlinear flux.");
    velocity.owned_data().update(1.0, d_impl->trial_velocity.owned_data(), 0.0);
    physical_pressure.owned_data().update(1.0, d_impl->trial_pressure.owned_data(), 0.0);
    flux.owned_data().update(1.0, d_impl->trial_flux.owned_data(), 0.0);
    velocity.sync_ghosts();
    physical_pressure.sync_ghosts();
    flux.sync_ghosts();
}
} // namespace SimpleFluid
