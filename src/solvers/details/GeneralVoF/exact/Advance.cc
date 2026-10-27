/** @file Advance.cc
 * @brief Compiled general VoF Advance operation.
 */
#include "solvers/details/GeneralVoF/Advance.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceRates.hh"
#include "solvers/details/GeneralVoF/NodeVelocity.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include "solvers/details/GeneralVoF/exact/FaceRates.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace SimpleFluid::detail::general_vof
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
double Advance<ArithmeticMode::Exact>::stable_timestep(
    const Resources& resources, const State& state, const std::vector<VoF::GeneralVoFNodeVelocity>& nodes)
{
    auto& mesh = resources.mesh;
    auto& material = resources.material;
    auto& options = resources.options;
    auto& flow = resources.flow;
    auto& volume = resources.volume;
    auto& thickness = resources.thickness;
    const Execution execution(*resources.mesh);
    const auto bound =
        flow.stable_timestep(state.face_velocity, std::min(material.gas_density, material.liquid_density),
            std::max(material.gas_viscosity, material.liquid_viscosity), material.surface_tension, options.flow);
    double local_thickness = std::numeric_limits<double>::infinity(), global_thickness = 0;
    for (double h : thickness)
        local_thickness = std::min(local_thickness, h);
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_thickness, &global_thickness);
    double speed = 0;
    for (const auto& node : nodes)
        speed = std::max(speed, std::hypot(node.velocity[0], node.velocity[1], node.velocity[2]));
    if (speed > 0 && std::isfinite(speed))
        speed = std::nextafter(speed, std::numeric_limits<double>::infinity());
    speed = execution.maximum(speed);
    // Supplement rounded face CFL with the authoritative exact rate, including
    // nonzero streams whose individual binary64 product is zero.
    exact::FaceMeasures owned;
    execution.collective(
        [&]
        {
            if (state.projection_arithmetic.face_rate.size() != mesh->num_owned_faces())
                throw std::runtime_error("Missing exact rate for timestep admission.");
            for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
                owned.emplace(mesh->face_global_id(LO(f)),
                    std::pair{state.projection_arithmetic.face_rate[f],
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>{}});
        });
    const auto rates = exact::exchange_face_measures(*mesh, *state.phases, owned, options.byte_budget / 2);
    double local_rate_bound = std::numeric_limits<double>::infinity();
    execution.collective(
        [&]
        {
            using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
            const auto geometry = state.phases->owned_geometry();
            const auto slots = geometry.size() + 32;
            const auto scalar_budget = options.byte_budget / 2 / slots;
            if (scalar_budget < 1024)
                throw std::runtime_error("Exact rate CFL cumulative reserve exhausted.");
            std::vector<M> sums(geometry.size());
            const auto lease = mesh->acquire_execution_view();
            for (const auto& tile : state.phases->tiles())
            {
                const auto face = mesh->overlap_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0)
                    throw std::runtime_error("Missing local exact CFL face.");
                const auto rate = rates.at(GO(tile.face_gid)).first;
                const auto magnitude = rate.sign() < 0 ? rate.scaled(-1) : rate;
                for (const auto cell : {mesh->owner_cell(face), mesh->neighbor_cell(face)})
                    if (cell >= 0 && std::size_t(cell) < sums.size())
                        sums[cell] = detail::vof_exact::add(sums[cell], magnitude, scalar_budget);
            }
            for (std::size_t c = 0; c < sums.size(); ++c)
                if (sums[c].sign())
                {
                    const auto exact =
                        M::volume(*geometry[c]).scaled(options.flow.max_courant).divided_by(sums[c], scalar_budget);
                    if (exact.compare(M::from_value(std::numeric_limits<double>::max())) > 0)
                        continue;
                    double value = exact.certified().as_double();
                    if (M::from_value(value).compare(exact) > 0)
                        value = std::nextafter(value, 0.0);
                    if (!(value > 0))
                        throw std::runtime_error("Exact rate has no positive representable timestep.");
                    local_rate_bound = std::min(local_rate_bound, value);
                }
        });
    double rate_bound = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_rate_bound, &rate_bound);
    // A trajectory displacement bound supplements net-face CFL: opposing
    // swept streams can exist on a face whose projected net Phi is zero.
    return std::min(
        rate_bound, speed > 0 ? std::min(bound, options.flow.max_courant * global_thickness / speed) : bound);
}
std::shared_ptr<State> Advance<ArithmeticMode::Exact>::prepare(Resources& resources, const State& old, double dt)
{
    auto& mesh = resources.mesh;
    auto& material = resources.material;
    auto& options = resources.options;
    auto& flow = resources.flow;
    auto& transport = resources.transport;
    auto& volume = resources.volume;
    auto& thickness = resources.thickness;
    const Execution execution(*resources.mesh);
    const auto phase_mass = [&](const PLIC::PhaseMeasures<ArithmeticMode::Exact>& phase)
    { return phase.gas.scaled(material.gas_density) + phase.carrier.scaled(material.liquid_density); };
    execution.require(std::isfinite(dt) && dt > 0, "General VoF timestep must be positive and finite.");
    double minimum = 0, maximum_dt = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &dt, &minimum);
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &dt, &maximum_dt);
    execution.require(minimum == maximum_dt, "General VoF timestep differs across ranks.");
    const auto nodes = node_velocity(*mesh, *old.phases, old.velocity, options.byte_budget);
    execution.require(dt <= stable_timestep(resources, old, nodes),
        "General VoF timestep exceeds its physical geometry/trajectory limit.");
    const auto controls = exact::face_volumes(*mesh, *old.phases, old.projection_arithmetic, dt, options.byte_budget);
    const auto trial = old.phases->plan_transport(nodes, controls, dt);
    double local_h = std::numeric_limits<double>::infinity(), global_h = 0;
    for (double h : thickness)
        local_h = std::min(local_h, h);
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_h, &global_h);
    execution.collective(
        [&]
        {
            const auto scalar_budget = options.byte_budget / 32;
            SimpleFluid::PLIC::TrajectoryLimits limits;
            limits.byte_budget = scalar_budget;
            const auto displacement_limit =
                SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(global_h)
                    .scaled(options.flow.max_courant)
                    .squared(scalar_budget);
            if (trial.node_trajectories().size() != nodes.size())
                throw std::runtime_error("Incomplete exact node trajectory CFL certificate.");
            for (const auto& node : trial.node_trajectories())
                if (node.trajectory.displacement_squared(limits).compare(displacement_limit) > 0)
                    throw std::runtime_error("Exact node trajectory exceeds physical displacement CFL.");
        });
    auto next = std::make_shared<State>();
    next->phases = trial.candidate();
    execution.require(
        next->phases->mesh().get() == mesh.get(), "General VoF transport changed its immutable native mesh.");
    PhaseStorage<ArithmeticMode::Exact>::validate(*mesh, *old.phases, trial, *next->phases, options.byte_budget);
    {
        auto storage = PhaseStorage<ArithmeticMode::Exact>::evaluate(
            *resources.mesh, *next->phases, resources.material, resources.options.byte_budget);
        next->fraction = std::move(storage.fraction);
        next->density = std::move(storage.density);
        next->physical_mass = std::move(storage.mass);
    }
    std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> outgoing(volume.size()),
        source(volume.size()), capacity(volume.size());
    std::vector<detail::ConservativeVoFTransport::ExactTransfer> incoming;
    std::vector<detail::ConservativeVoFTransport::ExactFaceContribution> face_contributions;
    execution.collective(
        [&]
        {
            capacity = next->physical_mass;
            std::set<GO> seen;
            for (const auto& donor : trial.donor_budgets())
            {
                const auto local = mesh->owned_cell_map()->getLocalElement(GO(donor.donor));
                if (local < 0 || !seen.insert(GO(donor.donor)).second)
                    throw std::runtime_error("General VoF donor budget is duplicate or foreign.");
                outgoing[local] = phase_mass(donor.exact_exported);
            }
            if (seen.size() != volume.size())
                throw std::runtime_error("General VoF donor budgets do not cover native owned cells.");
            for (const auto& route : trial.transfers())
                if (route.donor != route.recipient)
                    incoming.push_back({GO(route.donor), GO(route.recipient), phase_mass(route.exact)});
            for (const auto& face : trial.face_contributions())
            {
                const auto local = mesh->owned_face_map()->getLocalElement(GO(face.face));
                if (local < 0 || (face.direction != 1 && face.direction != -1))
                    throw std::runtime_error("General VoF raw donor stream is not a canonical owned face.");
                const auto mass = phase_mass(face.exact);
                face_contributions.push_back({local, GO(face.donor), mass.scaled(face.direction)});
            }
        });
    next->velocity.resize(volume.size());
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>> momentum(volume.size());
        execution.collective(
            [&]
            {
                for (std::size_t c = 0; c < volume.size(); ++c)
                {
                    momentum[c] = old.physical_mass[c].scaled(old.velocity[c][a]);
                }
            });
        const auto carried = detail::Exact::ConservativeVoFTransport::solve(transport, momentum, source, capacity,
            outgoing, incoming, face_contributions, false, options.maximum_momentum_iterations,
            options.momentum_relative_tolerance, options.byte_budget);
        next->momentum_transport[a] = {
            carried.physical_relative_residual, carried.iterations, carried.physical_refinements};
        for (std::size_t c = 0; c < volume.size(); ++c)
            next->velocity[c][a] = carried.cell_trace[c];
    }
    std::vector<double> viscosity(volume.size());
    const auto physical_geometry = old.phases->owned_geometry();
    std::vector<SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>> old_inventory;
    execution.collective(
        [&]
        {
            old_inventory = old.phases->inventory_measures();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto weighted = old_inventory[c].gas.scaled(material.gas_viscosity) +
                                      old_inventory[c].carrier.scaled(material.liquid_viscosity);
                const auto coefficient =
                    weighted.divided_by(SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::volume(
                                            *physical_geometry[c]),
                        options.byte_budget);
                viscosity[c] = coefficient.sign() ? coefficient.certified().as_double() : 0;
            }
        });
    const auto viscous = flow.viscous_force(old.velocity, viscosity);
    std::vector<Vector> change(volume.size());
    execution.collective(
        [&]
        {
            for (std::size_t c = 0; c < volume.size(); ++c)
                for (unsigned a = 0; a < 3; ++a)
                {
                    const auto mass = next->physical_mass[c];
                    const auto viscous_change =
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(
                            viscous[c][a])
                            .scaled(dt)
                            .divided_by(mass, options.byte_budget);
                    const auto updated =
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(
                            next->velocity[c][a]) +
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(
                            options.flow.gravity[a])
                            .scaled(dt) +
                        viscous_change;
                    next->velocity[c][a] = updated.sign() ? updated.certified().as_double() : 0;
                    change[c][a] = next->velocity[c][a] - old.velocity[c][a];
                }
        });
    next->face_velocity = old.face_velocity;
    const auto face_change = flow.interpolate(change);
    for (std::size_t f = 0; f < next->face_velocity.size(); ++f)
        next->face_velocity[f] += face_change[f];
    next->curvature = Curvature<ArithmeticMode::Exact>::assemble(*resources.mesh, resources.flow, *next->phases,
        resources.material.surface_tension, resources.options.byte_budget, resources.options.capillary,
        &next->capillary);
    next->projection_arithmetic.byte_budget = options.byte_budget;
    const auto closure_topology =
        projection_topology<detail::Exact::GeneralVoFProjectionClosureTopology>(*mesh, *next->phases);
    detail::Exact::GeneralVoFFlowWorkspace::project(flow, next->face_velocity, next->velocity, next->pressure,
        next->density, next->fraction, next->curvature, dt, material.surface_tension, options.flow, next->diagnostics,
        &next->projection_arithmetic,
        std::span<const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>(next->physical_mass),
        &closure_topology);
    next->diagnostics.time = old.diagnostics.time + dt;
    execution.require(std::isfinite(next->diagnostics.time) && next->diagnostics.time > old.diagnostics.time,
        "General VoF timestep did not advance its finite clock.");
    next->diagnostics = Diagnostics<ArithmeticMode::Exact>::evaluate(*resources.mesh, *next->phases, resources.material,
        next->physical_mass, next->velocity, next->density, resources.options.byte_budget, next->diagnostics);
    // The authority verifies source identity after every flow gate passes.
    // The coordinator publishes the returned immutable candidate.
    next->phases = old.phases->apply(trial);
    return next;
}

} // namespace SimpleFluid::detail::general_vof
