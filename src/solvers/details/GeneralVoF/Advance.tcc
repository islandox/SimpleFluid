/** @file Advance.tcc
 * @brief Compiled general VoF Advance operation.
 */
#include "FVM/details/FiniteGeometryAccess.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "solvers/details/GeneralVoF/Advance.hh"
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceRates.hh"
#include "solvers/details/GeneralVoF/NodeVelocity.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include <algorithm>
#include <bit>
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
template<ArithmeticMode Mode>
double Advance<Mode>::stable_timestep(
    const Resources& resources, const State& state, const std::vector<VoF::GeneralVoFNodeVelocity>& nodes)
{
    auto& mesh = resources.mesh;
    auto& material = resources.material;
    auto& options = resources.options;
    auto& flow = resources.flow;
    auto& volume = resources.volume;
    auto& thickness = resources.thickness;
    const auto arithmetic_mode_ = resources.arithmetic_mode;
    const Execution execution(*resources.mesh);
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
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
    namespace fp = VoF::finite_detail;
    const auto rates = face_rates<Mode>(*mesh, *state.phases, state.finite_projection, flow);
    const auto geometry = state.phases->owned_finite_geometry<Mode>();
    double local_rate = std::numeric_limits<double>::infinity();
    execution.collective(
        [&]
        {
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                VoF::finite_detail::SumFor<Mode> sum(arithmetic_mode_);
                for (const auto face : mesh->faces(LO(c)))
                    sum.add(A::make_value(std::abs(rates[face].value), rates[face].absolute_error));
                const auto outgoing = sum.value();
                const auto upper_rate = outgoing.value + (bounded_mode ? outgoing.absolute_error : 0);
                if (upper_rate > 0)
                {
                    const auto lower_volume =
                        VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume()).value -
                        (bounded_mode ? VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume()).absolute_error
                                      : 0);
                    const auto time = lower_volume * options.flow.max_courant / upper_rate;
                    double value = static_cast<double>(time);
                    if (static_cast<long double>(value) > time)
                        value = std::nextafter(value, 0.);
                    if (!(value > 0) || std::isnan(value))
                        throw std::runtime_error("Finite VoF canonical rates have no positive timestep.");
                    local_rate = std::min(local_rate, value);
                }
            }
        });
    double rate = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_rate, &rate);
    return std::min(rate, speed > 0 ? std::min(bound, options.flow.max_courant * global_thickness / speed) : bound);
}
template<ArithmeticMode Mode>
std::shared_ptr<State> Advance<Mode>::prepare(Resources& resources, const State& old, double dt)
{
    auto& mesh = resources.mesh;
    auto& material = resources.material;
    auto& options = resources.options;
    auto& flow = resources.flow;
    auto& transport = resources.transport;
    auto& volume = resources.volume;
    auto& thickness = resources.thickness;
    const auto arithmetic_mode_ = resources.arithmetic_mode;
    const Execution execution(*resources.mesh);
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    namespace fp = VoF::finite_detail;
    execution.require(std::isfinite(dt) && dt > 0, "Finite VoF timestep must be positive and finite.");
    double minimum = 0, maximum_dt = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &dt, &minimum);
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &dt, &maximum_dt);
    execution.require(minimum == maximum_dt, "Finite VoF timestep differs across ranks.");
    const bool face_backend =
        old.phases->options().transport_backend != VoF::GeneralVoFTransportBackend::SweptPartition;
    const bool homogeneous = !face_backend && old.phases->homogeneous_transport_eligible();
    const bool algebraic = face_backend || homogeneous;
    const auto nodes = algebraic ? std::vector<VoF::GeneralVoFNodeVelocity>{}
                                 : node_velocity(*mesh, *old.phases, old.velocity, options.byte_budget);
    execution.require(
        dt <= stable_timestep(resources, old, nodes), "Finite VoF timestep exceeds physical geometry/trajectory CFL.");
    const auto controls = face_volumes<Mode>(*mesh, *old.phases, old.finite_projection, flow, dt);
    const auto control_view = old.phases->transport_controls<Mode>(nodes, controls, dt);
    const auto trial =
        face_backend  ? old.phases->plan_face_transport(control_view, {material.gas_density, material.liquid_density},
                            std::bit_cast<std::uint64_t>(old.diagnostics.time))
        : homogeneous ? old.phases->plan_homogeneous_transport(control_view)
                      : old.phases->plan_transport(control_view);
    const auto stage = std::bit_cast<std::uint64_t>(old.diagnostics.time);
    const auto prepared_view = trial.template prepare_transport_view<Mode>(
        {material.gas_density, material.liquid_density}, stage);
    trial.validate_prepared_transport(prepared_view, {material.gas_density, material.liquid_density}, stage);
    const auto& phase_preparation = prepared_view.shared_ledger();
    if (!algebraic)
    {
        double local_h = std::numeric_limits<double>::infinity(), global_h = 0;
        for (auto h : thickness)
            local_h = std::min(local_h, h);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_h, &global_h);
        execution.collective(
            [&]
            {
                if (phase_preparation->node_trajectories.size() != nodes.size())
                    throw std::runtime_error("Finite VoF trajectory certificate is incomplete.");
                const auto limit = A::multiply(A::make_value(global_h, 0), A::make_value(options.flow.max_courant, 0));
                const auto square_limit = A::multiply(limit, limit);
                for (const auto& node : phase_preparation->node_trajectories)
                {
                    const auto displacement = A::subtract(node.point, node.departure);
                    const auto square = A::dot(displacement, displacement);
                    if (square.value + (bounded_mode ? square.absolute_error : 0) >
                        square_limit.value - (bounded_mode ? square_limit.absolute_error : 0))
                        throw std::runtime_error("Finite VoF bounded trajectory exceeds physical displacement CFL.");
                }
            });
    }
    auto next = std::make_shared<State>();
    next->phases = trial.candidate();
    execution.require(
        next->phases->mesh().get() == mesh.get(), "Finite VoF transport changed its immutable native mesh.");
    PhaseStorage<Mode>::validate(*mesh, *old.phases, *phase_preparation, *next->phases);
    {
        auto storage = PhaseStorage<Mode>::evaluate(*resources.mesh, *next->phases, resources.material);
        next->fraction = std::move(storage.fraction);
        next->density = std::move(storage.density);
        next->finite_mass = std::move(storage.mass);
        next->finite_mass_error = std::move(storage.mass_error);
        next->physical_mass.clear();
    }
    execution.collective([&]
    {
        if (phase_preparation->old_mass != old.finite_mass || phase_preparation->new_mass != next->finite_mass ||
            phase_preparation->old_mass_error != old.finite_mass_error ||
            phase_preparation->new_mass_error != next->finite_mass_error)
            throw std::runtime_error("Prepared phase/momentum mass capacities differ.");
    });
    std::vector<long double> outgoing(volume.size()), outgoing_error(volume.size()), source(volume.size());
    std::vector<detail::ConservativeVoFTransport::FiniteTransfer> incoming;
    std::vector<detail::ConservativeVoFTransport::FiniteFaceContribution> face_contributions;
    const auto mass = [&](const PLIC::DynamicPhaseMeasures& p)
    {
        return A::add(A::multiply(p.gas, A::make_value(phase_preparation->density[0], 0)),
            A::multiply(p.carrier, A::make_value(phase_preparation->density[1], 0)));
    };
    execution.collective(
        [&]
        {
            const bool signed_family = phase_preparation->family == VoF::GeneralVoFTransportFamily::InterfaceLocalSwept;
            if (signed_family && Mode != ArithmeticMode::Ordinary)
                throw std::runtime_error("Signed momentum transport requires Ordinary arithmetic.");
            const auto& budgets =
                signed_family ? phase_preparation->numerical_support_budgets : phase_preparation->donor_budgets;
            const auto& transfers =
                signed_family ? phase_preparation->numerical_support_transfers : phase_preparation->positive_transfers;
            std::set<GO> seen;
            for (const auto& donor : budgets)
            {
                const auto row = mesh->owned_cell_map()->getLocalElement(GO(donor.donor));
                if (row < 0 || !seen.insert(GO(donor.donor)).second)
                    throw std::runtime_error("Finite VoF donor budget is duplicate or foreign.");
                const auto exported_mass = mass(donor.exported);
                outgoing[row] = exported_mass.value;
                outgoing_error[row] = exported_mass.absolute_error;
            }
            if (seen.size() != volume.size())
                throw std::runtime_error("Finite VoF donor budgets are incomplete.");
            for (const auto& route : transfers)
                if (route.donor != route.recipient)
                {
                    const auto value = mass(route.amount);
                    incoming.push_back({GO(route.donor), GO(route.recipient), value.value, value.absolute_error});
                }
            for (const auto& face : phase_preparation->signed_face_contributions)
            {
                const auto row = mesh->owned_face_map()->getLocalElement(GO(face.face));
                if (row < 0 || (face.direction != 1 && face.direction != -1))
                    throw std::runtime_error("Finite VoF raw stream is not a canonical owned face.");
                const auto value = mass(face.amount);
                face_contributions.push_back({row, GO(face.donor), face.direction * value.value, value.absolute_error});
            }
        });
    old.phases->validate_prepared_transport_source(*phase_preparation);
    const auto preparation_before = transport.preparation_statistics();
    next->velocity.resize(volume.size());
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<long double> momentum(volume.size()), momentum_error(volume.size());
        for (std::size_t c = 0; c < volume.size(); ++c)
        {
            const auto value = A::multiply(
                A::make_value(phase_preparation->old_mass[c], phase_preparation->old_mass_error[c]),
                A::make_value(old.velocity[c][a], 0));
            momentum[c] = value.value;
            momentum_error[c] = value.absolute_error;
        }
        const auto carried =
            transport.solve(momentum, source, phase_preparation->new_mass, outgoing, incoming, face_contributions,
                false, options.maximum_momentum_iterations, options.momentum_relative_tolerance, options.byte_budget,
                momentum_error, {}, phase_preparation->new_mass_error, outgoing_error, arithmetic_mode_,
                phase_preparation->family == VoF::GeneralVoFTransportFamily::InterfaceLocalSwept
                    ? detail::ConservativeVoFTransport::FiniteOperatorMode::SignedFaceLedger
                    : detail::ConservativeVoFTransport::FiniteOperatorMode::EquivalentRoutes);
        next->momentum_transport[a] = {
            carried.physical_relative_residual, carried.iterations, carried.physical_refinements};
        for (std::size_t c = 0; c < volume.size(); ++c)
            next->velocity[c][a] = carried.cell_trace[c];
    }
    std::vector<double> viscosity(volume.size());
    execution.collective(
        [&]
        {
            const auto& phases = old.phases->phase_volumes();
            const auto geometry = old.phases->owned_finite_geometry<Mode>();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto weighted =
                    A::add(A::multiply(A::make_value(phases[c].gas, 0), A::make_value(material.gas_viscosity, 0)),
                        A::multiply(A::make_value(phases[c].carrier, 0), A::make_value(material.liquid_viscosity, 0)));
                viscosity[c] =
                    A::as_double(A::divide(weighted, VoF::finite_detail::runtime_value<Mode>(geometry[c]->volume())));
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
                    const auto force = A::divide(A::multiply(A::make_value(viscous[c][a], 0), A::make_value(dt, 0)),
                        A::make_value(next->finite_mass[c], next->finite_mass_error[c]));
                    const auto updated =
                        A::add(A::add(A::make_value(next->velocity[c][a], 0),
                                   A::multiply(A::make_value(options.flow.gravity[a], 0), A::make_value(dt, 0))),
                            force);
                    next->velocity[c][a] = A::as_double(updated);
                    change[c][a] = next->velocity[c][a] - old.velocity[c][a];
                }
        });
    next->face_velocity = old.face_velocity;
    const auto face_change = flow.interpolate(change);
    for (std::size_t f = 0; f < next->face_velocity.size(); ++f)
        next->face_velocity[f] += face_change[f];
    next->curvature = Curvature<Mode>::assemble(*resources.mesh, resources.flow, *next->phases,
        resources.material.surface_tension, resources.options.byte_budget, resources.options.capillary,
        &next->capillary);
    next->finite_projection.byte_budget = options.byte_budget;
    const auto closure_topology =
        projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*mesh, *next->phases);
    flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction, next->curvature,
        dt, material.surface_tension, options.flow, next->diagnostics, &next->finite_projection,
        std::span<const long double>(next->finite_mass), &closure_topology,
        std::span<const long double>(next->finite_mass_error));
    next->diagnostics.time = old.diagnostics.time + dt;
    execution.require(std::isfinite(next->diagnostics.time) && next->diagnostics.time > old.diagnostics.time,
        "Finite VoF timestep did not advance the accepted clock.");
    next->diagnostics = Diagnostics<Mode>::evaluate(
        *resources.mesh, *next->phases, resources.material, next->finite_mass, next->velocity, next->diagnostics);
    next->phases = old.phases->apply(trial);
    execution.collective([&]
    {
        phase_preparation->validate_byte_budget(options.byte_budget);
        auto prepared = std::make_shared<VoF::GeneralVoFPreparedTransport>(*phase_preparation);
        const auto after = transport.preparation_statistics();
        prepared->momentum_preparation = transport.finite_preparation_identity();
        prepared->momentum_builds = after.finite_builds - preparation_before.finite_builds;
        prepared->momentum_reuses = after.finite_hits - preparation_before.finite_hits;
        next->prepared_transport = std::move(prepared);
    });
    return next;
}

} // namespace SimpleFluid::detail::general_vof
