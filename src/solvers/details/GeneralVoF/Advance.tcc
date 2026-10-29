/** @file Advance.tcc
 * @brief Compiled general VoF Advance operation.
 */
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
#include "solvers/details/GeneralVoF/Advance.hh"
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Diagnostics.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceRates.hh"
#include "solvers/details/GeneralVoF/FlowTrace.hh"
#include "solvers/details/GeneralVoF/NodeVelocity.hh"
#include "solvers/details/GeneralVoF/OccupancyReadiness.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include "utils/CompensatedSum.hh"
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
    bool algebraic = false;
    if constexpr (Mode == ArithmeticMode::Ordinary)
        algebraic = state.phases->options().transport_backend != VoF::GeneralVoFTransportBackend::SweptPartition ||
                    state.phases->homogeneous_transport_eligible();
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
                VoF::finite_detail::SumFor<Mode> net(arithmetic_mode_);
                for (const auto face : mesh->faces(LO(c)))
                {
                    sum.add(A::make_value(std::abs(rates[face].value), rates[face].absolute_error));
                    if (algebraic)
                        net.add(mesh->owner_cell(face) == LO(c) ? rates[face] : A::negate(rates[face]));
                }
                const auto outgoing = sum.value();
                const auto upper_rate = outgoing.value + (bounded_mode ? outgoing.absolute_error : 0);
                flow_trace(mesh->owned_cell_map()->getComm()->getRank(), "canonical_timestep",
                    [&](auto& out)
                    {
                        const auto cell_volume =
                            SimpleFluid::geometry::swept_volume::runtime_value<Mode>(geometry[c]->volume()).value;
                        out << ",\"time\":" << state.diagnostics.time << ",\"cell\":" << mesh->cell_global_id(LO(c))
                            << ",\"volume\":" << cell_volume << ",\"net_rate\":" << net.value().value
                            << ",\"incident_absolute_rate\":" << upper_rate
                            << ",\"canonical_cfl_inverse\":" << upper_rate / (cell_volume * options.flow.max_courant)
                            << ",\"continuity_inverse\":"
                            << (algebraic ? 2.L * std::abs(net.value().value) /
                                                (cell_volume * state.phases->options().reconstruction.volume_tolerance)
                                          : 0);
                    });
                if (upper_rate > 0)
                {
                    const auto lower_volume =
                        SimpleFluid::geometry::swept_volume::runtime_value<Mode>(geometry[c]->volume()).value -
                        (bounded_mode ? SimpleFluid::geometry::swept_volume::runtime_value<Mode>(geometry[c]->volume()).absolute_error
                                      : 0);
                    const auto time = lower_volume * options.flow.max_courant / upper_rate;
                    double value = static_cast<double>(time);
                    if (static_cast<long double>(value) > time)
                        value = std::nextafter(value, 0.);
                    if (!(value > 0) || std::isnan(value))
                        throw std::runtime_error("Finite VoF canonical rates have no positive timestep.");
                    local_rate = std::min(local_rate, value);
                }
                if (algebraic && net.value().value != 0)
                {
                    // Accepted pressure rounding may leave a nonzero canonical
                    // rate defect. Bound its integrated volume at the future
                    // step, independently of CFL and the physical projection
                    // gate; a zero net rate imposes no additional restriction.
                    const auto cell_volume =
                        SimpleFluid::geometry::swept_volume::runtime_value<Mode>(geometry[c]->volume()).value;
                    const auto time = cell_volume * state.phases->options().reconstruction.volume_tolerance /
                                      (2.L * std::abs(net.value().value));
                    double value = static_cast<double>(time);
                    if (static_cast<long double>(value) > time)
                        value = std::nextafter(value, 0.);
                    if (!(value > 0) || std::isnan(value))
                        throw std::runtime_error("Finite VoF canonical continuity has no positive timestep.");
                    local_rate = std::min(local_rate, value);
                }
            }
        });
    double rate = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_rate, &rate);
    const auto selected =
        std::min(rate, speed > 0 ? std::min(bound, options.flow.max_courant * global_thickness / speed) : bound);
    flow_trace(mesh->owned_cell_map()->getComm()->getRank(), "selected_timestep",
        [&](auto& out)
        {
            out << ",\"time\":" << state.diagnostics.time << ",\"selected_inverse\":" << 1 / selected
                << ",\"trajectory_inverse\":" << speed / (options.flow.max_courant * global_thickness)
                << ",\"physical_inverse\":" << 1 / bound << ",\"canonical_inverse\":" << 1 / rate;
        });
    return selected;
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
    const auto trace_state = [&](const char* stage, const State& state)
    {
        flow_trace(mesh->owned_cell_map()->getComm()->getRank(), stage,
            [&](auto& out)
            {
                long double energy = 0;
                double maximum = 0, face_maximum = 0, change = 0;
                for (std::size_t c = 0; c < state.velocity.size(); ++c)
                {
                    double square = 0, delta_square = 0;
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        square += state.velocity[c][a] * state.velocity[c][a];
                        const auto delta = state.velocity[c][a] - old.velocity[c][a];
                        delta_square += delta * delta;
                    }
                    energy += .5L * state.finite_mass[c] * square;
                    maximum = std::max(maximum, std::sqrt(square));
                    change = std::max(change, std::sqrt(delta_square));
                }
                for (auto value : state.face_velocity)
                    face_maximum = std::max(face_maximum, std::abs(value));
                out << ",\"accepted_time\":" << old.diagnostics.time << ",\"dt\":" << dt
                    << ",\"local_kinetic_energy\":" << energy << ",\"local_maximum_cell_velocity\":" << maximum
                    << ",\"local_maximum_face_velocity\":" << face_maximum << ",\"local_velocity_change\":" << change
                    << ",\"pressure_iterations\":" << state.diagnostics.pressure_iterations
                    << ",\"physical_pressure_residual\":" << state.diagnostics.pressure_residual
                    << ",\"maximum_divergence\":" << state.diagnostics.maximum_divergence;
                for (unsigned a = 0; a < 3; ++a)
                    out << ",\"momentum_residual_" << a
                        << "\":" << state.momentum_transport[a].physical_relative_residual;
                const auto stats = transport.preparation_statistics();
                out << ",\"momentum_operator_builds\":" << stats.finite_builds
                    << ",\"momentum_operator_reuses\":" << stats.finite_hits;
                const auto& phases = state.phases->phase_volumes();
                if (phases.size() == state.velocity.size())
                {
                    const auto lease = mesh->acquire_execution_view();
                    out << ",\"cell_columns\":[\"gid\",\"x\",\"y\",\"z\",\"geometric_volume\",\"gas\","
                           "\"carrier\",\"ux\",\"uy\",\"uz\",\"pressure\",\"gas_history\",\"carrier_history\","
                           "\"declaration\",\"native_bulk\"],\"cells\":[";
                    const auto& readiness = state.finite_projection.transport_readiness;
                    for (std::size_t c = 0; c < phases.size(); ++c)
                    {
                        const auto center = mesh->cell_centroid(LO(c));
                        const auto& history = state.phases->accumulated_phase_errors()[c];
                        out << (c ? ",[" : "[") << mesh->cell_global_id(LO(c)) << ',' << center.x << ',' << center.y
                            << ',' << center.z << ','
                            << (readiness && readiness->cumulative ? readiness->volume[c] : phases[c].bulk) << ','
                            << phases[c].gas << ',' << phases[c].carrier << ',' << state.velocity[c][0] << ','
                            << state.velocity[c][1] << ',' << state.velocity[c][2] << ',';
                        if (state.pressure.size() == phases.size())
                            out << state.pressure[c];
                        else
                            out << "null";
                        out << ',' << history.gas << ',' << history.carrier << ','
                            << int(state.phases->phase_geometry_declarations()[c]) << ',' << phases[c].bulk << ']';
                    }
                    out << "],\"canonical_rate_columns\":[\"gid\",\"rate\",\"radius\"],\"canonical_rates\":[";
                    for (std::size_t f = 0; f < state.finite_projection.face_rate.size(); ++f)
                        out << (f ? ",[" : "[") << mesh->face_global_id(LO(f)) << ','
                            << state.finite_projection.face_rate[f] << ',' << state.finite_projection.face_radius[f]
                            << ']';
                    out << ']';
                }
            });
    };
    trace_state("accepted_state", old);
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
    const bool cleanup_enabled = old.phases->options().phase_cleanup.relative_cutoff > 0;
    std::vector<std::array<long double, 3>> cleanup_momentum;
    if (cleanup_enabled)
        execution.collective(
            [&]
            {
                if (volume.size() > options.byte_budget / sizeof(std::array<long double, 3>))
                    throw std::runtime_error("Finite VoF cleanup momentum exceeds its retained byte budget.");
                phase_preparation->validate_byte_budget(
                    options.byte_budget - volume.size() * sizeof(std::array<long double, 3>));
                cleanup_momentum.resize(volume.size());
            });
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
            transport.solve(momentum, source, phase_preparation->raw_new_mass, outgoing, incoming, face_contributions,
                false, options.maximum_momentum_iterations, options.momentum_relative_tolerance, options.byte_budget,
                momentum_error, {}, phase_preparation->raw_new_mass_error, outgoing_error, arithmetic_mode_,
                phase_preparation->family == VoF::GeneralVoFTransportFamily::InterfaceLocalSwept
                    ? detail::ConservativeVoFTransport::FiniteOperatorMode::SignedFaceLedger
                    : detail::ConservativeVoFTransport::FiniteOperatorMode::EquivalentRoutes);
        next->momentum_transport[a] = {
            carried.physical_relative_residual, carried.iterations, carried.physical_refinements};
        if (!cleanup_enabled)
        {
            for (std::size_t c = 0; c < volume.size(); ++c)
                next->velocity[c][a] = carried.cell_trace[c];
            continue;
        }
        VoF::finite_detail::SumFor<Mode> cleanup_sum(arithmetic_mode_);
        int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
        for (const auto value : momentum)
            if (value != 0)
            {
                int item_exponent = 0;
                (void) std::frexp(std::abs(value), &item_exponent);
                local_exponent = std::max(local_exponent, item_exponent);
            }
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
        const auto reference_scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
        CompensatedSum<> identity_squared, reference_squared;
        execution.collective(
            [&]
            {
                for (std::size_t c = 0; c < volume.size(); ++c)
                {
                    next->velocity[c][a] = carried.cell_trace[c];
                    // Transport uses the unmodified raw mass/face equation.
                    // Relabeling then changes mass and momentum together at
                    // this velocity; forces and projection see cleaned mass.
                    const auto delta = A::multiply(
                        A::make_value(phase_preparation->cleanup_mass[c], 0), A::make_value(carried.cell_trace[c], 0));
                    cleanup_momentum[c][a] = delta.value;
                    cleanup_sum.add(delta);
                    if (!std::isfinite(delta.value))
                        throw std::runtime_error("Finite VoF cleanup momentum is not representable.");
                    const auto velocity = A::make_value(carried.cell_trace[c], 0);
                    const auto cleaned = A::multiply(A::make_value(phase_preparation->new_mass[c], 0), velocity);
                    const auto raw = A::multiply(A::make_value(phase_preparation->raw_new_mass[c], 0), velocity);
                    const auto defect = A::subtract(A::subtract(cleaned, raw), delta).value;
                    if (!std::isfinite(cleaned.value) || !std::isfinite(raw.value) || !std::isfinite(defect))
                        throw std::runtime_error("Finite VoF cleaned/raw momentum identity is not representable.");
                    // Audit each declared local correction on the original
                    // momentum RHS scale. Neither correction magnitude nor a
                    // new numerical allowance enters the physical tolerance.
                    const auto normalized = defect / reference_scale;
                    const auto reference = momentum[c] / reference_scale;
                    identity_squared += normalized * normalized;
                    reference_squared += reference * reference;
                }
            });
        const auto local_cleanup = cleanup_sum.value().value;
        // Teuchos does not provide a long-double serializer on every supported
        // backend. Preserve the finite wide value as two binary64 mantissas
        // and an exact exponent, then accumulate all owners in rank order.
        const auto ranks = mesh->owned_cell_map()->getComm()->getSize();
        std::array<double, 3> cleanup_wire{};
        std::vector<double> cleanup_owners;
        execution.collective(
            [&]
            {
                if (!std::isfinite(local_cleanup) || ranks > std::numeric_limits<int>::max() / 3 ||
                    std::size_t(ranks) > options.byte_budget / (3 * sizeof(double)))
                    throw std::runtime_error("Finite VoF cleanup momentum reduction exceeds its admitted range.");
                int cleanup_exponent = 0;
                const auto mantissa = std::frexp(local_cleanup, &cleanup_exponent);
                cleanup_wire[0] = static_cast<double>(mantissa);
                cleanup_wire[1] = static_cast<double>(mantissa - cleanup_wire[0]);
                cleanup_wire[2] = cleanup_exponent;
                if (std::ldexp(static_cast<long double>(cleanup_wire[0]) + cleanup_wire[1], cleanup_exponent) !=
                    local_cleanup)
                    throw std::runtime_error("Finite VoF cleanup momentum wire conversion loses precision.");
                cleanup_owners.resize(3 * std::size_t(ranks));
            });
        Teuchos::gatherAll(
            *mesh->owned_cell_map()->getComm(), 3, cleanup_wire.data(), 3 * ranks, cleanup_owners.data());
        CompensatedSum<> cleanup_global_sum;
        for (int rank = 0; rank < ranks; ++rank)
        {
            const auto offset = 3 * std::size_t(rank);
            cleanup_global_sum +=
                std::ldexp(static_cast<long double>(cleanup_owners[offset]) + cleanup_owners[offset + 1],
                    int(cleanup_owners[offset + 2]));
        }
        const auto global_cleanup = cleanup_global_sum.value();
        const std::array<double, 2> local_norm{
            static_cast<double>(identity_squared.value()), static_cast<double>(reference_squared.value())};
        std::array<double, 2> global_norm{};
        Teuchos::reduceAll(
            *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 2, local_norm.data(), global_norm.data());
        execution.collective(
            [&]
            {
                auto& diagnostic = next->momentum_transport[a];
                diagnostic.cleanup_momentum = global_cleanup;
                diagnostic.accumulated_cleanup_momentum =
                    old.momentum_transport[a].accumulated_cleanup_momentum + global_cleanup;
                const auto reduction_u =
                    (mesh->owned_cell_map()->getComm()->getSize() + 8.L) * std::numeric_limits<double>::epsilon();
                if (!(reduction_u < .25L))
                    throw std::runtime_error("Finite VoF cleanup identity reduction bound is exhausted.");
                const auto residual_upper = global_norm[0] * (1 + reduction_u);
                const auto reference_lower = std::max(0.L, global_norm[1] * (1 - reduction_u));
                diagnostic.cleanup_identity_relative_residual =
                    std::sqrt(residual_upper / (reference_lower > 0 ? reference_lower : 1));
                if (!std::isfinite(global_cleanup) || !std::isfinite(diagnostic.accumulated_cleanup_momentum))
                    throw std::runtime_error("Finite VoF cumulative cleanup momentum is not representable.");
                // The triangle bound keeps the cleaned equation within the
                // same original tolerance after its explicit momentum source.
                if (!std::isfinite(global_norm[0]) || !std::isfinite(global_norm[1]) ||
                    !std::isfinite(diagnostic.cleanup_identity_relative_residual) ||
                    (global_norm[1] == 0 && global_norm[0] != 0) ||
                    diagnostic.physical_relative_residual + diagnostic.cleanup_identity_relative_residual >
                        options.momentum_relative_tolerance)
                    throw std::runtime_error("Finite VoF declared cleanup exceeds the original momentum residual.");
            });
    }
    trace_state("after_momentum_transport", *next);
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
                    A::as_double(A::divide(weighted, SimpleFluid::geometry::swept_volume::runtime_value<Mode>(geometry[c]->volume())));
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
    trace_state("before_projection", *next);
    next->curvature = Curvature<Mode>::assemble(*resources.mesh, resources.flow, *next->phases,
        resources.material.surface_tension, resources.options.byte_budget, resources.options.capillary,
        &next->capillary);
    next->finite_projection.byte_budget = options.byte_budget;
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        // Publish rates ready for the next state's transport family; pure
        // declarations are authenticated collectively, including empty owners.
        if (face_backend || next->phases->homogeneous_transport_eligible())
            next->finite_projection.transport_readiness =
                occupancy_readiness(*next->phases, options.flow.max_courant, options.byte_budget);
    }
    const auto closure_topology =
        projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*mesh, *next->phases);
    flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction, next->curvature,
        dt, material.surface_tension, options.flow, next->diagnostics, &next->finite_projection,
        std::span<const long double>(next->finite_mass), &closure_topology,
        std::span<const long double>(next->finite_mass_error));
    trace_state("after_projection", *next);
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
        if (cleanup_enabled)
        {
            prepared->cleanup_momentum = std::move(cleanup_momentum);
            prepared->validate_byte_budget(options.byte_budget);
        }
        next->prepared_transport = std::move(prepared);
    });
    return next;
}

} // namespace SimpleFluid::detail::general_vof
