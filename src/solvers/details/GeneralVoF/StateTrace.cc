/** @file StateTrace.cc
 * @brief Lazy finite state trace emission without accepted-state ownership.
 */
#include "solvers/details/GeneralVoF/StateTrace.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/GeneralVoF/FlowTrace.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include <algorithm>
#include <cmath>

namespace SimpleFluid::detail::general_vof
{
void trace_finite_state(const MeshHandle<>& mesh_view, const char* stage, double accepted_time, double dt,
    std::span<const VoF::Vector> accepted_velocity, const ConservativeVoFTransport& transport,
    const StateTraceView& state)
{
    using LO = DefaultTpetraTypes::local_ordinal_type;
    const auto* mesh = &mesh_view;
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
                    const auto delta = state.velocity[c][a] - accepted_velocity[c][a];
                    delta_square += delta * delta;
                }
                energy += .5L * state.finite_mass[c] * square;
                maximum = std::max(maximum, std::sqrt(square));
                change = std::max(change, std::sqrt(delta_square));
            }
            for (auto value : state.face_velocity)
                face_maximum = std::max(face_maximum, std::abs(value));
            out << ",\"accepted_time\":" << accepted_time << ",\"dt\":" << dt << ",\"local_kinetic_energy\":" << energy
                << ",\"local_maximum_cell_velocity\":" << maximum << ",\"local_maximum_face_velocity\":" << face_maximum
                << ",\"local_velocity_change\":" << change
                << ",\"pressure_iterations\":" << state.diagnostics.pressure_iterations
                << ",\"physical_pressure_residual\":" << state.diagnostics.pressure_residual
                << ",\"maximum_divergence\":" << state.diagnostics.maximum_divergence;
            for (unsigned a = 0; a < 3; ++a)
                out << ",\"momentum_residual_" << a << "\":" << state.momentum_transport[a].physical_relative_residual;
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
                    out << (c ? ",[" : "[") << mesh->cell_global_id(LO(c)) << ',' << center.x << ',' << center.y << ','
                        << center.z << ','
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
                        << state.finite_projection.face_rate[f] << ',' << state.finite_projection.face_radius[f] << ']';
                out << ']';
            }
        });
}
} // namespace SimpleFluid::detail::general_vof
