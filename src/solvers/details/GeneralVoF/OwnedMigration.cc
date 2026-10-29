/** @file OwnedMigration.cc
 * @brief Lossless keyed migration of an Ordinary accepted solver publication.
 */
#include "solvers/details/GeneralVoF/OwnedMigration.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/OwnedFaceTransfer.hh"
#include "solvers/details/GeneralVoF/PhaseStorage.hh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
namespace SimpleFluid::detail::general_vof
{
namespace
{
using Key = VoF::GeneralVoF::Key;
using Bytes = std::vector<char>;
struct Budget
{
    std::size_t limit = 0, used = 0;
    void add(std::size_t count, std::size_t width = 1)
    {
        if (width && count > (limit - used) / width)
            throw std::length_error("Ownership migration exceeds its cumulative byte budget.");
        used += count * width;
    }
};
void admit_key(Budget& budget, const Key& key, std::size_t copies = 1)
{
    budget.add(copies, sizeof(Key));
    std::visit(
        [&](const auto& ancestry)
        {
            if constexpr (requires { ancestry.path; })
                for (std::size_t copy = 0; copy < copies; ++copy)
                    budget.add(ancestry.path.size(), sizeof(std::uint64_t));
            else if constexpr (requires {
                                   ancestry.base;
                                   ancestry.axial;
                               })
                for (std::size_t copy = 0; copy < copies; ++copy)
                {
                    budget.add(ancestry.base.path.size(), sizeof(std::uint64_t));
                    budget.add(ancestry.axial.path.size(), sizeof(std::uint64_t));
                }
        },
        key.ancestry);
}

void word(Bytes& b, std::uint64_t v)
{
    append_face_word(b, v);
}
std::uint64_t word(const Bytes& b, std::size_t& p)
{
    const auto v = read_face_word(b, p);
    p += 8;
    return v;
}
void real(Bytes& b, double v)
{
    if (!std::isfinite(v))
        throw std::runtime_error("Ownership state contains nonfinite binary64.");
    word(b, std::bit_cast<std::uint64_t>(v));
}
double real(const Bytes& b, std::size_t& p)
{
    const auto v = std::bit_cast<double>(word(b, p));
    if (!std::isfinite(v))
        throw std::runtime_error("Ownership state contains nonfinite binary64.");
    return v;
}
void wide(Bytes& b, long double v)
{
    int exponent = 0;
    const auto mantissa = std::frexp(v, &exponent);
    const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
    if (!std::isfinite(v) || static_cast<long double>(high) + low != mantissa)
        throw std::runtime_error("Ownership state exceeds finite wide scalar codec.");
    real(b, high);
    real(b, low);
    word(b, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
}
long double wide(const Bytes& b, std::size_t& p)
{
    const auto high = real(b, p), low = real(b, p);
    const auto exponent = std::bit_cast<std::int64_t>(word(b, p));
    if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
        throw std::runtime_error("Ownership wide exponent is malformed.");
    const auto mantissa = high == 0 && low == 0 ? std::copysign(0.L, high) : static_cast<long double>(high) + low;
    const auto v = std::scalbn(mantissa, int(exponent));
    if (!std::isfinite(v))
        throw std::runtime_error("Ownership wide value is nonfinite.");
    return v;
}
void bytes(Bytes& b, const Bytes& v)
{
    word(b, v.size());
    b.insert(b.end(), v.begin(), v.end());
}
Bytes bytes(const Bytes& b, std::size_t& p)
{
    const auto n = word(b, p);
    if (p > b.size() || n > b.size() - p)
        throw std::runtime_error("Ownership keyed packet is truncated.");
    Bytes v(b.begin() + p, b.begin() + p + n);
    p += n;
    return v;
}
Bytes phase_record(
    const VoF::GeneralVoF& phases, std::size_t c, const std::vector<VoF::GeneralVoFPhaseGeometry>& declarations)
{
    Bytes b;
    const auto v = phases.phase_volumes().at(c);
    for (double x : {v.bulk, v.gas, v.carrier})
        real(b, x);
    for (const auto* values : {&phases.phase_roundoff_bounds(), &phases.accumulated_phase_errors()})
    {
        real(b, values->at(c).gas);
        real(b, values->at(c).carrier);
    }
    const auto e = phases.initial_occupancy_errors().at(c);
    for (double x : {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound, e.fit_residual,
             e.fit_bound, e.transferred_bound, e.fit_allocation_bound})
        real(b, x);
    word(b, std::uint64_t(declarations.at(c)));
    return b;
}
} // namespace
PreparedRemap prepare_owned_migration(
    const Resources& resources, const State& old, const VoF::GeneralVoF::Hierarchy::Plan& plan)
{
    const Execution execution(*resources.mesh);
    const auto source = old.phases;
    const auto hierarchy = source->hierarchy();
    const auto comm = hierarchy->communicator();
    const auto limit = resources.options.byte_budget;
    execution.require(resources.arithmetic_mode == ArithmeticMode::Ordinary && source->accepted_unchanged(),
        "Ownership migration requires an unchanged Ordinary publication.");
    const auto raw = hierarchy->apply(plan);
    const auto native = Meshes::AMR::assemble_adaptive_native(*raw, raw->configuration_descriptor());
    const auto bound = raw->with_partition(native.partition, native.mesh->owned_cell_map());
    const auto target_hierarchy = hierarchy->apply(plan, bound);
    VoF::GeneralVoFFiniteRemapOptions options;
    options.byte_budget = limit;
    const auto trial = source->plan_remap(plan, target_hierarchy, {}, options);
    trial.validate_observations();
    execution.require(trial.ownership_only(), "Ownership state transfer received an adaptation.");
    const auto phases = trial.candidate();
    std::unique_ptr<Resources> candidate;
    std::shared_ptr<State> state;
    execution.collective(
        [&]
        {
            if (source->phase_cleanup_checkpoint() != phases->phase_cleanup_checkpoint() ||
                !phases->latest_phase_cleanup().empty())
                throw std::runtime_error("Ownership migration changes cleanup accounting.");
        });
    candidate = std::make_unique<Resources>(phases, resources.material, resources.options);
    int local_optional[2] = {!old.finite_projection.velocity_error.empty(), !old.finite_projection.cell_radius.empty()};
    int optional[2] = {};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 2, local_optional, optional);
    Budget retained{limit};
    execution.collective(
        [&]
        {
            // Reserve the complete new accepted cell arrays before any allocation.
            retained.add(1, sizeof(State));
            const auto n = phases->owned_order().size();
            retained.add(n, sizeof(VoF::Vector) + 4 * sizeof(double) + 2 * sizeof(long double));
            if (optional[0])
                retained.add(n, sizeof(std::array<long double, 3>));
            if (optional[1])
                retained.add(n, sizeof(long double));
            if (limit - retained.used < 8)
                throw std::length_error("Ownership migration has no transient packet lease.");
            state = std::make_shared<State>();
            state->phases = phases;
        });
    const auto available = limit - retained.used;
    const auto packet_limit = available / 8;
    {
        // Separate simultaneous leases: indexes/decoded keys (1/4), local packets
        // and each retained exchange (1/8 each), owner lookup (1/8).
        Budget cell_resident{available / 4};
        execution.collective(
            [&]
            {
                cell_resident.add(source->owned_order().size(), sizeof(VoF::GeneralVoFPhaseGeometry));
                cell_resident.add(phases->owned_order().size(), sizeof(VoF::GeneralVoFPhaseGeometry));
                for (const auto& key : source->owned_order())
                {
                    admit_key(cell_resident, key);
                    cell_resident.add(1, sizeof(std::size_t) + 4 * sizeof(void*));
                }
                for (const auto& key : phases->owned_order())
                {
                    // Map key, owner-directory key, temporary encoding and seen key.
                    admit_key(cell_resident, key, 4);
                    cell_resident.add(1, sizeof(std::size_t) + sizeof(Meshes::AMR::AdaptiveOwner) + 8 * sizeof(void*));
                }
            });
        const auto owners = hierarchy->lookup_cells(
            phases->owned_order(), true, std::min(packet_limit, hierarchy->options().byte_budget));

        std::map<Key, std::size_t> source_rows, target_rows;
        std::vector<VoF::GeneralVoFPhaseGeometry> source_declarations, target_declarations;
        std::vector<OwnerPacket> packets;
        execution.collective(
            [&]
            {
                const auto n = source->owned_order().size();
                if (old.velocity.size() != n || old.pressure.size() != n || old.curvature.size() != n ||
                    old.fraction.size() != n || old.density.size() != n || old.finite_mass.size() != n ||
                    old.finite_mass_error.size() != n ||
                    (optional[0] && old.finite_projection.velocity_error.size() != n) ||
                    (optional[1] && old.finite_projection.cell_radius.size() != n))
                    throw std::runtime_error("Ownership accepted cell fields have incomplete shapes.");
                source_declarations = source->phase_geometry_declarations();
                target_declarations = phases->phase_geometry_declarations();
                for (std::size_t c = 0; c < n; ++c)
                    source_rows.emplace(source->owned_order()[c], c);
                Budget packet_admission{packet_limit};
                packet_admission.add(phases->owned_order().size(), 2048);
                packets.reserve(phases->owned_order().size());
                for (std::size_t c = 0; c < phases->owned_order().size(); ++c)
                {
                    const auto& key = phases->owned_order()[c];
                    const auto encoded = Meshes::AMR::encode_adaptive_key(key);
                    if (owners[c].key != encoded || owners[c].revision != target_hierarchy->owned().at(key).revision)
                        throw std::runtime_error("Ownership target key revision changed.");
                    // Requests, replies, their exchanges and decoded target coexist.
                    packet_admission.add(encoded.size(), 8);
                    target_rows.emplace(key, c);
                    OwnerPacket packet;
                    packet.peer = owners[c].owner;
                    word(packet.bytes, source->accepted_control());
                    word(packet.bytes, phases->accepted_control());
                    bytes(packet.bytes, encoded);
                    word(packet.bytes, owners[c].revision);
                    packets.push_back(std::move(packet));
                }
            });
        auto requests = exchange_owner_packets(packets, comm, packet_limit);
        std::vector<Key> requested_keys;
        execution.collective(
            [&]
            {
                for (const auto& packet : requests.packets)
                {
                    cell_resident.add(1, 2 * sizeof(Key) + sizeof(Meshes::AMR::AdaptiveOwner));
                    cell_resident.add(packet.bytes.size(), 4);
                }
                requested_keys.reserve(requests.packets.size());
                for (const auto& packet : requests.packets)
                {
                    std::size_t p = 0;
                    if (word(packet.bytes, p) != source->accepted_control() ||
                        word(packet.bytes, p) != phases->accepted_control())
                        throw std::runtime_error("Ownership request publication is stale.");
                    requested_keys.push_back(Meshes::AMR::decode_adaptive_key(bytes(packet.bytes, p)));
                }
            });
        const auto target_owners = target_hierarchy->lookup_cells(
            requested_keys, true, std::min(packet_limit, target_hierarchy->options().byte_budget));
        execution.collective(
            [&]
            {
                packets.clear();
                Budget packet_admission{packet_limit};
                packet_admission.add(requests.packets.size(), 2048);
                packets.reserve(std::max(packets.capacity(), requests.packets.size()));
                for (std::size_t i = 0; i < requests.packets.size(); ++i)
                {
                    const auto& request = requests.packets[i];
                    std::size_t p = 0;
                    (void) word(request.bytes, p);
                    (void) word(request.bytes, p);
                    const auto encoded = bytes(request.bytes, p);
                    const auto revision = word(request.bytes, p);
                    const auto& key = requested_keys[i];
                    const auto row = source_rows.find(key);
                    if (p != request.bytes.size() || row == source_rows.end() ||
                        target_owners[i].owner != request.peer || target_owners[i].revision != revision ||
                        hierarchy->owned().at(key).revision != revision)
                        throw std::runtime_error("Ownership state request has foreign owner/key/revision.");
                    packet_admission.add(encoded.size(), 8);
                    const auto c = row->second;
                    OwnerPacket reply;
                    reply.peer = request.peer;
                    word(reply.bytes, source->accepted_control());
                    word(reply.bytes, phases->accepted_control());
                    bytes(reply.bytes, encoded);
                    word(reply.bytes, revision);
                    bytes(reply.bytes, phase_record(*source, c, source_declarations));
                    for (double v : old.velocity[c])
                        real(reply.bytes, v);
                    for (double v : {old.pressure[c], old.curvature[c], old.fraction[c], old.density[c]})
                        real(reply.bytes, v);
                    wide(reply.bytes, old.finite_mass[c]);
                    wide(reply.bytes, old.finite_mass_error[c]);
                    if (optional[0])
                        for (long double v : old.finite_projection.velocity_error[c])
                            wide(reply.bytes, v);
                    if (optional[1])
                        wide(reply.bytes, old.finite_projection.cell_radius[c]);
                    packets.push_back(std::move(reply));
                }
            });
        auto replies = exchange_owner_packets(packets, comm, packet_limit);
        execution.collective(
            [&]
            {
                const auto n = phases->owned_order().size();
                state->velocity.resize(n);
                state->pressure.resize(n);
                state->curvature.resize(n);
                state->fraction.resize(n);
                state->density.resize(n);
                state->finite_mass.resize(n);
                state->finite_mass_error.resize(n);
                if (optional[0])
                    state->finite_projection.velocity_error.resize(n);
                if (optional[1])
                    state->finite_projection.cell_radius.resize(n);
                std::set<Key> seen;
                for (const auto& reply : replies.packets)
                {
                    std::size_t p = 0;
                    if (word(reply.bytes, p) != source->accepted_control() ||
                        word(reply.bytes, p) != phases->accepted_control())
                        throw std::runtime_error("Ownership reply publication is stale.");
                    const auto key = Meshes::AMR::decode_adaptive_key(bytes(reply.bytes, p));
                    const auto revision = word(reply.bytes, p);
                    const auto found = target_rows.find(key);
                    if (found == target_rows.end() || !seen.insert(key).second)
                        throw std::runtime_error("Ownership reply has duplicate or foreign key.");
                    const auto c = found->second;
                    if (reply.peer != owners[c].owner || revision != owners[c].revision ||
                        bytes(reply.bytes, p) != phase_record(*phases, c, target_declarations))
                        throw std::runtime_error(
                            "Ownership reply changes phase inventory/history/occupancy/declaration.");
                    for (auto& v : state->velocity[c])
                        v = real(reply.bytes, p);
                    state->pressure[c] = real(reply.bytes, p);
                    state->curvature[c] = real(reply.bytes, p);
                    state->fraction[c] = real(reply.bytes, p);
                    state->density[c] = real(reply.bytes, p);
                    state->finite_mass[c] = wide(reply.bytes, p);
                    state->finite_mass_error[c] = wide(reply.bytes, p);
                    if (optional[0])
                        for (auto& v : state->finite_projection.velocity_error[c])
                            v = wide(reply.bytes, p);
                    if (optional[1])
                        state->finite_projection.cell_radius[c] = wide(reply.bytes, p);
                    if (p != reply.bytes.size())
                        throw std::runtime_error("Ownership state reply has trailing bytes.");
                }
                if (seen.size() != n)
                    throw std::runtime_error("Ownership state transfer lacks target coverage.");
            });
    } // Release cell packets, key maps, directories and declarations before nested operations.
    {
        execution.collective(
            [&]
            {
                Budget storage_budget{available};
                storage_budget.add(
                    phases->owned_order().size(), 3 * sizeof(double) + 2 * sizeof(long double) + 2 * sizeof(void*));
            });
        const auto storage =
            PhaseStorage<ArithmeticMode::Ordinary>::evaluate(*candidate->mesh, *phases, candidate->material);
        execution.collective(
            [&]
            {
                for (std::size_t c = 0; c < state->velocity.size(); ++c)
                    if (std::bit_cast<std::uint64_t>(state->density[c]) !=
                            std::bit_cast<std::uint64_t>(storage.density[c]) ||
                        std::bit_cast<std::uint64_t>(state->fraction[c]) !=
                            std::bit_cast<std::uint64_t>(storage.fraction[c]) ||
                        state->finite_mass[c] != storage.mass[c] ||
                        state->finite_mass_error[c] != storage.mass_error[c])
                        throw std::runtime_error(
                            "Ownership copied storage disagrees with independent phase mass/density.");
            });
    } // Independent storage validation buffers do not coexist with curvature/face transfer.
    // Catalog contains native rank-local IDs; rebuild it without replacing the accepted curvature field.
    (void) Curvature<ArithmeticMode::Ordinary>::assemble(*candidate->mesh, candidate->flow, *phases,
        candidate->material.surface_tension, available, candidate->options.capillary, &state->capillary);
    execution.collective(
        [&]
        {
            retained.add(state->capillary.fits.capacity(), sizeof(VoF::GeneralVoFSurfaceFitDiagnostic));
            retained.add(state->capillary.components.capacity(), sizeof(VoF::GeneralVoFSurfaceComponent));
            if (retained.used == limit)
                throw std::length_error("Ownership migration has no face-transfer lease.");
        });
    const FaceTransferSource face_source{*resources.mesh, *source, old.velocity, old.finite_projection, resources.flow,
        ArithmeticMode::Ordinary, true, old.face_velocity};
    const FaceTransferTarget face_target{
        *candidate->mesh, *phases, candidate->volume, candidate->flow, ArithmeticMode::Ordinary};
    auto faces = transfer_owned_faces<ArithmeticMode::Ordinary>(
        face_source, face_target, {limit - retained.used, resources.options.flow.divergence_tolerance});
    auto velocity_error = std::move(state->finite_projection.velocity_error);
    auto cell_radius = std::move(state->finite_projection.cell_radius);
    state->face_velocity = std::move(faces.face_velocity);
    state->finite_projection = std::move(faces.projection);
    state->finite_projection.byte_budget = old.finite_projection.byte_budget;
    state->finite_projection.velocity_error = std::move(velocity_error);
    state->finite_projection.cell_radius = std::move(cell_radius);
    state->diagnostics = old.diagnostics;
    state->momentum_transport = old.momentum_transport;
    // Ordinary owns no Exact projection payload, but retains its configured scalar budget.
    state->projection_arithmetic.byte_budget = old.projection_arithmetic.byte_budget;
    // Preserve the existing independent physical phase-mass-weighted momentum gate.
    constexpr auto Mode = ArithmeticMode::Ordinary;
    constexpr bool bounded_mode = false;
    constexpr auto arithmetic_mode_ = Mode;
    using A = VoF::finite_detail::Arithmetic<Mode>;
    namespace fp = VoF::finite_detail;
    for (unsigned a = 0; a < 3; ++a)
    {
        VoF::finite_detail::Value residual, l1;
        execution.collective(
            [&]
            {
                VoF::finite_detail::SumFor<Mode> change(arithmetic_mode_), reference(arithmetic_mode_);
                for (std::size_t c = 0; c < old.velocity.size(); ++c)
                {
                    const auto amount = A::multiply(A::make_value(old.finite_mass[c], old.finite_mass_error[c]),
                        A::make_value(old.velocity[c][a], 0));
                    change.add(A::negate(amount));
                    reference.add(A::make_value(std::abs(amount.value), amount.absolute_error));
                }
                for (std::size_t c = 0; c < state->velocity.size(); ++c)
                {
                    const auto amount = A::multiply(A::make_value(state->finite_mass[c], state->finite_mass_error[c]),
                        A::make_value(state->velocity[c][a], 0));
                    change.add(amount);
                    reference.add(A::make_value(std::abs(amount.value), amount.absolute_error));
                }
                residual = change.value();
                l1 = reference.value();
            });
        int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
        if (l1.value != 0)
            (void) std::frexp(l1.value + (bounded_mode ? l1.absolute_error : 0), &local_exponent);
        Teuchos::reduceAll(
            *resources.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
        double local[4]{}, global[4]{};
        execution.collective(
            [&]
            {
                const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
                const auto center = residual.value / scale;
                const auto narrowed_center = static_cast<double>(center);
                const auto reference_difference =
                    !bounded_mode || l1.absolute_error == 0
                        ? l1.value
                        : std::nextafter(l1.value - l1.absolute_error, -std::numeric_limits<long double>::infinity());
                const auto scaled_reference =
                    A::divide(A::make_value(std::max(0.L, reference_difference), 0), A::make_value(scale, 0));
                const auto lower_reference = std::max(0.L, A::lower(scaled_reference));
                auto narrowed_reference = static_cast<double>(lower_reference);
                if (bounded_mode && static_cast<long double>(narrowed_reference) > lower_reference)
                    narrowed_reference = std::nextafter(narrowed_reference, 0.);
                const double values[4] = {narrowed_center,
                    bounded_mode ? fp::upper_double(fp::bound_add(
                                       residual.absolute_error / scale, std::abs(center - narrowed_center)))
                                 : 0,
                    narrowed_reference, bounded_mode ? fp::upper_double(std::abs(center)) : double(std::abs(center))};
                std::copy(std::begin(values), std::end(values), local);
            });
        Teuchos::reduceAll(*resources.mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 4, local, global);
        const auto u = bounded_mode ? (8.L * resources.mesh->owned_cell_map()->getComm()->getSize() + 16) *
                                          std::numeric_limits<double>::epsilon()
                                    : 0;
        const auto error = bounded_mode ? (global[1] + u * (global[3] + global[1])) / (1 - u) : 0;
        execution.require(
            std::abs(global[0]) + error <= resources.options.momentum_relative_tolerance * global[2] * (1 - u),
            "Finite fixed-clock remap changed physical phase-mass-weighted momentum.");
    }
    trial.validate_observations();
    execution.require(source->accepted_unchanged() && hierarchy->geometry_unchanged() && phases->accepted_unchanged(),
        "Ownership migration observations changed before publication.");
    return {std::move(candidate), std::move(state)};
}
} // namespace SimpleFluid::detail::general_vof
