/** @file Transport.tcc
 * @brief Moving phase transport; collective stages remain coordinated here.
 */
#include "FVM/details/GeneralVoF/ConstructionControls.tcc"
#include "FVM/details/GeneralVoF/LocalTransport.hh"
#include "FVM/details/GeneralVoF/PhaseAllocation.hh"
#include "FVM/details/GeneralVoF/Wire.hh"
namespace SimpleFluid::VoF
{
using SimpleFluid::geometry::swept_volume::runtime_phases;
using SimpleFluid::geometry::swept_volume::runtime_value;
namespace auth = construction_controls;
namespace
{
// Local stage. The operation coordinator provides collective error propagation.
template<ArithmeticMode Mode>
void prepare_transport_controls(const GeneralVoF::Data& data, std::span<const GeneralVoFNodeVelocity> node_velocity,
    std::span<const GeneralVoFFaceVolume> face_volume, double timestep,
    std::map<Key, PLIC::ConvexGeometryPtr<Mode>>& cell_geometry,
    std::map<std::uint64_t, std::pair<PLIC::Point<Mode>, PLIC::Point<Mode>>>& traces,
    std::map<std::uint64_t, PhaseValue>& flux, std::map<std::uint64_t, long double>& face_budgets,
    const PLIC::GeometryPacketContext& packet_context, std::vector<OwnerPacket>& packets)
{
    using Math = fp::Arithmetic<Mode>;
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    current(data);
    if (cell_geometry.empty())
    {
        wire::Budget geometry_budget{limit};
        for (const auto& [key, cell] : data.cells)
        {
            geometry_budget.add(sizeof(typename std::remove_reference_t<decltype(cell_geometry)>::value_type) +
                                4 * sizeof(void*) + sizeof(PLIC::ConvexCellGeometry<Mode>) +
                                AMR::encode_adaptive_key(key).size());
            cell_geometry.emplace(key, PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry));
        }
    }
    if (!std::isfinite(timestep) || timestep < 0 || node_velocity.size() != data.topology->nodes.size() ||
        face_volume.size() != data.topology->tiles.size())
        throw std::invalid_argument("Finite VoF controls do not cover retained nodes/faces.");
    wire::Budget budget{limit};
    std::map<std::uint64_t, SimpleFluid::PLIC::GeometryPoint> velocity;
    for (const auto& node : node_velocity)
    {
        if (!std::all_of(node.velocity.begin(), node.velocity.end(), [](auto x) { return std::isfinite(x); }) ||
            !velocity.emplace(node.node, node.velocity).second)
            throw std::invalid_argument("Finite VoF node control is nonfinite or duplicated.");
    }
    for (const auto& node : data.topology->nodes)
    {
        if (!velocity.contains(node.gid))
            throw std::invalid_argument("Finite VoF node controls contain a foreign identity.");
        const PLIC::Point<Mode> source(data.topology->finite_nodes.at(node.gid));
        const auto departure = auth::departure<Mode>(data, node.gid, velocity.at(node.gid), timestep);
        budget.add(sizeof(typename std::remove_reference_t<decltype(traces)>::value_type) + 4 * sizeof(void*) +
                   PLIC::dynamic_point_storage_bytes(source.dynamic()) +
                   PLIC::dynamic_point_storage_bytes(departure.dynamic()));
        traces.emplace(node.gid, std::pair{source, departure});
        auto bytes =
            auth::node_record<Mode>(data, node.gid, source, departure, velocity.at(node.gid), timestep, packet_context);
        budget.packet(bytes.size());
        packets.push_back({int(node.gid % comm->getSize()), std::move(bytes)});
    }
    for (const auto& face : face_volume)
    {
        if (face.exact_integrated_volume || !std::isfinite(face.integrated_volume))
            throw std::invalid_argument("Finite VoF cannot adopt an exact-policy face control.");
        const auto input = face.bounded_integrated_volume.value_or(
            Math::make_value(data.options.arithmetic_mode, face.integrated_volume));
        fp::validate(input);
        if (input.arithmetic_mode != data.options.arithmetic_mode)
            throw std::invalid_argument("Finite VoF face control arithmetic mode differs.");
        const auto value = Math::make_value(data.options.arithmetic_mode, input.value, input.absolute_error);
        if (!std::isfinite(value.value) || !std::isfinite(value.absolute_error) || value.absolute_error < 0 ||
            !flux.emplace(face.face, value).second)
            throw std::invalid_argument("Finite VoF face control is malformed or duplicated.");
    }
    for (const auto& tile : data.topology->tiles)
    {
        if (!flux.contains(tile.face_gid))
            throw std::invalid_argument("Finite VoF face controls contain a foreign identity.");
        const auto phi = flux.at(tile.face_gid);
        if ((tile.exterior || timestep == 0) && phi.value != 0)
            throw std::invalid_argument("Finite VoF impermeable boundary has nonzero flux.");
        long double allowed = 0;
        if constexpr (Mode == ArithmeticMode::Ordinary)
        {
            auto scale = data.cells.at(tile.first).ordinary_quota_scale;
            if (tile.second)
                scale = std::min(scale, data.cells.at(*tile.second).ordinary_quota_scale);
            allowed =
                std::nextafter(static_cast<long double>(data.options.reconstruction.volume_tolerance) * scale, 0.L);
            if (!(allowed > 0) || !std::isfinite(allowed))
                throw std::invalid_argument("General VoF canonical face budget has invalid capacity.");
        }
        face_budgets.emplace(tile.face_gid, allowed);
        auto bytes = auth::face_record(data, tile.face_gid, phi, allowed, timestep, packet_context);
        budget.packet(bytes.size());
        packets.push_back({int(tile.face_gid % comm->getSize()), std::move(bytes)});
    }
    wire::Writer w;
    auth::header(w, 2, 0, packet_context, Mode);
    real(w, timestep);
    w.bytes(auth::algorithm(data.options));
    budget.packet(w.data.size());
    packets.push_back({0, std::move(w.data)});
}
} // namespace
template<ArithmeticMode Mode>
GeneralVoF::Trial GeneralVoF::plan_transport_finite(std::span<const GeneralVoFNodeVelocity> node_velocity,
    std::span<const GeneralVoFFaceVolume> face_volume, double timestep) const
{
    using Math = fp::Arithmetic<Mode>;
    const auto& data = *data_;
    std::map<Key, PLIC::ConvexGeometryPtr<Mode>> cell_geometry;
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    namespace failure = ::SimpleFluid::detail::vof_failure;
    failure::Scope transport_context("swept_transport");
    transport_context.record.arithmetic_mode = Mode;
    transport_context.record.mark(failure::Field::Arithmetic);
    transport_context.record.byte_budget = limit;
    transport_context.record.mark(failure::Field::ByteBudget);
    Trial trial;
    trial.source_ = data_;
    trial.transaction_ = transaction(comm);
    trial.timestep_ = timestep;
    const PLIC::GeometryPacketContext packet_context{{data.control, data.hierarchy->topology_generation(),
        data.hierarchy->partition_generation(), trial.transaction_}};
    std::map<std::uint64_t, std::pair<PLIC::Point<Mode>, PLIC::Point<Mode>>> traces;
    std::map<std::uint64_t, PhaseValue> flux;
    std::map<std::uint64_t, long double> face_budgets;
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            prepare_transport_controls<Mode>(data, node_velocity, face_volume, timestep, cell_geometry, traces, flux,
                face_budgets, packet_context, packets);
        });
    auth::Records authoritative_controls;
    std::size_t authoritative_bytes = 0;
    {
        auto received = exchange_owner_packets(packets, comm, limit);
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{limit};
                budget.add(received.packets.capacity() * sizeof(OwnerPacket));
                for (const auto& packet : received.packets)
                    budget.add(packet.bytes.capacity());
                for (auto& packet : received.packets)
                {
                    wire::Reader r{packet.bytes};
                    const auto key = auth::header(r, packet_context, Mode);
                    if (key.first > 2 || int(key.second % comm->getSize()) != comm->getRank())
                        throw std::invalid_argument("Finite VoF control reached a foreign authority.");
                    auto found = authoritative_controls.find(key);
                    if (found != authoritative_controls.end())
                    {
                        if (found->second != packet.bytes)
                            throw std::invalid_argument("Finite VoF shared node/face controls disagree.");
                    }
                    else
                    {
                        budget.add(sizeof(auth::Records::value_type) + 4 * sizeof(void*));
                        authoritative_controls.emplace(key, std::move(packet.bytes));
                    }
                }
                wire::Budget retained{limit};
                for (const auto& [key, bytes] : authoritative_controls)
                    retained.add(sizeof(auth::Records::value_type) + 4 * sizeof(void*) + bytes.capacity());
                authoritative_bytes = retained.used;
                for (const auto& key : data.topology->order)
                {
                    fp::SumFor<Mode> sum;
                    for (auto index : data.topology->cell_tiles.at(key))
                    {
                        const auto& tile = data.topology->tiles[index];
                        const int sign = tile.first == key ? 1 : -1;
                        sum.add(sign > 0 ? flux.at(tile.face_gid) : Math::negate(flux.at(tile.face_gid)));
                    }
                    const auto residual = sum.value();
                    const auto bulk = runtime_value<Mode>(cell_geometry.at(key)->volume());
                    const auto bound = phase_allowance<Mode>(bulk, data.options.reconstruction.volume_tolerance);
                    if (!Math::closure_within(residual, bound))
                        throw std::invalid_argument("Finite VoF canonical controls violate source-free occupancy.");
                }
            });
    }
    std::vector<OwnerPacket>().swap(packets);
    int local_motion = 0, motion = 0;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(traces.size() * sizeof(GeneralVoFFiniteNodeTrajectory));
            trial.finite_trajectories_.reserve(traces.size());
            for (const auto& [id, trace] : traces)
            {
                local_motion |= !SimpleFluid::geometry::swept_volume::same_point(trace.first, trace.second);
                trial.finite_trajectories_.push_back({id, trace.first.dynamic(), trace.second.dynamic()});
            }
            for (const auto& [face, value] : flux)
                local_motion |= value.value != 0;
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_motion, &motion);
    trial.roundoff_.resize(data.topology->order.size());
    if (!motion)
    {
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{limit};
                budget.add(data.topology->tiles.size() * sizeof(GeneralVoFFaceChainDiagnostic));
                budget.add(data.topology->order.size() * sizeof(GeneralVoFCellClosureDiagnostic));
                trial.finite_face_chain_.reserve(data.topology->tiles.size());
                trial.finite_cell_closure_.reserve(data.topology->order.size());
                trial.candidate_ = data_;
                const auto zero = Math::make_value(data.options.arithmetic_mode, 0);
                for (const auto& tile : data.topology->tiles)
                    trial.finite_face_chain_.push_back({tile.face_gid, flux.at(tile.face_gid), zero,
                        Math::subtract(zero, flux.at(tile.face_gid)), face_budgets.at(tile.face_gid)});
                for (const auto& key : data.topology->order)
                {
                    const auto& cell = data.cells.at(key);
                    const auto old = phase_inventory<Mode>(cell, *cell_geometry.at(cell.key));
                    trial.finite_transfers_.push_back(
                        {cell.gid, cell.gid, cell.owner, cell.owner, old, cell.finite_history});
                    budget.add(AMR::encode_adaptive_key(key).size());
                    trial.finite_cell_closure_.push_back(
                        {key, cell.gid, cell.canonical_face_count, cell.ordinary_quota_scale, zero, zero, zero, 0,
                            phase_allowance<Mode>(old.bulk, data.options.reconstruction.volume_tolerance)});
                    trial.finite_budgets_.push_back({cell.gid, old, {zero, zero, zero}, old});
                }
            });
        return trial;
    }
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        if (homogeneous_transport_eligible())
        {
            auto algebraic = plan_face_transport_impl(face_volume, timestep, {1, 1}, 0, true, trial.transaction_);
            algebraic.finite_trajectories_ = std::move(trial.finite_trajectories_);
            algebraic.validate_retained_algebraic_payload();
            return algebraic;
        }
        return plan_interface_transport(std::move(trial), {face_volume, traces, flux, face_budgets, packet_context});
    }
    PLIC::GeometryLimits registry_limits = geometry_limits(Mode, limit);
    registry_limits.maximum_pair_tests = data.options.sweep.maximum_pair_tests;
    std::optional<PLIC::ConstructionRegistry<Mode>> construction;
    wire::collective(*comm,
        [&]
        {
            construction.emplace(packet_context, registry_limits);
            for (const auto& [key, cell] : data.cells)
                construction->register_native(cell.gid, cell.revision, cell_geometry.at(key));
            wire::Budget retained{limit};
            retained.add(authoritative_bytes);
            retained.add(construction->storage_bytes());
        });
    auto& registry = *construction;
    const auto planes = import_planes<Mode>(data);
    std::map<Key, PhaseLabel<Mode>> labels;
    std::map<std::uint64_t, SimpleFluid::geometry::swept_volume::Sweep<Mode>> sweeps;
    std::vector<SimpleFluid::geometry::swept_volume::Atom<Mode>> local_atoms;
    std::map<std::tuple<std::uint64_t, std::uint64_t, int>, PhaseSum<Mode>> face_amounts;
    wire::collective(*comm,
        [&]
        {
            SimpleFluid::geometry::swept_volume::Work work{data.options.sweep};
            work.arithmetic_mode = data.options.arithmetic_mode;
            work.volume_tolerance = data.options.reconstruction.volume_tolerance;
            work.limits.byte_budget = std::min(limit, work.limits.byte_budget);
            work.charge(authoritative_bytes);
            work.charge(registry.storage_bytes());
            work.charge(data.topology->tiles.size() * sizeof(GeneralVoFFaceChainDiagnostic));
            work.charge(data.topology->order.size() * sizeof(GeneralVoFCellClosureDiagnostic));
            trial.finite_face_chain_.reserve(data.topology->tiles.size());
            trial.finite_cell_closure_.reserve(data.topology->order.size());
            for (const auto& [key, cell] : data.cells)
            {
                work.charge(sizeof(typename decltype(labels)::value_type) + 4 * sizeof(void*));
                labels.emplace(key, phase_label<Mode>(cell, *cell_geometry.at(key), planes.at(key),
                                        data.options.reconstruction.volume_tolerance));
            }
            std::map<std::uint64_t, PLIC::ConvexGeometryPtr<Mode>> donors;
            std::vector<PLIC::ConvexGeometryPtr<Mode>> donor_geometries;
            std::vector<const Key*> donor_keys;
            work.charge(data.cells.size() * (sizeof(PLIC::ConvexGeometryPtr<Mode>) + sizeof(const Key*)));
            donor_geometries.reserve(data.cells.size());
            donor_keys.reserve(data.cells.size());
            for (const auto& [key, cell] : data.cells)
            {
                donors.emplace(cell.gid, cell_geometry.at(key));
                donor_geometries.push_back(cell_geometry.at(key));
                donor_keys.push_back(&key);
            }
            PLIC::GeometryLimits partition_limits;
            partition_limits.arithmetic_mode = data.options.arithmetic_mode;
            partition_limits.maximum_pair_tests = work.limits.maximum_pair_tests;
            partition_limits.byte_budget = work.limits.byte_budget;
            failure::Scope partition_context("donor_partition");
            const PLIC::ConvexPartition<Mode> donor_partition(donor_geometries, partition_limits);
            work.charge(donor_partition.storage_bytes());
            for (const auto& tile : data.topology->tiles)
            {
                failure::Scope face_context("face_sweep_partition");
                face_context.record.face = tile.face_gid;
                face_context.record.mark(failure::Field::Face);
                face_context.record.volume_budget = face_budgets.at(tile.face_gid);
                face_context.record.mark(failure::Field::VolumeBudget);
                if constexpr (Mode == ArithmeticMode::Ordinary)
                {
                    face_context.record.scale = data.cells.at(tile.first).ordinary_quota_scale;
                    if (tile.second)
                        face_context.record.scale =
                            std::min(face_context.record.scale, data.cells.at(*tile.second).ordinary_quota_scale);
                    face_context.record.mark(failure::Field::Scale);
                }
                face_context.record.pair_budget = work.limits.maximum_pair_tests;
                face_context.record.mark(failure::Field::PairBudget);
                work.pair_tests = 0; // Existing face-partition work-cap scope.
                auto sweep = SimpleFluid::geometry::swept_volume::make_sweep<Mode>(tile.face_gid, tile.node_gids, traces, flux.at(tile.face_gid), work,
                    face_budgets.at(tile.face_gid));
                for (std::size_t ordinal = 0; ordinal < sweep.pieces.size(); ++ordinal)
                {
                    const auto before = registry.storage_bytes();
                    registry.register_sweep(tile.face_gid, ordinal, sweep.pieces[ordinal].geometry);
                    work.charge(registry.storage_bytes() - before);
                }
                // Construct one complete complementary partition per signed simplex.
                // The immutable donor set has already proved disjointness; the
                // shared split tree certifies coverage without replaying geometry.
                for (std::size_t ordinal = 0; ordinal < sweep.pieces.size(); ++ordinal)
                {
                    const auto& piece = sweep.pieces[ordinal];
                    failure::Scope piece_context("donor_sweep_partition");
                    piece_context.record.construction = ordinal;
                    piece_context.record.mark(failure::Field::Construction);
                    piece_context.record.bytes_used = work.bytes;
                    piece_context.record.mark(failure::Field::BytesUsed);
                    const auto partition = donor_partition.partition(*piece.geometry, work.limits.maximum_atoms);
                    for (const auto& intersection : partition)
                    {
                        work.test();
                        const auto& key = *donor_keys.at(intersection.donor);
                        const auto& cell = data.cells.at(key);
                        failure::Scope donor_context("donor_phase_measure");
                        donor_context.record.donor = cell.gid;
                        donor_context.record.mark(failure::Field::Donor);
                        donor_context.record.scale = cell.ordinary_quota_scale;
                        donor_context.record.mark(failure::Field::Scale);
                        work.atom(intersection.geometry);
                        if (tile.first_owner == comm->getRank())
                            face_amounts[{tile.face_gid, cell.gid, piece.sign}].add(
                                labels.at(key).apply(*intersection.geometry, planes.at(key)));
                    }
                }
                trial.finite_face_chain_.push_back({tile.face_gid, sweep.control, sweep.constructed_volume,
                    sweep.signed_control_difference, sweep.ordinary_volume_budget});
                sweeps.emplace(tile.face_gid, std::move(sweep));
            }
            for (const auto& key : data.topology->order)
            {
                const auto& cell = data.cells.at(key);
                failure::Scope cell_context("recipient_closure");
                cell_context.record.cell = cell.gid;
                cell_context.record.mark(failure::Field::Cell);
                cell_context.record.scale = cell.ordinary_quota_scale;
                cell_context.record.mark(failure::Field::Scale);
                fp::SumFor<Mode> canonical(data.options.arithmetic_mode), signed_geometry(data.options.arithmetic_mode),
                    absolute_geometry(data.options.arithmetic_mode);
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    const int direction = tile.first == key ? 1 : -1;
                    const auto& sweep = sweeps.at(tile.face_gid);
                    canonical.add(direction > 0 ? sweep.control : Math::negate(sweep.control));
                    signed_geometry.add(direction > 0 ? sweep.signed_control_difference
                                                      : Math::negate(sweep.signed_control_difference));
                    absolute_geometry.add(Math::make_value(
                        data.options.arithmetic_mode, std::abs(sweep.signed_control_difference.value)));
                }
                const auto r_phi = canonical.value(), delta = signed_geometry.value();
                const auto combined = Math::add(r_phi, delta);
                const auto allowed = phase_allowance<Mode>(runtime_value<Mode>(cell_geometry.at(cell.key)->volume()),
                    data.options.reconstruction.volume_tolerance);
                if constexpr (Mode == ArithmeticMode::Ordinary)
                    if (!Math::closure_within(combined, allowed))
                        throw std::invalid_argument(
                            "Finite VoF combined canonical/geometry cell closure exceeds its original budget.");
                work.charge(AMR::encode_adaptive_key(key).size());
                trial.finite_cell_closure_.push_back({key, cell.gid, cell.canonical_face_count,
                    cell.ordinary_quota_scale, r_phi, delta, combined, absolute_geometry.value().value, allowed});
                std::vector<std::pair<const SimpleFluid::geometry::swept_volume::Sweep<Mode>*, int>> incident;
                for (auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    incident.push_back({&sweeps.at(tile.face_gid), tile.first == key ? 1 : -1});
                }
                work.pair_tests = 0; // Existing recipient-local partition work-cap scope.
                auto atoms = SimpleFluid::geometry::swept_volume::recipient_partition<Mode>(
                    cell.gid, donors, incident, work, data.options.reconstruction.volume_tolerance);
                local_atoms.insert(local_atoms.end(), atoms.begin(), atoms.end());
            }
            for (const auto& [key, sum] : face_amounts)
            {
                auto value = sum.value();
                const int direction = std::get<2>(key);
                for (const auto& amount : {value.bulk, value.gas, value.carrier})
                    if (amount.value < 0 || (amount.value != 0 && amount.value <= amount.absolute_error))
                        throw std::invalid_argument("Finite VoF raw donor stream positivity is unresolved.");
                const auto& donor = data.cells.at(data.topology->by_gid.at(std::get<1>(key)));
                trial.finite_faces_.push_back({std::get<0>(key), std::get<1>(key), donor.owner, direction, value});
            }
        });
    // Authenticate positive geometric fragments on the original donor owner.
    packets.clear();
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (std::size_t i = 0; i < local_atoms.size(); ++i)
            {
                const auto& atom = local_atoms[i];
                const auto& donor = data.cells.at(data.topology->by_gid.at(atom.donor));
                const auto& recipient = data.cells.at(data.topology->by_gid.at(atom.recipient));
                wire::Writer w;
                w.bytes(AMR::encode_adaptive_key(donor.key));
                w.word(donor.revision);
                w.word(donor.gid);
                w.bytes(AMR::encode_adaptive_key(recipient.key));
                w.word(recipient.gid);
                w.word(donor.owner == comm->getRank());
                if (donor.owner == comm->getRank())
                    w.word(i);
                else
                    w.bytes(atom.geometry->encode(registry, limit));
                budget.packet(w.data.size());
                packets.push_back({donor.owner, std::move(w.data)});
            }
        });
    const auto routed_packets = exchange_owner_packets(packets, comm, limit);
    struct Routed
    {
        Key donor, recipient;
        std::uint64_t recipient_gid;
        int recipient_owner;
        PLIC::ConvexGeometryPtr<Mode> geometry;
    };
    std::vector<Routed> routed;
    std::vector<Key> recipient_keys;
    struct PendingAtom
    {
        Routed value;
        std::span<const char> packet;
    };
    std::vector<PendingAtom> pending;
    std::set<PLIC::ConstructionRootKey> dependencies;
    std::set<std::uint64_t> missing_faces;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(authoritative_bytes);
            budget.add(registry.storage_bytes());
            for (const auto& packet : routed_packets.packets)
            {
                wire::Reader r{packet.bytes};
                auto donor_key = AMR::decode_adaptive_key(auth::packet_view(r));
                const auto revision = r.word(), gid = r.word();
                auto recipient_key = AMR::decode_adaptive_key(auth::packet_view(r));
                const auto recipient_gid = r.word(), local = r.word();
                const auto& donor = data.cells.at(donor_key);
                if (donor.owner != comm->getRank() || donor.gid != gid || donor.revision != revision)
                    throw std::invalid_argument("Finite VoF atom donor authentication failed.");
                PLIC::ConvexGeometryPtr<Mode> geometry;
                std::span<const char> encoded;
                if (local == 1)
                {
                    const auto index = r.word();
                    if (packet.peer != comm->getRank() || index >= local_atoms.size())
                        throw std::invalid_argument("Finite VoF foreign local atom reference.");
                    const auto& original = local_atoms[index];
                    if (original.donor != gid || original.recipient != recipient_gid ||
                        data.cells.at(data.topology->by_gid.at(recipient_gid)).key != recipient_key)
                        throw std::invalid_argument("Finite VoF local atom metadata differs.");
                    geometry = original.geometry;
                }
                else if (local == 0 && packet.peer != comm->getRank())
                {
                    encoded = auth::packet_view(r);
                    const auto roots =
                        PLIC::ConstructionRegistry<Mode>::required_roots(encoded, packet_context, registry_limits);
                    bool donor_root = false;
                    for (const auto& root : roots)
                    {
                        auth::admit_root(data, root, gid, revision, recipient_key);
                        donor_root = donor_root || root.kind == PLIC::ConstructionRootKey::Kind::Native;
                        if (registry.contains(root))
                            continue;
                        if (root.kind != PLIC::ConstructionRootKey::Kind::Sweep)
                            throw std::invalid_argument("Finite VoF packet lacks an admitted native source root.");
                        if (sweeps.contains(root.entity))
                            throw std::invalid_argument("Finite VoF packet requests a "
                                                        "nonexistent source-sweep ordinal.");
                        if (dependencies.insert(root).second)
                            budget.add(sizeof(root) + 4 * sizeof(void*));
                        if (missing_faces.insert(root.entity).second)
                            budget.add(sizeof(std::uint64_t) + 4 * sizeof(void*));
                    }
                    if (!donor_root)
                        throw std::invalid_argument(
                            "Finite VoF atom construction omits its authenticated native donor.");
                }
                else
                    throw std::invalid_argument("Finite VoF atom routing tag is malformed.");
                r.done();
                budget.add(sizeof(PendingAtom) + AMR::encode_adaptive_key(donor_key).size() +
                           AMR::encode_adaptive_key(recipient_key).size());
                pending.push_back(
                    {{std::move(donor_key), std::move(recipient_key), recipient_gid, packet.peer, std::move(geometry)},
                        encoded});
            }
        });
    std::size_t retained_protocol_bytes = 0;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(authoritative_bytes);
            budget.add(registry.storage_bytes());
            budget.add(dependencies.size() * (sizeof(PLIC::ConstructionRootKey) + 4 * sizeof(void*)));
            budget.add(missing_faces.size() * (sizeof(std::uint64_t) + 4 * sizeof(void*)));
            budget.add(pending.capacity() * sizeof(PendingAtom));
            for (const auto& item : pending)
                budget.add(AMR::encode_adaptive_key(item.value.donor).size() +
                           AMR::encode_adaptive_key(item.value.recipient).size());
            retained_protocol_bytes = budget.used;
        });
    auto face_records = auth::request_records(
        1, missing_faces, authoritative_controls, packet_context, Mode, comm, limit, retained_protocol_bytes);
    std::set<std::uint64_t> missing_nodes;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(retained_protocol_bytes);
            for (const auto& [face, bytes] : face_records)
            {
                budget.add(sizeof(decltype(face_records)::value_type) + 4 * sizeof(void*) + bytes.capacity());
                const auto [control, allowance] =
                    auth::accept_face_record<Mode>(data, face, bytes, timestep, packet_context);
                budget.add(sizeof(decltype(flux)::value_type) + sizeof(decltype(face_budgets)::value_type) +
                           8 * sizeof(void*));
                if (!flux.emplace(face, control).second || !face_budgets.emplace(face, allowance).second)
                    throw std::invalid_argument("Finite VoF source-sweep control was "
                                                "already retained without its root.");
                for (const auto node : data.topology->construction_faces.at(face).node_gids)
                {
                    if (!data.topology->construction_nodes.contains(node) ||
                        !data.topology->finite_nodes.contains(node) || !data.topology->node_walls.contains(node))
                        throw std::invalid_argument("Finite VoF source sweep needs a node "
                                                    "outside its admitted native halo.");
                    if (!traces.contains(node) && missing_nodes.insert(node).second)
                        budget.add(sizeof(std::uint64_t) + 4 * sizeof(void*));
                }
            }
            retained_protocol_bytes = budget.used;
        });
    auto node_records = auth::request_records(
        0, missing_nodes, authoritative_controls, packet_context, Mode, comm, limit, retained_protocol_bytes);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(retained_protocol_bytes);
            for (const auto& [node, bytes] : node_records)
            {
                budget.add(sizeof(decltype(node_records)::value_type) + 4 * sizeof(void*) + bytes.capacity());
                auto trace = auth::accept_node_record<Mode>(data, node, bytes, timestep, packet_context);
                budget.add(sizeof(typename decltype(traces)::value_type) + 4 * sizeof(void*) +
                           PLIC::dynamic_point_storage_bytes(trace.first.dynamic()) +
                           PLIC::dynamic_point_storage_bytes(trace.second.dynamic()));
                if (!traces.emplace(node, std::move(trace)).second)
                    throw std::invalid_argument("Finite VoF source-node trace was independently registered twice.");
            }
            SimpleFluid::geometry::swept_volume::Work reconstruction{data.options.sweep};
            reconstruction.arithmetic_mode = Mode;
            reconstruction.volume_tolerance = data.options.reconstruction.volume_tolerance;
            reconstruction.limits.byte_budget = std::min(limit, reconstruction.limits.byte_budget);
            reconstruction.charge(budget.used);
            for (const auto face : missing_faces)
            {
                reconstruction.pair_tests = 0;
                reconstruction.charge(sizeof(typename decltype(sweeps)::value_type) + 4 * sizeof(void*));
                const auto& native = data.topology->construction_faces.at(face);
                auto sweep = SimpleFluid::geometry::swept_volume::make_sweep<Mode>(
                    face, native.node_gids, traces, flux.at(face), reconstruction, face_budgets.at(face));
                for (std::size_t ordinal = 0; ordinal < sweep.pieces.size(); ++ordinal)
                {
                    const auto before = registry.storage_bytes();
                    registry.register_sweep(face, ordinal, sweep.pieces[ordinal].geometry);
                    reconstruction.charge(registry.storage_bytes() - before);
                }
                if (!sweeps.emplace(face, std::move(sweep)).second)
                    throw std::invalid_argument("Finite VoF source sweep was reconstructed twice.");
            }
            for (const auto& root : dependencies)
                if (!registry.contains(root))
                    throw std::invalid_argument("Finite VoF packet requires an "
                                                "unconstructed source-sweep ordinal.");
        });
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(authoritative_bytes);
            budget.add(registry.storage_bytes());
            for (auto& item : pending)
            {
                if (!item.value.geometry)
                    item.value.geometry =
                        PLIC::ConvexCellGeometry<Mode>::decode(item.packet, registry, registry_limits);
                const auto& donor = data.cells.at(item.value.donor);
                auth::require_donor_ancestry(registry, *item.value.geometry, donor.gid, donor.revision);
                budget.add(sizeof(Routed) + item.value.geometry->storage_bytes());
                recipient_keys.push_back(item.value.recipient);
                routed.push_back(std::move(item.value));
            }
        });
    std::vector<PendingAtom>().swap(pending);
    const auto recipients = data.hierarchy->lookup_cells(recipient_keys);
    // Signed fitting redistribution is distinct from arithmetic uncertainty.
    // The other entries carry source bounds by the same geometric bulk share.
    enum EvidenceIndex
    {
        Allocation,
        Occupancy,
        FitAllocation,
        PhaseInput,
        Transferred,
        PureFit,
        EvidenceCount
    };
    struct Evidence
    {
        std::array<fp::SumFor<Mode>, EvidenceCount> values;
        std::array<bool, 2> presence{};
        std::array<PhaseValue, EvidenceCount> value() const
        {
            std::array<PhaseValue, EvidenceCount> result;
            for (unsigned i = 0; i < EvidenceCount; ++i)
                result[i] = values[i].value();
            return result;
        }
    };
    struct Amount
    {
        Key donor, recipient;
        std::uint64_t donor_gid = 0, donor_revision = 0, recipient_gid = 0;
        int recipient_owner = -1;
        PhaseValues value, history;
        std::array<PhaseValue, EvidenceCount> provenance;
        // These bits exclude geometric absence, not independent trace labels.
        bool gas_geometry = false, carrier_geometry = false;
    };
    std::vector<Amount> amounts;
    wire::collective(*comm,
        [&]
        {
            SimpleFluid::geometry::swept_volume::Work work{data.options.sweep};
            work.arithmetic_mode = data.options.arithmetic_mode;
            work.volume_tolerance = data.options.reconstruction.volume_tolerance;
            work.limits.byte_budget = std::min(limit, work.limits.byte_budget);
            std::map<Key, std::vector<PLIC::ConvexGeometryPtr<Mode>>> covers;
            std::map<std::pair<Key, Key>, PhaseSum<Mode>> groups, history_groups;
            std::map<std::pair<Key, Key>, Evidence> evidence;
            std::map<Key, std::pair<int, std::uint64_t>> target_metadata;
            for (std::size_t i = 0; i < routed.size(); ++i)
            {
                const auto& atom = routed[i];
                if (recipients[i].owner != atom.recipient_owner || recipients[i].address != atom.recipient_gid)
                    throw std::invalid_argument("Finite VoF atom recipient owner authentication failed.");
                target_metadata.emplace(atom.recipient, std::pair{atom.recipient_owner, atom.recipient_gid});
                covers[atom.donor].push_back(atom.geometry);
                const auto& plane = planes.at(atom.donor);
                std::array<bool, 2> piece_presence{};
                PhaseValue allocation;
                const auto amount = labels.at(atom.donor).apply(*atom.geometry, plane, &piece_presence, &allocation);
                groups[{atom.donor, atom.recipient}].add(amount);
                auto [evidence_entry, inserted] = evidence.try_emplace({atom.donor, atom.recipient});
                if (inserted)
                    work.charge(sizeof(typename decltype(evidence)::value_type) + 4 * sizeof(void*));
                auto& entry = evidence_entry->second;
                const auto& source = data.cells.at(atom.donor);
                const auto share =
                    Math::divide(amount.bulk, runtime_value<Mode>(cell_geometry.at(source.key)->volume()));
                entry.values[Allocation].add(allocation);
                const std::array<double, EvidenceCount - 1> inherited{source.occupancy.admitted_bound,
                    source.occupancy.fit_allocation_bound, source.occupancy.phase_input_bound,
                    source.occupancy.transferred_bound, source.occupancy.fit_bound};
                for (unsigned i = 1; i < EvidenceCount; ++i)
                    entry.values[i].add(Math::multiply({inherited[i - 1], 0}, share));
                auto& presence = entry.presence;
                // A pure source may carry an independent minority trace label without
                // asserting minority geometry. Mixed-source pieces lacking an
                // absence proof and opposite pure-source pieces invalidate the hint.
                presence[0] = presence[0] || piece_presence[0];
                presence[1] = presence[1] || piece_presence[1];
                history_groups[{atom.donor, atom.recipient}].add(labels.at(atom.donor).history(*atom.geometry, plane));
                work.atom(atom.geometry);
            }
            for (const auto& key : data.topology->order)
            {
                const auto& donor = data.cells.at(key);
                cell_geometry.at(donor.key)->certify_fragment_partition(
                    covers[key], data.options.reconstruction.volume_tolerance);
                PhaseSum<Mode> exported(data.options.arithmetic_mode), retained(data.options.arithmetic_mode),
                    total(data.options.arithmetic_mode);
                for (const auto& [route, sum] : groups)
                    if (route.first == key)
                    {
                        const auto amount = sum.value();
                        total.add(amount);
                        (route.second == key ? retained : exported).add(amount);
                        const auto metadata = target_metadata.at(route.second);
                        amounts.push_back({key, route.second, donor.gid, donor.revision, metadata.second,
                            metadata.first, amount, history_groups.at(route).value(), evidence.at(route).value(),
                            evidence.at(route).presence[0], evidence.at(route).presence[1]});
                    }
                const auto old = phase_inventory<Mode>(donor, *cell_geometry.at(donor.key)), delivered = total.value();
                for (auto pair : {std::pair{old.gas, delivered.gas}, std::pair{old.carrier, delivered.carrier}})
                {
                    const auto difference = Math::subtract(pair.first, pair.second);
                    const auto bound = phase_allowance<Mode>(pair.first, data.options.reconstruction.volume_tolerance);
                    if (!Math::closure_within(difference, bound, pair.first.absolute_error))
                        throw std::invalid_argument("Finite VoF donor phase closure exceeds its independent budget.");
                }
                trial.finite_budgets_.push_back({donor.gid, old, exported.value(), retained.value()});
            }
        });
    packets.clear();
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& amount : amounts)
            {
                wire::Writer w;
                w.bytes(AMR::encode_adaptive_key(amount.donor));
                w.word(amount.donor_gid);
                w.word(amount.donor_revision);
                w.bytes(AMR::encode_adaptive_key(amount.recipient));
                w.word(amount.recipient_gid);
                finite_phases(w, amount.value);
                finite_phases(w, amount.history);
                for (const auto value : amount.provenance)
                    finite_value(w, value);
                w.word(amount.gas_geometry);
                w.word(amount.carrier_geometry);
                budget.packet(w.data.size());
                packets.push_back({amount.recipient_owner, std::move(w.data)});
            }
        });
    const auto incoming_amounts = exchange_owner_packets(packets, comm, limit);
    std::vector<Key> donor_keys;
    struct Transfer
    {
        Key donor, recipient;
        std::uint64_t gid, revision;
        int owner;
        PhaseValues amount, history;
        std::array<PhaseValue, EvidenceCount> provenance;
        bool gas_geometry, carrier_geometry;
    };
    std::vector<Transfer> transfers;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::set<std::pair<Key, Key>> seen;
            for (const auto& packet : incoming_amounts.packets)
            {
                wire::Reader r{packet.bytes};
                auto dk = AMR::decode_adaptive_key(r.bytes());
                const auto gid = r.word(), revision = r.word();
                auto rk = AMR::decode_adaptive_key(r.bytes());
                const auto recipient_gid = r.word();
                const auto& recipient = data.cells.at(rk);
                if (recipient.owner != comm->getRank() || recipient.gid != recipient_gid ||
                    !seen.emplace(dk, rk).second)
                    throw std::invalid_argument("Finite VoF positive transfer recipient is duplicate or foreign.");
                auto amount = finite_phases(r, data.options.arithmetic_mode);
                auto history = finite_phases(r, data.options.arithmetic_mode);
                std::array<PhaseValue, EvidenceCount> provenance;
                for (unsigned i = 0; i < EvidenceCount; ++i)
                {
                    provenance[i] = finite_value(r, data.options.arithmetic_mode);
                    if (i && Math::lower(provenance[i]) < 0)
                        throw std::invalid_argument("Finite VoF inherited occupancy evidence is negative.");
                }
                const auto gas_geometry = r.word(), carrier_geometry = r.word();
                if (gas_geometry > 1 || carrier_geometry > 1 || !(gas_geometry || carrier_geometry))
                    throw std::invalid_argument("Finite VoF positive transfer geometric phase proof is malformed.");
                r.done();
                for (const auto& p : {amount.bulk, amount.gas, amount.carrier})
                    if (p.value < 0 || (p.value != 0 && p.value <= p.absolute_error))
                        throw std::invalid_argument("Finite VoF positive transfer has unresolved nonnegativity.");
                budget.add(sizeof(Transfer));
                donor_keys.push_back(dk);
                transfers.push_back({dk, rk, gid, revision, packet.peer, amount, history, provenance,
                    bool(gas_geometry), bool(carrier_geometry)});
            }
        });
    const auto donor_owners = data.hierarchy->lookup_cells(donor_keys);
    std::shared_ptr<Data> candidate;
    wire::collective(*comm,
        [&]
        {
            candidate = std::make_shared<Data>(data);
            candidate->control = trial.transaction_;
            std::map<Key, PhaseSum<Mode>> totals, histories;
            std::map<Key, Evidence> evidence;
            wire::Budget evidence_budget{limit};
            for (std::size_t i = 0; i < transfers.size(); ++i)
            {
                const auto& transfer = transfers[i];
                const auto& owner = donor_owners[i];
                if (owner.owner != transfer.owner || owner.revision != transfer.revision ||
                    owner.address != transfer.gid)
                    throw std::invalid_argument("Finite VoF positive transfer donor is stale or foreign.");
                totals[transfer.recipient].add(transfer.amount);
                histories[transfer.recipient].add(transfer.history);
                auto [evidence_entry, inserted] = evidence.try_emplace(transfer.recipient);
                if (inserted)
                    evidence_budget.add(sizeof(typename decltype(evidence)::value_type) + 4 * sizeof(void*));
                auto& entry = evidence_entry->second;
                for (unsigned j = 0; j < EvidenceCount; ++j)
                    entry.values[j].add(transfer.provenance[j]);
                auto& presence = entry.presence;
                presence[0] = presence[0] || transfer.gas_geometry;
                presence[1] = presence[1] || transfer.carrier_geometry;
                trial.finite_transfers_.push_back({transfer.gid, data.cells.at(transfer.recipient).gid, transfer.owner,
                    comm->getRank(), transfer.amount,
                    {fp::upper_double(fp::bound_add(transfer.history.gas.value, transfer.history.gas.absolute_error)),
                        fp::upper_double(
                            fp::bound_add(transfer.history.carrier.value, transfer.history.carrier.absolute_error))}});
            }
            candidate->gas.clear();
            candidate->carrier.clear();
            candidate->phases.clear();
            candidate->roundoff.clear();
            candidate->occupancy.clear();
            candidate->phase_history.clear();
            for (const auto& key : data.topology->order)
            {
                auto& cell = candidate->cells.at(key);
                const auto total = totals.at(key).value();
                const auto actual = runtime_value<Mode>(cell_geometry.at(cell.key)->volume());
                const auto closure = Math::subtract(total.bulk, actual);
                const auto geometry_bound = phase_allowance<Mode>(actual, data.options.reconstruction.volume_tolerance);
                if (!Math::closure_within(closure, geometry_bound))
                    throw std::invalid_argument("Finite VoF transfers do not fill the admitted recipient geometry.");
                Math::admit_receipt(total.gas, data.options.reconstruction.volume_tolerance, "gas");
                Math::admit_receipt(total.carrier, data.options.reconstruction.volume_tolerance, "carrier");
                const auto gas = Math::publish(total.gas), carrier = Math::publish(total.carrier);
                cell.phase.gas_volume = gas.first;
                cell.phase.carrier_volume = carrier.first;
                cell.roundoff = {gas.second, carrier.second};
                const auto transported_history = histories.at(key).value();
                cell.finite_history.gas = fp::upper_double(fp::bound_add(
                    gas.second, fp::bound_add(transported_history.gas.value, transported_history.gas.absolute_error)));
                cell.finite_history.carrier = fp::upper_double(fp::bound_add(carrier.second,
                    fp::bound_add(transported_history.carrier.value, transported_history.carrier.absolute_error)));
                const auto inherited = evidence.at(key).value();
                fp::SumFor<Mode> occupancy_sum(data.options.arithmetic_mode);
                occupancy_sum.add(Math::make_value(data.options.arithmetic_mode, gas.first, gas.second));
                occupancy_sum.add(Math::make_value(data.options.arithmetic_mode, carrier.first, carrier.second));
                occupancy_sum.add(Math::negate(actual));
                const auto published = occupancy_sum.value();
                cell.occupancy.difference = static_cast<double>(published.value);
                const auto publication_bound = fp::bound_add(gas.second, carrier.second);
                const auto provenance = allocation_provenance<Mode>(published, inherited[Allocation],
                    inherited[Occupancy], inherited[FitAllocation], closure, publication_bound);
                cell.occupancy.admitted_bound = fp::upper_double(provenance.admitted_bound);
                cell.occupancy.fit_allocation_bound = fp::upper_double(provenance.fit_allocation_bound);
                cell.occupancy.phase_input_bound =
                    fp::upper_double(fp::bound_add(Math::upper(inherited[PhaseInput]), publication_bound));
                cell.occupancy.transferred_bound = fp::upper_double(fp::bound_add(Math::upper(inherited[Transferred]),
                    fp::bound_add(total.gas.absolute_error, total.carrier.absolute_error)));
                cell.retained_interface.reset();
                const auto presence = evidence.at(key).presence;
                if ((cell.declaration == GeneralVoFPhaseGeometry::PureGas && presence[1]) ||
                    (cell.declaration == GeneralVoFPhaseGeometry::PureCarrier && presence[0]))
                    cell.declaration = GeneralVoFPhaseGeometry::Infer;
                cell.occupancy.fit_bound = cell.declaration == GeneralVoFPhaseGeometry::Infer
                                               ? 0
                                               : fp::upper_double(Math::upper(inherited[PureFit]));
                cell.occupancy.fit_residual =
                    cell.declaration == GeneralVoFPhaseGeometry::Infer
                        ? 0
                        : (cell.declaration == GeneralVoFPhaseGeometry::PureGas ? carrier.first : gas.first);
                cell.inherited = true;
                validate_cell<Mode>(cell, data.options);
                candidate->gas.push_back(gas.first);
                candidate->carrier.push_back(carrier.first);
                candidate->phases.push_back({cell.native_volume, gas.first, carrier.first});
                candidate->roundoff.push_back(cell.roundoff);
                candidate->occupancy.push_back(cell.occupancy);
                candidate->phase_history.push_back(cell.finite_history);
            }
            trial.roundoff_ = candidate->roundoff;
            current(data);
        });
    // Shared signed transfers give opposite incidences; an independent global
    // phase check accounts for wire reduction and one-time publication error.
    std::array<double, 8> local{}, global{};
    wire::collective(*comm,
        [&]
        {
            std::array<fp::SumFor<Mode>, 6> sums;
            std::array<fp::SumFor<Mode>, 2> publication;
            for (auto& sum : sums)
                sum = fp::SumFor<Mode>(data.options.arithmetic_mode);
            for (auto& sum : publication)
                sum = fp::SumFor<Mode>(data.options.arithmetic_mode);
            for (std::size_t i = 0; i < data.topology->order.size(); ++i)
            {
                sums[0].add(Math::make_value(data.options.arithmetic_mode,
                    static_cast<long double>(candidate->gas[i]) - data.gas[i], candidate->roundoff[i].gas));
                sums[1].add(Math::make_value(data.options.arithmetic_mode,
                    static_cast<long double>(candidate->carrier[i]) - data.carrier[i], candidate->roundoff[i].carrier));
                sums[2].add({std::abs(static_cast<long double>(candidate->gas[i])) + std::abs(data.gas[i]), 0});
                sums[3].add(Math::make_value(data.options.arithmetic_mode,
                    std::abs(static_cast<long double>(candidate->carrier[i])) + std::abs(data.carrier[i])));
                sums[4].add(Math::make_value(data.options.arithmetic_mode, data.gas[i]));
                sums[5].add(Math::make_value(data.options.arithmetic_mode, data.carrier[i]));
                publication[0].add(Math::make_value(data.options.arithmetic_mode, candidate->roundoff[i].gas));
                publication[1].add(Math::make_value(data.options.arithmetic_mode, candidate->roundoff[i].carrier));
            }
            for (unsigned a = 0; a < 2; ++a)
            {
                const auto sum = sums[a].value();
                local[a] = static_cast<double>(sum.value);
                const auto narrowing = [&]
                {
                    if constexpr (Mode == ArithmeticMode::Ordinary)
                        return publication[a].value().value;
                    else
                        return sum.absolute_error;
                }();
                local[2 + a] = fp::upper_double(fp::upward(narrowing + std::abs(sum.value - local[a])));
                local[4 + a] = fp::upper_double(sums[2 + a].value().value);
                local[6 + a] = fp::upper_double(sums[4 + a].value().value);
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    wire::collective(*comm,
        [&]
        {
            const auto operations = 8.L * comm->getSize() + 16;
            const auto u = operations * std::numeric_limits<double>::epsilon();
            if (!(u < .25L))
                throw std::invalid_argument("Finite VoF global phase reduction bound is exhausted.");
            for (unsigned phase = 0; phase < 2; ++phase)
            {
                const auto physical = [&]
                {
                    if constexpr (Mode == ArithmeticMode::Ordinary)
                        return data.options.reconstruction.volume_tolerance * global[6 + phase];
                    else
                        return 0.L;
                }();
                const auto bound = physical + global[2 + phase] + u / (1 - u) * (global[4 + phase] + global[2 + phase]);
                if (!std::isfinite(global[phase]) || !std::isfinite(bound) || std::abs(global[phase]) > bound)
                    throw std::invalid_argument("Finite VoF global independent phase conservation failed.");
            }
            current(data);
        });
    trial.candidate_ = std::move(candidate);
    return trial;
}
} // namespace SimpleFluid::VoF
