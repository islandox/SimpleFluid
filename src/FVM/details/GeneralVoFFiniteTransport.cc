/** @file GeneralVoFFiniteTransport.cc
 * @brief Finite phase transport and conservative remap planning.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF
{
namespace
{
namespace fp = finite_detail;
using FValue = Meshes::BoundedGeometryValue;
using FPoint = Meshes::BoundedPoint;
using FPhases = Meshes::BoundedPhaseMeasures;
void finite_point(wire::Writer& w, const FPoint& p)
{
    for (unsigned a = 0; a < 3; ++a)
        finite_value(w, {p.value[a], p.absolute_error[a]});
}
FPoint finite_point(wire::Reader& r)
{
    FPoint p;
    for (unsigned a = 0; a < 3; ++a)
    {
        const auto v = finite_value(r);
        p.value[a] = v.value;
        p.absolute_error[a] = v.absolute_error;
    }
    return p;
}
void finite_phases(wire::Writer& w, const FPhases& p)
{
    finite_value(w, p.bulk);
    finite_value(w, p.gas);
    finite_value(w, p.carrier);
}
FPhases finite_phases(wire::Reader& r)
{
    return {finite_value(r), finite_value(r), finite_value(r)};
}
struct FPhaseSum
{
    fp::Sum bulk, gas, carrier;
    void add(const FPhases& p, int sign = 1)
    {
        bulk.add(sign > 0 ? p.bulk : fp::negate(p.bulk));
        gas.add(sign > 0 ? p.gas : fp::negate(p.gas));
        carrier.add(sign > 0 ? p.carrier : fp::negate(p.carrier));
    }
    FPhases value() const { return {bulk.value(), gas.value(), carrier.value()}; }
};
FPhases finite_inventory(const Cell& cell)
{
    return {cell.finite_geometry->volume(), {cell.phase.gas_volume, 0}, {cell.phase.carrier_volume, 0}};
}
FPoint constrained_finite_departure(const GeneralVoF::Data& data, std::uint64_t node,
    const Meshes::GeometryPoint& velocity, double dt)
{
    std::vector<Meshes::BoundedFaceConstraint> walls;
    for (const auto& wall : data.topology->node_walls.at(node))
        walls.push_back(wall.finite_constraint);
    if (walls.size() > Meshes::ExactTrajectoryLimits{}.maximum_constraints)
        throw std::invalid_argument("Finite VoF complete wall catalog exceeds the unchanged constraint limit.");
    Meshes::BoundedGeometryLimits limits;
    limits.byte_budget = data.options.byte_budget;
    limits.maximum_operations = Meshes::ExactTrajectoryLimits{}.maximum_operations;
    return Meshes::bounded_backtrace_constrained(data.topology->finite_nodes.at(node), velocity, dt, walls, limits);
}
struct FLabel
{
    FValue gas_factor, carrier_factor, gas_history_factor, carrier_history_factor;
    bool gas_bulk = false, carrier_bulk = false;
    FPhases apply(const Meshes::BoundedConvexGeometry& geometry, const PolyhedralInterface& plane) const
    {
        const auto bulk = geometry.volume();
        const auto raw = (plane.pure_gas || plane.pure_carrier) ? FPhases{bulk, {}, {}} : geometry.phases(plane.gas);
        return {bulk, fp::multiply(gas_bulk ? bulk : raw.gas, gas_factor),
            fp::multiply(carrier_bulk ? bulk : raw.carrier, carrier_factor)};
    }
    FPhases history(const Meshes::BoundedConvexGeometry& geometry, const PolyhedralInterface& plane) const
    {
        const auto bulk = geometry.volume();
        const auto raw = (plane.pure_gas || plane.pure_carrier) ? FPhases{bulk, {}, {}} : geometry.phases(plane.gas);
        return {{}, fp::multiply(gas_bulk ? bulk : raw.gas, gas_history_factor),
            fp::multiply(carrier_bulk ? bulk : raw.carrier, carrier_history_factor)};
    }
};
FLabel finite_label(const Cell& cell, const PolyhedralInterface& plane, double tolerance)
{
    const auto inventory = finite_inventory(cell);
    FLabel result;
    if (plane.pure_gas || plane.pure_carrier)
    {
        result.gas_bulk = result.carrier_bulk = true;
        result.gas_factor = fp::divide(inventory.gas, inventory.bulk);
        result.carrier_factor = fp::divide(inventory.carrier, inventory.bulk);
        result.gas_history_factor = fp::divide({cell.finite_history.gas, 0}, inventory.bulk);
        result.carrier_history_factor = fp::divide({cell.finite_history.carrier, 0}, inventory.bulk);
        return result;
    }
    const auto raw = cell.finite_geometry->phases(plane.gas);
    const auto check = [&](FValue geometric, FValue physical)
    {
        if (physical.value == 0)
            return FValue{};
        const auto defect = fp::subtract(geometric, physical);
        const auto bound = std::abs(physical.value) * tolerance;
        if (std::abs(defect.value) + defect.absolute_error > bound + physical.absolute_error)
            throw std::invalid_argument("Finite VoF phase fit exceeds its independent phase budget.");
        return fp::divide(physical, geometric);
    };
    // The minority fit determines the plane; the dominant label additionally
    // includes the already admitted physical/native occupancy discrepancy.
    result.gas_factor = fp::divide(inventory.gas, raw.gas);
    result.carrier_factor = fp::divide(inventory.carrier, raw.carrier);
    result.gas_history_factor = fp::divide({cell.finite_history.gas, 0}, raw.gas);
    result.carrier_history_factor = fp::divide({cell.finite_history.carrier, 0}, raw.carrier);
    if (inventory.gas.value <= inventory.carrier.value)
        result.gas_factor = check(raw.gas, inventory.gas);
    else
        result.carrier_factor = check(raw.carrier, inventory.carrier);
    return result;
}
} // namespace
GeneralVoF::Trial GeneralVoF::plan_transport(std::span<const GeneralVoFNodeVelocity> node_velocity,
    std::span<const GeneralVoFFaceVolume> face_volume, double timestep) const
{
    if (inventory_policy() != PhaseInventoryPolicy::Finite)
        return Exact::GeneralVoF::plan_transport(*this, node_velocity, face_volume, timestep);
    const auto& data = *data_;
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    Meshes::BoundedGeometryPacketContext packet_context{{data.control,
        data.hierarchy->topology_generation(), data.hierarchy->partition_generation(), 0}};
    std::map<std::uint64_t, std::pair<FPoint, FPoint>> traces;
    std::map<std::uint64_t, FValue> flux;
    std::vector<OwnerPacket> packets;
    wire::collective(*comm, [&]
    {
        current(data);
        if (!std::isfinite(timestep) || timestep < 0 || node_velocity.size() != data.topology->nodes.size() ||
            face_volume.size() != data.topology->tiles.size())
            throw std::invalid_argument("Finite VoF controls do not cover retained nodes/faces.");
        wire::Budget budget{limit};
        std::map<std::uint64_t, Meshes::GeometryPoint> velocity;
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
            const auto& source = data.topology->finite_nodes.at(node.gid);
            const auto departure = constrained_finite_departure(data, node.gid, velocity.at(node.gid), timestep);
            budget.add(sizeof(decltype(traces)::value_type) + 4 * sizeof(void*) +
                Meshes::bounded_point_storage_bytes(source) + Meshes::bounded_point_storage_bytes(departure));
            traces.emplace(node.gid, std::pair{source, departure});
            wire::Writer w;
            w.word(0);
            w.word(node.gid);
            w.bytes(node.identity);
            finite_point(w, source);
            finite_point(w, departure);
            for (auto x : velocity.at(node.gid))
                real(w, x);
            real(w, timestep);
            w.word(data.topology->node_walls.at(node.gid).size());
            for (const auto& wall : data.topology->node_walls.at(node.gid))
            {
                w.bytes(wall.identity);
                finite_point(w, wall.finite_normal);
            }
            budget.packet(w.data.size());
            packets.push_back({int(node.gid % comm->getSize()), std::move(w.data)});
        }
        for (const auto& face : face_volume)
        {
            if (face.exact_integrated_volume || !std::isfinite(face.integrated_volume))
                throw std::invalid_argument("Finite VoF cannot adopt an exact-policy face control.");
            const auto value = face.bounded_integrated_volume.value_or(FValue{face.integrated_volume, 0});
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
            wire::Writer w;
            w.word(1);
            w.word(tile.face_gid);
            w.bytes(tile.domain);
            w.bytes(tile.surface);
            w.bytes(tile.identity);
            finite_value(w, phi);
            budget.packet(w.data.size());
            packets.push_back({int(tile.face_gid % comm->getSize()), std::move(w.data)});
        }
        wire::Writer w;
        w.word(2);
        w.word(0);
        w.word(data.control);
        real(w, timestep);
        packets.push_back({0, std::move(w.data)});
    });
    const auto controls = exchange_owner_packets(packets, comm, limit);
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<char>> canonical;
        for (const auto& packet : controls.packets)
        {
            wire::Reader r{packet.bytes};
            const auto type = r.word(), id = r.word();
            if (type > 2 || int(id % comm->getSize()) != comm->getRank())
                throw std::invalid_argument("Finite VoF control reached a foreign authority.");
            budget.add(packet.bytes.size() + sizeof(decltype(canonical)::value_type));
            auto [it, inserted] = canonical.emplace(std::pair{type, id}, packet.bytes);
            if (!inserted && it->second != packet.bytes)
                throw std::invalid_argument("Finite VoF shared node/face controls disagree.");
        }
        for (const auto& key : data.topology->order)
        {
            fp::Sum sum;
            for (auto index : data.topology->cell_tiles.at(key))
            {
                const auto& tile = data.topology->tiles[index];
                const int sign = tile.first == key ? 1 : -1;
                sum.add(sign > 0 ? flux.at(tile.face_gid) : fp::negate(flux.at(tile.face_gid)));
            }
            const auto residual = sum.value();
            if (std::abs(residual.value) > residual.absolute_error || residual.absolute_error >
                data.cells.at(key).finite_geometry->volume().value * data.options.reconstruction.volume_tolerance)
                throw std::invalid_argument("Finite VoF canonical controls violate source-free occupancy.");
        }
    });
    Trial trial;
    trial.source_ = data_;
    trial.transaction_ = transaction(comm);
    packet_context.publication[3] = trial.transaction_;
    int local_motion = 0, motion = 0;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        budget.add(traces.size() * sizeof(GeneralVoFFiniteNodeTrajectory));
        trial.finite_trajectories_.reserve(traces.size());
        for (const auto& [id, trace] : traces)
        {
            local_motion |= !fp::same_point(trace.first, trace.second);
            trial.finite_trajectories_.push_back({id, trace.first, trace.second});
        }
        for (const auto& [face, value] : flux) local_motion |= value.value != 0;
    });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_motion, &motion);
    trial.roundoff_.resize(data.topology->order.size());
    if (!motion)
    {
        trial.candidate_ = data_;
        for (const auto& key : data.topology->order)
        {
            const auto& cell = data.cells.at(key);
            const auto old = finite_inventory(cell);
            trial.finite_transfers_.push_back({cell.gid, cell.gid, cell.owner, cell.owner, old, cell.finite_history});
            trial.finite_budgets_.push_back({cell.gid, old, {}, old});
        }
        return trial;
    }
    const auto planes = import_planes(data);
    std::map<Key, FLabel> labels;
    std::map<std::uint64_t, fp::Sweep> sweeps;
    std::vector<fp::Atom> local_atoms;
    std::map<std::tuple<std::uint64_t, std::uint64_t, int>, FPhaseSum> face_amounts;
    wire::collective(*comm, [&]
    {
        fp::Work work{data.options.sweep};
        work.limits.byte_budget = std::min(limit, work.limits.byte_budget);
        for (const auto& [key, cell] : data.cells)
            labels.emplace(key, finite_label(cell, planes.at(key), data.options.reconstruction.volume_tolerance));
        std::map<std::uint64_t, Meshes::BoundedConvexGeometryPtr> donors;
        for (const auto& [key, cell] : data.cells)
            donors.emplace(cell.gid, cell.finite_geometry);
        for (const auto& tile : data.topology->tiles)
        {
            work.pair_tests = 0; // Existing face-partition work-cap scope.
            auto sweep = fp::make_sweep(tile.face_gid, tile.node_gids, traces, flux.at(tile.face_gid), work);
            // Each signed simplex is covered by actual nonoverlapping native
            // donors before it can contribute a raw canonical face amount.
            for (const auto& piece : sweep.pieces)
            {
                std::vector<Meshes::BoundedConvexGeometryPtr> cover;
                for (const auto& [key, cell] : data.cells)
                {
                    work.test();
                    if (auto intersection = piece.geometry->intersect(*cell.finite_geometry))
                    {
                        work.atom(intersection);
                        cover.push_back(intersection);
                        if (tile.first_owner == comm->getRank())
                            face_amounts[{tile.face_gid, cell.gid, piece.sign}].add(
                                labels.at(key).apply(*intersection, planes.at(key)));
                    }
                }
                piece.geometry->certify_partition(cover, data.options.reconstruction.volume_tolerance);
            }
            sweeps.emplace(tile.face_gid, std::move(sweep));
        }
        for (const auto& key : data.topology->order)
        {
            const auto& cell = data.cells.at(key);
            std::vector<std::pair<const fp::Sweep*, int>> incident;
            for (auto index : data.topology->cell_tiles.at(key))
            {
                const auto& tile = data.topology->tiles[index];
                incident.push_back({&sweeps.at(tile.face_gid), tile.first == key ? 1 : -1});
            }
            work.pair_tests = 0; // Existing recipient-local partition work-cap scope.
            auto atoms = fp::recipient_partition(cell.gid, donors, incident, work,
                data.options.reconstruction.volume_tolerance);
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
    wire::collective(*comm, [&]
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
                w.bytes(atom.geometry->encode(packet_context, limit));
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
        Meshes::BoundedConvexGeometryPtr geometry;
    };
    std::vector<Routed> routed;
    std::vector<Key> recipient_keys;
    std::vector<Meshes::BoundedConvexGeometryPtr> source_registry;
    for (const auto& [key, cell] : data.cells)
        source_registry.push_back(cell.finite_geometry);
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& packet : routed_packets.packets)
        {
            wire::Reader r{packet.bytes};
            auto donor_key = AMR::decode_adaptive_key(r.bytes());
            const auto revision = r.word(), gid = r.word();
            auto recipient_key = AMR::decode_adaptive_key(r.bytes());
            const auto recipient_gid = r.word(), local = r.word();
            const auto& donor = data.cells.at(donor_key);
            if (donor.owner != comm->getRank() || donor.gid != gid || donor.revision != revision)
                throw std::invalid_argument("Finite VoF atom donor authentication failed.");
            Meshes::BoundedConvexGeometryPtr geometry;
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
                Meshes::BoundedGeometryLimits decode_limits;
                decode_limits.byte_budget = limit;
                geometry = Meshes::BoundedConvexGeometry::decode(r.bytes(), packet_context, source_registry, decode_limits);
            }
            else
                throw std::invalid_argument("Finite VoF atom routing tag is malformed.");
            r.done();
            budget.add(sizeof(Routed) + geometry->storage_bytes());
            recipient_keys.push_back(recipient_key);
            routed.push_back({donor_key, recipient_key, recipient_gid, packet.peer, std::move(geometry)});
        }
    });
    const auto recipients = data.hierarchy->lookup_cells(recipient_keys);
    struct Amount
    {
        Key donor, recipient;
        std::uint64_t donor_gid = 0, donor_revision = 0, recipient_gid = 0;
        int recipient_owner = -1;
        FPhases value, history;
    };
    std::vector<Amount> amounts;
    wire::collective(*comm, [&]
    {
        fp::Work work{data.options.sweep};
        work.limits.byte_budget = std::min(limit, work.limits.byte_budget);
        std::map<Key, std::vector<Meshes::BoundedConvexGeometryPtr>> covers;
        std::map<std::pair<Key, Key>, FPhaseSum> groups, history_groups;
        std::map<Key, std::pair<int, std::uint64_t>> target_metadata;
        for (std::size_t i = 0; i < routed.size(); ++i)
        {
            const auto& atom = routed[i];
            if (recipients[i].owner != atom.recipient_owner || recipients[i].address != atom.recipient_gid)
                throw std::invalid_argument("Finite VoF atom recipient owner authentication failed.");
            target_metadata.emplace(atom.recipient, std::pair{atom.recipient_owner, atom.recipient_gid});
            covers[atom.donor].push_back(atom.geometry);
            groups[{atom.donor, atom.recipient}].add(labels.at(atom.donor).apply(*atom.geometry, planes.at(atom.donor)));
            history_groups[{atom.donor, atom.recipient}].add(labels.at(atom.donor).history(*atom.geometry, planes.at(atom.donor)));
            work.atom(atom.geometry);
        }
        for (const auto& key : data.topology->order)
        {
            const auto& donor = data.cells.at(key);
            donor.finite_geometry->certify_partition(covers[key], data.options.reconstruction.volume_tolerance);
            FPhaseSum exported, retained, total;
            for (const auto& [route, sum] : groups)
                if (route.first == key)
                {
                    const auto amount = sum.value();
                    total.add(amount);
                    (route.second == key ? retained : exported).add(amount);
                    const auto metadata = target_metadata.at(route.second);
                    amounts.push_back({key, route.second, donor.gid, donor.revision,
                        metadata.second, metadata.first, amount, history_groups.at(route).value()});
                }
            const auto old = finite_inventory(donor), delivered = total.value();
            for (auto pair : {std::pair{old.gas, delivered.gas}, std::pair{old.carrier, delivered.carrier}})
            {
                const auto difference = fp::subtract(pair.first, pair.second);
                if (std::abs(difference.value) > difference.absolute_error || difference.absolute_error >
                    std::abs(pair.first.value) * data.options.reconstruction.volume_tolerance + pair.first.absolute_error)
                    throw std::invalid_argument("Finite VoF donor phase closure exceeds its independent budget.");
            }
            trial.finite_budgets_.push_back({donor.gid, old, exported.value(), retained.value()});
        }
    });
    packets.clear();
    wire::collective(*comm, [&]
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
        FPhases amount, history;
    };
    std::vector<Transfer> transfers;
    wire::collective(*comm, [&]
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
            if (recipient.owner != comm->getRank() || recipient.gid != recipient_gid || !seen.emplace(dk, rk).second)
                throw std::invalid_argument("Finite VoF positive transfer recipient is duplicate or foreign.");
            auto amount = finite_phases(r);
            auto history = finite_phases(r);
            r.done();
            for (const auto& p : {amount.bulk, amount.gas, amount.carrier})
                if (p.value < 0 || (p.value != 0 && p.value <= p.absolute_error))
                    throw std::invalid_argument("Finite VoF positive transfer has unresolved nonnegativity.");
            budget.add(sizeof(Transfer));
            donor_keys.push_back(dk);
            transfers.push_back({dk, rk, gid, revision, packet.peer, amount, history});
        }
    });
    const auto donor_owners = data.hierarchy->lookup_cells(donor_keys);
    std::shared_ptr<Data> candidate;
    wire::collective(*comm, [&]
    {
        candidate = std::make_shared<Data>(data);
        candidate->control = trial.transaction_;
        std::map<Key, FPhaseSum> totals, histories;
        for (std::size_t i = 0; i < transfers.size(); ++i)
        {
            const auto& transfer = transfers[i];
            const auto& owner = donor_owners[i];
            if (owner.owner != transfer.owner || owner.revision != transfer.revision || owner.address != transfer.gid)
                throw std::invalid_argument("Finite VoF positive transfer donor is stale or foreign.");
            totals[transfer.recipient].add(transfer.amount);
            histories[transfer.recipient].add(transfer.history);
            trial.finite_transfers_.push_back({transfer.gid, data.cells.at(transfer.recipient).gid,
                transfer.owner, comm->getRank(), transfer.amount,
                {fp::upper_double(fp::bound_add(transfer.history.gas.value, transfer.history.gas.absolute_error)),
                 fp::upper_double(fp::bound_add(transfer.history.carrier.value, transfer.history.carrier.absolute_error))}});
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
            const auto actual = cell.finite_geometry->volume();
            const auto closure = fp::subtract(total.bulk, actual);
            if (std::abs(closure.value) > closure.absolute_error || closure.absolute_error >
                actual.value * data.options.reconstruction.volume_tolerance)
                throw std::invalid_argument("Finite VoF transfers do not fill the admitted recipient geometry.");
            const auto gas = fp::publish(total.gas), carrier = fp::publish(total.carrier);
            cell.phase.gas_volume = gas.first;
            cell.phase.carrier_volume = carrier.first;
            cell.roundoff = {gas.second, carrier.second};
            const auto transported_history = histories.at(key).value();
            cell.finite_history.gas = fp::upper_double(fp::bound_add(gas.second,
                fp::bound_add(transported_history.gas.value, transported_history.gas.absolute_error)));
            cell.finite_history.carrier = fp::upper_double(fp::bound_add(carrier.second,
                fp::bound_add(transported_history.carrier.value, transported_history.carrier.absolute_error)));
            const auto previous = data.cells.at(key).occupancy;
            fp::Sum occupancy_sum;
            occupancy_sum.add({gas.first, gas.second});
            occupancy_sum.add({carrier.first, carrier.second});
            occupancy_sum.add(fp::negate(actual));
            const auto published = occupancy_sum.value();
            cell.occupancy.difference = static_cast<double>(published.value);
            cell.occupancy.admitted_bound = fp::upper_double(fp::upward(previous.admitted_bound +
                published.absolute_error + std::abs(closure.value)));
            cell.occupancy.phase_input_bound = fp::upper_double(fp::upward(previous.phase_input_bound +
                gas.second + carrier.second));
            cell.occupancy.transferred_bound = fp::upper_double(fp::upward(previous.transferred_bound +
                total.gas.absolute_error + total.carrier.absolute_error));
            cell.retained_interface.reset();
            if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
            {
                const auto minority = cell.declaration == GeneralVoFPhaseGeometry::PureGas ? carrier.first : gas.first;
                if (minority > actual.absolute_error + 8 * std::numeric_limits<double>::epsilon() * actual.value)
                    cell.declaration = GeneralVoFPhaseGeometry::Infer;
            }
            cell.inherited = true;
            validate_cell(cell, data.options);
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
    std::array<double, 6> local{}, global{};
    wire::collective(*comm, [&]
    {
        std::array<fp::Sum, 4> sums;
        for (std::size_t i = 0; i < data.topology->order.size(); ++i)
        {
            sums[0].add({static_cast<long double>(candidate->gas[i]) - data.gas[i], candidate->roundoff[i].gas});
            sums[1].add({static_cast<long double>(candidate->carrier[i]) - data.carrier[i], candidate->roundoff[i].carrier});
            sums[2].add({std::abs(static_cast<long double>(candidate->gas[i])) + std::abs(data.gas[i]), 0});
            sums[3].add({std::abs(static_cast<long double>(candidate->carrier[i])) + std::abs(data.carrier[i]), 0});
        }
        for (unsigned a = 0; a < 2; ++a)
        {
            const auto sum = sums[a].value();
            local[a] = static_cast<double>(sum.value);
            local[2 + a] = fp::upper_double(fp::upward(sum.absolute_error + std::abs(sum.value - local[a])));
            local[4 + a] = fp::upper_double(sums[2 + a].value().value);
        }
    });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    wire::collective(*comm, [&]
    {
        const auto operations = 8.L * comm->getSize() + 16;
        const auto u = operations * std::numeric_limits<double>::epsilon();
        if (!(u < .25L))
            throw std::invalid_argument("Finite VoF global phase reduction bound is exhausted.");
        for (unsigned phase = 0; phase < 2; ++phase)
        {
            const auto bound = global[2 + phase] + u / (1 - u) * (global[4 + phase] + global[2 + phase]);
            if (!std::isfinite(global[phase]) || !std::isfinite(bound) || std::abs(global[phase]) > bound)
                throw std::invalid_argument("Finite VoF global independent phase conservation failed.");
        }
        current(data);
    });
    trial.candidate_ = std::move(candidate);
    return trial;
}
namespace
{
FValue finite_remap_weighted(const FPhases& p, GeneralVoFFiniteRemapCarrier carrier,
    const std::array<double, 2>& density)
{
    switch (carrier)
    {
    case GeneralVoFFiniteRemapCarrier::BulkVolume: return p.bulk;
    case GeneralVoFFiniteRemapCarrier::GasVolume: return p.gas;
    case GeneralVoFFiniteRemapCarrier::CarrierVolume: return p.carrier;
    case GeneralVoFFiniteRemapCarrier::MixtureMass:
        return fp::add(fp::multiply(p.gas, {density[0], 0}), fp::multiply(p.carrier, {density[1], 0}));
    }
    throw std::invalid_argument("Unknown finite remap channel carrier.");
}
void finite_remap_error(wire::Writer& w, const PhaseError& e)
{
    for (double v : {e.difference, e.admitted_bound, e.native_conversion, e.phase_input_bound,
             e.fit_residual, e.fit_bound, e.transferred_bound, e.fit_allocation_bound})
        real(w, v);
}
PhaseError finite_remap_error(wire::Reader& r)
{
    PhaseError e;
    e.difference = real(r); e.admitted_bound = real(r); e.native_conversion = real(r);
    e.phase_input_bound = real(r); e.fit_residual = real(r); e.fit_bound = real(r);
    e.transferred_bound = real(r); e.fit_allocation_bound = real(r);
    return e;
}
} // namespace

void GeneralVoF::FiniteRemapTrial::validate_observations() const
{
    if (!source_)
        throw std::invalid_argument("Finite remap has no authenticated source.");
    wire::collective(*source_->hierarchy->communicator(), [&]
    {
        current(*source_);
        if (!candidate_ || source_->options.inventory_policy != PhaseInventoryPolicy::Finite ||
            candidate_->options.inventory_policy != PhaseInventoryPolicy::Finite)
            throw std::invalid_argument("Finite remap has no compatible candidate.");
        current(*candidate_);
    });
}
std::shared_ptr<const GeneralVoF> GeneralVoF::FiniteRemapTrial::candidate() const
{
    validate_observations();
    return std::shared_ptr<const GeneralVoF>(new GeneralVoF(candidate_));
}
GeneralVoF::FiniteRemapTrial GeneralVoF::plan_remap(const Hierarchy::Plan& plan,
    std::shared_ptr<const Hierarchy> target, std::span<const GeneralVoFFiniteOwnedChannels> channels,
    GeneralVoFFiniteRemapOptions options) const
{
    const auto& data = *data_;
    const auto comm = data.hierarchy->communicator();
    const auto limit = std::min(data.options.byte_budget, options.byte_budget);
    std::map<Key, std::vector<FValue>> source_channels;
    std::vector<char> manifest;
    wire::collective(*comm, [&]
    {
        current(data);
        if (data.options.inventory_policy != PhaseInventoryPolicy::Finite || !target ||
            target->communicator().getRawPtr() != comm.getRawPtr() || !target->native_partition() || !limit ||
            !plan.collective_identity() || !std::isfinite(options.maximum_relative_target_defect) ||
            !(options.maximum_relative_target_defect > 0) || !std::isfinite(options.maximum_relative_target_fit) ||
            !(options.maximum_relative_target_fit > 0))
            throw std::invalid_argument("Finite remap requires compatible source, actual native target and options.");
        wire::Budget budget{limit};
        wire::Writer w;
        w.word(data.control);
        w.word(plan.collective_identity());
        w.word(options.channels.size());
        for (double density : options.phase_density)
        {
            if (!std::isfinite(density) || !(density > 0))
                throw std::invalid_argument("Finite remap phase density must be finite and positive.");
            real(w, density);
        }
        real(w, options.maximum_relative_target_defect);
        real(w, options.maximum_relative_target_fit);
        std::set<std::string> names;
        for (const auto& channel : options.channels)
        {
            if (channel.name.empty() || !names.insert(channel.name).second ||
                static_cast<unsigned>(channel.carrier) > static_cast<unsigned>(GeneralVoFFiniteRemapCarrier::MixtureMass))
                throw std::invalid_argument("Finite remap channel schema is malformed.");
            budget.add(channel.name.size() + sizeof(GeneralVoFFiniteRemapChannel));
            w.bytes(std::vector<char>(channel.name.begin(), channel.name.end()));
            w.word(static_cast<unsigned>(channel.carrier));
            w.word(channel.nonnegative);
        }
        budget.packet(w.data.size());
        manifest = std::move(w.data);
        if (options.channels.empty() && channels.empty())
            for (const auto& key : data.topology->order)
                source_channels.emplace(key, std::vector<FValue>{});
        else
            for (const auto& item : channels)
            {
                if (!data.hierarchy->owned().contains(item.key) || item.amounts.size() != options.channels.size() ||
                    !source_channels.emplace(item.key, item.amounts).second)
                    throw std::invalid_argument("Finite remap source channels have duplicate, foreign or missing keys.");
                budget.add(sizeof(GeneralVoFFiniteOwnedChannels) + item.amounts.size() * sizeof(FValue));
                for (std::size_t i = 0; i < item.amounts.size(); ++i)
                {
                    const auto v = item.amounts[i];
                    if (!std::isfinite(v.value) || !std::isfinite(v.absolute_error) || v.absolute_error < 0 ||
                        (options.channels[i].nonnegative && (v.lower() < 0 || v.value < 0)))
                        throw std::invalid_argument("Finite remap source channel is nonfinite or unresolved nonnegative.");
                }
            }
        if (source_channels.size() != data.topology->order.size())
            throw std::invalid_argument("Finite remap channels do not cover every source owner.");
    });
    std::uint64_t manifest_size = manifest.size();
    Teuchos::broadcast(*comm, 0, 1, &manifest_size);
    std::vector<char> agreed;
    wire::collective(*comm, [&]
    {
        if (manifest_size > limit || manifest_size > std::uint64_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Finite remap collective schema exceeds its payload cap.");
        agreed = comm->getRank() == 0 ? manifest : std::vector<char>(std::size_t(manifest_size));
    });
    Teuchos::broadcast(*comm, 0, int(manifest_size), agreed.data());
    wire::collective(*comm, [&]
    {
        if (agreed != manifest)
            throw std::invalid_argument("Finite remap schema, source publication or plan differs across ranks.");
    });
    target = data.hierarchy->apply(plan, std::move(target));
    const auto target_observation = target->observation();
    const bool ownership = target->topology_generation() == data.hierarchy->topology_generation();
    const auto catalog = detail::build_adaptive_overlap_catalog(*data.hierarchy, *target, limit);
    std::map<std::uint64_t, const AMR::AdaptiveCellBinding*> from, to;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& cell : catalog.source_cells)
        {
            budget.add(sizeof(decltype(from)::value_type) + 4 * sizeof(void*));
            if (!from.emplace(cell.gid, &cell).second)
                throw std::invalid_argument("Finite remap source catalog repeats a global ID.");
        }
        for (const auto& cell : catalog.target_cells)
        {
            budget.add(sizeof(decltype(to)::value_type) + 4 * sizeof(void*));
            if (!to.emplace(cell.gid, &cell).second)
                throw std::invalid_argument("Finite remap target catalog repeats a global ID.");
        }
    });
    struct Relation
    {
        Key source, target;
        std::uint64_t source_gid, source_revision, target_gid, target_revision;
        AMR::AdaptiveAncestryRelation::Kind kind;
        int target_owner;
        Meshes::BoundedConvexGeometryPtr overlap;
    };
    std::vector<Relation> requests;
    std::vector<Key> source_keys;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& relation : catalog.relations)
        {
            const auto& source_cell = *from.at(relation.source_gid);
            const auto& target_cell = *to.at(relation.target_gid);
            if (!target->owned().contains(target_cell.key))
                continue;
            budget.add(sizeof(Relation) + 2 * sizeof(Key));
            source_keys.push_back(source_cell.key);
            requests.push_back({source_cell.key, target_cell.key, source_cell.gid, 0, target_cell.gid,
                target->owned().at(target_cell.key).revision, relation.kind, comm->getRank(), {}});
        }
    });
    const auto locations = data.hierarchy->lookup_cells(source_keys);
    std::vector<OwnerPacket> packets;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (std::size_t i = 0; i < requests.size(); ++i)
        {
            auto& request = requests[i];
            request.source_revision = locations[i].revision;
            if (request.source_gid != locations[i].address)
                throw std::invalid_argument("Finite remap source catalog is stale.");
            wire::Writer w;
            w.bytes(AMR::encode_adaptive_key(request.source));
            w.word(request.source_gid); w.word(request.source_revision);
            w.bytes(AMR::encode_adaptive_key(request.target));
            w.word(request.target_gid); w.word(request.target_revision);
            w.word(static_cast<unsigned>(request.kind));
            budget.packet(w.data.size());
            packets.push_back({locations[i].owner, std::move(w.data)});
        }
    });
    const auto incoming = exchange_owner_packets(packets, comm, limit);
    std::vector<Relation> columns;
    std::vector<Key> target_keys;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        std::set<std::pair<Key, Key>> seen;
        std::map<std::pair<Key, Key>, AMR::AdaptiveAncestryRelation::Kind> admitted_relations;
        for (const auto& relation : catalog.relations)
            admitted_relations.emplace(std::pair{from.at(relation.source_gid)->key,
                to.at(relation.target_gid)->key}, relation.kind);
        for (const auto& packet : incoming.packets)
        {
            wire::Reader r{packet.bytes};
            Relation column;
            column.source = AMR::decode_adaptive_key(r.bytes());
            column.source_gid = r.word(); column.source_revision = r.word();
            column.target = AMR::decode_adaptive_key(r.bytes());
            column.target_gid = r.word(); column.target_revision = r.word();
            const auto kind = r.word();
            r.done();
            if (!data.hierarchy->owned().contains(column.source) ||
                !seen.emplace(column.source, column.target).second ||
                kind > static_cast<unsigned>(AMR::AdaptiveAncestryRelation::Kind::ProductIntersection))
                throw std::invalid_argument("Finite remap received duplicate, foreign or malformed overlap request.");
            column.kind = static_cast<AMR::AdaptiveAncestryRelation::Kind>(kind);
            const auto authenticated = admitted_relations.find({column.source, column.target});
            if (authenticated == admitted_relations.end() || authenticated->second != column.kind)
                throw std::invalid_argument("Finite remap request does not match authenticated native ancestry.");
            const auto& cell = data.cells.at(column.source);
            if (cell.gid != column.source_gid || cell.revision != column.source_revision)
                throw std::invalid_argument("Finite remap overlap source revision differs.");
            column.target_owner = packet.peer;
            budget.add(sizeof(Relation) + sizeof(Key));
            target_keys.push_back(column.target);
            columns.push_back(std::move(column));
        }
    });
    const auto target_locations = target->lookup_cells(target_keys);
    const auto target_leaves = target->fetch_leaves(target_keys, limit);
    std::map<Key, Meshes::BoundedConvexGeometryPtr> imported_targets;
    std::map<Key, std::vector<std::size_t>> by_source;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (std::size_t i = 0; i < columns.size(); ++i)
        {
            auto& column = columns[i];
            const auto& location = target_locations[i];
            const auto& leaf = target_leaves[i];
            const auto& source_leaf = data.hierarchy->owned().at(column.source);
            if (location.owner != column.target_owner || location.address != column.target_gid ||
                location.revision != column.target_revision || leaf.key != column.target ||
                leaf.revision != column.target_revision || leaf.material != source_leaf.material ||
                leaf.key.region != source_leaf.key.region)
                throw std::invalid_argument("Finite remap target owner/material/revision differs.");
            if (!imported_targets.contains(column.target))
            {
                Meshes::BoundedGeometryLimits limits;
                limits.byte_budget = limit;
                auto geometry = Meshes::BoundedConvexGeometry::from_geometry(*target->geometry(leaf), limits);
                budget.add(geometry->storage_bytes());
                imported_targets.emplace(column.target, std::move(geometry));
            }
            const auto& source_geometry = data.cells.at(column.source).finite_geometry;
            const auto& target_geometry = imported_targets.at(column.target);
            // Native plan ancestry authenticates exact containment for immediate
            // refinement/coarsening. Numerical partition checks below still
            // establish that the supplied whole pieces form a complete cover.
            if (column.kind == AMR::AdaptiveAncestryRelation::Kind::Unchanged ||
                column.kind == AMR::AdaptiveAncestryRelation::Kind::Coarsening)
                column.overlap = source_geometry;
            else if (column.kind == AMR::AdaptiveAncestryRelation::Kind::Refinement)
                column.overlap = target_geometry;
            else
                column.overlap = source_geometry->intersect(*target_geometry);
            if (!column.overlap)
                throw std::invalid_argument("Finite remap ancestry overlap is geometrically empty.");
            budget.add(sizeof(std::size_t) + column.overlap->storage_bytes());
            by_source[column.source].push_back(i);
        }
    });
    struct Output
    {
        Relation relation;
        FPhases phases, history;
        std::vector<FValue> amounts;
        FValue inherited_occupancy;
        PhaseError original_error;
        PhaseRoundoff original_roundoff;
        GeneralVoFPhaseGeometry declaration = GeneralVoFPhaseGeometry::Infer;
        std::optional<PolyhedralInterface> retained_interface;
    };
    std::vector<Output> outgoing;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& key : data.topology->order)
        {
            const auto& source = data.cells.at(key);
            const auto found = by_source.find(key);
            if (found == by_source.end() || found->second.empty())
                throw std::invalid_argument("Finite remap has an uncovered source owner.");
            const auto& indices = found->second;
            std::vector<Meshes::BoundedConvexGeometryPtr> cover;
            for (auto i : indices)
                cover.push_back(columns[i].overlap);
            source.finite_geometry->certify_partition(cover, options.maximum_relative_target_defect);
            const bool whole = indices.size() == 1 &&
                (columns[indices[0]].kind == AMR::AdaptiveAncestryRelation::Kind::Unchanged ||
                 columns[indices[0]].kind == AMR::AdaptiveAncestryRelation::Kind::Coarsening);
            PolyhedralInterface fitted;
            FLabel labels;
            if (!whole)
            {
                fitted = reconstruct_finite_cell(data, key);
                labels = finite_label(source, fitted, data.options.reconstruction.volume_tolerance);
            }
            const auto old = finite_inventory(source);
            // Carrier existence is a source contract even on ownership-only
            // copies; the whole-cell fast path cannot legitimize an unsupported
            // nonzero channel merely because it needs no geometric weighting.
            for (std::size_t c = 0; c < options.channels.size(); ++c)
            {
                const auto carrier = finite_remap_weighted(old, options.channels[c].carrier, options.phase_density);
                const auto amount = source_channels.at(key)[c];
                if (carrier.value == 0 && carrier.absolute_error == 0)
                {
                    if (amount.value != 0 || amount.absolute_error != 0)
                        throw std::invalid_argument("Finite remap nonzero channel has no physical carrier.");
                }
                else if (!(carrier.lower() > 0))
                    throw std::invalid_argument("Finite remap source channel carrier is unresolved.");
            }
            FPhaseSum total;
            std::vector<fp::Sum> channel_totals(options.channels.size());
            for (auto i : indices)
            {
                const auto& column = columns[i];
                Output output;
                output.relation = column;
                output.phases = whole ? old : labels.apply(*column.overlap, fitted);
                output.history = whole ? FPhases{{}, {source.finite_history.gas, 0},
                    {source.finite_history.carrier, 0}} : labels.history(*column.overlap, fitted);
                output.declaration = source.declaration;
                output.original_error = source.occupancy;
                output.original_roundoff = source.roundoff;
                if (ownership)
                    output.retained_interface = source.retained_interface;
                const auto bulk_share = whole ? FValue{1, 0} : fp::divide(output.phases.bulk, old.bulk);
                output.inherited_occupancy = fp::multiply({source.occupancy.admitted_bound, 0}, bulk_share);
                for (std::size_t c = 0; c < options.channels.size(); ++c)
                {
                    const auto& input = source_channels.at(key)[c];
                    FValue amount;
                    if (whole)
                        amount = input;
                    else
                    {
                        const auto denominator = finite_remap_weighted(old, options.channels[c].carrier, options.phase_density);
                        const auto numerator = finite_remap_weighted(output.phases, options.channels[c].carrier, options.phase_density);
                        if (denominator.value == 0 && denominator.absolute_error == 0)
                        {
                            if (input.value != 0 || input.absolute_error != 0 ||
                                numerator.value != 0 || numerator.absolute_error != 0)
                                throw std::invalid_argument("Finite remap nonzero channel has no physical carrier.");
                        }
                        else
                            amount = fp::multiply(input, fp::divide(numerator, denominator));
                    }
                    output.amounts.push_back(amount);
                    channel_totals[c].add(amount);
                }
                total.add(output.phases);
                budget.add(sizeof(Output) + output.amounts.size() * sizeof(FValue));
                outgoing.push_back(std::move(output));
            }
            const auto delivered = total.value();
            for (auto pair : {std::pair{old.gas, delivered.gas}, std::pair{old.carrier, delivered.carrier}})
            {
                const auto residual = fp::subtract(pair.second, pair.first);
                if (std::abs(residual.value) > residual.absolute_error ||
                    residual.absolute_error > std::abs(pair.first.value) * options.maximum_relative_target_fit)
                    throw std::invalid_argument("Finite remap source phase closure exceeds its unchanged target-fit bound.");
            }
            for (std::size_t c = 0; c < options.channels.size(); ++c)
            {
                const auto residual = fp::subtract(channel_totals[c].value(), source_channels.at(key)[c]);
                if (std::abs(residual.value) > residual.absolute_error)
                    throw std::invalid_argument("Finite remap weighted channel column is not conservative.");
            }
        }
    });
    packets.clear();
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& output : outgoing)
        {
            const auto& column = output.relation;
            wire::Writer w;
            w.bytes(AMR::encode_adaptive_key(column.source));
            w.word(column.source_gid); w.word(column.source_revision);
            w.bytes(AMR::encode_adaptive_key(column.target));
            w.word(column.target_gid); w.word(column.target_revision);
            finite_phases(w, output.phases);
            finite_phases(w, output.history);
            finite_value(w, output.inherited_occupancy);
            w.word(static_cast<unsigned>(output.declaration));
            w.word(output.amounts.size());
            for (auto amount : output.amounts)
                finite_value(w, amount);
            if (ownership)
            {
                finite_remap_error(w, output.original_error);
                real(w, output.original_roundoff.gas); real(w, output.original_roundoff.carrier);
                w.word(bool(output.retained_interface));
                if (output.retained_interface)
                    plane(w, *output.retained_interface);
            }
            budget.packet(w.data.size());
            packets.push_back({column.target_owner, std::move(w.data)});
        }
    });
    const auto responses = exchange_owner_packets(packets, comm, limit);
    struct Row
    {
        FPhaseSum phases, history;
        fp::Sum occupancy;
        std::vector<fp::Sum> channels;
        std::optional<GeneralVoFPhaseGeometry> declaration;
        PhaseError original_error;
        PhaseRoundoff original_roundoff;
        std::optional<PolyhedralInterface> retained_interface;
        std::size_t pieces = 0;
    };
    std::map<Key, Row> rows;
    FiniteRemapTrial result;
    result.source_ = data_;
    result.ownership_only_ = ownership;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& column : columns)
        {
            budget.add(sizeof(GeneralVoFFiniteCommonCell) + column.overlap->storage_bytes());
            result.common_cells_.push_back({column.source, column.target, column.source_gid,
                column.target_gid, column.target_owner, column.overlap});
        }
    });
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        std::map<std::pair<Key, Key>, std::size_t> expected;
        for (std::size_t i = 0; i < requests.size(); ++i)
            if (!expected.emplace(std::pair{requests[i].source, requests[i].target}, i).second)
                throw std::invalid_argument("Finite remap ancestry repeats a source/target pair.");
        std::set<std::pair<Key, Key>> seen;
        for (const auto& packet : responses.packets)
        {
            wire::Reader r{packet.bytes};
            const auto sk = AMR::decode_adaptive_key(r.bytes());
            const auto sgid = r.word(), revision = r.word();
            const auto tk = AMR::decode_adaptive_key(r.bytes());
            const auto tgid = r.word(), target_revision = r.word();
            const auto item = expected.find({sk, tk});
            if (item == expected.end() || !seen.emplace(sk, tk).second)
                throw std::invalid_argument("Finite remap output is duplicate or not an authenticated overlap.");
            const auto& request = requests[item->second];
            if (packet.peer != locations[item->second].owner || sgid != request.source_gid ||
                revision != request.source_revision || tgid != request.target_gid || target_revision != request.target_revision)
                throw std::invalid_argument("Finite remap result source/target ownership or revision differs.");
            const auto phases = finite_phases(r), history = finite_phases(r);
            const auto inherited_occupancy = finite_value(r);
            const auto declaration = r.word(), count = r.word();
            if (declaration > static_cast<unsigned>(GeneralVoFPhaseGeometry::PureCarrier) || count != options.channels.size())
                throw std::invalid_argument("Finite remap result schema is malformed.");
            for (auto value : {phases.bulk, phases.gas, phases.carrier, history.gas, history.carrier, inherited_occupancy})
                if (value.value < 0 || value.lower() < 0)
                    throw std::invalid_argument("Finite remap output has unresolved nonnegative amounts.");
            auto& row = rows[tk];
            if (row.pieces++ == 0)
            {
                row.channels.resize(options.channels.size());
                row.declaration = static_cast<GeneralVoFPhaseGeometry>(declaration);
            }
            else if (*row.declaration != static_cast<GeneralVoFPhaseGeometry>(declaration))
                row.declaration = GeneralVoFPhaseGeometry::Infer;
            row.phases.add(phases);
            row.history.add(history);
            row.occupancy.add(inherited_occupancy);
            for (auto& channel : row.channels)
                channel.add(finite_value(r));
            if (ownership)
            {
                if (row.pieces != 1 || sk != tk)
                    throw std::invalid_argument("Finite ownership migration changes cell ancestry.");
                row.original_error = finite_remap_error(r);
                row.original_roundoff = {real(r), real(r)};
                const auto retained = r.word();
                if (retained > 1)
                    throw std::invalid_argument("Finite migration interface tag is malformed.");
                if (retained)
                {
                    row.retained_interface = plane(r);
                    if (!row.retained_interface->finite_geometry)
                        throw std::invalid_argument("Finite migration cannot acquire an exact interface.");
                }
            }
            r.done();
            budget.add(sizeof(Row) + sizeof(GeneralVoFFiniteRemapRoute) + row.channels.size() * sizeof(fp::Sum));
            result.routes_.push_back({sk, tk, sgid, tgid, packet.peer, comm->getRank(), phases});
        }
        if (seen.size() != expected.size() || rows.size() != target->owned().size())
            throw std::invalid_argument("Finite remap outputs do not cover every target owner.");
    });
    // Independently validate recipient geometry before physical publication.
    // Sparse ancestry selects candidates; a net volume sum cannot certify cover.
    std::map<Key, Meshes::BoundedConvexGeometryPtr> local_source_geometry;
    std::map<Key, std::vector<Meshes::BoundedConvexGeometryPtr>> target_covers;
    wire::collective(*comm, [&]
    {
        wire::Budget budget{limit};
        for (const auto& request : requests)
        {
            if (!local_source_geometry.contains(request.source))
            {
                const auto existing = data.cells.find(request.source);
                auto geometry = existing != data.cells.end() ? existing->second.finite_geometry :
                    Meshes::BoundedConvexGeometry::from_geometry(*from.at(request.source_gid)->geometry);
                budget.add(geometry->storage_bytes());
                local_source_geometry.emplace(request.source, std::move(geometry));
            }
            if (!imported_targets.contains(request.target))
            {
                auto geometry = Meshes::BoundedConvexGeometry::from_geometry(*to.at(request.target_gid)->geometry);
                budget.add(geometry->storage_bytes());
                imported_targets.emplace(request.target, std::move(geometry));
            }
            const auto& source_geometry = local_source_geometry.at(request.source);
            const auto& target_geometry = imported_targets.at(request.target);
            auto overlap = source_geometry;
            if (request.kind == AMR::AdaptiveAncestryRelation::Kind::Refinement)
                overlap = target_geometry;
            else if (request.kind == AMR::AdaptiveAncestryRelation::Kind::ProductIntersection)
                overlap = source_geometry->intersect(*target_geometry);
            if (!overlap)
                throw std::invalid_argument("Finite remap recipient overlap is empty.");
            target_covers[request.target].push_back(std::move(overlap));
        }
        for (const auto& [key, leaf] : target->owned())
            imported_targets.at(key)->certify_partition(target_covers.at(key), options.maximum_relative_target_defect);
    });
    // Build only geometry/maps in a private pure placeholder. Actual remapped
    // inventories and their inherited error evidence are installed together.
    std::vector<Key> order;
    std::vector<double> zero, native_volume;
    wire::collective(*comm, [&]
    {
        for (const auto& [key, leaf] : target->owned())
        {
            order.push_back(key);
            zero.push_back(0);
            native_volume.push_back(target->metrics(leaf).volume);
        }
    });
    const auto placeholder = std::make_shared<const GeneralVoF>(target, order, zero, native_volume, data.options);
    auto candidate = std::make_shared<Data>(*placeholder->data_);
    wire::collective(*comm, [&]
    {
        candidate->gas.clear(); candidate->carrier.clear(); candidate->phases.clear();
        candidate->roundoff.clear(); candidate->phase_history.clear(); candidate->occupancy.clear();
        wire::Budget budget{limit};
        for (const auto& key : candidate->topology->order)
        {
            auto& cell = candidate->cells.at(key);
            auto& row = rows.at(key);
            const auto totals = row.phases.value(), history = row.history.value();
            const auto actual = cell.finite_geometry->volume();
            const auto closure = fp::subtract(totals.bulk, actual);
            if (std::abs(closure.value) > closure.absolute_error ||
                closure.absolute_error > actual.value * options.maximum_relative_target_defect)
                throw std::invalid_argument("Finite remap recipient coverage exceeds its geometric error budget.");
            const auto gas = fp::publish(totals.gas), carrier = fp::publish(totals.carrier);
            cell.phase.gas_volume = gas.first;
            cell.phase.carrier_volume = carrier.first;
            cell.roundoff = ownership ? row.original_roundoff : PhaseRoundoff{gas.second, carrier.second};
            cell.finite_history = {fp::upper_double(history.gas.upper()), fp::upper_double(history.carrier.upper())};
            if (!ownership)
            {
                cell.finite_history.gas = fp::upper_double(fp::upward(cell.finite_history.gas + gas.second));
                cell.finite_history.carrier = fp::upper_double(fp::upward(cell.finite_history.carrier + carrier.second));
            }
            cell.declaration = row.declaration.value_or(GeneralVoFPhaseGeometry::Infer);
            cell.retained_interface = ownership ? row.retained_interface : std::optional<PolyhedralInterface>{};
            cell.inherited = true;
            if (ownership)
                cell.occupancy = row.original_error;
            else
            {
                const auto discrepancy = fp::subtract(fp::add({gas.first, gas.second}, {carrier.first, carrier.second}), actual);
                const auto inherited = row.occupancy.value();
                const auto admitted = fp::upward(inherited.upper() + discrepancy.absolute_error + std::abs(closure.value));
                if (std::abs(discrepancy.value) > admitted ||
                    admitted > actual.value * options.maximum_relative_target_defect)
                    throw std::invalid_argument("Finite remap physical occupancy exceeds its original target-defect bound.");
                cell.occupancy.difference = static_cast<double>(discrepancy.value);
                cell.occupancy.admitted_bound = fp::upper_double(admitted);
                cell.occupancy.phase_input_bound = fp::upper_double(fp::upward(gas.second + carrier.second));
                cell.occupancy.transferred_bound = fp::upper_double(fp::upward(totals.gas.absolute_error + totals.carrier.absolute_error));
            }
            validate_cell(cell, data.options);
            candidate->gas.push_back(gas.first); candidate->carrier.push_back(carrier.first);
            candidate->phases.push_back({cell.native_volume, gas.first, carrier.first});
            candidate->roundoff.push_back(cell.roundoff); candidate->phase_history.push_back(cell.finite_history);
            candidate->occupancy.push_back(cell.occupancy);
            GeneralVoFFiniteOwnedChannels output;
            output.key = key;
            for (std::size_t i = 0; i < row.channels.size(); ++i)
            {
                const auto value = row.channels[i].value();
                if (options.channels[i].nonnegative && value.lower() < 0)
                    throw std::invalid_argument("Finite remap nonnegative channel output is unresolved.");
                output.amounts.push_back(value);
            }
            budget.add(sizeof(output) + output.amounts.size() * sizeof(FValue));
            result.channels_.push_back(std::move(output));
        }
        current(data);
        if (!target->geometry_unchanged() || target->observation() != target_observation)
            throw std::invalid_argument("Finite remap target observation changed during preparation.");
    });
    import_phases(*candidate);
    wire::collective(*comm, [&] { retained_budget(*candidate); });
    result.candidate_ = std::move(candidate);
    result.validate_observations();
    return result;
}

}
