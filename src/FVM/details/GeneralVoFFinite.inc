/** @file GeneralVoFFinite.inc
 * @brief Separate bounded numerical transport and independent finite phase publication.
 * Included only by GeneralVoF.cc inside SimpleFluid::VoF.
 */
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
GeneralVoF::Trial GeneralVoF::plan_transport_finite(std::span<const GeneralVoFNodeVelocity> node_velocity,
    std::span<const GeneralVoFFaceVolume> face_volume, double timestep) const
{
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
