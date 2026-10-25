/** @file GeneralVoFExactTransport.cc
 * @brief Optional exact phase transport planning.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF
{
namespace Exact
{
namespace
{
// Sharded equality authorities compare full certificates/represented controls,
// never a geometry hash alone. All ranks, including empty owners, participate.
void controls(const GeneralVoF::Data& data, std::span<const GeneralVoFNodeVelocity> nodes,
    std::span<const GeneralVoFFaceVolume> faces, double dt,
    std::map<std::uint64_t, SimpleFluid::PLIC::GeometryPoint>& velocity, std::map<std::uint64_t, Measure>& flux,
    std::map<std::uint64_t, SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>>& trajectories)
{
    auto comm = data.hierarchy->communicator();
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (!std::isfinite(dt) || dt < 0 || nodes.size() != data.topology->nodes.size() ||
                faces.size() != data.topology->tiles.size())
                throw std::invalid_argument("General VoF controls do not cover retained nodes/faces.");
            wire::Budget budget{data.options.byte_budget};
            for (const auto& node : nodes)
            {
                for (auto x : node.velocity)
                    if (!std::isfinite(x))
                        throw std::invalid_argument("General VoF node velocity is nonfinite.");
                if (!velocity.emplace(node.node, node.velocity).second)
                    throw std::invalid_argument("General VoF repeats a node control.");
            }
            for (const auto& face : faces)
                if (!std::isfinite(face.integrated_volume) ||
                    !flux
                        .emplace(face.face, face.exact_integrated_volume ? *face.exact_integrated_volume
                                                                         : Measure::from_value(face.integrated_volume))
                        .second)
                    throw std::invalid_argument("General VoF invalid/repeated face control.");
            for (const auto& node : data.topology->nodes)
            {
                auto v = velocity.at(node.gid);
                std::vector<SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Exact>> walls;
                const auto& catalog = data.topology->node_walls.at(node.gid);
                budget.add(
                    catalog.size() * sizeof(SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Exact>));
                for (const auto& wall : catalog)
                    walls.push_back(PLIC::PlanarFace<ArithmeticMode::Exact>::from_geometry(*wall.geometry));
                SimpleFluid::PLIC::TrajectoryLimits limits;
                limits.byte_budget = data.options.byte_budget;
                auto trajectory =
                    SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>::backtrace_constrained(
                        node.coordinate_identity, v, dt, walls, limits);
                budget.add(sizeof(trajectory) + trajectory.point.size() + trajectory.departure.size());
                budget.packet(56 + node.coordinate_identity.size());
                wire::Writer w;
                w.word(0);
                w.word(node.gid);
                w.bytes(node.coordinate_identity);
                for (auto x : v)
                    real(w, x);
                real(w, dt);
                budget.add(16 + trajectory.departure.size());
                w.bytes(trajectory.departure);
                w.word(catalog.size());
                for (const auto& wall : catalog)
                {
                    const auto certificate = wall.geometry->encode_incidence();
                    budget.add(16 + wall.identity.size() + certificate.size());
                    w.bytes(wall.identity);
                    w.bytes(certificate);
                }
                trajectories.emplace(node.gid, std::move(trajectory));
                packets.push_back({int(node.gid % comm->getSize()), std::move(w.data)});
            }
            for (const auto& tile : data.topology->tiles)
            {
                const auto phi = flux.at(tile.face_gid);
                if ((tile.exterior || dt == 0) && phi.sign() != 0)
                    throw std::invalid_argument("General VoF physical boundary has nonzero flux.");
                auto face = tile.geometry->encode_incidence();
                auto exact_phi = phi.encode(data.options.byte_budget);
                budget.packet(face.size() + exact_phi.size() + 32);
                wire::Writer w;
                w.word(1);
                w.word(tile.face_gid);
                w.bytes(face);
                w.bytes(exact_phi);
                packets.push_back({int(tile.face_gid % comm->getSize()), std::move(w.data)});
            }
            budget.packet(32);
            wire::Writer w;
            w.word(2);
            w.word(0);
            real(w, dt);
            w.word(data.control);
            packets.push_back({0, std::move(w.data)});
        });
    auto received = exchange_owner_packets(packets, comm, data.options.byte_budget);
    wire::collective(*comm,
        [&]
        {
            std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<char>> canonical;
            wire::Budget budget{data.options.byte_budget};
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto type = r.word(), id = r.word();
                if (type > 2 || int(id % comm->getSize()) != comm->getRank())
                    throw std::invalid_argument("General VoF control reached wrong authority.");
                budget.add(packet.bytes.size() +
                           sizeof(std::pair<const std::pair<std::uint64_t, std::uint64_t>, std::vector<char>>));
                auto [found, inserted] = canonical.emplace(std::pair{type, id}, packet.bytes);
                if (!inserted && found->second != packet.bytes)
                    throw std::invalid_argument("General VoF shared node/face controls disagree.");
            }
        });
}
std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> donor_support(
    const GeneralVoF::Data& data, const Key& first, const std::optional<Key>& second = {})
{
    std::set<Key> seen{first}, front{first};
    if (second)
    {
        seen.insert(*second);
        front.insert(*second);
    }
    for (unsigned ring = 0; ring < data.options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& key : front)
        {
            auto found = data.topology->adjacency.find(key);
            if (found == data.topology->adjacency.end())
                continue;
            for (const auto& neighbor : found->second)
                if (data.cells.contains(neighbor) && seen.insert(neighbor).second)
                {
                    if (seen.size() > data.options.maximum_stencil_cells)
                        throw std::invalid_argument("General VoF donor support budget exhausted.");
                    next.insert(neighbor);
                }
        }
        front = std::move(next);
    }
    std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> result;
    for (const auto& key : seen)
    {
        const auto& cell = data.cells.at(key);
        result.push_back({cell.gid, cell.phase.geometry});
    }
    return result;
}
std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> face_donor_support(
    const GeneralVoF::Data& data, const AMR::AdaptiveTile& face, wire::Budget& budget)
{
    std::map<Key, SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> selected;
    auto insert = [&](const Key& key)
    {
        const auto cell = data.cells.find(key);
        if (cell == data.cells.end() || !cell->second.phase.geometry)
            throw std::invalid_argument("General VoF face donor fan requires an authenticated full native shell.");
        if (!selected.contains(key))
        {
            if (selected.size() >= data.options.maximum_stencil_cells)
                throw std::invalid_argument("General VoF complete face donor fan budget exhausted.");
            budget.add(sizeof(Key) + sizeof(SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>) +
                       AMR::encode_adaptive_key(key).size() + 8 * sizeof(void*));
            selected.emplace(key, SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>{
                                      cell->second.gid, cell->second.phase.geometry});
        }
    };
    for (const auto& donor : donor_support(data, face.first, face.second))
        insert(data.topology->by_gid.at(donor.id));
    // A high-valence native corner is not bounded by two face hops. These
    // complete fans were registered by ALL incident owners and their full
    // shells fetched before construction; geometric coincidence is irrelevant.
    for (auto node : face.node_gids)
        for (const auto& key : data.topology->node_fans.at(node))
            insert(key);
    std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> result;
    budget.add(selected.size() * sizeof(SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>));
    result.reserve(selected.size());
    for (const auto& [key, donor] : selected)
        result.push_back(donor);
    return result;
}
struct ExactAmounts
{
    Measure bulk, gas, carrier;
    void add(const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>& p, int sign = 1)
    {
        bulk = bulk + p.bulk.scaled(sign);
        gas = gas + p.gas.scaled(sign);
        carrier = carrier + p.carrier.scaled(sign);
    }
};
void transport_charge(wire::Budget& budget, std::size_t bytes, const char* phase, const char* kind = "")
{
    if (bytes > budget.limit - budget.used)
        throw std::invalid_argument(std::string("General VoF ") + phase +
                                    " budget exhausted: used=" + std::to_string(budget.used) +
                                    ", requested=" + std::to_string(bytes) + ", limit=" + std::to_string(budget.limit) +
                                    ", remaining=" + std::to_string(budget.limit - budget.used));
    budget.add(bytes);
}
struct ExactStorageBudget
{
    wire::Budget& bytes;
    std::set<const void*> fragments;
    const char* phase = "transport retained exact";
    void fragment(const SimpleFluid::PLIC::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>& value)
    {
        if (fragments.insert(value.storage_identity()).second)
            transport_charge(bytes, value.storage_bytes() + 4 * sizeof(void*), phase, " fragment");
    }
    void measure(const Measure& value)
    {
        // Accumulator intermediates may already have died and their allocator
        // address may be reused. Charge each produced value conservatively;
        // pointer deduplication is valid only for the retained live catalog.
        transport_charge(bytes, value.storage_bytes(), phase, " measure");
    }
    void phases(const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>& value)
    {
        measure(value.bulk);
        measure(value.gas);
        measure(value.carrier);
    }
};
} // namespace
GeneralVoF::Trial GeneralVoF::plan_transport(const State& self, std::span<const GeneralVoFNodeVelocity> node_velocity,
    std::span<const GeneralVoFFaceVolume> face_volume, double timestep)
{
    const auto& data = *self.data_;
    auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::map<std::uint64_t, SimpleFluid::PLIC::GeometryPoint> velocity;
    std::map<std::uint64_t, Measure> flux;
    std::map<std::uint64_t, SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>>
        canonical_trajectories;
    controls(data, node_velocity, face_volume, timestep, velocity, flux, canonical_trajectories);
    auto retain_trajectories = [&](Trial& trial)
    {
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{limit};
                for (const auto& [node, trajectory] : canonical_trajectories)
                {
                    budget.add(
                        sizeof(GeneralVoFNodeTrajectory) + trajectory.point.size() + trajectory.departure.size());
                    trial.trajectories_.push_back({node, trajectory});
                }
            });
    };
    wire::collective(*comm,
        [&]
        {
            for (const auto& key : data.topology->order)
            {
                Measure divergence;
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    const int sign = tile.first == key ? 1 : (tile.second && *tile.second == key ? -1 : 0);
                    if (sign)
                        divergence = divergence + flux.at(tile.face_gid).scaled(sign);
                }
                if (divergence.sign())
                    throw std::invalid_argument("General VoF projected flux violates source-free occupancy.");
            }
        });
    int local_motion = 0, motion = 0;
    for (const auto& [node, trajectory] : canonical_trajectories)
        local_motion |= trajectory.point != trajectory.departure;
    for (const auto& [face, value] : flux)
        local_motion |= value.sign() != 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_motion, &motion);
    if (!motion)
    {
        // A proven identity trajectory and zero canonical Phi need no inferred
        // interface normal. Net-zero Phi alone is NOT an identity test: crossed
        // nonzero trajectories continue through the full exact sweep proof.
        Trial trial;
        trial.source_ = self.data_;
        trial.candidate_ = self.data_;
        trial.transaction_ = transaction(comm);
        retain_trajectories(trial);
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{limit};
                budget.add(data.topology->order.size() * sizeof(PhaseRoundoff));
                trial.roundoff_.resize(data.topology->order.size());
                for (const auto& key : data.topology->order)
                {
                    budget.add(sizeof(GeneralVoFTransfer) + sizeof(GeneralVoFDonorBudget));
                    const auto& cell = data.cells.at(key);
                    trial.transfers_.push_back({cell.gid, cell.gid, cell.owner, cell.owner, phase_measures(cell),
                        diagnostic(phase_measures(cell))});
                    SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> zero;
                    trial.budgets_.push_back({cell.gid, phase_measures(cell), zero, phase_measures(cell),
                        diagnostic(phase_measures(cell)), diagnostic(zero), diagnostic(phase_measures(cell))});
                }
            });
        return trial;
    }
    auto planes = import_planes(data);
    std::map<Key, detail::ExactPhaseLabelMap> labels;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
            {
                const auto raw =
                    phases(*SimpleFluid::PLIC::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>::from_geometry(
                               *cell.phase.geometry),
                        planes.at(key));
                const bool gas = phase_measures(cell).gas.compare(phase_measures(cell).carrier) <= 0;
                const auto target = gas ? phase_measures(cell).gas : phase_measures(cell).carrier;
                const auto view = Measure::from_value(gas ? cell.phase.gas_volume : cell.phase.carrier_volume);
                const auto conversion = target - view;
                const auto absolute_conversion = conversion.sign() < 0 ? conversion.scaled(-1) : conversion;
                auto bound = view.scaled(data.options.reconstruction.volume_tolerance) + absolute_conversion;
                if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
                {
                    if (cell.experimental_exact)
                    {
                        const auto& e = cell.experimental_exact->error;
                        bound = e.admitted_bound.compare(e.fit_bound) >= 0 ? e.admitted_bound : e.fit_bound;
                    }
                    else
                        bound = finite_bound(std::max(cell.occupancy.admitted_bound, cell.occupancy.fit_bound));
                }
                auto label = detail::ExactPhaseLabelMap::create(raw, gas, target, bound, limit);
                budget.add(label.storage_bytes());
                labels.emplace(key, std::move(label));
            }
        });
    std::map<std::uint64_t, std::shared_ptr<const SimpleFluid::PLIC::SweptFace<SimpleFluid::ArithmeticMode::Exact>>>
        sweeps;
    std::map<std::uint64_t, std::vector<SimpleFluid::PLIC::SweepIntersection<SimpleFluid::ArithmeticMode::Exact>>>
        face_parts;
    std::map<std::uint64_t, std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>>> face_donors;
    std::vector<SimpleFluid::PLIC::TransportAtom<SimpleFluid::ArithmeticMode::Exact>> local_atoms;
    std::map<Key, std::pair<Measure, Measure>> signed_repair;
    Trial trial;
    trial.source_ = self.data_;
    trial.transaction_ = transaction(comm);
    retain_trajectories(trial);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            ExactStorageBudget exact_storage{budget};
            detail::ExactLiveStorage raw_live(budget, limit / 4);
            budget.add(data.topology->order.size() * sizeof(PhaseRoundoff));
            trial.roundoff_.resize(data.topology->order.size());
            for (const auto& tile : data.topology->tiles)
            {
                std::vector<SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>> trajectories;
                budget.add(tile.node_gids.size() *
                           sizeof(SimpleFluid::PLIC::NodeTrajectory<SimpleFluid::ArithmeticMode::Exact>));
                for (auto gid : tile.node_gids)
                    trajectories.push_back(canonical_trajectories.at(gid));
                auto face = SimpleFluid::PLIC::SweptFace<SimpleFluid::ArithmeticMode::Exact>::make(
                    *PLIC::PlanarFace<ArithmeticMode::Exact>::from_geometry(*tile.geometry), trajectories,
                    flux.at(tile.face_gid), data.options.sweep);
                auto candidates = face_donor_support(data, tile, budget);
                auto pieces = [&]
                {
                    try
                    {
                        return face->partition(candidates, data.options.sweep);
                    }
                    catch (const std::invalid_argument& error)
                    {
                        throw std::invalid_argument(
                            "General VoF raw face=" + std::to_string(tile.face_gid) +
                            " first=" + std::to_string(data.cells.at(tile.first).gid) +
                            " second=" + (tile.second ? std::to_string(data.cells.at(*tile.second).gid) : "wall") +
                            " source_control=" + std::to_string(data.control) + ": " + error.what());
                    }
                }();
                for (const auto& simplex : face->simplices())
                    exact_storage.fragment(*simplex.geometry);
                for (const auto& piece : pieces)
                    exact_storage.fragment(*piece.geometry);
                sweeps.emplace(tile.face_gid, std::move(face));
                face_parts.emplace(tile.face_gid, std::move(pieces));
                face_donors.emplace(tile.face_gid, std::move(candidates));
            }
            for (const auto& key : data.topology->order)
            {
                const auto& cell = data.cells.at(key);
                SimpleFluid::PLIC::SweepRecipient<SimpleFluid::ArithmeticMode::Exact> recipient{
                    cell.gid, cell.phase.geometry, {}};
                Measure divergence;
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    int sign = tile.first == key ? 1 : (tile.second && *tile.second == key ? -1 : 0);
                    if (!sign)
                        continue;
                    recipient.outward_faces.push_back({sweeps.at(tile.face_gid), sign});
                    divergence = divergence + flux.at(tile.face_gid).scaled(sign);
                }
                if (divergence.sign())
                    throw std::invalid_argument("General VoF projected flux violates source-free occupancy.");
                std::map<Key, SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> selected;
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    for (const auto& donor : face_donors.at(tile.face_gid))
                    {
                        const auto& donor_key = data.topology->by_gid.at(donor.id);
                        if (!selected.contains(donor_key))
                        {
                            if (selected.size() >= data.options.maximum_stencil_cells)
                                throw std::invalid_argument("General VoF recipient face-donor union budget exhausted.");
                            budget.add(sizeof(Key) +
                                       sizeof(SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>) +
                                       AMR::encode_adaptive_key(donor_key).size() + 8 * sizeof(void*));
                            selected.emplace(donor_key, donor);
                        }
                    }
                }
                std::vector<SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>> donors;
                budget.add(selected.size() * sizeof(SimpleFluid::PLIC::SweepDonor<SimpleFluid::ArithmeticMode::Exact>));
                for (const auto& [donor_key, donor] : selected)
                    donors.push_back(donor);
                auto partition = [&]
                {
                    try
                    {
                        return SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::local(
                            donors, std::span(&recipient, 1), data.options.sweep);
                    }
                    catch (const std::invalid_argument& error)
                    {
                        std::ostringstream context;
                        context << "General VoF local recipient=" << cell.gid << " source_control=" << data.control
                                << " donors=[";
                        for (const auto& donor : donors)
                            context << donor.id << ',';
                        context << "] incident_faces=[";
                        for (const auto index : data.topology->cell_tiles.at(key))
                        {
                            const auto& tile = data.topology->tiles[index];
                            std::uint64_t hash = 1469598103934665603ULL;
                            for (unsigned char byte : tile.geometry->encode())
                                hash = (hash ^ byte) * 1099511628211ULL;
                            context << tile.face_gid << ':' << hash << ',';
                        }
                        context << "]: " << error.what();
                        throw std::invalid_argument(context.str());
                    }
                }();
                budget.add(partition.atoms().size() *
                           sizeof(SimpleFluid::PLIC::TransportAtom<SimpleFluid::ArithmeticMode::Exact>));
                for (const auto& atom : partition.atoms())
                    exact_storage.fragment(*atom.geometry);
                local_atoms.insert(local_atoms.end(), partition.atoms().begin(), partition.atoms().end());
                // Independently verify phase row identities against the canonical
                // signed face ledger before any native-range conversion.
                std::map<std::uint64_t, detail::ExactPhaseAccumulator> row;
                const auto row_for = [&](std::uint64_t donor) -> detail::ExactPhaseAccumulator&
                {
                    if (!row.contains(donor))
                        budget.add(sizeof(decltype(row)::value_type) + 4 * sizeof(void*));
                    return row.try_emplace(donor, raw_live).first->second;
                };
                for (const auto& atom : partition.atoms())
                    row_for(atom.donor).add(phases(*atom.geometry, planes.at(data.topology->by_gid.at(atom.donor))));
                for (const auto index : data.topology->cell_tiles.at(key))
                {
                    const auto& tile = data.topology->tiles[index];
                    const int sign = tile.first == key ? 1 : (tile.second && *tile.second == key ? -1 : 0);
                    if (!sign)
                        continue;
                    for (const auto& piece : face_parts.at(tile.face_gid))
                        row_for(piece.donor)
                            .add(phases(*piece.geometry, planes.at(data.topology->by_gid.at(piece.donor))),
                                sign * piece.sign);
                }
                const auto original =
                    phases(*SimpleFluid::PLIC::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>::from_geometry(
                               *cell.phase.geometry),
                        planes.at(key));
                row_for(cell.gid).add(original, -1);
                for (const auto& [donor, amount] : row)
                {
                    // Keep the original MAPPED-zero gate, including singular
                    // donor maps; raw-zero alone is not the same acceptance.
                    const auto paired = amount.apply(labels.at(data.topology->by_gid.at(donor)));
                    if (paired.bulk.sign() || paired.gas.sign() || paired.carrier.sign())
                        throw std::invalid_argument(
                            "General VoF signed donor face ledger differs from positive recipient atoms.");
                }
            }
            for (const auto& tile : data.topology->tiles)
                if (tile.first_owner == comm->getRank())
                {
                    using GroupKey = std::pair<std::uint64_t, int>;
                    std::map<GroupKey, detail::ExactPhaseAccumulator> grouped;
                    for (const auto& piece : face_parts.at(tile.face_gid))
                    {
                        const GroupKey key{piece.donor, piece.sign};
                        if (!grouped.contains(key))
                            budget.add(sizeof(decltype(grouped)::value_type) + 4 * sizeof(void*));
                        grouped.try_emplace(key, raw_live)
                            .first->second.add(
                                phases(*piece.geometry, planes.at(data.topology->by_gid.at(piece.donor))));
                    }
                    for (const auto& [group, amount] : grouped)
                    {
                        const auto& key = data.topology->by_gid.at(group.first);
                        const auto& donor = data.cells.at(key);
                        auto paired = amount.apply(labels.at(key));
                        detail::ExactLiveStorage::Scope paired_scope(
                            raw_live, {&paired.bulk, &paired.gas, &paired.carrier});
                        auto difference = raw_live.difference(paired.gas, amount.raw().gas);
                        detail::ExactLiveStorage::Scope difference_scope(raw_live, {&difference});
                        auto absolute = difference.sign() < 0 ? raw_live.scaled(difference, -1) : difference;
                        detail::ExactLiveStorage::Scope absolute_scope(raw_live, {&absolute});
                        if (!signed_repair.contains(key))
                            budget.add(sizeof(decltype(signed_repair)::value_type) + 4 * sizeof(void*) +
                                       AMR::encode_adaptive_key(key).size());
                        auto& repair = signed_repair[key];
                        raw_live.track(repair.first);
                        raw_live.track(repair.second);
                        // create() changes only a or b: on positive pieces the
                        // gas correction has one fixed sign. Hence this grouped
                        // absolute value equals the original sum of absolutes.
                        raw_live.add(repair.first, absolute);
                        // Preserve the original per-PIECE allowance and keep
                        // opposite simplex signs in distinct groups.
                        if (amount.count() > 9007199254740992ULL)
                            throw std::invalid_argument(
                                "Exact phase group count exceeds exact binary64 integer range.");
                        auto bound = raw_live.scaled(labels.at(key).bound(), double(amount.count()));
                        detail::ExactLiveStorage::Scope bound_scope(raw_live, {&bound});
                        raw_live.add(repair.second, bound);
                        budget.add(3 * sizeof(GeneralVoFFaceContribution));
                        raw_live.retain(paired);
                        trial.faces_.push_back(
                            {tile.face_gid, donor.gid, donor.owner, group.second, paired, diagnostic(paired)});
                    }
                }
        });
    std::vector<OwnerPacket> repair_packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, repair] : signed_repair)
            {
                if (repair.first.compare(repair.second) > 0)
                    throw std::invalid_argument("General VoF cumulative signed phase correction exceeds bound.");
                const auto& donor = data.cells.at(key);
                const auto bytes = AMR::encode_adaptive_key(key);
                const auto absolute = repair.first.encode(limit), bound = repair.second.encode(limit);
                budget.packet(bytes.size() + absolute.size() + bound.size() + 40);
                wire::Writer w;
                w.bytes(bytes);
                w.word(donor.revision);
                w.bytes(absolute);
                w.bytes(bound);
                repair_packets.push_back({donor.owner, std::move(w.data)});
            }
        });
    auto received_repairs = exchange_owner_packets(repair_packets, comm, limit);
    std::map<Key, std::pair<Measure, Measure>> donor_signed_repair;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            ExactStorageBudget exact_storage{budget};
            std::set<std::pair<int, Key>> seen;
            for (const auto& packet : received_repairs.packets)
            {
                wire::Reader r{packet.bytes};
                const auto key = AMR::decode_adaptive_key(r.bytes());
                const auto revision = r.word();
                const auto& donor = data.cells.at(key);
                if (donor.owner != comm->getRank() || donor.revision != revision ||
                    !seen.emplace(packet.peer, key).second)
                    throw std::invalid_argument("General VoF unauthenticated signed correction record.");
                const auto absolute = Measure::decode(r.bytes(), limit), bound = Measure::decode(r.bytes(), limit);
                r.done();
                if (absolute.sign() < 0 || bound.sign() < 0 || absolute.compare(bound) > 0)
                    throw std::invalid_argument("General VoF invalid signed correction bound.");
                auto& total = donor_signed_repair[key];
                total.first = total.first + absolute;
                total.second = total.second + bound;
                exact_storage.measure(total.first);
                exact_storage.measure(total.second);
            }
        });
    std::vector<OwnerPacket>().swap(repair_packets);
    std::vector<OwnerPacket>().swap(received_repairs.packets);
    // Positive route atoms travel to the ORIGINAL actual donor owner. The
    // owner certifies a complete/disjoint source cover across all recipients.
    std::vector<OwnerPacket> atom_packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (std::size_t index = 0; index < local_atoms.size(); ++index)
            {
                const auto& atom = local_atoms[index];
                const auto& donor = data.cells.at(data.topology->by_gid.at(atom.donor));
                const auto& recipient = data.cells.at(data.topology->by_gid.at(atom.recipient));
                const bool local = donor.owner == comm->getRank();
                auto dk = AMR::encode_adaptive_key(donor.key), rk = AMR::encode_adaptive_key(recipient.key);
                std::vector<char> geometry;
                if (!local)
                    geometry = atom.geometry->encode(limit);
                budget.packet(dk.size() + rk.size() + geometry.size() + 64);
                wire::Writer w;
                w.bytes(dk);
                w.word(donor.revision);
                w.word(donor.gid);
                w.bytes(rk);
                w.word(recipient.gid);
                w.word(local ? 1 : 0);
                if (local)
                    w.word(index);
                else
                    w.bytes(geometry);
                atom_packets.push_back({donor.owner, std::move(w.data)});
            }
        });
    auto incoming = exchange_owner_packets(atom_packets, comm, limit);
    std::vector<OwnerPacket>().swap(atom_packets);
    struct Routed
    {
        Key donor, recipient;
        std::uint64_t recipient_gid = 0;
        int recipient_owner = -1;
        SimpleFluid::PLIC::TransportAtom<SimpleFluid::ArithmeticMode::Exact> atom;
    };
    std::vector<Routed> routed;
    std::vector<Key> recipient_keys;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto dk = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word(), gid = r.word();
                auto rk = AMR::decode_adaptive_key(r.bytes());
                auto recipient = r.word();
                const auto tag = r.word();
                const auto& donor = data.cells.at(dk);
                if (donor.owner != comm->getRank() || donor.revision != revision || donor.gid != gid)
                    throw std::invalid_argument("General VoF atom donor authentication failed.");
                SimpleFluid::PLIC::ConvexFragmentPtr<SimpleFluid::ArithmeticMode::Exact> fragment;
                if (tag == 1)
                {
                    // Only a self packet can name this invocation's immutable,
                    // already certified local atom. Remote data never acquires
                    // certified provenance through an integer reference.
                    const auto index = r.word();
                    r.done();
                    if (packet.peer != comm->getRank() || index >= local_atoms.size())
                        throw std::invalid_argument("General VoF foreign/out-of-range local atom reference.");
                    const auto& original = local_atoms[std::size_t(index)];
                    const auto& original_donor = data.cells.at(data.topology->by_gid.at(original.donor));
                    const auto& original_recipient = data.cells.at(data.topology->by_gid.at(original.recipient));
                    if (!original.geometry || original.donor != gid || original.recipient != recipient ||
                        original_donor.key != dk || original_recipient.key != rk ||
                        original_recipient.owner != comm->getRank() || original_recipient.gid != recipient)
                        throw std::invalid_argument("General VoF local atom metadata mismatch.");
                    fragment = original.geometry;
                    budget.add(sizeof(Routed));
                }
                else if (tag == 0)
                {
                    if (packet.peer == comm->getRank())
                        throw std::invalid_argument("General VoF self atom omitted its local reference.");
                    auto bytes = r.bytes();
                    r.done();
                    budget.add(sizeof(Routed) + bytes.size());
                    fragment =
                        SimpleFluid::PLIC::ConvexFragment<SimpleFluid::ArithmeticMode::Exact>::decode(bytes, limit);
                }
                else
                    throw std::invalid_argument("General VoF invalid atom route tag.");
                // Charge the referenced immutable payload in the receiver's
                // existing phase bound as well; aliasing does not grow its cap.
                budget.add(fragment->storage_bytes());
                routed.push_back({dk, rk, recipient, packet.peer, {gid, recipient, std::move(fragment)}});
                recipient_keys.push_back(rk);
            }
        });
    std::vector<OwnerPacket>().swap(incoming.packets);
    std::vector<SimpleFluid::PLIC::TransportAtom<SimpleFluid::ArithmeticMode::Exact>>().swap(local_atoms);
    auto recipient_owners = data.hierarchy->lookup_cells(recipient_keys);
    struct LocalAmount
    {
        Key donor, recipient;
        std::uint64_t donor_gid = 0, donor_revision = 0, recipient_gid = 0;
        ExactAmounts amount;
    };
    std::vector<LocalAmount> local_amounts;
    std::vector<OwnerPacket> transfer_packets;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            detail::ExactLiveStorage live(budget, limit / 4);
            std::map<Key, std::vector<SimpleFluid::PLIC::TransportAtom<SimpleFluid::ArithmeticMode::Exact>>>
                donor_atoms;
            std::map<Key, std::map<Key, ExactAmounts>> amounts;
            std::map<Key, Measure> route_repair;
            ExactStorageBudget exact_storage{budget};
            exact_storage.phase = "donor phase retained exact";
            std::map<Key, std::pair<int, std::uint64_t>> recipients;
            const auto metadata = [&](const Key& key, std::size_t record)
            {
                transport_charge(budget, record + 4 * sizeof(void*), "donor phase map metadata");
                std::visit(
                    [&](const auto& ancestry)
                    {
                        const auto path = [&](const auto& values)
                        {
                            if (values.capacity() > limit / sizeof(std::uint64_t))
                                throw std::invalid_argument("Donor phase key capacity exceeds budget.");
                            transport_charge(
                                budget, values.capacity() * sizeof(std::uint64_t), "donor phase key payload");
                        };
                        if constexpr (requires { ancestry.path; })
                            path(ancestry.path);
                        else if constexpr (requires { ancestry.base.path; })
                        {
                            path(ancestry.base.path);
                            path(ancestry.axial.path);
                        }
                    },
                    key.ancestry);
            };
            const auto amount_for = [&](const Key& donor, const Key& recipient) -> ExactAmounts&
            {
                if (!amounts.contains(donor))
                    metadata(donor, sizeof(decltype(amounts)::value_type));
                auto& row = amounts[donor];
                if (!row.contains(recipient))
                    metadata(recipient, sizeof(std::map<Key, ExactAmounts>::value_type));
                auto& result = row[recipient];
                live.track(result.bulk);
                live.track(result.gas);
                live.track(result.carrier);
                return result;
            };
            for (std::size_t i = 0; i < routed.size(); ++i)
            {
                const auto& route = routed[i];
                const auto& owner = recipient_owners[i];
                if (owner.owner != route.recipient_owner || owner.address != route.recipient_gid)
                    throw std::invalid_argument("General VoF atom recipient authentication failed.");
                if (!donor_atoms.contains(route.donor))
                    metadata(route.donor, sizeof(decltype(donor_atoms)::value_type));
                transport_charge(budget,
                    3 * sizeof(SimpleFluid::PLIC::TransportAtom<SimpleFluid::ArithmeticMode::Exact>),
                    "donor phase atom handles");
                donor_atoms[route.donor].push_back(route.atom);
                exact_storage.fragment(*route.atom.geometry);
                auto raw = phases(*route.atom.geometry, planes.at(route.donor));
                detail::ExactLiveStorage::Scope raw_scope(live, {&raw.bulk, &raw.gas, &raw.carrier});
                auto paired = labels.at(route.donor).apply(raw, live.operation_bytes());
                detail::ExactLiveStorage::Scope paired_scope(live, {&paired.bulk, &paired.gas, &paired.carrier});
                auto difference = live.difference(paired.gas, raw.gas);
                detail::ExactLiveStorage::Scope difference_scope(live, {&difference});
                if (!route_repair.contains(route.donor))
                    metadata(route.donor, sizeof(decltype(route_repair)::value_type));
                auto& correction = route_repair[route.donor];
                auto absolute = difference.sign() < 0 ? live.scaled(difference, -1) : difference;
                detail::ExactLiveStorage::Scope absolute_scope(live, {&absolute});
                live.add(correction, absolute);
                auto& amount = amount_for(route.donor, route.recipient);
                live.add(amount.bulk, paired.bulk);
                live.add(amount.gas, paired.gas);
                live.add(amount.carrier, paired.carrier);
                if (!recipients.contains(route.recipient))
                    metadata(route.recipient, sizeof(decltype(recipients)::value_type));
                recipients[route.recipient] = {route.recipient_owner, route.recipient_gid};
            }
            for (const auto& key : data.topology->order)
            {
                const auto& donor = data.cells.at(key);
                SimpleFluid::PLIC::TransportPartition<SimpleFluid::ArithmeticMode::Exact>::certify_donor_cover(
                    {donor.gid, donor.phase.geometry}, donor_atoms[key], data.options.sweep);
                const auto discrepancy = labels.at(key).discrepancy();
                auto absolute_discrepancy = discrepancy.sign() < 0 ? live.scaled(discrepancy, -1) : discrepancy;
                detail::ExactLiveStorage::Scope discrepancy_scope(live, {&absolute_discrepancy});
                if (route_repair[key].compare(absolute_discrepancy) ||
                    route_repair[key].compare(labels.at(key).bound()) > 0)
                    throw std::invalid_argument(
                        "General VoF cumulative positive phase correction exceeds donor proof.");
                ExactAmounts exported;
                detail::ExactLiveStorage::Scope exported_scope(
                    live, {&exported.bulk, &exported.gas, &exported.carrier});
                for (const auto& [recipient, amount] : amounts[key])
                    if (recipient != key)
                    {
                        live.add(exported.bulk, amount.bulk);
                        live.add(exported.gas, amount.gas);
                        live.add(exported.carrier, amount.carrier);
                    }
                auto retained_gas = live.difference(phase_measures(donor).gas, exported.gas);
                detail::ExactLiveStorage::Scope retained_gas_scope(live, {&retained_gas});
                auto retained_carrier = live.difference(phase_measures(donor).carrier, exported.carrier);
                detail::ExactLiveStorage::Scope retained_carrier_scope(live, {&retained_carrier});
                if (retained_gas.sign() < 0 || retained_carrier.sign() < 0)
                    throw std::invalid_argument(
                        "General VoF donor phase capacity exceeded after bounded canonical correction.");
                SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> exported_exact{
                    exported.bulk, exported.gas, exported.carrier};
                SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> retained_exact{
                    live.difference(phase_measures(donor).bulk, exported.bulk), retained_gas, retained_carrier};
                detail::ExactLiveStorage::Scope retained_exact_scope(
                    live, {&retained_exact.bulk, &retained_exact.gas, &retained_exact.carrier});
                if (!retained_exact.bulk.sign() && (retained_gas.sign() || retained_carrier.sign()))
                    throw std::invalid_argument(
                        "General VoF full donor export cannot retain a numerical occupancy defect.");
                transport_charge(budget, 3 * sizeof(GeneralVoFDonorBudget), "donor phase proof records");
                trial.budgets_.push_back({donor.gid, phase_measures(donor), exported_exact, retained_exact,
                    diagnostic(phase_measures(donor)), diagnostic(exported_exact), diagnostic(retained_exact)});
                auto& proof = trial.budgets_.back();
                proof.signed_repair_absolute = donor_signed_repair[key].first;
                proof.signed_repair_bound = donor_signed_repair[key].second;
                proof.route_repair_absolute = route_repair[key];
                proof.route_repair_bound = labels.at(key).bound();
                proof.full_cell_fit_discrepancy = discrepancy;
                live.retain(phase_measures(donor));
                live.retain(exported_exact);
                live.retain(retained_exact);
                live.retain(proof.signed_repair_absolute);
                live.retain(proof.signed_repair_bound);
                live.retain(proof.route_repair_absolute);
                live.retain(proof.route_repair_bound);
                live.retain(proof.full_cell_fit_discrepancy);
                auto& self = amount_for(key, key);
                live.replace(self.gas, retained_gas);
                live.replace(self.carrier, retained_carrier);
                if (!recipients.contains(key))
                    metadata(key, sizeof(decltype(recipients)::value_type));
                recipients[key] = {comm->getRank(), donor.gid};
                ExactAmounts column;
                detail::ExactLiveStorage::Scope column_scope(live, {&column.bulk, &column.gas, &column.carrier});
                for (const auto& [recipient, amount] : amounts[key])
                {
                    if (amount.gas.sign() < 0 || amount.carrier.sign() < 0)
                        throw std::invalid_argument("General VoF negative positive-route phase.");
                    live.add(column.gas, amount.gas);
                    live.add(column.carrier, amount.carrier);
                    if (recipient != key && live.sum(amount.gas, amount.carrier).compare(amount.bulk))
                        throw std::invalid_argument("General VoF routed phase complement differs from bulk.");
                }
                if (column.gas.compare(phase_measures(donor).gas) ||
                    column.carrier.compare(phase_measures(donor).carrier))
                    throw std::invalid_argument("General VoF independent donor phase column does not close.");
                for (const auto& [recipient, amount] : amounts[key])
                {
                    const auto [recipient_owner, recipient_gid] = recipients.at(recipient);
                    const bool self = recipient_owner == comm->getRank();
                    const auto key_bound = [&](const Key& value)
                    {
                        wire::Budget bound{live.operation_bytes() / 8};
                        bound.add(128); // all fixed typed-key fields and framing
                        std::visit(
                            [&](const auto& ancestry)
                            {
                                const auto path = [&](const auto& values)
                                {
                                    if (values.capacity() > bound.limit / sizeof(std::uint64_t))
                                        throw std::invalid_argument("Donor packet key exceeds operation arena.");
                                    bound.add(values.capacity() * sizeof(std::uint64_t));
                                };
                                if constexpr (requires { ancestry.path; })
                                    path(ancestry.path);
                                else if constexpr (requires { ancestry.base.path; })
                                {
                                    path(ancestry.base.path);
                                    path(ancestry.axial.path);
                                }
                            },
                            value.ancestry);
                        return bound.used;
                    };
                    const auto donor_key_bound = key_bound(key), recipient_key_bound = key_bound(recipient);
                    wire::Budget codec{live.operation_bytes() / 8};
                    codec.add(donor_key_bound);
                    codec.add(recipient_key_bound);
                    codec.add(72);
                    const auto measure_bound = [&](const Measure& value)
                    {
                        const auto bytes = value.storage_bytes();
                        codec.add(bytes);
                        codec.add(56); // exact measure certificate framing
                        return bytes + 56;
                    };
                    std::size_t bulk_bound = 0, gas_bound = 0, carrier_bound = 0;
                    if (!self)
                    {
                        bulk_bound = measure_bound(amount.bulk);
                        gas_bound = measure_bound(amount.gas);
                        carrier_bound = measure_bound(amount.carrier);
                    }
                    // Eight times this complete envelope was reserved before
                    // allocation. It covers all three returned encodings plus
                    // the next encode's integer copies/byte-vector growth and
                    // key vectors; no scalar operation overlaps this codec.
                    auto dk = AMR::encode_adaptive_key(key), rk = AMR::encode_adaptive_key(recipient);
                    if (dk.capacity() > 2 * donor_key_bound || rk.capacity() > 2 * recipient_key_bound)
                        throw std::invalid_argument("Donor packet key capacity exceeds its admission.");
                    std::vector<char> bulk, gas, carrier;
                    if (!self)
                    {
                        bulk = amount.bulk.encode(bulk_bound);
                        gas = amount.gas.encode(gas_bound);
                        carrier = amount.carrier.encode(carrier_bound);
                        if (bulk.capacity() > 2 * bulk_bound || gas.capacity() > 2 * gas_bound ||
                            carrier.capacity() > 2 * carrier_bound)
                            throw std::invalid_argument("Donor phase encoded capacity exceeds its admission.");
                    }
                    const auto payload_bound = dk.size() + rk.size() + bulk.size() + gas.size() + carrier.size() + 72;
                    if (payload_bound > limit / 2)
                        throw std::invalid_argument("Donor phase packet capacity exceeds its budget.");
                    transport_charge(budget, 3 * sizeof(OwnerPacket), "donor phase packet metadata");
                    transport_charge(budget, 2 * payload_bound, "donor phase packet capacity");
                    wire::Writer w;
                    w.data.reserve(payload_bound);
                    if (w.data.capacity() > 2 * payload_bound)
                        throw std::invalid_argument("Donor phase writer capacity exceeds admission.");
                    w.bytes(dk);
                    w.word(donor.gid);
                    w.word(donor.revision);
                    w.bytes(rk);
                    w.word(recipient_gid);
                    w.word(self ? 1 : 0);
                    if (self)
                    {
                        // Keep immutable exact handles until this invocation's
                        // self packets have been authenticated and consumed.
                        // Cover retained capacity and simultaneous old/new
                        // vector buffers during geometric capacity growth.
                        transport_charge(
                            budget, 3 * sizeof(LocalAmount) + dk.size() + rk.size(), "donor phase local amount record");
                        // Immutable handles alias the tracked amount slots, which
                        // remain live through this builder; do not charge twice.
                        live.retain({amount.bulk, amount.gas, amount.carrier});
                        w.word(local_amounts.size());
                        local_amounts.push_back({key, recipient, donor.gid, donor.revision, recipient_gid, amount});
                    }
                    else
                    {
                        w.bytes(bulk);
                        w.bytes(gas);
                        w.bytes(carrier);
                    }
                    transfer_packets.push_back({recipient_owner, std::move(w.data)});
                }
            }
        });
    std::vector<Routed>().swap(routed);
    auto received = exchange_owner_packets(transfer_packets, comm, limit);
    std::vector<OwnerPacket>().swap(transfer_packets);
    std::shared_ptr<Data> candidate;
    std::vector<Key> donor_keys;
    struct Transfer
    {
        Key donor, recipient;
        std::uint64_t gid = 0, revision = 0;
        int owner = -1;
        ExactAmounts amount;
    };
    std::vector<Transfer> transfers;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::set<std::pair<Key, Key>> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto dk = AMR::decode_adaptive_key(r.bytes());
                auto gid = r.word(), revision = r.word();
                auto rk = AMR::decode_adaptive_key(r.bytes());
                auto recipient_gid = r.word();
                const auto& recipient = data.cells.at(rk);
                if (recipient.owner != comm->getRank() || recipient.gid != recipient_gid ||
                    !seen.emplace(dk, rk).second)
                    throw std::invalid_argument("General VoF positive transfer recipient differs.");
                ExactAmounts amount;
                const auto tag = r.word();
                if (tag == 1)
                {
                    const auto index = r.word();
                    r.done();
                    if (packet.peer != comm->getRank() || index >= local_amounts.size())
                        throw std::invalid_argument("General VoF foreign/out-of-range local amount reference.");
                    const auto& original = local_amounts[std::size_t(index)];
                    const auto& donor = data.cells.at(dk);
                    if (donor.owner != comm->getRank() || donor.gid != gid || donor.revision != revision ||
                        original.donor != dk || original.recipient != rk || original.donor_gid != gid ||
                        original.donor_revision != revision || original.recipient_gid != recipient_gid)
                        throw std::invalid_argument("General VoF local amount metadata mismatch.");
                    amount = original.amount;
                }
                else if (tag == 0)
                {
                    if (packet.peer == comm->getRank())
                        throw std::invalid_argument("General VoF self amount omitted its local reference.");
                    amount.bulk = Measure::decode(r.bytes(), limit);
                    amount.gas = Measure::decode(r.bytes(), limit);
                    amount.carrier = Measure::decode(r.bytes(), limit);
                    r.done();
                }
                else
                    throw std::invalid_argument("General VoF invalid amount route tag.");
                budget.add(sizeof(Transfer));
                budget.add(amount.bulk.storage_bytes());
                budget.add(amount.gas.storage_bytes());
                budget.add(amount.carrier.storage_bytes());
                transfers.push_back({dk, rk, gid, revision, packet.peer, std::move(amount)});
                donor_keys.push_back(dk);
            }
        });
    std::vector<OwnerPacket>().swap(received.packets);
    std::vector<LocalAmount>().swap(local_amounts);
    auto donor_owners = data.hierarchy->lookup_cells(donor_keys);
    wire::collective(*comm,
        [&]
        {
            candidate = std::make_shared<Data>(data);
            candidate->control = trial.transaction_;
            std::map<Key, ExactAmounts> totals;
            for (std::size_t i = 0; i < transfers.size(); ++i)
            {
                const auto& transfer = transfers[i];
                const auto& owner = donor_owners[i];
                if (owner.owner != transfer.owner || owner.revision != transfer.revision ||
                    owner.address != transfer.gid)
                    throw std::invalid_argument("General VoF positive transfer donor differs.");
                const auto& amount = transfer.amount;
                auto& total = totals[transfer.recipient];
                total.bulk = total.bulk + amount.bulk;
                total.gas = total.gas + amount.gas;
                total.carrier = total.carrier + amount.carrier;
                SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact> exact{
                    amount.bulk, amount.gas, amount.carrier};
                trial.transfers_.push_back({transfer.gid, data.cells.at(transfer.recipient).gid, transfer.owner,
                    comm->getRank(), exact, diagnostic(exact)});
            }
            candidate->gas.clear();
            candidate->carrier.clear();
            candidate->phases.clear();
            candidate->roundoff.clear();
            candidate->occupancy.clear();
            for (const auto& key : data.topology->order)
            {
                auto& cell = candidate->cells.at(key);
                const auto& total = totals.at(key);
                if (total.bulk.compare(Measure::volume(*cell.phase.geometry)))
                    throw std::invalid_argument("General VoF positive transfers do not fill actual target.");
                const auto before = phase_measures(data.cells.at(key));
                if (((total.gas + total.carrier) - total.bulk).compare((before.gas + before.carrier) - before.bulk))
                    throw std::invalid_argument(
                        "General VoF transport changed the independent source occupancy defect.");
                cell.retained_interface.reset();
                if (inventory_policy_for(data.options.arithmetic_mode) == PhaseInventoryPolicy::ExperimentalExact)
                {
                    cell.experimental_exact = detail::experimental_advance(*data.cells.at(key).experimental_exact,
                        {total.bulk, total.gas, total.carrier}, *cell.phase.geometry, cell.native_volume, limit);
                    const auto view = detail::experimental_view(*cell.experimental_exact, cell.native_volume);
                    cell.phase.gas_volume = view.phases.gas;
                    cell.phase.carrier_volume = view.phases.carrier;
                    cell.occupancy = view.error;
                    cell.roundoff = view.roundoff;
                }
                else
                {
                    // Round each independently conserved phase ONCE. The next step
                    // starts from these published values, never total's rational tail.
                    const auto gas = detail::round_phase_value(total.gas);
                    const auto carrier = detail::round_phase_value(total.carrier);
                    cell.phase.gas_volume = gas.value;
                    cell.phase.carrier_volume = carrier.value;
                    cell.roundoff = {gas.absolute_error, carrier.absolute_error};
                    const auto rounding = finite_bound(gas.absolute_error) + finite_bound(carrier.absolute_error);
                    cell.occupancy.admitted_bound =
                        detail::phase_upper_bound(finite_bound(cell.occupancy.admitted_bound) + rounding);
                    cell.occupancy.phase_input_bound =
                        detail::phase_upper_bound(finite_bound(cell.occupancy.phase_input_bound) + rounding);
                    const auto published = phase_measures(cell);
                    const auto diagnostic =
                        detail::round_phase_residual(published.gas + published.carrier - published.bulk);
                    cell.occupancy.difference = diagnostic.value;
                    cell.occupancy.admitted_bound = detail::phase_upper_bound(
                        finite_bound(cell.occupancy.admitted_bound) + finite_bound(diagnostic.absolute_error));
                    cell.occupancy.phase_input_bound = detail::phase_upper_bound(
                        finite_bound(cell.occupancy.phase_input_bound) + finite_bound(diagnostic.absolute_error));
                }
                candidate->phases.push_back({cell.native_volume, cell.phase.gas_volume, cell.phase.carrier_volume});
                candidate->roundoff.push_back(cell.roundoff);
                candidate->occupancy.push_back(cell.occupancy);
                if (cell.declaration != GeneralVoFPhaseGeometry::Infer)
                {
                    const auto metric = cell.phase.geometry->moments().volume;
                    const double minority = cell.declaration == GeneralVoFPhaseGeometry::PureGas
                                                ? cell.phase.carrier_volume
                                                : cell.phase.gas_volume;
                    if (minority >
                        metric.binary64_absolute_error() + 8 * std::numeric_limits<double>::epsilon() * metric.value)
                        cell.declaration = GeneralVoFPhaseGeometry::Infer;
                }
                validate_cell(cell, data.options);
                candidate->gas.push_back(cell.phase.gas_volume);
                candidate->carrier.push_back(cell.phase.carrier_volume);
            }
            refresh_experimental_catalog(*candidate);
            std::copy(candidate->roundoff.begin(), candidate->roundoff.end(), trial.roundoff_.begin());
            current(data);
        });
    // Independent global phase checks use only fixed-size finite reductions.
    // Exact donor columns above certify the pre-publication route algebra;
    // these checks account explicitly for the once-rounded accepted values.
    std::array<double, 12> local{}, global{};
    wire::collective(*comm,
        [&]
        {
            std::array<::SimpleFluid::detail::CompensatedSum<>, 6> sums;
            for (const auto& key : data.topology->order)
            {
                const auto& old = data.cells.at(key);
                const auto& next = candidate->cells.at(key);
                sums[0] += static_cast<long double>(next.phase.gas_volume) - old.phase.gas_volume;
                sums[1] += static_cast<long double>(next.phase.carrier_volume) - old.phase.carrier_volume;
                sums[2] += next.roundoff.gas;
                sums[3] += next.roundoff.carrier;
                if (inventory_policy_for(data.options.arithmetic_mode) == PhaseInventoryPolicy::ExperimentalExact)
                {
                    sums[2] += old.roundoff.gas;
                    sums[3] += old.roundoff.carrier;
                }
                sums[4] += std::abs(static_cast<long double>(next.phase.gas_volume)) +
                           std::abs(static_cast<long double>(old.phase.gas_volume));
                sums[5] += std::abs(static_cast<long double>(next.phase.carrier_volume)) +
                           std::abs(static_cast<long double>(old.phase.carrier_volume));
            }
            for (std::size_t i = 0; i < sums.size(); ++i)
            {
                const auto exact_local = sums[i].value();
                local[i] = static_cast<double>(exact_local);
                if (!std::isfinite(local[i]))
                    throw std::invalid_argument("Finite phase reduction wire range is inconclusive.");
                const auto error = std::abs(exact_local - static_cast<long double>(local[i]));
                auto bound = static_cast<double>(error);
                if (static_cast<long double>(bound) < error)
                    bound = std::nextafter(bound, std::numeric_limits<double>::infinity());
                if (!std::isfinite(bound) || static_cast<long double>(bound) < error)
                    throw std::invalid_argument("Finite phase reduction conversion bound is inconclusive.");
                local[i + 6] = bound;
            }
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, int(local.size()), local.data(), global.data());
    wire::collective(*comm,
        [&]
        {
            const long double count =
                48 * static_cast<long double>(data.hierarchy->global_cells()) + 64 * comm->getSize() + 128;
            const auto product = count * std::numeric_limits<double>::epsilon();
            if (!(product < .25L))
                throw std::invalid_argument("Finite phase reduction operation bound is exhausted.");
            // Per phase: three compensated streams cost at most14N
            // arithmetic operations, six value/radius reductions6(P-1), plus
            // conversion/finalization.48N+64P+128 conservatively covers those
            // paths AND the uncertainty of their observed L1 scale. The guard
            // implies an integer count<2^51, exactly representable in double.
            const auto u =
                Measure::from_value(static_cast<double>(count)).scaled(std::numeric_limits<double>::epsilon());
            const auto gamma_exact = u.divided_by(Measure::from_value(1) - u);
            const auto gamma = detail::phase_upper_bound(gamma_exact.divided_by(Measure::from_value(1) - gamma_exact));
            const auto up_add = [](long double x, long double y)
            {
                const auto sum = x + y;
                return sum == 0 && x == 0 && y == 0 ? 0.L
                                                    : std::nextafter(sum, std::numeric_limits<long double>::infinity());
            };
            const auto up_product = [](long double x, long double y)
            {
                if (x == 0 || y == 0)
                    return 0.L;
                return std::nextafter(x * y, std::numeric_limits<long double>::infinity());
            };
            for (unsigned phase = 0; phase < 2; ++phase)
            {
                // Convert/reduce only supported binary64 scalars. Include the
                // local wire radii and reduction/local arithmetic on phase L1;
                // no geometric/native bulk sets a minority tolerance floor.
                const long double radius = up_add(global[phase + 6], global[phase + 8]);
                const long double publication = global[phase + 2];
                const long double scale =
                    up_add(up_add(global[phase + 4], global[phase + 10]), up_add(publication, radius));
                const auto bound = up_add(up_add(publication, radius), up_product(gamma, scale));
                if (!std::isfinite(global[phase]) || !std::isfinite(bound) ||
                    std::abs(static_cast<long double>(global[phase])) > bound)
                    throw std::invalid_argument("Finite General VoF phase conservation exceeds its publication bound.");
            }
        });
    import_phases(*candidate);
    wire::collective(*comm, [&] { retained_budget(*candidate); });
    trial.candidate_ = std::move(candidate);
    return trial;
}
} // namespace Exact

}
