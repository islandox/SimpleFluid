/** @file HaloExchange.cc
 * @brief Authenticated complete-node fans and owner phase halo exchange.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
// Every incident owner registers its actual cell once. The node authority
// authenticates those registrations and returns the complete canonical fan;
// geometric coincidence and whichever ghost rows happen to be resident play no
// part in defining the interpolation stencil.
void complete_node_fans(GeneralVoF::Data& data, GeneralVoF::Data::Topology& topology)
{
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::vector<OwnerPacket> requests;
    std::map<std::uint64_t, const AMR::AdaptiveNode*> native_nodes;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::map<std::uint64_t, std::set<Key>> local;
            for (const auto& node : topology.draft.nodes)
            {
                budget.add(sizeof(decltype(native_nodes)::value_type) + 4 * sizeof(void*));
                native_nodes.emplace(node.gid, &node);
            }
            for (const auto& tile : topology.draft.tiles)
                for (auto node : tile.node_gids)
                    for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                        if (key && data.hierarchy->owned().contains(*key))
                        {
                            budget.add(sizeof(Key) + AMR::encode_adaptive_key(*key).size() + 8 * sizeof(void*));
                            local[node].insert(*key);
                        }
            for (const auto& [node, keys] : local)
            {
                wire::Writer w;
                const auto& identity = native_nodes.at(node)->identity;
                budget.packet(32 + identity.size());
                w.word(node);
                w.bytes(identity);
                w.word(keys.size());
                for (const auto& key : keys)
                {
                    const auto bytes = AMR::encode_adaptive_key(key);
                    budget.add(16 + bytes.size());
                    w.bytes(bytes);
                    w.word(data.hierarchy->owned().at(key).revision);
                }
                requests.push_back({int(node % std::uint64_t(comm->getSize())), std::move(w.data)});
                budget.add(sizeof(decltype(topology.node_fans)::value_type) + 4 * sizeof(void*));
                topology.node_fans.emplace(node, KeySet{});
            }
        });
    auto received = exchange_owner_packets(requests, comm, limit);
    struct Registration
    {
        std::uint64_t node, revision;
        Key key;
        int peer;
    };
    std::vector<Registration> registrations;
    std::vector<Key> keys;
    std::map<std::uint64_t, std::set<int>> subscribers;
    std::map<std::uint64_t, AMR::AdaptiveBytes> identities;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                const auto node = r.word();
                auto identity = r.bytes();
                budget.add(identity.size() + sizeof(identity) + 8 * sizeof(void*));
                if (node % std::uint64_t(comm->getSize()) != std::uint64_t(comm->getRank()) ||
                    !subscribers[node].insert(packet.peer).second)
                    throw std::invalid_argument("General VoF duplicate or misrouted node fan registration.");
                const auto [it, fresh] = identities.emplace(node, identity);
                if (!fresh && it->second != identity)
                    throw std::invalid_argument("General VoF node fan identity differs.");
                const auto count = r.word();
                if (!count || count > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF node fan budget exhausted.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto bytes = r.bytes();
                    budget.add(bytes.size() + sizeof(Registration) + sizeof(Key));
                    auto key = AMR::decode_adaptive_key(bytes);
                    const auto revision = r.word();
                    registrations.push_back({node, revision, key, packet.peer});
                    keys.push_back(std::move(key));
                }
                r.done();
            }
        });
    const auto owners = data.hierarchy->lookup_cells(keys);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::map<std::uint64_t, std::set<Key>> fans;
            for (std::size_t i = 0; i < registrations.size(); ++i)
            {
                const auto& entry = registrations[i];
                if (owners[i].owner != entry.peer || owners[i].revision != entry.revision ||
                    !fans[entry.node].insert(entry.key).second)
                    throw std::invalid_argument("General VoF unauthenticated node fan cell.");
                if (fans[entry.node].size() > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF complete node fan budget exhausted.");
            }
            for (const auto& [node, fan] : fans)
                for (auto peer : subscribers.at(node))
                {
                    wire::Writer w;
                    budget.packet(24 + identities.at(node).size());
                    w.word(node);
                    w.bytes(identities.at(node));
                    w.word(fan.size());
                    for (const auto& key : fan)
                    {
                        const auto bytes = AMR::encode_adaptive_key(key);
                        budget.add(8 + bytes.size());
                        w.bytes(bytes);
                    }
                    replies.push_back({peer, std::move(w.data)});
                }
        });
    auto returned = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(native_nodes.size() * (sizeof(decltype(native_nodes)::value_type) + 4 * sizeof(void*)));
            budget.add(
                topology.node_fans.size() * (sizeof(decltype(topology.node_fans)::value_type) + 4 * sizeof(void*)));
            std::set<std::uint64_t> seen;
            for (const auto& packet : returned.packets)
            {
                wire::Reader r{packet.bytes};
                const auto node = r.word();
                const auto identity = r.bytes();
                if (!topology.node_fans.contains(node) || !seen.insert(node).second ||
                    packet.peer != int(node % std::uint64_t(comm->getSize())))
                    throw std::invalid_argument("General VoF unexpected node fan reply.");
                const auto native = native_nodes.find(node);
                if (native == native_nodes.end() || native->second->identity != identity)
                    throw std::invalid_argument("General VoF node fan reply identity differs.");
                const auto count = r.word();
                if (!count || count > data.options.maximum_stencil_cells)
                    throw std::invalid_argument("General VoF node fan reply budget exhausted.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    const auto bytes = r.bytes();
                    budget.add(bytes.size() + sizeof(Key) + 8 * sizeof(void*));
                    if (!topology.node_fans.at(node)
                            .insert(AMR::decode_adaptive_key(bytes), topology.graph_keys, &budget)
                            .second)
                        throw std::invalid_argument("General VoF duplicate node fan member.");
                }
                r.done();
            }
            if (seen.size() != topology.node_fans.size())
                throw std::invalid_argument("General VoF incomplete canonical node fan.");
        });
}
template<ArithmeticMode Mode> void import_phases(GeneralVoF::Data& data)
{
    auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
                if (cell.owner != comm->getRank())
                {
                    const auto encoded = AMR::encode_adaptive_key(key);
                    budget.packet(encoded.size() + 16);
                    wire::Writer w;
                    w.bytes(encoded);
                    w.word(cell.revision);
                    requests.push_back({cell.owner, std::move(w.data)});
                }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto bytes = r.bytes();
                auto key = AMR::decode_adaptive_key(bytes);
                auto revision = r.word();
                r.done();
                const auto& cell = data.cells.at(key);
                if (cell.owner != comm->getRank() || cell.revision != revision)
                    throw std::invalid_argument("General VoF stale phase request.");
                wire::Writer w;
                w.bytes(bytes);
                w.word(revision);
                w.word(Mode == ArithmeticMode::Ordinary ? 4 : 3);
                w.word(std::uint64_t(data.options.arithmetic_mode));
                w.word(cell.canonical_face_count);
                finite_real(w, cell.ordinary_quota_scale);
                real(w, cell.phase.gas_volume);
                real(w, cell.phase.carrier_volume);
                w.word(std::uint64_t(cell.declaration));
                w.word(cell.inherited);
                if constexpr (Mode == ArithmeticMode::Ordinary)
                    w.word(cell.retained_interface && !cell.retained_interface->pure_gas &&
                           !cell.retained_interface->pure_carrier);
                real(w, cell.roundoff.gas);
                real(w, cell.roundoff.carrier);
                real(w, cell.finite_history.gas);
                real(w, cell.finite_history.carrier);
                for (const auto* value :
                    {&cell.occupancy.difference, &cell.occupancy.admitted_bound, &cell.occupancy.native_conversion,
                        &cell.occupancy.phase_input_bound, &cell.occupancy.fit_residual, &cell.occupancy.fit_bound,
                        &cell.occupancy.transferred_bound, &cell.occupancy.fit_allocation_bound})
                    real(w, *value);
                if (data.options.arithmetic_mode == ArithmeticMode::Exact)
                {
                    if (!cell.experimental_exact)
                        throw std::invalid_argument("Missing exact owner phase state.");
                    const auto encoded = detail::encode_experimental(*cell.experimental_exact, limit);
                    budget.add(8 + encoded.size());
                    w.bytes(encoded);
                }
                budget.packet(w.data.size());
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto received = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto key = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word();
                auto& cell = data.cells.at(key);
                if (cell.owner != packet.peer || cell.owner == comm->getRank() || cell.revision != revision ||
                    !seen.insert(key).second)
                    throw std::invalid_argument("General VoF unauthenticated phase reply.");
                if (r.word() != (Mode == ArithmeticMode::Ordinary ? 4 : 3) ||
                    r.word() != std::uint64_t(data.options.arithmetic_mode))
                    throw std::invalid_argument("General VoF ghost inventory policy differs.");
                const auto count = r.word();
                const auto quota = finite_real(r);
                if (!count || count > limit / sizeof(std::size_t) || !std::isfinite(quota) || quota < 0 ||
                    (data.options.arithmetic_mode == ArithmeticMode::Ordinary && !(quota > 0)))
                    throw std::invalid_argument("General VoF owner face quota is malformed.");
                cell.canonical_face_count = static_cast<std::size_t>(count);
                cell.ordinary_quota_scale = quota;
                cell.phase.gas_volume = real(r);
                cell.phase.carrier_volume = real(r);
                cell.declaration = GeneralVoFPhaseGeometry(r.word());
                cell.inherited = r.word();
                if constexpr (Mode == ArithmeticMode::Ordinary)
                {
                    const auto requires_interface = r.word();
                    if (requires_interface > 1)
                        throw std::invalid_argument("General VoF malformed owner interface requirement.");
                    cell.requires_interface = requires_interface;
                }
                cell.roundoff.gas = real(r);
                cell.roundoff.carrier = real(r);
                cell.finite_history.gas = real(r);
                cell.finite_history.carrier = real(r);
                for (auto* value :
                    {&cell.occupancy.difference, &cell.occupancy.admitted_bound, &cell.occupancy.native_conversion,
                        &cell.occupancy.phase_input_bound, &cell.occupancy.fit_residual, &cell.occupancy.fit_bound,
                        &cell.occupancy.transferred_bound, &cell.occupancy.fit_allocation_bound})
                    *value = real(r);
                if (data.options.arithmetic_mode == ArithmeticMode::Exact)
                    cell.experimental_exact = detail::decode_experimental(r.bytes(), limit);
                else
                    cell.experimental_exact.reset();
                r.done();
                validate_cell<Mode>(cell, data.options);
            }
            if (seen.size() != requests.size())
                throw std::invalid_argument("General VoF incomplete phase replies.");
        });
}
template void import_phases<ArithmeticMode::Ordinary>(GeneralVoF::Data&);
template void import_phases<ArithmeticMode::BoundedCertification>(GeneralVoF::Data&);
template void import_phases<ArithmeticMode::Exact>(GeneralVoF::Data&);
} // namespace SimpleFluid::VoF::general_vof_detail
