#include "FVM/details/GeneralVoF/AggregateBudgets.hh"

namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
constexpr std::uint64_t request_tag = 0x5346414747433031ULL, reply_tag = 0x5346414747433032ULL;
constexpr std::size_t packet_bytes = 8 * sizeof(std::uint64_t);
using GO = MeshHandle<>::global_ordinal_type;
void stamp(wire::Writer& writer, std::uint64_t tag, const PLIC::GeometryPacketContext& context)
{
    writer.word(tag);
    for (auto word : context.publication)
        writer.word(word);
}
void stamp(wire::Reader& reader, std::uint64_t tag, const PLIC::GeometryPacketContext& context)
{
    if (reader.word() != tag)
        throw std::invalid_argument("Aggregate authority packet protocol differs.");
    for (auto word : context.publication)
        if (reader.word() != word)
            throw std::invalid_argument("Aggregate authority source or trial context differs.");
}
const Cell& binding(const GeneralVoF::Data& data, std::uint64_t gid)
{
    if (gid > std::uint64_t(std::numeric_limits<GO>::max()))
        throw std::invalid_argument("Aggregate authority donor GID exceeds the native map range.");
    const auto key = data.topology->by_gid.find(gid);
    if (key == data.topology->by_gid.end())
        throw std::invalid_argument("Aggregate authority donor is not retained.");
    const auto cell = data.cells.find(key->second);
    if (cell == data.cells.end() || cell->second.gid != gid || cell->second.key != key->second ||
        cell->second.owner < 0 || cell->second.owner >= data.hierarchy->communicator()->getSize())
        throw std::invalid_argument("Aggregate authority donor identity/owner binding differs.");
    return cell->second;
}
std::size_t count(std::uint64_t word)
{
    if (!word || word > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Aggregate authority count is zero or outside the host range.");
    return static_cast<std::size_t>(word);
}
} // namespace
std::map<std::uint64_t, std::size_t> aggregate_authority_counts(const GeneralVoF::Data& data,
    const std::map<std::uint64_t, std::size_t>& local_counts, const PLIC::GeometryPacketContext& context)
{
    namespace failure = ::SimpleFluid::detail::vof_failure;
    failure::Scope scope("interface_aggregate_authority_counts");
    scope.record.arithmetic_mode = ArithmeticMode::Ordinary;
    scope.record.mark(failure::Field::Arithmetic);
    const auto comm = data.hierarchy->communicator();
    const auto rank = comm->getRank();
    const auto limit = data.options.byte_budget;
    scope.record.byte_budget = limit;
    scope.record.mark(failure::Field::ByteBudget);
    wire::Budget budget{limit};
    std::vector<OwnerPacket> packets;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (data.options.arithmetic_mode != ArithmeticMode::Ordinary || context.publication[0] != data.control ||
                context.publication[1] != data.hierarchy->topology_generation() ||
                context.publication[2] != data.hierarchy->partition_generation())
                throw std::invalid_argument(
                    "Aggregate authority source context is stale or has another arithmetic mode.");
            for (const auto& [gid, local] : local_counts)
            {
                const auto& cell = binding(data, gid);
                if (!local || local > std::numeric_limits<std::uint64_t>::max())
                    throw std::invalid_argument("Aggregate authority local count is zero or exceeds the wire range.");
                budget.packet(packet_bytes);
                wire::Writer writer;
                stamp(writer, request_tag, context);
                writer.word(gid);
                writer.word(cell.revision);
                writer.word(local);
                packets.push_back({cell.owner, std::move(writer.data)});
            }
        });
    const auto incoming = exchange_owner_packets(packets, comm, limit);
    packets.clear();
    struct Subscription
    {
        int peer;
        std::uint64_t gid, revision;
    };
    std::vector<Subscription> subscriptions;
    std::map<std::uint64_t, std::size_t> totals;
    wire::collective(*comm,
        [&]
        {
            std::set<std::pair<int, std::uint64_t>> seen;
            for (const auto& packet : incoming.packets)
            {
                budget.packet(packet.bytes.size());
                wire::Reader reader{packet.bytes};
                stamp(reader, request_tag, context);
                const auto gid = reader.word(), revision = reader.word();
                const auto demand = count(reader.word());
                reader.done();
                const auto& cell = binding(data, gid);
                if (packet.peer < 0 || packet.peer >= comm->getSize() || cell.owner != rank ||
                    cell.revision != revision ||
                    !data.topology->native.mesh->owned_cell_map()->isNodeGlobalElement(GO(gid)))
                    throw std::invalid_argument("Aggregate authority request donor ownership/revision differs.");
                budget.add(sizeof(decltype(seen)::value_type) + 4 * sizeof(void*));
                if (!seen.emplace(packet.peer, gid).second)
                    throw std::invalid_argument("Aggregate authority request repeats a peer/donor pair.");
                const auto found = totals.find(gid);
                const auto prior = found == totals.end() ? 0 : found->second;
                if (demand > std::numeric_limits<std::size_t>::max() - prior)
                    throw std::invalid_argument("Aggregate authority global count overflows.");
                if (found == totals.end())
                {
                    budget.add(sizeof(decltype(totals)::value_type) + 4 * sizeof(void*));
                    totals.emplace(gid, demand);
                }
                else
                    found->second += demand;
                budget.add(sizeof(Subscription));
                subscriptions.push_back({packet.peer, gid, revision});
            }
            for (const auto& subscription : subscriptions)
            {
                const auto total = totals.at(subscription.gid);
                if (total > std::numeric_limits<std::uint64_t>::max())
                    throw std::invalid_argument("Aggregate authority global count exceeds the wire range.");
                budget.packet(packet_bytes);
                wire::Writer writer;
                stamp(writer, reply_tag, context);
                writer.word(subscription.gid);
                writer.word(subscription.revision);
                writer.word(total);
                packets.push_back({subscription.peer, std::move(writer.data)});
            }
        });
    const auto returned = exchange_owner_packets(packets, comm, limit);
    std::map<std::uint64_t, std::size_t> result;
    wire::collective(*comm,
        [&]
        {
            for (const auto& packet : returned.packets)
            {
                budget.packet(packet.bytes.size());
                wire::Reader reader{packet.bytes};
                stamp(reader, reply_tag, context);
                const auto gid = reader.word(), revision = reader.word();
                const auto global = count(reader.word());
                reader.done();
                const auto local = local_counts.find(gid);
                const auto& cell = binding(data, gid);
                if (local == local_counts.end() || packet.peer != cell.owner || revision != cell.revision ||
                    global < local->second)
                    throw std::invalid_argument("Aggregate authority reply source/revision/count differs.");
                budget.add(sizeof(decltype(result)::value_type) + 4 * sizeof(void*));
                if (!result.emplace(gid, global).second)
                    throw std::invalid_argument("Aggregate authority reply repeats a donor.");
            }
            if (result.size() != local_counts.size())
                throw std::invalid_argument("Aggregate authority reply coverage is incomplete.");
        });
    return result;
}
} // namespace SimpleFluid::VoF::general_vof_detail
