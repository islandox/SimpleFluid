#include "solvers/details/GeneralVoFPhiClosureSupport.hh"
#include <limits>
namespace SimpleFluid::detail::phi_support
{
Codec::Codec(std::size_t budget) : limit(budget)
{}
Codec::Codec(const Bytes& input, std::size_t budget) : limit(input.size()), decode_limit(budget)
{
    if (input.size() > decode_limit)
        throw PhiProposalResource("Phi decode copy exceeds its live packet arena.");
    bytes = input;
    if (bytes.capacity() > decode_limit)
        throw PhiProposalResource("Phi decode capacity exceeds its live packet arena.");
    decode_used = bytes.capacity();
}
void Codec::put(std::uint64_t v)
{
    if (bytes.size() > limit || limit - bytes.size() < 8)
        throw PhiProposalResource("Phi closure framing exceeds its budget.");
    for (unsigned i = 0; i < 8; ++i)
        bytes.push_back(char(v >> (8 * i)));
}
std::uint64_t Codec::get()
{
    if (cursor > bytes.size() || bytes.size() - cursor < 8)
        throw std::runtime_error("Truncated Phi closure packet.");
    std::uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i)
        v |= std::uint64_t(static_cast<unsigned char>(bytes[cursor++])) << (8 * i);
    return v;
}
void Codec::put(const Bytes& v)
{
    if (bytes.size() > limit || limit - bytes.size() < 8 || v.size() > limit - bytes.size() - 8)
        throw PhiProposalResource("Phi closure payload exceeds its budget.");
    put(v.size());
    bytes.insert(bytes.end(), v.begin(), v.end());
}
Bytes Codec::blob()
{
    const auto n = get();
    if (n > bytes.size() - cursor)
        throw std::runtime_error("Invalid Phi closure framing.");
    if (n > decode_limit - decode_used)
        throw PhiProposalResource("Phi decoded blobs exceed their live packet arena.");
    Bytes v(bytes.begin() + cursor, bytes.begin() + cursor + n);
    if (v.capacity() > decode_limit - decode_used)
        throw PhiProposalResource("Phi decoded blob capacity exceeds its live packet arena.");
    // Conservatively retain every blob's capacity until this Codec dies,
    // even when a scalar decoder has already released its temporary blob.
    decode_used += v.capacity();
    cursor += n;
    return v;
}
void Codec::done()
{
    if (cursor != bytes.size())
        throw std::runtime_error("Trailing Phi closure bytes.");
}
std::size_t packet_capacity(const std::vector<OwnerPacket>& packets, std::size_t budget)
{
    if (packets.capacity() > budget / sizeof(OwnerPacket))
        throw PhiProposalResource("Phi packet metadata capacity exceeds its arena.");
    std::size_t used = packets.capacity() * sizeof(OwnerPacket);
    for (const auto& packet : packets)
    {
        if (packet.bytes.capacity() > budget - used)
            throw PhiProposalResource("Phi packet payload capacities exceed their arena.");
        used += packet.bytes.capacity();
    }
    return used;
}
std::size_t phi_decode_budget(const std::vector<OwnerPacket>& inputs, const std::vector<OwnerPacket>& outputs,
    const Teuchos::Comm<int>& comm, std::size_t arena)
{
    std::size_t remaining = 0;
    collective(comm,
        [&]
        {
            const auto input = packet_capacity(inputs, arena);
            const auto output = packet_capacity(outputs, arena - input);
            remaining = arena - input - output;
        });
    return remaining;
}
OwnerExchange exchange_phi_packets(const std::vector<OwnerPacket>& packets,
    const Teuchos::RCP<const Teuchos::Comm<int>>& comm, std::size_t input_cap, std::size_t exchange_cap,
    std::size_t arena_cap, std::size_t* resident_input)
{
    std::size_t resident = 0, payload = 0, wire = 0;
    collective(*comm,
        [&]
        {
            if (exchange_cap > arena_cap || packets.capacity() > (arena_cap - exchange_cap) / sizeof(OwnerPacket))
                throw PhiProposalResource("Phi caller packet metadata exceeds its arena share.");
            resident = packets.capacity() * sizeof(OwnerPacket);
            for (const auto& packet : packets)
            {
                if (packet.bytes.capacity() > arena_cap - exchange_cap - resident)
                    throw PhiProposalResource("Phi caller packet capacities exceed their arena share.");
                resident += packet.bytes.capacity();
                if (packet.bytes.size() > input_cap - payload)
                    throw PhiProposalResource("Phi outgoing payload exceeds its unchanged input cap.");
                payload += packet.bytes.size();
                if (packet.bytes.size() > std::numeric_limits<std::size_t>::max() - sizeof(std::uint64_t) ||
                    wire > std::numeric_limits<std::size_t>::max() - sizeof(std::uint64_t) - packet.bytes.size())
                    throw PhiProposalResource("Phi packet wire count overflow.");
                wire += sizeof(std::uint64_t) + packet.bytes.size();
            }
        });
    if (resident_input)
        *resident_input = resident;
    const auto context = [&](const std::exception& error)
    {
        return std::string(error.what()) + " Phi packet arena: payload=" + std::to_string(payload) +
               ", sent_wire=" + std::to_string(wire) +
               ", serial_received_wire=" + (comm->getSize() == 1 ? std::to_string(wire) : std::string("not-local")) +
               ", resident_input=" + std::to_string(resident) + ", input_cap=" + std::to_string(input_cap) +
               ", exchange_cap=" + std::to_string(exchange_cap) + ", arena_cap=" + std::to_string(arena_cap);
    };
    try
    {
        // The exchange cap includes its packed send, two receive copies and
        // explicit route/output metadata. Caller inputs are additional; their
        // audited capacities occupy the disjoint remainder of the same arena.
        return exchange_owner_packets(packets, comm, exchange_cap);
    }
    catch (const std::length_error& error)
    {
        throw PhiProposalResource(context(error));
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(context(error));
    }
}
} // namespace SimpleFluid::detail::phi_support
