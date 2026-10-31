/** @file SparseOwnerExchange.cc
 * @brief Two-phase, peer-coalesced variable packet exchange with collective gates.
 */
#include "parallel/SparseOwnerExchange.hh"
#include "parallel/SparseOwnerExchangeFraming.hh"
#include <Kokkos_Core.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Distributor.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
namespace SimpleFluid
{
namespace
{
using Word = std::uint64_t;
std::size_t plus(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::length_error("Owner packet size overflow.");
    return a + b;
}
std::size_t times(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::length_error("Owner packet size overflow.");
    return a * b;
}
template<class F> void collectively(const Teuchos::Comm<int>& comm, F&& f)
{
    std::exception_ptr error;
    int failed = 0, any = 0;
    try
    {
        f();
    }
    catch (const std::length_error&)
    {
        error = std::current_exception();
        failed = 1;
    }
    catch (...)
    {
        error = std::current_exception();
        failed = 2;
    }
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &failed, &any);
    if (any == 1)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::length_error("Owner packet validation failed on another rank.");
    }
    if (any == 2)
    {
        if (failed == 2)
            std::rethrow_exception(error);
        throw std::invalid_argument("Owner packet validation failed on another rank.");
    }
}
template<class T> auto read_view(const std::vector<T>& v)
{
    // The pointer constructor wraps without owning the vector allocation. Keep
    // default traits: Distributor also uses this view type for managed scratch.
    return Kokkos::View<const T*, Kokkos::HostSpace>(v.data(), v.size());
}
template<class T> auto write_view(std::vector<T>& v)
{
    return Kokkos::View<T*, Kokkos::HostSpace>(v.data(), v.size());
}
} // namespace
owner_exchange_detail::StreamAudit owner_exchange_detail::walk_owner_streams(std::span<const char> packed,
    std::span<const std::size_t> stream_sizes, std::span<const std::uint64_t> headers, void* context,
    void (*visitor)(void*, std::size_t, std::span<const char>))
{
    if (headers.size() != times(stream_sizes.size(), 2))
        throw std::invalid_argument("Owner stream header count differs from stream count.");
    StreamAudit audit;
    std::size_t offset = 0;
    for (std::size_t stream = 0; stream < stream_sizes.size(); ++stream)
    {
        const auto count = headers[2 * stream], declared_bytes = headers[2 * stream + 1];
        if (!count || declared_bytes != stream_sizes[stream] || declared_bytes < sizeof(Word) ||
            count > declared_bytes / sizeof(Word))
            throw std::invalid_argument("Invalid owner stream size declaration.");
        const auto end = plus(offset, stream_sizes[stream]);
        if (end > packed.size())
            throw std::invalid_argument("Owner stream exceeds receive buffer.");
        for (Word record = 0; record < count; ++record)
        {
            if (end - offset < sizeof(Word))
                throw std::invalid_argument("Truncated owner packet length.");
            Word size = 0;
            std::memcpy(&size, packed.data() + offset, sizeof(size));
            offset += sizeof(size);
            if (size > end - offset)
                throw std::invalid_argument("Truncated owner packet payload.");
            audit.records = plus(audit.records, 1);
            audit.payload_bytes = plus(audit.payload_bytes, static_cast<std::size_t>(size));
            if (visitor)
                visitor(context, stream, packed.subspan(offset, static_cast<std::size_t>(size)));
            offset += static_cast<std::size_t>(size);
        }
        if (offset != end)
            throw std::invalid_argument("Trailing owner stream bytes.");
    }
    if (offset != packed.size())
        throw std::invalid_argument("Owner streams do not cover receive buffer.");
    return audit;
}
OwnerExchange exchange_owner_packets(
    std::span<const OwnerPacket> input, Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget)
{
    if (comm.is_null())
        throw std::invalid_argument("Owner exchange needs a communicator.");
    OwnerExchange output;
    std::size_t sent_wire = 0, group_storage = 0;
    collectively(*comm,
        [&]
        {
            if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error("Owner exchange packet count exceeds MPI/Teuchos range.");
            if (input.size() > budget / sizeof(std::size_t))
                throw std::length_error("Owner packet metadata exceeds byte budget.");
            for (const auto& p : input)
            {
                if (p.peer < 0 || p.peer >= comm->getSize())
                    throw std::invalid_argument("Owner packet peer is out of range.");
                sent_wire = plus(sent_wire, plus(sizeof(Word), p.bytes.size()));
                output.stats.sent_bytes = plus(output.stats.sent_bytes, p.bytes.size());
            }
            if (sent_wire > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error("Owner exchange packed exports exceed supported MPI count range.");
            constexpr auto record_metadata = 2 * sizeof(std::size_t) +
                                             sizeof(std::pair<const int, std::vector<std::size_t>>) +
                                             4 * sizeof(void*) + sizeof(int) + sizeof(std::size_t) + 2 * sizeof(Word);
            if (plus(sent_wire, times(input.size(), record_metadata)) > budget)
                throw std::length_error("Owner packet exports/framing exceed byte budget.");
        });
    // One Distributor item per actual peer bounds planning by peer count, not by
    // potentially enormous incoming record counts. Record framing stays in bytes.
    std::map<int, std::vector<std::size_t>> groups;
    std::vector<int> peers;
    std::vector<std::size_t> send_sizes;
    std::vector<Word> send_header;
    collectively(*comm,
        [&]
        {
            for (std::size_t i = 0; i < input.size(); ++i)
                groups[input[i].peer].push_back(i);
            peers.reserve(groups.size());
            send_sizes.reserve(groups.size());
            send_header.reserve(times(2, groups.size()));
            for (const auto& [peer, records] : groups)
            {
                peers.push_back(peer);
                std::size_t bytes = 0;
                for (auto i : records)
                    bytes = plus(bytes, plus(sizeof(Word), input[i].bytes.size()));
                send_sizes.push_back(bytes);
                send_header.push_back(records.size());
                send_header.push_back(bytes);
                group_storage = plus(group_storage, plus(times(records.capacity(), sizeof(std::size_t)),
                                                        sizeof(decltype(groups)::value_type) + 4 * sizeof(void*)));
            }
            const auto metadata = plus(group_storage,
                plus(times(peers.capacity(), sizeof(int)), plus(times(send_sizes.capacity(), sizeof(std::size_t)),
                                                               times(send_header.capacity(), sizeof(Word)))));
            if (plus(sent_wire, metadata) > budget)
                throw std::length_error("Owner packet export framing exceeds byte budget.");
        });
    Tpetra::Distributor distributor(comm);
    const auto imports = distributor.createFromSends(Teuchos::arrayViewFromVector(peers));
    std::vector<Word> recv_header;
    std::vector<int> senders;
    collectively(*comm,
        [&]
        {
            if (imports > static_cast<std::size_t>(comm->getSize()))
                throw std::invalid_argument("Owner exchange received duplicate peer streams.");
            const auto existing = plus(group_storage,
                plus(times(peers.capacity(), sizeof(int)), plus(times(send_sizes.capacity(), sizeof(std::size_t)),
                                                               times(send_header.capacity(), sizeof(Word)))));
            const auto receive_metadata = times(imports, 2 * sizeof(Word) + sizeof(int) + sizeof(std::size_t));
            if (plus(existing, receive_metadata) > budget)
                throw std::length_error("Owner packet receive metadata exceeds byte budget.");
            recv_header.resize(times(imports, 2));
            senders.reserve(imports);
            const auto from = distributor.getProcsFrom();
            const auto lengths = distributor.getLengthsFrom();
            for (std::size_t i = 0; i < static_cast<std::size_t>(from.size()); ++i)
            {
                if (lengths[i] != 1)
                    throw std::invalid_argument("Owner exchange expected one stream per sender.");
                senders.push_back(from[i]);
            }
            if (senders.size() != imports)
                throw std::invalid_argument("Owner exchange sender routing differs from Distributor imports.");
        });
    distributor.doPostsAndWaits(read_view(send_header), std::size_t(2), write_view(recv_header));
    std::vector<std::size_t> recv_sizes;
    std::size_t received_wire = 0, records = 0, metadata = 0;
    std::size_t retired_send_metadata = 0, planned_decode_metadata = 0, framed_record_bytes = 0;
    std::size_t planned_decoded_payload = 0;
    collectively(*comm,
        [&]
        {
            recv_sizes.reserve(imports);
            for (std::size_t i = 0; i < imports; ++i)
            {
                const auto n = recv_header[2 * i], bytes = recv_header[2 * i + 1];
                if (!n || bytes < sizeof(Word) || n > bytes / sizeof(Word) ||
                    bytes > std::numeric_limits<std::size_t>::max() || n > std::numeric_limits<std::size_t>::max())
                    throw std::invalid_argument("Invalid owner stream size declaration.");
                received_wire = plus(received_wire, static_cast<std::size_t>(bytes));
                records = plus(records, static_cast<std::size_t>(n));
                recv_sizes.push_back(static_cast<std::size_t>(bytes));
            }
            if (received_wire > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                records > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error("Owner exchange packed imports exceed supported MPI count range.");
            metadata = group_storage;
            metadata = plus(metadata, times(peers.capacity() + senders.capacity(), sizeof(int)));
            metadata = plus(metadata, times(send_sizes.capacity() + recv_sizes.capacity(), sizeof(std::size_t)));
            metadata = plus(metadata, times(send_header.capacity() + recv_header.capacity(), sizeof(Word)));
            metadata = plus(metadata, times(records, sizeof(OwnerPacket)));
            retired_send_metadata = plus(group_storage,
                plus(times(peers.capacity(), sizeof(int)), plus(times(send_sizes.capacity(), sizeof(std::size_t)),
                                                               times(send_header.capacity(), sizeof(Word)))));
            planned_decode_metadata = metadata - retired_send_metadata;
            framed_record_bytes = times(records, sizeof(Word));
            planned_decoded_payload = received_wire - framed_record_bytes;
            // Transfer and decode have distinct lifetimes: packed sends are
            // destroyed after the completed exchange, before output allocation.
            // Keep output record metadata charged in both phases conservatively.
            const auto transfer_bound = plus(metadata, plus(sent_wire, received_wire));
            const auto decode_bound = plus(planned_decode_metadata, plus(received_wire, planned_decoded_payload));
            if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            {
                std::size_t input_size = times(input.size(), sizeof(OwnerPacket)), input_capacity = input_size;
                for (const auto& packet : input)
                {
                    input_size = plus(input_size, packet.bytes.size());
                    input_capacity = plus(input_capacity, packet.bytes.capacity());
                }
                std::fprintf(stderr,
                    "ADAPTIVE_EXCHANGE rank=%d packets=%zu input_range_size=%zu input_range_capacity=%zu "
                    "borrowed_input=1 scratch_excludes_input=1 "
                    "planned_send_size=%zu planned_send_capacity=%zu planned_receive_size=%zu "
                    "planned_receive_capacity=%zu planned_decoded_size=%zu metadata_capacity=%zu "
                    "decode_metadata_capacity=%zu retired_send_metadata_capacity=%zu frame_bytes=%zu "
                    "transfer_bound=%zu decode_bound=%zu all_buffers_bound=%zu scratch_bound=%zu limit=%zu\n",
                    comm->getRank(), input.size(), input_size, input_capacity, sent_wire, sent_wire, received_wire,
                    received_wire, planned_decoded_payload, metadata, planned_decode_metadata, retired_send_metadata,
                    framed_record_bytes, transfer_bound, decode_bound,
                    plus(metadata, plus(sent_wire, times(received_wire, 2))), std::max(transfer_bound, decode_bound),
                    budget);
            }
            if (std::max(transfer_bound, decode_bound) > budget)
                throw std::length_error("Owner packet receive buffers exceed byte budget.");
        });
    std::vector<char> received;
    std::size_t send_capacity = 0, transfer_capacity = 0, decoded_payload_capacity = 0, decode_capacity = 0;
    {
        std::vector<char> send;
        collectively(*comm,
            [&]
            {
                send.reserve(sent_wire);
                received.resize(received_wire);
                send_capacity = send.capacity();
                transfer_capacity = plus(metadata, plus(send_capacity, received.capacity()));
                if (transfer_capacity > budget ||
                    plus(planned_decode_metadata, plus(received.capacity(), planned_decoded_payload)) > budget)
                    throw std::length_error("Owner packet buffer capacities exceed byte budget.");
                for (const auto& [peer, ids] : groups)
                    for (auto i : ids)
                    {
                        const Word count = input[i].bytes.size();
                        const auto* bytes = reinterpret_cast<const char*>(&count);
                        send.insert(send.end(), bytes, bytes + sizeof(count));
                        send.insert(send.end(), input[i].bytes.begin(), input[i].bytes.end());
                    }
            });
        distributor.doPostsAndWaits(read_view(send), Teuchos::arrayViewFromVector(send_sizes), write_view(received),
            Teuchos::arrayViewFromVector(recv_sizes));
    } // Release packed sends before allocating any decoded packet storage.
    // These allocations were needed only to build/export the completed stream.
    // Destroy the real owners before using the smaller decode-phase charge.
    decltype(groups)().swap(groups);
    std::vector<int>().swap(peers);
    std::vector<std::size_t>().swap(send_sizes);
    std::vector<Word>().swap(send_header);
    owner_exchange_detail::StreamAudit validated;
    collectively(*comm,
        [&]
        {
            // Declared counts bound exact decoded payload only after every
            // inner frame has passed. A malicious early record cannot allocate
            // against that declaration and hide a truncated later frame.
            validated = owner_exchange_detail::walk_owner_streams(received, recv_sizes, recv_header);
            if (validated.records != records || validated.payload_bytes != planned_decoded_payload)
                throw std::invalid_argument("Owner stream audit differs from declared complete payload.");
        });
    std::size_t actual_decode_metadata = 0;
    collectively(*comm,
        [&]
        {
            output.packets.reserve(records);
            actual_decode_metadata = plus(planned_decode_metadata - times(records, sizeof(OwnerPacket)),
                times(output.packets.capacity(), sizeof(OwnerPacket)));
            if (plus(actual_decode_metadata, plus(received.capacity(), validated.payload_bytes)) > budget)
                throw std::length_error("Owner packet decode capacities exceed byte budget.");
            struct DecodeState
            {
                OwnerExchange& output;
                const std::vector<int>& senders;
                std::size_t metadata, packed_capacity, budget;
                std::size_t payload_capacity = 0, capacity = 0;
            };
            DecodeState state{output, senders, actual_decode_metadata, received.capacity(), budget};
            const auto decoded = owner_exchange_detail::walk_owner_streams(received, recv_sizes, recv_header, &state,
                [](void* context, std::size_t stream, std::span<const char> payload)
                {
                    auto& state = *static_cast<DecodeState*>(context);
                    state.output.packets.push_back(
                        {state.senders[stream], std::vector<char>(payload.begin(), payload.end())});
                    state.payload_capacity = plus(state.payload_capacity, state.output.packets.back().bytes.capacity());
                    state.capacity = plus(state.metadata, plus(state.packed_capacity, state.payload_capacity));
                    if (state.capacity > state.budget)
                        throw std::length_error("Owner packet decoded payload capacities exceed byte budget.");
                    state.output.stats.received_bytes = plus(state.output.stats.received_bytes, payload.size());
                });
            if (decoded.records != validated.records || decoded.payload_bytes != validated.payload_bytes)
                throw std::invalid_argument("Owner decoding differs from complete frame audit.");
            decoded_payload_capacity = state.payload_capacity;
            decode_capacity = plus(actual_decode_metadata, plus(received.capacity(), decoded_payload_capacity));
        });
    output.stats.sent_packets = input.size();
    output.stats.received_packets = records;
    output.stats.peak_scratch_bytes = std::max(transfer_capacity, decode_capacity);
    if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
    {
        std::size_t decoded_capacity = times(output.packets.capacity(), sizeof(OwnerPacket));
        for (const auto& packet : output.packets)
            decoded_capacity = plus(decoded_capacity, packet.bytes.capacity());
        std::fprintf(stderr,
            "ADAPTIVE_EXCHANGE_DONE rank=%d send_size=%zu send_capacity=%zu receive_size=%zu "
            "receive_capacity=%zu decoded_size=%zu decoded_capacity=%zu metadata_capacity=%zu "
            "decode_metadata_capacity=%zu retired_send_metadata_capacity=%zu validated_payload_bytes=%zu "
            "decoded_payload_capacity=%zu "
            "transfer_capacity=%zu decode_capacity=%zu send_capacity_at_decode=0 scratch_capacity=%zu limit=%zu\n",
            comm->getRank(), sent_wire, send_capacity, received.size(), received.capacity(),
            plus(times(output.packets.size(), sizeof(OwnerPacket)), output.stats.received_bytes), decoded_capacity,
            metadata, actual_decode_metadata, retired_send_metadata, validated.payload_bytes, decoded_payload_capacity,
            transfer_capacity, decode_capacity, output.stats.peak_scratch_bytes, budget);
    }
    return output;
}
OwnerExchange consume_owner_packets(
    std::vector<OwnerPacket>&& input, Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget)
{
    if (comm.is_null())
        throw std::invalid_argument("Owner exchange needs a communicator.");
    int remote = 0, any_remote = 0;
    std::size_t bytes = 0, capacity = 0;
    collectively(*comm,
        [&]
        {
            if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error("Owner exchange packet count exceeds MPI/Teuchos range.");
            capacity = times(input.capacity(), sizeof(OwnerPacket));
            for (const auto& packet : input)
            {
                if (packet.peer < 0 || packet.peer >= comm->getSize())
                    throw std::invalid_argument("Owner packet peer is out of range.");
                remote |= packet.peer != comm->getRank();
                bytes = plus(bytes, packet.bytes.size());
                capacity = plus(capacity, packet.bytes.capacity());
            }
            if (plus(bytes, times(input.size(), sizeof(Word))) >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error("Owner exchange packed exports exceed supported MPI count range.");
        });
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &remote, &any_remote);
    if (any_remote)
    {
        if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            std::fprintf(stderr,
                "ADAPTIVE_EXCHANGE_INPUT rank=%d consumed_input_size=%zu consumed_input_capacity=%zu "
                "scratch_excludes_input=1\n",
                comm->getRank(), plus(times(input.size(), sizeof(OwnerPacket)), bytes), capacity);
        auto result = exchange_owner_packets(std::span<const OwnerPacket>(input), comm, budget);
        std::vector<OwnerPacket>().swap(input);
        return result;
    }
    collectively(*comm,
        [&]
        {
            if (capacity > budget)
                throw std::length_error("Local owner packet capacities exceed byte budget.");
        });
    OwnerExchange result;
    result.stats.sent_packets = result.stats.received_packets = input.size();
    result.stats.sent_bytes = result.stats.received_bytes = bytes;
    result.stats.peak_scratch_bytes = capacity;
    if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
        std::fprintf(stderr,
            "ADAPTIVE_EXCHANGE rank=%d local_delivery=1 packets=%zu input_size=%zu "
            "input_capacity=%zu send_size=0 send_capacity=0 receive_size=0 receive_capacity=0 "
            "decoded_size=%zu decoded_capacity=%zu scratch_capacity=%zu limit=%zu\n",
            comm->getRank(), input.size(), input.size() * sizeof(OwnerPacket) + bytes, capacity,
            input.size() * sizeof(OwnerPacket) + bytes, capacity, capacity, budget);
    result.packets = std::move(input);
    return result;
}
} // namespace SimpleFluid
