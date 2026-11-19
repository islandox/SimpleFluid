/** @file SparseOwnerExchange.cc
 * @brief Two-phase, peer-coalesced variable packet exchange with collective gates.
 */
#include "parallel/SparseOwnerExchange.hh"
#include "parallel/NativePacketAdmission.hh"
#include "geometry/details/AllocationDomainScope.hh"
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include "parallel/SparseOwnerExchangeFraming.hh"
#include <Kokkos_Core.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Distributor.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
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
using StreamBlocks = owner_exchange_detail::OwnerStreamBlocks;
// A cursor borrows block owners during audit and retires their real allocations
// during decode. A crossing record is allocated whole before any payload copy;
// its admission therefore includes the partially consumed source block.
class BlockCursor
{
public:
    BlockCursor(const StreamBlocks& stream, StreamBlocks* destructive, void* retirement_context = nullptr,
        void (*after_deallocation)(void*, const char*, std::size_t) noexcept = nullptr)
        : blocks_(stream.blocks), destructive_(destructive), retirement_context_(retirement_context),
          after_deallocation_(after_deallocation)
    {
        for (const auto& block : blocks_)
            remaining_ = plus(remaining_, block.size());
    }
    std::size_t remaining() const { return remaining_; }
    std::size_t released() const { return released_; }
    void finish()
    {
        if (remaining_)
            throw std::invalid_argument("Trailing segmented owner stream bytes.");
        // Explicit seam callers may retain capacity in empty trailing owners.
        // They are real live allocations until the completed stream retires.
        while (block_ < blocks_.size())
            retire();
    }
    void read(char* destination, std::size_t count)
    {
        if (count > remaining_)
            throw std::invalid_argument("Truncated segmented owner stream.");
        while (count)
        {
            while (block_ < blocks_.size() && offset_ == blocks_[block_].size())
                retire();
            const auto n = std::min(count, blocks_[block_].size() - offset_);
            if (destination)
            {
                std::memcpy(destination, blocks_[block_].data() + offset_, n);
                destination += n;
            }
            offset_ += n;
            count -= n;
            remaining_ -= n;
            if (offset_ == blocks_[block_].size())
                retire();
        }
    }

private:
    void retire()
    {
        released_ = plus(released_, blocks_[block_].capacity());
        if (destructive_)
        {
            const auto* address = blocks_[block_].data();
            const auto capacity = blocks_[block_].capacity();
            std::vector<char>().swap(destructive_->blocks[block_]);
            if (after_deallocation_ && capacity)
                after_deallocation_(retirement_context_, address, capacity);
        }
        ++block_;
        offset_ = 0;
    }
    const std::vector<std::vector<char>>& blocks_;
    StreamBlocks* destructive_;
    void* retirement_context_;
    void (*after_deallocation_)(void*, const char*, std::size_t) noexcept;
    std::size_t block_ = 0, offset_ = 0, remaining_ = 0, released_ = 0;
};
template<class Visitor>
owner_exchange_detail::StreamAudit walk_blocks(std::span<const StreamBlocks> streams,
    std::span<const std::size_t> sizes, std::span<const Word> headers, Visitor&& visit,
    std::span<StreamBlocks> destructive = {}, void* retirement_context = nullptr,
    void (*after_deallocation)(void*, const char*, std::size_t) noexcept = nullptr)
{
    if (streams.size() != sizes.size() || headers.size() != times(sizes.size(), 2) ||
        (!destructive.empty() && destructive.size() != streams.size()))
        throw std::invalid_argument("Owner block/header count differs from stream count.");
    owner_exchange_detail::StreamAudit audit;
    std::size_t retired = 0;
    for (std::size_t stream = 0; stream < streams.size(); ++stream)
    {
        BlockCursor cursor(streams[stream], destructive.empty() ? nullptr : &destructive[stream], retirement_context,
            after_deallocation);
        const auto count = headers[2 * stream], bytes = headers[2 * stream + 1];
        if (!count || bytes < sizeof(Word) || count > bytes / sizeof(Word) || bytes != sizes[stream] ||
            cursor.remaining() != sizes[stream])
            throw std::invalid_argument("Invalid segmented owner stream declaration.");
        for (Word record = 0; record < count; ++record)
        {
            Word length = 0;
            cursor.read(reinterpret_cast<char*>(&length), sizeof(length));
            if (length > cursor.remaining())
                throw std::invalid_argument("Truncated segmented owner packet payload.");
            const auto size = static_cast<std::size_t>(length);
            audit.records = plus(audit.records, 1);
            audit.payload_bytes = plus(audit.payload_bytes, size);
            audit.largest_payload = std::max(audit.largest_payload, size);
            const auto before = cursor.remaining();
            visit(stream, size, cursor, retired);
            if (cursor.remaining() != before - size)
                throw std::invalid_argument("Segmented owner visitor consumed a different payload size.");
        }
        if (cursor.remaining())
            throw std::invalid_argument("Trailing segmented owner stream bytes.");
        cursor.finish();
        retired = plus(retired, cursor.released());
    }
    return audit;
}
} // namespace
owner_exchange_detail::StreamAudit owner_exchange_detail::walk_owner_block_streams(
    std::span<const OwnerStreamBlocks> streams, std::span<const std::size_t> sizes, std::span<const Word> headers)
{
    return walk_blocks(streams, sizes, headers,
        [](std::size_t, std::size_t size, BlockCursor& cursor, std::size_t) { cursor.read(nullptr, size); });
}
OwnerExchange owner_exchange_detail::decode_owner_block_streams(std::span<OwnerStreamBlocks> streams,
    std::span<const std::size_t> sizes, std::span<const Word> headers, std::span<const int> senders,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget, std::size_t retained_metadata,
    void* observer_context, void (*before_payload_allocation)(void*, std::size_t),
    void (*after_block_deallocation)(void*, const char*, std::size_t) noexcept)
{
    if (comm.is_null())
        throw std::invalid_argument("Owner block decode needs a communicator.");
    OwnerExchange output;
    StreamAudit validated;
    std::size_t raw_capacity = 0;
    collectively(*comm,
        [&]
        {
            if (senders.size() != streams.size())
                throw std::invalid_argument("Owner block sender count differs from stream count.");
            // Audit every frame before considering allocation or resource
            // admission. Invalid tails retain priority over remote cap errors.
            validated = walk_owner_block_streams(streams, sizes, headers);
            for (const auto& stream : streams)
                for (const auto& block : stream.blocks)
                    raw_capacity = plus(raw_capacity, block.capacity());
        });
    std::size_t metadata = 0, planned_peak = 0, payload_capacity = 0;
    collectively(*comm,
        [&]
        {
            metadata = plus(retained_metadata, times(validated.records, sizeof(OwnerPacket)));
            planned_peak = plus(metadata, raw_capacity);
            (void) walk_blocks(streams, sizes, headers,
                [&](std::size_t, std::size_t size, BlockCursor& cursor, std::size_t retired)
                {
                    const auto remaining_raw = raw_capacity - retired - cursor.released();
                    planned_peak =
                        std::max(planned_peak, plus(metadata, plus(remaining_raw, plus(payload_capacity, size))));
                    payload_capacity = plus(payload_capacity, size);
                    cursor.read(nullptr, size);
                });
            if (planned_peak > budget)
                throw std::length_error("Owner packet destructive decode exceeds byte budget.");
        });
    std::size_t decode_peak = 0;
    payload_capacity = 0;
    collectively(*comm,
        [&]
        {
            output.packets.reserve(validated.records);
            const auto actual_metadata = plus(retained_metadata, times(output.packets.capacity(), sizeof(OwnerPacket)));
            if (plus(planned_peak - metadata, actual_metadata) > budget)
                throw std::length_error("Owner packet output capacities exceed byte budget.");
            decode_peak = plus(actual_metadata, raw_capacity);
            const auto decoded = walk_blocks(
                streams, sizes, headers,
                [&](std::size_t stream, std::size_t size, BlockCursor& cursor, std::size_t retired)
                {
                    const auto remaining_raw = raw_capacity - retired - cursor.released();
                    if (plus(actual_metadata, plus(remaining_raw, plus(payload_capacity, size))) > budget)
                        throw std::length_error("Owner packet next payload exceeds byte budget.");
                    if (size && before_payload_allocation)
                        before_payload_allocation(observer_context, size);
                    output.packets.push_back({senders[stream], std::vector<char>(size)});
                    auto& packet = output.packets.back();
                    payload_capacity = plus(payload_capacity, packet.bytes.capacity());
                    const auto live = plus(actual_metadata, plus(remaining_raw, payload_capacity));
                    if (live > budget)
                        throw std::length_error("Owner packet decoded capacities exceed byte budget.");
                    decode_peak = std::max(decode_peak, live);
                    cursor.read(packet.bytes.data(), size);
                    output.stats.received_bytes = plus(output.stats.received_bytes, size);
                },
                streams, observer_context, after_block_deallocation);
            if (decoded.records != validated.records || decoded.payload_bytes != validated.payload_bytes)
                throw std::invalid_argument("Owner block decode differs from complete audit.");
        });
    output.stats.received_packets = validated.records;
    output.stats.decode_peak_bytes = decode_peak;
    output.stats.peak_scratch_bytes = decode_peak;
    output.stats.largest_received_record = validated.largest_payload;
    output.stats.receive_block_capacity = raw_capacity;
    return output;
}
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
            audit.largest_payload = std::max(audit.largest_payload, static_cast<std::size_t>(size));
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
OwnerExchange owner_exchange_detail::exchange_owner_packets_with_round_limit(std::span<const OwnerPacket> input,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget, std::size_t round_limit)
{
    if (comm.is_null())
        throw std::invalid_argument("Owner exchange needs a communicator.");
    Word local_limit = round_limit, minimum_limit = 0, maximum_limit = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_limit, &minimum_limit);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_limit, &maximum_limit);
    if (!minimum_limit || minimum_limit != maximum_limit)
        throw std::invalid_argument("Owner exchange round limits must be nonzero and identical on every rank.");
    OwnerExchange output;
    std::size_t sent_wire = 0, group_storage = 0, planning_peak = 0;
    Word largest_input_payload = 0, global_payload_bound = 0;
    collectively(*comm,
        [&]
        {
            if (input.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error("Owner exchange packet count exceeds MPI/Teuchos range.");
            for (const auto& p : input)
                if (p.peer < 0 || p.peer >= comm->getSize())
                    throw std::invalid_argument("Owner packet peer is out of range.");
            if (input.size() > budget / sizeof(std::size_t))
                throw std::length_error("Owner packet metadata exceeds byte budget.");
            for (const auto& p : input)
            {
                sent_wire = plus(sent_wire, plus(sizeof(Word), p.bytes.size()));
                output.stats.sent_bytes = plus(output.stats.sent_bytes, p.bytes.size());
                largest_input_payload = std::max(largest_input_payload, static_cast<Word>(p.bytes.size()));
            }
            if (sent_wire > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::length_error("Owner exchange packed exports exceed supported MPI count range.");
            constexpr auto record_metadata = 2 * sizeof(std::size_t) +
                                             sizeof(std::pair<const int, std::vector<std::size_t>>) +
                                             4 * sizeof(void*) + sizeof(int) + sizeof(std::size_t) + 2 * sizeof(Word);
            if (times(input.size(), record_metadata) > budget)
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
            std::size_t current_group_storage = 0;
            for (std::size_t i = 0; i < input.size(); ++i)
            {
                auto [group, inserted] = groups.try_emplace(input[i].peer);
                if (inserted)
                    current_group_storage =
                        plus(current_group_storage, sizeof(decltype(groups)::value_type) + 4 * sizeof(void*));
                auto& indices = group->second;
                const auto old_capacity = indices.capacity();
                indices.push_back(i);
                if (indices.capacity() != old_capacity)
                {
                    // At reallocation both old and new index owners coexist.
                    // current_group_storage still includes the old owner here.
                    planning_peak = std::max(
                        planning_peak, plus(current_group_storage, times(indices.capacity(), sizeof(std::size_t))));
                    current_group_storage =
                        plus(current_group_storage, times(indices.capacity() - old_capacity, sizeof(std::size_t)));
                }
                planning_peak = std::max(planning_peak, current_group_storage);
            }
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
            planning_peak = std::max(planning_peak, metadata);
            if (metadata > budget)
                throw std::length_error("Owner packet export framing exceeds byte budget.");
            if (planning_peak > budget)
                throw std::length_error("Owner packet planning capacities exceed byte budget.");
        });
    std::unique_ptr<Tpetra::Distributor> distributor_owner;
    collectively(*comm, [&]
    {
        const geometry_detail::AllocationDomainScope domain(geometry_detail::AllocationDomain::ExternalThirdParty);
        distributor_owner = std::make_unique<Tpetra::Distributor>(comm);
    });
    auto& distributor = *distributor_owner;
    const auto imports = [&]
    {
        const geometry_detail::AllocationDomainScope domain(geometry_detail::AllocationDomain::ExternalThirdParty);
        return distributor.createFromSends(Teuchos::arrayViewFromVector(peers));
    }();
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
    {
        const auto export_view = read_view(send_header);
        const auto import_view = write_view(recv_header);
        const geometry_detail::AllocationDomainScope domain(geometry_detail::AllocationDomain::ExternalThirdParty);
        distributor.doPostsAndWaits(export_view, std::size_t(2), import_view);
    }
    std::vector<std::size_t> recv_sizes;
    std::size_t received_wire = 0, records = 0, metadata = 0;
    std::size_t retired_send_metadata = 0, planned_decode_metadata = 0, framed_record_bytes = 0;
    std::size_t planned_decoded_payload = 0;
    std::size_t transfer_bound = 0, decode_bound = 0;
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
            planning_peak = std::max(planning_peak, metadata - times(records, sizeof(OwnerPacket)));
            retired_send_metadata = plus(group_storage,
                plus(times(peers.capacity(), sizeof(int)), plus(times(send_sizes.capacity(), sizeof(std::size_t)),
                                                               times(send_header.capacity(), sizeof(Word)))));
            planned_decode_metadata = metadata - retired_send_metadata;
            framed_record_bytes = times(records, sizeof(Word));
            planned_decoded_payload = received_wire - framed_record_bytes;
            // Transfer and decode have distinct lifetimes: packed sends are
            // destroyed after the completed exchange, before output allocation.
            // Keep output record metadata charged in both phases conservatively.
            transfer_bound = plus(metadata, plus(sent_wire, received_wire));
            decode_bound = plus(planned_decode_metadata, plus(received_wire, planned_decoded_payload));
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
        });
    // This scalar conservatively bounds every incoming payload without
    // changing either the two-word peer headers or the framed packet bytes.
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &largest_input_payload, &global_payload_bound);
    int old_admitted = std::max(transfer_bound, decode_bound) <= budget, all_old_admitted = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &old_admitted, &all_old_admitted);
    const int force_blocks = round_limit != 16 * 1024 * 1024;
    int want_blocks = force_blocks || !all_old_admitted || std::max(sent_wire, received_wire) > 1024 * 1024;
    int any_want_blocks = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &want_blocks, &any_want_blocks);
    std::size_t quantum = round_limit, block_metadata = 0, block_transfer_base = 0, max_stream_send = 0;
    std::size_t block_decode_base = 0;
    std::size_t planned_block_transfer = 0, planned_block_decode = 0, decode_payload_headroom = 0;
    for (const auto size : send_sizes)
        max_stream_send = std::max(max_stream_send, size);
    bool use_blocks = false;
    if (any_want_blocks)
    {
        // A smaller quantum lowers staging but increases block descriptors.
        // Check both costs on every rank at every candidate, including empty
        // owners and owners with distinct budgets. No candidate allocates.
        while (quantum)
        {
            int fits = 0, all_fit = 0, compatible = 0, all_compatible = 0;
            try
            {
                block_metadata = times(imports, sizeof(StreamBlocks) + sizeof(std::size_t));
                for (const auto size : recv_sizes)
                    block_metadata =
                        plus(block_metadata, times(size / quantum + (size % quantum != 0), sizeof(std::vector<char>)));
                block_transfer_base = plus(metadata - times(records, sizeof(OwnerPacket)), block_metadata);
                block_decode_base = plus(planned_decode_metadata, block_metadata - times(imports, sizeof(std::size_t)));
                planned_block_transfer =
                    plus(block_transfer_base, plus(received_wire, std::min(quantum, max_stream_send)));
                // Completed source blocks are retired in stream order. Prior
                // decoded payload plus remaining raw bytes exceed the original
                // receive size by at most one partially consumed block. The
                // next packet is allocated whole before copying, even when it
                // crosses several blocks. Frame bytes only reduce this bound.
                // The contiguous R+D bound remains a useful smaller bound for
                // oversized records and owners with little decoded payload.
                decode_payload_headroom = std::min(planned_decoded_payload,
                    plus(std::min(quantum, received_wire),
                        std::min(planned_decoded_payload, static_cast<std::size_t>(global_payload_bound))));
                planned_block_decode = plus(block_decode_base, plus(received_wire, decode_payload_headroom));
                const auto candidate_peak = std::max(planned_block_transfer, planned_block_decode);
                fits = candidate_peak <= budget;
                compatible = candidate_peak <= budget && candidate_peak <= std::max(transfer_bound, decode_bound);
            }
            catch (const std::length_error&)
            {
                // This candidate's descriptor bound overflows; it cannot fit.
            }
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &fits, &all_fit);
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &compatible, &all_compatible);
            if (all_fit)
            {
                // Preserve previously admitted tight contiguous budgets. A
                // forced private test seam deliberately bypasses that choice.
                use_blocks = force_blocks || !all_old_admitted || all_compatible;
                break;
            }
            quantum /= 2;
        }
    }
    if (use_blocks)
    {
        std::vector<StreamBlocks> streams;
        std::vector<std::size_t> round_receive_sizes;
        std::size_t raw_capacity = 0, transfer_peak = 0, actual_block_metadata = 0;
        collectively(*comm,
            [&]
            {
                streams.resize(imports);
                round_receive_sizes.resize(imports);
                for (std::size_t i = 0; i < imports; ++i)
                    streams[i].blocks.reserve(recv_sizes[i] / quantum + (recv_sizes[i] % quantum != 0));
                actual_block_metadata = plus(times(streams.capacity(), sizeof(StreamBlocks)),
                    times(round_receive_sizes.capacity(), sizeof(std::size_t)));
                for (const auto& stream : streams)
                    actual_block_metadata =
                        plus(actual_block_metadata, times(stream.blocks.capacity(), sizeof(std::vector<char>)));
                block_transfer_base = plus(metadata - times(records, sizeof(OwnerPacket)), actual_block_metadata);
                block_decode_base = plus(planned_decode_metadata,
                    actual_block_metadata - times(round_receive_sizes.capacity(), sizeof(std::size_t)));
                planned_block_transfer =
                    plus(block_transfer_base, plus(received_wire, std::min(quantum, max_stream_send)));
                planned_block_decode = plus(block_decode_base, plus(received_wire, decode_payload_headroom));
                if (std::max(planned_block_transfer, planned_block_decode) > budget)
                    throw std::length_error("Owner packet block capacities exceed byte budget.");
            });
        if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            std::fprintf(stderr,
                "ADAPTIVE_EXCHANGE_BLOCK_PLAN rank=%d round_limit=%zu global_payload_bound=%llu "
                "block_metadata_capacity=%zu planned_block_transfer=%zu planned_block_decode=%zu limit=%zu\n",
                comm->getRank(), quantum, static_cast<unsigned long long>(global_payload_bound), actual_block_metadata,
                planned_block_transfer, planned_block_decode, budget);
        // Each shift is a permutation: there is at most one outgoing and one
        // incoming block per rank. Counts for all other original plan slots are
        // zero. The global extent keeps empty ranks in precisely the same rounds.
        for (int shift = 0; shift < comm->getSize(); ++shift)
        {
            const int destination = (comm->getRank() + shift) % comm->getSize();
            const int source = (comm->getRank() - shift + comm->getSize()) % comm->getSize();
            const auto send_it = std::find(peers.begin(), peers.end(), destination);
            const auto receive_it = std::find(senders.begin(), senders.end(), source);
            const auto si = static_cast<std::size_t>(send_it - peers.begin());
            const auto ri = static_cast<std::size_t>(receive_it - senders.begin());
            const std::size_t local_send = si == peers.size() ? 0 : send_header[2 * si + 1];
            const std::size_t local_receive = ri == senders.size() ? 0 : recv_sizes[ri];
            Word extent = local_send, global_extent = 0;
            Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &extent, &global_extent);
            const auto group = groups.find(destination);
            std::size_t record = 0, framed_offset = 0;
            for (std::size_t offset = 0; offset < global_extent; offset = plus(offset, quantum))
            {
                const auto send_count = offset < local_send ? std::min(quantum, local_send - offset) : 0;
                const auto receive_count = offset < local_receive ? std::min(quantum, local_receive - offset) : 0;
                std::vector<char> send, empty_receive;
                std::vector<char>* receive = &empty_receive;
                collectively(*comm,
                    [&]
                    {
                        std::fill(send_sizes.begin(), send_sizes.end(), 0);
                        std::fill(round_receive_sizes.begin(), round_receive_sizes.end(), 0);
                        // Admission precedes both allocations; no old/new staging
                        // owners overlap between rounds.
                        if (plus(block_transfer_base, plus(raw_capacity, plus(send_count, receive_count))) > budget)
                            throw std::length_error("Owner packet round staging exceeds byte budget.");
                        send.resize(send_count);
                        if (send_count)
                            send_sizes[si] = send_count;
                        if (receive_count)
                        {
                            streams[ri].blocks.emplace_back(receive_count);
                            receive = &streams[ri].blocks.back();
                            raw_capacity = plus(raw_capacity, receive->capacity());
                            round_receive_sizes[ri] = receive_count;
                        }
                        const auto live = plus(block_transfer_base, plus(raw_capacity, send.capacity()));
                        if (live > budget)
                            throw std::length_error("Owner packet round capacities exceed byte budget.");
                        transfer_peak = std::max(transfer_peak, live);
                        std::size_t filled = 0;
                        while (filled < send_count)
                        {
                            const auto& packet = input[group->second[record]];
                            const Word length = packet.bytes.size();
                            const auto* part = framed_offset < sizeof(Word)
                                                   ? reinterpret_cast<const char*>(&length) + framed_offset
                                                   : packet.bytes.data() + (framed_offset - sizeof(Word));
                            const auto available = framed_offset < sizeof(Word)
                                                       ? sizeof(Word) - framed_offset
                                                       : packet.bytes.size() - (framed_offset - sizeof(Word));
                            const auto count = std::min(available, send_count - filled);
                            std::memcpy(send.data() + filled, part, count);
                            filled += count;
                            framed_offset += count;
                            if (framed_offset == sizeof(Word) + packet.bytes.size())
                            {
                                ++record;
                                framed_offset = 0;
                            }
                        }
                    });
                {
                    const auto export_view = read_view(send);
                    const auto import_view = write_view(*receive);
                    const geometry_detail::AllocationDomainScope domain(
                        geometry_detail::AllocationDomain::ExternalThirdParty);
                    distributor.doPostsAndWaits(export_view, Teuchos::arrayViewFromVector(send_sizes),
                        import_view, Teuchos::arrayViewFromVector(round_receive_sizes));
                }
                ++output.stats.transport_rounds;
            }
        }
        decltype(groups)().swap(groups);
        std::vector<int>().swap(peers);
        std::vector<std::size_t>().swap(send_sizes);
        std::vector<Word>().swap(send_header);
        const auto retired_round_metadata = times(round_receive_sizes.capacity(), sizeof(std::size_t));
        std::vector<std::size_t>().swap(round_receive_sizes);
        actual_block_metadata -= retired_round_metadata;
        const auto retained_decode_metadata =
            plus(planned_decode_metadata - times(records, sizeof(OwnerPacket)), actual_block_metadata);
        auto decoded_output = owner_exchange_detail::decode_owner_block_streams(
            streams, recv_sizes, recv_header, senders, comm, budget, retained_decode_metadata);
        decoded_output.stats.sent_bytes = output.stats.sent_bytes;
        decoded_output.stats.transport_rounds = output.stats.transport_rounds;
        output = std::move(decoded_output);
        const auto decode_peak = output.stats.decode_peak_bytes;
        std::size_t payload_capacity = 0;
        for (const auto& packet : output.packets)
            payload_capacity = plus(payload_capacity, packet.bytes.capacity());
        output.stats.sent_packets = input.size();
        output.stats.received_packets = records;
        output.stats.transfer_peak_bytes = transfer_peak;
        output.stats.decode_peak_bytes = decode_peak;
        output.stats.planning_peak_bytes = planning_peak;
        output.stats.peak_scratch_bytes = std::max(planning_peak, std::max(transfer_peak, decode_peak));
        output.stats.round_byte_limit = quantum;
        output.stats.receive_block_capacity = raw_capacity;
        output.stats.block_metadata_capacity = actual_block_metadata;
        output.stats.segmented_receive = true;
        if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            std::fprintf(stderr,
                "ADAPTIVE_EXCHANGE_DONE rank=%d segmented_receive=1 rounds=%zu round_limit=%zu "
                "receive_capacity=%zu block_metadata_capacity=%zu largest_record=%zu "
                "decoded_payload_capacity=%zu planning_capacity=%zu transfer_capacity=%zu decode_capacity=%zu "
                "send_capacity_at_decode=0 scratch_capacity=%zu limit=%zu scratch_excludes_input=1\n",
                comm->getRank(), output.stats.transport_rounds, quantum, raw_capacity, actual_block_metadata,
                output.stats.largest_received_record, payload_capacity, planning_peak, transfer_peak, decode_peak,
                output.stats.peak_scratch_bytes, budget);
        return output;
    }
    collectively(*comm,
        [&]
        {
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
        {
            const auto export_view = read_view(send);
            const auto import_view = write_view(received);
            const geometry_detail::AllocationDomainScope domain(geometry_detail::AllocationDomain::ExternalThirdParty);
            distributor.doPostsAndWaits(export_view, Teuchos::arrayViewFromVector(send_sizes), import_view,
                Teuchos::arrayViewFromVector(recv_sizes));
        }
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
    output.stats.planning_peak_bytes = planning_peak;
    output.stats.peak_scratch_bytes = std::max(planning_peak, std::max(transfer_capacity, decode_capacity));
    output.stats.transfer_peak_bytes = transfer_capacity;
    output.stats.decode_peak_bytes = decode_capacity;
    output.stats.transport_rounds = sent_wire || received_wire ? 1 : 0;
    output.stats.largest_received_record = validated.largest_payload;
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
            "planning_capacity=%zu transfer_capacity=%zu decode_capacity=%zu send_capacity_at_decode=0 "
            "scratch_capacity=%zu limit=%zu\n",
            comm->getRank(), sent_wire, send_capacity, received.size(), received.capacity(),
            plus(times(output.packets.size(), sizeof(OwnerPacket)), output.stats.received_bytes), decoded_capacity,
            metadata, actual_decode_metadata, retired_send_metadata, validated.payload_bytes, decoded_payload_capacity,
            planning_peak, transfer_capacity, decode_capacity, output.stats.peak_scratch_bytes, budget);
    }
    return output;
}
OwnerExchange exchange_owner_packets(
    std::span<const OwnerPacket> input, Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget)
{
    return owner_exchange_detail::exchange_owner_packets_with_round_limit(input, comm, budget, 16 * 1024 * 1024);
}
namespace
{
class NativePacketComm
{
public:
    explicit NativePacketComm(const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
        : size(comm->getSize()), rank(comm->getRank())
    {
#ifdef HAVE_TEUCHOS_MPI
        if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.getRawPtr()))
        {
            raw = *mpi->getRawMpiComm();
            tag_ = const_cast<Teuchos::Comm<int>*>(comm.getRawPtr())->incrementTag();
            mpi_ = true;
            return;
        }
#endif
        if (size != 1 || rank || !dynamic_cast<const Teuchos::SerialComm<int>*>(comm.getRawPtr()))
            throw std::invalid_argument("Native packet adapter needs its actual MPI or serial communicator.");
    }
    template<class F> void gate(F&& action) const
    {
        std::exception_ptr error;
        try
        {
            action();
        }
        catch (...)
        {
            error = std::current_exception();
        }
        const int failed = bool(error);
        int any = failed;
#ifdef HAVE_TEUCHOS_MPI
        if (mpi_)
            check(MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, raw));
#endif
        if (any)
        {
            if (error)
                std::rethrow_exception(error);
            throw std::invalid_argument("Native owner packet allocation/validation failed on another rank.");
        }
    }
    std::uint64_t maximum(std::uint64_t value) const
    {
#ifdef HAVE_TEUCHOS_MPI
        if (mpi_)
        {
            std::uint64_t result = 0;
            check(MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, MPI_MAX, raw));
            return result;
        }
#endif
        return value;
    }
    void headers(const Word* send, Word* receive) const
    {
#ifdef HAVE_TEUCHOS_MPI
        if (mpi_)
        {
            check(MPI_Alltoall(send, 2, MPI_UINT64_T, receive, 2, MPI_UINT64_T, raw));
            return;
        }
#endif
        std::copy_n(send, 2, receive);
    }
    void transfer(const char* send, int send_count, int destination, char* receive, int receive_count, int source) const
    {
#ifdef HAVE_TEUCHOS_MPI
        if (mpi_)
        {
            check(MPI_Sendrecv(send, send_count, MPI_CHAR, destination, tag_, receive, receive_count, MPI_CHAR, source,
                tag_, raw, MPI_STATUS_IGNORE));
            return;
        }
#endif
        if (send_count != receive_count || destination || source)
            throw std::invalid_argument("Native serial packet counts differ.");
        if (send_count)
            std::memcpy(receive, send, static_cast<std::size_t>(send_count));
    }
    int size, rank;

private:
#ifdef HAVE_TEUCHOS_MPI
    MPI_Comm raw = MPI_COMM_NULL;
    bool mpi_ = false;
    int tag_ = 0;
    static void check(int status)
    {
        if (status != MPI_SUCCESS)
            throw std::invalid_argument("Native owner packet MPI transport failed.");
    }
#endif
};
struct NativePacketSendCursor
{
    std::size_t record = 0, offset = 0;
    void copy(std::span<const OwnerPacket> input, std::span<const std::size_t> indices, std::span<char> destination)
    {
        std::size_t written = 0;
        while (written < destination.size())
        {
            if (record == indices.size())
                throw std::invalid_argument("Native owner stream cursor exceeds records.");
            const auto& packet = input[indices[record]].bytes;
            const Word length = packet.size();
            const auto extent = plus(sizeof(Word), packet.size());
            const auto count = std::min(extent - offset, destination.size() - written);
            const auto prefix = offset < sizeof(Word) ? std::min(sizeof(Word) - offset, count) : 0;
            if (prefix)
                std::memcpy(destination.data() + written, reinterpret_cast<const char*>(&length) + offset, prefix);
            if (count > prefix)
                std::memcpy(destination.data() + written + prefix, packet.data() + offset + prefix - sizeof(Word),
                    count - prefix);
            offset += count;
            written += count;
            if (offset == extent)
            {
                ++record;
                offset = 0;
            }
        }
    }
};
} // namespace
NativeOwnerExchange exchange_owner_packets(std::span<const OwnerPacket> input,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, const geometry_detail::NativeConstructionAdmission& admission,
    std::size_t round_limit)
{
    if (comm.is_null())
        throw std::invalid_argument("Native owner exchange needs its communicator.");
    const NativePacketComm transport(comm);
    using geometry_detail::NativeConstructorVector;
    NativeOwnerExchange output;
    // Every phase owner precedes the concrete buffers it pays for.
    geometry_detail::NativeConstructorStorageOwner receive_storage;
    std::vector<StreamBlocks> streams;
    NativeConstructorVector<Word> send_header(admission.allocator<Word>()), recv_header(admission.allocator<Word>());
    NativeConstructorVector<std::size_t> sizes(admission.allocator<std::size_t>());
    NativeConstructorVector<int> senders(admission.allocator<int>());
    NativeConstructorVector<NativeConstructorVector<std::size_t>> groups(
        admission.allocator<NativeConstructorVector<std::size_t>>());
    NativeConstructorVector<NativePacketSendCursor> cursors(admission.allocator<NativePacketSendCursor>());
    std::size_t sent_wire = 0, received_wire = 0, imports = 0, records = 0;
    const auto observe = [&](unsigned phase, std::size_t send_chunk = 0)
    {
        const auto stored = [](const geometry_detail::NativeConstructorStorageOwner& storage)
        {
            return storage.owner ? plus(storage.owner->exclusive_bytes(), storage.envelope->fixed_allocation_bytes())
                                 : 0;
        };
        std::size_t bytes = plus(stored(output.storage()), stored(receive_storage));
        bytes = plus(bytes, times(send_header.capacity() + recv_header.capacity(), sizeof(Word)));
        bytes = plus(bytes, times(sizes.capacity(), sizeof(std::size_t)));
        bytes = plus(bytes, times(senders.capacity(), sizeof(int)));
        bytes = plus(bytes, times(groups.capacity(), sizeof(NativeConstructorVector<std::size_t>)));
        for (const auto& group : groups)
            bytes = plus(bytes, times(group.capacity(), sizeof(std::size_t)));
        bytes = plus(bytes, times(cursors.capacity(), sizeof(NativePacketSendCursor)));
        bytes = plus(bytes, send_chunk);
        auto& value = phase == 0   ? output.stats.planning_peak_bytes
                      : phase == 1 ? output.stats.transfer_peak_bytes
                                   : output.stats.decode_peak_bytes;
        value = std::max(value, bytes);
        output.stats.peak_scratch_bytes = std::max(output.stats.peak_scratch_bytes, bytes);
    };
    const auto quantum64 = transport.maximum(round_limit);
    transport.gate(
        [&]
        {
            geometry_detail::NativeConstructorStorage::require_profile();
            if (!round_limit || quantum64 != round_limit || round_limit > static_cast<std::size_t>(INT_MAX))
                throw std::invalid_argument("Native packet round limits must be identical, nonzero and represented.");
            if (input.size() > static_cast<std::size_t>(INT_MAX))
                throw std::length_error("Native packet count exceeds range.");
            output.storage() = geometry_detail::native_constructor_storage(admission);
            receive_storage = geometry_detail::native_constructor_storage(admission);
            send_header.resize(times(transport.size, 2), 0);
            recv_header.resize(times(transport.size, 2), 0);
            groups.reserve(transport.size);
            cursors.resize(transport.size);
            for (int peer = 0; peer < transport.size; ++peer)
                groups.emplace_back(admission.allocator<std::size_t>());
            for (const auto& packet : input)
            {
                if (packet.peer < 0 || packet.peer >= transport.size)
                    throw std::invalid_argument("Native packet peer is out of range.");
                auto& count = send_header[2 * packet.peer];
                auto& bytes = send_header[2 * packet.peer + 1];
                ++count;
                bytes = plus(bytes, plus(sizeof(Word), packet.bytes.size()));
                sent_wire = plus(sent_wire, plus(sizeof(Word), packet.bytes.size()));
                output.stats.sent_bytes = plus(output.stats.sent_bytes, packet.bytes.size());
            }
            if (sent_wire > static_cast<std::size_t>(INT_MAX))
                throw std::length_error("Native packed exports exceed range.");
            for (int peer = 0; peer < transport.size; ++peer)
                groups[peer].reserve(send_header[2 * peer]);
            for (std::size_t i = 0; i < input.size(); ++i)
                groups[input[i].peer].push_back(i);
            observe(0);
        });
    transport.headers(send_header.data(), recv_header.data());
    transport.gate(
        [&]
        {
            for (int peer = 0; peer < transport.size; ++peer)
            {
                const auto count = recv_header[2 * peer], bytes = recv_header[2 * peer + 1];
                if ((!count) != (!bytes) || (count && (bytes < sizeof(Word) || count > bytes / sizeof(Word))))
                    throw std::invalid_argument("Native peer stream declaration is invalid.");
                if (!count)
                    continue;
                received_wire = plus(received_wire, bytes);
                records = plus(records, count);
                ++imports;
            }
            if (received_wire > static_cast<std::size_t>(INT_MAX) || records > static_cast<std::size_t>(INT_MAX))
                throw std::length_error("Native packed imports exceed range.");
            receive_storage.owner->reserve(streams, transport.size);
            streams.resize(transport.size);
            sizes.resize(transport.size);
            senders.resize(transport.size);
            for (int peer = 0; peer < transport.size; ++peer)
            {
                sizes[peer] = recv_header[2 * peer + 1];
                senders[peer] = peer;
                const auto count = sizes[peer] / round_limit + bool(sizes[peer] % round_limit);
                receive_storage.owner->reserve(streams[peer].blocks, count);
            }
            output.stats.block_metadata_capacity = times(streams.capacity(), sizeof(StreamBlocks));
            for (const auto& stream : streams)
                output.stats.block_metadata_capacity = plus(
                    output.stats.block_metadata_capacity, times(stream.blocks.capacity(), sizeof(std::vector<char>)));
            observe(0);
        });
    // All ranks execute the same peer and round schedule, including empty owners.
    for (int step = 0; step < transport.size; ++step)
    {
        const int destination = (transport.rank + step) % transport.size;
        const int source = (transport.rank - step + transport.size) % transport.size;
        const auto sending = send_header[2 * destination + 1], receiving = recv_header[2 * source + 1];
        const auto rounds = transport.maximum(std::max(sending / round_limit + bool(sending % round_limit),
            receiving / round_limit + bool(receiving % round_limit)));
        for (std::size_t round = 0; round < rounds; ++round)
        {
            NativeConstructorVector<char> send(admission.allocator<char>());
            std::vector<char>* receive = nullptr;
            const auto offset = times(round, round_limit);
            const auto send_count = std::min<std::size_t>(round_limit, sending > offset ? sending - offset : 0);
            const auto receive_count = std::min<std::size_t>(round_limit, receiving > offset ? receiving - offset : 0);
            transport.gate(
                [&]
                {
                    send.resize(send_count);
                    cursors[destination].copy(input, groups[destination], send);
                    if (receive_count)
                    {
                        receive = &streams[source].blocks.emplace_back();
                        receive_storage.owner->reserve(*receive, receive_count);
                        receive->resize(receive_count);
                    }
                    observe(1, send.capacity());
                });
            transport.transfer(send.data(), static_cast<int>(send_count), destination,
                receive ? receive->data() : nullptr, static_cast<int>(receive_count), source);
            ++output.stats.transport_rounds;
        } // The actual send chunk dies here, before the next phase allocation.
    }
    // Export-only ownership retires before decoded payloads are allocated.
    groups.clear();
    NativeConstructorVector<NativeConstructorVector<std::size_t>>(
        admission.allocator<NativeConstructorVector<std::size_t>>())
        .swap(groups);
    NativeConstructorVector<NativePacketSendCursor>(admission.allocator<NativePacketSendCursor>()).swap(cursors);
    NativeConstructorVector<Word>(admission.allocator<Word>()).swap(send_header);
    // Compact empty peer streams without changing sender order or framing.
    std::size_t active = 0;
    for (int peer = 0; peer < transport.size; ++peer)
        if (recv_header[2 * peer])
        {
            if (active != static_cast<std::size_t>(peer))
                streams[active].blocks.swap(streams[peer].blocks);
            sizes[active] = recv_header[2 * peer + 1];
            senders[active] = peer;
            recv_header[2 * active] = recv_header[2 * peer];
            recv_header[2 * active + 1] = recv_header[2 * peer + 1];
            ++active;
        }
    owner_exchange_detail::StreamAudit audit;
    transport.gate(
        [&]
        {
            audit = owner_exchange_detail::walk_owner_block_streams(std::span(streams).first(active),
                std::span(sizes).first(active), std::span(recv_header).first(2 * active));
            if (audit.records != records)
                throw std::invalid_argument("Native complete stream record count differs.");
            output.storage().owner->reserve(output.packets, records);
            observe(2);
        });
    transport.gate(
        [&]
        {
            (void) walk_blocks(
                std::span(streams).first(active), std::span(sizes).first(active),
                std::span(recv_header).first(2 * active),
                [&](std::size_t stream, std::size_t size, BlockCursor& cursor, std::size_t)
                {
                    auto& packet = output.packets.emplace_back();
                    packet.peer = senders[stream];
                    output.storage().owner->reserve(packet.bytes, size);
                    packet.bytes.resize(size);
                    observe(2);
                    cursor.read(packet.bytes.data(), size);
                    output.stats.received_bytes = plus(output.stats.received_bytes, size);
                },
                std::span(streams).first(active), receive_storage.owner.get(),
                [](void* storage, const char* address, std::size_t bytes) noexcept
                {
                    static_cast<geometry_detail::NativeConstructorStorage*>(storage)->retire_freed_buffer(
                        address, bytes);
                });
        });
    // Blocks and their descriptor arrays die before their receipt owner.
    for (auto& stream : streams)
        receive_storage.owner->reset_buffer(stream.blocks);
    receive_storage.owner->reset_buffer(streams);
    output.stats.sent_packets = input.size();
    output.stats.received_packets = audit.records;
    output.stats.largest_received_record = audit.largest_payload;
    output.stats.segmented_receive = true;
    output.stats.round_byte_limit = round_limit;
    output.stats.receive_block_capacity = received_wire;
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
    result.stats.decode_peak_bytes = capacity;
    for (const auto& packet : input)
        result.stats.largest_received_record = std::max(result.stats.largest_received_record, packet.bytes.size());
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
