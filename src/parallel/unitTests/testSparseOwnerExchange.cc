/** @file testSparseOwnerExchange.cc
 * @brief Actual variable peer streams, empty ranks and collective rejection.
 */
#include "parallel/SparseOwnerExchange.hh"
#include "parallel/SparseOwnerExchangeFraming.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <Tpetra_Distributor.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
namespace
{
using namespace SimpleFluid;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr std::size_t budget = 1024 * 1024;

std::vector<char> framed_payloads(const std::vector<std::vector<char>>& payloads)
{
    std::vector<char> bytes;
    for (const auto& payload : payloads)
    {
        const std::uint64_t size = payload.size();
        const auto offset = bytes.size();
        bytes.resize(offset + sizeof(size));
        std::memcpy(bytes.data() + offset, &size, sizeof(size));
        bytes.insert(bytes.end(), payload.begin(), payload.end());
    }
    return bytes;
}

owner_exchange_detail::OwnerStreamBlocks split_stream(std::span<const char> bytes, std::size_t block_size)
{
    owner_exchange_detail::OwnerStreamBlocks result;
    for (std::size_t offset = 0; offset < bytes.size(); offset += block_size)
    {
        const auto block = bytes.subspan(offset, std::min(block_size, bytes.size() - offset));
        result.blocks.emplace_back(block.begin(), block.end());
    }
    return result;
}

struct PayloadAllocations
{
    std::size_t count = 0;
    std::array<std::size_t, 3> sizes{};
};

void observe_payload_allocation(void* context, std::size_t size)
{
    auto& allocations = *static_cast<PayloadAllocations*>(context);
    if (allocations.count < allocations.sizes.size())
        allocations.sizes[allocations.count] = size;
    ++allocations.count;
}

std::size_t retained_block_metadata(const std::vector<owner_exchange_detail::OwnerStreamBlocks>& streams,
    const std::vector<std::size_t>& sizes, const std::vector<std::uint64_t>& headers,
    const std::vector<int>& senders)
{
    auto bytes = streams.capacity() * sizeof(owner_exchange_detail::OwnerStreamBlocks) +
                 sizes.capacity() * sizeof(std::size_t) + headers.capacity() * sizeof(std::uint64_t) +
                 senders.capacity() * sizeof(int);
    for (const auto& stream : streams)
        bytes += stream.blocks.capacity() * sizeof(std::vector<char>);
    return bytes;
}

TEST(SparseOwnerExchange, SegmentedAuditAcceptsEveryWordAndPayloadSplit)
{
    using owner_exchange_detail::OwnerStreamBlocks;
    using owner_exchange_detail::walk_owner_block_streams;
    const std::array<std::vector<char>, 2> wire{
        framed_payloads({{'a', 'b', 'c'}, {}}), framed_payloads({{'d', 'e'}})};
    const std::array<std::size_t, 2> sizes{19, 10};
    const std::array<std::uint64_t, 4> headers{2, 19, 1, 10};
    auto check = [&](const std::array<OwnerStreamBlocks, 2>& streams)
    {
        const auto audit = walk_owner_block_streams(streams, sizes, headers);
        EXPECT_EQ(audit.records, 3u);
        EXPECT_EQ(audit.payload_bytes, 5u);
        EXPECT_EQ(audit.largest_payload, 3u);
    };
    // Every possible split independently exercises all seven interior word
    // offsets, payload boundaries, and an empty record after a nonempty one.
    for (std::size_t stream = 0; stream < wire.size(); ++stream)
        for (std::size_t cut = 1; cut < wire[stream].size(); ++cut)
        {
            SCOPED_TRACE(testing::Message() << "stream=" << stream << " cut=" << cut);
            std::array<OwnerStreamBlocks, 2> streams;
            for (std::size_t i = 0; i < wire.size(); ++i)
                streams[i] = split_stream(wire[i], wire[i].size());
            streams[stream].blocks.clear();
            streams[stream].blocks.emplace_back(wire[stream].begin(), wire[stream].begin() + cut);
            streams[stream].blocks.emplace_back(wire[stream].begin() + cut, wire[stream].end());
            check(streams);
        }
    for (const auto width : {1u, 7u, 8u, 9u})
    {
        SCOPED_TRACE(width);
        check({split_stream(wire[0], width), split_stream(wire[1], width)});
    }
    EXPECT_EQ(walk_owner_block_streams({}, {}, {}).records, 0u);
}

TEST(SparseOwnerExchange, SegmentedAuditRejectsLateFramingAndDeclaredCoverageErrors)
{
    using owner_exchange_detail::OwnerStreamBlocks;
    using owner_exchange_detail::walk_owner_block_streams;
    // A valid first record is much larger than a block. None of its contents
    // excuses an invalid final record or a mismatch in complete coverage.
    const auto bytes = framed_payloads({std::vector<char>(37, 'p'), {}, {'x', 'y'}});
    const std::array<std::size_t, 1> sizes{bytes.size()};
    const std::array<std::uint64_t, 2> headers{3, bytes.size()};
    const auto valid = split_stream(bytes, 7);
    auto reject = [&](const std::vector<char>& bad, std::array<std::size_t, 1> declared_sizes,
                      std::array<std::uint64_t, 2> declared_headers)
    {
        const std::array<OwnerStreamBlocks, 1> streams{split_stream(bad, 7)};
        EXPECT_THROW((void) walk_owner_block_streams(streams, declared_sizes, declared_headers),
            std::invalid_argument);
    };
    auto bad = bytes;
    bad.pop_back();
    reject(bad, {bad.size()}, {3, bad.size()});
    bad = bytes;
    bad.resize(8 + 37 + 8 + 7);
    reject(bad, {bad.size()}, {3, bad.size()});
    bad = bytes;
    const auto huge = std::numeric_limits<std::uint64_t>::max();
    std::memcpy(bad.data() + 8 + 37 + 8, &huge, sizeof(huge));
    reject(bad, sizes, headers);
    bad = bytes;
    bad.push_back('!');
    reject(bad, {bad.size()}, {3, bad.size()});
    reject(bytes, sizes, {2, bytes.size()});
    reject(bytes, sizes, {4, bytes.size()});
    reject(bytes, {bytes.size() - 1}, headers);
    const std::array<OwnerStreamBlocks, 1> streams{valid};
    EXPECT_THROW((void) walk_owner_block_streams(streams, sizes, std::span(headers).first(1)),
        std::invalid_argument);
    EXPECT_EQ(walk_owner_block_streams(streams, sizes, headers).payload_bytes, 39u);
}

TEST(SparseOwnerExchange, LateMalformedRankAllocatesNoDecodedPayloadOnAnyRank)
{
    using owner_exchange_detail::decode_owner_block_streams;
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank(), last = comm->getSize() - 1;
    const auto valid = framed_payloads({std::vector<char>(37, 'p'), {}, {'x', 'y'}});
    auto bytes = valid;
    if (rank == last)
    {
        // The final word crosses a block boundary and claims one unavailable
        // byte. A decoder that starts before the global audit would first
        // allocate the valid 37-byte payload on one or more ranks.
        const std::uint64_t invalid_final_length = 3;
        std::memcpy(bytes.data() + 8 + 37 + 8, &invalid_final_length, sizeof(invalid_final_length));
    }
    std::vector<owner_exchange_detail::OwnerStreamBlocks> streams;
    streams.push_back(split_stream(bytes, 7));
    const std::vector<std::size_t> sizes{bytes.size()};
    const std::vector<std::uint64_t> headers{3, bytes.size()};
    const std::vector<int> senders{(rank + 1) % comm->getSize()};
    auto metadata = retained_block_metadata(streams, sizes, headers, senders);
    PayloadAllocations allocations;
    // An independent rank's resource shortage must not change the malformed
    // framing result into a resource fallback or allow any rank to decode.
    EXPECT_THROW((void) decode_owner_block_streams(streams, sizes, headers, senders, comm,
                     rank == 0 ? 0 : budget, metadata, &allocations, observe_payload_allocation),
        std::invalid_argument);
    EXPECT_EQ(allocations.count, 0u);
    std::vector<char> unchanged;
    for (const auto& block : streams.front().blocks)
        unchanged.insert(unchanged.end(), block.begin(), block.end());
    EXPECT_EQ(unchanged, bytes);
    streams.front() = split_stream(valid, 7);
    metadata = retained_block_metadata(streams, sizes, headers, senders);
    const auto result = decode_owner_block_streams(streams, sizes, headers, senders, comm, budget, metadata,
        &allocations, observe_payload_allocation);
    EXPECT_EQ(allocations.count, 2u);
    EXPECT_EQ(allocations.sizes[0], 37u);
    EXPECT_EQ(allocations.sizes[1], 2u);
    EXPECT_EQ(result.stats.received_bytes, 39u);
    EXPECT_EQ(result.packets.size(), 3u);
    const std::array<std::vector<char>, 3> expected{
        std::vector<char>(37, 'p'), std::vector<char>{}, std::vector<char>{'x', 'y'}};
    for (std::size_t i = 0; i < std::min(result.packets.size(), expected.size()); ++i)
    {
        EXPECT_EQ(result.packets[i].peer, senders.front());
        EXPECT_EQ(result.packets[i].bytes, expected[i]);
    }
    for (const auto& block : streams.front().blocks)
        EXPECT_EQ(block.capacity(), 0u);
}

TEST(SparseOwnerExchange, DestructiveDecodeAdmitsExactCapacitiesAndRejectsOneByteShort)
{
    using owner_exchange_detail::decode_owner_block_streams;
    const auto comm = Tpetra::getDefaultComm();
    const auto wire = framed_payloads({{'a', 'b', 'c'}, {}, {'d', 'e'}});
    std::vector<owner_exchange_detail::OwnerStreamBlocks> streams(1);
    streams.front().blocks.reserve(3);
    const std::array<std::size_t, 3> block_sizes{11, 8, 10}, block_capacities{64, 16, 32};
    std::size_t offset = 0, receive_capacity = 0;
    for (std::size_t i = 0; i < block_sizes.size(); ++i)
    {
        auto& block = streams.front().blocks.emplace_back();
        block.reserve(block_capacities[i]);
        block.insert(block.end(), wire.begin() + offset, wire.begin() + offset + block_sizes[i]);
        offset += block_sizes[i];
        receive_capacity += block.capacity();
    }
    const std::vector<std::size_t> sizes{29};
    const std::vector<std::uint64_t> headers{3, 29};
    const std::vector<int> senders{comm->getRank()};
    const auto retained_metadata = retained_block_metadata(streams, sizes, headers, senders);
    const auto metadata = retained_metadata + 3 * sizeof(OwnerPacket);
    // Initial receive owners retain 64+16+32 bytes. The first 3-byte payload
    // allocates while all remain live, making the exact peak M+115. Its final
    // source byte then retires the 64-byte block. Reading the empty packet's
    // word retires the 16-byte block, so the final 2-byte allocation peaks only
    // at M+37. This oracle uses actual capacity, independent of production stats.
    EXPECT_EQ(receive_capacity, 112u);
    const auto exact_peak = metadata + receive_capacity + 3;
    PayloadAllocations allocations;
    EXPECT_THROW((void) decode_owner_block_streams(streams, sizes, headers, senders, comm,
                     comm->getRank() == comm->getSize() - 1 ? exact_peak - 1 : exact_peak,
                     retained_metadata, &allocations, observe_payload_allocation),
        std::length_error);
    EXPECT_EQ(allocations.count, 0u);
    for (std::size_t i = 0; i < block_sizes.size(); ++i)
    {
        EXPECT_EQ(streams.front().blocks[i].size(), block_sizes[i]);
        EXPECT_EQ(streams.front().blocks[i].capacity(), block_capacities[i]);
    }
    const auto result = decode_owner_block_streams(streams, sizes, headers, senders, comm, exact_peak,
        retained_metadata, &allocations, observe_payload_allocation);
    EXPECT_EQ(allocations.count, 2u);
    EXPECT_EQ(allocations.sizes[0], 3u);
    EXPECT_EQ(allocations.sizes[1], 2u);
    EXPECT_EQ(result.stats.receive_block_capacity, receive_capacity);
    EXPECT_EQ(result.stats.decode_peak_bytes, exact_peak);
    EXPECT_EQ(result.stats.peak_scratch_bytes, exact_peak);
    EXPECT_EQ(result.stats.received_packets, 3u);
    EXPECT_EQ(result.stats.received_bytes, 5u);
    EXPECT_EQ(result.packets.capacity(), 3u);
    const std::array<std::vector<char>, 3> expected{{{'a', 'b', 'c'}, {}, {'d', 'e'}}};
    EXPECT_EQ(result.packets.size(), expected.size());
    for (std::size_t i = 0; i < std::min(result.packets.size(), expected.size()); ++i)
    {
        EXPECT_EQ(result.packets[i].peer, comm->getRank());
        EXPECT_EQ(result.packets[i].bytes, expected[i]);
    }
    for (const auto& block : streams.front().blocks)
    {
        EXPECT_TRUE(block.empty());
        EXPECT_EQ(block.capacity(), 0u);
    }
}

TEST(SparseOwnerExchange, DestructiveDecodeRetiresReservedEmptyTailAfterEveryFinalPayload)
{
    using owner_exchange_detail::decode_owner_block_streams;
    const auto comm = Tpetra::getDefaultComm();
    for (const std::size_t payload_size : {0u, 3u})
    {
        SCOPED_TRACE(payload_size);
        const std::vector<char> payload(payload_size, 'x');
        const auto wire = framed_payloads({payload});
        std::vector<owner_exchange_detail::OwnerStreamBlocks> streams;
        streams.push_back(split_stream(wire, wire.size()));
        streams.front().blocks.emplace_back().reserve(41);
        streams.front().blocks.emplace_back().reserve(17);
        const std::vector<std::size_t> sizes{wire.size()};
        const std::vector<std::uint64_t> headers{1, wire.size()};
        const std::vector<int> senders{comm->getRank()};
        const auto retained_metadata = retained_block_metadata(streams, sizes, headers, senders);
        std::size_t receive_capacity = 0;
        for (const auto& block : streams.front().blocks)
            receive_capacity += block.capacity();
        // Empty trailing owners count at admission even though they contribute
        // no wire bytes. A final empty payload still requires their retirement.
        const auto exact_peak = retained_metadata + sizeof(OwnerPacket) + receive_capacity + payload_size;
        PayloadAllocations allocations;
        EXPECT_THROW((void) decode_owner_block_streams(streams, sizes, headers, senders, comm,
                         comm->getRank() == comm->getSize() - 1 ? exact_peak - 1 : exact_peak,
                         retained_metadata, &allocations, observe_payload_allocation),
            std::length_error);
        EXPECT_EQ(allocations.count, 0u);
        EXPECT_EQ(streams.front().blocks[1].capacity(), 41u);
        EXPECT_EQ(streams.front().blocks[2].capacity(), 17u);
        const auto result = decode_owner_block_streams(streams, sizes, headers, senders, comm, exact_peak,
            retained_metadata, &allocations, observe_payload_allocation);
        EXPECT_EQ(allocations.count, payload_size ? 1u : 0u);
        EXPECT_EQ(result.packets.size(), 1u);
        if (!result.packets.empty())
        {
            EXPECT_EQ(result.packets.front().peer, comm->getRank());
            EXPECT_EQ(result.packets.front().bytes, payload);
        }
        EXPECT_EQ(result.stats.receive_block_capacity, receive_capacity);
        EXPECT_EQ(result.stats.peak_scratch_bytes, exact_peak);
        for (const auto& block : streams.front().blocks)
        {
            EXPECT_TRUE(block.empty());
            EXPECT_EQ(block.capacity(), 0u);
        }
    }
}

TEST(SparseOwnerExchange, BoundedRoundsPreserveDistributorOrderAndInputAcrossFailure)
{
    using owner_exchange_detail::exchange_owner_packets_with_round_limit;
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank(), last = comm->getSize() - 1;
    constexpr std::size_t chunk = 17;
    auto payload = [](int source, std::size_t ordinal, std::size_t size)
    {
        std::vector<char> bytes(size);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<char>((i + 29 * source + 11 * ordinal) % 127);
        return bytes;
    };
    auto source_packets = [&](int source)
    {
        return std::vector<OwnerPacket>{{last, payload(source, 0, 3 * chunk + 5)}, {source, {}},
            {0, payload(source, 2, 7)}, {last, payload(source, 3, chunk - 1)},
            {0, payload(source, 4, chunk + 1)}, {source, payload(source, 5, 8)}};
    };
    const bool active = rank == 0 || rank == last;
    auto packets = active ? source_packets(rank) : std::vector<OwnerPacket>{};
    // Construct only the established routing plan to obtain its sender order;
    // payload bytes and within-peer order come independently from the fixture.
    std::vector<int> peers;
    if (active)
    {
        peers.push_back(0);
        if (last)
            peers.push_back(last);
    }
    Tpetra::Distributor reference_plan(comm);
    (void) reference_plan.createFromSends(Teuchos::arrayViewFromVector(peers));
    std::vector<OwnerPacket> expected;
    for (const auto source : reference_plan.getProcsFrom())
        for (const auto& packet : source_packets(source))
            if (packet.peer == rank)
                expected.push_back({source, packet.bytes});
    const std::size_t local_budget = !active ? 0 : rank == 0 ? 64 * 1024 : budget;
    auto check = [&](const OwnerExchange& result)
    {
        ASSERT_EQ(result.packets.size(), expected.size());
        std::size_t received_bytes = 0, retained = result.packets.capacity() * sizeof(OwnerPacket);
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_EQ(result.packets[i].peer, expected[i].peer);
            EXPECT_EQ(result.packets[i].bytes, expected[i].bytes);
            received_bytes += expected[i].bytes.size();
            retained += result.packets[i].bytes.capacity();
        }
        EXPECT_EQ(result.stats.received_packets, expected.size());
        EXPECT_EQ(result.stats.received_bytes, received_bytes);
        EXPECT_GE(result.stats.peak_scratch_bytes, retained);
        EXPECT_LE(result.stats.peak_scratch_bytes, local_budget);
    };
    const auto original = packets;
    const auto* packet_owner = packets.data();
    const auto packet_capacity = packets.capacity();
    std::vector<const char*> payload_owners;
    std::vector<std::size_t> payload_capacities;
    for (const auto& packet : packets)
    {
        payload_owners.push_back(packet.bytes.data());
        payload_capacities.push_back(packet.bytes.capacity());
    }
    auto check_unchanged = [&]
    {
        ASSERT_EQ(packets.size(), original.size());
        EXPECT_EQ(packets.data(), packet_owner);
        EXPECT_EQ(packets.capacity(), packet_capacity);
        for (std::size_t i = 0; i < packets.size(); ++i)
        {
            EXPECT_EQ(packets[i].peer, original[i].peer);
            EXPECT_EQ(packets[i].bytes, original[i].bytes);
            EXPECT_EQ(packets[i].bytes.data(), payload_owners[i]);
            EXPECT_EQ(packets[i].bytes.capacity(), payload_capacities[i]);
        }
    };
    auto result = exchange_owner_packets_with_round_limit(packets, comm, local_budget, chunk);
    check(result);
    EXPECT_TRUE(result.stats.segmented_receive);
    EXPECT_EQ(result.stats.round_byte_limit, chunk);
    EXPECT_GT(result.stats.transport_rounds, 1u);
    check_unchanged();
    EXPECT_THROW((void) consume_owner_packets(std::move(packets), comm, rank == 0 ? 1 : local_budget),
        std::length_error);
    check_unchanged();
    check(consume_owner_packets(std::move(packets), comm, local_budget));
    EXPECT_TRUE(packets.empty());
    const auto empty = exchange_owner_packets_with_round_limit({}, comm, 0, chunk);
    EXPECT_TRUE(empty.packets.empty());
    EXPECT_EQ(empty.stats.sent_packets, 0u);
    EXPECT_EQ(empty.stats.received_packets, 0u);
    EXPECT_EQ(empty.stats.peak_scratch_bytes, 0u);
}

TEST(SparseOwnerExchange, BoundedSendHeavyRankAdmitsOnlyItsRoundStaging)
{
    using owner_exchange_detail::exchange_owner_packets_with_round_limit;
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank(), last = comm->getSize() - 1;
    constexpr std::size_t count = 400, payload_size = 1024, chunk = 1024;
    std::vector<OwnerPacket> packets;
    if (rank == 0)
        for (std::size_t i = 0; i < count; ++i)
            packets.push_back({last, std::vector<char>(payload_size, static_cast<char>(i % 127))});
    // In MPI, the send-only owner's complete stream exceeds its budget, while
    // its actual metadata plus one send chunk fit. Receive-only and idle owners
    // have independently chosen budgets and must execute the same rounds.
    const std::size_t local_budget = rank == last ? budget : rank == 0 ? 96 * 1024 : 0;
    if (rank == 0 && rank != last)
        EXPECT_GT(count * (payload_size + sizeof(std::uint64_t)), local_budget);
    const auto result = exchange_owner_packets_with_round_limit(packets, comm, local_budget, chunk);
    EXPECT_TRUE(result.stats.segmented_receive);
    EXPECT_GT(result.stats.transport_rounds, 1u);
    EXPECT_EQ(result.stats.sent_bytes, rank == 0 ? count * payload_size : 0);
    EXPECT_EQ(result.stats.received_bytes, rank == last ? count * payload_size : 0);
    EXPECT_EQ(result.packets.size(), rank == last ? count : 0);
    for (std::size_t i = 0; i < result.packets.size(); ++i)
    {
        EXPECT_EQ(result.packets[i].peer, 0);
        EXPECT_EQ(result.packets[i].bytes, std::vector<char>(payload_size, static_cast<char>(i % 127)));
    }
    EXPECT_LE(result.stats.peak_scratch_bytes, local_budget);
    EXPECT_TRUE(exchange_owner_packets_with_round_limit({}, comm, 0, chunk).packets.empty());
}

TEST(SparseOwnerExchange, BoundedRoundsShrinkWhenTransferFitsButDecodeDoesNot)
{
    using owner_exchange_detail::exchange_owner_packets_with_round_limit;
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank(), last = comm->getSize() - 1;
    constexpr std::size_t count = 128, payload_size = 128, maximum_chunk = 8192;
    constexpr std::size_t fixed_budget = 28 * 1024;
    const bool active = rank == 0 || rank == last;
    const int peer = rank == 0 ? last : 0;
    auto payload = [](int source, std::size_t ordinal)
    {
        std::vector<char> bytes(payload_size);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<char>((i + 31 * source + 17 * ordinal) % 127);
        return bytes;
    };
    std::vector<OwnerPacket> packets;
    if (active)
        for (std::size_t i = 0; i < count; ++i)
            packets.push_back({peer, payload(rank, i)});
    const auto* input_owner = packets.data();
    const auto input_capacity = packets.capacity();
    std::vector<const char*> payload_owners;
    for (const auto& packet : packets)
        payload_owners.push_back(packet.bytes.data());
    auto check_input = [&]
    {
        EXPECT_EQ(packets.data(), input_owner);
        EXPECT_EQ(packets.capacity(), input_capacity);
        EXPECT_EQ(packets.size(), active ? count : 0);
        for (std::size_t i = 0; i < std::min(packets.size(), payload_owners.size()); ++i)
        {
            EXPECT_EQ(packets[i].peer, peer);
            EXPECT_EQ(packets[i].bytes, payload(rank, i));
            EXPECT_EQ(packets[i].bytes.data(), payload_owners[i]);
            EXPECT_EQ(packets[i].bytes.capacity(), payload_size);
        }
    };
    const auto reference = exchange_owner_packets_with_round_limit(
        packets, comm, active ? 64 * 1024 : 0, maximum_chunk);
    EXPECT_TRUE(reference.stats.segmented_receive);
    EXPECT_EQ(reference.stats.round_byte_limit, maximum_chunk);
    if (active)
    {
        // Before payload 61 allocates, its length word ends at byte 8168:
        // the first 8192-byte receive block is still live. All received wire
        // bytes, 61 whole payload owners, and the admitted output packet array
        // therefore coexist. This lower bound needs no private route-layout
        // constants and already exceeds the unchanged cap.
        constexpr auto wire_bytes = count * (sizeof(std::uint64_t) + payload_size);
        constexpr auto word_end = 60 * (sizeof(std::uint64_t) + payload_size) + sizeof(std::uint64_t);
        EXPECT_LT(word_end, maximum_chunk);
        EXPECT_GT(wire_bytes + 61 * payload_size + count * sizeof(OwnerPacket), fixed_budget);
        EXPECT_LE(reference.stats.transfer_peak_bytes, fixed_budget);
        EXPECT_GT(reference.stats.decode_peak_bytes, fixed_budget);
    }
    check_input();
    const auto result = exchange_owner_packets_with_round_limit(
        packets, comm, active ? fixed_budget : 0, maximum_chunk);
    EXPECT_TRUE(result.stats.segmented_receive);
    EXPECT_GT(result.stats.round_byte_limit, 0u);
    EXPECT_LT(result.stats.round_byte_limit, maximum_chunk);
    EXPECT_GT(result.stats.transport_rounds, reference.stats.transport_rounds);
    EXPECT_LE(result.stats.peak_scratch_bytes, active ? fixed_budget : 0);
    EXPECT_EQ(result.stats.sent_packets, active ? count : 0);
    EXPECT_EQ(result.stats.received_packets, active ? count : 0);
    EXPECT_EQ(result.stats.received_bytes, active ? count * payload_size : 0);
    EXPECT_EQ(result.packets.size(), active ? count : 0);
    EXPECT_EQ(result.packets.size(), reference.packets.size());
    for (std::size_t i = 0; i < std::min(result.packets.size(), reference.packets.size()); ++i)
    {
        EXPECT_EQ(result.packets[i].peer, peer);
        EXPECT_EQ(result.packets[i].bytes, payload(peer, i));
        EXPECT_EQ(result.packets[i].peer, reference.packets[i].peer);
        EXPECT_EQ(result.packets[i].bytes, reference.packets[i].bytes);
    }
    check_input();
}

TEST(SparseOwnerExchange, CompleteFrameAuditRejectsOversizedPrefixAndMalformedTails)
{
    using owner_exchange_detail::walk_owner_streams;
    auto word = [](std::vector<char>& bytes, std::uint64_t value)
    {
        const auto offset = bytes.size();
        bytes.resize(offset + sizeof(value));
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    };
    std::vector<char> bytes;
    word(bytes, 3);
    bytes.insert(bytes.end(), {'a', 'b', 'c'});
    word(bytes, 0);
    word(bytes, 2);
    bytes.insert(bytes.end(), {'d', 'e'});
    const std::array<std::size_t, 2> sizes{19, 10};
    const std::array<std::uint64_t, 4> headers{2, 19, 1, 10};
    const auto audit = walk_owner_streams(bytes, sizes, headers);
    EXPECT_EQ(audit.records, 3u);
    EXPECT_EQ(audit.payload_bytes, 5u);
    EXPECT_EQ(walk_owner_streams({}, {}, {}).records, 0u);
    EXPECT_THROW((void) walk_owner_streams(std::span(bytes).first(28), sizes, headers), std::invalid_argument);
    EXPECT_THROW((void) walk_owner_streams(bytes, sizes, std::span(headers).first(3)), std::invalid_argument);
    auto bad_headers = headers;
    bad_headers[0] = 1;
    EXPECT_THROW((void) walk_owner_streams(bytes, sizes, bad_headers), std::invalid_argument);
    bad_headers[0] = 0;
    EXPECT_THROW((void) walk_owner_streams(bytes, sizes, bad_headers), std::invalid_argument);
    auto bad = bytes;
    const auto huge = std::numeric_limits<std::uint64_t>::max();
    std::memcpy(bad.data(), &huge, sizeof(huge));
    EXPECT_THROW((void) walk_owner_streams(bad, sizes, headers), std::invalid_argument);
    // The first payload would allocate 16 bytes although a declaration of two
    // frames leaves only eight payload bytes. Audit must reject the missing
    // second frame before production invokes any allocating decode visitor.
    bad.clear();
    word(bad, 16);
    bad.resize(24, 'x');
    const std::array<std::size_t, 1> misleading_size{24};
    const std::array<std::uint64_t, 2> misleading_header{2, 24};
    EXPECT_THROW((void) walk_owner_streams(bad, misleading_size, misleading_header), std::invalid_argument);
    bad = bytes;
    bad.push_back('x');
    EXPECT_THROW((void) walk_owner_streams(bad, sizes, headers), std::invalid_argument);
    EXPECT_EQ(walk_owner_streams(bytes, sizes, headers).payload_bytes, 5u);
}
TEST(SparseOwnerExchange, ReceiveHeavyDecodeRetiresExportMetadataAndRecovers)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank(), last = comm->getSize() - 1;
    constexpr std::size_t payload_size = 512, phase_budget = 427000;
    auto payload = [](int source, std::size_t ordinal)
    {
        std::vector<char> result(payload_size);
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = char((i + 17 * source + 31 * ordinal) % 127);
        return result;
    };
    auto make_packets = [&]
    {
        std::vector<OwnerPacket> packets;
        const auto count = rank == 0 ? 400u : rank == last ? 200u : 0u;
        for (unsigned i = 0; i < count; ++i)
            packets.push_back({rank == 0 ? last : 0, payload(rank, i)});
        return packets;
    };
    auto check = [&](const OwnerExchange& result)
    {
        const auto count = rank == last ? 400u : rank == 0 ? 200u : 0u;
        const auto sender = rank == last ? 0 : last;
        ASSERT_EQ(result.packets.size(), count);
        EXPECT_EQ(result.stats.received_bytes, count * payload_size);
        EXPECT_EQ(result.stats.received_packets, count);
        EXPECT_LE(result.stats.peak_scratch_bytes, phase_budget);
        for (unsigned i = 0; i < count; ++i)
        {
            EXPECT_EQ(result.packets[i].peer, sender);
            EXPECT_EQ(result.packets[i].bytes, payload(sender, i));
        }
    };
    auto packets = make_packets();
    check(consume_owner_packets(std::move(packets), comm, phase_budget));
    EXPECT_TRUE(packets.empty());
    packets = make_packets();
    EXPECT_THROW((void) consume_owner_packets(std::move(packets), comm, rank == last ? 1 : phase_budget),
        std::length_error);
    EXPECT_EQ(packets.size(), make_packets().size());
    check(consume_owner_packets(std::move(packets), comm, phase_budget));
    EXPECT_TRUE(exchange_owner_packets({}, comm, 0).packets.empty());
}
TEST(SparseOwnerExchange, HeterogeneousDuplicatePeersAndEmptyOwners)
{
    auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank(), last = comm->getSize() - 1;
    std::vector<OwnerPacket> sent;
    if (rank == 0)
        sent = {{0, {}}, {last, {'a', 'b', 'c'}}, {last, std::vector<char>(8192, 'q')}};
    auto result = exchange_owner_packets(sent, comm, budget);
    std::size_t expected_packets = 0, expected_bytes = 0;
    if (rank == 0)
        ++expected_packets;
    if (rank == last)
    {
        expected_packets += 2;
        expected_bytes += 8195;
    }
    ASSERT_EQ(result.packets.size(), expected_packets);
    EXPECT_EQ(result.stats.received_bytes, expected_bytes);
    EXPECT_EQ(result.stats.received_packets, expected_packets);
    for (const auto& packet : result.packets)
        EXPECT_EQ(packet.peer, 0);
    std::size_t index = 0;
    if (rank == 0)
        EXPECT_TRUE(result.packets[index++].bytes.empty());
    if (rank == last)
    {
        EXPECT_EQ(result.packets[index++].bytes, (std::vector<char>{'a', 'b', 'c'}));
        EXPECT_EQ(result.packets[index++].bytes, std::vector<char>(8192, 'q'));
    }
    EXPECT_LE(result.stats.peak_scratch_bytes, budget);
    EXPECT_EQ(result.stats.sent_bytes, rank == 0 ? 8195 : 0);
}
TEST(SparseOwnerExchange, RankLocalBadPeerAndReceiveBudgetRejectCollectively)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<OwnerPacket> packets;
    if (comm->getRank() == 0)
        packets.push_back({comm->getSize(), {'x'}});
    EXPECT_THROW(exchange_owner_packets(packets, comm, budget), std::invalid_argument);
    // Invalid routing also outranks a resource shortage on that same owner;
    // rejecting metadata admission first would incorrectly allow a cap fallback.
    EXPECT_THROW(exchange_owner_packets(packets, comm, comm->getRank() == 0 ? 0 : budget),
        std::invalid_argument);
    packets.clear();
    if (comm->getRank() == 0)
        packets.push_back({comm->getSize() - 1, std::vector<char>(8192, 'x')});
    EXPECT_THROW(
        exchange_owner_packets(packets, comm, comm->getRank() == comm->getSize() - 1 ? 1 : budget), std::length_error);
    // Invalid metadata outranks a simultaneous resource failure on another
    // owner; every rank must make the same reject-versus-fallback decision.
    if (comm->getSize() > 1)
    {
        packets.clear();
        if (comm->getRank() == 0)
            packets.push_back({comm->getSize(), {'x'}});
        else if (comm->getRank() == comm->getSize() - 1)
            packets.push_back({0, {'x'}});
        EXPECT_THROW(exchange_owner_packets(packets, comm, comm->getRank() == 0 ? budget : 0), std::invalid_argument);
    }
    // A failed round does not poison later exchanges, including all-empty ranks.
    EXPECT_TRUE(exchange_owner_packets({}, comm, 0).packets.empty());
}
TEST(SparseOwnerExchange, ConsumedLocalDeliveryMovesPayloadAndAdmitsRetainedCapacities)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<OwnerPacket> packets;
    packets.reserve(16);
    packets.push_back({comm->getRank(), {'x'}});
    packets.front().bytes.reserve(4096);
    const auto capacity = packets.capacity() * sizeof(OwnerPacket) + packets.front().bytes.capacity();
    const auto* payload = packets.front().bytes.data();
    EXPECT_THROW(
        consume_owner_packets(std::move(packets), comm, comm->getRank() == comm->getSize() - 1 ? capacity - 1 : budget),
        std::length_error);
    // Rejection leaves the ownership transfer unperformed and the next
    // collective can succeed, including otherwise empty application owners.
    auto result = consume_owner_packets(std::move(packets), comm, capacity);
    ASSERT_EQ(result.packets.size(), 1);
    EXPECT_EQ(result.packets.front().bytes.data(), payload);
    EXPECT_EQ(result.packets.front().peer, comm->getRank());
    EXPECT_EQ(result.stats.sent_bytes, 1);
    EXPECT_EQ(result.stats.received_bytes, 1);
    EXPECT_EQ(result.stats.peak_scratch_bytes, capacity);
    EXPECT_TRUE(consume_owner_packets({}, comm, 0).packets.empty());
}
TEST(SparseOwnerExchange, TransferAndDecodePhasesShareBudgetAfterPackedSendRelease)
{
    auto comm = Tpetra::getDefaultComm();
    const auto rank = comm->getRank(), last = comm->getSize() - 1;
    constexpr std::size_t payload_size = 32 * 1024;
    // Two equal packets are sent and received by each active owner. Either live
    // phase needs roughly four payloads, whereas all three buffers need six.
    // The extra payload allowance covers framing without copying its layout.
    constexpr std::size_t phase_budget = 5 * payload_size;
    auto payload = [](int source, bool local)
    {
        std::vector<char> bytes(payload_size);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<char>((i + 17 * source + (local ? 31 : 67)) % 127);
        return bytes;
    };
    auto make_packets = [&]
    {
        std::vector<OwnerPacket> packets;
        if (rank == 0 || rank == last)
        {
            packets.push_back({rank, payload(rank, true)});
            packets.push_back({rank == 0 ? last : 0, payload(rank, false)});
        }
        return packets;
    };
    auto check = [&](const OwnerExchange& result, bool transported)
    {
        if (rank == 0 || rank == last)
        {
            ASSERT_EQ(result.packets.size(), 2);
            EXPECT_EQ(result.stats.sent_bytes, 2 * payload_size);
            EXPECT_EQ(result.stats.received_bytes, 2 * payload_size);
            if (transported)
                EXPECT_GE(result.stats.peak_scratch_bytes, 4 * payload_size);
            // Independently show why retaining packed sends through decode
            // cannot meet this budget, even before counting record framing.
            EXPECT_GT(result.stats.sent_bytes + 2 * result.stats.received_bytes, phase_budget);
            const int other = rank == 0 ? last : 0;
            const auto local_payload = payload(rank, true), remote_payload = payload(other, false);
            std::size_t local_matches = 0, remote_matches = 0;
            for (const auto& packet : result.packets)
            {
                local_matches += packet.peer == rank && packet.bytes == local_payload;
                remote_matches += packet.peer == other && packet.bytes == remote_payload;
            }
            EXPECT_EQ(local_matches, 1);
            EXPECT_EQ(remote_matches, 1);
        }
        else
            EXPECT_TRUE(result.packets.empty());
        EXPECT_LE(result.stats.peak_scratch_bytes, phase_budget);
    };
    auto packets = make_packets();
    check(exchange_owner_packets(packets, comm, phase_budget), true);
    // The borrowed input remains intact, independently of temporary lifetimes.
    ASSERT_EQ(packets.size(), make_packets().size());
    if (!packets.empty())
    {
        EXPECT_EQ(packets[0].bytes, payload(rank, true));
        EXPECT_EQ(packets[1].bytes, payload(rank, false));
    }
    EXPECT_THROW(exchange_owner_packets(packets, comm, rank == 0 ? 3 * payload_size : phase_budget), std::length_error);
    check(consume_owner_packets(std::move(packets), comm, phase_budget), comm->getSize() > 1);
    EXPECT_TRUE(packets.empty());
    // A later valid round has empty sending ranks, including an owner that
    // receives a packet but contributes no export stream.
    if (rank == 0)
        packets.push_back({last, payload(0, false)});
    auto reused = exchange_owner_packets(packets, comm, phase_budget);
    ASSERT_EQ(reused.packets.size(), rank == last ? 1 : 0);
    if (rank == last)
    {
        EXPECT_EQ(reused.packets.front().peer, 0);
        EXPECT_EQ(reused.packets.front().bytes, payload(0, false));
    }
    EXPECT_TRUE(exchange_owner_packets({}, comm, 0).packets.empty());
}
} // namespace
