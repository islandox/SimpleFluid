/** @file testSparseOwnerExchange.cc
 * @brief Actual variable peer streams, empty ranks and collective rejection.
 */
#include "parallel/SparseOwnerExchange.hh"
#include "parallel/SparseOwnerExchangeFraming.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace
{
using namespace SimpleFluid;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr std::size_t budget = 1024 * 1024;
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
