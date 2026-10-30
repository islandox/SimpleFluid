/** @file testSparseOwnerExchange.cc
 * @brief Actual variable peer streams, empty ranks and collective rejection.
 */
#include "parallel/SparseOwnerExchange.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
namespace
{
using namespace SimpleFluid;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
constexpr std::size_t budget = 1024 * 1024;
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
