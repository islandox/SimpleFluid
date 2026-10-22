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
    EXPECT_THROW(exchange_owner_packets(packets, comm, comm->getRank() == comm->getSize() - 1 ? 1 : budget),
        std::length_error);
    // Invalid metadata outranks a simultaneous resource failure on another
    // owner; every rank must make the same reject-versus-fallback decision.
    if (comm->getSize() > 1)
    {
        packets.clear();
        if (comm->getRank() == 0)
            packets.push_back({comm->getSize(), {'x'}});
        else if (comm->getRank() == comm->getSize() - 1)
            packets.push_back({0, {'x'}});
        EXPECT_THROW(exchange_owner_packets(packets, comm, comm->getRank() == 0 ? budget : 0),
            std::invalid_argument);
    }
    // A failed round does not poison later exchanges, including all-empty ranks.
    EXPECT_TRUE(exchange_owner_packets({}, comm, 0).packets.empty());
}
} // namespace
