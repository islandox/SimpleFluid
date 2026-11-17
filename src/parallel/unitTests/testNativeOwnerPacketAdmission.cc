#include "parallel/NativePacketAdmission.hh"
#include "utils/testing_environment.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/unitTests/native_constructor_allocation_operators.inc"
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <gtest/gtest.h>
#include <optional>
namespace
{
namespace G = SimpleFluid::geometry_detail;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using SimpleFluid::NativeOwnerExchange;
using SimpleFluid::OwnerExchange;
using SimpleFluid::OwnerPacket;
struct Counter
{
    std::size_t limit = 0, used = 0, peak = 0, cumulative = 0;
    G::NativeCounterAllocationEnvelope native_allocation;
    void add(std::size_t bytes)
    {
        if (used > limit || bytes > limit - used)
            throw std::invalid_argument("native packet test cap");
        used += bytes;
        peak = std::max(peak, used);
        cumulative += bytes;
    }
};
struct Domain
{
    G::NativeOwnedCounter<Counter> counter;
    G::NativeConstructionAdmission admission;
    explicit Domain(std::size_t cap = 8 * 1024 * 1024)
        : counter(G::NativeConstructionCounterAccess::create<Counter>(cap)), admission(counter)
    {
    }
};
std::vector<OwnerPacket> packets(const Teuchos::Comm<int>& comm)
{
    const int rank = comm.getRank(), size = comm.getSize();
    std::vector<OwnerPacket> result;
    // Rank 3 is a genuinely empty sender in the four-rank case. Rank 0 still
    // sends to it, so send-zero and receive-positive are exercised independently.
    if (size >= 4 && rank == 3)
        return result;
    result.push_back({rank, std::vector<char>(2 + rank, char('a' + rank))});
    result.push_back({(rank + 1) % size, std::vector<char>(37 + 3 * rank, char('A' + rank))});
    result.push_back({(rank + 1) % size, {}});
    result.push_back({(rank + size - 1) % size, {'x', char(rank)}});
    return result;
}
void equal(const OwnerExchange& actual, const OwnerExchange& expected)
{
    ASSERT_EQ(actual.packets.size(), expected.packets.size());
    for (std::size_t i = 0; i < actual.packets.size(); ++i)
    {
        EXPECT_EQ(actual.packets[i].peer, expected.packets[i].peer) << i;
        EXPECT_EQ(actual.packets[i].bytes, expected.packets[i].bytes) << i;
    }
    EXPECT_EQ(actual.stats.sent_packets, expected.stats.sent_packets);
    EXPECT_EQ(actual.stats.received_packets, expected.stats.received_packets);
    EXPECT_EQ(actual.stats.sent_bytes, expected.stats.sent_bytes);
    EXPECT_EQ(actual.stats.received_bytes, expected.stats.received_bytes);
}
void oracle(const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    const auto input = packets(*comm);
    const auto expected = SimpleFluid::exchange_owner_packets(input, comm, 8 * 1024 * 1024);
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    std::optional<NativeOwnerExchange> result;
    result.emplace(SimpleFluid::exchange_owner_packets(input, comm, domain.admission, 7));
    const auto live = domain.counter.owner()->used - prefix;
    const auto trace = capture.finish();
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(live, trace.all_live);
    EXPECT_GT(result->stats.peak_scratch_bytes, 0u);
    EXPECT_TRUE(result->stats.segmented_receive);
    equal(*result, expected);
    result.reset();
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativeOwnerPacketAdmission, SerialAndActualCommunicatorMatchPublicPacketBytes)
{
    oracle(Teuchos::rcp(new Teuchos::SerialComm<int>()));
    oracle(Tpetra::getDefaultComm());
}
TEST(NativeOwnerPacketAdmission, SplitCommunicatorPreservesLocalRanksAndPacketOrder)
{
    const auto parent = Tpetra::getDefaultComm();
    const auto sub = parent->split(parent->getRank() % 2, -parent->getRank());
    oracle(sub);
}
TEST(NativeOwnerPacketAdmission, EmptyAllRanksAndInvalidPeerRejectBeforeHeadersThenRetry)
{
    const auto comm = Tpetra::getDefaultComm();
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    {
        const auto empty = SimpleFluid::exchange_owner_packets({}, comm, domain.admission, 3);
        EXPECT_TRUE(empty.packets.empty());
        EXPECT_EQ(empty.stats.sent_bytes, 0u);
        EXPECT_EQ(empty.stats.received_bytes, 0u);
    }
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    auto input = packets(*comm);
    if (comm->getRank() == comm->getSize() - 1)
        input.push_back({comm->getSize(), {'z'}});
    EXPECT_THROW((void) SimpleFluid::exchange_owner_packets(input, comm, domain.admission, 7), std::invalid_argument);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
    input = packets(*comm);
    {
        const auto retry = SimpleFluid::exchange_owner_packets(input, comm, domain.admission, 7);
        const auto expected = SimpleFluid::exchange_owner_packets(input, comm, 8 * 1024 * 1024);
        equal(retry, expected);
    }
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
TEST(NativeOwnerPacketAdmission, EveryCppFailurePrefixRejectsCollectivelyAndLeavesDomainReusable)
{
    const auto comm = Tpetra::getDefaultComm();
    const auto input = packets(*comm);
    Domain measured;
    native_constructor_test::Capture capture(measured.counter.owner()->used);
    {
        const auto result = SimpleFluid::exchange_owner_packets(input, comm, measured.admission, 11);
    }
    const auto trace = capture.finish();
    auto local = static_cast<unsigned long long>(trace.phase[0].allocations);
    unsigned long long attempts = 0;
    Teuchos::broadcast(*comm, comm->getSize() - 1, 1, &local);
    attempts = local;
    for (std::size_t ordinal = 0; ordinal < attempts; ++ordinal)
    {
        Domain domain;
        const auto accepted = domain.admission.shared<unsigned>(101);
        const auto prefix = domain.counter.owner()->used;
        bool rejected = false;
        {
            std::optional<native_constructor_test::Failure> failure;
            if (comm->getRank() == comm->getSize() - 1)
                failure.emplace(ordinal);
            try
            {
                const auto result = SimpleFluid::exchange_owner_packets(input, comm, domain.admission, 11);
            }
            catch (const std::exception&)
            {
                rejected = true;
            }
        }
        EXPECT_TRUE(rejected) << ordinal;
        EXPECT_EQ(*accepted, 101u);
        EXPECT_EQ(domain.counter.owner()->used, prefix) << ordinal;
        {
            const auto retry = SimpleFluid::exchange_owner_packets(input, comm, domain.admission, 11);
        }
        EXPECT_EQ(domain.counter.owner()->used, prefix);
    }
}
TEST(NativeOwnerPacketAdmission, QualifiedFailureGateUsesNoCppAllocation)
{
    const auto comm = Tpetra::getDefaultComm();
    Domain domain;
    const auto prefix = domain.counter.owner()->used;
    native_constructor_test::Capture capture(domain.counter.owner()->used);
    SimpleFluid::Meshes::AMR::adaptive_detail::collective(*comm, [] {});
    const auto trace = capture.finish();
    std::size_t allocations = 0;
    for (const auto& phase : trace.phase)
        allocations += phase.allocations;
    EXPECT_EQ(allocations, 0u);
    EXPECT_EQ(trace.all_live, 0u);
    EXPECT_EQ(domain.counter.owner()->used, prefix);
}
} // namespace
