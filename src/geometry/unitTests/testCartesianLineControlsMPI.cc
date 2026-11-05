/** Complete native-line control messages on real communicators, including empty owners. */
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/PooledLineExchange.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
using Access = P::details::GeometryIdentityAccess;
using Wide = std::array<long double, 3>;
using Captured = std::vector<std::pair<bool, AMR::AdaptiveBytes>>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

AMR::AdaptiveBytes line_key()
{
    D::Writer writer;
    writer.bytes(std::array<char, 1>{'d'});
    writer.bytes(P::canonical_key(Access::line({0, {1, 0, 0}, {0, 0, 0}})));
    return std::move(writer.data);
}

void run(bool source_protocol, D::PooledNativeLineFault fault, bool empty, Captured* captured = nullptr)
{
    const auto comm = Tpetra::getDefaultComm();
    auto budget = std::make_shared<D::Budget>(D::Budget{8 * 1024 * 1024});
    D::IdentityBudget identities;
    D::IncidencePools pools(*budget, identities, budget);
    D::PooledLineRegistry registry(*budget, pools);
    const auto key = line_key();
    std::vector<D::PooledLineEdit> edits;
    D::collective(*comm,
        [&]
        {
            registry.begin_replay(true);
            // Only rank zero contributes. Other ranks still authenticate and
            // complete every control phase, including ranks owning no queries.
            if (!empty && comm->getRank() == 0)
                for (unsigned i = 0; i < 16; ++i)
                {
                    const AMR::AdaptiveBytes support{'s', char(i)};
                    D::PooledLineSubscriber subscriber;
                    subscriber.support = pools.supports.intern(support);
                    subscriber.resident = 0;
                    subscriber.support_authority = D::authority(support, comm->getSize());
                    subscriber.first = pools.points.intern(Access::point(Wide{(long double) i, 0, 0}));
                    subscriber.last = pools.points.intern(Access::point(Wide{(long double) (i + 1), 0, 0}));
                    subscriber.emission = i;
                    subscriber.native_source = source_protocol;
                    subscriber.source_generation = source_protocol ? 1 : 0;
                    edits.push_back({pools.lines.intern(key), std::move(subscriber), false});
                }
        });
    std::size_t cursor = 0, replies = 0;
    const auto observe =
        [&](bool reply, unsigned, std::uint64_t, std::uint64_t, std::span<const SimpleFluid::OwnerPacket> packets)
    {
        if (!captured)
            return;
        for (const auto& packet : packets)
        {
            D::Reader reader{packet.bytes};
            EXPECT_EQ(reader.word(), 2u);
            captured->emplace_back(reply, packet.bytes);
        }
    };
    D::exchange_pooled_line_edits(
        registry, pools,
        [&](D::PooledLineEdit& item)
        {
            if (cursor == edits.size())
                return false;
            item = edits[cursor++];
            return true;
        },
        0, {7, 11, 13, 17, 19}, comm, *budget, 64 * 1024,
        [&](std::size_t line, const D::PooledLineSubscriber& subscriber)
        {
            EXPECT_EQ(comm->getRank(), 0);
            EXPECT_EQ(pools.lines.get(line), key);
            EXPECT_EQ(subscriber.native_source, source_protocol);
            EXPECT_EQ(subscriber.source_generation, source_protocol ? 1u : 0u);
            EXPECT_EQ(subscriber.result_size(), 2u);
            if (subscriber.result_size() == 2)
            {
                EXPECT_EQ(Access::point(pools.points.get(subscriber.result_at(0))),
                    (Wide{(long double) subscriber.emission, 0, 0}));
                EXPECT_EQ(Access::point(pools.points.get(subscriber.result_at(1))),
                    (Wide{(long double) (subscriber.emission + 1), 0, 0}));
            }
            ++replies;
        },
        {}, source_protocol, fault, observe);
    D::collective(*comm, [&] { registry.finish_replay(0); });
    EXPECT_EQ(replies, !empty && comm->getRank() == 0 ? 16u : 0u);
}

void reject(D::PooledNativeLineFault fault, bool empty)
{
    int local_rejected = 0;
    try
    {
        run(true, fault, empty);
    }
    catch (const std::exception&)
    {
        local_rejected = 1;
    }
    int all_rejected = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MIN, 1, &local_rejected, &all_rejected);
    EXPECT_EQ(all_rejected, 1);
    // A failed temporary wave cannot poison a later clean session on this communicator.
    run(true, {}, empty);
}

TEST(CartesianLineControlsMPI, ExistingDataPacketsStayByteIdenticalWithEmptyOwners)
{
    Captured original, native;
    run(false, {}, false, &original);
    run(true, {}, false, &native);
    EXPECT_EQ(original, native);
    const auto comm = Tpetra::getDefaultComm();
    const std::uint64_t local = native.size();
    std::uint64_t global = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &global);
    EXPECT_EQ(global, 32u);
    run(true, {}, true);
}

TEST(CartesianLineControlsMPI, RankLocalMalformedEditAndReplyControlsRejectCollectivelyAndRecover)
{
    const auto comm = Tpetra::getDefaultComm();
    for (bool reply : {false, true})
        for (auto kind :
            {D::PooledNativeLineFault::Kind::MissingCompletion, D::PooledNativeLineFault::Kind::DuplicateCompletion,
                D::PooledNativeLineFault::Kind::StaleContext, D::PooledNativeLineFault::Kind::ReorderedReference,
                D::PooledNativeLineFault::Kind::TruncatedCompletion})
        {
            SCOPED_TRACE(reply);
            SCOPED_TRACE(static_cast<int>(kind));
            D::PooledNativeLineFault fault;
            fault.kind = kind;
            fault.reply = reply;
            fault.rank = reply ? D::authority(line_key(), comm->getSize()) : 0;
            reject(fault, false);
        }
    // A wholly empty edit stream still requires every sender's edit completion.
    {
        D::PooledNativeLineFault fault;
        fault.kind = D::PooledNativeLineFault::Kind::MissingCompletion;
        fault.reply = false;
        fault.rank = comm->getSize() - 1;
        reject(fault, true);
    }
    // An actual dirty wave is needed for a reply phase. A rank other than the
    // sole line authority has no reply data, but must still complete that phase.
    if (comm->getSize() > 1)
    {
        D::PooledNativeLineFault fault;
        fault.kind = D::PooledNativeLineFault::Kind::MissingCompletion;
        fault.reply = true;
        fault.rank = (D::authority(line_key(), comm->getSize()) + 1) % comm->getSize();
        reject(fault, false);
    }
}
} // namespace
