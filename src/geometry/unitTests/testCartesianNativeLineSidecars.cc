#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/PooledLineExchange.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <gtest/gtest.h>
#include <span>
#include <vector>
namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
template<class F> bool invalid(F&& f)
{
    try
    {
        f();
    }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}
TEST(CartesianNativeLineSidecars, ExistingV2BytesStayExactAndMalformedCompleteControlsReject)
{
    using Captured = std::vector<std::pair<bool, AdaptiveBytes>>;
    auto run = [&](bool source, PooledNativeLineFault fault, bool empty, Captured* captured)
    {
        auto arena = std::make_shared<Budget>(Budget{8 * 1024 * 1024});
        IdentityBudget identities;
        IncidencePools pools(*arena, identities, arena);
        PooledLineRegistry registry(*arena, pools);
        auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
        Writer key;
        key.bytes(std::array<char, 1>{'d'});
        key.bytes(PLIC::canonical_key(PLIC::details::GeometryIdentityAccess::line({0, {1, 0, 0}, {0, 0, 0}})));
        const auto line = pools.lines.intern(std::move(key.data));
        std::vector<PooledLineEdit> edits;
        for (unsigned i = 0; i < (empty ? 0U : 16U); ++i)
        {
            PooledLineSubscriber sub;
            sub.support = pools.supports.intern(AdaptiveBytes{'s', char(i)});
            sub.resident = sub.support_authority = 0;
            sub.first = pools.points.intern(PLIC::details::GeometryIdentityAccess::point({(long double) i, 0, 0}));
            sub.last = pools.points.intern(PLIC::details::GeometryIdentityAccess::point({(long double) (i + 1), 0, 0}));
            sub.source_generation = source ? 1 : 0;
            sub.native_source = source;
            edits.push_back({line, std::move(sub), false});
        }
        registry.begin_replay(true);
        std::size_t cursor = 0, replies = 0;
        const auto next = [&](PooledLineEdit& item)
        {
            if (cursor == edits.size())
                return false;
            item = edits[cursor++];
            return true;
        };
        const auto observe =
            [&](bool reply, unsigned, std::uint64_t, std::uint64_t, std::span<const OwnerPacket> packets)
        {
            if (!captured)
                return;
            for (const auto& packet : packets)
            {
                Reader reader{packet.bytes};
                EXPECT_TRUE(reader.word() == 2);
                captured->emplace_back(reply, packet.bytes);
            }
        };
        exchange_pooled_line_edits(
            registry, pools, next, 0, {7, 11, 13, 17, 19}, comm, *arena, 64 * 1024,
            [&](auto, const auto& subscriber)
            {
                EXPECT_TRUE(subscriber.result_size() == 2);
                EXPECT_TRUE(subscriber.native_source == source && subscriber.source_generation == (source ? 1U : 0U));
                ++replies;
            },
            {}, source, fault, observe);
        EXPECT_TRUE(replies == (empty ? 0 : 16));
    };
    Captured original, native;
    run(false, {}, false, &original);
    run(true, {}, false, &native);
    EXPECT_TRUE(original == native && original.size() == 32);
    for (bool reply : {false, true})
        for (auto kind : {PooledNativeLineFault::Kind::MissingCompletion,
                 PooledNativeLineFault::Kind::DuplicateCompletion, PooledNativeLineFault::Kind::StaleContext,
                 PooledNativeLineFault::Kind::ReorderedReference, PooledNativeLineFault::Kind::TruncatedCompletion})
        {
            PooledNativeLineFault fault;
            fault.kind = kind;
            fault.reply = reply;
            EXPECT_TRUE(invalid([&] { run(true, fault, false, nullptr); }));
        }
    run(true, {}, true, nullptr);
    PooledNativeLineFault missing;
    missing.kind = PooledNativeLineFault::Kind::MissingCompletion;
    missing.reply = false;
    EXPECT_TRUE(invalid([&] { run(true, missing, true, nullptr); }));
}
} // namespace
