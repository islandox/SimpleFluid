#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/GeometryOwnerImportAccess.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/PLIC/details/NativeIdentityOwnerAccess.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <gtest/gtest.h>
#include <map>

namespace
{
namespace P = SimpleFluid::PLIC;
namespace G = SimpleFluid::geometry_detail;
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
namespace M = P::details::geometry_demand;
namespace R = P::details::retention;
using Key = std::pair<unsigned, const void*>;
using Rows = std::map<Key, M::OwnerRecord>;
constexpr std::size_t cap = 16 * 1024 * 1024;
Rows rows(const M::GeometryOwnerManifest& manifest)
{
    Rows result;
    manifest.visit([&](const M::OwnerRecord& value)
        { EXPECT_TRUE(result.emplace(Key{value.key.kind, value.owner.get()}, value).second); });
    return result;
}
M::Admission charge(std::size_t& used)
{
    return {&used, [](void* raw, std::size_t bytes) { *static_cast<std::size_t*>(raw) += bytes; }};
}
TEST(NativeIdentityProjection, GraphMatchesFullRetainAndBorrowedCensusForAdmittedSupport)
{
    if (!G::native_constructor_capacity_profile_supported())
        return; // constructor tests explicitly exercise unsupported-profile rejection
    auto counter = G::NativeConstructionCounterAccess::create<D::Budget>(cap);
    G::NativeConstructionAdmission admission(counter);
    const auto support = P::details::GeometryIdentityAccess::support_axis_admitted(1, .5L, admission);
    std::size_t graph_used = 0;
    M::GeometryOwnerManifest graph_manifest;
    {
        P::details::GeometryOwnerGraphEditor graph({}, 0, graph_used, cap);
        graph.identity(support, true);
        graph_manifest = graph.finish();
    }
    P::GeometryAllocationLedger full;
    std::size_t full_used = 0;
    full.retain(support, full_used, cap);
    const auto oracle = P::details::GeometryOwnerImportAccess::export_manifest(full, charge(full_used));
    const auto expected = rows(oracle), actual = rows(graph_manifest);
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& [key, value] : expected)
    {
        const auto found = actual.find(key);
        ASSERT_NE(found, actual.end());
        EXPECT_EQ(found->second.bytes, value.bytes);
        EXPECT_FALSE(found->second.owner.owner_before(value.owner));
        EXPECT_FALSE(value.owner.owner_before(found->second.owner));
    }
    EXPECT_EQ(full.payload_bytes(), graph_manifest.payload_bytes());
    std::vector<std::byte> scratch_bytes(4 * 1024 * 1024);
    R::ProbeScratch scratch(scratch_bytes);
    const std::array roots{R::root(support)};
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&full};
    const auto census = R::inspect_retention(roots, ledgers, scratch, {1000, 1000, true});
    ASSERT_TRUE(census.complete()) << census.detail;
    EXPECT_EQ(census.owner_union.bytes, full.payload_bytes());
    EXPECT_EQ(census.ledger_only.owners, 0u);
    EXPECT_EQ(census.application_only.owners, 0u);
}
TEST(NativeIdentityProjection, ImportedGraphThenFullRetentionDoesNotDoubleBillOrLoseClosure)
{
    if (!G::native_constructor_capacity_profile_supported())
        return;
    auto counter = G::NativeConstructionCounterAccess::create<D::Budget>(cap);
    G::NativeConstructionAdmission admission(counter);
    const auto support = P::details::GeometryIdentityAccess::support_axis_admitted(2, 1.L, admission);
    std::size_t graph_used = 0;
    M::GeometryOwnerManifest manifest;
    {
        P::details::GeometryOwnerGraphEditor graph({}, 0, graph_used, cap);
        graph.identity(support, true);
        manifest = graph.finish();
    }
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    (void) P::details::GeometryOwnerImportAccess::import(ledger, manifest, used, cap);
    const auto payload = ledger.payload_bytes(), owners = ledger.owner_count();
    ASSERT_NO_THROW(ledger.retain(support, used, cap));
    EXPECT_EQ(ledger.payload_bytes(), payload);
    EXPECT_EQ(ledger.owner_count(), owners);
    {
        P::details::GeometryOwnerGraphEditor graph(manifest, 0, graph_used, cap);
        graph.identity(support, false);
        EXPECT_EQ(graph.finish().owner_count(), 0u);
    }
}
TEST(NativeIdentityProjection, TightAdmittedIdentityBudgetIncludesEachConcreteBacking)
{
    ASSERT_TRUE(G::native_constructor_capacity_profile_supported());
    using Access = P::details::GeometryIdentityAccess;
    using OwnerAccess = P::details::NativeIdentityOwnerAccess;
    const G::NativeConstructionAdmission admission(G::NativeConstructionCounterAccess::create<D::Budget>(cap));
    const auto axis = Access::support_axis_admitted(1, .5L, admission);
    auto translated = axis;
    for (unsigned layer = 0; layer != 3; ++layer)
        translated = Access::translated_support_admitted(translated, {1.L, .25L, -2.L}, admission);
    for (const auto& sample : {axis, translated})
    {
        const auto key = P::canonical_key(sample);
        std::size_t depth = 0, logical_bytes = 0;
        for (auto owner = Access::backing(sample); owner; owner = owner->parent)
        {
            const auto* typed = OwnerAccess::backing(owner);
            ASSERT_TRUE(typed);
            std::size_t payload = 0;
            typed->storage().owner->visit_allocations([&](const auto& allocation)
                {
                    if (allocation.address == owner->native.get())
                        payload = allocation.bytes;
                });
            ASSERT_GT(payload, 0u);
            logical_bytes += sizeof(P::details::NativeGeometryIdentityBacking) + payload;
            ++depth;
        }
        // The complete backing path and the outer translation point stack
        // coexist at the final hash. These inputs have no other payload buffers.
        logical_bytes += depth * sizeof(const P::details::GeometryIdentityBacking*) +
            (depth - 1) * sizeof(P::details::OrdinaryIdentityPoint);
        const auto before = admission.statistics().actual_cumulative;
        EXPECT_THROW((void) Access::decode_support_admitted(key, logical_bytes - 1, admission), std::invalid_argument);
        EXPECT_EQ(admission.statistics().actual_cumulative, before);
        EXPECT_EQ(Access::decode_support_admitted(key, logical_bytes, admission), sample);
    }
}
} // namespace
