#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include <gtest/gtest.h>

namespace
{
namespace G = SimpleFluid::geometry_detail;
namespace D = SimpleFluid::Meshes::AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
namespace M = P::details::geometry_demand;
constexpr std::size_t cap = 16 * 1024 * 1024;
TEST(NativePublicationOwnerGraph, WeakFixedPresenceCannotSuppressOrPinStrongSchemaChildren)
{
    auto counter = G::NativeConstructionCounterAccess::create<D::Budget>(cap);
    G::NativeConstructionAdmission admission(counter);
    const std::array<char, 1024> bytes{'s'};
    auto schema = G::NativeSchemaOwnerAccess::copy(bytes, admission);
    auto fixed = schema->allocation();
    const auto state = fixed->state();
    const auto external = P::details::GeometryOwnerGraphEditor::external_kind();
    std::size_t index_used = 0;
    M::GeometryOwnerManifest both;
    {
        P::details::GeometryOwnerGraphEditor graph({}, 0, index_used, cap);
        D::NativePublicationOwnerGraph::fixed(fixed, graph, true, true);
        D::NativePublicationOwnerGraph::schema(schema, graph, true, true);
        both = graph.finish();
    }
    EXPECT_EQ(both.payload_bytes(), counter.owner()->used);
    EXPECT_EQ(both.references(0, external, state), 6u); // five fixed controls plus one live byte representation
    EXPECT_EQ(both.references(0, external, state->counter_owner), 2u);
    EXPECT_EQ(both.references(0, external, fixed), 2u); // independent weak root and semantic object's envelope field
    M::GeometryOwnerManifest weak_only;
    {
        P::details::GeometryOwnerGraphEditor graph(both, 0, index_used, cap);
        D::NativePublicationOwnerGraph::schema(schema, graph, false, true);
        weak_only = graph.finish();
    }
    both = {};
    std::weak_ptr<const G::NativeSchemaOwner> object = schema;
    schema.reset();
    EXPECT_TRUE(object.expired());
    EXPECT_FALSE(fixed->object_alive());
    EXPECT_EQ(weak_only.references(0, external, state), 5u);
    EXPECT_EQ(weak_only.references(0, external, fixed), 1u);
    EXPECT_EQ(weak_only.payload_bytes(), counter.owner()->used);
    EXPECT_EQ(weak_only.payload_bytes(), fixed->fixed_allocation_bytes() + state->allocation_admission_bytes +
                                             state->counter_allocations.object.bytes +
                                             state->counter_allocations.control.bytes);
}
TEST(NativePublicationOwnerGraph, RepeatedSourceAndNativeEdgesShareOneActualFixedAllocation)
{
    auto counter = G::NativeConstructionCounterAccess::create<D::Budget>(cap);
    G::NativeConstructionAdmission admission(counter);
    auto value = G::NativeAllocationEnvelopeAccess::make<std::uint64_t>(admission, 42);
    std::size_t index_used = 0;
    M::GeometryOwnerManifest manifest;
    {
        P::details::GeometryOwnerGraphEditor graph({}, 0, index_used, cap);
        D::NativePublicationOwnerGraph::fixed(value.envelope, graph, true, true);
        D::NativePublicationOwnerGraph::fixed(value.envelope, graph, true, true);
        manifest = graph.finish();
    }
    EXPECT_EQ(manifest.payload_bytes(), counter.owner()->used);
    const auto external = P::details::GeometryOwnerGraphEditor::external_kind();
    EXPECT_EQ(manifest.references(0, external, value.envelope), 2u);
    EXPECT_EQ(manifest.references(0, external, value.envelope->state()), 5u);
}
TEST(NativePublicationOwnerGraph, BorrowedUnreceiptedCounterCannotEnterQualifiedGraph)
{
    auto counter = std::make_shared<D::Budget>(D::Budget{cap});
    G::NativeConstructionAdmission admission(counter);
    auto value = G::NativeAllocationEnvelopeAccess::make<std::uint64_t>(admission, 42);
    std::size_t index_used = 0;
    P::details::GeometryOwnerGraphEditor graph({}, 0, index_used, cap);
    EXPECT_THROW(D::NativePublicationOwnerGraph::fixed(value.envelope, graph, true, true), std::invalid_argument);
    EXPECT_EQ(index_used, 0u);
    EXPECT_THROW((void) graph.finish(), P::GeometryError);
}
} // namespace
