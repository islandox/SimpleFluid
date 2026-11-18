#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/NativeDescriptorCodecAccess.hh"
#include <array>
#include <gtest/gtest.h>
#include <span>

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

TEST(NativePublicationOwnerGraph, FlatFragmentDescriptorPinDoesNotSuppressRecipeParts)
{
    auto descriptor = std::make_shared<SimpleFluid::Meshes::NativeFragmentDescriptor>();
    const auto part = std::make_shared<const SimpleFluid::Meshes::NativeFragmentBytes>(99, 's');
    descriptor->parts.push_back(part);
    const auto descriptor_bytes =
        sizeof(*descriptor) + descriptor->view.capacity() + descriptor->parts.capacity() * sizeof(descriptor->parts[0]);
    const auto part_bytes = sizeof(*part) + part->capacity();
    using Editor = P::details::GeometryOwnerGraphEditor;
    const auto kind = Editor::external_kind();
    std::size_t used = 0;
    Editor build({}, 0, used, cap);
    build.external(descriptor, descriptor_bytes, true);
    D::NativePublicationOwnerGraph::descriptor(descriptor, build, true);
    const auto both = build.finish();
    EXPECT_EQ(both.references(0, kind, descriptor), 2u);
    EXPECT_TRUE(both.admits(0, kind, part, part_bytes));

    Editor index_retire(both, 0, used, cap);
    EXPECT_FALSE(index_retire.external(descriptor, descriptor_bytes, false));
    const auto typed = index_retire.finish();
    EXPECT_EQ(typed.references(0, kind, descriptor), 1u);
    EXPECT_EQ(typed.references(0, kind, part), 1u);

    Editor typed_retire(typed, 0, used, cap);
    D::NativePublicationOwnerGraph::descriptor(descriptor, typed_retire, false);
    EXPECT_EQ(typed_retire.finish().owner_count(), 0u);
}

TEST(NativePublicationOwnerGraph, AdmittedAdaptivePartsRetireIndependentlyOfFlatComponentPins)
{
    namespace A = SimpleFluid::Meshes::AMR;
    namespace Mesh = SimpleFluid::Meshes;
    using Editor = P::details::GeometryOwnerGraphEditor;
    using Bytes = G::NativeObjectOwnerAccess<A::AdaptiveBytes>;
    using Factor = G::NativeObjectOwnerAccess<A::AdaptiveFactorRecipe>;
    using Descriptor = G::NativeObjectOwnerAccess<Mesh::NativeFragmentDescriptor>;
    auto counter = G::NativeConstructionCounterAccess::create<D::Budget>(cap);
    const G::NativeConstructionAdmission admission(counter);
    const auto legacy = std::make_shared<Mesh::OrthogonalCartesian3D>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{{.1, .3, .5}, {-1, 2}, {3, 5}}});
    Mesh::NativeFragmentCache cache;
    const auto source = Mesh::extract_native_fragment(Mesh::native_region("legacy_static", legacy), cache);
    const auto descriptor = Mesh::NativeDescriptorCodecAccess::decode(
        Mesh::encode_native_fragment_descriptor(*source), admission);
    ASSERT_EQ(descriptor->parts.size(), 1u);
    const auto& root = descriptor->parts.front();
    const std::array<char, 512> axis_bytes{'z'};
    const auto axis = Bytes::make(admission);
    Bytes::storage(axis.owner).owner->copy(*axis.owner, std::span<const char>(axis_bytes));
    const auto factor = Factor::make_without_buffers(admission);
    factor.owner->root = root;
    factor.owner->base = root;
    factor.owner->axis = axis.owner;
    A::AdaptiveLeaf leaf;
    leaf.native_recipe = descriptor;
    leaf.factor_recipe = factor.owner;
    leaf.join_recipe = root;

    std::size_t used = 0;
    Editor build_component({}, 0, used, cap);
    D::NativeSourceOwnerStore::leaf(leaf, build_component, true);
    const auto component = build_component.finish();
    const auto kind = Editor::external_kind();
    ASSERT_EQ(component.payload_bytes(), counter.owner()->used);
    EXPECT_EQ(component.references(0, kind, root), 4u);
    EXPECT_EQ(component.references(0, kind, Bytes::envelope(root)), 1u);
    EXPECT_EQ(component.references(0, kind, Bytes::storage(root).owner), 1u);
    EXPECT_EQ(component.references(0, kind, factor.owner), 1u);
    EXPECT_EQ(component.references(0, kind, descriptor), 1u);
    EXPECT_EQ(component.references(0, kind, Descriptor::storage(descriptor).owner), 1u);

    const auto project_flat = [&](Editor& graph, bool retain)
    {
        struct Projection { Editor& graph; bool retain; } projection{graph, retain};
        component.visit_records({&projection, [](void* raw, const M::OwnerRecord& row)
            {
                const auto& p = *static_cast<const Projection*>(raw);
                p.graph.owner(row, p.retain);
            }});
    };
    Editor build_both({}, 0, used, cap);
    project_flat(build_both, true);
    D::NativeSourceOwnerStore::leaf(leaf, build_both, true);
    const auto both = build_both.finish();
    EXPECT_EQ(both.payload_bytes(), component.payload_bytes());

    for (const bool flat_first : {true, false})
    {
        SCOPED_TRACE(flat_first);
        Editor first(both, 0, used, cap);
        if (flat_first)
            project_flat(first, false);
        else
            D::NativeSourceOwnerStore::leaf(leaf, first, false);
        const auto survivor = first.finish();
        EXPECT_EQ(survivor.payload_bytes(), component.payload_bytes());
        EXPECT_EQ(survivor.owner_count(), component.owner_count());
        struct Comparison { const M::GeometryOwnerManifest& survivor; bool component_left; }
            comparison{survivor, !flat_first};
        component.visit_records({&comparison, [](void* raw, const M::OwnerRecord& row)
            {
                const auto& c = *static_cast<const Comparison*>(raw);
                EXPECT_EQ(c.survivor.references(row.key.channel, row.key.kind, row.owner),
                    c.component_left ? 1u : row.references);
            }});

        Editor second(survivor, 0, used, cap);
        if (flat_first)
            D::NativeSourceOwnerStore::leaf(leaf, second, false);
        else
            project_flat(second, false);
        EXPECT_EQ(second.finish().owner_count(), 0u);
    }
}
