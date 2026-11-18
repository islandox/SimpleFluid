#include "geometry/mesh/AMR/NativePublicationOwnerGraph.hh"
#include "geometry/details/NativeAllocationGraph.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
NativePublicationOwnerGraph::Context::Context(PLIC::details::GeometryOwnerGraphEditor& editor, bool add,
    bool paid, PLIC::details::NativeGeometryAccounting* live)
    : graph(editor), retain(add), prepaid(paid), accounting(live)
{
    sink.context = this;
    sink.owner_edge = [](void* raw, const auto& owner, geometry_detail::NativeConstructorOwnerKind, std::size_t bytes)
    {
        auto& c = *static_cast<Context*>(raw);
        return c.graph.external(owner, bytes, c.retain, c.prepaid);
    };
    sink.branch_edge = [](void* raw, const auto& owner, geometry_detail::NativeConstructorOwnerKind, std::size_t bytes)
    {
        auto& c = *static_cast<Context*>(raw);
        return c.graph.external_with_children(owner, bytes, c.retain, c.prepaid);
    };
    sink.fixed_edge = [](void* raw, const auto& envelope)
    {
        auto& c = *static_cast<Context*>(raw);
        NativePublicationOwnerGraph::fixed(envelope, c.graph, c.retain, c.prepaid);
    };
    sink.state_edge = [](void* raw, const auto& state)
    {
        auto& c = *static_cast<Context*>(raw);
        NativePublicationOwnerGraph::state(state, c.graph, c.retain, c.prepaid);
    };
    sink.semantic_receipt = [](void* raw, const auto& owner, const auto& envelope)
    {
        auto& c = *static_cast<Context*>(raw);
        if (c.accounting)
            c.accounting->authorize_semantic(0, PLIC::details::GeometryOwnerGraphEditor::external_kind(), owner, envelope);
    };
    sink.schema_receipt = [](void* raw, const auto& schema)
    {
        auto& c = *static_cast<Context*>(raw);
        if (c.accounting) c.accounting->authorize_schema(schema);
    };
    sink.storage_receipt = [](void* raw, const auto& storage)
    {
        auto& c = *static_cast<Context*>(raw);
        if (c.accounting)
            c.accounting->authorize_storage(storage.owner, storage.envelope);
        c.graph.native_storage(storage);
    };
}
void NativePublicationOwnerGraph::Context::identity(
    void* raw, const PLIC::PlaneSupportIdentity<ArithmeticMode::Ordinary>& identity)
{
    auto& c = *static_cast<Context*>(raw);
    c.graph.identity(identity, c.retain);
}
void NativePublicationOwnerGraph::Context::geometry(
    void* raw, const PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary>& geometry)
{
    auto& c = *static_cast<Context*>(raw);
    c.graph.geometry(geometry, c.retain);
}
void NativePublicationOwnerGraph::Context::coverage(
    void* raw, const std::shared_ptr<const NativeSourceSubsetCoverage>& coverage)
{
    auto& c = *static_cast<Context*>(raw);
    NativeSourceOwnerStore::record_subset(coverage, c.graph, c.retain);
    NativeSourceOwnerStore::project_subset(coverage, c.graph, c.retain, c.accounting);
}
void NativePublicationOwnerGraph::descriptor(const std::shared_ptr<const NativeFragmentDescriptor>& descriptor,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, bool prepaid,
    PLIC::details::NativeGeometryAccounting* accounting)
{
    Context context(graph, retain, prepaid, accounting);
    NativeFragmentAccess::visit_descriptor(descriptor, context.sink, Context::identity, Context::geometry, Context::coverage);
}
void NativePublicationOwnerGraph::fragment(const std::shared_ptr<const NativeFragmentRecord>& record,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, bool prepaid,
    PLIC::details::NativeGeometryAccounting* accounting)
{
    Context context(graph, retain, prepaid, accounting);
    NativeFragmentAccess::visit(record, context.sink, Context::identity, Context::geometry, Context::coverage);
}
void NativePublicationOwnerGraph::validate_state(
    const std::shared_ptr<const geometry_detail::NativeAllocationState>& state)
{
    (void) geometry_detail::NativeAllocationGraphAccess::validate_state(state);
}
void NativePublicationOwnerGraph::state(const std::shared_ptr<const geometry_detail::NativeAllocationState>& state,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, bool prepaid)
{
    graph.native_state(state, retain, prepaid);
}
void NativePublicationOwnerGraph::fixed(const std::shared_ptr<const geometry_detail::NativeAllocationEnvelope>& owner,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, bool prepaid)
{
    graph.native_fixed(owner, retain, prepaid);
}
void NativePublicationOwnerGraph::schema(const std::shared_ptr<const geometry_detail::NativeSchemaOwner>& owner,
    PLIC::details::GeometryOwnerGraphEditor& graph, bool retain, bool prepaid)
{
    if (!owner)
        return;
    const auto& allocation = owner->allocation();
    if (!allocation || !allocation->matches(owner))
        throw std::invalid_argument("Native schema projection has a foreign fixed allocation receipt.");
    validate_state(allocation->state());
    (void) allocation->fixed_allocation_bytes();
    if (graph.external(owner, 0, retain, prepaid))
    {
        fixed(allocation, graph, retain, prepaid);
        // Both the admitted-vector and adopted-input representation hold one
        // direct admission state for the buffer's complete lifetime.
        state(allocation->state(), graph, retain, prepaid);
        if (owner->capacity_bytes())
            graph.external(
                std::shared_ptr<const void>(owner, owner->span().data()), owner->capacity_bytes(), retain, prepaid);
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
