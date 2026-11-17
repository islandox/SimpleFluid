/** @file GeometryRetentionCensus.tcc
 * @brief Private observational provider, compiled once by GeometryAllocationLedger.cc.
 */
#include "geometry/PLIC/details/GeometryAllocationLedgerInternal.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/GeometryIdentityRetention.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/PLIC/details/GeometryRetentionCensusAccess.hh"
#include "geometry/PLIC/details/PointSnapshot.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <new>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace SimpleFluid::PLIC::details::retention
{
namespace
{
using K = allocation_detail::Kind;
constexpr unsigned planar_wrapper = static_cast<unsigned>(K::PlanarWrapper),
                   planar_payload = static_cast<unsigned>(K::PlanarPayload);
struct Overflow
{
};
struct InvalidGraph
{
};
std::size_t census_add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw Overflow{};
    return a + b;
}
std::size_t census_mul(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw Overflow{};
    return a * b;
}
struct CensusKey
{
    unsigned kind;
    const void* pointer;
    ArithmeticMode mode;
    bool operator<(const CensusKey& b) const noexcept
    {
        if (kind != b.kind)
            return kind < b.kind;
        if (pointer != b.pointer)
            return std::less<const void*>{}(pointer, b.pointer);
        return false;
    }
};
struct CensusNode
{
    CensusKey key;
    std::size_t id;
    std::uint64_t groups = 0;
    bool ledger = false, required = false;
};
struct CensusWork
{
    CensusKey key;
    std::uint64_t groups;
    bool ledger, required;
    std::size_t depth = 0;
};
struct RecordKey
{
    const void* owner;
    std::size_t index;
    unsigned kind, role;
    bool operator<(const RecordKey& b) const noexcept
    {
        if (owner != b.owner)
            return std::less<const void*>{}(owner, b.owner);
        if (kind != b.kind)
            return kind < b.kind;
        if (role != b.role)
            return role < b.role;
        return index < b.index;
    }
};
const char* owner_label(unsigned kind)
{
    static constexpr const char* labels[] = {"typed_wrapper", "dynamic_wrapper", "geometry", "topology", "export_cache",
        "dynamic_export", "typed_export", "compact_batch", "plane", "point_context", "native_facts", "moment",
        "ancestry", "split", "certificate", "recipe", "clip", "identity", "opaque_authority", "cartesian_source",
        "cell_catalog", "native_source_body", "external_owner", "cell_shell_page"};
    if (kind < std::size(labels))
        return labels[kind];
    return kind == planar_wrapper ? "planar_wrapper" : "planar_payload";
}
class Census
{
public:
    Census(Report& r, ProbeScratch& s, ProbeLimits l)
        : report(r), limits(l), nodes(&s), work(&s), records(&s), contexts(&s), external_sizes(&s), owner_edges(&s),
          point_edges(&s), colors(&s), depths(&s), frames(&s)
    {
    }
    Report& report;
    ProbeLimits limits;
    std::pmr::map<CensusKey, CensusNode> nodes;
    std::pmr::vector<CensusWork> work;
    std::pmr::map<RecordKey, std::size_t> records;
    std::pmr::map<const PointSnapshotContextBase*, std::size_t, std::less<const PointSnapshotContextBase*>> contexts;
    std::pmr::map<const void*, std::array<std::size_t, 3>, std::less<const void*>> external_sizes;
    using EdgeMap = std::pmr::map<std::pair<std::size_t, std::size_t>, bool>;
    EdgeMap owner_edges, point_edges;
    struct Frame
    {
        std::size_t id;
        EdgeMap::const_iterator next, end;
    };
    std::pmr::vector<unsigned char> colors;
    std::pmr::vector<std::size_t> depths;
    std::pmr::vector<Frame> frames;
    static constexpr auto no_parent = std::numeric_limits<std::size_t>::max();
    std::size_t owner_parent = no_parent, point_parent = no_parent;
    CensusNode& ensure_node(CensusKey key)
    {
        auto found = nodes.find(key);
        if (found != nodes.end())
            return found->second;
        if (nodes.size() >= limits.maximum_owners)
            throw std::bad_alloc{};
        const auto id = report.owners.size();
        OwnerRow row;
        row.id = id;
        row.kind = owner_label(key.kind);
        row.mode = key.mode;
        report.owners.push_back(row);
        return nodes.emplace(key, CensusNode{key, id}).first->second;
    }

    void push(
        unsigned k, const void* p, ArithmeticMode m, std::uint64_t g, bool ledger, bool required, std::size_t depth = 0)
    {
        if (!p)
            return;
        if (k == static_cast<unsigned>(K::PointSnapshot))
            m = static_cast<const PointSnapshotContextBase*>(p)->mode;
        else if (k == static_cast<unsigned>(K::Geometry))
            m = static_cast<const DynamicConvexGeometry::Impl*>(p)->limits.arithmetic_mode;
        else if (k == static_cast<unsigned>(K::Clip))
            m = static_cast<const GeometryClipConstruction*>(p)->mode;
        else if (k == static_cast<unsigned>(K::DynamicWrapper))
            m = static_cast<const DynamicConvexGeometry*>(p)->arithmetic_mode();
        if (k == static_cast<unsigned>(K::IdentityPayload))
        {
            const auto* identity = static_cast<const GeometryIdentityBacking*>(p);
            m = identity->native && identity->native->native_kind() != GeometryIdentityNativeKind::OrdinarySupport
                    ? ArithmeticMode::Exact
                    : ArithmeticMode::Ordinary;
        }
        if (k == planar_payload)
        {
            const auto* face = static_cast<const PlanarFace<ArithmeticMode::Ordinary>*>(p);
            p = face->storage_identity();
            planar_faces.emplace(p, face);
        }
        const CensusKey key{k, p, m};
        if (owner_parent != no_parent)
            owner_edges.emplace(std::pair{owner_parent, ensure_node(key).id}, true);
        work.push_back({key, g, ledger, required, depth});
    }
    void push(K k, const void* p, const CensusWork& w)
    {
        push(static_cast<unsigned>(k), p, w.key.mode, w.groups, w.ledger, w.required, w.depth);
    }
    void authority(const PlaneAuthority& p, const CensusWork& w)
    {
        if (const auto* support = OriginalSupportAuthorityAccess::retained(p))
            push(static_cast<unsigned>(K::IdentityPayload), GeometryIdentityAccess::backing(*support).get(),
                ArithmeticMode::Ordinary, w.groups, w.ledger, w.required, w.depth);
        else
            push(K::AuthorityPayload, RetentionAccess::opaque(p), w);
    }
    void point(PointProvenanceView p, std::uint64_t groups, bool ledger, bool required)
    {
        if (!p)
            return;
        // The normalized view resolves external dependency slots without owning().
        push(static_cast<unsigned>(K::PointSnapshot), p.context(), p.arithmetic_mode(), groups, ledger, required);
        if (!required)
            return;
        const auto ref = p.local_ref();
        const RecordKey key{p.context(), ref.index, 0, static_cast<unsigned>(ref.role)};
        const bool fresh = record(key, 0);
        if (point_parent != no_parent)
            point_edges.emplace(std::pair{point_parent, records.at(key)}, true);
        if (fresh)
            point_work.push_back(p);
    }
    bool record(RecordKey key, std::size_t bytes)
    {
        if (records.contains(key))
            return false;
        if (records.size() >= limits.maximum_records)
            throw std::bad_alloc{};
        records.emplace(key, records.size());
        report.required_record_bytes_lower_bound = census_add(report.required_record_bytes_lower_bound, bytes);
        return true;
    }
    void require_context(const PointSnapshotContextBase* context)
    {
        // Comparing the address is safe even when a broken manifest has left a
        // borrowed plane context without an owner. Never inspect that pointee.
        const auto found = nodes.find({static_cast<unsigned>(K::PointSnapshot), context, ArithmeticMode::Ordinary});
        if (found == nodes.end() || (!found->second.groups && !found->second.ledger))
        {
            report.uncovered_required_owners = census_add(report.uncovered_required_owners, 1);
            throw InvalidGraph{};
        }
    }
    template<ArithmeticMode M>
    void incidence(const PointSnapshotContext<M>* context, LocalAuthorityRange<M> range, const CensusWork& w)
    {
        if (!context && !range.empty())
            throw InvalidGraph{};
        if (!context)
            return;
        require_context(context);
        push(static_cast<unsigned>(K::PointSnapshot), context, M, 0, false, true);
        auto& row = context_row(context);
        for (std::size_t i = 0; i < range.size(); ++i)
        {
            const auto slot = census_add(range.first(), i);
            const auto authority_index = context->incidence.at(slot).index();
            if (record({context, slot, 1, 0}, sizeof(PrivateAuthorityRef<M>)))
                row.required_incidence = census_add(row.required_incidence, 1);
            if (record({context, authority_index, 2, 0}, sizeof(PlaneAuthority)))
            {
                row.required_authorities = census_add(row.required_authorities, 1);
                authority(context->authority_table.at(authority_index), w);
            }
        }
    }
    template<ArithmeticMode M> void required_point(const PointSnapshotContext<M>& context, DynamicStoredPointRef ref)
    {
        auto& summary = context_row(&context);
        summary.required_records.at(static_cast<unsigned>(ref.role)) =
            census_add(summary.required_records.at(static_cast<unsigned>(ref.role)), 1);
        report.required_records = census_add(report.required_records, 1);
        CensusWork w{{static_cast<unsigned>(K::PointSnapshot), &context, M}, 0, false, true};
        std::size_t bytes = sizeof(PointCoordinates<M>);
        LocalAuthorityRange<M> supports;
        switch (ref.role)
        {
            case PointProvenanceRole::Native: {
                const auto& row = context.native_refs.at(ref.index);
                bytes = sizeof(row);
                supports = row.incidence;
                const auto* facts = context.native_fact_blocks.at(row.fact_block).get();
                push(K::NativeFactBlock, facts, w);
                (void) facts->points.at(row.fact_index);
                if (record({&context, row.fact_index, 6, row.fact_block}, 0))
                    summary.required_fact_rows = census_add(summary.required_fact_rows, 1);
                record({facts, row.fact_index, 3, 0}, sizeof(PointCoordinates<M>));
                for (std::size_t i = 0; i < facts->witnesses.size(); ++i)
                    if (facts->witnesses[i].index == row.fact_index)
                        record({facts, i, 4, 0}, sizeof(PointErrorWitness));
                break;
            }
            case PointProvenanceRole::Generated: {
                const auto& row = context.generated_recipes.at(ref.index);
                supports = row.incidence;
                bytes = census_add(bytes, sizeof(row));
                for (std::size_t i = 0; i < row.edges.count; ++i)
                {
                    const auto slot = census_add(row.edges.first, i);
                    const auto& edge = context.edges.at(slot);
                    if (record({&context, slot, 5, 0}, sizeof(StoredPointEdge<M>)))
                        summary.required_edges = census_add(summary.required_edges, 1);
                    for (const auto endpoint : edge)
                        point(resolve_point(&context, endpoint), 0, false, true);
                }
                break;
            }
            case PointProvenanceRole::Export: {
                const auto& row = context.export_aliases.at(ref.index);
                supports = row.incidence;
                bytes = census_add(bytes, sizeof(row));
                point(resolve_point(&context, row.source), 0, false, true);
                break;
            }
            case PointProvenanceRole::Numerical:
                break;
        }
        for (std::size_t i = 0; i < context.witnesses.size(); ++i)
            if (context.witnesses[i].role == ref.role && context.witnesses[i].index == ref.index)
                record({&context, i, 4, 0}, sizeof(PointErrorWitness));
        report.required_record_bytes_lower_bound = census_add(report.required_record_bytes_lower_bound, bytes);
        incidence(&context, supports, w);
    }
    template<ArithmeticMode M> ContextRow& context_row(const PointSnapshotContext<M>* p)
    {
        auto [it, inserted] = contexts.emplace(p, report.contexts.size());
        if (inserted)
        {
            ContextRow row;
            row.mode = M;
            row.retained_records = {p->native_refs.size(), p->generated_recipes.size(), p->export_aliases.size(),
                p->numerical_points.size()};
            row.authorities = p->authority_table.size();
            row.incidence = p->incidence.size();
            row.edges = p->edges.size();
            for (const auto& block : p->native_fact_blocks)
                row.fact_rows = census_add(row.fact_rows, block->points.size());
            report.contexts.push_back(row);
        }
        return report.contexts.at(it->second);
    }
    void account(CensusNode& n, std::size_t metadata, std::size_t populated, std::size_t unused)
    {
        auto& row = report.owners.at(n.id);
        row.metadata_bytes = metadata;
        row.populated_bytes = populated;
        row.unused_capacity_bytes = unused;
        row.bytes = census_add(metadata, census_add(populated, unused));
    }
    template<class T> void table(const std::vector<T>& values, std::size_t& populated, std::size_t& unused)
    {
        populated = census_add(populated, census_mul(values.size(), sizeof(T)));
        unused = census_add(unused, census_mul(values.capacity() - values.size(), sizeof(T)));
    }
    template<ArithmeticMode M>
    void table(const FrozenVertexBuffer<M>& values, std::size_t& populated, std::size_t& unused)
    {
        if (values.borrowed())
            return; // The owning geometry/body edge accounts for this storage.
        populated = census_add(populated, census_mul(values.size(), sizeof(FrozenVertex<M>)));
        unused = census_add(unused, census_mul(values.owned_capacity() - values.size(), sizeof(FrozenVertex<M>)));
    }
    template<ArithmeticMode M> void visit(CensusNode& n, const CensusWork& w)
    {
        using namespace SimpleFluid::PLIC::details;
        const auto p = w.key.pointer;
        std::size_t metadata = 0, populated = 0, unused = 0;
        const auto k = w.key.kind;
        if (k == planar_wrapper)
        {
            const auto& face = *static_cast<const PlanarFace<ArithmeticMode::Ordinary>*>(p);
            metadata = sizeof(face);
            push(planar_payload, &face, ArithmeticMode::Ordinary, w.groups, w.ledger, w.required);
        }
        else if (k == planar_payload)
        {
            // The key is normalized to impl storage below. The object borrow is
            // kept in a separate lookup because its implementation stays opaque.
            const auto& face = *planar_faces.at(p);
            metadata = RetentionAccess::planar_metadata_bytes();
            table(face.points(), populated, unused);
            if (const auto* support = RetentionAccess::planar_support_borrowed(face))
                push(static_cast<unsigned>(K::IdentityPayload), GeometryIdentityAccess::backing(*support).get(),
                    ArithmeticMode::Ordinary, w.groups, w.ledger, w.required);
        }
        else
            switch (static_cast<K>(k))
            {
                case K::TypedWrapper: {
                    const auto& g = *static_cast<const ConvexCellGeometry<M>*>(p);
                    metadata = sizeof(g);
                    push(K::DynamicWrapper, g.dynamic_geometry().get(), w);
                    break;
                }
                case K::DynamicWrapper: {
                    const auto& g = *static_cast<const DynamicConvexGeometry*>(p);
                    metadata = sizeof(g);
                    push(K::Geometry, DynamicGeometryAccess::implementation(g).get(), w);
                    break;
                }
                case K::Geometry: {
                    const auto& g =
                        *static_cast<const GeometryStorage<M>*>(static_cast<const DynamicConvexGeometry::Impl*>(p));
                    metadata = sizeof(g);
                    if (!g.vertices.borrowed())
                        table(g.vertices, populated, unused);
                    if (const auto* owner =
                            std::get_if<std::shared_ptr<const DynamicConvexGeometry::Impl>>(&g.vertex_owner))
                        push(K::Geometry, owner->get(), w);
                    else if (const auto* owner =
                                 std::get_if<std::shared_ptr<const NativeGeometryBody>>(&g.vertex_owner))
                        push(K::NativeSourceBody, owner->get(), w);
                    push(K::Topology, g.topology.get(), w);
                    push(K::PointSnapshot, g.point_context.get(), w);
                    push(K::ExportCache, g.point_export_cache.get(), w);
                    push(K::Geometry, g.export_source.get(), w);
                    push(K::Recipe, g.native_export_source.get(), w);
                    push(K::Moment, g.moment_cache.get(), w);
                    push(K::Ancestry, g.ancestry.get(), w);
                    push(K::Certificate, g.partition_certificate.get(), w);
                    push(K::Recipe, g.construction.get(), w);
                    if (w.required)
                        for (const auto& vertex : g.vertices)
                        {
                            incidence(g.point_context.get(), vertex.authorities, w);
                            point(resolve_point(g.point_context.get(), vertex.source_identity), 0, false, true);
                        }
                    break;
                }
                case K::ExternalOwner: {
                    const auto bytes = external_sizes.at(p);
                    metadata = bytes[0];
                    populated = bytes[1];
                    unused = bytes[2];
                    break;
                }
                case K::NativeSourceBody: {
                    const auto& body = *static_cast<const NativeGeometryBody*>(p);
                    metadata = sizeof(body);
                    table(body.vertices, populated, unused);
                    push(K::Topology, body.topology.get(), w);
                    push(K::PointSnapshot, body.point_context.get(), w);
                    push(K::ExportCache, body.export_cache.get(), w);
                    push(K::Ancestry, body.ancestry.get(), w);
                    if (w.required)
                        for (const auto& vertex : body.vertices)
                        {
                            incidence(body.point_context.get(), vertex.authorities, w);
                            point(resolve_point(body.point_context.get(), vertex.source_identity), 0, false, true);
                        }
                    break;
                }
                case K::Topology: {
                    const auto& t = *static_cast<const FrozenTopology<M>*>(p);
                    metadata = sizeof(t);
                    table(t.planes, populated, unused);
                    std::visit(
                        [&](const auto& a)
                        {
                            table(a.face_offsets, populated, unused);
                            table(a.face_vertices, populated, unused);
                            table(a.incident_offsets, populated, unused);
                            table(a.incident_planes, populated, unused);
                            table(a.surface_planes, populated, unused);
                            table(a.surface_orientations, populated, unused);
                        },
                        t.arrays);
                    for (const auto& plane : t.planes)
                        push(K::Plane, plane.get(), w);
                    break;
                }
                case K::ExportCache: {
                    const auto& cache = *static_cast<const PointExportCache*>(p);
                    metadata = sizeof(cache);
                    table(cache.vertex_provenance, populated, unused);
                    break;
                }
                case K::DynamicPointExport: {
                    const auto& payload = *static_cast<const DynamicPointExportPayload*>(p);
                    metadata = sizeof(payload);
                    table(payload.points, populated, unused);
                    push(K::PointSnapshot, payload.point_context.get(), w);
                    for (const auto& value : payload.points)
                        point(RetentionAccess::provenance(value), w.groups, w.ledger, w.required);
                    break;
                }
                case K::TypedPointExport: {
                    const auto& payload = *static_cast<const PointExportPayload<M>*>(p);
                    metadata = sizeof(payload);
                    table(payload.points, populated, unused);
                    push(K::PointSnapshot, payload.point_context.get(), w);
                    for (const auto& value : payload.points)
                        point(RetentionAccess::provenance(value), w.groups, w.ledger, w.required);
                    break;
                }
                case K::CompactPointBatch: {
                    const auto& payload = *static_cast<const PointSnapshotPayload<M>*>(p);
                    metadata = sizeof(payload);
                    table(payload.refs, populated, unused);
                    push(K::PointSnapshot, payload.context.get(), w);
                    if (w.required)
                        for (const auto ref : payload.refs)
                            point(resolve_point(payload.context.get(), ref), 0, false, true);
                    break;
                }
                case K::Plane: {
                    const auto& plane = *static_cast<const StoredPlane<M>*>(p);
                    metadata = sizeof(plane);
                    table(plane.source_vertices, populated, unused);
                    // point_context is borrowed. It never contributes a strong edge.
                    if (w.required)
                    {
                        if (plane.point_context)
                            require_context(plane.point_context);
                        incidence(plane.point_context, plane.authorities, w);
                        for (const auto ref : plane.source_vertices)
                            point(resolve_point(plane.point_context, ref), 0, false, true);
                    }
                    if constexpr (M == ArithmeticMode::Ordinary)
                        if (plane.native_support)
                            push(K::IdentityPayload, GeometryIdentityAccess::backing(*plane.native_support).get(), w);
                    break;
                }
                case K::PointSnapshot: {
                    const auto& context =
                        *static_cast<const PointSnapshotContext<M>*>(static_cast<const PointSnapshotContextBase*>(p));
                    metadata = sizeof(context);
                    table(context.dependencies, populated, unused);
                    table(context.native_fact_blocks, populated, unused);
                    table(context.native_refs, populated, unused);
                    table(context.generated_recipes, populated, unused);
                    table(context.export_aliases, populated, unused);
                    table(context.generated_points, populated, unused);
                    table(context.export_points, populated, unused);
                    table(context.numerical_points, populated, unused);
                    table(context.witnesses, populated, unused);
                    table(context.authority_table, populated, unused);
                    table(context.incidence, populated, unused);
                    table(context.edges, populated, unused);
                    context_row(&context).owner_id = n.id;
                    // Logical necessity is record-specific; capacity and unrelated
                    // dependency tables are retained physically but not all required.
                    if (!w.required)
                    {
                        for (const auto& dependency : context.dependencies)
                            push(K::PointSnapshot, dependency.get(), w);
                        for (const auto& block : context.native_fact_blocks)
                            push(K::NativeFactBlock, block.get(), w);
                        for (const auto& a : context.authority_table)
                            authority(a, w);
                    }
                    break;
                }
                case K::NativeFactBlock: {
                    const auto& b = *static_cast<const NativePointFactBlock<M>*>(p);
                    metadata = sizeof(b);
                    table(b.points, populated, unused);
                    table(b.witnesses, populated, unused);
                    break;
                }
                case K::Moment:
                    metadata = sizeof(StoredMomentCache<M>);
                    break;
                case K::Ancestry: {
                    const auto& a = *static_cast<const Ancestry*>(p);
                    metadata = sizeof(a);
                    report.owners.at(n.id).depth = a.depth;
                    push(K::Ancestry, a.parent.get(), w);
                    push(K::Split, a.split.get(), w);
                    break;
                }
                case K::Split: {
                    const auto& s = *static_cast<const Split*>(p);
                    metadata = sizeof(s);
                    push(K::Ancestry, s.parent.get(), w);
                    break;
                }
                case K::Certificate: {
                    const auto& c = *static_cast<const PartitionCertificate*>(p);
                    metadata = sizeof(c);
                    table(c.members, populated, unused);
                    push(K::Ancestry, c.parent.get(), w);
                    break;
                }
                case K::Recipe: {
                    const auto& r = *static_cast<const GeometryConstruction*>(p);
                    metadata = sizeof(r);
                    report.maximum_recipe_depth = std::max(report.maximum_recipe_depth, w.depth);
                    report.owners.at(n.id).depth = std::max(report.owners.at(n.id).depth, w.depth);
                    report.owners.at(n.id).recipe_kind = r.kind() == GeometryConstruction::Kind::Source ? "source"
                                                         : r.kind() == GeometryConstruction::Kind::Clip ? "clip"
                                                                                                        : "rebase";
                    auto child = w;
                    child.depth = census_add(w.depth, 1);
                    if (r.kind() == GeometryConstruction::Kind::Source)
                    {
                        if (const auto* native = r.source().native())
                        {
                            metadata = census_add(metadata, sizeof(NativeGeometryConstruction));
                            push(K::CartesianSource, native->source.get(), child);
                            push(K::NativeSourceBody, native->body.get(), child);
                            push(K::Moment, native->moments.get(), child);
                        }
                        else
                            push(K::Geometry, r.source().geometry().get(), child);
                    }
                    else if (r.kind() == GeometryConstruction::Kind::Clip)
                        push(K::Clip, r.clip().clip.get(), child);
                    else
                    {
                        metadata = census_add(metadata, sizeof(GeometryConstruction::Rebase));
                        push(K::Recipe, r.rebase().parent.get(), child);
                    }
                    break;
                }
                case K::Clip: {
                    const auto& c = *static_cast<const GeometryClipConstruction*>(p);
                    metadata = sizeof(c);
                    table(c.retained_points, populated, unused);
                    push(K::Recipe, c.parent.get(), w);
                    push(K::Recipe, c.native_plane_source.get(), w);
                    push(K::Plane, c.plane.get(), w);
                    push(K::PointSnapshot, c.point_context.get(), w);
                    if (w.required)
                        for (const auto ref : c.retained_points)
                            point(resolve_point(c.point_context.get(), ref), 0, false, true);
                    break;
                }
                case K::IdentityPayload: {
                    const auto& identity = *static_cast<const GeometryIdentityBacking*>(p);
                    metadata = sizeof(identity);
                    if (identity.native)
                    {
                        const auto native =
                            identity.native->native_kind() == GeometryIdentityNativeKind::OrdinarySupport
                                ? ordinary_identity_retention_bytes(*identity.native)
                                : exact_identity_retention_bytes(*identity.native);
                        if (native.overflow)
                            throw Overflow{};
                        if (!native.complete)
                            throw InvalidGraph{};
                        metadata = census_add(metadata, native.metadata);
                        populated = native.populated;
                        unused = native.unused;
                    }
                    report.maximum_identity_depth = std::max(report.maximum_identity_depth, identity.depth);
                    report.owners.at(n.id).depth = identity.depth;
                    push(K::IdentityPayload, identity.parent.get(), w);
                    break;
                }
                case K::CellCatalog: {
                    const auto& catalog = *static_cast<const CellGeometryCatalog<M>*>(p);
                    metadata = sizeof(catalog);
                    if (RetentionAccess::catalog_persistent(catalog))
                    {
                        namespace G = SimpleFluid::PLIC::details::geometry_demand;
                        struct Context
                        {
                            Census& census;
                            const CensusWork& work;
                            std::size_t catalog_parent;
                        } context{*this, w, n.id};
                        RetentionAccess::visit_catalog_dependencies(catalog,
                            {&context,
                                [](void* raw, const void* owner, const void* parent, std::size_t bytes)
                                {
                                    auto& value = *static_cast<Context*>(raw);
                                    auto& census = value.census;
                                    const auto [entry, inserted] =
                                        census.external_sizes.emplace(owner, std::array<std::size_t, 3>{bytes, 0, 0});
                                    if (!inserted && entry->second != std::array<std::size_t, 3>{bytes, 0, 0})
                                        throw InvalidGraph{};
                                    const auto saved = census.owner_parent;
                                    census.owner_parent =
                                        parent ? census
                                                     .ensure_node({static_cast<unsigned>(K::ExternalOwner), parent,
                                                         value.work.key.mode})
                                                     .id
                                               : value.catalog_parent;
                                    census.push(K::ExternalOwner, owner, value.work);
                                    census.owner_parent = saved;
                                    return true;
                                }},
                            {&context, [](void* raw, const void* parent, G::SourceKind kind, const void* owner)
                                {
                                    auto& value = *static_cast<Context*>(raw);
                                    auto& census = value.census;
                                    const auto saved = census.owner_parent;
                                    census.owner_parent = census
                                                              .ensure_node({static_cast<unsigned>(K::ExternalOwner),
                                                                  parent, value.work.key.mode})
                                                              .id;
                                    census.push(
                                        kind == G::SourceKind::Cartesian ? K::CartesianSource : K::CellShellPage, owner,
                                        value.work);
                                    census.owner_parent = saved;
                                }});
                        break;
                    }
                    table(RetentionAccess::catalog_pages(catalog), populated, unused);
                    table(RetentionAccess::catalog_selection(catalog), populated, unused);
                    for (const auto& page : RetentionAccess::catalog_selection(catalog))
                        table(page.irregular, populated, unused);
                    table(RetentionAccess::catalog_sources(catalog), populated, unused);
                    if constexpr (M == ArithmeticMode::Ordinary)
                        for (const auto& source : RetentionAccess::catalog_sources(catalog))
                            push(K::CartesianSource, source.get(), w);
                    for (const auto& page : RetentionAccess::catalog_pages(catalog))
                        push(K::CellShellPage, page.get(), w);
                    break;
                }
                case K::CellShellPage: {
                    const auto& page = *static_cast<const typename CellGeometryCatalog<M>::ExplicitPage*>(p);
                    metadata = sizeof(page);
                    table(page.cells, populated, unused);
                    for (const auto& geometry : page.cells)
                        push(K::TypedWrapper, geometry.get(), w);
                    break;
                }
                case K::CartesianSource: {
                    const auto& source = *static_cast<const CartesianCellSource*>(p);
                    const auto bytes = RetentionAccess::cartesian_bytes(source);
                    metadata = bytes[0];
                    populated = bytes[1];
                    unused = bytes[2];
                    const auto base = NativeCellGeometrySourceAccess::base_owner(source);
                    external_sizes.insert_or_assign(base.get(), NativeCellGeometrySourceAccess::base_bytes(source));
                    push(K::ExternalOwner, base.get(), w);
                    for (const auto& support : RetentionAccess::cartesian_supports(source))
                        push(K::IdentityPayload, GeometryIdentityAccess::backing(support).get(), w);
                    break;
                }
                case K::AuthorityPayload: {
                    const auto& a = *static_cast<const std::vector<std::uint64_t>*>(p);
                    metadata = sizeof(a);
                    table(a, populated, unused);
                    break;
                }
                case K::PlanarWrapper:
                case K::PlanarPayload:
                    throw InvalidGraph{}; // Opaque planar owners use the normalizing branches above.
            }
        account(n, metadata, populated, unused);
    }
    std::pmr::map<const void*, const PlanarFace<ArithmeticMode::Ordinary>*, std::less<const void*>> planar_faces{
        nodes.get_allocator().resource()};
    void drain()
    {
        while (!work.empty() || !point_work.empty())
        {
            if (!point_work.empty())
            {
                const auto p = point_work.back();
                point_work.pop_back();
                const auto ref = p.local_ref();
                point_parent = records.at({p.context(), ref.index, 0, static_cast<unsigned>(ref.role)});
                snapshot_detail::dispatch(p.context(), [&](const auto& context) { required_point(context, ref); });
                point_parent = no_parent;
                continue;
            }
            auto w = work.back();
            work.pop_back();
            auto& node = ensure_node(w.key);
            const auto new_groups = w.groups & ~node.groups;
            const bool new_ledger = w.ledger && !node.ledger, new_required = w.required && !node.required;
            if (!new_groups && !new_ledger && !new_required)
                continue;
            node.groups |= new_groups;
            node.ledger |= new_ledger;
            node.required |= new_required;
            auto& row = report.owners.at(node.id);
            row.root_groups = node.groups;
            row.application = bool(node.groups);
            row.ledger = node.ledger;
            row.required = node.required;
            w.groups = new_groups;
            w.ledger = new_ledger;
            w.required = new_required;
            // Traverse physical and logical edges separately so a requirement
            // cannot accidentally make borrowed plane contexts strong owners.
            if (new_groups || new_ledger)
            {
                auto physical = w;
                physical.required = false;
                owner_parent = node.id;
                dispatch(node, physical);
                owner_parent = no_parent;
            }
            if (new_required)
            {
                auto logical = w;
                logical.groups = 0;
                logical.ledger = false;
                logical.required = true;
                dispatch(node, logical);
            }
        }
    }
    void dispatch(CensusNode& node, const CensusWork& w)
    {
        if (w.key.mode == ArithmeticMode::Ordinary)
            visit<ArithmeticMode::Ordinary>(node, w);
        else if (w.key.mode == ArithmeticMode::BoundedCertification)
            visit<ArithmeticMode::BoundedCertification>(node, w);
        else if (w.key.kind == static_cast<unsigned>(K::IdentityPayload))
            visit<ArithmeticMode::Ordinary>(node, w);
        else
            throw InvalidGraph{};
    }
    std::pmr::vector<PointProvenanceView> point_work{nodes.get_allocator().resource()};

    template<class Include> std::size_t dependency_depth(const EdgeMap& edges, std::size_t count, Include include)
    {
        colors.assign(count, 0);
        depths.assign(count, 0);
        frames.clear();
        const auto frame = [&](std::size_t id)
        { return Frame{id, edges.lower_bound({id, 0}), edges.upper_bound({id, no_parent})}; };
        std::size_t maximum = 0;
        for (std::size_t start = 0; start < count; ++start)
        {
            if (colors[start])
                continue;
            colors[start] = 1;
            frames.push_back(frame(start));
            while (!frames.empty())
            {
                auto& current = frames.back();
                if (current.next == current.end)
                {
                    const auto id = current.id;
                    maximum = std::max(maximum, depths[id]);
                    colors[id] = 2;
                    frames.pop_back();
                    if (!frames.empty())
                        depths[frames.back().id] = std::max(depths[frames.back().id], census_add(depths[id], 1));
                    continue;
                }
                const auto child = (current.next++)->first.second;
                if (!include(current.id, child))
                    continue;
                if (colors[child] == 1)
                    throw InvalidGraph{};
                if (colors[child] == 2)
                {
                    depths[current.id] = std::max(depths[current.id], census_add(depths[child], 1));
                    continue;
                }
                colors[child] = 1;
                frames.push_back(frame(child));
            }
        }
        return maximum;
    }
    void validate_owner_dependencies()
    {
        const auto count = report.owners.size();
        report.maximum_owner_dependency_depth = dependency_depth(owner_edges, count, [](auto, auto) { return true; });
        for (std::size_t i = 0; i < count; ++i)
            report.owners[i].dependency_depth = depths[i];
        const auto kind = [&](std::size_t id) { return std::string_view(report.owners[id].kind); };
        report.maximum_context_dependency_depth = dependency_depth(owner_edges, count,
            [&](auto from, auto to) { return kind(from) == "point_context" && kind(to) == "point_context"; });
        report.maximum_identity_dependency_depth = dependency_depth(
            owner_edges, count, [&](auto from, auto to) { return kind(from) == "identity" && kind(to) == "identity"; });
        report.maximum_ancestry_dependency_depth = dependency_depth(owner_edges, count,
            [&](auto from, auto to)
            {
                const auto ancestry = [&](auto id) { return kind(id) == "ancestry" || kind(id) == "split"; };
                return ancestry(from) && ancestry(to);
            });
        const auto replay = [&](auto id)
        { return kind(id) == "geometry" || kind(id) == "recipe" || kind(id) == "clip"; };
        report.maximum_recipe_depth =
            dependency_depth(owner_edges, count, [&](auto from, auto to) { return replay(from) && replay(to); });
        for (std::size_t i = 0; i < count; ++i)
            if (replay(i))
                report.owners[i].depth = depths[i];
    }
    void validate_point_dependencies()
    {
        report.maximum_required_point_dependency_depth =
            dependency_depth(point_edges, records.size(), [](auto, auto) { return true; });
    }
    void root_input(const BorrowedRoot& r)
    {
        if (r.group >= 63)
            throw InvalidGraph{};
        auto found =
            std::find_if(report.groups.begin(), report.groups.end(), [&](const auto& g) { return g.group == r.group; });
        if (found == report.groups.end())
        {
            report.groups.push_back({r.group});
            found = std::prev(report.groups.end());
        }
        found->roots = census_add(found->roots, 1);
        const auto mask = std::uint64_t(1) << r.group;
        root_one(r, mask, false);
        logical_roots.push_back(r);
    }
    void root_one(const BorrowedRoot& r, std::uint64_t mask, bool required)
    {
        if (!r.object)
            return;
        CensusWork w{{0, r.object, r.mode}, mask, false, required};
        switch (r.kind)
        {
            case RootKind::TypedGeometry:
                push(K::TypedWrapper, r.object, w);
                break;
            case RootKind::DynamicGeometry:
                push(K::DynamicWrapper, r.object, w);
                break;
            case RootKind::DynamicPoint:
                point(RetentionAccess::provenance(*static_cast<const DynamicPoint*>(r.object)), mask, false, required);
                break;
            case RootKind::TypedPoint:
                if (r.mode == ArithmeticMode::Ordinary)
                    point(RetentionAccess::provenance(*static_cast<const Point<ArithmeticMode::Ordinary>*>(r.object)),
                        mask, false, required);
                else if (r.mode == ArithmeticMode::BoundedCertification)
                    point(RetentionAccess::provenance(
                              *static_cast<const Point<ArithmeticMode::BoundedCertification>*>(r.object)),
                        mask, false, required);
                else
                    throw InvalidGraph{};
                break;
            case RootKind::Snapshot:
                if (r.mode == ArithmeticMode::Ordinary)
                    push(K::CompactPointBatch,
                        RetentionAccess::payload(
                            *static_cast<const PointSnapshot<ArithmeticMode::Ordinary>*>(r.object)),
                        w);
                else if (r.mode == ArithmeticMode::BoundedCertification)
                    push(K::CompactPointBatch,
                        RetentionAccess::payload(
                            *static_cast<const PointSnapshot<ArithmeticMode::BoundedCertification>*>(r.object)),
                        w);
                else
                    throw InvalidGraph{};
                break;
            case RootKind::TypedExport:
                if (r.mode == ArithmeticMode::Ordinary)
                    push(K::TypedPointExport,
                        RetentionAccess::payload(*static_cast<const PointExport<ArithmeticMode::Ordinary>*>(r.object)),
                        w);
                else if (r.mode == ArithmeticMode::BoundedCertification)
                    push(K::TypedPointExport,
                        RetentionAccess::payload(
                            *static_cast<const PointExport<ArithmeticMode::BoundedCertification>*>(r.object)),
                        w);
                else
                    throw InvalidGraph{};
                break;
            case RootKind::DynamicExport: {
                const auto* p = RetentionAccess::payload(*static_cast<const DynamicPointExport*>(r.object));
                if (p)
                {
                    w.key.mode = p->mode;
                    push(K::DynamicPointExport, p, w);
                }
                break;
            }
            case RootKind::Context: {
                const auto* p = static_cast<const PointSnapshotContextBase*>(r.object);
                push(K::PointSnapshot, p, w);
                // An explicitly declared whole-context root declares every row.
                if (required)
                    snapshot_detail::dispatch(p,
                        [&](const auto& context)
                        {
                            const std::array counts{context.native_refs.size(), context.generated_recipes.size(),
                                context.export_aliases.size(), context.numerical_points.size()};
                            for (unsigned role = 0; role < counts.size(); ++role)
                                for (std::size_t i = 0; i < counts[role]; ++i)
                                    point(resolve_point(p, {0, static_cast<std::uint32_t>(i),
                                                               static_cast<PointProvenanceRole>(role)}),
                                        0, false, true);
                        });
                break;
            }
            case RootKind::OrdinaryPlanarFace:
                push(planar_wrapper, r.object, ArithmeticMode::Ordinary, mask, false, required);
                break;
            case RootKind::IdentityBacking:
                push(K::IdentityPayload, r.object, w);
                break;
            case RootKind::Authority:
                authority(*static_cast<const PlaneAuthority*>(r.object), w);
                break;
            case RootKind::CellCatalog:
                push(K::CellCatalog, r.object, w);
                break;
            case RootKind::ExternalOwner:
                external_sizes.emplace(r.object, std::array<std::size_t, 3>{r.exclusive_bytes, 0, 0});
                push(K::ExternalOwner, r.object, w);
                break;
            case RootKind::CartesianSource:
                push(K::CartesianSource, r.object, w);
                break;
            case RootKind::OrdinaryBoundary: {
                const auto& b = *static_cast<const OrdinaryBoundary*>(r.object);
                push(K::TypedWrapper, RetentionAccess::boundary_geometry(b), w);
                authority(RetentionAccess::boundary_authority(b), w);
                break;
            }
        }
    }
    std::pmr::vector<BorrowedRoot> logical_roots{nodes.get_allocator().resource()};
    void finish()
    {
        auto add_count = [](ByteCount& value, std::size_t bytes)
        {
            value.owners = census_add(value.owners, 1);
            value.bytes = census_add(value.bytes, bytes);
        };
        for (const auto& row : report.owners)
        {
            if (row.application || row.ledger)
                add_count(report.owner_union, row.bytes);
            if (row.application)
                report.application_bytes = census_add(report.application_bytes, row.bytes);
            if (row.ledger)
                report.ledger_bytes = census_add(report.ledger_bytes, row.bytes);
            if (row.application && row.ledger)
                add_count(report.intersection, row.bytes);
            else if (row.application)
                add_count(report.application_only, row.bytes);
            else if (row.ledger)
            {
                add_count(report.ledger_only, row.bytes);
                add_count(row.required ? report.required_ledger_only : report.unneeded_ledger_only, row.bytes);
            }
            else if (row.required)
                report.uncovered_required_owners = census_add(report.uncovered_required_owners, 1);
            if (std::popcount(row.root_groups) > 1)
                report.group_overlap_bytes = census_add(report.group_overlap_bytes, row.bytes);
            for (auto& group : report.groups)
                if (row.root_groups & (std::uint64_t(1) << group.group))
                {
                    group.inclusive_bytes = census_add(group.inclusive_bytes, row.bytes);
                    if (std::popcount(row.root_groups) == 1)
                        group.exclusive_bytes = census_add(group.exclusive_bytes, row.bytes);
                }
        }
        for (const auto& context : report.contexts)
            for (const auto count : context.retained_records)
                report.retained_records = census_add(report.retained_records, count);
    }
};
} // namespace
void* ProbeScratch::do_allocate(std::size_t bytes, std::size_t alignment)
{
    if (bytes_.empty())
        throw std::bad_alloc{};
    void* pointer = bytes_.data() + used_;
    auto available = bytes_.size() - used_;
    if (!std::align(alignment, bytes, pointer, available))
        throw std::bad_alloc{};
    used_ = census_add(static_cast<std::byte*>(pointer) - bytes_.data(), bytes);
    return pointer;
}
BorrowedRoot root(const PointSnapshotContextBase& p, std::uint32_t group)
{
    return {RootKind::Context, &p, p.mode, group};
}
Report inspect_retention(std::span<const BorrowedRoot> roots, std::span<const GeometryAllocationLedger* const> ledgers,
    ProbeScratch& scratch, ProbeLimits limits) noexcept
{
    Report report(scratch);
    report.complete_root_manifest = limits.complete_root_manifest;
    report.root_count = roots.size();
    report.ledger_count = ledgers.size();
    report.probe_capacity_bytes = scratch.capacity_bytes();
    try
    {
        Census census(report, scratch, limits);
        for (const auto& r : roots)
            census.root_input(r);
        std::pmr::vector<const GeometryAllocationLedger*> seen(&scratch);
        for (const auto* ledger : ledgers)
        {
            if (!ledger || std::find(seen.begin(), seen.end(), ledger) != seen.end())
                continue;
            seen.push_back(ledger);
            if (RetentionAccess::failed(*ledger))
                report.failed_ledgers = census_add(report.failed_ledgers, 1);
            const auto* impl = RetentionAccess::ledger(*ledger);
            if (!impl)
                continue;
            report.ledger_payload_admission = census_add(report.ledger_payload_admission, impl->payload);
            report.ledger_bookkeeping_admission = census_add(report.ledger_bookkeeping_admission, impl->bookkeeping);
            auto tracking =
                census_add(sizeof(*impl), census_mul(impl->pending.capacity(), sizeof(allocation_detail::Item)));
            tracking = census_add(tracking, census_mul(impl->additional_subsets.capacity(),
                                                sizeof(GeometryAllocationLedger::Impl::SubsetCertificate)));
            tracking =
                census_add(tracking, census_mul(impl->owners.size(),
                                         sizeof(typename decltype(impl->owners)::value_type) + 4 * sizeof(void*)));
            tracking = census_add(
                tracking, census_mul(impl->metadata_credits.size(),
                              sizeof(typename decltype(impl->metadata_credits)::value_type) + 4 * sizeof(void*)));
            report.ledger_tracking_bytes = census_add(report.ledger_tracking_bytes, tracking);
            const auto mode = impl->mode.value_or(ArithmeticMode::Ordinary);
            for (const auto& [key, value] : impl->owners)
            {
                // Polygon payload pins name opaque implementation storage.
                // Its retained face wrapper supplies the normalizing borrow
                // and publishes the same physical payload key when visited.
                if (key.kind == K::PlanarPayload)
                    continue;
                if (key.kind == K::ExternalOwner)
                    census.external_sizes.emplace(
                        value.pin.get(), std::array<std::size_t, 3>{value.payload_bytes, 0, 0});
                census.push(static_cast<unsigned>(key.kind), value.pin.get(), mode, 0, true, false);
            }
            if (impl->imported)
            {
                report.ledger_imported_owners =
                    census_add(report.ledger_imported_owners, impl->imported->owner_count());
                impl->imported->visit(
                    [&](const geometry_demand::OwnerRecord& row)
                    {
                        const auto imported_kind = static_cast<unsigned>(row.key.kind);
                        if (imported_kind == static_cast<unsigned>(K::ExternalOwner))
                            census.external_sizes.emplace(row.owner.get(),
                                std::array<std::size_t, 3>{row.exclusive_payload_bytes(), 0, 0});
                        census.push(imported_kind, row.owner.get(), mode, 0, true, false);
                    });
            }
            report.pending_pins = census_add(report.pending_pins, impl->pending.size());
            for (const auto& pending : impl->pending)
                census.push(static_cast<unsigned>(pending.kind), pending.owner.get(), mode, 0, true, false);
        }
        census.drain();
        for (const auto& r : census.logical_roots)
            census.root_one(r, 0, true);
        census.drain();
        census.validate_owner_dependencies();
        census.validate_point_dependencies();
        census.finish();
    }
    catch (const std::bad_alloc&)
    {
        report.status = Status::ScratchExhausted;
        report.detail = "probe scratch or count limit exhausted; totals unavailable";
    }
    catch (const Overflow&)
    {
        report.status = Status::IntegerOverflow;
        report.detail = "probe integer count overflow; totals unavailable";
    }
    catch (...)
    {
        report.status = Status::InvalidGraph;
        report.detail = "invalid or uncovered borrowed graph; totals unavailable";
    }
    report.probe_used_bytes = scratch.used_bytes();
    return report;
}
} // namespace SimpleFluid::PLIC::details::retention
