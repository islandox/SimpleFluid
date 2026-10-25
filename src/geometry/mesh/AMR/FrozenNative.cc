/** @file FrozenNative.cc
 * @brief Accepted-source detachment, bounded record transport and ownership-only plans.
 */
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/FrozenNativeDetails.hh"
#include "geometry/mesh/AMR/MixedAMR.hh"
#include "geometry/mesh/AMR/PolyhedralAMR.hh"
#include "geometry/mesh/SparseNativeMesh.hh"
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using namespace adaptive_detail;
using ID = std::uint64_t;
void real(Writer& w, double x)
{
    w.word(std::bit_cast<ID>(x));
}
double real(Reader& r)
{
    return std::bit_cast<double>(r.word());
}
void point(Writer& w, PLIC::GeometryPoint p)
{
    for (auto x : p)
        real(w, x);
}
PLIC::GeometryPoint point(Reader& r)
{
    PLIC::GeometryPoint p;
    for (auto& x : p)
        x = real(r);
    return p;
}
PLIC::GeometryPoint point(MeshUtils::Vec3 p)
{
    return {p.x, p.y, p.z};
}
void ids(Writer& w, const std::vector<ID>& values)
{
    w.word(values.size());
    for (auto n : values)
        w.word(n);
}
std::vector<ID> ids(Reader& r)
{
    auto n = r.word();
    if (n > (r.data.size() - r.offset) / 8)
        throw std::invalid_argument("Frozen ordinal list exceeds packet.");
    std::vector<ID> result;
    for (ID i = 0; i < n; ++i)
        result.push_back(r.word());
    return result;
}
void names(Writer& w, const std::map<int, std::string>& values)
{
    w.word(values.size());
    for (const auto& [id, name] : values)
    {
        w.word(ID(id));
        w.bytes(std::span(name));
    }
}
std::map<int, std::string> names(Reader& r)
{
    auto n = r.word();
    if (n > (r.data.size() - r.offset) / 16)
        throw std::invalid_argument("Frozen boundary list exceeds packet.");
    std::map<int, std::string> result;
    for (ID i = 0; i < n; ++i)
    {
        int id = int(r.word());
        auto b = r.bytes();
        if (!result.emplace(id, std::string(b.begin(), b.end())).second)
            throw std::invalid_argument("Duplicate frozen boundary.");
    }
    return result;
}
bool source_unchanged(
    const AdaptiveHierarchy& h, const MultiRegionMesh& canonical, const HierarchyObservation& observation)
{
    if (canonical.sparse_native() || canonical.geometry_epoch() != observation.geometry)
        return false;
    if (auto p = dynamic_cast<const CartesianAMR*>(&h))
        return p->geometry_unchanged();
    if (auto p = dynamic_cast<const ExtrudedAMR*>(&h))
        return p->geometry_unchanged();
    if (auto p = dynamic_cast<const PolyhedralAMR*>(&h))
        return p->geometry_unchanged();
    if (auto p = dynamic_cast<const MixedAMR*>(&h))
        return p->geometry_unchanged();
    return observation.native_geometry == &canonical;
}
ID transaction(const Teuchos::Comm<int>& comm)
{
    static std::atomic<ID> next{0};
    ID id = 0;
    if (comm.getRank() == 0)
        id = ++next;
    Teuchos::broadcast(comm, 0, 1, &id);
    if (!id)
        throw std::overflow_error("Frozen transaction exhausted.");
    return id;
}
void agree(std::span<const char> bytes, const Teuchos::Comm<int>& comm, std::size_t budget)
{
    int n = comm.getRank() == 0 ? int(bytes.size()) : 0;
    collective(comm,
        [&]
        {
            if (bytes.size() > budget || bytes.size() > std::size_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument("Frozen common schema exceeds budget.");
        });
    Teuchos::broadcast(comm, 0, 1, &n);
    AdaptiveBytes root;
    collective(comm,
        [&]
        {
            if (n < 0 || std::size_t(n) > budget)
                throw std::invalid_argument("Frozen schema receiver budget exhausted.");
            root.resize(n);
            if (comm.getRank() == 0)
                std::copy(bytes.begin(), bytes.end(), root.begin());
        });
    Teuchos::broadcast(comm, 0, n, root.data());
    collective(comm,
        [&]
        {
            if (!std::equal(root.begin(), root.end(), bytes.begin(), bytes.end()))
                throw std::invalid_argument("Frozen source schema differs across ranks.");
        });
}
} // namespace
AdaptiveBytes encode_frozen_part(const FrozenPart& p)
{
    Writer w;
    w.word(p.source);
    w.word(p.node_prefix);
    w.word(bool(p.procedural));
    if (p.procedural)
    {
        w.bytes(encode_native_fragment_descriptor(*p.procedural));
        return std::move(w.data);
    }
    w.word(p.layout.cells);
    w.word(p.layout.faces);
    w.word(p.layout.nodes);
    names(w, p.boundaries);
    w.word(p.cells.size());
    for (const auto& c : p.cells)
    {
        w.word(c.id);
        w.word(ID(c.type));
        ids(w, c.faces);
        ids(w, c.nodes);
        point(w, point(c.centroid));
        real(w, c.volume);
    }
    w.word(p.faces.size());
    for (const auto& f : p.faces)
    {
        w.word(f.id);
        w.word(f.owner);
        w.word(f.neighbor);
        ids(w, f.nodes);
        point(w, point(f.centroid));
        point(w, point(f.area_vector));
        w.word(ID(f.boundary));
    }
    w.word(p.nodes.size());
    for (const auto& n : p.nodes)
    {
        w.word(n.id);
        point(w, point(n.point));
    }
    return std::move(w.data);
}
std::shared_ptr<const FrozenPart> decode_frozen_part(std::span<const char> bytes)
{
    Reader r{bytes};
    auto p = std::make_shared<FrozenPart>();
    p->source = r.word();
    p->node_prefix = r.word();
    auto kind = r.word();
    if (kind > 1)
        throw std::invalid_argument("Invalid frozen provider kind.");
    if (kind)
    {
        NativeFragmentCache cache;
        p->procedural = decode_native_fragment_descriptor(r.bytes(), cache);
    }
    else
    {
        p->layout = {RegionLayout::Family::Explicit, r.word(), r.word(), r.word(), {}};
        p->boundaries = names(r);
        auto n = r.word();
        if (n > bytes.size() / 48)
            throw std::invalid_argument("Frozen cells exceed packet.");
        for (ID i = 0; i < n; ++i)
        {
            ExplicitRegionProvider::Cell c;
            c.id = r.word();
            c.type = MeshUtils::CellType(r.word());
            c.faces = ids(r);
            c.nodes = ids(r);
            c.centroid = point(r);
            c.volume = real(r);
            p->cells.push_back(std::move(c));
        }
        n = r.word();
        if (n > bytes.size() / 64)
            throw std::invalid_argument("Frozen faces exceed packet.");
        for (ID i = 0; i < n; ++i)
        {
            ExplicitRegionProvider::Face f;
            f.id = r.word();
            f.owner = r.word();
            f.neighbor = r.word();
            f.nodes = ids(r);
            f.centroid = point(r);
            f.area_vector = point(r);
            f.boundary = int(r.word());
            p->faces.push_back(std::move(f));
        }
        n = r.word();
        if (n > bytes.size() / 32)
            throw std::invalid_argument("Frozen nodes exceed packet.");
        for (ID i = 0; i < n; ++i)
        {
            auto id = r.word();
            p->nodes.push_back({id, point(r)});
        }
    }
    r.done();
    if (encode_frozen_part(*p) != AdaptiveBytes(bytes.begin(), bytes.end()))
        throw std::invalid_argument("Noncanonical frozen provider record.");
    return p;
}
namespace
{
using BundleData = FrozenNativeBundle::Data;
AdaptiveBytes encode_batch(std::span<const std::shared_ptr<const BundleData>> bundles, std::size_t limit)
{
    Budget budget{limit};
    budget.add(32);
    std::map<AdaptiveBytes, ID> explicit_dictionary, chunk_dictionary;
    std::map<ID, std::pair<ID, std::shared_ptr<const NativeFragmentDescriptor>>> procedural_dictionary;
    std::vector<AdaptiveBytes> parts, chunks;
    std::map<ID, ID> prefixes;
    std::map<const FrozenPart*, ID> part_addresses;
    auto part_id = [&](const std::shared_ptr<const FrozenPart>& p) -> ID
    {
        if (!p)
            return std::numeric_limits<ID>::max();
        if (auto cached = part_addresses.find(p.get()); cached != part_addresses.end())
            return cached->second;
        budget.add(sizeof(std::pair<const FrozenPart* const, ID>) + 4 * sizeof(void*));
        auto [prefix, added_prefix] = prefixes.emplace(p->source, p->node_prefix);
        if (!added_prefix && prefix->second != p->node_prefix)
            throw std::invalid_argument("Frozen source node prefixes disagree.");
        if (p->procedural)
        {
            auto existing = procedural_dictionary.find(p->source);
            if (existing != procedural_dictionary.end())
            {
                const auto& old = existing->second.second;
                const auto& current = p->procedural;
                if (old.get() != current.get())
                {
                    if (old->kind != current->kind || old->view != current->view ||
                        old->parts.size() != current->parts.size())
                        throw std::invalid_argument("Frozen shared procedural source differs.");
                    for (std::size_t i = 0; i < old->parts.size(); ++i)
                        if (*old->parts[i] != *current->parts[i])
                            throw std::invalid_argument("Frozen shared native part differs.");
                }
                part_addresses.emplace(p.get(), existing->second.first);
                return existing->second.first;
            }
            budget.add(2 * p->procedural->view.size() + p->procedural->parts.size() * 16 + 64);
            Writer w;
            w.word(p->source);
            w.word(p->node_prefix);
            w.word(1);
            w.word(ID(p->procedural->kind));
            w.bytes(p->procedural->view);
            w.word(p->procedural->parts.size());
            for (const auto& chunk : p->procedural->parts)
            {
                if (!chunk)
                    throw std::invalid_argument("Frozen procedural chunk missing.");
                auto found = chunk_dictionary.find(*chunk);
                if (found == chunk_dictionary.end())
                {
                    budget.add(3 * chunk->size() + sizeof(AdaptiveBytes) + 64);
                    const ID index = chunks.size();
                    found = chunk_dictionary.emplace(*chunk, index).first;
                    chunks.push_back(*chunk);
                }
                w.word(found->second);
            }
            budget.add(w.data.size() + sizeof(FrozenPart) + 64);
            const ID id = parts.size();
            parts.push_back(std::move(w.data));
            procedural_dictionary.emplace(p->source, std::pair{id, p->procedural});
            part_addresses.emplace(p.get(), id);
            return id;
        }
        auto encoded = encode_frozen_part(*p);
        auto found = explicit_dictionary.find(encoded);
        if (found != explicit_dictionary.end())
        {
            part_addresses.emplace(p.get(), found->second);
            return found->second;
        }
        budget.add(3 * encoded.size() + sizeof(FrozenPart) + 64);
        const ID id = parts.size();
        explicit_dictionary.emplace(encoded, id);
        Writer w;
        w.word(p->source);
        w.word(p->node_prefix);
        w.word(0);
        w.bytes(encoded);
        parts.push_back(std::move(w.data));
        part_addresses.emplace(p.get(), id);
        return id;
    };
    std::vector<AdaptiveBytes> records;
    for (const auto& b : bundles)
    {
        std::size_t estimate = 128 + encode_adaptive_key(b->key).size() + b->rows.size() * 16;
        for (const auto& row : b->rows)
            for (const auto& [face, sign] : row.faces)
                estimate += 112 + encode_adaptive_key(face->first).size() +
                            (face->second ? encode_adaptive_key(*face->second).size() : 0);
        budget.add(2 * estimate + sizeof(BundleData));
        Writer w;
        w.bytes(encode_adaptive_key(b->key));
        w.word(b->source_uid);
        w.word(b->revision);
        w.word(b->material);
        w.word(b->source_gid);
        w.word(b->native);
        w.word(part_id(b->part));
        w.word(bool(b->metric));
        if (b->metric)
        {
            real(w, b->metric->volume);
            point(w, b->metric->center);
        }
        w.word(b->rows.size());
        for (const auto& row : b->rows)
        {
            w.word(row.native);
            w.word(row.faces.size());
            for (const auto& [f, sign] : row.faces)
            {
                w.word(sign == 1);
                w.word(f->gid);
                w.word(f->native);
                w.word(part_id(f->part));
                w.bytes(encode_adaptive_key(f->first));
                w.word(bool(f->second));
                if (f->second)
                    w.bytes(encode_adaptive_key(*f->second));
                point(w, f->second_image);
                w.word(ID(f->boundary));
            }
        }
        records.push_back(std::move(w.data));
    }
    Writer result;
    result.word(chunks.size());
    for (const auto& chunk : chunks)
        result.bytes(chunk);
    result.word(parts.size());
    for (const auto& p : parts)
        result.bytes(p);
    result.word(records.size());
    for (const auto& record : records)
        result.bytes(record);
    if (result.data.size() > limit)
        throw std::invalid_argument("Frozen batch wire budget exhausted.");
    return std::move(result.data);
}
std::vector<std::shared_ptr<const BundleData>> decode_batch(std::span<const char> bytes, std::size_t limit)
{
    Budget budget{limit};
    budget.add(bytes.size());
    Reader r{bytes};
    auto n = r.word();
    if (n > bytes.size() / 8)
        throw std::invalid_argument("Frozen chunk dictionary exceeds packet.");
    std::vector<std::shared_ptr<const AdaptiveBytes>> chunks;
    for (ID i = 0; i < n; ++i)
    {
        auto chunk = r.bytes();
        budget.add(chunk.size() + sizeof(AdaptiveBytes));
        chunks.push_back(std::make_shared<const AdaptiveBytes>(std::move(chunk)));
    }
    n = r.word();
    if (n > bytes.size() / 8)
        throw std::invalid_argument("Frozen part dictionary exceeds packet.");
    std::vector<std::shared_ptr<const FrozenPart>> parts;
    for (ID i = 0; i < n; ++i)
    {
        auto record = r.bytes();
        budget.add(record.size() + sizeof(FrozenPart));
        Reader q{record};
        auto source = q.word(), prefix = q.word(), kind = q.word();
        if (kind > 1)
            throw std::invalid_argument("Invalid frozen wire provider kind.");
        if (!kind)
        {
            auto p = decode_frozen_part(q.bytes());
            if (p->source != source || p->node_prefix != prefix || p->procedural)
                throw std::invalid_argument("Frozen explicit dictionary provenance differs.");
            parts.push_back(std::move(p));
        }
        else
        {
            auto p = std::make_shared<FrozenPart>();
            p->source = source;
            p->node_prefix = prefix;
            auto descriptor = std::make_shared<NativeFragmentDescriptor>();
            auto tag = q.word();
            if (tag > 2)
                throw std::invalid_argument("Frozen native descriptor kind invalid.");
            descriptor->kind = NativeFragmentDescriptor::Kind(tag);
            descriptor->view = q.bytes();
            auto count = q.word();
            if (count > record.size() / 8)
                throw std::invalid_argument("Frozen chunk references exceed packet.");
            for (ID j = 0; j < count; ++j)
            {
                auto index = q.word();
                if (index >= chunks.size())
                    throw std::invalid_argument("Frozen chunk reference missing.");
                descriptor->parts.push_back(chunks[index]);
            }
            p->procedural = std::move(descriptor);
            parts.push_back(std::move(p));
        }
        q.done();
    }
    auto part = [&](ID i) -> std::shared_ptr<const FrozenPart>
    {
        if (i == std::numeric_limits<ID>::max())
            return {};
        if (i >= parts.size())
            throw std::invalid_argument("Invalid frozen part reference.");
        return parts[i];
    };
    n = r.word();
    if (n > bytes.size() / 8)
        throw std::invalid_argument("Frozen bundle count exceeds packet.");
    std::vector<std::shared_ptr<const BundleData>> result;
    std::map<ID, std::shared_ptr<const FrozenFace>> canonical_faces;
    for (ID i = 0; i < n; ++i)
    {
        auto record = r.bytes();
        budget.add(record.size() + sizeof(BundleData));
        Reader q{record};
        auto b = std::make_shared<BundleData>();
        b->key = decode_adaptive_key(q.bytes());
        b->source_uid = q.word();
        b->revision = q.word();
        b->material = q.word();
        b->source_gid = q.word();
        b->native = q.word();
        b->part = part(q.word());
        const auto metric = q.word();
        if (metric > 1)
            throw std::invalid_argument("Invalid frozen metric route kind.");
        if (metric)
            b->metric = FrozenMetric{real(q), point(q)};
        auto count = q.word();
        if (count > record.size() / 16)
            throw std::invalid_argument("Frozen cell rows exceed packet.");
        for (ID j = 0; j < count; ++j)
        {
            FrozenRow row;
            row.native = q.word();
            auto faces = q.word();
            if (faces > record.size() / 128)
                throw std::invalid_argument("Frozen row faces exceed packet.");
            for (ID k = 0; k < faces; ++k)
            {
                auto sign = q.word();
                if (sign > 1)
                    throw std::invalid_argument("Invalid frozen orientation.");
                auto f = std::make_shared<FrozenFace>();
                f->gid = q.word();
                f->native = q.word();
                f->part = part(q.word());
                f->first = decode_adaptive_key(q.bytes());
                auto second = q.word();
                if (second > 1)
                    throw std::invalid_argument("Invalid frozen incidence.");
                if (second)
                    f->second = decode_adaptive_key(q.bytes());
                f->second_image = point(q);
                f->boundary = int(q.word());
                if (!f->part)
                    throw std::invalid_argument("Frozen face lacks native provider.");
                auto [shared, added] = canonical_faces.emplace(f->gid, f);
                if (!added)
                {
                    const auto& old = *shared->second;
                    if (old.native != f->native || old.part->source != f->part->source ||
                        old.part->node_prefix != f->part->node_prefix || old.first != f->first ||
                        old.second != f->second || old.second_image != f->second_image || old.boundary != f->boundary)
                        throw std::invalid_argument("Frozen shared canonical routes disagree.");
                }
                row.faces.emplace_back(shared->second, sign ? 1 : -1);
            }
            b->rows.push_back(std::move(row));
        }
        q.done();
        result.push_back(std::move(b));
    }
    r.done();
    return result;
}
} // namespace
AdaptiveBytes encode_frozen_batch(std::span<const std::shared_ptr<const BundleData>> b, std::size_t limit)
{
    return encode_batch(b, limit);
}
const MultiRegionMesh::Region& frozen_provider(
    const std::shared_ptr<const FrozenPart>& part, FrozenEvaluationCache& cache)
{
    if (!part)
        throw std::invalid_argument("Frozen native cell lacks provider.");
    auto found = cache.providers.find(part.get());
    if (found != cache.providers.end())
        return found->second;
    if (part->procedural)
        return cache.providers.emplace(part.get(), materialize_native_fragment(*part->procedural, cache.fragments))
            .first->second;
    auto provider = std::make_shared<const ExplicitRegionProvider>(
        part->layout, part->cells, part->faces, part->nodes, part->boundaries);
    return cache.providers.emplace(part.get(), DistributedExplicitRegion("frozen_evaluation", provider, provider))
        .first->second;
}
PLIC::GeometryPoint frozen_transform(const FrozenNativeRoot::Data& root, MeshUtils::Vec3 p)
{
    const auto scale = (root.frame[1] - root.frame[0]) / (root.frame[1] - root.frame[0]);
    p.z = root.frame[0] + (p.z - root.frame[0]) * scale;
    return {p.x, p.y, p.z};
}
FrozenMetric frozen_metric(
    const FrozenNativeBundle::Data& bundle, const FrozenNativeRoot::Data& root, FrozenEvaluationCache& cache)
{
    if (bundle.metric)
    {
        if (bundle.part || !bundle.rows.empty())
            throw std::invalid_argument("Frozen metric-only reply has native topology.");
        return *bundle.metric;
    }
    const auto& region = frozen_provider(bundle.part, cache);
    return std::visit(
        [&](const auto& r)
        {
            const auto scale = (root.frame[1] - root.frame[0]) / (root.frame[1] - root.frame[0]);
            return FrozenMetric{scale * r.geometry().cell_volume(bundle.native),
                frozen_transform(root, r.geometry().cell_centroid(bundle.native))};
        },
        region);
}
class FrozenNativeAccess
{
public:
    static FrozenNativeBootstrap detach(const AdaptiveHierarchy& hierarchy, std::span<const ID> selected,
        std::span<const char> schema, Teuchos::RCP<const Teuchos::Comm<int>> comm, FrozenNativeOptions options)
    {
        const auto observation = hierarchy.observation();
        auto source = hierarchy.native_geometry();
        FrozenNativeBootstrap result;
        std::shared_ptr<FrozenNativeRoot::Data> root;
        NativeFragmentCache cache;
        std::map<ID, std::shared_ptr<const FrozenPart>> procedural;
        collective(*comm,
            [&]
            {
                if (!source || !observation.identity || !source_unchanged(hierarchy, *source, observation) ||
                    observation.geometry == std::numeric_limits<ID>::max() ||
                    source->d_axial_edges != source->d_reference_axial_edges || source->d_axial_edges.size() != 2 ||
                    selected.size() > options.maximum_owned)
                    throw std::invalid_argument(
                        "Frozen native export requires an unchanged accepted source and an identity axial frame.");
                root = std::make_shared<FrozenNativeRoot::Data>();
                root->observation = {observation.identity, observation.topology_identity,
                    observation.partition_identity, observation.topology, observation.partition, observation.geometry,
                    observation.native_geometry, observation.communicator};
                root->global_cells = source->num_cells();
                root->global_faces = source->num_faces();
                root->global_nodes = source->num_nodes();
                root->frame = {source->d_reference_axial_edges[0], source->d_reference_axial_edges[1]};
                for (auto b : source->boundary_batch_ids())
                    root->boundaries.emplace(b, source->boundary_batch_name(b));
            });
        root->uid = transaction(*comm);
        Writer common;
        collective(*comm,
            [&]
            {
                common.word(root->uid);
                common.word(root->global_cells);
                common.word(root->global_faces);
                common.word(root->global_nodes);
                common.word(observation.topology);
                common.word(observation.partition);
                common.word(observation.geometry);
                real(common, root->frame[0]);
                real(common, root->frame[1]);
                names(common, root->boundaries);
                common.bytes(schema);
                root->schema = common.data;
            });
        agree(root->schema, *comm, options.byte_budget);
        auto part = [&](ID region, std::optional<ID> cell, std::optional<ID> face,
                        std::optional<ID> node) -> std::shared_ptr<const FrozenPart>
        {
            const auto& native = source->d_regions.at(region);
            const auto layout = std::visit([](const auto& r) { return r.layout(); }, native);
            if (layout.family != RegionLayout::Family::Explicit)
            {
                auto found = procedural.find(region);
                if (found != procedural.end())
                    return found->second;
                auto p = std::make_shared<FrozenPart>();
                p->source = region;
                p->node_prefix = source->d_nodes.at(region);
                p->procedural = extract_native_fragment(native, cache);
                procedural.emplace(region, p);
                return p;
            }
            auto p = std::make_shared<FrozenPart>();
            p->source = region;
            p->node_prefix = source->d_nodes.at(region);
            p->layout = layout;
            std::set<ID> full, faces, nodes, cells;
            if (cell)
            {
                full.insert(*cell);
                cells.insert(*cell);
            }
            if (face)
                faces.insert(*face);
            if (node)
                nodes.insert(*node);
            std::visit(
                [&](const auto& r)
                {
                    for (auto c : full)
                    {
                        for (auto f : r.topology().cell_faces(c))
                            faces.insert(f);
                        for (auto n : r.topology().cell_nodes(c))
                            nodes.insert(n);
                    }
                    for (auto f : faces)
                    {
                        auto a = r.topology().owner_cell(f), b = r.topology().neighbor_cell(f);
                        cells.insert(a);
                        if (b != MultiRegionMesh::invalid_cell_id())
                            cells.insert(b);
                        for (auto n : r.topology().face_nodes(f))
                            nodes.insert(n);
                        ExplicitRegionProvider::Face x;
                        x.id = f;
                        x.owner = a;
                        x.neighbor = b;
                        x.boundary = r.topology().boundary_id(f);
                        x.centroid = r.geometry().face_centroid(f);
                        x.area_vector = r.geometry().face_area_vector(f);
                        for (auto n : r.topology().face_nodes(f))
                            x.nodes.push_back(n);
                        p->faces.push_back(std::move(x));
                    }
                    for (auto c : cells)
                    {
                        ExplicitRegionProvider::Cell x;
                        x.id = c;
                        x.type = r.topology().cell_type(c);
                        x.centroid = r.geometry().cell_centroid(c);
                        x.volume = r.geometry().cell_volume(c);
                        if (full.contains(c))
                        {
                            for (auto f : r.topology().cell_faces(c))
                                x.faces.push_back(f);
                            for (auto n : r.topology().cell_nodes(c))
                                x.nodes.push_back(n);
                        }
                        p->cells.push_back(std::move(x));
                    }
                    for (auto n : nodes)
                        p->nodes.push_back({n, r.geometry().node_coordinates(n)});
                    for (auto b : r.topology().boundary_batch_ids())
                        p->boundaries.emplace(b, r.topology().boundary_batch_name(b));
                },
                native);
            return p;
        };
        FrozenEvaluationCache evaluated;
        std::map<ID, std::shared_ptr<const FrozenFace>> canonical_faces;
        collective(*comm,
            [&]
            {
                Budget budget{options.byte_budget};
                std::set<AdaptiveCellKey> seen;
                for (auto gid : selected)
                {
                    budget.add(sizeof(BundleData));
                    auto b = std::make_shared<BundleData>();
                    b->source_uid = root->uid;
                    b->key = hierarchy.cell_key(gid);
                    if (!seen.insert(b->key).second)
                        throw std::invalid_argument("Duplicate frozen source selection.");
                    b->source_gid = gid;
                    b->material = hierarchy.cell_material(gid);
                    auto [region, native] = source->native_cell(gid);
                    b->native = native;
                    b->part = part(region, native, {}, {});
                    const auto metric = frozen_metric(*b, *root, evaluated);
                    if (metric.volume != source->cell_volume(gid) || metric.center != point(source->cell_centroid(gid)))
                        throw std::invalid_argument("Frozen detached cell metric differs from accepted source.");
                    const auto& copied = frozen_provider(b->part, evaluated);
                    std::visit(
                        [&](const auto& provider)
                        {
                            const auto nodes = provider.topology().cell_nodes(native);
                            const auto original = source->cell_nodes(gid);
                            if (nodes.size() != original.size())
                                throw std::invalid_argument("Frozen detached cell node incidence differs from source.");
                            for (std::size_t i = 0; i < nodes.size(); ++i)
                            {
                                const auto global = b->part->node_prefix + nodes[i];
                                if (global != original[i] ||
                                    frozen_transform(*root, provider.geometry().node_coordinates(nodes[i])) !=
                                        point(source->node_coordinates(global)))
                                    throw std::invalid_argument("Frozen detached physical node differs from source.");
                            }
                        },
                        copied);
                    source->topology(region,
                        [&](const auto& t)
                        {
                            for (auto nf : t.cell_faces(native))
                            {
                                budget.add(sizeof(FrozenRow));
                                FrozenRow row;
                                row.native = nf;
                                for (auto logical : source->logical_faces({region, nf}))
                                {
                                    auto found = canonical_faces.find(logical);
                                    if (found != canonical_faces.end())
                                    {
                                        row.faces.emplace_back(
                                            found->second, int(source->native_flux_orientation({region, nf}, logical)));
                                        continue;
                                    }
                                    budget.add(sizeof(FrozenFace));
                                    auto f = std::make_shared<FrozenFace>();
                                    f->gid = logical;
                                    auto address = source->native_face(logical);
                                    f->native = address.face;
                                    f->part = part(address.region, {}, address.face, {});
                                    auto first = source->owner_cell(logical), second = source->neighbor_cell(logical);
                                    if (first == second)
                                        throw std::invalid_argument(
                                            "Frozen native self-periodic incidence is unsupported.");
                                    f->first = hierarchy.cell_key(first);
                                    if (second != MultiRegionMesh::invalid_cell_id())
                                        f->second = hierarchy.cell_key(second);
                                    const auto& retained = frozen_provider(f->part, evaluated);
                                    std::visit(
                                        [&](const auto& provider)
                                        {
                                            const auto center =
                                                frozen_transform(*root, provider.geometry().face_centroid(f->native));
                                            auto vector = provider.geometry().face_area_vector(f->native);
                                            const auto scale =
                                                (root->frame[1] - root->frame[0]) / (root->frame[1] - root->frame[0]);
                                            vector = {vector.x * scale, vector.y * scale, vector.z};
                                            const auto area = source->area_magnitude(vector);
                                            if (center != point(source->face_centroid(logical)) ||
                                                area != source->face_area(logical) ||
                                                point(vector / area) != point(source->face_normal(logical)))
                                                throw std::invalid_argument("Frozen detached canonical face metric "
                                                                            "differs from accepted source.");
                                            const auto nodes = provider.topology().face_nodes(f->native);
                                            const auto original = source->face_nodes(logical);
                                            if (nodes.size() != original.size())
                                                throw std::invalid_argument(
                                                    "Frozen detached canonical node count differs.");
                                            for (std::size_t i = 0; i < nodes.size(); ++i)
                                            {
                                                const auto global = f->part->node_prefix + nodes[i];
                                                if (global != original[i] ||
                                                    frozen_transform(
                                                        *root, provider.geometry().node_coordinates(nodes[i])) !=
                                                        point(source->node_coordinates(global)))
                                                    throw std::invalid_argument(
                                                        "Frozen detached canonical node route differs from source.");
                                            }
                                        },
                                        retained);
                                    f->second_image = point(source->periodic_translation(logical) * -1.0);
                                    f->boundary = source->boundary_id(logical);
                                    canonical_faces.emplace(logical, f);
                                    row.faces.emplace_back(
                                        std::move(f), int(source->native_flux_orientation({region, nf}, logical)));
                                }
                                b->rows.push_back(std::move(row));
                            }
                        });

                    result.source_cells.push_back({b->key, gid});
                    result.owned.push_back(FrozenNativeBundle(std::move(b)));
                }
                std::vector<std::shared_ptr<const BundleData>> records;
                for (const auto& bundle : result.owned)
                    records.push_back(bundle.data_);
                (void) encode_batch(records, options.byte_budget);
                if (hierarchy.observation() != observation || !source_unchanged(hierarchy, *source, observation))
                    throw std::invalid_argument("Frozen source changed during detachment.");
                result.root = std::shared_ptr<const FrozenNativeRoot>(new FrozenNativeRoot(root));
            });
        return result;
    }
};
FrozenNativeBootstrap detach_frozen_native(const AdaptiveHierarchy& source, std::span<const ID> owned,
    std::span<const char> schema, Teuchos::RCP<const Teuchos::Comm<int>> comm, FrozenNativeOptions options)
{
    if (comm.is_null())
        throw std::invalid_argument("Frozen export needs communicator.");
    return FrozenNativeAccess::detach(source, owned, schema, comm, options);
}
const FrozenNativeRoot::SourceObservation& FrozenNativeRoot::source_observation() const noexcept
{
    return data_->observation;
}
std::span<const char> FrozenNativeRoot::schema() const noexcept
{
    return data_->schema;
}
const AdaptiveCellKey& FrozenNativeBundle::key() const noexcept
{
    return data_->key;
}
ID FrozenNativeBundle::revision() const noexcept
{
    return data_->revision;
}
ID FrozenNativeBundle::material() const noexcept
{
    return data_->material;
}
ID FrozenNativeBundle::source_gid() const noexcept
{
    return data_->source_gid;
}
std::size_t FrozenNativeBundle::encoded_payload_bytes() const
{
    std::array<std::shared_ptr<const BundleData>, 1> record{data_};
    return encode_batch(record, std::numeric_limits<std::size_t>::max()).size();
}
std::size_t FrozenNativeBundle::storage_bytes() const
{
    std::array<std::shared_ptr<const BundleData>, 1> record{data_};
    return sizeof(*this) + encode_batch(record, std::numeric_limits<std::size_t>::max()).capacity();
}
struct FrozenNativeModel::Data
{
    std::map<AdaptiveCellKey, FrozenNativeBundle> owned;
    std::shared_ptr<const FrozenNativeRoot> root;
    Teuchos::RCP<const Teuchos::Comm<int>> comm;
    FrozenNativeOptions options;
    std::shared_ptr<const DistributedAncestryDirectory> directory;
    std::shared_ptr<const void> token = std::make_shared<const char>(0);
    ID topology = 0, partition = 0;
};
namespace
{
void register_model(FrozenNativeModel::Data& d)
{
    ID local = d.owned.size(), inclusive = 0, total = 0;
    collective(*d.comm,
        [&]
        {
            if (local > std::numeric_limits<ID>::max() / ID(d.comm->getSize()))
                throw std::overflow_error("Frozen cell prefix overflow.");
        });
    Teuchos::scan(*d.comm, Teuchos::REDUCE_SUM, 1, &local, &inclusive);
    Teuchos::reduceAll(*d.comm, Teuchos::REDUCE_SUM, 1, &local, &total);
    std::vector<AdaptiveOwner> entries;
    collective(*d.comm,
        [&]
        {
            Budget budget{d.options.byte_budget};
            ID gid = inclusive - local;
            for (const auto& [key, b] : d.owned)
            {
                auto encoded = encode_adaptive_key(key);
                budget.add(encoded.size() + sizeof(AdaptiveOwner));
                entries.push_back({std::move(encoded), d.comm->getRank(), b.revision(), gid++});
            }
        });
    d.directory = std::make_shared<const DistributedAncestryDirectory>(entries, d.comm, d.options.byte_budget);
}
} // namespace
FrozenNativeModel::FrozenNativeModel(
    FrozenNativeBootstrap bootstrap, Teuchos::RCP<const Teuchos::Comm<int>> comm, FrozenNativeOptions options)
{
    if (comm.is_null())
        throw std::invalid_argument("Frozen model needs communicator.");
    std::shared_ptr<Data> d;
    collective(*comm,
        [&]
        {
            if (!bootstrap.root || !options.byte_budget || !options.maximum_owned ||
                bootstrap.owned.size() > options.maximum_owned)
                throw std::invalid_argument("Invalid frozen bootstrap or limits.");
            d = std::make_shared<Data>();
            d->root = bootstrap.root;
            d->comm = comm;
            d->options = options;
            d->topology = bootstrap.root->data_->observation.topology;
            d->partition = bootstrap.root->data_->observation.partition;
            std::vector<std::shared_ptr<const BundleData>> originals;
            for (const auto& b : bootstrap.owned)
                originals.push_back(b.data_);
            auto canonical = decode_batch(encode_batch(originals, options.byte_budget), options.byte_budget);
            bootstrap.owned.clear();
            for (auto& b : canonical)
                bootstrap.owned.push_back(FrozenNativeBundle(std::move(b)));
            for (auto& b : bootstrap.owned)
            {
                if (!b.data_ || b.data_->source_uid != d->root->data_->uid || !b.data_->part || b.data_->metric ||
                    !d->owned.emplace(b.key(), std::move(b)).second)
                    throw std::invalid_argument("Invalid or duplicate frozen owned bundle.");
            }
        });
    agree(d->root->schema(), *comm, options.byte_budget);
    Writer limits;
    limits.word(options.byte_budget);
    limits.word(options.maximum_owned);
    limits.word(options.maximum_imported);
    agree(limits.data, *comm, options.byte_budget);
    register_model(*d);
    ID count = d->owned.size(), total = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &count, &total);
    collective(*comm,
        [&]
        {
            if (total != d->root->data_->global_cells)
                throw std::invalid_argument("Frozen bootstrap ownership is incomplete.");
        });
    data_ = std::move(d);
}
const std::map<AdaptiveCellKey, FrozenNativeBundle>& FrozenNativeModel::owned() const noexcept
{
    return data_->owned;
}
const std::shared_ptr<const void>& FrozenNativeModel::identity() const noexcept
{
    return data_->token;
}
ID FrozenNativeModel::topology_generation() const noexcept
{
    return data_->topology;
}
ID FrozenNativeModel::partition_generation() const noexcept
{
    return data_->partition;
}
Teuchos::RCP<const Teuchos::Comm<int>> FrozenNativeModel::communicator() const noexcept
{
    return data_->comm;
}
FrozenNativeOptions FrozenNativeModel::options() const noexcept
{
    return data_->options;
}
std::span<const char> FrozenNativeModel::configuration_descriptor() const noexcept
{
    return data_->root->schema();
}
ID FrozenNativeModel::global_cells() const noexcept
{
    return data_->root->data_->global_cells;
}
std::vector<AdaptiveOwner> FrozenNativeModel::lookup_cells(std::span<const AdaptiveCellKey> keys) const
{
    std::vector<AdaptiveBytes> encoded;
    collective(*data_->comm,
        [&]
        {
            Budget budget{data_->options.byte_budget};
            for (const auto& key : keys)
            {
                auto bytes = encode_adaptive_key(key);
                budget.add(bytes.size() + sizeof(AdaptiveBytes));
                encoded.push_back(std::move(bytes));
            }
        });
    return data_->directory->lookup(encoded);
}
FrozenNativeModel::Storage FrozenNativeModel::storage() const
{
    Storage result;
    result.owned_cells = data_->owned.size();
    result.metadata_bytes = data_->directory->storage_bytes() + data_->root->schema().size();
    std::set<const FrozenPart*> parts;
    std::set<const NativeFragmentDescriptor*> descriptors;
    std::set<const AdaptiveBytes*> chunks;
    std::set<const FrozenFace*> faces;
    auto count = [&](const std::shared_ptr<const FrozenPart>& part)
    {
        if (!parts.insert(part.get()).second)
            return;
        if (part->procedural)
        {
            if (descriptors.insert(part->procedural.get()).second)
            {
                ++result.procedural_fragments;
                result.procedural_bytes +=
                    sizeof(FrozenPart) + sizeof(NativeFragmentDescriptor) + part->procedural->view.capacity() +
                    part->procedural->parts.capacity() * sizeof(std::shared_ptr<const AdaptiveBytes>);
            }
            for (const auto& bytes : part->procedural->parts)
                if (chunks.insert(bytes.get()).second)
                {
                    ++result.shared_native_parts;
                    result.procedural_bytes += sizeof(AdaptiveBytes) + bytes->capacity();
                }
        }
        else
        {
            result.explicit_payload_bytes += encode_frozen_part(*part).size();
            result.explicit_bytes += sizeof(FrozenPart) +
                                     part->cells.capacity() * sizeof(ExplicitRegionProvider::Cell) +
                                     part->faces.capacity() * sizeof(ExplicitRegionProvider::Face) +
                                     part->nodes.capacity() * sizeof(ExplicitRegionProvider::Node);
            for (const auto& cell : part->cells)
                result.explicit_bytes += (cell.faces.capacity() + cell.nodes.capacity()) * sizeof(ID);
            for (const auto& face : part->faces)
                result.explicit_bytes += face.nodes.capacity() * sizeof(ID);
            for (const auto& [id, name] : part->boundaries)
                result.explicit_bytes +=
                    sizeof(std::pair<const int, std::string>) + 4 * sizeof(void*) + name.capacity();
        }
    };
    for (const auto& [key, bundle] : data_->owned)
    {
        const auto& b = *bundle.data_;
        result.routing_bytes += sizeof(b) + encode_adaptive_key(key).size() + b.rows.capacity() * sizeof(FrozenRow);
        count(b.part);
        for (const auto& row : b.rows)
        {
            result.routing_bytes += row.faces.capacity() * sizeof(decltype(row.faces)::value_type);
            for (const auto& [face, sign] : row.faces)
            {
                if (faces.insert(face.get()).second)
                {
                    ++result.canonical_routes;
                    result.routing_bytes += sizeof(FrozenFace) + encode_adaptive_key(face->first).size() +
                                            (face->second ? encode_adaptive_key(*face->second).size() : 0);
                }
                count(face->part);
            }
        }
    }
    result.metadata_bytes += result.routing_bytes;
    return result;
}
FrozenNativeModel::PreparedMigration FrozenNativeModel::prepare_migration(std::span<const int> destinations) const
{
    const auto id = transaction(*data_->comm);
    std::vector<OwnerPacket> packets;
    ID changed = 0, global_changed = 0, bytes = 0;
    collective(*data_->comm,
        [&]
        {
            if (destinations.size() != data_->owned.size() || data_->partition == std::numeric_limits<ID>::max())
                throw std::invalid_argument("Frozen destination extent or generation invalid.");
            Budget budget{data_->options.byte_budget};
            std::map<int, std::vector<std::shared_ptr<const BundleData>>> groups;
            std::size_t i = 0;
            for (const auto& [key, b] : data_->owned)
            {
                auto rank = destinations[i++];
                if (rank < 0 || rank >= data_->comm->getSize())
                    throw std::invalid_argument("Invalid frozen destination.");
                changed += rank != data_->comm->getRank();
                groups[rank].push_back(b.data_);
            }
            for (const auto& [rank, records] : groups)
            {
                Writer w;
                w.word(data_->root->data_->uid);
                w.bytes(encode_batch(records, data_->options.byte_budget));
                budget.packet(w.data.size());
                if (rank != data_->comm->getRank())
                    bytes += w.data.size() + 24;
                packets.push_back({rank, std::move(w.data)});
            }
        });
    Teuchos::reduceAll(*data_->comm, Teuchos::REDUCE_SUM, 1, &changed, &global_changed);
    PreparedMigration prepared;
    prepared.source_ = data_->token;
    prepared.topology_ = data_->topology;
    prepared.partition_ = data_->partition;
    prepared.transaction_ = id;
    prepared.global_changed_ = global_changed;
    prepared.local_bytes_ = bytes;
    prepared.packets_ = std::move(packets);
    collective(*data_->comm,
        [&]
        {
            if (bytes > std::numeric_limits<ID>::max() / ID(data_->comm->getSize()))
                throw std::overflow_error("Frozen prepared byte sum overflows.");
        });
    Teuchos::reduceAll(*data_->comm, Teuchos::REDUCE_SUM, 1, &bytes, &prepared.global_bytes_);
    return prepared;
}
FrozenNativeModel::Plan FrozenNativeModel::plan_migration(std::span<const int> destinations) const
{
    return plan_migration(prepare_migration(destinations));
}
FrozenNativeModel::Plan FrozenNativeModel::plan_migration(const PreparedMigration& prepared) const
{
    std::array<ID, 3> descriptor{prepared.transaction_, prepared.global_changed_, prepared.global_bytes_},
        root = descriptor;
    Teuchos::broadcast(*data_->comm, 0, int(root.size()), root.data());
    collective(*data_->comm,
        [&]
        {
            if (root != descriptor || prepared.source_ != data_->token || prepared.topology_ != data_->topology ||
                prepared.partition_ != data_->partition)
                throw std::invalid_argument("Foreign, stale or mixed frozen prepared migration.");
        });
    Plan plan;
    plan.source_ = data_->token;
    plan.topology_ = data_->topology;
    plan.partition_ = data_->partition;
    plan.transaction_ = prepared.transaction_;
    if (!prepared.global_changed_)
    {
        plan.candidate_ = std::shared_ptr<const FrozenNativeModel>(new FrozenNativeModel(data_));
        return plan;
    }
    auto received = exchange_owner_packets(prepared.packets_, data_->comm, data_->options.byte_budget);
    plan.packet_phase_peak_ = received.stats.peak_scratch_bytes;
    std::vector<std::shared_ptr<const BundleData>> records;
    std::vector<int> peers;
    collective(*data_->comm,
        [&]
        {
            for (const auto& p : received.packets)
            {
                Reader r{p.bytes};
                if (r.word() != data_->root->data_->uid)
                    throw std::invalid_argument("Foreign frozen copy source.");
                auto batch = decode_batch(r.bytes(), data_->options.byte_budget);
                r.done();
                for (auto& b : batch)
                {
                    records.push_back(std::move(b));
                    peers.push_back(p.peer);
                }
            }
        });
    std::vector<AdaptiveCellKey> keys;
    collective(*data_->comm,
        [&]
        {
            for (const auto& b : records)
                keys.push_back(b->key);
        });
    auto owners = lookup_cells(keys);
    std::shared_ptr<Data> next;
    collective(*data_->comm,
        [&]
        {
            records = decode_batch(encode_batch(records, data_->options.byte_budget), data_->options.byte_budget);
            next = std::make_shared<Data>(*data_);
            next->token = std::make_shared<const char>(0);
            ++next->partition;
            next->owned.clear();
            if (records.size() > data_->options.maximum_owned)
                throw std::invalid_argument("Frozen imported ownership budget exhausted.");
            for (std::size_t i = 0; i < records.size(); ++i)
            {
                auto& b = records[i];
                if (b->source_uid != data_->root->data_->uid || !b->part || b->metric || owners[i].owner != peers[i] ||
                    owners[i].revision != b->revision || !next->owned.emplace(b->key, FrozenNativeBundle(b)).second)
                    throw std::invalid_argument("Stale, foreign or duplicate frozen ownership record.");
            }
        });
    register_model(*next);
    ID local = next->owned.size(), total = 0;
    Teuchos::reduceAll(*data_->comm, Teuchos::REDUCE_SUM, 1, &local, &total);
    collective(*data_->comm,
        [&]
        {
            if (total != global_cells())
                throw std::invalid_argument("Frozen migration lost ownership.");
        });
    plan.local_application_bytes_ = prepared.local_bytes_;
    plan.application_bytes_ = prepared.global_bytes_;
    plan.candidate_ = std::shared_ptr<const FrozenNativeModel>(new FrozenNativeModel(std::move(next)));
    return plan;
}
FrozenNativeModel::Plan FrozenNativeModel::plan_migration(const AdaptiveBalancePlan& p) const
{
    collective(*data_->comm,
        [&]
        {
            if (p.source.identity != identity() || p.source.topology != topology_generation() ||
                p.source.partition != partition_generation() || p.keys.size() != owned().size() ||
                p.revisions.size() != owned().size() || (p.accepted && !p.proposed.wire_bytes_exact))
                throw std::invalid_argument("Foreign or unqualified frozen balance proposal.");
            std::size_t i = 0;
            for (const auto& [key, b] : owned())
            {
                if (p.keys[i] != key || p.revisions[i] != b.revision())
                    throw std::invalid_argument("Frozen balance key/revision differs.");
                ++i;
            }
        });
    return plan_migration(p.localdestinations);
}
FrozenNativeModel::Plan FrozenNativeModel::plan_adaptation(
    std::span<const AdaptiveMark> marks, std::span<const AdaptiveBytes> coarsen) const
{
    ID local = marks.size() + coarsen.size(), global = 0;
    Teuchos::reduceAll(*data_->comm, Teuchos::REDUCE_SUM, 1, &local, &global);
    collective(*data_->comm,
        [&]
        {
            if (global)
                throw std::invalid_argument("Frozen native whole-model copy contract prohibits all topology changes.");
        });
    Plan p;
    p.source_ = identity();
    p.topology_ = topology_generation();
    p.partition_ = partition_generation();
    p.transaction_ = transaction(*data_->comm);
    p.candidate_ = std::shared_ptr<const FrozenNativeModel>(new FrozenNativeModel(data_));
    return p;
}
std::shared_ptr<const FrozenNativeModel> FrozenNativeModel::apply(const Plan& p) const
{
    Writer descriptor;
    descriptor.word(p.transaction_);
    agree(descriptor.data, *data_->comm, data_->options.byte_budget);
    collective(*data_->comm,
        [&]
        {
            if (p.source_ != identity() || p.topology_ != topology_generation() ||
                p.partition_ != partition_generation() || !p.candidate_)
                throw std::invalid_argument("Foreign or stale frozen source plan.");
        });
    return p.candidate_;
}
FrozenNativePublication FrozenNativeModel::publication(std::span<const AdaptiveCellKey> extra) const
{
    auto fetch = [&](const std::vector<AdaptiveCellKey>& keys, bool metrics)
    {
        collective(*data_->comm,
            [&]
            {
                if (keys.size() > data_->options.maximum_imported)
                    throw std::invalid_argument("Frozen requested/imported work budget exhausted.");
            });
        auto owners = lookup_cells(keys);
        std::vector<OwnerPacket> requests;
        collective(*data_->comm,
            [&]
            {
                Budget budget{data_->options.byte_budget};
                for (std::size_t i = 0; i < keys.size(); ++i)
                {
                    auto key = encode_adaptive_key(keys[i]);
                    budget.packet(16 + key.size());
                    Writer w;
                    w.bytes(key);
                    w.word(owners[i].revision);
                    requests.push_back({owners[i].owner, std::move(w.data)});
                }
            });
        auto incoming = exchange_owner_packets(requests, data_->comm, data_->options.byte_budget);
        std::vector<OwnerPacket> replies;
        collective(*data_->comm,
            [&]
            {
                Budget budget{data_->options.byte_budget};
                if (incoming.packets.size() > data_->options.maximum_imported)
                    throw std::invalid_argument("Frozen served ghost work budget exhausted.");
                std::map<int, std::vector<std::shared_ptr<const BundleData>>> groups;
                FrozenEvaluationCache evaluated;
                for (const auto& p : incoming.packets)
                {
                    budget.add(p.bytes.size() + sizeof(std::shared_ptr<const BundleData>) +
                               sizeof(decltype(groups)::value_type) + 4 * sizeof(void*));
                    Reader r{p.bytes};
                    auto key = decode_adaptive_key(r.bytes());
                    auto rev = r.word();
                    r.done();
                    const auto& b = data_->owned.at(key);
                    if (rev != b.revision())
                        throw std::invalid_argument("Stale frozen ghost request.");
                    if (metrics)
                    {
                        budget.add(sizeof(BundleData) + encode_adaptive_key(key).size());
                        auto m = std::make_shared<BundleData>();
                        m->key = b.data_->key;
                        m->source_uid = b.data_->source_uid;
                        m->revision = b.data_->revision;
                        m->material = b.data_->material;
                        m->source_gid = b.data_->source_gid;
                        m->native = b.data_->native;
                        m->metric = frozen_metric(*b.data_, *data_->root->data_, evaluated);
                        groups[p.peer].push_back(std::move(m));
                    }
                    else
                        groups[p.peer].push_back(b.data_);
                }
                for (const auto& [peer, b] : groups)
                {
                    budget.add(sizeof(OwnerPacket) + 16);
                    auto encoded = encode_batch(b, budget.limit - budget.used);
                    budget.add(encoded.size());
                    budget.add(encoded.size());
                    Writer w;
                    w.word(data_->root->data_->uid);
                    w.bytes(encoded);
                    replies.push_back({peer, std::move(w.data)});
                }
            });
        auto returned = exchange_owner_packets(replies, data_->comm, data_->options.byte_budget);
        std::map<AdaptiveCellKey, std::shared_ptr<const BundleData>> result;
        collective(*data_->comm,
            [&]
            {
                Budget budget{data_->options.byte_budget};
                std::map<AdaptiveCellKey, AdaptiveOwner> expected;
                for (std::size_t i = 0; i < keys.size(); ++i)
                {
                    budget.add(sizeof(decltype(expected)::value_type) + encode_adaptive_key(keys[i]).size() +
                               4 * sizeof(void*));
                    expected.emplace(keys[i], owners[i]);
                }
                for (const auto& p : returned.packets)
                {
                    budget.add(p.bytes.size());
                    Reader r{p.bytes};
                    if (r.word() != data_->root->data_->uid)
                        throw std::invalid_argument("Foreign frozen ghost root.");
                    for (auto b : decode_batch(r.bytes(), data_->options.byte_budget))
                    {
                        budget.add(sizeof(decltype(result)::value_type) + sizeof(BundleData) +
                                   encode_adaptive_key(b->key).size() + 4 * sizeof(void*));
                        if (result.size() >= data_->options.maximum_imported)
                            throw std::invalid_argument("Frozen returned ghost count budget exhausted.");
                        auto e = expected.find(b->key);
                        if (b->source_uid != data_->root->data_->uid || e == expected.end() ||
                            e->second.owner != p.peer || e->second.revision != b->revision || (!metrics && !b->part) ||
                            !result.emplace(b->key, b).second)
                            throw std::invalid_argument("Invalid frozen ghost response.");
                    }
                    r.done();
                }
                if (result.size() != expected.size())
                    throw std::invalid_argument("Incomplete frozen ghost response.");
            });
        return result;
    };
    Budget retained{data_->options.byte_budget};
    std::shared_ptr<FrozenNativeMeshInput::Data> input;
    std::vector<AdaptiveCellKey> requested;
    collective(*data_->comm,
        [&]
        {
            if (extra.size() > data_->options.maximum_imported)
                throw std::invalid_argument("Frozen extra-ghost work budget exhausted.");
            retained.add(sizeof(FrozenNativeMeshInput::Data));
            input = std::make_shared<FrozenNativeMeshInput::Data>();
            input->root = data_->root->data_;
            input->byte_budget = data_->options.byte_budget;
            std::set<AdaptiveCellKey> needed;
            const auto need = [&](const AdaptiveCellKey& key)
            {
                if (!needed.contains(key))
                {
                    retained.add(sizeof(AdaptiveCellKey) + encode_adaptive_key(key).size() + 4 * sizeof(void*));
                    needed.insert(key);
                }
            };
            for (const auto& key : extra)
                need(key);
            for (const auto& [key, b] : owned())
            {
                retained.add(sizeof(AdaptiveCellKey) + encode_adaptive_key(key).size() + 4 * sizeof(void*) +
                             sizeof(std::shared_ptr<const BundleData>));
                input->owned.insert(key);
                input->full.push_back(b.data_);
                for (const auto& row : b.data_->rows)
                    for (const auto& [face, sign] : row.faces)
                    {
                        need(face->first);
                        if (face->second)
                            need(*face->second);
                    }
            }
            for (const auto& key : needed)
                if (!owned().contains(key))
                {
                    if (requested.size() >= data_->options.maximum_imported)
                        throw std::invalid_argument("Frozen reconstruction ghost budget exhausted.");
                    retained.add(sizeof(AdaptiveCellKey) + encode_adaptive_key(key).size());
                    requested.push_back(key);
                }
            if (requested.size() > data_->options.maximum_imported)
                throw std::invalid_argument("Frozen reconstruction ghost budget exhausted.");
        });
    auto ghosts = fetch(requested, false);
    collective(*data_->comm,
        [&]
        {
            for (auto& [key, b] : ghosts)
            {
                retained.add(sizeof(std::shared_ptr<const BundleData>));
                input->full.push_back(std::move(b));
            }
        });
    std::set<AdaptiveCellKey> all;
    collective(*data_->comm,
        [&]
        {
            const auto needed = [&](const AdaptiveCellKey& key)
            {
                if (!all.contains(key))
                {
                    retained.add(sizeof(AdaptiveCellKey) + encode_adaptive_key(key).size() + 4 * sizeof(void*));
                    all.insert(key);
                }
            };
            for (const auto& b : input->full)
            {
                retained.add(sizeof(decltype(input->metrics)::value_type) + encode_adaptive_key(b->key).size() +
                             4 * sizeof(void*));
                input->metrics.emplace(b->key, b);
                needed(b->key);
                for (const auto& row : b->rows)
                    for (const auto& [face, sign] : row.faces)
                    {
                        needed(face->first);
                        if (face->second)
                            needed(*face->second);
                    }
            }
            requested.clear();
            for (const auto& key : all)
                if (!input->metrics.contains(key))
                {
                    if (requested.size() >= data_->options.maximum_imported)
                        throw std::invalid_argument("Frozen outer metric work budget exhausted.");
                    retained.add(sizeof(AdaptiveCellKey) + encode_adaptive_key(key).size());
                    requested.push_back(key);
                }
        });
    auto metrics = fetch(requested, true);
    collective(*data_->comm,
        [&]
        {
            for (const auto& [key, b] : metrics)
            {
                retained.add(
                    sizeof(decltype(input->metrics)::value_type) + encode_adaptive_key(key).size() + 4 * sizeof(void*));
                input->metrics.emplace(key, b);
            }
            retained.add(all.size() * sizeof(AdaptiveCellKey));
            requested.assign(all.begin(), all.end());
        });
    auto owners = lookup_cells(requested);
    collective(*data_->comm,
        [&]
        {
            for (std::size_t i = 0; i < requested.size(); ++i)
            {
                retained.add(sizeof(decltype(input->bindings)::value_type) + encode_adaptive_key(requested[i]).size() +
                             4 * sizeof(void*));
                input->bindings.emplace(requested[i], owners[i]);
            }
        });
    auto geometry = MultiRegionMesh::from_frozen_native(FrozenNativeMeshInput(input), data_->comm);
    FrozenNativePublication result;
    std::vector<ID> owned_gids, ghost_gids;
    collective(*data_->comm,
        [&]
        {
            for (const auto& b : input->full)
            {
                const auto gid = input->bindings.at(b->key).address;
                result.local_keys.emplace(gid, b->key);
                if (input->owned.contains(b->key))
                {
                    owned_gids.push_back(gid);
                    result.owned_gids.emplace(b->key, gid);
                }
                else
                    ghost_gids.push_back(gid);
            }
        });
    auto partition = CompositePartition<>::from_sparse_native(geometry, owned_gids, ghost_gids);
    result.partition = std::make_shared<const CompositePartition<>>(std::move(partition));
    result.mesh = std::make_shared<const MeshHandle<>>(*result.partition);
    result.storage = geometry->storage_report();
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
