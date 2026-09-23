/** @file MultiRegionMeshPartition.cc
 * @brief Versioned native region transfer and sparse explicit region residency.
 */
#include "geometry/mesh/MultiRegionMesh.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include <cstring>
#include <set>
#include <numeric>
#include <type_traits>

namespace SimpleFluid::Meshes
{
namespace
{
using ID = MultiRegionMesh::ID;
using Vec3 = MeshUtils::Vec3;
// Internal MPI wire format: one executable, native scalar representation.
// No pointers, size_t container objects, or borrowed storage are transmitted.
constexpr uint64_t partition_magic = 0x5346524547490002ULL;
struct Writer
{
    std::vector<char> data;
    template<class T> void value(const T& v)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p = reinterpret_cast<const char*>(&v);
        data.insert(data.end(), p, p + sizeof(T));
    }
    template<class Range> void array(const Range& values)
    {
        value(uint64_t(values.size()));
        for (const auto& v : values) value(v);
    }
    void string(const std::string& v) { array(v); }
    void boolean(bool v) { value(uint8_t(v)); }
};
struct Reader
{
    std::span<const char> data;
    template<class T> T value()
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (data.size() < sizeof(T)) throw std::invalid_argument("Truncated composite partition packet.");
        T result;
        std::memcpy(&result, data.data(), sizeof(T));
        data = data.subspan(sizeof(T));
        return result;
    }
    size_t count(size_t minimum_bytes = 1)
    {
        const auto n = value<uint64_t>();
        if (n > data.size() / minimum_bytes)
            throw std::invalid_argument("Composite partition count exceeds remaining payload.");
        return static_cast<size_t>(n);
    }
    bool boolean()
    {
        const auto v = value<uint8_t>();
        if (v > 1) throw std::invalid_argument("Invalid boolean in composite partition packet.");
        return v != 0;
    }
    template<class T> std::vector<T> array()
    {
        std::vector<T> result(count(sizeof(T)));
        for (auto& v : result) v = value<T>();
        return result;
    }
    std::string string()
    {
        const auto chars = array<char>();
        return {chars.begin(), chars.end()};
    }
};
void write_layout(Writer& w, const RegionLayout& l)
{
    w.value(l.family); w.value(uint64_t(l.cells)); w.value(uint64_t(l.faces));
    w.value(uint64_t(l.nodes));
    for (auto n : l.extents) w.value(uint64_t(n));
    for (auto p : l.periodic) w.boolean(p);
}
RegionLayout read_layout(Reader& r)
{
    RegionLayout l{};
    l.family = r.value<RegionLayout::Family>();
    l.cells = r.value<uint64_t>(); l.faces = r.value<uint64_t>(); l.nodes = r.value<uint64_t>();
    for (auto& n : l.extents) n = r.value<uint64_t>();
    for (auto& p : l.periodic) p = r.boolean();
    if (!l.cells || !l.faces || !l.nodes || l.cells == MultiRegionMesh::invalid_cell_id()
        || l.faces == MultiRegionMesh::invalid_cell_id() || l.nodes == MultiRegionMesh::invalid_cell_id()
        || (l.family != RegionLayout::Family::Rectilinear && l.family != RegionLayout::Family::Extruded
            && l.family != RegionLayout::Family::Explicit && l.family != RegionLayout::Family::Cylindrical))
        throw std::invalid_argument("Invalid global region layout in partition packet.");
    return l;
}
template<class T, class G>
void write_explicit(Writer& w, const T& t, const G& g, const std::set<ID>& visible)
{
    std::set<ID> faces, nodes, metrics = visible;
    for (auto c : visible)
    {
        for (auto f : t.cell_faces(c)) faces.insert(f);
        for (auto n : t.cell_nodes(c)) nodes.insert(n);
    }
    for (auto f : faces)
    {
        metrics.insert(t.owner_cell(f));
        const auto n = t.neighbor_cell(f);
        if (n != MultiRegionMesh::invalid_cell_id()) metrics.insert(n);
        for (auto node : t.face_nodes(f)) nodes.insert(node);
    }
    w.value(uint64_t(metrics.size()));
    for (auto c : metrics)
    {
        w.value(c); w.value(t.cell_type(c)); w.value(g.cell_centroid(c)); w.value(g.cell_volume(c));
        if (visible.contains(c)) { w.array(t.cell_faces(c)); w.array(t.cell_nodes(c)); }
        else { w.value(uint64_t(0)); w.value(uint64_t(0)); }
    }
    w.value(uint64_t(faces.size()));
    for (auto f : faces)
    {
        w.value(f); w.value(ID(t.owner_cell(f))); w.value(ID(t.neighbor_cell(f)));
        w.array(t.face_nodes(f)); w.value(g.face_centroid(f)); w.value(g.face_area_vector(f));
        w.value(t.boundary_id(f));
    }
    w.value(uint64_t(nodes.size()));
    for (auto n : nodes) { w.value(n); w.value(g.node_coordinates(n)); }
    const auto boundaries = t.boundary_batch_ids();
    w.value(uint64_t(boundaries.size()));
    for (auto b : boundaries) { w.value(b); w.string(t.boundary_batch_name(b)); }
}
DistributedExplicitRegion read_explicit(Reader& r, std::string name, RegionLayout layout)
{
    using Provider = ExplicitRegionProvider;
    std::vector<Provider::Cell> cells(r.count(sizeof(ID) + sizeof(MeshUtils::CellType) + sizeof(Vec3) + sizeof(real_t) + 2 * sizeof(uint64_t)));
    for (auto& c : cells)
    {
        c.id = r.value<ID>(); c.type = r.value<MeshUtils::CellType>();
        c.centroid = r.value<Vec3>(); c.volume = r.value<real_t>();
        c.faces = r.array<ID>(); c.nodes = r.array<ID>();
    }
    std::vector<Provider::Face> faces(r.count(4 * sizeof(ID) + 2 * sizeof(Vec3) + sizeof(int)));
    for (auto& f : faces)
    {
        f.id = r.value<ID>(); f.owner = r.value<ID>(); f.neighbor = r.value<ID>();
        f.nodes = r.array<ID>(); f.centroid = r.value<Vec3>(); f.area_vector = r.value<Vec3>();
        f.boundary = r.value<int>();
    }
    std::vector<Provider::Node> nodes(r.count(sizeof(ID) + sizeof(Vec3)));
    for (auto& n : nodes) { n.id = r.value<ID>(); n.point = r.value<Vec3>(); }
    std::map<int, std::string> boundaries;
    for (size_t i = r.count(sizeof(int) + sizeof(uint64_t)); i; --i)
    {
        const auto id = r.value<int>(); auto name = r.string();
        if (!boundaries.emplace(id, std::move(name)).second)
            throw std::invalid_argument("Duplicate explicit boundary ID in partition packet.");
    }
    auto provider = std::make_shared<const Provider>(layout, std::move(cells), std::move(faces),
                                                   std::move(nodes), std::move(boundaries));
    return {std::move(name), provider, provider};
}
template<class Region>
void write_region(Writer& w, const Region& region, uint32_t tag, const std::set<ID>& visible)
{
    w.value(tag); w.string(region.name()); write_layout(w, region.layout());
    const auto& t = region.topology(); const auto& g = region.geometry();
    if constexpr (std::same_as<Region, NativeIsoRegion<UnstructuredMesh>>
                  || std::same_as<Region, DistributedExplicitRegion>)
        write_explicit(w, t, g, visible);
    else
    {
        const auto& native = [&]() -> const auto& {
            if constexpr (requires { g.native(); }) return g.native(); else return g;
        }();
        if constexpr (requires { native.cell_edges(); })
        {
            for (const auto& axis : native.cell_edges()) w.array(axis);
            std::array<std::string, 6> names;
            for (auto b : t.boundary_batch_ids()) names.at(b) = t.boundary_batch_name(b);
            for (const auto& name : names) w.string(name);
        }
        else
        {
            const auto& base = [&]() -> const auto& {
                if constexpr (requires { native.rz_nodes(); }) return native.rz_nodes();
                else return native.xy_nodes();
            }();
            const auto& layers = [&]() -> const auto& {
                if constexpr (requires { native.theta_edges(); }) return native.theta_edges();
                else return native.z_edges();
            }();
            w.array(base); w.array(layers);
            const auto& topo = [&]() -> const auto& {
                if constexpr (requires { t.native_topology(); }) return t.native_topology();
                else return t.native().topology();
            }();
            w.value(uint64_t(region.layout().extents[0]));
            for (size_t c = 0; c < region.layout().extents[0]; ++c)
            {
                std::vector<unsigned> loop;
                for (auto edge_id : topo.cell_side_faces(c))
                {
                    const auto& edge = topo.side_face(edge_id);
                    loop.push_back(edge.nodes[edge.owner == c ? 0 : 1]);
                }
                w.array(loop);
            }
            std::vector<SemiStructMeshTopo::BoundaryEdge> boundaries;
            for (const auto& edge : topo.side_faces())
                if (edge.neighbor == SemiStructMeshTopo::invalid_ordinal)
                    boundaries.push_back({edge.nodes[0], edge.nodes[1], topo.boundary_batch_name(edge.boundary_id)});
            w.value(uint64_t(boundaries.size()));
            for (const auto& b : boundaries) { w.value(b.node0); w.value(b.node1); w.string(b.batch_name); }
            const auto ids = topo.boundary_batch_ids();
            w.value(uint64_t(ids.size()));
            for (auto b : ids) { w.string(topo.boundary_batch_name(b)); w.string(t.boundary_batch_name(b)); }
        }
    }
}
MultiRegionMesh::Region read_region(Reader& r)
{
    const auto tag = r.value<uint32_t>(); auto name = r.string(); const auto layout = read_layout(r);
    const auto checked = [&layout](MultiRegionMesh::Region result)
    {
        const auto actual = std::visit([](const auto& region) { return region.layout(); }, result);
        if (actual.family != layout.family || actual.cells != layout.cells || actual.faces != layout.faces
            || actual.nodes != layout.nodes || actual.extents != layout.extents || actual.periodic != layout.periodic)
            throw std::invalid_argument("Reconstructed partition region differs from its declared layout.");
        return result;
    };
    if (tag == 3 || tag == 7) return checked(read_explicit(r, std::move(name), layout));
    if (tag == 0 || tag == 1 || tag == 4)
    {
        Vec3D<ArrReal> edges;
        for (auto& axis : edges) axis = r.array<real_t>();
        OrthoMeshTopo::BoundaryNames names;
        for (auto& n : names) n = r.string();
        if (tag == 1) return checked(native_region(std::move(name), std::make_shared<OrthogonalCartesian3D>(edges,
            Vec3D<bool>{layout.periodic[0], layout.periodic[1], layout.periodic[2]})));
        if (tag == 4) return checked(native_region(std::move(name), std::make_shared<OrthogonalCylindrial3D>(edges)));
        for (size_t axis = 0; axis < 3; ++axis)
            if (!layout.extents[axis] || layout.extents[axis] >= std::numeric_limits<unsigned>::max()
                || edges[axis].size() != layout.extents[axis] + 1 || layout.periodic[axis])
                throw std::invalid_argument("Invalid rectilinear partition descriptor extents.");
        auto topology = std::make_shared<const RectilinearTopology>(std::make_shared<const OrthoMeshTopo>(
            layout.extents[0], layout.extents[1], layout.extents[2], false, false, false, std::move(names)));
        return checked(CartesianRegion(std::move(name), std::move(topology), std::make_shared<const RectilinearGeometry>(std::move(edges))));
    }
    if (tag != 2 && tag != 5 && tag != 6) throw std::invalid_argument("Unknown partition region provider tag.");
    auto base = r.array<Vec3>(); auto layers = r.array<real_t>();
    if (base.empty() || base.size() > std::numeric_limits<unsigned>::max() || layers.size() < 2
        || layers.size() > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("Invalid extruded partition descriptor extents.");
    Arr<Arr<unsigned>> loops(r.count(sizeof(uint64_t)));
    for (auto& loop : loops) loop = r.array<unsigned>();
    Arr<SemiStructMeshTopo::BoundaryEdge> boundaries(r.count(2 * sizeof(unsigned) + sizeof(uint64_t)));
    for (auto& b : boundaries) { b.node0 = r.value<unsigned>(); b.node1 = r.value<unsigned>(); b.batch_name = r.string(); }
    std::map<std::string, std::string> aliases;
    std::set<std::string> alias_sources;
    for (size_t n = r.count(2 * sizeof(uint64_t)); n; --n)
    {
        auto source = r.string(); auto target = r.string();
        if (!alias_sources.insert(source).second || source.empty() || target.empty())
            throw std::invalid_argument("Invalid or repeated extruded partition boundary alias.");
        if (source != target) aliases.emplace(std::move(source), std::move(target));
    }
    if (tag == 2)
    {
        if (!aliases.empty()) throw std::invalid_argument("Native extrusion cannot carry boundary aliases.");
        return checked(native_region(std::move(name), std::make_shared<SemiStructuredXY_Z>(base, loops, layers, boundaries)));
    }
    auto topology = std::make_shared<const ExtrudedTopology>(base.size(), loops, layers.size() - 1, boundaries, std::move(aliases));
    if (tag == 5) return checked(ExtrudedRegion(std::move(name), topology,
        std::make_shared<const ExtrudedGeometry>(topology, std::move(base), std::move(layers))));
    return checked(SweptRZRegion(std::move(name), topology,
        std::make_shared<const SweptRZGeometry>(topology, std::move(base), std::move(layers))));
}
void write_patch(Writer& w, const StructuredPatch& p)
{
    w.value(uint64_t(p.region)); w.value(p.boundary);
    for (auto n : p.begin) w.value(uint64_t(n));
    for (auto n : p.extent) w.value(uint64_t(n));
    for (auto a : p.axes) w.value(a);
    for (auto b : p.reversed) w.boolean(b);
}
StructuredPatch read_patch(Reader& r)
{
    StructuredPatch p;
    p.region = r.value<uint64_t>(); p.boundary = r.value<int>();
    for (auto& n : p.begin) n = r.value<uint64_t>();
    for (auto& n : p.extent) n = r.value<uint64_t>();
    for (auto& a : p.axes) a = r.value<unsigned>();
    for (auto& b : p.reversed) b = r.boolean();
    return p;
}
void write_interface(Writer& w, const MultiRegionMesh::Interface& interface)
{
    w.value(uint32_t(interface.index()));
    std::visit([&](const auto& v)
    {
        using T = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::same_as<T, StructuredPatchInterface>)
        {
            write_patch(w, v.first); write_patch(w, v.second); w.value(v.permutation);
            for (auto b : v.reversed) w.boolean(b);
            w.boolean(bool(v.periodic_translation));
            if (v.periodic_translation) w.value(*v.periodic_translation);
        }
        else if constexpr (std::same_as<T, ExplicitConformingInterface>)
        {
            w.value(v.first_region); w.value(v.second_region); w.value(v.first_boundary); w.value(v.second_boundary);
            w.value(uint64_t(v.faces.size()));
            for (auto [a,b] : v.faces) { w.value(a); w.value(b); }
        }
        else
        {
            w.value(v.coarse_region); w.value(v.fine_region); w.value(v.coarse_boundary); w.value(v.fine_boundary);
            w.value(uint64_t(v.faces.size()));
            for (const auto& f : v.faces) { w.value(f.coarse_face); w.array(f.fine_faces); }
        }
    }, interface);
}
MultiRegionMesh::Interface read_interface(Reader& r)
{
    const auto tag = r.value<uint32_t>();
    if (tag == 0)
    {
        StructuredPatchInterface s;
        s.first = read_patch(r); s.second = read_patch(r);
        s.permutation = r.value<decltype(s.permutation)>();
        for (auto& b : s.reversed) b = r.boolean();
        if (r.boolean()) s.periodic_translation = r.value<Vec3>();
        s.validate_mapping();
        return s;
    }
    if (tag == 1)
    {
        ExplicitConformingInterface e;
        e.first_region = r.value<size_t>(); e.second_region = r.value<size_t>();
        e.first_boundary = r.value<int>(); e.second_boundary = r.value<int>();
        e.faces.resize(r.count(2 * sizeof(ID)));
        for (auto& [a,b] : e.faces) { a = r.value<ID>(); b = r.value<ID>(); }
        return e;
    }
    if (tag != 2) throw std::invalid_argument("Unknown partition interface tag.");
    NonconformingInterface n;
    n.coarse_region = r.value<size_t>(); n.fine_region = r.value<size_t>();
    n.coarse_boundary = r.value<int>(); n.fine_boundary = r.value<int>();
    n.faces.resize(r.count(sizeof(ID) + sizeof(uint64_t)));
    for (auto& f : n.faces) { f.coarse_face = r.value<ID>(); f.fine_faces = r.array<ID>(); }
    return n;
}
} // namespace

std::vector<char> MultiRegionMesh::serialize_partition(std::span<const ID> visible_cells) const
{
    const auto execution = acquire_execution_view();
    std::vector<std::set<ID>> selected(d_regions.size());
    for (auto c : visible_cells)
    {
        const auto [region, native] = native_cell(c);
        selected[region].insert(native);
    }
    // A canonical seam can refer to the other side of an outer halo face.
    // Retain that side's native owner geometry without enlarging field halos.
    for (auto c : visible_cells) for (auto f : cell_faces(c))
    {
        for (auto neighbor : {owner_cell(f), neighbor_cell(f)})
            if (neighbor != invalid_cell_id())
            {
                const auto [region, native] = native_cell(neighbor);
                selected[region].insert(native);
            }
    }
    Writer w;
    w.value(partition_magic); w.value(uint64_t(d_regions.size()));
    for (size_t i = 0; i < d_regions.size(); ++i)
        std::visit([&](const auto& region) { write_region(w, region, d_regions[i].index(), selected[i]); }, d_regions[i]);
    w.value(uint64_t(d_interfaces.size()));
    for (const auto& interface : d_interfaces) write_interface(w, interface);
    w.value(d_tolerance);
    w.array(d_cells); w.array(d_faces); w.array(d_nodes); w.array(d_native_faces);
    w.value(uint64_t(d_boundaries.size()));
    for (const auto& b : d_boundaries) { w.value(b.region); w.value(b.native_id); w.value(b.id); w.string(b.name); }
    w.array(d_reference_axial_edges); w.array(d_axial_edges); w.value(d_geometry_state.epoch);
    return std::move(w.data);
}

std::shared_ptr<MultiRegionMesh> MultiRegionMesh::deserialize_partition(
    std::span<const char> packet, Teuchos::RCP<const Teuchos::Comm<int>> comm)
{
    if (comm.is_null()) throw std::invalid_argument("Partition geometry requires a communicator.");
    Reader r{packet};
    if (r.value<uint64_t>() != partition_magic) throw std::invalid_argument("Unsupported composite partition packet version.");
    auto mesh = std::shared_ptr<MultiRegionMesh>(new MultiRegionMesh(PartitionTag{}));
    mesh->d_comm = std::move(comm);
    for (size_t count = r.count(); count; --count) mesh->d_regions.push_back(read_region(r));
    for (size_t count = r.count(); count; --count) mesh->d_interfaces.push_back(read_interface(r));
    mesh->d_tolerance = r.value<InterfaceTolerance>();
    mesh->d_cells = r.array<ID>(); mesh->d_faces = r.array<ID>(); mesh->d_nodes = r.array<ID>(); mesh->d_native_faces = r.array<ID>();
    mesh->d_boundaries.resize(r.count(sizeof(uint64_t) + 2 * sizeof(int) + sizeof(uint64_t)));
    for (auto& b : mesh->d_boundaries)
    { b.region = r.value<size_t>(); b.native_id = r.value<int>(); b.id = r.value<int>(); b.name = r.string(); }
    mesh->d_reference_axial_edges = r.array<real_t>(); mesh->d_axial_edges = r.array<real_t>();
    mesh->d_geometry_state.epoch = r.value<uint64_t>();
    if (!r.data.empty()) throw std::invalid_argument("Trailing composite partition bytes.");
    const auto regions = mesh->d_regions.size();
    if (!regions || !std::isfinite(mesh->d_tolerance.absolute_length) || !std::isfinite(mesh->d_tolerance.relative)
        || mesh->d_tolerance.absolute_length < 0 || !(mesh->d_tolerance.relative > 0))
        throw std::invalid_argument("Invalid partition catalog dimensions.");
    std::set<std::string> region_names;
    for (size_t region = 0; region < regions; ++region)
        if (!region_names.insert(mesh->region_name(region)).second)
            throw std::invalid_argument("Duplicate partition region name.");

    // Directories, strides and inverse lookups are derived only from validated
    // descriptors. They are never accepted as executable indexing data on wire.
    mesh->d_explicit.resize(mesh->d_interfaces.size());
    const auto boundary_exists = [&](size_t region, int boundary) {
        if (region >= regions) throw std::invalid_argument("Partition interface region is out of range.");
        const auto ids = mesh->topology(region, [](const auto& t) { return t.boundary_batch_ids(); });
        if (std::ranges::find(ids, boundary) == ids.end())
            throw std::invalid_argument("Partition interface boundary is unknown.");
    };
    const auto check_face = [&](size_t region, ID face, int boundary) {
        if (face >= mesh->region_layout(region).faces)
            throw std::invalid_argument("Partition interface face is out of range.");
        mesh->topology(region, [&](const auto& t) {
            // Sparse records retain the complete catalog, but only local face
            // geometry. Validate available incidence without expanding residency.
            if constexpr (std::same_as<std::remove_cvref_t<decltype(t)>, ExplicitRegionProvider>)
                if (!t.has_face(face)) return;
            if (t.boundary_id(face) != boundary || t.neighbor_cell(face) != invalid_cell_id())
                throw std::invalid_argument("Partition interface face is not on its declared exterior boundary.");
        });
    };
    const auto sort_unique = [](auto& pairs) {
        std::ranges::sort(pairs);
        if (std::adjacent_find(pairs.begin(), pairs.end(), [](auto a, auto b) { return a.first == b.first; }) != pairs.end())
            throw std::invalid_argument("Partition interface repeats a native face.");
    };
    std::vector<size_t> component(regions);
    std::iota(component.begin(), component.end(), 0);
    const auto root = [&](size_t a) { while (component[a] != a) a = component[a]; return a; };
    for (size_t i = 0; i < mesh->d_interfaces.size(); ++i)
    {
        auto& lookup = mesh->d_explicit[i];
        size_t first = 0, second = 0;
        if (auto* s = std::get_if<StructuredPatchInterface>(&mesh->d_interfaces[i]))
        {
            if (s->first.region >= regions || s->second.region >= regions)
                throw std::invalid_argument("Partition patch region is out of range.");
            mesh->normalize(s->first); mesh->normalize(s->second); s->validate_mapping();
            first = s->first.region; second = s->second.region;
            if (s->periodic_translation && (!std::isfinite(s->periodic_translation->x)
                || !std::isfinite(s->periodic_translation->y) || !std::isfinite(s->periodic_translation->z)))
                throw std::invalid_argument("Partition periodic translation is not finite.");
            if (first == second && !s->periodic_translation)
                throw std::invalid_argument("Partition self interface requires a periodic translation.");
        }
        else if (const auto* e = std::get_if<ExplicitConformingInterface>(&mesh->d_interfaces[i]))
        {
            first = e->first_region; second = e->second_region;
            boundary_exists(first, e->first_boundary); boundary_exists(second, e->second_boundary);
            if (e->faces.empty()) throw std::invalid_argument("Empty partition conforming interface.");
            lookup.first_to_second = e->faces;
            for (auto [a, b] : e->faces)
            {
                check_face(first, a, e->first_boundary); check_face(second, b, e->second_boundary);
                if (first == second && a == b) throw std::invalid_argument("Partition interface pairs a face with itself.");
                lookup.second_to_first.emplace_back(b, a);
            }
        }
        else
        {
            const auto& n = std::get<NonconformingInterface>(mesh->d_interfaces[i]);
            first = n.coarse_region; second = n.fine_region;
            boundary_exists(first, n.coarse_boundary); boundary_exists(second, n.fine_boundary);
            if (first == second || n.faces.empty()) throw std::invalid_argument("Invalid partition coarse/fine interface.");
            for (size_t group = 0; group < n.faces.size(); ++group)
            {
                const auto& faces = n.faces[group];
                check_face(first, faces.coarse_face, n.coarse_boundary);
                if (faces.fine_faces.empty()) throw std::invalid_argument("Empty partition fine-face coverage.");
                lookup.second_to_first.emplace_back(faces.coarse_face, group);
                for (auto f : faces.fine_faces)
                {
                    check_face(second, f, n.fine_boundary);
                    lookup.first_to_second.emplace_back(f, faces.coarse_face);
                }
            }
        }
        sort_unique(lookup.first_to_second); sort_unique(lookup.second_to_first);
        component[root(first)] = root(second);
    }
    for (size_t region = 1; region < regions; ++region)
        if (root(region) != root(0)) throw std::invalid_argument("Disconnected partition region catalog.");
    std::vector<StructuredPatch> patches;
    std::set<RegionFace> explicit_faces;
    const auto unique_explicit = [&](size_t region, ID face) {
        if (!explicit_faces.insert({region, face}).second)
            throw std::invalid_argument("Partition face belongs to multiple interfaces.");
    };
    for (const auto& interface : mesh->d_interfaces)
        if (const auto* s = std::get_if<StructuredPatchInterface>(&interface))
        {
            for (const auto& patch : {s->first, s->second})
            {
                for (const auto& previous : patches)
                    if (patch.region == previous.region && patch.boundary == previous.boundary
                        && patch.begin[0] < previous.begin[0] + previous.extent[0]
                        && previous.begin[0] < patch.begin[0] + patch.extent[0]
                        && patch.begin[1] < previous.begin[1] + previous.extent[1]
                        && previous.begin[1] < patch.begin[1] + patch.extent[1])
                        throw std::invalid_argument("Overlapping partition interface patches.");
                patches.push_back(patch);
            }
        }
        else if (const auto* e = std::get_if<ExplicitConformingInterface>(&interface))
            for (auto [a, b] : e->faces)
            { unique_explicit(e->first_region, a); unique_explicit(e->second_region, b); }
        else
        {
            const auto& n = std::get<NonconformingInterface>(interface);
            for (const auto& f : n.faces)
            {
                unique_explicit(n.coarse_region, f.coarse_face);
                for (auto fine : f.fine_faces) unique_explicit(n.fine_region, fine);
            }
        }
    for (const auto& patch : patches)
        for (auto face : explicit_faces)
            if (face.region == patch.region && mesh->patch_coordinate(patch, face.face))
                throw std::invalid_argument("Partition face belongs to both structured and explicit interfaces.");
    mesh->initialize_interface_directory();
    for (const auto* offsets : {&mesh->d_cells, &mesh->d_faces, &mesh->d_nodes, &mesh->d_native_faces})
        if (offsets->size() != regions + 1 || offsets->front() != 0
            || !std::is_sorted(offsets->begin(), offsets->end()))
            throw std::invalid_argument("Invalid partition region offsets.");
    size_t boundary_cursor = 0;
    std::map<int, std::string> boundary_names;
    std::map<std::string, int> boundary_ids;
    for (size_t region = 0; region < regions; ++region)
    {
        const auto& l = mesh->region_layout(region);
        auto& directory = mesh->d_region_interfaces[region];
        directory.boundary_begin = boundary_cursor;
        std::set<int> native_ids;
        while (boundary_cursor < mesh->d_boundaries.size() && mesh->d_boundaries[boundary_cursor].region == region)
        {
            const auto& b = mesh->d_boundaries[boundary_cursor++];
            boundary_exists(b.region, b.native_id);
            if (b.id < 0 || b.name.empty() || !native_ids.insert(b.native_id).second)
                throw std::invalid_argument("Invalid or repeated partition boundary.");
            const auto [by_id, new_id] = boundary_names.emplace(b.id, b.name);
            const auto [by_name, new_name] = boundary_ids.emplace(b.name, b.id);
            if ((!new_id && by_id->second != b.name) || (!new_name && by_name->second != b.id))
                throw std::invalid_argument("Inconsistent partition boundary names.");
        }
        directory.boundary_end = boundary_cursor;
        if (mesh->d_cells[region+1] - mesh->d_cells[region] != l.cells
            || mesh->d_nodes[region+1] - mesh->d_nodes[region] != l.nodes
            || mesh->d_native_faces[region+1] - mesh->d_native_faces[region] != l.faces
            || directory.removed_faces > l.faces
            || mesh->d_faces[region+1] - mesh->d_faces[region] != l.faces - directory.removed_faces
            || !std::is_sorted(directory.offsets.begin(), directory.offsets.end())
            || directory.offsets.back() > mesh->d_interface_sides.size()
            || directory.boundary_begin > directory.boundary_end || directory.boundary_end > mesh->d_boundaries.size())
            throw std::invalid_argument("Inconsistent partition region catalog.");
    }
    if (boundary_cursor != mesh->d_boundaries.size()
        || (!boundary_names.empty() && static_cast<size_t>(boundary_names.rbegin()->first) != boundary_names.size() - 1))
        throw std::invalid_argument("Invalid partition boundary ordering or IDs.");
    for (const auto* edges : {&mesh->d_reference_axial_edges, &mesh->d_axial_edges})
        if (edges->size() != 2 || !std::isfinite((*edges)[0]) || !std::isfinite((*edges)[1]) || !((*edges)[1] > (*edges)[0]))
            throw std::invalid_argument("Invalid partition axial transformation.");
    mesh->d_num_cells = mesh->d_num_local_cells = mesh->d_num_owned_cells = mesh->d_cells.back();
    mesh->d_num_faces = mesh->d_num_owned_faces = mesh->d_faces.back(); mesh->d_num_nodes = mesh->d_nodes.back();
    mesh->d_indexer = {mesh->d_num_cells, mesh->d_num_faces, mesh->d_num_nodes};
    mesh->d_partition_resident = true;
    mesh->validate_static();
    return mesh;
}
} // namespace SimpleFluid::Meshes
