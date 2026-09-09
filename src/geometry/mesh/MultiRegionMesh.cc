/**
 * @file MultiRegionMesh.cc
 * @author islandox
 * @brief Validated compact conforming interface resolution.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#include "geometry/mesh/MultiRegionMesh.hh"
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace SimpleFluid::Meshes
{
namespace
{
using ID = uint64_t;
ID checked_add(ID a, ID b)
{
    if (b >= std::numeric_limits<ID>::max() - a) throw std::overflow_error("Composite entity count overflow.");
    return a + b;
}
std::array<size_t, 2> parameter_shape(const StructuredPatch& p)
{
    return {p.extent[p.axes[0]], p.extent[p.axes[1]]};
}
void check_permutation(const std::array<unsigned, 2>& p)
{
    if (p[0] > 1 || p[1] > 1 || p[0] == p[1]) throw std::invalid_argument("Invalid tangential permutation.");
}
OrthogonalIndexer rectilinear_indexer(const RegionLayout& l)
{
    return {static_cast<unsigned>(l.extents[0]), static_cast<unsigned>(l.extents[1]), static_cast<unsigned>(l.extents[2])};
}
std::array<size_t, 2> tangent_axes(int boundary)
{
    std::array<size_t, 2> result{};
    size_t t = 0;
    for (size_t axis = 0; axis < 3; ++axis) if (axis != static_cast<size_t>(boundary / 2)) result[t++] = axis;
    return result;
}
} // namespace

void StructuredPatchInterface::validate_mapping() const
{
    check_permutation(permutation);
    check_permutation(first.axes);
    check_permutation(second.axes);
    const auto a = parameter_shape(first), b = parameter_shape(second);
    for (size_t i = 0; i < 2; ++i)
        if (!a[i] || b[i] != a[permutation[i]]) throw std::invalid_argument("Structured patch extents are not bijective.");
}
std::array<size_t, 2> StructuredPatchInterface::map(std::array<size_t, 2> p) const
{
    validate_mapping();
    const auto a = parameter_shape(first), b = parameter_shape(second);
    if (p[0] >= a[0] || p[1] >= a[1]) throw std::out_of_range("Interface source index out of bounds.");
    std::array<size_t, 2> result;
    for (size_t i = 0; i < 2; ++i) result[i] = reversed[i] ? b[i] - 1 - p[permutation[i]] : p[permutation[i]];
    return result;
}
std::array<size_t, 2> StructuredPatchInterface::inverse(std::array<size_t, 2> p) const
{
    validate_mapping();
    const auto b = parameter_shape(second);
    if (p[0] >= b[0] || p[1] >= b[1]) throw std::out_of_range("Interface target index out of bounds.");
    std::array<size_t, 2> result;
    for (size_t i = 0; i < 2; ++i) result[permutation[i]] = reversed[i] ? b[i] - 1 - p[i] : p[i];
    return result;
}
const RegionLayout& MultiRegionMesh::region_layout(size_t r) const
{
    return std::visit([](const auto& r) -> const RegionLayout& { return r.layout(); }, region(r));
}
const std::string& MultiRegionMesh::region_name(size_t r) const
{
    return std::visit([](const auto& r) -> const std::string& { return r.name(); }, region(r));
}
void MultiRegionMesh::normalize(StructuredPatch& p) const
{
    const auto& l = region_layout(p.region);
    if (l.family != RegionLayout::Family::Rectilinear || p.boundary < 0 || p.boundary >= 6)
        throw std::invalid_argument("Structured interfaces require Cartesian exterior patches.");
    check_permutation(p.axes);
    const auto axes = tangent_axes(p.boundary);
    for (size_t i = 0; i < 2; ++i)
    {
        const auto n = l.extents[axes[i]];
        if (p.begin[i] >= n) throw std::out_of_range("Structured patch begin out of bounds.");
        if (p.extent[i] == 0) p.extent[i] = n - p.begin[i];
        if (p.extent[i] > n - p.begin[i]) throw std::out_of_range("Structured patch extent out of bounds.");
    }
}
ID MultiRegionMesh::patch_face(const StructuredPatch& p, std::array<size_t, 2> coordinate) const
{
    const auto shape = parameter_shape(p);
    if (coordinate[0] >= shape[0] || coordinate[1] >= shape[1]) throw std::out_of_range("Patch coordinate out of bounds.");
    const auto& l = region_layout(p.region);
    const auto axes = tangent_axes(p.boundary);
    std::array<unsigned, 3> n{};
    n[p.boundary / 2] = p.boundary % 2 ? static_cast<unsigned>(l.extents[p.boundary / 2]) : 0;
    for (size_t i = 0; i < 2; ++i)
        n[axes[p.axes[i]]] = static_cast<unsigned>(p.begin[p.axes[i]] + (p.reversed[i] ? shape[i] - 1 - coordinate[i] : coordinate[i]));
    return rectilinear_indexer(l).face_ordinal({n[0], n[1], n[2], static_cast<uint8_t>(p.boundary / 2)});
}
std::optional<std::array<size_t, 2>> MultiRegionMesh::patch_coordinate(const StructuredPatch& p, ID face) const
{
    const auto& l = region_layout(p.region);
    if (face >= l.faces) return {};
    const auto f = rectilinear_indexer(l).face_id(face);
    if (f.orientation != p.boundary / 2) return {};
    const std::array<size_t, 3> n{f.i, f.j, f.k};
    if (n[f.orientation] != (p.boundary % 2 ? l.extents[f.orientation] : 0)) return {};
    const auto axes = tangent_axes(p.boundary), shape = parameter_shape(p);
    std::array<size_t, 2> result;
    for (size_t i = 0; i < 2; ++i)
    {
        const auto a = p.axes[i];
        if (n[axes[a]] < p.begin[a] || n[axes[a]] - p.begin[a] >= p.extent[a]) return {};
        const auto c = n[axes[a]] - p.begin[a];
        result[i] = p.reversed[i] ? shape[i] - 1 - c : c;
    }
    return result;
}
size_t MultiRegionMesh::patch_count_before(const StructuredPatch& p, ID face) const
{
    auto natural = p;
    natural.axes = {0, 1}; natural.reversed = {};
    const auto origin = patch_face(natural, {0, 0});
    if (face <= origin) return 0;
    const auto indexer = rectilinear_indexer(region_layout(p.region));
    const auto axes = tangent_axes(p.boundary);
    size_t fast = 0, slow = 1;
    if (indexer.face_strides[p.boundary / 2][axes[fast]] > indexer.face_strides[p.boundary / 2][axes[slow]]) std::swap(fast, slow);
    const auto stride_fast = indexer.face_strides[p.boundary / 2][axes[fast]];
    const auto stride_slow = indexer.face_strides[p.boundary / 2][axes[slow]];
    const auto delta = face - origin;
    const auto rows = std::min<size_t>(p.extent[slow], delta / stride_slow);
    if (rows == p.extent[slow]) return p.extent[0] * p.extent[1];
    const auto remaining = delta - rows * stride_slow;
    return rows * p.extent[fast] + std::min<size_t>(p.extent[fast], remaining / stride_fast + (remaining % stride_fast != 0));
}
size_t MultiRegionMesh::removed_before(size_t r, ID face) const
{
    size_t count = 0;
    for (size_t i = 0; i < d_interfaces.size(); ++i)
    {
        if (const auto* s = std::get_if<StructuredPatchInterface>(&d_interfaces[i]))
        {
            if (s->second.region == r) count += patch_count_before(s->second, face);
        }
        else
        {
            const auto& e = std::get<ExplicitConformingInterface>(d_interfaces[i]);
            if (e.second_region == r)
            {
                const auto& pairs = d_explicit[i].second_to_first;
                count += std::lower_bound(pairs.begin(), pairs.end(), face,
                    [](const auto& pair, ID f) { return pair.first < f; }) - pairs.begin();
            }
        }
    }
    return count;
}
std::optional<RegionFace> MultiRegionMesh::partner(RegionFace f, bool first) const
{
    for (size_t i = 0; i < d_interfaces.size(); ++i)
    {
        if (const auto* s = std::get_if<StructuredPatchInterface>(&d_interfaces[i]))
        {
            const auto& p = first ? s->first : s->second;
            if (p.region != f.region) continue;
            if (const auto coordinate = patch_coordinate(p, f.face))
            {
                const auto& other = first ? s->second : s->first;
                return RegionFace{other.region, patch_face(other, first ? s->map(*coordinate) : s->inverse(*coordinate))};
            }
        }
        else
        {
            const auto& e = std::get<ExplicitConformingInterface>(d_interfaces[i]);
            if ((first ? e.first_region : e.second_region) != f.region) continue;
            const auto& pairs = first ? d_explicit[i].first_to_second : d_explicit[i].second_to_first;
            const auto it = std::lower_bound(pairs.begin(), pairs.end(), f.face,
                [](const auto& pair, ID face) { return pair.first < face; });
            if (it != pairs.end() && it->first == f.face)
                return RegionFace{first ? e.second_region : e.first_region, it->second};
        }
    }
    return {};
}
bool MultiRegionMesh::stitched(RegionFace f) const { return partner(f, true).has_value() || partner(f, false).has_value(); }
ID MultiRegionMesh::canonical_face(RegionFace f) const
{
    if (f.face >= region_layout(f.region).faces) throw std::out_of_range("Native face out of bounds.");
    if (topology(f.region, [&](const auto& t) { return t.neighbor_cell(f.face); }) == invalid_cell_id())
        if (const auto owner = partner(f, false)) f = *owner;
    return d_faces[f.region] + f.face - removed_before(f.region, f.face);
}
RegionFace MultiRegionMesh::native_face(ID f) const
{
    if (f >= num_faces()) throw std::out_of_range("Composite face out of bounds.");
    const size_t r = std::upper_bound(d_faces.begin(), d_faces.end(), f) - d_faces.begin() - 1;
    const ID ordinal = f - d_faces[r];
    ID lo = ordinal, hi = region_layout(r).faces;
    // Rank/select through removed rectangular ranges or sorted irregular IDs.
    while (lo < hi)
    {
        const auto mid = lo + (hi - lo) / 2;
        if (mid + 1 - removed_before(r, mid + 1) <= ordinal) lo = mid + 1;
        else hi = mid;
    }
    return {r, lo};
}
std::pair<size_t, ID> MultiRegionMesh::native_cell(ID c) const
{
    if (c >= num_cells()) throw std::out_of_range("Composite cell out of bounds.");
    const size_t r = std::upper_bound(d_cells.begin(), d_cells.end(), c) - d_cells.begin() - 1;
    return {r, c - d_cells[r]};
}
ID MultiRegionMesh::composite_cell(size_t r, ID c) const
{
    if (c >= region_layout(r).cells) throw std::out_of_range("Native cell out of bounds.");
    return d_cells[r] + c;
}
void MultiRegionMesh::validate_static() const
{
    for (const auto& r : d_regions) std::visit([](const auto& r) { r.validate_static(); }, r);
}
void MultiRegionMesh::check_cell_id(ID c) const { validate_static(); if (c >= num_cells()) throw std::out_of_range("Composite cell out of bounds."); }
void MultiRegionMesh::check_face_id(ID f) const { validate_static(); if (f >= num_faces()) throw std::out_of_range("Composite face out of bounds."); }
void MultiRegionMesh::check_node_id(ID n) const { validate_static(); if (n >= num_nodes()) throw std::out_of_range("Composite node out of bounds."); }
real_t MultiRegionMesh::cell_volume_impl(ID c) const
{
    const auto [r, id] = native_cell(c); return geometry(r, [&](const auto& g) { return g.cell_volume(id); });
}
MultiRegionMesh::Vec3 MultiRegionMesh::cell_centroid_impl(ID c) const
{
    const auto [r, id] = native_cell(c); return geometry(r, [&](const auto& g) { return g.cell_centroid(id); });
}
EntityRange<ID> MultiRegionMesh::cell_faces_impl(ID c) const
{
    const auto [r, id] = native_cell(c);
    const auto count = topology(r, [&](const auto& t) { return t.cell_faces(id).size(); });
    return {this, c, count, [](const void* source, size_t c, size_t i) -> ID
    {
        const auto& mesh = *static_cast<const MultiRegionMesh*>(source);
        mesh.validate_static();
        const auto [r, id] = mesh.native_cell(c);
        return mesh.canonical_face({r, mesh.topology(r, [&](const auto& t) { return t.cell_faces(id)[i]; })});
    }};
}
ID MultiRegionMesh::owner_cell_impl(ID f) const
{
    const auto n = native_face(f);
    return composite_cell(n.region, topology(n.region, [&](const auto& t) { return t.owner_cell(n.face); }));
}
ID MultiRegionMesh::neighbor_cell_impl(ID f) const
{
    const auto n = native_face(f);
    const auto c = topology(n.region, [&](const auto& t) { return t.neighbor_cell(n.face); });
    if (c != invalid_cell_id()) return composite_cell(n.region, c);
    if (const auto other = partner(n, true))
        return composite_cell(other->region, topology(other->region, [&](const auto& t) { return t.owner_cell(other->face); }));
    return invalid_cell_id();
}

real_t MultiRegionMesh::face_area_impl(ID f) const
{
    const auto n = native_face(f); return geometry(n.region, [&](const auto& g) { return g.face_area(n.face); });
}
MultiRegionMesh::Vec3 MultiRegionMesh::face_centroid_impl(ID f) const
{
    const auto n = native_face(f); return geometry(n.region, [&](const auto& g) { return g.face_centroid(n.face); });
}
MultiRegionMesh::Vec3 MultiRegionMesh::face_normal_impl(ID f) const
{
    const auto n = native_face(f); return geometry(n.region, [&](const auto& g) { return g.face_area_vector(n.face) / g.face_area(n.face); });
}
MultiRegionMesh::Vec3 MultiRegionMesh::node_coordinates_impl(ID n) const
{
    const size_t r = std::upper_bound(d_nodes.begin(), d_nodes.end(), n) - d_nodes.begin() - 1;
    return geometry(r, [&](const auto& g) { return g.node_coordinates(n - d_nodes[r]); });
}
int MultiRegionMesh::boundary_id_impl(ID f) const
{
    const auto n = native_face(f);
    const auto id = topology(n.region, [&](const auto& t) { return t.boundary_id(n.face); });
    if (id == invalid_boundary_id || partner(n, true)) return invalid_boundary_id;
    for (const auto& b : d_boundaries) if (b.region == n.region && b.native_id == id) return b.id;
    return invalid_boundary_id;
}
const std::string& MultiRegionMesh::boundary_batch_name_impl(int b) const
{
    for (const auto& entry : d_boundaries) if (entry.id == b) return entry.name;
    throw std::out_of_range("Unknown composite boundary.");
}
std::vector<int> MultiRegionMesh::boundary_batch_ids_impl() const
{
    std::vector<int> ids;
    for (const auto& b : d_boundaries) if (std::find(ids.begin(), ids.end(), b.id) == ids.end()) ids.push_back(b.id);
    return ids;
}
int MultiRegionMesh::num_boundary_batches_impl() const noexcept
{
    int count = 0;
    for (const auto& b : d_boundaries) count = std::max(count, b.id + 1);
    return count;
}
MeshUtils::CellType MultiRegionMesh::cell_type(ID c) const
{
    check_cell_id(c); const auto [r, id] = native_cell(c); return topology(r, [&](const auto& t) { return t.cell_type(id); });
}
EntityRange<ID> MultiRegionMesh::cell_nodes(ID c) const
{
    check_cell_id(c); const auto [r, id] = native_cell(c);
    return {this, c, topology(r, [&](const auto& t) { return t.cell_nodes(id).size(); }),
        [](const void* source, size_t c, size_t i) -> ID
        {
            const auto& m = *static_cast<const MultiRegionMesh*>(source); const auto [r, id] = m.native_cell(c);
            return m.d_nodes[r] + m.topology(r, [&](const auto& t) { return t.cell_nodes(id)[i]; });
        }};
}
EntityRange<ID> MultiRegionMesh::face_nodes(ID f) const
{
    check_face_id(f); const auto n = native_face(f);
    return {this, f, topology(n.region, [&](const auto& t) { return t.face_nodes(n.face).size(); }),
        [](const void* source, size_t f, size_t i) -> ID
        {
            const auto& m = *static_cast<const MultiRegionMesh*>(source); const auto n = m.native_face(f);
            return m.d_nodes[n.region] + m.topology(n.region, [&](const auto& t) { return t.face_nodes(n.face)[i]; });
        }};
}

void MultiRegionMesh::validate_pair(RegionFace a, RegionFace b) const
{
    if (a.region == b.region || a.face >= region_layout(a.region).faces || b.face >= region_layout(b.region).faces)
        throw std::invalid_argument("Conforming interface requires valid faces in distinct regions.");
    for (auto f : {a, b})
        if (topology(f.region, [&](const auto& t) { return t.neighbor_cell(f.face); }) != invalid_cell_id())
            throw std::invalid_argument("Only standalone exterior faces may be stitched.");
    const auto area_a = geometry(a.region, [&](const auto& g) { return g.face_area(a.face); });
    const auto area_b = geometry(b.region, [&](const auto& g) { return g.face_area(b.face); });
    const auto ca = geometry(a.region, [&](const auto& g) { return g.face_centroid(a.face); });
    const auto cb = geometry(b.region, [&](const auto& g) { return g.face_centroid(b.face); });
    const auto va = geometry(a.region, [&](const auto& g) { return g.face_area_vector(a.face); });
    const auto vb = geometry(b.region, [&](const auto& g) { return g.face_area_vector(b.face); });
    const auto nodes_a = topology(a.region, [&](const auto& t) { return t.face_nodes(a.face); });
    const auto nodes_b = topology(b.region, [&](const auto& t) { return t.face_nodes(b.face); });
    if (nodes_a.size() != nodes_b.size()) throw std::invalid_argument("Nonconforming interface vertex count or subdivision.");
    real_t length = std::sqrt(std::max(area_a, area_b));
    for (auto n : nodes_a)
        length = std::max(length, (geometry(a.region, [&](const auto& g) { return g.node_coordinates(n); }) - ca).norm());
    const auto eps = d_tolerance.absolute_length + d_tolerance.relative * length;
    const auto area_eps = d_tolerance.relative * std::max(area_a, area_b) + d_tolerance.absolute_length * length;
    if (!std::isfinite(area_a) || !std::isfinite(area_b) || area_a <= 0 || area_b <= 0
        || std::abs(area_a - area_b) > area_eps || (ca - cb).norm() > eps
        || (va / area_a + vb / area_b).norm() > d_tolerance.relative)
        throw std::invalid_argument("Interface areas, centroids or opposite outward orientations do not match.");
    // Cyclic or reversed vertex-loop agreement verifies both vertices and edges.
    bool matching_loop = false;
    for (size_t shift = 0; shift < nodes_b.size(); ++shift)
        for (int direction : {1, -1})
        {
            bool match = true;
            for (size_t i = 0; i < nodes_a.size(); ++i)
            {
                const size_t j = (shift + (direction == 1 ? i : nodes_b.size() - i)) % nodes_b.size();
                const auto x = geometry(a.region, [&](const auto& g) { return g.node_coordinates(nodes_a[i]); });
                const auto y = geometry(b.region, [&](const auto& g) { return g.node_coordinates(nodes_b[j]); });
                match = match && (x - y).norm() <= eps;
            }
            matching_loop = matching_loop || match;
        }
    if (!matching_loop) throw std::invalid_argument("Interface face shapes or vertex correspondence do not match.");
}

void MultiRegionMesh::validate_regions() const
{
    validate_static();
    std::set<std::string> names;
    for (size_t r = 0; r < d_regions.size(); ++r)
    {
        if (!names.insert(region_name(r)).second) throw std::invalid_argument("Topology region names must be unique.");
        const auto& l = region_layout(r);
        if (!l.cells || !l.faces || !l.nodes) throw std::invalid_argument("Empty topology region.");
        for (ID c = 0; c < l.cells; ++c)
        {
            const auto v = geometry(r, [&](const auto& g) { return g.cell_volume(c); });
            if (!(v > 0) || !std::isfinite(v)) throw std::invalid_argument("Region cell volume must be finite and positive.");
            if (topology(r, [&](const auto& t) { return t.cell_type(c); }) == MeshUtils::CellType::INVALID)
                throw std::invalid_argument("Composite regions currently support HEX_8 and triangular prisms.");
        }
        // Rectilinear templates are connected by construction. Validate explicit
        // and base-plane adjacency without expanding layered cell connectivity.
        if (l.family != RegionLayout::Family::Rectilinear)
        {
            const size_t count = l.family == RegionLayout::Family::Extruded ? l.extents[0] : l.cells;
            std::vector<unsigned char> seen(count);
            std::vector<ID> queue{0}; seen[0] = 1;
            for (size_t i = 0; i < queue.size(); ++i)
                for (const auto f : topology(r, [&](const auto& t) { return t.cell_faces(queue[i]); }))
                {
                    const auto owner = topology(r, [&](const auto& t) { return t.owner_cell(f); });
                    const auto neighbor = topology(r, [&](const auto& t) { return t.neighbor_cell(f); });
                    const auto other = owner == queue[i] ? neighbor : owner;
                    if (other < count && !seen[other]) { seen[other] = 1; queue.push_back(other); }
                }
            if (queue.size() != count) throw std::invalid_argument("Disconnected constituent regions are unsupported.");
        }
    }
}

MultiRegionMesh::MultiRegionMesh(std::vector<Region> regions, std::vector<Interface> interfaces,
    InterfaceTolerance tolerance, BoundaryNamePolicy names)
    : d_regions(std::move(regions)), d_interfaces(std::move(interfaces)), d_tolerance(tolerance)
{
    // The communicator-size check precedes any map setup or collective.
    if (Tpetra::getDefaultComm()->getSize() != 1) throw std::runtime_error("MultiRegionMesh supports serial communicators only.");
    if (d_regions.empty()) throw std::invalid_argument("MultiRegionMesh requires at least one region.");
    if (!std::isfinite(tolerance.absolute_length) || !std::isfinite(tolerance.relative)
        || tolerance.absolute_length < 0 || tolerance.relative <= 0)
        throw std::invalid_argument("Invalid interface matching tolerance.");
    validate_regions();
    d_explicit.resize(d_interfaces.size());
    std::vector<size_t> component(d_regions.size()); std::iota(component.begin(), component.end(), 0);
    const auto root = [&](size_t r) { while (component[r] != r) r = component[r]; return r; };
    for (size_t i = 0; i < d_interfaces.size(); ++i)
    {
        size_t first, second;
        if (auto* s = std::get_if<StructuredPatchInterface>(&d_interfaces[i]))
        {
            normalize(s->first); normalize(s->second); s->validate_mapping();
            first = s->first.region; second = s->second.region;
        }
        else
        {
            const auto& e = std::get<ExplicitConformingInterface>(d_interfaces[i]);
            first = e.first_region; second = e.second_region;
            region(first); region(second);
            for (auto [r, b] : {std::pair{first, e.first_boundary}, std::pair{second, e.second_boundary}})
            {
                const auto ids = topology(r, [&](const auto& t) { return t.boundary_batch_ids(); });
                if (std::find(ids.begin(), ids.end(), b) == ids.end()) throw std::invalid_argument("Unknown interface boundary patch.");
                size_t count = 0;
                for (ID f = 0; f < region_layout(r).faces; ++f)
                    count += topology(r, [&](const auto& t) { return t.boundary_id(f); }) == b;
                if (count == 0 || count != e.faces.size()) throw std::invalid_argument("Explicit mapping is incomplete for its declared patch.");
            }
            auto& lookup = d_explicit[i]; lookup.first_to_second = e.faces;
            for (const auto& [a, b] : e.faces)
            {
                if (a >= region_layout(first).faces || b >= region_layout(second).faces)
                    throw std::out_of_range("Explicit interface face out of bounds.");
                if (topology(first, [&](const auto& t) { return t.boundary_id(a); }) != e.first_boundary
                    || topology(second, [&](const auto& t) { return t.boundary_id(b); }) != e.second_boundary)
                    throw std::invalid_argument("Explicit interface face is outside its declared patch.");
                lookup.second_to_first.emplace_back(b, a);
            }
            for (auto* pairs : {&lookup.first_to_second, &lookup.second_to_first})
            {
                std::sort(pairs->begin(), pairs->end());
                if (std::adjacent_find(pairs->begin(), pairs->end(), [](const auto& a, const auto& b) { return a.first == b.first; }) != pairs->end())
                    throw std::invalid_argument("Duplicate explicit interface face.");
            }
        }
        if (first == second) throw std::invalid_argument("Self interfaces and composite periodicity are unsupported.");
        component[root(first)] = root(second);
    }
    for (size_t r = 1; r < d_regions.size(); ++r)
        if (root(r) != root(0)) throw std::invalid_argument("Disconnected composite fluid domains are unsupported.");
    auto check_pair = [&](RegionFace a, RegionFace b)
    {
        validate_pair(a, b);
        for (auto f : {a, b})
        {
            size_t uses = 0;
            for (size_t i = 0; i < d_interfaces.size(); ++i)
            {
                if (const auto* s = std::get_if<StructuredPatchInterface>(&d_interfaces[i]))
                {
                    for (const auto& p : {s->first, s->second}) uses += p.region == f.region && patch_coordinate(p, f.face).has_value();
                }
                else
                {
                    const auto& e = std::get<ExplicitConformingInterface>(d_interfaces[i]);
                    const auto contains = [&](const auto& pairs)
                    {
                        const auto it = std::lower_bound(pairs.begin(), pairs.end(), f.face,
                            [](const auto& pair, ID id) { return pair.first < id; });
                        return it != pairs.end() && it->first == f.face;
                    };
                    if (e.first_region == f.region) uses += contains(d_explicit[i].first_to_second);
                    if (e.second_region == f.region) uses += contains(d_explicit[i].second_to_first);
                }
            }
            if (uses != 1) throw std::invalid_argument("A native face is stitched more than once.");
        }
    };
    for (const auto& interface : d_interfaces)
    {
        if (const auto* s = std::get_if<StructuredPatchInterface>(&interface))
        {
            const auto shape = parameter_shape(s->first);
            for (size_t j = 0; j < shape[1]; ++j) for (size_t i = 0; i < shape[0]; ++i)
            {
                const std::array<size_t, 2> p{i, j};
                if (s->inverse(s->map(p)) != p) throw std::invalid_argument("Interface inverse mismatch.");
                check_pair({s->first.region, patch_face(s->first, p)}, {s->second.region, patch_face(s->second, s->map(p))});
            }
        }
        else
        {
            const auto& e = std::get<ExplicitConformingInterface>(interface);
            for (const auto& [a, b] : e.faces) check_pair({e.first_region, a}, {e.second_region, b});
        }
    }
    d_cells.push_back(0); d_faces.push_back(0); d_nodes.push_back(0);
    int next_boundary = 0;
    for (size_t r = 0; r < d_regions.size(); ++r)
    {
        const auto& l = region_layout(r);
        d_cells.push_back(checked_add(d_cells.back(), l.cells));
        d_faces.push_back(checked_add(d_faces.back(), l.faces - removed_before(r, l.faces)));
        d_nodes.push_back(checked_add(d_nodes.back(), l.nodes));
        auto ids = topology(r, [&](const auto& t) { return t.boundary_batch_ids(); });
        std::sort(ids.begin(), ids.end());
        for (const auto b : ids)
        {
            bool has_exterior = false;
            for (ID f = 0; f < l.faces && !has_exterior; ++f)
                has_exterior = topology(r, [&](const auto& t) { return t.boundary_id(f); }) == b && !stitched({r, f});
            if (!has_exterior) continue;
            auto name = topology(r, [&](const auto& t) { return std::string(t.boundary_batch_name(b)); });
            int id = next_boundary;
            if (names == BoundaryNamePolicy::NamespaceRegions)
            {
                name = region_name(r) + "/" + name;
                for (const auto& entry : d_boundaries)
                    if (entry.name == name)
                        throw std::invalid_argument("Composite boundary names collide; use unique names or an explicit merge policy.");
            }
            else for (const auto& entry : d_boundaries) if (entry.name == name) { id = entry.id; break; }
            if (id == next_boundary)
            {
                if (next_boundary == std::numeric_limits<int>::max()) throw std::overflow_error("Composite boundary ID overflow.");
                ++next_boundary;
            }
            d_boundaries.push_back({r, b, id, std::move(name)});
        }
    }
    d_num_cells = d_num_local_cells = d_num_owned_cells = d_cells.back();
    d_num_faces = d_num_owned_faces = d_faces.back(); d_num_nodes = d_nodes.back();
    d_indexer = {d_num_cells, d_num_faces, d_num_nodes};
}

MeshStorageReport MultiRegionMesh::storage_report() const
{
    MeshStorageReport result;
    result.objects = sizeof(*this) + d_regions.capacity() * sizeof(Region);
    result.indexing = (d_cells.capacity() + d_faces.capacity() + d_nodes.capacity()) * sizeof(ID);
    result.interfaces = d_interfaces.capacity() * sizeof(Interface) + d_explicit.capacity() * sizeof(ExplicitLookup);
    for (size_t i = 0; i < d_interfaces.size(); ++i)
    {
        if (const auto* e = std::get_if<ExplicitConformingInterface>(&d_interfaces[i])) result.interfaces += e->faces.capacity() * sizeof(e->faces[0]);
        result.interfaces += (d_explicit[i].first_to_second.capacity() + d_explicit[i].second_to_first.capacity()) * sizeof(std::pair<ID, ID>);
    }
    result.boundaries = d_boundaries.capacity() * sizeof(Boundary);
    std::set<const void*> topology_seen, geometry_seen, providers_seen, native_objects_seen;
    const auto count_provider = [&](const auto& provider)
    {
        if (providers_seen.insert(&provider).second) result.objects += sizeof(provider);
        if constexpr (requires { provider.native(); })
        {
            const auto& native = provider.native();
            if (native_objects_seen.insert(&native).second)
            {
                result.objects += sizeof(native);
                if constexpr (requires { native.topology(); }) result.objects -= sizeof(native.topology());
            }
        }
    };
    for (const auto& region : d_regions) std::visit([&](const auto& r)
    {
        count_provider(r.topology()); count_provider(r.geometry());
        if (topology_seen.insert(r.topology().storage_identity()).second) result.topology += r.topology().storage_report().topology;
        if (geometry_seen.insert(r.geometry().storage_identity()).second) result.geometry += r.geometry().storage_report().geometry;
    }, region);
    return result;
}
VTUWriter::TopologyHandle MultiRegionMesh::vtu_topology() const
{
    validate_static();
    VTUWriter::VectorData points; VTUWriter::Int64Data connectivity, offsets; VTUWriter::UInt8Data types;
    points.reserve(num_nodes()); offsets.reserve(num_cells()); types.reserve(num_cells());
    for (ID n = 0; n < num_nodes(); ++n) points.push_back(node_coordinates(n));
    for (ID c = 0; c < num_cells(); ++c)
    {
        const auto nodes = cell_nodes(c);
        for (const auto n : nodes) connectivity.push_back(static_cast<global_index_t>(n));
        offsets.push_back(static_cast<global_index_t>(connectivity.size()));
        types.push_back(MeshUtils::vtu_cell_type_code(cell_type(c)));
    }
    return VTUWriter::make_topology(std::move(points), std::move(connectivity), std::move(offsets), std::move(types));
}
} // namespace SimpleFluid::Meshes
