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
#include <Teuchos_CommHelpers.hpp>
#include <sstream>
#include <iomanip>
#include <unordered_map>
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
    return {static_cast<unsigned>(l.extents[0]), static_cast<unsigned>(l.extents[1]), static_cast<unsigned>(l.extents[2]),l.periodic[0],l.periodic[1],l.periodic[2]};
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
    if ((l.family != RegionLayout::Family::Rectilinear && l.family != RegionLayout::Family::Cylindrical) || p.boundary < 0 || p.boundary >= 6)
        throw std::invalid_argument("Structured interfaces require supported coordinate-normal exterior patches.");
    if(l.periodic[p.boundary/2]) throw std::invalid_argument("A native periodic axis has no exterior patch to stitch.");
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
std::optional<size_t> MultiRegionMesh::structured_boundary_size(size_t r, int boundary) const
{
    const auto& layout = region_layout(r);
    if ((layout.family != RegionLayout::Family::Rectilinear && layout.family != RegionLayout::Family::Cylindrical)
        || boundary < 0 || boundary >= 6) return {};
    if (layout.periodic[boundary/2]) return size_t{0};
    const auto axes = tangent_axes(boundary);
    return layout.extents[axes[0]] * layout.extents[axes[1]];
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
void MultiRegionMesh::initialize_interface_directory()
{
    d_region_interfaces.resize(d_regions.size());
    const auto visit_sides = [&](auto&& visitor)
    {
        for (size_t i = 0; i < d_interfaces.size(); ++i)
            if (const auto* s = std::get_if<StructuredPatchInterface>(&d_interfaces[i]))
            {
                visitor(i, true, s->first.region, static_cast<size_t>(s->first.boundary));
                visitor(i, false, s->second.region, static_cast<size_t>(s->second.boundary));
            }
            else if (const auto* e = std::get_if<ExplicitConformingInterface>(&d_interfaces[i]))
            {
                visitor(i, true, e->first_region, size_t{6});
                visitor(i, false, e->second_region, size_t{6});
            }
            else
            {
                const auto& n = std::get<NonconformingInterface>(d_interfaces[i]);
                visitor(i, true, n.fine_region, size_t{6});
                visitor(i, false, n.coarse_region, size_t{6});
            }
    };
    visit_sides([&](size_t i, bool first, size_t r, size_t bucket)
    {
        auto& directory = d_region_interfaces[r];
        ++directory.offsets[bucket];
        if (!first)
        {
            const auto* s = std::get_if<StructuredPatchInterface>(&d_interfaces[i]);
            const auto removed = s ? s->second.extent[0] * s->second.extent[1]
                : d_explicit[i].second_to_first.size();
            directory.removed_faces = checked_add(directory.removed_faces, removed);
        }
    });
    size_t offset = 0;
    for (auto& directory : d_region_interfaces)
    {
        for (size_t bucket = 0; bucket < 7; ++bucket)
        {
            const auto count = directory.offsets[bucket];
            directory.offsets[bucket] = offset;
            offset += count;
        }
        directory.offsets.back() = offset;
    }
    d_interface_sides.resize(offset);
    // The cursor is temporary descriptor-sized construction storage.
    std::vector<std::array<size_t, 8>> cursor;
    cursor.reserve(d_region_interfaces.size());
    for (const auto& directory : d_region_interfaces) cursor.push_back(directory.offsets);
    visit_sides([&](size_t i, bool first, size_t r, size_t bucket)
    {
        d_interface_sides[cursor[r][bucket]++] = {i, first};
    });
}
std::span<const MultiRegionMesh::InterfaceSide> MultiRegionMesh::interface_sides(size_t r, size_t bucket) const
{
    const auto& offsets = d_region_interfaces.at(r).offsets;
    return std::span<const InterfaceSide>(d_interface_sides).subspan(offsets[bucket], offsets[bucket+1]-offsets[bucket]);
}
std::span<const MultiRegionMesh::InterfaceSide> MultiRegionMesh::structured_interface_sides(RegionFace face) const
{
    const auto& layout = region_layout(face.region);
    if ((layout.family != RegionLayout::Family::Rectilinear && layout.family != RegionLayout::Family::Cylindrical)
        || face.face >= layout.faces) return {};
    const auto native = rectilinear_indexer(layout).face_id(face.face);
    if (layout.periodic[native.orientation]) return {};
    const std::array<size_t, 3> coordinate{native.i, native.j, native.k};
    const auto normal = coordinate[native.orientation];
    if (normal != 0 && normal != layout.extents[native.orientation]) return {};
    return interface_sides(face.region, 2 * native.orientation + (normal != 0));
}
size_t MultiRegionMesh::removed_before(size_t r, ID face) const
{
    const auto& directory = d_region_interfaces.at(r);
    if (!directory.removed_faces) return 0;
    size_t count=0;
    for (size_t side = directory.offsets.front(); side < directory.offsets.back(); ++side)
    {
        const auto [i, first] = d_interface_sides[side];
        if (first) continue;
        if(const auto* s=std::get_if<StructuredPatchInterface>(&d_interfaces[i]))
            count+=patch_count_before(s->second,face);
        else
        {
            const auto& pairs=d_explicit[i].second_to_first;
            count+=std::lower_bound(pairs.begin(),pairs.end(),face,[](const auto& p,ID f){return p.first<f;})-pairs.begin();
        }
    }
    return count;
}
std::optional<RegionFace> MultiRegionMesh::partner(RegionFace f, bool first) const
{
    for (const auto side : structured_interface_sides(f))
    {
        if (side.first != first) continue;
        const auto& s = std::get<StructuredPatchInterface>(d_interfaces[side.interface]);
        const auto& p = first ? s.first : s.second;
        if (const auto coordinate = patch_coordinate(p, f.face))
        {
            const auto& other = first ? s.second : s.first;
            return RegionFace{other.region, patch_face(other, first ? s.map(*coordinate) : s.inverse(*coordinate))};
        }
    }
    for (const auto [i, side_first] : interface_sides(f.region, 6))
    {
        if (side_first != first) continue;
        if (const auto* refined=std::get_if<NonconformingInterface>(&d_interfaces[i]))
        {
            if (!first) continue;
            const auto& pairs=d_explicit[i].first_to_second;
            const auto it=std::lower_bound(pairs.begin(),pairs.end(),f.face,[](const auto& p,ID f){return p.first<f;});
            if(it!=pairs.end() && it->first==f.face) return RegionFace{refined->coarse_region,it->second};
        }
        else
        {
            const auto& e = std::get<ExplicitConformingInterface>(d_interfaces[i]);
            const auto& pairs = first ? d_explicit[i].first_to_second : d_explicit[i].second_to_first;
            const auto it = std::lower_bound(pairs.begin(), pairs.end(), f.face,
                [](const auto& pair, ID face) { return pair.first < face; });
            if (it != pairs.end() && it->first == f.face)
                return RegionFace{first ? e.second_region : e.first_region, it->second};
        }
    }
    return {};
}
bool MultiRegionMesh::stitched(RegionFace f) const { return refinement(f).has_value() || partner(f,true).has_value() || partner(f,false).has_value(); }
std::optional<MultiRegionMesh::RefinedFaceView> MultiRegionMesh::refinement(RegionFace f) const
{
    for(const auto [i, first] : interface_sides(f.region, 6))
        if(const auto* n=std::get_if<NonconformingInterface>(&d_interfaces[i]); n && !first)
        {
            const auto& pairs=d_explicit[i].second_to_first;
            const auto it=std::lower_bound(pairs.begin(),pairs.end(),f.face,[](const auto& p,ID f){return p.first<f;});
            if(it!=pairs.end() && it->first==f.face) return RefinedFaceView{n->fine_region,n->faces[it->second].fine_faces};
        }
    return {};
}
MultiRegionMesh::ID MultiRegionMesh::encode_native_face(RegionFace f) const
{
    if(f.face>=region_layout(f.region).faces) throw std::out_of_range("Native face out of bounds.");
    return d_native_faces[f.region]+f.face;
}
RegionFace MultiRegionMesh::decode_native_face(ID key) const
{
    const auto r=static_cast<size_t>(std::upper_bound(d_native_faces.begin(),d_native_faces.end(),key)-d_native_faces.begin()-1);
    return {r,key-d_native_faces[r]};
}
EntityRange<MultiRegionMesh::ID> MultiRegionMesh::logical_faces(RegionFace f) const
{
    const auto refined=refinement(f);
    return {this,encode_native_face(f),refined?refined->faces.size():1,[](const void* source,size_t key,size_t i)->ID
    {
        const auto& m=*static_cast<const MultiRegionMesh*>(source); const auto f=m.decode_native_face(key);
        if(const auto refined=m.refinement(f)) return m.canonical_face({refined->region,refined->faces[i]});
        return m.canonical_face(f);
    }};
}
EntityRange<FaceFluxWeight> MultiRegionMesh::face_flux_weights(RegionFace f) const
{
    return {this,encode_native_face(f),logical_faces(f).size(),[](const void* source,size_t key,size_t i)->FaceFluxWeight
    {
        const auto& m=*static_cast<const MultiRegionMesh*>(source); const auto native=m.decode_native_face(key);
        const auto f=m.logical_faces(native)[i];
        const auto native_vector=m.transformed_area_vector(m.geometry(native.region,[&](const auto& g){return g.face_area_vector(native.face);}));
        const auto vector=m.face_area_vector(f);
        real_t total_area=0;
        for(const auto logical:m.logical_faces(native)) total_area+=m.face_area(logical);
        // Normalize flux transfer, never geometry: allowed matching tolerance
        // must not create or destroy a supplied integrated native-face flux.
        return {f,(vector.dot(native_vector)>0?1.0:-1.0)*m.face_area(f)/total_area};
    }};
}
InterfaceFacePolygon MultiRegionMesh::interface_polygon(RegionFace f) const
{
    InterfaceFacePolygon p;
    p.area=geometry(f.region,[&](const auto& g){return g.face_area(f.face);});
    p.centroid=geometry(f.region,[&](const auto& g){return g.face_centroid(f.face);});
    p.area_vector=geometry(f.region,[&](const auto& g){return g.face_area_vector(f.face);});
    for(const auto n:topology(f.region,[&](const auto& t){return t.face_nodes(f.face);}))
        p.vertices.push_back(geometry(f.region,[&](const auto& g){return g.node_coordinates(n);}));
    return p;
}
ID MultiRegionMesh::canonical_face(RegionFace f) const
{
    if (f.face >= region_layout(f.region).faces) throw std::out_of_range("Native face out of bounds.");
    if (refinement(f)) throw std::invalid_argument("A subdivided native face has multiple logical faces; use logical_faces().");
    if (topology(f.region, [&](const auto& t) { return t.neighbor_cell(f.face); }) == invalid_cell_id())
        if (const auto owner = partner(f, false)) f = *owner;
    return d_faces[f.region] + f.face - removed_before(f.region, f.face);
}
RegionFace MultiRegionMesh::native_face(ID f) const
{
    if (f >= num_faces()) throw std::out_of_range("Composite face out of bounds.");
    const size_t r = std::upper_bound(d_faces.begin(), d_faces.end(), f) - d_faces.begin() - 1;
    const ID ordinal = f - d_faces[r];
    if (!d_region_interfaces[r].removed_faces) return {r, ordinal};
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
thread_local const MultiRegionMesh::ExecutionView* MultiRegionMesh::ExecutionView::d_current = nullptr;
bool MultiRegionMesh::executing_on_this_thread() const noexcept
{
    for (auto* view = ExecutionView::d_current; view; view = view->d_previous)
        if (view->d_mesh == this) return true;
    return false;
}
MultiRegionMesh::ExecutionView::ExecutionView(const MultiRegionMesh* mesh)
    : d_mesh(mesh), d_previous(d_current)
{
    if (mesh && !mesh->executing_on_this_thread())
    {
        d_leases.reserve(1 + 4 * mesh->d_regions.size());
        d_leases.push_back(mesh->acquire_geometry_read());
        const auto lock = [&](const auto& provider)
        {
            if constexpr (requires { provider.acquire_geometry_read(); })
                d_leases.push_back(provider.acquire_geometry_read());
            if constexpr (requires { provider.native(); })
                d_leases.push_back(provider.native().acquire_geometry_read());
            if constexpr (requires { provider.native_topology().acquire_geometry_read(); })
                d_leases.push_back(provider.native_topology().acquire_geometry_read());
        };
        for (const auto& region : mesh->d_regions)
            std::visit([&](const auto& r) { lock(r.topology()); lock(r.geometry()); }, region);
        mesh->validate_static();
    }
    d_current = this;
}
MultiRegionMesh::ExecutionView::~ExecutionView() { d_current = d_previous; }
void MultiRegionMesh::validate_query() const
{
    if (!executing_on_this_thread()) validate_static();
}
void MultiRegionMesh::check_cell_id(ID c) const { validate_query(); if (c >= num_cells()) throw std::out_of_range("Composite cell out of bounds."); }
void MultiRegionMesh::check_face_id(ID f) const { validate_query(); if (f >= num_faces()) throw std::out_of_range("Composite face out of bounds."); }
void MultiRegionMesh::check_node_id(ID n) const { validate_query(); if (n >= num_nodes()) throw std::out_of_range("Composite node out of bounds."); }
real_t MultiRegionMesh::cell_volume_impl(ID c) const
{
    const auto [r, id] = native_cell(c); return axial_scale() * geometry(r, [&](const auto& g) { return g.cell_volume(id); });
}
MultiRegionMesh::Vec3 MultiRegionMesh::cell_centroid_impl(ID c) const
{
    const auto [r, id] = native_cell(c); return transformed_point(geometry(r, [&](const auto& g) { return g.cell_centroid(id); }));
}
EntityRange<ID> MultiRegionMesh::cell_faces_impl(ID c) const
{
    const auto [r,id]=native_cell(c);
    size_t count=0;
    for(const auto f:topology(r,[&](const auto& t){return t.cell_faces(id);}))
        count+=logical_faces({r,f}).size();
    return {this,c,count,[](const void* source,size_t c,size_t i)->ID
    {
        const auto& m=*static_cast<const MultiRegionMesh*>(source); m.validate_query();
        const auto [r,id]=m.native_cell(c);
        for(const auto f:m.topology(r,[&](const auto& t){return t.cell_faces(id);}))
        {
            const auto faces=m.logical_faces({r,f});
            if(i<faces.size()) return faces[i];
            i-=faces.size();
        }
        throw std::out_of_range("Composite cell-face entry out of bounds.");
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
    const auto n = native_face(f); return transformed_area_vector(geometry(n.region, [&](const auto& g) { return g.face_area_vector(n.face); })).norm();
}
MultiRegionMesh::Vec3 MultiRegionMesh::face_centroid_impl(ID f) const
{
    const auto n = native_face(f); return transformed_point(geometry(n.region, [&](const auto& g) { return g.face_centroid(n.face); }));
}
MultiRegionMesh::Vec3 MultiRegionMesh::face_normal_impl(ID f) const
{
    const auto n = native_face(f); const auto vector = transformed_area_vector(geometry(n.region, [&](const auto& g) { return g.face_area_vector(n.face); }));
    return vector / vector.norm();
}
MultiRegionMesh::Vec3 MultiRegionMesh::node_coordinates_impl(ID n) const
{
    const size_t r = std::upper_bound(d_nodes.begin(), d_nodes.end(), n) - d_nodes.begin() - 1;
    return transformed_point(geometry(r, [&](const auto& g) { return g.node_coordinates(n - d_nodes[r]); }));
}
int MultiRegionMesh::boundary_id_impl(ID f) const
{
    const auto n = native_face(f);
    const auto id = topology(n.region, [&](const auto& t) { return t.boundary_id(n.face); });
    if (id == invalid_boundary_id || partner(n, true)) return invalid_boundary_id;
    const auto& directory = d_region_interfaces[n.region];
    for (size_t i = directory.boundary_begin; i < directory.boundary_end; ++i)
        if (d_boundaries[i].native_id == id) return d_boundaries[i].id;
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

void MultiRegionMesh::validate_pair(RegionFace a, RegionFace b, std::optional<Vec3> translation) const
{
    if ((!translation && a.region == b.region) || a.face >= region_layout(a.region).faces || b.face >= region_layout(b.region).faces)
        throw std::invalid_argument("Conforming interface requires valid faces in distinct regions.");
    if (translation && (!std::isfinite(translation->norm()) || (a.region==b.region
        && topology(a.region,[&](const auto& t){return t.owner_cell(a.face);})==topology(b.region,[&](const auto& t){return t.owner_cell(b.face);}))))
        throw std::invalid_argument("Periodic interfaces require finite translations and distinct incident cells.");
    const auto periodic_shift=translation.value_or(Vec3{});
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
        || std::abs(area_a - area_b) > area_eps || (ca + periodic_shift - cb).norm() > eps
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
                match = match && (x + periodic_shift - y).norm() <= eps;
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
        for(size_t a=0;a<3;++a) if(l.periodic[a] && l.extents[a]<2)
            throw std::invalid_argument("Composite native periodic directions require at least two cells.");
        if(l.family==RegionLayout::Family::Cylindrical)
            geometry(r,[&](const auto& g)
            {
                if constexpr (requires { g.native(); })
                    if constexpr (std::same_as<std::remove_cvref_t<decltype(g.native())>,OrthogonalCylindrial3D>)
                    {
                        const auto& theta=g.native().cell_edges()[1];
                        for(size_t i=1;i<theta.size();++i)
                            if(theta[i]-theta[i-1]>=std::acos(-1.0))
                                throw std::invalid_argument("Composite cylindrical cells require angular widths below pi for nondegenerate HEX output.");
                    }
            });
        const auto validate_volume = [](real_t volume)
        {
            if (!(volume > 0) || !std::isfinite(volume))
                throw std::invalid_argument("Region cell volume must be finite and positive.");
        };
        const auto validate_center = [](Vec3 center)
        {
            if(!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(center.z))
                throw std::invalid_argument("Region cell centroid must be finite.");
        };
        const auto validate_type = [&](ID c)
        {
            if (topology(r, [&](const auto& t) { return t.cell_type(c); }) == MeshUtils::CellType::INVALID)
                throw std::invalid_argument("Composite regions currently support HEX_8 and triangular prisms.");
        };
        // Positive tensor-product widths make the actual minimum/maximum
        // volume cells sufficient, including floating-point under/overflow.
        // Sample centers through the provider to preserve its native arithmetic.
        geometry(r, [&](const auto& g)
        {
            const auto& source = [&]() -> const auto&
            {
                if constexpr (requires { g.native(); }) return g.native();
                else return g;
            }();
            using Source = std::remove_cvref_t<decltype(source)>;
            if constexpr (std::same_as<Source, RectilinearGeometry> || std::same_as<Source, OrthogonalCartesian3D>)
            {
                const auto& edges = source.cell_edges();
                std::array<unsigned, 3> minimum{}, maximum{};
                const auto indexer = rectilinear_indexer(l);
                for (size_t axis = 0; axis < 3; ++axis)
                {
                    real_t smallest = std::numeric_limits<real_t>::infinity(), largest = 0;
                    for (size_t i = 0; i < l.extents[axis]; ++i)
                    {
                        const auto width = edges[axis][i+1] - edges[axis][i];
                        if (!std::isfinite(edges[axis][i]) || !std::isfinite(edges[axis][i+1])
                            || !(width > 0) || !std::isfinite(width))
                            throw std::invalid_argument("Region coordinate widths must be finite and positive.");
                        if (width < smallest) { smallest = width; minimum[axis] = static_cast<unsigned>(i); }
                        if (width > largest) { largest = width; maximum[axis] = static_cast<unsigned>(i); }
                        std::array<unsigned, 3> cell{}; cell[axis] = static_cast<unsigned>(i);
                        validate_center(g.cell_centroid(indexer.cell_ordinal({cell[0], cell[1], cell[2]})));
                    }
                }
                validate_volume(g.cell_volume(indexer.cell_ordinal({minimum[0], minimum[1], minimum[2]})));
                validate_volume(g.cell_volume(indexer.cell_ordinal({maximum[0], maximum[1], maximum[2]})));
                validate_type(0);
            }
            else if constexpr (std::same_as<Source, ExtrudedGeometry> || std::same_as<Source, SemiStructuredXY_Z>)
            {
                const auto& edges = source.z_edges();
                size_t minimum = 0, maximum = 0;
                real_t smallest = std::numeric_limits<real_t>::infinity(), largest = 0;
                for (size_t k = 0; k < l.extents[2]; ++k)
                {
                    const auto width = edges[k+1] - edges[k];
                    if (!std::isfinite(edges[k]) || !std::isfinite(edges[k+1]) || !(width > 0) || !std::isfinite(width))
                        throw std::invalid_argument("Region coordinate widths must be finite and positive.");
                    if (width < smallest) { smallest = width; minimum = k; }
                    if (width > largest) { largest = width; maximum = k; }
                    validate_center(g.cell_centroid(k*l.extents[0]));
                }
                for (ID base = 0; base < l.extents[0]; ++base)
                {
                    validate_volume(g.cell_volume(minimum*l.extents[0]+base));
                    validate_volume(g.cell_volume(maximum*l.extents[0]+base));
                    validate_center(g.cell_centroid(base));
                    validate_type(base);
                }
            }
            else
                for (ID c = 0; c < l.cells; ++c)
                {
                    validate_volume(g.cell_volume(c));
                    validate_center(g.cell_centroid(c));
                    validate_type(c);
                }
        });
        // Rectilinear templates are connected by construction. Validate explicit
        // and base-plane adjacency without expanding layered cell connectivity.
        if (l.family != RegionLayout::Family::Rectilinear && l.family != RegionLayout::Family::Cylindrical)
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
    const auto comm = Tpetra::getDefaultComm();
    std::exception_ptr error;
    try
    {
        if (comm->getSize() > 1)
            for (size_t r = 0; r < d_regions.size(); ++r)
                if (region_layout(r).family == RegionLayout::Family::Explicit)
                    throw std::invalid_argument("Composite MPI requires compact implicit regions; partitioned explicit regions are not supported.");
        initialize_regions(tolerance, names);
    }
    catch (...) { error = std::current_exception(); }
    int failed = error ? 1 : 0, any_failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (any_failed)
    {
        if (error) std::rethrow_exception(error);
        throw std::invalid_argument("Composite mesh construction failed on another rank.");
    }
    if (comm->getSize() > 1)
    {
        auto local = configuration_signature();
        const int overflow = local.size() > static_cast<size_t>(std::numeric_limits<int>::max());
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &overflow, &any_failed);
        if (any_failed) throw std::overflow_error("Composite descriptor exceeds collective message bounds.");
        int length = comm->getRank() == 0 ? static_cast<int>(local.size()) : 0;
        Teuchos::broadcast(*comm, 0, 1, &length);
        std::string reference = comm->getRank() == 0 ? local : std::string(static_cast<size_t>(length), '\0');
        Teuchos::broadcast(*comm, 0, length, reference.data());
        failed = reference != local;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
        if (any_failed) throw std::invalid_argument("Composite geometry/topology descriptors differ across ranks.");
    }
}

void MultiRegionMesh::initialize_regions(InterfaceTolerance tolerance, BoundaryNamePolicy names)
{
    if (d_regions.empty()) throw std::invalid_argument("MultiRegionMesh requires at least one region.");
    if (!std::isfinite(tolerance.absolute_length) || !std::isfinite(tolerance.relative)
        || tolerance.absolute_length < 0 || tolerance.relative <= 0)
        throw std::invalid_argument("Invalid interface matching tolerance.");
    validate_regions();
    d_explicit.resize(d_interfaces.size());
    std::vector<size_t> component(d_regions.size()); std::iota(component.begin(), component.end(), 0);
    const auto root=[&](size_t r){while(component[r]!=r) r=component[r]; return r;};
    const auto patch_count=[&](size_t r,int boundary)
    {
        const auto ids=topology(r,[&](const auto& t){return t.boundary_batch_ids();});
        if(std::find(ids.begin(),ids.end(),boundary)==ids.end()) throw std::invalid_argument("Unknown interface boundary patch.");
        if (const auto count = structured_boundary_size(r, boundary)) return *count;
        size_t count=0;
        for(ID f=0;f<region_layout(r).faces;++f) count+=topology(r,[&](const auto& t){return t.boundary_id(f);})==boundary;
        return count;
    };
    const auto check_face=[&](size_t r,ID face,int boundary)
    {
        if(face>=region_layout(r).faces) throw std::out_of_range("Interface face out of bounds.");
        if(topology(r,[&](const auto& t){return t.boundary_id(face);})!=boundary
            || topology(r,[&](const auto& t){return t.neighbor_cell(face);})!=invalid_cell_id())
            throw std::invalid_argument("An interface face must be exterior and belong to its declared patch.");
    };
    const auto sort_unique=[](auto& pairs)
    {
        std::sort(pairs.begin(),pairs.end());
        if(std::adjacent_find(pairs.begin(),pairs.end(),[](const auto& a,const auto& b){return a.first==b.first;})!=pairs.end())
            throw std::invalid_argument("A native face occurs multiple times in an interface.");
    };
    for(size_t i=0;i<d_interfaces.size();++i)
    {
        size_t first,second;
        auto& lookup=d_explicit[i];
        if(auto* structured=std::get_if<StructuredPatchInterface>(&d_interfaces[i]))
        {
            normalize(structured->first); normalize(structured->second); structured->validate_mapping();
            first=structured->first.region; second=structured->second.region;
        }
        else if(const auto* explicit_interface=std::get_if<ExplicitConformingInterface>(&d_interfaces[i]))
        {
            const auto& e=*explicit_interface; first=e.first_region; second=e.second_region;
            if(e.faces.empty() || patch_count(first,e.first_boundary)!=e.faces.size()
                || patch_count(second,e.second_boundary)!=e.faces.size())
                throw std::invalid_argument("Explicit mapping is incomplete for its declared patch.");
            lookup.first_to_second=e.faces;
            for(const auto& [a,b]:e.faces)
            { check_face(first,a,e.first_boundary); check_face(second,b,e.second_boundary); lookup.second_to_first.emplace_back(b,a); }
        }
        else
        {
            const auto& e=std::get<NonconformingInterface>(d_interfaces[i]); first=e.coarse_region; second=e.fine_region;
            if(e.faces.empty() || patch_count(first,e.coarse_boundary)!=e.faces.size())
                throw std::invalid_argument("Nonconforming mapping is incomplete for the coarse patch.");
            for(size_t group=0;group<e.faces.size();++group)
            {
                const auto& mapping=e.faces[group]; check_face(first,mapping.coarse_face,e.coarse_boundary);
                if(mapping.fine_faces.empty()) throw std::invalid_argument("Coarse faces require nonempty fine coverage.");
                lookup.second_to_first.emplace_back(mapping.coarse_face,group);
                for(const auto fine:mapping.fine_faces)
                { check_face(second,fine,e.fine_boundary); lookup.first_to_second.emplace_back(fine,mapping.coarse_face); }
            }
            if(patch_count(second,e.fine_boundary)!=lookup.first_to_second.size())
                throw std::invalid_argument("Nonconforming mapping is incomplete for the fine patch.");
        }
        sort_unique(lookup.first_to_second); sort_unique(lookup.second_to_first);
        const auto* structured=std::get_if<StructuredPatchInterface>(&d_interfaces[i]);
        if(first==second && (!structured || !structured->periodic_translation))
            throw std::invalid_argument("Self interfaces require an explicit periodic mapping.");
        component[root(first)]=root(second);
    }
    for(size_t r=1;r<d_regions.size();++r)
        if(root(r)!=root(0)) throw std::invalid_argument("Disconnected composite fluid domains are unsupported.");
    initialize_interface_directory();
    const auto unique_face=[&](RegionFace f)
    {
        if(interface_uses(f)!=1) throw std::invalid_argument("A native face is stitched more than once.");
    };
    for(const auto& interface:d_interfaces)
    {
        if(const auto* structured=std::get_if<StructuredPatchInterface>(&interface))
        {
            const auto& e=*structured; const auto shape=parameter_shape(e.first);
            for(size_t j=0;j<shape[1];++j) for(size_t i=0;i<shape[0];++i)
            {
                const std::array<size_t,2> p{i,j};
                if(e.inverse(e.map(p))!=p) throw std::invalid_argument("Interface inverse mismatch.");
                const RegionFace a{e.first.region,patch_face(e.first,p)}, b{e.second.region,patch_face(e.second,e.map(p))};
                unique_face(a); unique_face(b); validate_pair(a,b,e.periodic_translation);
            }
        }
        else if(const auto* explicit_interface=std::get_if<ExplicitConformingInterface>(&interface))
        {
            const auto& e=*explicit_interface;
            for(const auto& [x,y]:e.faces)
            {
                const RegionFace a{e.first_region,x}, b{e.second_region,y};
                unique_face(a); unique_face(b); validate_pair(a,b);
            }
        }
        else
        {
            const auto& e=std::get<NonconformingInterface>(interface);
            for(const auto& group:e.faces)
            {
                const RegionFace coarse{e.coarse_region,group.coarse_face}; unique_face(coarse);
                std::vector<InterfaceFacePolygon> fine;
                for(const auto id:group.fine_faces) { const RegionFace f{e.fine_region,id}; unique_face(f); fine.push_back(interface_polygon(f)); }
                validate_planar_subdivision(interface_polygon(coarse),fine,tolerance.absolute_length,tolerance.relative);
            }
        }
    }
    d_cells.push_back(0); d_faces.push_back(0); d_nodes.push_back(0); d_native_faces.push_back(0);
    int next_boundary = 0;
    for (size_t r = 0; r < d_regions.size(); ++r)
    {
        const auto& l = region_layout(r);
        d_cells.push_back(checked_add(d_cells.back(), l.cells));
        d_faces.push_back(checked_add(d_faces.back(), l.faces - d_region_interfaces[r].removed_faces));
        d_nodes.push_back(checked_add(d_nodes.back(), l.nodes));
        d_native_faces.push_back(checked_add(d_native_faces.back(), l.faces));
        auto ids = topology(r, [&](const auto& t) { return t.boundary_batch_ids(); });
        std::sort(ids.begin(), ids.end());
        d_region_interfaces[r].boundary_begin = d_boundaries.size();
        for (const auto b : ids)
        {
            bool has_exterior = false;
            if (const auto count = structured_boundary_size(r, b))
            {
                size_t joined = 0;
                for (const auto [i, first] : interface_sides(r, static_cast<size_t>(b)))
                {
                    const auto& s = std::get<StructuredPatchInterface>(d_interfaces[i]);
                    const auto& patch = first ? s.first : s.second;
                    joined += patch.extent[0] * patch.extent[1];
                }
                for (const auto [i, first] : interface_sides(r, 6))
                {
                    const auto& pairs = first ? d_explicit[i].first_to_second : d_explicit[i].second_to_first;
                    if (!pairs.empty() && topology(r, [&](const auto& t) { return t.boundary_id(pairs.front().first); }) == b)
                        joined += pairs.size();
                }
                // Unique-face and interface-coverage checks have already run.
                has_exterior = joined < *count;
            }
            else
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
        d_region_interfaces[r].boundary_end = d_boundaries.size();
    }
    d_num_cells = d_num_local_cells = d_num_owned_cells = d_cells.back();
    d_num_faces = d_num_owned_faces = d_faces.back(); d_num_nodes = d_nodes.back();
    d_indexer = {d_num_cells, d_num_faces, d_num_nodes};
    real_t bottom = std::numeric_limits<real_t>::infinity(), top = -bottom;
    for (size_t r = 0; r < d_regions.size(); ++r)
        geometry(r, [&](const auto& g)
        {
            const auto bounds = [&](const auto& source)
            {
                if constexpr (requires { source.cell_edges(); })
                { bottom = std::min(bottom, source.cell_edges()[2].front()); top = std::max(top, source.cell_edges()[2].back()); }
                else if constexpr (requires { source.z_edges(); })
                { bottom = std::min(bottom, source.z_edges().front()); top = std::max(top, source.z_edges().back()); }
                else if constexpr (requires { source.nodes(); })
                    for (const auto& p : source.nodes()) { bottom = std::min(bottom,p.z); top = std::max(top,p.z); }
            };
            if constexpr (requires { g.native(); }) bounds(g.native()); else bounds(g);
        });
    if (!std::isfinite(bottom) || !std::isfinite(top) || !(top > bottom))
        throw std::invalid_argument("Composite axial extent must be finite and positive.");
    d_reference_axial_edges = d_axial_edges = {bottom, top};
}

size_t MultiRegionMesh::interface_uses(RegionFace f) const
{
    size_t uses=0;
    const auto contains=[&](const auto& pairs)
    {
        const auto it=std::lower_bound(pairs.begin(),pairs.end(),f.face,[](const auto& p,ID face){return p.first<face;});
        return it!=pairs.end() && it->first==f.face;
    };
    for(const auto [i, first] : structured_interface_sides(f))
    {
        const auto& s=std::get<StructuredPatchInterface>(d_interfaces[i]);
        uses+=patch_coordinate(first?s.first:s.second,f.face).has_value();
    }
    for(const auto [i, first] : interface_sides(f.region, 6))
        uses+=contains(first?d_explicit[i].first_to_second:d_explicit[i].second_to_first);
    return uses;
}

bool MultiRegionMesh::supports_axial_motion() const noexcept
{
    for(size_t r=0;r<d_regions.size();++r)
        if(region_layout(r).family==RegionLayout::Family::Explicit) return false;
    for(const auto& interface:d_interfaces)
        if(const auto* s=std::get_if<StructuredPatchInterface>(&interface); s && s->periodic_translation && s->periodic_translation->z!=0)
            return false;
    return true;
}
MultiRegionMesh::Vec3 MultiRegionMesh::periodic_translation(ID f) const
{
    const auto n=native_face(f);
    for(const auto [i, first] : structured_interface_sides(n))
    {
        const auto& s=std::get<StructuredPatchInterface>(d_interfaces[i]);
        if(first && s.periodic_translation && patch_coordinate(s.first,n.face))
        { auto shift=*s.periodic_translation; shift.z*=axial_scale(); return shift; }
    }
    return {};
}
MultiRegionMesh::Vec3 MultiRegionMesh::face_center_vector(ID f,ID c) const
{
    check_face_id(f); check_cell_id(c);
    if(c==owner_cell(f)) return face_centroid(f)-cell_centroid(c);
    if(c==neighbor_cell(f)) return face_centroid(f)+periodic_translation(f)-cell_centroid(c);
    throw std::invalid_argument("Cell is not incident to composite face.");
}
MultiRegionMesh::Vec3 MultiRegionMesh::cell_center_vector(ID f,ID c) const
{
    check_face_id(f); check_cell_id(c);
    const auto other=opposite_cell(f,c);
    if(other==invalid_cell_id()) throw std::invalid_argument("Exterior face has no adjacent cell.");
    return cell_centroid(other)-cell_centroid(c)+periodic_translation(f)*(c==owner_cell(f)?-1.0:1.0);
}

real_t MultiRegionMesh::axial_scale() const noexcept
{
    return (d_axial_edges[1]-d_axial_edges[0])/(d_reference_axial_edges[1]-d_reference_axial_edges[0]);
}
MultiRegionMesh::Vec3 MultiRegionMesh::transformed_point(Vec3 p) const
{
    p.z = d_axial_edges[0] + (p.z-d_reference_axial_edges[0])*axial_scale();
    return p;
}
MultiRegionMesh::Vec3 MultiRegionMesh::transformed_area_vector(Vec3 v) const
{
    const auto scale=axial_scale(); return {v.x*scale,v.y*scale,v.z};
}
void MultiRegionMesh::replace_axial_edges_fixed_topology(ArrReal edges)
{
    require_geometry_writable();
    validate_static();
    if (edges.size()!=2 || !std::isfinite(edges[0]) || !std::isfinite(edges[1]) || !(edges[1]>edges[0]))
        throw std::invalid_argument("Composite planar ALE requires a finite positive affine axial interval.");
    if (d_geometry_state.epoch==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Composite geometry epoch overflow.");
    d_axial_edges=std::move(edges); ++d_geometry_state.epoch;
}

MeshStorageReport MultiRegionMesh::storage_report() const
{
    MeshStorageReport result;
    result.geometry = (d_reference_axial_edges.capacity() + d_axial_edges.capacity()) * sizeof(real_t);
    result.objects = sizeof(*this) + d_regions.capacity() * sizeof(Region);
    result.indexing = (d_cells.capacity() + d_faces.capacity() + d_nodes.capacity() + d_native_faces.capacity()) * sizeof(ID);
    result.indexing += d_region_interfaces.capacity() * sizeof(RegionInterfaceDirectory)
        + d_interface_sides.capacity() * sizeof(InterfaceSide);
    result.interfaces = d_interfaces.capacity() * sizeof(Interface) + d_explicit.capacity() * sizeof(ExplicitLookup);
    for (size_t i = 0; i < d_interfaces.size(); ++i)
    {
        if (const auto* e = std::get_if<ExplicitConformingInterface>(&d_interfaces[i])) result.interfaces += e->faces.capacity() * sizeof(e->faces[0]);
        if (const auto* n=std::get_if<NonconformingInterface>(&d_interfaces[i]))
        {
            result.interfaces+=n->faces.capacity()*sizeof(CoarseFineFaceMapping);
            for(const auto& group:n->faces) result.interfaces+=group.fine_faces.capacity()*sizeof(ID);
        }
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
std::string MultiRegionMesh::configuration_signature() const
{
    std::ostringstream out;
    out << std::hexfloat << d_tolerance.absolute_length << ' ' << d_tolerance.relative << ' ';
    const auto values = [&](const auto& range) { out << range.size() << ' '; for (const auto v : range) out << v << ' '; };
    for (size_t r = 0; r < d_regions.size(); ++r)
    {
        const auto& l = region_layout(r);
        out << std::quoted(region_name(r)) << ' ' << static_cast<int>(l.family) << ' ' << l.cells << ' ' << l.faces << ' ' << l.nodes << ' ';
        geometry(r, [&](const auto& g)
        {
            const auto append = [&](const auto& native)
            {
                if constexpr (requires { native.cell_edges(); })
                    for (size_t a = 0; a < 3; ++a) values(native.cell_edges()[a]);
                else if constexpr (requires { native.xy_nodes(); native.z_edges(); })
                {
                    for (const auto& p : native.xy_nodes()) out << p.x << ' ' << p.y << ' ';
                    for (const auto& row : native.xy_cell_nodes()) values(row);
                    values(native.z_edges());
                }
            };
            if constexpr (requires { g.native(); }) append(g.native());
            else append(g);
        });
    }
    for (const auto& b : d_boundaries) out << b.region << ' ' << b.native_id << ' ' << b.id << ' ' << std::quoted(b.name) << ' ';
    const auto patch = [&](const StructuredPatch& p)
    {
        out << p.region << ' ' << p.boundary << ' '; values(p.begin); values(p.extent); values(p.axes); values(p.reversed);
    };
    for (const auto& interface : d_interfaces)
    {
        out << interface.index() << ' ';
        if (const auto* s = std::get_if<StructuredPatchInterface>(&interface))
        {
            patch(s->first); patch(s->second); values(s->permutation); values(s->reversed);
            out << s->periodic_translation.has_value() << ' ';
            if(s->periodic_translation) out << s->periodic_translation->x << ' ' << s->periodic_translation->y << ' ' << s->periodic_translation->z << ' ';
        }
        else if(const auto* n=std::get_if<NonconformingInterface>(&interface))
        {
            out << n->coarse_region << ' ' << n->fine_region << ' ' << n->coarse_boundary << ' ' << n->fine_boundary << ' ' << n->faces.size() << ' ';
            for(const auto& group:n->faces) { out << group.coarse_face << ' '; values(group.fine_faces); }
        }
        else
        {
            const auto& e = std::get<ExplicitConformingInterface>(interface);
            out << e.first_region << ' ' << e.second_region << ' ' << e.first_boundary << ' ' << e.second_boundary << ' ' << e.faces.size() << ' ';
            for (const auto& [a,b] : e.faces) out << a << ' ' << b << ' ';
        }
    }
    return out.str();
}

VTUWriter::TopologyHandle MultiRegionMesh::vtu_topology() const
{
    return vtu_topology({this, 0, num_cells(), [](const void*, size_t, size_t c) -> ID { return c; }});
}

VTUWriter::TopologyHandle MultiRegionMesh::vtu_topology(const EntityRange<ID>& owned_cells) const
{
    const auto execution = acquire_execution_view();
    VTUWriter::VectorData points; VTUWriter::Int64Data connectivity, offsets; VTUWriter::UInt8Data types;
    std::unordered_map<ID, global_index_t> local_nodes;
    offsets.reserve(owned_cells.size()); types.reserve(owned_cells.size());
    for (const auto c : owned_cells)
    {
        for (const auto n : cell_nodes(c))
        {
            auto [it, added] = local_nodes.emplace(n, static_cast<global_index_t>(points.size()));
            if (added) points.push_back(node_coordinates(n));
            connectivity.push_back(it->second);
        }
        offsets.push_back(static_cast<global_index_t>(connectivity.size()));
        types.push_back(MeshUtils::vtu_cell_type_code(cell_type(c)));
    }
    return VTUWriter::make_topology(std::move(points), std::move(connectivity), std::move(offsets), std::move(types));
}
} // namespace SimpleFluid::Meshes
