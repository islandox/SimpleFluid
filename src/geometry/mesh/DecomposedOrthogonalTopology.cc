/** @file DecomposedOrthogonalTopology.cc
 * @brief Arithmetic local topology over disjoint boxes and exact L1 halos.
 */
#include "geometry/mesh/DecomposedOrthogonalTopology.hh"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::Meshes
{
using Topology = DecomposedOrthogonalTopology;
auto Topology::subtract(Box b, const Box& r) -> Boxes
{
    Box intersection;
    for (unsigned a = 0; a < 3; ++a)
    {
        intersection.begin[a] = std::max(b.begin[a], r.begin[a]);
        intersection.end[a] = std::min(b.end[a], r.end[a]);
        if (intersection.begin[a] >= intersection.end[a])
            return {b};
    }
    Boxes result;
    for (unsigned a = 0; a < 3; ++a)
    {
        if (b.begin[a] < intersection.begin[a])
        {
            auto part = b;
            part.end[a] = intersection.begin[a];
            result.push_back(part);
            b.begin[a] = intersection.begin[a];
        }
        if (b.end[a] > intersection.end[a])
        {
            auto part = b;
            part.begin[a] = intersection.end[a];
            result.push_back(part);
            b.end[a] = intersection.end[a];
        }
    }
    return result;
}
void Topology::insert(Boxes& boxes, Box b)
{
    if (!b.count())
        return;
    Boxes pending{b};
    for (const auto& existing : boxes)
    {
        Boxes next;
        for (const auto& part : pending)
        {
            auto remainder = subtract(part, existing);
            next.insert(next.end(), remainder.begin(), remainder.end());
        }
        pending = std::move(next);
        if (pending.empty())
            return;
    }
    boxes.insert(boxes.end(), pending.begin(), pending.end());
    // Coalesce neighboring products, keeping metadata independent of core volume.
    bool merged = true;
    while (merged)
    {
        merged = false;
        for (size_t i = 0; i < boxes.size() && !merged; ++i)
            for (size_t j = i + 1; j < boxes.size() && !merged; ++j)
                for (unsigned a = 0; a < 3; ++a)
                {
                    bool same = true;
                    for (unsigned t = 0; t < 3; ++t)
                        if (t != a && (boxes[i].begin[t] != boxes[j].begin[t] || boxes[i].end[t] != boxes[j].end[t]))
                            same = false;
                    if (same && (boxes[i].end[a] == boxes[j].begin[a] || boxes[j].end[a] == boxes[i].begin[a]))
                    {
                        boxes[i].begin[a] = std::min(boxes[i].begin[a], boxes[j].begin[a]);
                        boxes[i].end[a] = std::max(boxes[i].end[a], boxes[j].end[a]);
                        boxes.erase(boxes.begin() + j);
                        merged = true;
                        break;
                    }
                }
    }
}
size_t Topology::count(const Boxes& boxes) noexcept
{
    size_t result = 0;
    for (const auto& b : boxes)
        result += b.count();
    return result;
}
void Topology::finalize(Boxes& boxes) noexcept
{
    size_t offset = 0;
    for (auto& box : boxes)
    {
        box.offset = offset;
        offset += box.count();
    }
}
size_t Topology::flatten(Coordinates p, Coordinates n) noexcept
{
    return p[0] + n[0] * (p[1] + n[1] * p[2]);
}
auto Topology::expand(size_t id, Coordinates n) noexcept -> Coordinates
{
    return {id % n[0], (id / n[0]) % n[1], id / (n[0] * n[1])};
}
size_t Topology::lookup(const Boxes& boxes, Coordinates p) noexcept
{
    for (const auto& b : boxes)
        if (p[0] >= b.begin[0] && p[0] < b.end[0] && p[1] >= b.begin[1] && p[1] < b.end[1] && p[2] >= b.begin[2] &&
            p[2] < b.end[2])
            return b.offset + flatten({p[0] - b.begin[0], p[1] - b.begin[1], p[2] - b.begin[2]},
                                  {b.end[0] - b.begin[0], b.end[1] - b.begin[1], b.end[2] - b.begin[2]});
    return invalid_local_id();
}
auto Topology::select(const Boxes& boxes, size_t local) -> Coordinates
{
    const auto next =
        std::upper_bound(boxes.begin(), boxes.end(), local, [](size_t id, const Box& box) { return id < box.offset; });
    if (next == boxes.begin())
        throw std::out_of_range("Decomposed orthogonal local ID is absent.");
    const auto& b = *(next - 1);
    if (local - b.offset >= b.count())
        throw std::out_of_range("Decomposed orthogonal local ID is absent.");
    auto p = expand(local - b.offset, {b.end[0] - b.begin[0], b.end[1] - b.begin[1], b.end[2] - b.begin[2]});
    for (unsigned a = 0; a < 3; ++a)
        p[a] += b.begin[a];
    return p;
}
auto Topology::shape(unsigned kind, unsigned orientation) const noexcept -> Coordinates
{
    const auto& i = d_source->indexer();
    Coordinates result{};
    for (unsigned a = 0; a < 3; ++a)
        result[a] = kind == 0   ? i.num_cells_per_dim[a]
                    : kind == 2 ? i.num_nodes_per_dim[a]
                                : i.num_faces_per_dim_per_orientation[orientation][a];
    return result;
}
void Topology::add_wrapped(Boxes& boxes, Box box, Coordinates n) const
{
    Boxes pending{box};
    for (unsigned a = 0; a < 3; ++a)
    {
        Boxes next;
        for (auto b : pending)
        {
            if (b.begin[a] >= n[a])
            {
                b.begin[a] -= n[a];
                b.end[a] -= n[a];
            }
            if (b.end[a] <= n[a])
                next.push_back(b);
            else
            {
                auto wrap = b;
                wrap.begin[a] = 0;
                wrap.end[a] = b.end[a] - n[a];
                b.end[a] = n[a];
                next.push_back(b);
                next.push_back(wrap);
            }
        }
        pending = std::move(next);
    }
    for (const auto& b : pending)
        insert(boxes, b);
}
Topology::DecomposedOrthogonalTopology(
    std::shared_ptr<const OrthoMeshTopo> source, Coordinates begin, Coordinates end, size_t depth)
    : d_source(std::move(source)), d_owned_begin(begin), d_owned_end(end)
{
    if (!d_source)
        throw std::invalid_argument("Missing orthogonal source topology.");
    const auto n = shape(0);
    bool empty = false;
    for (unsigned a = 0; a < 3; ++a)
    {
        if (begin[a] > end[a] || end[a] > n[a])
            throw std::invalid_argument("Invalid orthogonal ownership bounds.");
        empty = empty || begin[a] == end[a];
    }
    if (empty)
        return;
    const Box owned{begin, end};
    d_cells[0].push_back(owned);
    Boxes all{owned};
    // One union expansion per graph edge preserves the exact L1 halo even on
    // thin periodic axes; repeated wraps are removed by disjoint subtraction.
    for (size_t layer = 0; layer < depth; ++layer)
    {
        auto previous = all;
        const auto before = count(all);
        for (const auto& b : previous)
            for (unsigned a = 0; a < 3; ++a)
            {
                auto upper = b;
                ++upper.begin[a];
                ++upper.end[a];
                if (d_source->indexer().periodic_dimensions[a])
                    add_wrapped(all, upper, n);
                else
                {
                    upper.end[a] = std::min(upper.end[a], n[a]);
                    insert(all, upper);
                }
                auto lower = b;
                if (lower.begin[a] > 0)
                {
                    --lower.begin[a];
                    --lower.end[a];
                    insert(all, lower);
                }
                else
                {
                    if (lower.end[a] > 1)
                    {
                        --lower.end[a];
                        insert(all, lower);
                    }
                    if (d_source->indexer().periodic_dimensions[a])
                    {
                        auto wrap = b;
                        wrap.begin[a] = n[a] - 1;
                        wrap.end[a] = n[a];
                        insert(all, wrap);
                    }
                }
            }
        if (count(all) == before)
            break;
    }
    for (const auto& b : all)
        for (const auto& part : subtract(b, owned))
            insert(d_cells[1], part);
    d_owned_cells = count(d_cells[0]);
    d_local_cells = d_owned_cells + count(d_cells[1]);
    for (unsigned o = 0; o < 3; ++o)
    {
        auto owner_faces = owned;
        ++owner_faces.begin[o];
        ++owner_faces.end[o];
        add_wrapped(d_faces[0][o], owner_faces, shape(1, o));
        if (!d_source->indexer().periodic_dimensions[o] && begin[o] == 0)
        {
            auto lower = owned;
            lower.end[o] = 1;
            insert(d_faces[0][o], lower);
        }
        Boxes faces;
        for (auto b : all)
        {
            ++b.end[o];
            add_wrapped(faces, b, shape(1, o));
        }
        for (const auto& b : faces)
        {
            Boxes remaining{b};
            for (const auto& owner : d_faces[0][o])
            {
                Boxes next;
                for (const auto& part : remaining)
                {
                    auto pieces = subtract(part, owner);
                    next.insert(next.end(), pieces.begin(), pieces.end());
                }
                remaining = std::move(next);
            }
            for (const auto& part : remaining)
                insert(d_faces[1][o], part);
        }
        d_owned_faces += count(d_faces[0][o]);
        d_local_faces += count(faces);
    }
    for (auto b : all)
    {
        for (unsigned a = 0; a < 3; ++a)
            ++b.end[a];
        add_wrapped(d_nodes, b, shape(2));
    }
    d_local_nodes = count(d_nodes);
    for (auto& boxes : d_cells)
        finalize(boxes);
    finalize(d_nodes);
    for (unsigned group = 0; group < 2; ++group)
        for (unsigned o = 0; o < 3; ++o)
        {
            finalize(d_faces[group][o]);
            d_face_offsets[group][o + 1] = d_face_offsets[group][o] + count(d_faces[group][o]);
        }
    for (unsigned o = 0; o < 3; ++o)
    {
        for (unsigned a = 0; a < 3; ++a)
            d_owned_face_shapes[o][a] = end[a] - begin[a];
        if (!d_source->indexer().periodic_dimensions[o] && begin[o] == 0)
            ++d_owned_face_shapes[o][o];
    }
}
auto Topology::source_cell(size_t local) const -> CellID
{
    Coordinates p;
    if (local < d_owned_cells)
    {
        p = expand(local,
            {d_owned_end[0] - d_owned_begin[0], d_owned_end[1] - d_owned_begin[1], d_owned_end[2] - d_owned_begin[2]});
        for (unsigned a = 0; a < 3; ++a)
            p[a] += d_owned_begin[a];
    }
    else
        p = select(d_cells[1], local - d_owned_cells);
    return {static_cast<Ordinal>(p[0]), static_cast<Ordinal>(p[1]), static_cast<Ordinal>(p[2])};
}
size_t Topology::owned_face_local_id(FaceID f) const noexcept
{
    if (f.orientation >= 3 || !d_owned_cells)
        return invalid_local_id();
    const unsigned o = f.orientation;
    Coordinates p{f.i, f.j, f.k};
    const auto periodic = d_source->indexer().periodic_dimensions[o];
    p[o] = p[o] ? p[o] - 1 : periodic ? shape(0)[o] - 1 : 0;
    for (unsigned a = 0; a < 3; ++a)
        if (p[a] < d_owned_begin[a] || p[a] >= d_owned_end[a])
            return invalid_local_id();
    p = {f.i, f.j, f.k};
    for (unsigned a = 0; a < 3; ++a)
        if (a != o)
            p[a] -= d_owned_begin[a];
    const auto seam = periodic && d_owned_end[o] == shape(0)[o];
    if (seam)
        p[o] = p[o] == 0 ? 0 : p[o] - d_owned_begin[o];
    else
        p[o] -= (!periodic && d_owned_begin[o] == 0) ? 0 : d_owned_begin[o] + 1;
    return d_face_offsets[0][o] + flatten(p, d_owned_face_shapes[o]);
}
auto Topology::owned_source_face(size_t local) const -> FaceID
{
    unsigned o = 0;
    while (o < 2 && local >= d_face_offsets[0][o + 1])
        ++o;
    auto p = expand(local - d_face_offsets[0][o], d_owned_face_shapes[o]);
    for (unsigned a = 0; a < 3; ++a)
        if (a != o)
            p[a] += d_owned_begin[a];
    const auto periodic = d_source->indexer().periodic_dimensions[o];
    const auto seam = periodic && d_owned_end[o] == shape(0)[o];
    if (seam)
        p[o] = p[o] == 0 ? 0 : p[o] + d_owned_begin[o];
    else
        p[o] += (!periodic && d_owned_begin[o] == 0) ? 0 : d_owned_begin[o] + 1;
    return {
        static_cast<Ordinal>(p[0]), static_cast<Ordinal>(p[1]), static_cast<Ordinal>(p[2]), static_cast<uint8_t>(o)};
}
auto Topology::source_face(size_t local) const -> FaceID
{
    if (local < d_owned_faces)
        return owned_source_face(local);
    local -= d_owned_faces;
    for (unsigned o = 0; o < 3; ++o)
        if (local < d_face_offsets[1][o + 1])
        {
            const auto p = select(d_faces[1][o], local - d_face_offsets[1][o]);
            return {static_cast<Ordinal>(p[0]), static_cast<Ordinal>(p[1]), static_cast<Ordinal>(p[2]),
                static_cast<uint8_t>(o)};
        }
    throw std::out_of_range("Decomposed orthogonal face ID is absent.");
}
auto Topology::source_node(size_t local) const -> NodeID
{
    const auto p = select(d_nodes, local);
    return {static_cast<Ordinal>(p[0]), static_cast<Ordinal>(p[1]), static_cast<Ordinal>(p[2])};
}
size_t Topology::cell_local_id(CellID c) const noexcept
{
    const Coordinates p{c.i, c.j, c.k};
    if (p[0] >= d_owned_begin[0] && p[0] < d_owned_end[0] && p[1] >= d_owned_begin[1] && p[1] < d_owned_end[1] &&
        p[2] >= d_owned_begin[2] && p[2] < d_owned_end[2])
        return flatten({p[0] - d_owned_begin[0], p[1] - d_owned_begin[1], p[2] - d_owned_begin[2]},
            {d_owned_end[0] - d_owned_begin[0], d_owned_end[1] - d_owned_begin[1], d_owned_end[2] - d_owned_begin[2]});
    const auto id = lookup(d_cells[1], p);
    return id == invalid_local_id() ? id : id + d_owned_cells;
}
size_t Topology::face_local_id(FaceID f) const noexcept
{
    if (f.orientation >= 3)
        return invalid_local_id();
    const auto n = shape(1, f.orientation);
    if (f.i >= n[0] || f.j >= n[1] || f.k >= n[2])
        return invalid_local_id();
    const auto owned = owned_face_local_id(f);
    if (owned != invalid_local_id())
        return owned;
    const auto id = lookup(d_faces[1][f.orientation], {f.i, f.j, f.k});
    return id == invalid_local_id() ? id : d_owned_faces + d_face_offsets[1][f.orientation] + id;
}
size_t Topology::node_local_id(NodeID n) const noexcept
{
    return lookup(d_nodes, {n.i, n.j, n.k});
}
size_t Topology::owner_cell(size_t f) const
{
    return cell_local_id(d_source->owner_cell(source_face(f)));
}
size_t Topology::neighbor_cell(size_t f) const
{
    return cell_local_id(d_source->neighbor_cell(source_face(f)));
}
int Topology::boundary_id(size_t f) const
{
    return d_source->boundary_id(source_face(f));
}
EntityRange<size_t> Topology::cell_faces(size_t cell) const
{
    (void) source_cell(cell);
    return {this, cell, 6, [](const void* object, size_t c, size_t entry)
        {
            const auto& self = *static_cast<const Topology*>(object);
            const auto id = self.source_cell(c);
            Coordinates p{id.i, id.j, id.k};
            const auto o = static_cast<unsigned>(entry / 2);
            if (entry % 2)
                ++p[o];
            if (p[o] == self.shape(1, o)[o])
                p[o] = 0;
            return self.face_local_id({static_cast<Ordinal>(p[0]), static_cast<Ordinal>(p[1]),
                static_cast<Ordinal>(p[2]), static_cast<uint8_t>(o)});
        }};
}
size_t Topology::storage_bytes() const noexcept
{
    size_t bytes = sizeof(*this) + d_nodes.capacity() * sizeof(Box);
    for (const auto& boxes : d_cells)
        bytes += boxes.capacity() * sizeof(Box);
    for (const auto& group : d_faces)
        for (const auto& boxes : group)
            bytes += boxes.capacity() * sizeof(Box);
    return bytes;
}
} // namespace SimpleFluid::Meshes
