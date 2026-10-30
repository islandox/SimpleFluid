/** @file ExactFaceTransfer.cc
 * @brief Conservative native face-rate transfer at a fixed physical clock.
 */
#include "FVM/ExactFaceTransfer.hh"
#include "FVM/GeneralVoF.hh"
#include "FVM/details/DistributedAMROverlap.hh"
#include "FVM/details/ExactPhaseLabelMap.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
namespace SimpleFluid::VoF
{
namespace
{
namespace A = Meshes::AMR;
namespace W = A::adaptive_detail;
using H = A::DistributedAdaptiveHierarchy;
using Key = A::AdaptiveCellKey;
using Tile = A::AdaptiveTile;
using Face = SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Exact>;
using NativeFace = SimpleFluid::PLIC::PlanarFacePtr<SimpleFluid::ArithmeticMode::Ordinary>;
using ExactCell = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Exact>;
using ExactCellPtr = SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Exact>;
using M = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>;
using GO = DefaultTpetraTypes::global_ordinal_type;
using LO = DefaultTpetraTypes::local_ordinal_type;
using Bytes = A::AdaptiveBytes;
using SupportIdentity = PLIC::PlaneSupportIdentity<ArithmeticMode::Exact>;
struct Budget
{
    ExactFaceTransferOptions options;
    W::Budget memory;
    std::size_t tests = 0, patches = 0, scalar_limit;
    // Retain both the finite source and its explicit exact import. Ownership
    // identity prevents a recycled temporary face address from aliasing a cache entry.
    std::map<NativeFace, Face, std::owner_less<NativeFace>> imported_faces;
    explicit Budget(ExactFaceTransferOptions o) : options(o), memory{o.byte_budget}, scalar_limit(o.byte_budget / 160)
    {
        if (!o.byte_budget || !o.maximum_common_cells || !o.maximum_patches || !o.maximum_pair_tests ||
            scalar_limit < 1024)
            throw std::invalid_argument("Invalid exact face transfer resource limits.");
        memory.add(o.byte_budget / 3); // caller/transport/codec live arena
        memory.add(o.byte_budget / 4); // hierarchy service lease (16 bounded slices)
        memory.add(20 * scalar_limit); // scalar operation/pending-result lease
    }
    std::size_t service_cap() const { return options.byte_budget / 64; }
    void records(std::size_t count, std::size_t bytes)
    {
        if (bytes && count > options.byte_budget / bytes)
            throw std::invalid_argument("Face transfer metadata count exceeds budget.");
        memory.add(count * bytes);
    }
    static std::size_t key_bytes(const Key& key)
    {
        return std::visit(
            [](const auto& p) -> std::size_t
            {
                if constexpr (requires { p.path; })
                    return p.path.capacity() * sizeof(std::uint64_t);
                else if constexpr (requires { p.base.path; })
                    return (p.base.path.capacity() + p.axial.path.capacity()) * sizeof(std::uint64_t);
                else
                    return 0;
            },
            key.ancestry);
    }
    void key(const Key& value, std::size_t copies = 1) { records(copies, key_bytes(value)); }
    void blob(const Bytes& value, std::size_t copies = 1) { records(copies, value.capacity()); }
    template<class T> void grow(std::vector<T>& values)
    {
        if (values.size() != values.capacity())
            return;
        const auto old = values.capacity();
        if (old > options.byte_budget / 2 / sizeof(T))
            throw std::invalid_argument("Face transfer vector growth exceeds budget.");
        const auto next = old ? 2 * old : 1;
        records(next, sizeof(T));
        values.reserve(next);
        if (values.capacity() > next)
            records(values.capacity() - next, sizeof(T));
    }
    void operands(const M& a, const M& b) const
    {
        if (a.storage_bytes() > scalar_limit || b.storage_bytes() > scalar_limit)
            throw std::invalid_argument("Face transfer exact operands exceed operation lease.");
    }
    void tile(const Tile& value)
    {
        records(1, sizeof(Tile));
        key(value.first);
        if (value.second)
            key(*value.second);
        blob(value.domain);
        blob(value.surface);
        blob(value.identity);
        records(1, value.second_image_identity.payload_bytes());
        records(value.node_gids.capacity() + value.second_node_gids.capacity(), sizeof(std::uint64_t));
    }
    void test()
    {
        if (tests >= options.maximum_pair_tests)
            throw std::invalid_argument("Exact face transfer pair-work budget exhausted.");
        ++tests;
    }
    template<class Geometry, class Parts> void certify(const Geometry& whole, const Parts& parts)
    {
        const auto n = parts.size();
        if (n > 1)
        {
            const auto first = n % 2 ? n : n / 2;
            const auto second = n % 2 ? (n - 1) / 2 : n - 1;
            if (first && second > (options.maximum_pair_tests - tests) / first)
                throw std::invalid_argument("Exact face coverage pair-work budget exhausted.");
            tests += first * second;
        }
        whole.certify_partition(parts);
    }
    void patch(const Face& p)
    {
        if (patches >= options.maximum_patches)
            throw std::invalid_argument("Exact face transfer patch budget exhausted.");
        ++patches;
        memory.add(sizeof(Face) + p->storage_bytes());
    }
    Face face(const NativeFace& source)
    {
        if (!source)
            throw std::invalid_argument("Exact face transfer received a null finite source face.");
        if (const auto found = imported_faces.find(source); found != imported_faces.end())
            return found->second;
        records(1, sizeof(decltype(imported_faces)::value_type) + 4 * sizeof(void*));
        memory.add(source->storage_bytes());
        auto exact = SimpleFluid::PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Exact>::from_geometry(*source);
        patch(exact);
        imported_faces.emplace(source, exact);
        return exact;
    }
    M keep(M value)
    {
        if (value.storage_bytes() > scalar_limit)
            throw std::invalid_argument("Exact face transfer scalar growth exhausted.");
        // Conservative cumulative admitted arithmetic payload, including
        // intermediates that may already have died; never an RSS claim.
        memory.add(value.storage_bytes() + sizeof(M));
        return value;
    }
    M add(const M& a, const M& b) { return keep(a + b); }
    M neg(const M& a) { return keep(a.scaled(-1)); }
    M product(const M& a, const M& b) { return keep(detail::exact_phase_product(a, b, scalar_limit)); }
};
Bytes identity(const Tile& tile)
{
    W::Writer w;
    w.bytes(tile.domain);
    w.bytes(tile.surface);
    w.bytes(tile.identity);
    w.bytes(A::encode_adaptive_key(tile.first));
    w.word(bool(tile.second));
    if (tile.second)
        w.bytes(A::encode_adaptive_key(*tile.second));
    return std::move(w.data);
}
SupportIdentity support(const Face& p)
{
    return SimpleFluid::PLIC::planar_support<SimpleFluid::ArithmeticMode::Exact>(*p);
}
Face outward(const Tile& tile, const Key& key, Budget& budget)
{
    if (tile.first == key)
        return budget.face(tile.geometry);
    if (tile.second && *tile.second == key)
    {
        auto reversed = budget.face(tile.geometry)->reversed();
        budget.patch(reversed);
        return reversed;
    }
    throw std::invalid_argument("Face transfer cell is not incident on its canonical tile.");
}
std::map<GO, int> owners(const DefaultTpetraTypes::map_type& map, const std::set<GO>& query)
{
    const auto comm = map.getComm();
    Teuchos::Array<GO> ids;
    Teuchos::Array<int> ranks;
    W::collective(*comm,
        [&]
        {
            ids.assign(query.begin(), query.end());
            ranks.resize(ids.size(), -1);
        });
    (void) map.getRemoteIndexList(ids(), ranks());
    std::map<GO, int> result;
    W::collective(*comm,
        [&]
        {
            for (std::size_t i = 0; i < std::size_t(ids.size()); ++i)
            {
                if (ranks[i] < 0)
                    throw std::invalid_argument("Missing actual face transfer native owner.");
                result.emplace(ids[i], ranks[i]);
            }
        });
    return result;
}
struct Communication
{
    Teuchos::RCP<const Teuchos::Comm<int>> comm;
    std::size_t codec, exchange;
    explicit Communication(const H& h, std::size_t limit)
        : comm(h.communicator()), codec(limit / 16), exchange(limit / 4)
    {
    }
    OwnerExchange run(const std::vector<OwnerPacket>& packets) const
    {
        W::collective(*comm,
            [&]
            {
                W::Budget b{codec};
                b.add(packets.capacity() * sizeof(OwnerPacket));
                for (const auto& p : packets)
                    b.add(p.bytes.capacity());
            });
        return exchange_owner_packets(packets, comm, exchange);
    }
};
std::map<Key, std::vector<Tile>> target_tiles(
    const GeneralVoF& target, const std::set<Key>& needed, Communication& communication, Budget& budget)
{
    const auto& h = *target.hierarchy();
    std::vector<Key> keys;
    W::collective(*communication.comm,
        [&]
        {
            budget.records(needed.size(), 6 * sizeof(Key) + sizeof(A::AdaptiveOwner) + 24 * sizeof(void*));
            for (const auto& key : needed)
                budget.key(key, 6);
            keys.assign(needed.begin(), needed.end());
        });
    const auto located = h.lookup_cells(keys, true, std::min(budget.service_cap(), h.options().byte_budget));
    std::vector<OwnerPacket> requests;
    W::collective(*communication.comm,
        [&]
        {
            W::Budget arena{communication.codec};
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                W::Writer w;
                w.bytes(A::encode_adaptive_key(keys[i]));
                w.word(located[i].revision);
                arena.packet(w.data.size());
                requests.push_back({located[i].owner, std::move(w.data)});
            }
        });
    auto incoming = communication.run(requests);
    std::vector<OwnerPacket>().swap(requests);
    std::vector<OwnerPacket> replies;
    W::collective(*communication.comm,
        [&]
        {
            W::Budget arena{communication.codec};
            std::set<std::pair<int, Key>> seen;
            for (const auto& p : incoming.packets)
            {
                W::Reader r{p.bytes};
                auto key = A::decode_adaptive_key(r.bytes());
                const auto revision = r.word();
                r.done();
                const auto& leaf = h.owned().at(key);
                if (leaf.revision != revision || !seen.emplace(p.peer, key).second)
                    throw std::invalid_argument("Invalid target native tile request.");
                W::Writer w;
                w.bytes(A::encode_adaptive_key(key));
                w.word(revision);
                std::vector<const Tile*> tiles;
                for (const auto& tile : target.tiles())
                    if (tile.first == key || (tile.second && *tile.second == key))
                        tiles.push_back(&tile);
                w.word(tiles.size());
                for (const auto* tile : tiles)
                    w.bytes(A::encode_adaptive_tile(*tile));
                arena.packet(w.data.size());
                replies.push_back({p.peer, std::move(w.data)});
            }
        });
    requests.clear();
    incoming = {};
    const auto received = communication.run(replies);
    std::map<Key, std::vector<Tile>> result;
    W::collective(*communication.comm,
        [&]
        {
            std::map<Key, A::AdaptiveOwner> expected;
            for (std::size_t i = 0; i < keys.size(); ++i)
                expected.emplace(keys[i], located[i]);
            for (const auto& p : received.packets)
            {
                W::Reader r{p.bytes};
                const auto key = A::decode_adaptive_key(r.bytes());
                const auto revision = r.word();
                const auto& owner = expected.at(key);
                if (p.peer != owner.owner || revision != owner.revision || result.contains(key))
                    throw std::invalid_argument("Foreign target native tile package.");
                auto count = r.word();
                if (count > budget.options.maximum_patches)
                    throw std::invalid_argument("Target tile package count exceeds budget.");
                std::vector<Tile> tiles;
                std::set<Bytes> seen;
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    const auto encoded = r.bytes();
                    budget.records(4, encoded.size());
                    auto tile = A::decode_adaptive_tile(encoded);
                    if (tile.first != key && (!tile.second || *tile.second != key))
                        throw std::invalid_argument("Target native package has foreign incidence.");
                    const auto tile_identity = identity(tile);
                    budget.blob(tile_identity, 2);
                    budget.records(1, sizeof(Bytes) + 4 * sizeof(void*));
                    if (!tile.second_image_identity.is_zero() || !seen.insert(tile_identity).second)
                        throw std::invalid_argument("Periodic or duplicate target native tile.");
                    budget.memory.add(sizeof(Tile) + A::encode_adaptive_tile(tile).size());
                    tiles.push_back(std::move(tile));
                }
                r.done();
                result.emplace(key, std::move(tiles));
            }
            if (result.size() != keys.size())
                throw std::invalid_argument("Incomplete target native tile packages.");
        });
    return result;
}
std::map<std::uint64_t, M> import_rates(const GeneralVoF& source, const std::map<std::uint64_t, Tile>& tiles,
    std::span<const ExactOwnedFaceRate> input, Communication& communication, Budget& budget)
{
    std::map<GO, M> owned;
    std::set<GO> query;
    W::collective(*communication.comm,
        [&]
        {
            budget.records(input.size() + tiles.size() + source.tiles().size(),
                4 * (sizeof(std::pair<const GO, M>) + 4 * sizeof(void*)));
            if (input.size() != source.mesh()->num_owned_faces())
                throw std::invalid_argument("Source face rates do not cover actual owned map.");
            for (const auto& item : input)
            {
                const auto row = source.mesh()->owned_face_map()->getLocalElement(GO(item.face));
                if (row < 0 || !owned.emplace(GO(item.face), item.rate).second)
                    throw std::invalid_argument("Foreign or duplicate accepted source rate.");
                if (source.mesh()->is_geometry_exterior_face(row) && item.rate.sign())
                    throw std::invalid_argument("Nonzero impermeable source face rate.");
                budget.keep(item.rate);
            }
            for (const auto& [gid, tile] : tiles)
                query.insert(GO(gid));
        });
    const auto located = owners(*source.mesh()->owned_face_map(), query);
    std::vector<OwnerPacket> requests;
    W::collective(*communication.comm,
        [&]
        {
            W::Budget arena{communication.codec};
            for (const auto& [gid, tile] : tiles)
            {
                W::Writer w;
                w.word(gid);
                w.bytes(identity(tile));
                arena.packet(w.data.size());
                requests.push_back({located.at(GO(gid)), std::move(w.data)});
            }
        });
    auto incoming = communication.run(requests);
    std::vector<OwnerPacket>().swap(requests);
    std::map<std::uint64_t, const Tile*> native;
    std::vector<OwnerPacket> replies;
    W::collective(*communication.comm,
        [&]
        {
            for (const auto& tile : source.tiles())
                native.emplace(tile.face_gid, &tile);
            W::Budget arena{communication.codec};
            std::set<std::pair<int, std::uint64_t>> seen;
            for (const auto& p : incoming.packets)
            {
                W::Reader r{p.bytes};
                const auto gid = r.word();
                const auto key = r.bytes();
                r.done();
                if (!owned.contains(GO(gid)) || identity(*native.at(gid)) != key || !seen.emplace(p.peer, gid).second)
                    throw std::invalid_argument("Invalid accepted source face request.");
                W::Writer w;
                w.word(gid);
                w.bytes(key);
                w.bytes(owned.at(GO(gid)).encode(budget.scalar_limit));
                arena.packet(w.data.size());
                replies.push_back({p.peer, std::move(w.data)});
            }
        });
    incoming = {};
    const auto received = communication.run(replies);
    std::map<std::uint64_t, M> result;
    W::collective(*communication.comm,
        [&]
        {
            for (const auto& p : received.packets)
            {
                W::Reader r{p.bytes};
                const auto gid = r.word();
                const auto key = r.bytes();
                const auto value = budget.keep(M::decode(r.bytes(), budget.scalar_limit));
                r.done();
                if (located.at(GO(gid)) != p.peer || identity(tiles.at(gid)) != key ||
                    !result.emplace(gid, value).second)
                    throw std::invalid_argument("Foreign or duplicate accepted source face response.");
            }
            if (result.size() != tiles.size())
                throw std::invalid_argument("Incomplete accepted source rates.");
        });
    return result;
}
struct Part
{
    Face shape;
    M rate;
    std::optional<std::size_t> edge;
    int sign = 1;
};
struct Facet
{
    std::size_t cell;
    Face shape;
    std::vector<Part> parts;
    bool inherited = false;
    std::optional<SupportIdentity> plane;
};
struct Common
{
    Key target;
    SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> geometry;
    std::vector<std::size_t> facets;
};
struct Internal
{
    std::size_t first, second;
    Face shape;
    M rate;
    Bytes identity;
};
struct Output
{
    Key source, target;
    std::uint64_t source_revision, face;
    Bytes face_identity;
    Face shape;
    M rate;
};
std::vector<Output> extend(const Key& key, const A::AdaptiveLeaf& source_leaf,
    const SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>& source_geometry,
    std::vector<Common> cells, const std::vector<const Tile*>& boundary, const std::map<std::uint64_t, M>& old_rates,
    SimpleFluid::PLIC::GeometryPoint velocity, const std::map<Key, std::vector<Tile>>& targets, Budget& budget)
{
    std::sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) { return a.target < b.target; });
    if (cells.empty())
        throw std::invalid_argument("Empty source common refinement.");
    budget.records(cells.size(), sizeof(ExactCellPtr) + 2 * sizeof(std::size_t));
    const auto exact_source = ExactCell::from_geometry(source_geometry);
    budget.memory.add(sizeof(ExactCellPtr) + exact_source->storage_bytes());
    std::vector<ExactCellPtr> cover;
    cover.reserve(cells.size());
    for (const auto& cell : cells)
    {
        auto imported = ExactCell::from_geometry(*cell.geometry);
        budget.memory.add(imported->storage_bytes());
        cover.push_back(std::move(imported));
    }
    budget.certify(*exact_source, cover);
    M old_divergence;
    for (const auto* tile : boundary)
        old_divergence = budget.add(old_divergence,
            tile->first == key ? old_rates.at(tile->face_gid) : budget.neg(old_rates.at(tile->face_gid)));
    if (old_divergence.sign())
        throw std::invalid_argument("Accepted source face rate is not exactly divergence-free.");
    std::vector<Facet> facets;
    std::map<SupportIdentity, std::vector<std::size_t>> supports;
    std::map<std::uint64_t, std::vector<Face>> old_cover;
    for (std::size_t i = 0; i < cells.size(); ++i)
        for (std::size_t f = 0; f < cells[i].geometry->num_faces(); ++f)
        {
            Facet facet{i, cover[i]->face_geometry(f), {}};
            budget.patch(facet.shape);
            facet.plane = support(facet.shape);
            budget.records(1, facet.plane->payload_bytes());
            budget.grow(facets);
            std::vector<Face> pieces;
            for (const auto* tile : boundary)
            {
                budget.test();
                const auto old = outward(*tile, key, budget);
                if (support(old) != support(facet.shape))
                    continue;
                if (const auto piece = facet.shape->intersection(*old))
                {
                    budget.patch(piece);
                    const auto sign = tile->first == key ? 1 : -1;
                    const auto rate = budget.product(old_rates.at(tile->face_gid),
                        piece->coplanar_area_ratio(*budget.face(tile->geometry), budget.scalar_limit));
                    budget.grow(facet.parts);
                    budget.grow(pieces);
                    if (!old_cover.contains(tile->face_gid))
                        budget.records(1, sizeof(decltype(old_cover)::value_type) + 4 * sizeof(void*));
                    budget.grow(old_cover[tile->face_gid]);
                    facet.parts.push_back({piece, sign == 1 ? rate : budget.neg(rate), {}, 1});
                    pieces.push_back(piece);
                    old_cover[tile->face_gid].push_back(piece);
                }
            }
            if (!pieces.empty())
            {
                budget.certify(*facet.shape, pieces);
                facet.inherited = true;
            }
            budget.grow(cells[i].facets);
            cells[i].facets.push_back(facets.size());
            if (!supports.contains(*facet.plane))
            {
                budget.records(1, sizeof(decltype(supports)::value_type) + 4 * sizeof(void*));
                budget.records(1, facet.plane->payload_bytes());
            }
            budget.grow(supports[*facet.plane]);
            supports[*facet.plane].push_back(facets.size());
            facets.push_back(std::move(facet));
        }
    for (const auto* tile : boundary)
        budget.certify(*budget.face(tile->geometry), old_cover.at(tile->face_gid));
    std::vector<Internal> internal;
    for (std::size_t f = 0; f < facets.size(); ++f)
    {
        auto& face = facets[f];
        if (face.inherited)
            continue;
        std::vector<Face> pieces;
        for (const auto g : supports.at(*face.plane))
        {
            if (g == f || facets[g].cell == face.cell)
                continue;
            budget.test();
            const auto piece = face.shape->intersection(*facets[g].shape);
            if (!piece)
                continue;
            if (!face.shape->oppositely_oriented(*facets[g].shape))
                throw std::invalid_argument("Common cells have equal-oriented interior faces.");
            budget.patch(piece);
            budget.grow(pieces);
            pieces.push_back(piece);
            if (f < g)
            {
                const auto index = internal.size();
                budget.grow(internal);
                budget.grow(face.parts);
                budget.grow(facets[g].parts);
                auto a = PLIC::canonical_face_key(*piece), b = PLIC::canonical_face_key(*piece->reversed());
                auto shape_identity = std::min(a, b);
                budget.blob(shape_identity);
                internal.push_back({face.cell, facets[g].cell, piece,
                    budget.keep(piece->area_vector_dot(velocity, budget.scalar_limit)), std::move(shape_identity)});
                face.parts.push_back({piece, {}, index, 1});
                auto reversed = piece->reversed();
                budget.patch(reversed);
                facets[g].parts.push_back({std::move(reversed), {}, index, -1});
            }
        }
        budget.certify(*face.shape, pieces);
    }
    // Check every interior facet independently, including those populated by an
    // earlier partner. No one-sided or overlapping interface is admitted.
    for (const auto& face : facets)
    {
        std::vector<Face> pieces;
        budget.records(face.parts.size(), sizeof(Face));
        pieces.reserve(face.parts.size());
        for (const auto& p : face.parts)
            pieces.push_back(p.shape);
        budget.certify(*face.shape, pieces);
    }
    auto divergences = [&]
    {
        budget.records(cells.size(), sizeof(M));
        std::vector<M> result(cells.size());
        for (const auto& face : facets)
            for (const auto& part : face.parts)
                result[face.cell] = budget.add(result[face.cell],
                    part.edge ? (part.sign == 1 ? internal[*part.edge].rate : budget.neg(internal[*part.edge].rate))
                              : part.rate);
        return result;
    };
    budget.records(cells.size(), sizeof(std::vector<std::size_t>));
    std::vector<std::vector<std::size_t>> adjacent(cells.size());
    for (std::size_t e = 0; e < internal.size(); ++e)
    {
        budget.grow(adjacent[internal[e].first]);
        budget.grow(adjacent[internal[e].second]);
        adjacent[internal[e].first].push_back(e);
        adjacent[internal[e].second].push_back(e);
    }
    for (std::size_t row = 0; row < adjacent.size(); ++row)
        std::sort(adjacent[row].begin(), adjacent[row].end(),
            [&](const auto a, const auto b)
            {
                const auto other_a = internal[a].first == row ? internal[a].second : internal[a].first;
                const auto other_b = internal[b].first == row ? internal[b].second : internal[b].first;
                if (cells[other_a].target != cells[other_b].target)
                    return cells[other_a].target < cells[other_b].target;
                return internal[a].identity < internal[b].identity;
            });
    const auto none = std::numeric_limits<std::size_t>::max();
    budget.records(2 * cells.size(), sizeof(std::size_t));
    std::vector<std::size_t> parent(cells.size(), none), parent_edge(cells.size(), none), order;
    budget.grow(order);
    order.push_back(0);
    parent[0] = 0;
    for (std::size_t i = 0; i < order.size(); ++i)
        for (const auto e : adjacent[order[i]])
        {
            const auto other = internal[e].first == order[i] ? internal[e].second : internal[e].first;
            if (parent[other] == none)
            {
                parent[other] = order[i];
                parent_edge[other] = e;
                budget.grow(order);
                order.push_back(other);
            }
        }
    if (order.size() != cells.size())
        throw std::invalid_argument("Common refinement interior is not face-connected.");
    auto defect = divergences();
    for (std::size_t i = order.size(); i-- > 1;)
    {
        const auto child = order[i], e = parent_edge[child];
        internal[e].rate =
            budget.add(internal[e].rate, internal[e].first == child ? budget.neg(defect[child]) : defect[child]);
        defect[parent[child]] = budget.add(defect[parent[child]], defect[child]);
    }
    for (const auto& d : divergences())
        if (d.sign())
            throw std::invalid_argument("New common-face extension did not close exact divergence.");
    std::vector<Output> result;
    for (const auto& face : facets)
        for (const auto& part : face.parts)
            for (const auto& tile : targets.at(cells[face.cell].target))
            {
                if (tile.first != cells[face.cell].target)
                    continue;
                const auto target_face = budget.face(tile.geometry);
                if (support(part.shape) != support(target_face))
                    continue;
                budget.test();
                const auto piece = part.shape->intersection(*target_face);
                if (!piece)
                    continue;
                if (part.shape->oppositely_oriented(*target_face))
                    throw std::invalid_argument("Target face has inconsistent outward orientation.");
                budget.patch(piece);
                const auto rate =
                    part.edge ? (part.sign == 1 ? internal[*part.edge].rate : budget.neg(internal[*part.edge].rate))
                              : part.rate;
                budget.grow(result);
                budget.key(key);
                budget.key(cells[face.cell].target);
                auto face_identity = identity(tile);
                budget.blob(face_identity);
                result.push_back(
                    {key, cells[face.cell].target, source_leaf.revision, tile.face_gid, std::move(face_identity), piece,
                        budget.product(rate, piece->coplanar_area_ratio(*part.shape, budget.scalar_limit))});
            }
    return result;
}
} // namespace
struct ExactFaceTransferResult::Data
{
    explicit Data(ExactPhaseTransferResult value) : phase(std::move(value)) {}
    ExactPhaseTransferResult phase;
    std::shared_ptr<const void> source_identity, target_identity;
    std::uint64_t source_control = 0, target_control = 0;
    A::HierarchyObservation source_observation, target_observation;
    std::vector<ExactOwnedFaceRate> rates;
};
ExactFaceTransferResult::ExactFaceTransferResult(std::shared_ptr<const Data> value) : data_(std::move(value)) {}
const std::vector<ExactOwnedFaceRate>& ExactFaceTransferResult::owned_rates() const noexcept
{
    return data_->rates;
}
void ExactFaceTransferResult::validate_observations(const GeneralVoF& source, const GeneralVoF& target) const
{
    const auto comm = data_->phase.source_hierarchy()->communicator();
    W::collective(*comm,
        [&]
        {
            if (source.hierarchy()->communicator().getRawPtr() != comm.getRawPtr() ||
                target.hierarchy()->communicator().getRawPtr() != comm.getRawPtr() || !source.accepted_unchanged() ||
                !target.accepted_unchanged())
                throw std::invalid_argument("Face transfer publication communicator differs.");
        });
    data_->phase.validate_observations();
    W::collective(*comm,
        [&]
        {
            if (source.accepted_identity() != data_->source_identity ||
                target.accepted_identity() != data_->target_identity ||
                source.accepted_control() != data_->source_control ||
                target.accepted_control() != data_->target_control ||
                source.hierarchy()->observation() != data_->source_observation ||
                target.hierarchy()->observation() != data_->target_observation || !source.accepted_unchanged() ||
                !target.accepted_unchanged())
                throw std::invalid_argument("Exact face transfer source/target publication is stale or foreign.");
        });
}
ExactFaceTransferResult ExactFaceTransfer::prepare(const GeneralVoF& source, const GeneralVoF& target,
    const ExactPhaseTransferResult& phase, std::span<const ExactOwnedFaceRate> rates,
    std::span<const ExactOwnedCellVelocity> velocity, ExactFaceTransferOptions options)
{
    const auto comm = phase.source_hierarchy()->communicator();
    W::collective(*comm,
        [&]
        {
            if (source.hierarchy()->communicator().getRawPtr() != comm.getRawPtr() ||
                target.hierarchy()->communicator().getRawPtr() != comm.getRawPtr() || !source.accepted_unchanged() ||
                !target.accepted_unchanged())
                throw std::invalid_argument("Face transfer publication communicator differs.");
        });
    phase.validate_observations();
    target.validate_transfer_target(phase);
    std::shared_ptr<ExactFaceTransferResult::Data> result;
    std::unique_ptr<Budget> budget;
    std::map<Key, SimpleFluid::PLIC::GeometryPoint> seeds;
    W::collective(*comm,
        [&]
        {
            budget = std::make_unique<Budget>(options);
            if (target.hierarchy()->communicator().getRawPtr() != comm.getRawPtr() ||
                source.accepted_identity() != phase.source_identity() ||
                source.accepted_control() != phase.source_control() ||
                source.hierarchy()->observation() != phase.source_hierarchy()->observation() ||
                target.hierarchy()->topology_generation() != phase.target_hierarchy()->topology_generation() ||
                target.hierarchy()->partition_generation() < phase.target_hierarchy()->partition_generation())
                throw std::invalid_argument("Face transfer is not pinned to the accepted phase transaction.");
            if (!phase.ownership_only() && velocity.size() != source.hierarchy()->owned().size())
                throw std::invalid_argument(
                    "Topology face transfer needs one physical velocity seed per source owner.");
            budget->records(
                velocity.size(), sizeof(std::pair<const Key, SimpleFluid::PLIC::GeometryPoint>) + 4 * sizeof(void*));
            for (const auto& item : velocity)
                budget->key(item.key);
            for (const auto& item : velocity)
            {
                if (!source.hierarchy()->owned().contains(item.key) || !seeds.emplace(item.key, item.velocity).second)
                    throw std::invalid_argument("Foreign or duplicate physical velocity seed.");
                for (const auto x : item.velocity)
                    if (!std::isfinite(x))
                        throw std::invalid_argument("Nonfinite physical velocity seed.");
            }
            budget->records(1, sizeof(ExactFaceTransferResult::Data));
            result = std::make_shared<ExactFaceTransferResult::Data>(phase);
            result->source_identity = source.accepted_identity();
            result->source_control = source.accepted_control();
            result->target_identity = target.accepted_identity();
            result->target_control = target.accepted_control();
            result->source_observation = source.hierarchy()->observation();
            result->target_observation = target.hierarchy()->observation();
        });
    Communication communication(*source.hierarchy(), options.byte_budget);
    const auto overlaps = detail::build_adaptive_overlap(
        *source.hierarchy(), *target.hierarchy(), budget->service_cap(), options.byte_budget / 16);
    std::map<Key, std::vector<Common>> groups;
    std::set<Key> needed;
    std::map<std::uint64_t, Tile> old_tiles;
    W::collective(*comm,
        [&]
        {
            budget->records(
                overlaps->source_cells().size() + overlaps->target_cells().size(), sizeof(A::AdaptiveCellBinding));
            for (const auto* catalog : {&overlaps->source_cells(), &overlaps->target_cells()})
                for (const auto& cell : *catalog)
                {
                    budget->key(cell.key);
                    budget->memory.add(cell.geometry->storage_bytes());
                }
            budget->records(overlaps->records().size(), sizeof(A::AdaptiveOverlapRecord));
            for (const auto& record : overlaps->records())
            {
                budget->key(record.source_key);
                budget->key(record.target_key);
            }
            std::size_t count = 0;
            for (const auto& record : overlaps->records())
                if (source.hierarchy()->owned().contains(record.source_key))
                {
                    if (++count > options.maximum_common_cells)
                        throw std::invalid_argument("Common refinement cell budget exhausted.");
                    budget->memory.add(sizeof(Common) + record.intersection->storage_bytes() +
                                       A::encode_adaptive_key(record.target_key).size());
                    if (!groups.contains(record.source_key))
                    {
                        budget->records(1, sizeof(decltype(groups)::value_type) + 4 * sizeof(void*));
                        budget->key(record.source_key);
                    }
                    budget->grow(groups[record.source_key]);
                    budget->key(record.target_key);
                    groups[record.source_key].push_back({record.target_key, record.intersection, {}});
                    if (!needed.contains(record.target_key))
                    {
                        budget->records(1, sizeof(Key) + 4 * sizeof(void*));
                        budget->key(record.target_key);
                    }
                    needed.insert(record.target_key);
                }
            for (const auto& tile : source.tiles())
            {
                if (!tile.second_image_identity.is_zero())
                    throw std::invalid_argument("General face transfer periodic images remain unqualified.");
                if (source.hierarchy()->owned().contains(tile.first) ||
                    (tile.second && source.hierarchy()->owned().contains(*tile.second)))
                {
                    budget->records(1, sizeof(decltype(old_tiles)::value_type) + 4 * sizeof(void*));
                    budget->tile(tile);
                    if (!old_tiles.emplace(tile.face_gid, tile).second)
                        throw std::invalid_argument("Duplicate source native tile publication.");
                    budget->memory.add(sizeof(Tile) + A::encode_adaptive_tile(tile).size());
                }
            }
        });
    const auto old_rates = import_rates(source, old_tiles, rates, communication, *budget);
    const auto native_targets = target_tiles(target, needed, communication, *budget);
    std::vector<Output> outputs;
    W::collective(*comm,
        [&]
        {
            for (const auto& [key, leaf] : source.hierarchy()->owned())
            {
                budget->records(old_tiles.size(), sizeof(const Tile*));
                std::vector<const Tile*> boundary;
                boundary.reserve(old_tiles.size());
                for (const auto& [gid, tile] : old_tiles)
                    if (tile.first == key || (tile.second && *tile.second == key))
                        boundary.push_back(&tile);
                const auto geometry = source.hierarchy()->geometry(leaf);
                auto generated = extend(key, leaf, *geometry, std::move(groups.at(key)), boundary, old_rates,
                    phase.ownership_only() ? SimpleFluid::PLIC::GeometryPoint{} : seeds.at(key), native_targets,
                    *budget);
                for (auto& item : generated)
                {
                    budget->memory.add(sizeof(Output) + A::encode_adaptive_key(item.source).size() +
                                       A::encode_adaptive_key(item.target).size());
                    budget->grow(outputs);
                    outputs.push_back(std::move(item));
                }
            }
        });
    std::set<GO> query;
    W::collective(*comm,
        [&]
        {
            budget->records(outputs.size(), 5 * (sizeof(GO) + sizeof(int) + 4 * sizeof(void*)));
            for (const auto& item : outputs)
                query.insert(GO(item.face));
        });
    const auto destinations = owners(*target.mesh()->owned_face_map(), query);
    std::vector<OwnerPacket> packets;
    W::collective(*comm,
        [&]
        {
            W::Budget arena{communication.codec};
            for (const auto& item : outputs)
            {
                W::Writer w;
                w.bytes(A::encode_adaptive_key(item.source));
                w.word(item.source_revision);
                w.bytes(A::encode_adaptive_key(item.target));
                w.word(item.face);
                w.bytes(item.face_identity);
                w.bytes(item.shape->encode());
                w.bytes(item.rate.encode(budget->scalar_limit));
                arena.packet(w.data.size());
                packets.push_back({destinations.at(GO(item.face)), std::move(w.data)});
            }
        });
    auto incoming = communication.run(packets);
    std::vector<OwnerPacket>().swap(packets);
    std::vector<Output>().swap(outputs);
    struct Received
    {
        int peer;
        Output output;
    };
    std::vector<Received> decoded;
    std::vector<Key> source_keys;
    W::collective(*comm,
        [&]
        {
            for (const auto& p : incoming.packets)
            {
                budget->records(4, p.bytes.size());
                budget->grow(decoded);
                budget->grow(source_keys);
                W::Reader r{p.bytes};
                Output item;
                item.source = A::decode_adaptive_key(r.bytes());
                item.source_revision = r.word();
                item.target = A::decode_adaptive_key(r.bytes());
                item.face = r.word();
                item.face_identity = r.bytes();
                item.shape = SimpleFluid::PLIC::PlanarFace<SimpleFluid::ArithmeticMode::Exact>::decode(r.bytes());
                item.rate = budget->keep(M::decode(r.bytes(), budget->scalar_limit));
                r.done();
                budget->patch(item.shape);
                source_keys.push_back(item.source);
                decoded.push_back({p.peer, std::move(item)});
            }
        });
    incoming = {};
    const auto source_owners = source.hierarchy()->lookup_cells(
        source_keys, true, std::min(budget->service_cap(), source.hierarchy()->options().byte_budget));
    std::map<std::uint64_t, M> target_rates;
    W::collective(*comm,
        [&]
        {
            budget->records(target.tiles().size() + decoded.size(),
                6 * (sizeof(Key) + sizeof(Face) + sizeof(M) + 4 * sizeof(void*)));
            for (const auto& item : decoded)
            {
                budget->key(item.output.source);
                budget->key(item.output.target);
                budget->blob(item.output.face_identity);
            }
            std::map<std::uint64_t, const Tile*> actual;
            for (const auto& tile : target.tiles())
                if (target.mesh()->owned_face_map()->getLocalElement(GO(tile.face_gid)) >= 0)
                    if (!actual.emplace(tile.face_gid, &tile).second)
                        throw std::invalid_argument("Duplicate actual target owned tile.");
            std::map<std::uint64_t, std::vector<Face>> cover;
            std::set<std::pair<std::uint64_t, Bytes>> seen;
            for (std::size_t i = 0; i < decoded.size(); ++i)
            {
                const auto& item = decoded[i].output;
                const auto& owner = source_owners[i];
                const auto& tile = *actual.at(item.face);
                const auto target_face = budget->face(tile.geometry);
                if (owner.owner != decoded[i].peer || owner.revision != item.source_revision ||
                    tile.first != item.target || identity(tile) != item.face_identity ||
                    item.shape->oppositely_oriented(*target_face) || !target_face->contains(*item.shape) ||
                    !seen.emplace(item.face, PLIC::canonical_face_key(*item.shape)).second)
                    throw std::invalid_argument("Foreign, overlapping or duplicate target face proof.");
                budget->grow(cover[item.face]);
                cover[item.face].push_back(item.shape);
                target_rates[item.face] = budget->add(target_rates[item.face], item.rate);
            }
            if (actual.size() != target.mesh()->num_owned_faces())
                throw std::invalid_argument("Actual target owned face publication is incomplete.");
            for (std::size_t i = 0; i < target.mesh()->num_owned_faces(); ++i)
            {
                const auto gid = std::uint64_t(target.mesh()->face_global_id(LO(i)));
                budget->certify(*budget->face(actual.at(gid)->geometry), cover.at(gid));
                if (actual.at(gid)->exterior && target_rates.at(gid).sign())
                    throw std::invalid_argument("Transferred physical wall rate is nonzero.");
                budget->grow(result->rates);
                result->rates.push_back({gid, target_rates.at(gid)});
            }
        });
    std::vector<Received>().swap(decoded);
    std::vector<Key>().swap(source_keys);
    // Re-import the actual target canonical rates to cell owners and certify
    // every target row independently. No correction or normalization here.
    std::map<std::uint64_t, Tile> target_owned_incidence;
    W::collective(*comm,
        [&]
        {
            for (const auto& tile : target.tiles())
                if (target.hierarchy()->owned().contains(tile.first) ||
                    (tile.second && target.hierarchy()->owned().contains(*tile.second)))
                {
                    budget->records(1, sizeof(decltype(target_owned_incidence)::value_type) + 4 * sizeof(void*));
                    budget->tile(tile);
                    target_owned_incidence.emplace(tile.face_gid, tile);
                }
        });
    const auto target_cell_rates = import_rates(target, target_owned_incidence, result->rates, communication, *budget);
    W::collective(*comm,
        [&]
        {
            std::map<Key, M> divergence;
            budget->records(
                target.hierarchy()->owned().size(), sizeof(decltype(divergence)::value_type) + 4 * sizeof(void*));
            for (const auto& [key, leaf] : target.hierarchy()->owned())
            {
                budget->key(key);
                divergence.emplace(key, M{});
            }
            for (const auto& [gid, tile] : target_owned_incidence)
            {
                if (divergence.contains(tile.first))
                    divergence[tile.first] = budget->add(divergence[tile.first], target_cell_rates.at(gid));
                if (tile.second && divergence.contains(*tile.second))
                    divergence[*tile.second] =
                        budget->add(divergence[*tile.second], budget->neg(target_cell_rates.at(gid)));
            }
            for (const auto& [key, value] : divergence)
                if (value.sign())
                    throw std::invalid_argument("Transferred target face rate has nonzero exact divergence.");
        });
    ExactFaceTransferResult validated(std::move(result));
    validated.validate_observations(source, target);
    return validated;
}
} // namespace SimpleFluid::VoF
