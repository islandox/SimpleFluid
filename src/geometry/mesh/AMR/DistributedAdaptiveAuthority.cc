/** @file DistributedAdaptiveAuthority.cc
 * @brief Complete support authority overlays and canonical edge-point routing.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <bit>
#include <limits>
#include <map>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
using namespace adaptive_detail;
namespace
{
std::pair<std::uint64_t, std::uint64_t> entity_range(std::uint64_t local, const Teuchos::Comm<int>& comm)
{
    collective(comm,
        [&]
        {
            if (local > std::numeric_limits<std::uint64_t>::max() / std::uint64_t(comm.getSize()))
                throw std::overflow_error("Canonical entity count exceeds safe prefix range.");
        });
    std::uint64_t inclusive = 0, global = 0;
    Teuchos::scan(comm, Teuchos::REDUCE_SUM, 1, &local, &inclusive);
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local, &global);
    return {inclusive - local, global};
}
struct Occurrence
{
    AdaptiveCellKey cell;
    int rank, orientation;
    AdaptiveBytes domain, image;
    std::uint64_t native_face;
    bool exterior, native_whole = false;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> whole_geometry;
    std::string boundary;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> face;
};
AdaptiveBytes encode_tile(const AdaptiveTile& tile)
{
    Writer w;
    w.word(tile.face_gid);
    w.bytes(tile.surface);
    w.bytes(tile.domain);
    w.bytes(tile.identity);
    w.bytes(tile.geometry->encode_incidence());
    w.bytes(tile.second_image_identity);
    w.bytes(encode_adaptive_key(tile.first));
    w.word(tile.second.has_value());
    if (tile.second)
        w.bytes(encode_adaptive_key(*tile.second));
    w.word(tile.first_owner);
    w.word(tile.second_owner + 1);
    w.word(tile.first_native_face);
    w.word(tile.second_native_face);
    w.word(tile.exterior);
    w.bytes(std::span(tile.boundary));
    for (const auto* incidence : {&tile.node_gids, &tile.second_node_gids})
    {
        w.word(incidence->size());
        for (auto node : *incidence)
            w.word(node);
    }
    return std::move(w.data);
}
AdaptiveTile decode_tile(std::span<const char> bytes)
{
    Reader r{bytes};
    AdaptiveTile tile;
    tile.face_gid = r.word();
    tile.surface = r.bytes();
    tile.domain = r.bytes();
    tile.identity = r.bytes();
    tile.geometry = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes());
    tile.second_image_identity = r.bytes();
    tile.second_image = PLIC::displacement_display<ArithmeticMode::Ordinary>(tile.second_image_identity);
    tile.first = decode_adaptive_key(r.bytes());
    auto second = r.word();
    if (second > 1)
        throw std::invalid_argument("Invalid authority incidence.");
    if (second)
        tile.second = decode_adaptive_key(r.bytes());
    tile.first_owner = r.word();
    tile.second_owner = int(r.word()) - 1;
    tile.first_native_face = r.word();
    tile.second_native_face = r.word();
    tile.exterior = r.word();
    auto name = r.bytes();
    tile.boundary.assign(name.begin(), name.end());
    for (auto* incidence : {&tile.node_gids, &tile.second_node_gids})
    {
        auto count = r.word();
        if (count > (r.data.size() - r.offset) / 8)
            throw std::invalid_argument("Truncated adaptive face incidence.");
        for (std::uint64_t i = 0; i < count; ++i)
            incidence->push_back(r.word());
    }
    r.done();
    return tile;
}
} // namespace
AdaptiveBytes encode_adaptive_tile(const AdaptiveTile& tile)
{
    return encode_tile(tile);
}
AdaptiveTile decode_adaptive_tile(std::span<const char> bytes)
{
    return decode_tile(bytes);
}
std::vector<AdaptiveTile> decide_adaptive_tiles(std::span<const AdaptiveLeaf> owned,
    Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t budget, std::size_t tile_budget,
    AdaptiveAuthorityStats* statistics, std::vector<AdaptiveNode>* node_catalog)
{
    if (statistics)
        *statistics = {};
    if (comm.is_null())
        throw std::invalid_argument("Surface authority needs communicator.");
    std::vector<OwnerPacket> requests;
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            for (const auto& leaf : owned)
                for (const auto& surface : leaf.surfaces)
                {
                    if (!surface.geometry || surface.key.empty() ||
                        (surface.orientation != 1 && surface.orientation != -1))
                        throw std::invalid_argument("Incomplete support authority input.");
                    if (!surface.authority_image.empty() &&
                        (leaf.provider != AdaptiveProviderKind::Cartesian || surface.exterior ||
                            PLIC::displacement_difference<ArithmeticMode::Ordinary>(surface.authority_image, {}) !=
                                surface.authority_image))
                        throw std::invalid_argument(
                            "Periodic authority images require canonical Cartesian interior provenance.");
                    auto face = surface.geometry->encode_incidence();
                    auto key = encode_adaptive_key(leaf.key);
                    const auto whole = surface.native_whole_geometry ? surface.native_whole_geometry->encode_incidence()
                                                                     : AdaptiveBytes{};
                    if (surface.require_native_whole_tile && whole.empty())
                        throw std::invalid_argument("Missing actual native face nesting certificate.");
                    allocation.packet(88);
                    allocation.add(whole.size());
                    allocation.add(surface.authority_image.size());
                    allocation.add(leaf.domain.size());
                    allocation.add(surface.key.size());
                    allocation.add(face.size());
                    allocation.add(key.size());
                    allocation.add(surface.boundary.size());
                    Writer w;
                    w.bytes(surface.key);
                    w.bytes(leaf.domain);
                    w.bytes(key);
                    w.word(surface.native_face);
                    w.bytes(surface.authority_image);
                    w.word(surface.require_native_whole_tile);
                    w.bytes(whole);
                    w.word(surface.orientation == 1);
                    w.word(surface.exterior);
                    w.bytes(std::span(surface.boundary));
                    w.bytes(face);
                    requests.push_back({authority(surface.key, comm->getSize()), std::move(w.data)});
                }
        });
    auto incoming = exchange_owner_packets(requests, comm, budget);
    std::map<AdaptiveBytes, std::vector<Occurrence>> groups;
    std::vector<OwnerPacket> replies;
    std::uint64_t authority_faces = 0;
    std::map<AdaptiveBytes, std::uint64_t> face_identity_order;
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            for (const auto& packet : incoming.packets)
            {
                allocation.add(packet.bytes.size());
                allocation.add(sizeof(Occurrence));
                Reader r{packet.bytes};
                auto support = r.bytes();
                Occurrence entry;
                entry.domain = r.bytes();
                entry.cell = decode_adaptive_key(r.bytes());
                entry.native_face = r.word();
                entry.image = r.bytes();
                auto nesting = r.word();
                if (nesting > 1)
                    throw std::invalid_argument("Invalid native face nesting policy.");
                entry.native_whole = nesting;
                auto whole = r.bytes();
                if (!whole.empty())
                    entry.whole_geometry =
                        PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(whole)->translated_packet(entry.image);
                if (entry.native_whole && !entry.whole_geometry)
                    throw std::invalid_argument("Missing native whole-face authority proof.");
                entry.rank = packet.peer;
                auto orientation = r.word(), exterior = r.word();
                if (orientation > 1 || exterior > 1 || authority(support, comm->getSize()) != comm->getRank())
                    throw std::invalid_argument("Invalid authority support routing.");
                entry.orientation = orientation ? 1 : -1;
                entry.exterior = exterior;
                auto name = r.bytes();
                entry.boundary.assign(name.begin(), name.end());
                entry.face =
                    PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes())->translated_packet(entry.image);
                if (entry.whole_geometry && !entry.whole_geometry->contains(*entry.face))
                    throw std::invalid_argument("Rewritten join lies outside its actual native face.");
                r.done();
                groups[std::move(support)].push_back(std::move(entry));
            }
            std::size_t tiles = 0;
            for (const auto& [support, entries] : groups)
            {
                if (entries.empty())
                    continue;
                const bool exterior = entries[0].exterior;
                const auto boundary = entries[0].boundary;
                allocation.add(entries.size() * sizeof(std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>>));
                // Charge the maximum retained overlap table before quadratic pair work.
                if (entries.size() &&
                    entries.size() > budget / entries.size() / sizeof(PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>))
                    throw std::invalid_argument("Authority overlap budget exhausted.");
                allocation.add(entries.size() * entries.size() * sizeof(PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>));
                std::vector<std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>>> coverage(entries.size());
                std::set<AdaptiveCellKey> unique;
                for (std::size_t i = 0; i < entries.size(); ++i)
                {
                    const auto& a = entries[i];
                    if (a.domain.empty() || a.domain != entries[0].domain || a.exterior != exterior ||
                        a.boundary != boundary || !unique.insert(a.cell).second || (!exterior && !boundary.empty()))
                        throw std::invalid_argument("Inconsistent support membership.");
                    for (std::size_t j = 0; j < i; ++j)
                        if (a.orientation == entries[j].orientation && a.face->intersection(*entries[j].face))
                            throw std::invalid_argument("Same-side support overlap.");
                }
                auto emit =
                    [&](std::size_t i, std::optional<std::size_t> j, PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> face)
                {
                    if (tiles >= tile_budget)
                        throw std::invalid_argument("Authority tile budget exhausted.");
                    ++tiles;
                    AdaptiveTile tile;
                    tile.face_gid = authority_faces;
                    tile.surface = support;
                    tile.domain = entries[i].domain;
                    tile.geometry = std::move(face);
                    auto first_shift = entries[i].image, second_shift = j ? entries[*j].image : AdaptiveBytes{};
                    tile.first = entries[i].cell;
                    tile.first_owner = entries[i].rank;
                    tile.first_native_face = entries[i].native_face;
                    tile.exterior = exterior;
                    tile.boundary = boundary;
                    if (j)
                    {
                        tile.second = entries[*j].cell;
                        tile.second_owner = entries[*j].rank;
                        tile.second_native_face = entries[*j].native_face;
                        if (*tile.second < tile.first)
                        {
                            std::swap(tile.first, *tile.second);
                            std::swap(tile.first_owner, tile.second_owner);
                            std::swap(tile.first_native_face, tile.second_native_face);
                            std::swap(first_shift, second_shift);
                            tile.geometry = tile.geometry->reversed();
                        }
                    }
                    Writer identity;
                    identity.bytes(support);
                    auto forward = tile.geometry->encode(), reverse = tile.geometry->reversed()->encode();
                    identity.bytes(std::min(forward, reverse));
                    tile.identity = std::move(identity.data);
                    allocation.add(
                        tile.identity.size() + sizeof(AdaptiveBytes) + sizeof(std::uint64_t) + 3 * sizeof(void*));
                    if (!face_identity_order.emplace(tile.identity, authority_faces++).second)
                        throw std::invalid_argument("Duplicate canonical authority face identity.");
                    if (j)
                    {
                        tile.second_image_identity =
                            PLIC::displacement_difference<ArithmeticMode::Ordinary>(second_shift, first_shift);
                        tile.second_image =
                            PLIC::displacement_display<ArithmeticMode::Ordinary>(tile.second_image_identity);
                    }
                    tile.geometry = tile.geometry->translated_packet(
                        PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, first_shift));
                    auto bytes = encode_tile(tile);
                    allocation.packet(bytes.size());
                    replies.push_back({tile.first_owner, bytes});
                    if (j && tile.second_owner != tile.first_owner)
                    {
                        allocation.packet(bytes.size());
                        replies.push_back({tile.second_owner, std::move(bytes)});
                    }
                };
                if (exterior)
                {
                    for (std::size_t i = 0; i < entries.size(); ++i)
                    {
                        if (entries[i].orientation != entries[0].orientation)
                            throw std::invalid_argument("Exterior support has opposite side.");
                        emit(i, {}, entries[i].face);
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < entries.size(); ++i)
                        if (entries[i].orientation > 0)
                            for (std::size_t j = 0; j < entries.size(); ++j)
                                if (entries[j].orientation < 0)
                                {
                                    auto face = entries[i].face->intersection(*entries[j].face);
                                    if (!face)
                                        continue;
                                    if (entries[i].native_whole && entries[j].native_whole &&
                                        !face->equivalent(*entries[i].whole_geometry) &&
                                        !face->equivalent(*entries[j].whole_geometry))
                                        throw std::invalid_argument("Crossed procedural traces need compatible native "
                                                                    "closure; mortar tiles are not admitted.");
                                    if (!entries[i].face->oppositely_oriented(*entries[j].face))
                                        throw std::invalid_argument("Interior support normals disagree.");
                                    coverage[i].push_back(face);
                                    coverage[j].push_back(face);
                                    emit(i, j, std::move(face));
                                }
                    for (std::size_t i = 0; i < entries.size(); ++i)
                        entries[i].face->certify_partition(coverage[i]);
                }
            }
        });
    std::vector<std::uint64_t> sorted_face_ordinals;
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            allocation.add(authority_faces * sizeof(std::uint64_t));
            sorted_face_ordinals.resize(authority_faces);
            std::uint64_t ordinal = 0;
            for (const auto& [identity, emitted] : face_identity_order)
                sorted_face_ordinals[emitted] = ordinal++;
        });
    const auto [face_offset, global_faces] = entity_range(authority_faces, *comm);
    if (statistics)
        statistics->global_faces = global_faces;
    // The first wire word is the authority-local face ordinal. Patch its
    // numeric address after the collective prefix, without re-encoding geometry.
    collective(*comm,
        [&]
        {
            for (auto& packet : replies)
            {
                Reader r{packet.bytes};
                const auto emitted = r.word();
                if (emitted >= sorted_face_ordinals.size())
                    throw std::invalid_argument("Invalid authority face ordinal.");
                const auto gid = sorted_face_ordinals[emitted] + face_offset;
                for (unsigned i = 0; i < 8; ++i)
                    packet.bytes[i] = char(gid >> (8 * i));
            }
        });
    auto returned = exchange_owner_packets(replies, comm, budget);
    std::vector<AdaptiveTile> result;
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            for (const auto& packet : returned.packets)
            {
                allocation.add(packet.bytes.size());
                allocation.add(sizeof(AdaptiveTile));
                if (result.size() >= tile_budget)
                    throw std::invalid_argument("Resident tile budget exhausted.");
                auto tile = decode_tile(packet.bytes);
                if (tile.face_gid >= global_faces || packet.peer != authority(tile.surface, comm->getSize()) ||
                    (tile.first_owner != comm->getRank() && tile.second_owner != comm->getRank()))
                    throw std::invalid_argument("Foreign authority tile.");
                result.push_back(std::move(tile));
            }
        });
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            std::map<std::pair<AdaptiveCellKey, AdaptiveBytes>,
                std::vector<PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>>>
                coverage;
            std::set<AdaptiveBytes> identities;
            std::map<AdaptiveCellKey, const AdaptiveLeaf*> local;
            for (const auto& leaf : owned)
            {
                allocation.add(sizeof(AdaptiveCellKey) + sizeof(void*) + 3 * sizeof(void*));
                if (!local.emplace(leaf.key, &leaf).second)
                    throw std::invalid_argument("Duplicate owned authority cell.");
            }
            for (const auto& tile : result)
            {
                allocation.add(tile.identity.size() + sizeof(AdaptiveBytes));
                if (tile.domain.empty() || !identities.insert(tile.identity).second)
                    throw std::invalid_argument("Duplicate resident authority tile.");
                auto register_incident =
                    [&](const AdaptiveCellKey& cell, PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> face)
                {
                    const auto found = local.find(cell);
                    const AdaptiveLeaf* leaf = found == local.end() ? nullptr : found->second;
                    if (!leaf || leaf->domain != tile.domain)
                        throw std::invalid_argument("Authority tile has foreign incidence.");
                    allocation.add(sizeof(PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>) + sizeof(AdaptiveCellKey) +
                                   tile.surface.size());
                    coverage[{cell, tile.surface}].push_back(std::move(face));
                };
                if (tile.first_owner == comm->getRank())
                    register_incident(tile.first, tile.geometry);
                if (tile.second_owner == comm->getRank())
                {
                    if (!tile.second)
                        throw std::invalid_argument("Missing second authority incidence.");
                    register_incident(*tile.second,
                        tile.geometry->reversed()->translated_packet(
                            PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile.second_image_identity)));
                }
            }
            for (const auto& leaf : owned)
                for (const auto& surface : leaf.surfaces)
                {
                    auto found = coverage.find({leaf.key, surface.key});
                    if (found == coverage.end())
                        throw std::invalid_argument("Missing owned authority support.");
                    for (const auto& face : found->second)
                        if (surface.geometry->oppositely_oriented(*face))
                            throw std::invalid_argument("Authority face has inward orientation.");
                    surface.geometry->certify_partition(found->second);
                    coverage.erase(found);
                }
            if (!coverage.empty())
                throw std::invalid_argument("Unrequested authority support.");
        });
    // A line authority receives endpoint events, but replies only with points
    // contained in each requested closed segment. Long collinear meshes thus
    // retain local incidence payload rather than broadcasting the whole line.
    struct ExpectedEdge
    {
        AdaptiveBytes key;
        std::size_t tile;
        bool received = false;
        AdaptiveBytes image;
    };
    int local_periodic = 0, periodic = 0;
    for (const auto& tile : result)
        local_periodic |= !tile.second_image_identity.empty();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_periodic, &periodic);
    for (unsigned propagation = 0; propagation < 64; ++propagation)
    {
        int local_changed = 0, changed = 0;
        std::vector<ExpectedEdge> expected;
        requests.clear();
        collective(*comm,
            [&]
            {
                Budget allocation{budget};
                for (std::size_t t = 0; t < result.size(); ++t)
                {
                    auto subscribe =
                        [&](const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& geometry, const AdaptiveBytes& image)
                    {
                        auto lines = geometry->edge_support_identities();
                        auto points = geometry->vertex_identities();
                        if (lines.size() != points.size())
                            throw std::invalid_argument("Invalid tile edge incidence.");
                        for (std::size_t e = 0; e < lines.size(); ++e)
                        {
                            allocation.add(sizeof(ExpectedEdge) + sizeof(OwnerPacket) + 48 + image.size());
                            allocation.add(result[t].domain.size() + lines[e].size() + points[e].size() +
                                           points[(e + 1) % points.size()].size());
                            Writer support;
                            support.bytes(result[t].domain);
                            support.bytes(lines[e]);
                            auto key = std::move(support.data);
                            Writer w;
                            w.bytes(key);
                            w.word(expected.size());
                            w.bytes(points[e]);
                            w.bytes(points[(e + 1) % points.size()]);
                            allocation.add(w.data.size() + key.size());
                            requests.push_back({authority(key, comm->getSize()), std::move(w.data)});
                            expected.push_back({std::move(key), t, false, image});
                        }
                    };
                    subscribe(result[t].geometry, {});
                    if (!result[t].second_image_identity.empty())
                        subscribe(result[t].geometry->translated_packet(
                                      PLIC::displacement_difference<ArithmeticMode::Ordinary>(
                                          {}, result[t].second_image_identity)),
                            result[t].second_image_identity);
                }
            });
        if (statistics)
            for (const auto& packet : requests)
                statistics->edge_request_bytes += packet.bytes.size();
        incoming = exchange_owner_packets(requests, comm, budget);
        replies.clear();
        collective(*comm,
            [&]
            {
                struct Subscriber
                {
                    int rank;
                    std::uint64_t ordinal;
                    AdaptiveBytes first, last;
                };
                struct Line
                {
                    std::vector<AdaptiveBytes> points;
                    std::vector<Subscriber> subscribers;
                };
                std::map<AdaptiveBytes, Line> lines;
                Budget allocation{budget};
                std::set<std::pair<int, std::uint64_t>> unique;
                for (const auto& packet : incoming.packets)
                {
                    allocation.add(packet.bytes.size() + sizeof(Line) + sizeof(Subscriber) + 2 * sizeof(AdaptiveBytes));
                    Reader r{packet.bytes};
                    auto key = r.bytes();
                    auto ordinal = r.word();
                    if (authority(key, comm->getSize()) != comm->getRank() ||
                        !unique.emplace(packet.peer, ordinal).second)
                        throw std::invalid_argument("Invalid edge authority request.");
                    auto first = r.bytes(), last = r.bytes();
                    r.done();
                    allocation.add(key.size() + 2 * (first.size() + last.size()));
                    auto& line = lines[key];
                    line.points.push_back(first);
                    line.points.push_back(last);
                    line.subscribers.push_back({packet.peer, ordinal, std::move(first), std::move(last)});
                }
                for (const auto& [key, line] : lines)
                {
                    for (const auto& point : line.points)
                        allocation.add(4 * point.size() + 256);
                    PLIC::EdgePointIndex<ArithmeticMode::Ordinary> index(line.points);
                    for (const auto& subscriber : line.subscribers)
                    {
                        auto points = index.closed_segment(subscriber.first, subscriber.last);
                        std::size_t size = 24 + key.size();
                        for (const auto& point : points)
                        {
                            allocation.add(point.size() + sizeof(AdaptiveBytes));
                            size += 8 + point.size();
                        }
                        allocation.packet(size);
                        Writer w;
                        w.bytes(key);
                        w.word(subscriber.ordinal);
                        w.word(points.size());
                        for (const auto& point : points)
                            w.bytes(point);
                        replies.push_back({subscriber.rank, std::move(w.data)});
                    }
                }
            });
        returned = exchange_owner_packets(replies, comm, budget);
        collective(*comm,
            [&]
            {
                Budget allocation{budget};
                allocation.add(result.size() * sizeof(std::vector<AdaptiveBytes>));
                std::vector<std::vector<AdaptiveBytes>> points(result.size());
                for (const auto& packet : returned.packets)
                {
                    allocation.add(packet.bytes.size());
                    Reader r{packet.bytes};
                    auto key = r.bytes();
                    auto ordinal = r.word(), n = r.word();
                    if (ordinal >= expected.size() || n > packet.bytes.size() / 8)
                        throw std::invalid_argument("Invalid edge authority response.");
                    if (statistics)
                    {
                        statistics->edge_reply_bytes += packet.bytes.size();
                        statistics->edge_reply_points += n;
                    }
                    auto& edge = expected[ordinal];
                    if (edge.received || key != edge.key || packet.peer != authority(key, comm->getSize()))
                        throw std::invalid_argument("Foreign or duplicate edge authority response.");
                    edge.received = true;
                    for (std::uint64_t i = 0; i < n; ++i)
                    {
                        auto point = r.bytes();
                        allocation.add(point.size() + sizeof(AdaptiveBytes));
                        points[edge.tile].push_back(
                            edge.image.empty() ? std::move(point)
                                               : PLIC::point_translated<ArithmeticMode::Ordinary>(point, edge.image));
                    }
                    r.done();
                }
                for (const auto& edge : expected)
                    if (!edge.received)
                        throw std::invalid_argument("Missing edge authority response.");
                for (std::size_t i = 0; i < result.size(); ++i)
                {
                    const auto before = result[i].geometry->vertex_identities().size();
                    result[i].geometry = result[i].geometry->with_edge_vertices(points[i]);
                    local_changed |= result[i].geometry->vertex_identities().size() != before;
                }
            });
        if (!periodic)
            break;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_changed, &changed);
        if (!changed)
            break;
        collective(*comm,
            [&]
            {
                if (propagation == 63)
                    throw std::invalid_argument("Periodic edge-incidence propagation exhausted its bounded rounds.");
            });
    }
    // Numeric node addresses are assigned by full (domain, exact point) keys.
    // Face incidence ordering is retained; display coordinates never weld nodes.
    requests.clear();
    replies.clear();
    struct NeededNode
    {
        AdaptiveBytes key, point;
    };
    std::vector<NeededNode> needed;
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            std::map<AdaptiveBytes, std::uint64_t> indices;
            for (auto& tile : result)
            {
                auto assign =
                    [&](const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& geometry, std::vector<std::uint64_t>& gids)
                {
                    auto points = geometry->vertex_identities();
                    allocation.add(points.size() * sizeof(std::uint64_t));
                    gids.clear();
                    for (const auto& point : points)
                    {
                        Writer key_writer;
                        key_writer.bytes(tile.domain);
                        key_writer.bytes(point);
                        auto key = std::move(key_writer.data);
                        auto found = indices.find(key);
                        if (found == indices.end())
                        {
                            allocation.add(3 * key.size() + point.size() + sizeof(NeededNode) + sizeof(AdaptiveBytes) +
                                           3 * sizeof(void*));
                            const auto ordinal = needed.size();
                            allocation.packet(24 + key.size() + point.size());
                            Writer request;
                            request.word(ordinal);
                            request.bytes(key);
                            request.bytes(point);
                            requests.push_back({authority(key, comm->getSize()), std::move(request.data)});
                            found = indices.emplace(key, ordinal).first;
                            needed.push_back({std::move(key), point});
                        }
                        gids.push_back(found->second);
                    }
                };
                assign(tile.geometry, tile.node_gids);
                if (tile.second)
                    assign(tile.geometry->translated_packet(
                               PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile.second_image_identity)),
                        tile.second_node_gids);
            }
        });
    incoming = exchange_owner_packets(requests, comm, budget);
    struct NodeAuthority
    {
        PLIC::GeometryPoint coordinate;
        std::vector<std::pair<int, std::uint64_t>> subscribers;
    };
    std::map<AdaptiveBytes, NodeAuthority> nodes;
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            std::set<std::pair<int, std::uint64_t>> unique;
            for (const auto& packet : incoming.packets)
            {
                allocation.add(2 * packet.bytes.size() + sizeof(NodeAuthority) + 6 * sizeof(void*));
                Reader r{packet.bytes};
                auto ordinal = r.word();
                auto key = r.bytes(), point = r.bytes();
                r.done();
                Reader identity{key};
                auto domain = identity.bytes(), certified_point = identity.bytes();
                identity.done();
                if (domain.empty() || certified_point != point || authority(key, comm->getSize()) != comm->getRank() ||
                    !unique.emplace(packet.peer, ordinal).second)
                    throw std::invalid_argument("Invalid canonical node request.");
                auto found = nodes.find(key);
                if (found == nodes.end())
                    found = nodes
                                .emplace(std::move(key),
                                    NodeAuthority{PLIC::point_display<ArithmeticMode::Ordinary>(point), {}})
                                .first;
                found->second.subscribers.emplace_back(packet.peer, ordinal);
            }
        });
    const auto [node_offset, global_nodes] = entity_range(nodes.size(), *comm);
    if (statistics)
        statistics->global_nodes = global_nodes;
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            std::uint64_t ordinal = 0;
            for (const auto& [key, node] : nodes)
            {
                const auto gid = node_offset + ordinal++;
                for (const auto& [rank, request] : node.subscribers)
                {
                    allocation.packet(48 + key.size());
                    Writer w;
                    w.word(request);
                    w.bytes(key);
                    w.word(gid);
                    for (double coordinate : node.coordinate)
                        w.word(std::bit_cast<std::uint64_t>(coordinate));
                    replies.push_back({rank, std::move(w.data)});
                }
            }
        });
    returned = exchange_owner_packets(replies, comm, budget);
    collective(*comm,
        [&]
        {
            Budget allocation{budget};
            allocation.add(needed.size() * (sizeof(AdaptiveNode) + 1));
            std::vector<AdaptiveNode> catalog(needed.size());
            std::vector<bool> seen(needed.size());
            for (const auto& packet : returned.packets)
            {
                allocation.add(packet.bytes.size());
                Reader r{packet.bytes};
                auto ordinal = r.word();
                auto key = r.bytes();
                auto gid = r.word();
                PLIC::GeometryPoint coordinate;
                for (auto& value : coordinate)
                    value = std::bit_cast<double>(r.word());
                r.done();
                if (ordinal >= needed.size() || seen[ordinal] || key != needed[ordinal].key ||
                    packet.peer != authority(key, comm->getSize()) || gid >= global_nodes ||
                    coordinate != PLIC::point_display<ArithmeticMode::Ordinary>(needed[ordinal].point))
                    throw std::invalid_argument("Foreign, duplicate or malformed canonical node reply.");
                seen[ordinal] = true;
                catalog[ordinal] = {gid, std::move(key), coordinate, needed[ordinal].point};
            }
            for (bool value : seen)
                if (!value)
                    throw std::invalid_argument("Missing canonical node reply.");
            for (auto& tile : result)
                for (auto* incidence : {&tile.node_gids, &tile.second_node_gids})
                    for (auto& node : *incidence)
                    {
                        if (node >= catalog.size())
                            throw std::invalid_argument("Invalid canonical face incidence node.");
                        node = catalog[node].gid;
                    }
            if (node_catalog)
                *node_catalog = std::move(catalog);
        });
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
