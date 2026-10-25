/** @file DistributedAdaptiveClosure.cc
 * @brief Point-authority closed-neighbor proofs without global leaf catalogs.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <map>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
namespace
{
using namespace adaptive_detail;
struct Contact
{
    AdaptiveCellKey key;
    AdaptiveBytes domain, physical_point;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry;
    std::array<std::uint64_t, 3> factors{};
    std::vector<AdaptiveTraceFace> traces;
    int owner = -1;
    std::uint64_t ordinal = 0;
    std::optional<AdaptiveSplit> required;
};
AdaptiveBytes point_key(std::span<const char> domain, std::span<const char> point)
{
    Writer w;
    w.bytes(domain);
    w.bytes(point);
    return std::move(w.data);
}
AdaptiveBytes encode_contact(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, Budget& budget)
{
    auto shape = kernel.geometry(leaf);
    auto trace = kernel.traces(leaf);
    if (trace.key != leaf.key)
        throw std::invalid_argument("Provider trace identifies another leaf.");
    budget.add(shape->storage_bytes());
    Writer w;
    w.bytes(encode_adaptive_key(leaf.key));
    w.bytes(leaf.domain);
    w.bytes(shape->encode_certificate());
    for (std::size_t a = 0; a < 3; ++a)
        w.word(leaf.provider == AdaptiveProviderKind::Static ? 3 : std::uint64_t(kernel.closure_split(leaf, a)));
    w.word(trace.faces.size());
    for (const auto& face : trace.faces)
    {
        if (face.stem.empty())
            throw std::invalid_argument("Comparable trace requires explicit provenance.");
        budget.add(face.stem.size() + sizeof(AdaptiveTraceFace));
        w.bytes(face.stem);
        w.word(face.depth);
        w.word(std::uint64_t(face.split));
        w.word(face.terminal);
        w.word(bool(face.face));
        if (face.face)
        {
            budget.add(face.face->encode_incidence().size());
            w.bytes(face.face->encode_incidence());
        }
    }
    budget.add(w.data.size());
    return std::move(w.data);
}
Contact decode_contact(std::span<const char> bytes)
{
    Reader r{bytes};
    Contact contact;
    contact.key = decode_adaptive_key(r.bytes());
    contact.domain = r.bytes();
    if (contact.domain.empty())
        throw std::invalid_argument("Closure contact requires connected-domain provenance.");
    contact.geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::decode_certificate(r.bytes());
    for (auto& factor : contact.factors)
    {
        factor = r.word();
        if (factor > 3)
            throw std::invalid_argument("Invalid closure refinement factor.");
    }
    auto n = r.word();
    if (n > bytes.size() / 32)
        throw std::invalid_argument("Invalid comparable trace count.");
    for (std::uint64_t i = 0; i < n; ++i)
    {
        AdaptiveTraceFace face;
        face.stem = r.bytes();
        auto depth = r.word(), split = r.word(), terminal = r.word(), geometry = r.word();
        if (face.stem.empty() || depth > 64 || split > 2 || terminal > 1 || geometry > 1)
            throw std::invalid_argument("Invalid comparable trace.");
        face.depth = unsigned(depth);
        face.split = AdaptiveSplit(split);
        face.terminal = terminal;
        if (geometry)
            face.face = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes());
        contact.traces.push_back(std::move(face));
    }
    r.done();
    return contact;
}
void require(Contact& contact, std::uint64_t split)
{
    if (split > 2 ||
        std::all_of(contact.factors.begin(), contact.factors.end(), [](auto factor) { return factor == 3; }))
        throw std::invalid_argument("Closed-contact closure needs an unsupported static refinement.");
    auto kind = AdaptiveSplit(split);
    if (!contact.required)
        contact.required = kind;
    else if (*contact.required != kind)
        contact.required = AdaptiveSplit::Isotropic;
}
} // namespace
std::vector<AdaptiveMark> required_adaptive_refinements(std::span<const AdaptiveLeaf> owned,
    const AdaptiveProviderKernel& kernel, Teuchos::RCP<const Teuchos::Comm<int>> comm, AdaptiveClosureOptions options)
{
    using namespace adaptive_detail;
    if (comm.is_null())
        throw std::invalid_argument("Adaptive closure needs a communicator.");
    std::array<std::uint64_t, 5> semantic{options.byte_budget, options.contact_budget, options.tile_budget,
        std::bit_cast<std::uint64_t>(options.maximum_span_ratio), options.maximum_trace_jump},
        root = semantic;
    Teuchos::broadcast(*comm, 0, 5, root.data());
    collective(*comm,
        [&]
        {
            if (root != semantic)
                throw std::invalid_argument("Closure options differ across ranks.");
        });
    std::vector<AdaptiveLeaf> transient;
    collective(*comm,
        [&]
        {
            if (!options.byte_budget || !options.contact_budget || !options.tile_budget ||
                !std::isfinite(options.maximum_span_ratio) || options.maximum_span_ratio < 1 ||
                options.maximum_trace_jump != 1)
                throw std::invalid_argument("Invalid closed-contact closure bounds.");
            Budget budget{options.byte_budget};
            for (const auto& source : owned)
            {
                budget.add(sizeof(AdaptiveLeaf));
                auto leaf = source;
                leaf.geometry = kernel.geometry(source);
                leaf.surfaces = kernel.surfaces(source);
                budget.add(leaf.geometry->storage_bytes());
                for (const auto& surface : leaf.surfaces)
                    budget.add(surface.key.size() + surface.geometry->encode_incidence().size());
                transient.push_back(std::move(leaf));
            }
        });
    auto tiles = decide_adaptive_tiles(transient, comm, options.byte_budget, options.tile_budget);
    std::vector<OwnerPacket> requests;
    struct Expected
    {
        AdaptiveBytes point_key;
        AdaptiveCellKey cell;
    };
    std::vector<Expected> expected;
    collective(*comm,
        [&]
        {
            Budget budget{options.byte_budget};
            std::map<AdaptiveCellKey, std::set<AdaptiveBytes>> points;
            for (const auto& tile : tiles)
            {
                auto record_points =
                    [&](const AdaptiveCellKey& cell, const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& face)
                {
                    for (const auto& point : face->vertex_identities())
                    {
                        budget.add(point.size() + sizeof(AdaptiveBytes) + 3 * sizeof(void*));
                        points[cell].insert(point);
                    }
                };
                if (tile.first_owner == comm->getRank())
                    record_points(tile.first, tile.geometry);
                if (tile.second_owner == comm->getRank())
                    record_points(*tile.second,
                        tile.geometry->translated_packet(
                            PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile.second_image_identity)));
            }
            // Provider queries consume retained provider records, not the
            // canonical authority copies above. A Mixed retained polyhedron
            // stores original support keys; mapping its already-rewritten
            // transient surfaces again would lose the physical selections.
            for (const auto& leaf : owned)
            {
                auto found = points.find(leaf.key);
                if (found == points.end())
                    throw std::invalid_argument("Closure has no complete owned contact incidence.");
                auto record = encode_contact(leaf, kernel, budget);
                for (const auto& point : found->second)
                {
                    auto normalized = kernel.contact_point(leaf, point);
                    auto key = point_key(leaf.domain, normalized);
                    budget.packet(40 + key.size() + normalized.size() + point.size() + record.size());
                    budget.add(sizeof(Expected) + key.size());
                    Writer w;
                    w.word(expected.size());
                    w.bytes(key);
                    w.bytes(normalized);
                    w.bytes(point);
                    w.bytes(record);
                    requests.push_back({authority(key, comm->getSize()), std::move(w.data)});
                    expected.push_back({std::move(key), leaf.key});
                }
            }
        });
    auto incoming = exchange_owner_packets(requests, comm, options.byte_budget);
    std::vector<OwnerPacket> replies;
    collective(*comm,
        [&]
        {
            struct Group
            {
                AdaptiveBytes point;
                std::vector<Contact> cells;
            };
            std::map<AdaptiveBytes, Group> groups;
            Budget budget{options.byte_budget};
            for (const auto& packet : incoming.packets)
            {
                budget.add(packet.bytes.size() + sizeof(Contact) + sizeof(Group) + 6 * sizeof(void*));
                Reader r{packet.bytes};
                auto ordinal = r.word();
                auto key = r.bytes(), point = r.bytes(), physical_point = r.bytes();
                auto contact = decode_contact(r.bytes());
                contact.physical_point = std::move(physical_point);
                r.done();
                if (key != point_key(contact.domain, point) || authority(key, comm->getSize()) != comm->getRank())
                    throw std::invalid_argument("Foreign closed-contact request.");
                contact.owner = packet.peer;
                contact.ordinal = ordinal;
                auto& group = groups[key];
                if (!group.point.empty() && group.point != point)
                    throw std::invalid_argument("Closure point key disagreement.");
                group.point = std::move(point);
                for (const auto& existing : group.cells)
                    if (existing.key == contact.key)
                        throw std::invalid_argument("Duplicate closed-contact cell owner.");
                group.cells.push_back(std::move(contact));
            }
            std::size_t pairs = 0;
            for (auto& [key, group] : groups)
            {
                const auto n = group.cells.size();
                std::size_t left = n, right = n ? n - 1 : 0;
                if (left % 2 == 0)
                    left /= 2;
                else
                    right /= 2;
                if (right && left > (options.contact_budget - pairs) / right)
                    throw std::invalid_argument("Closure contact work budget exhausted.");
                pairs += left * right;
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < i; ++j)
                    {
                        auto& a = group.cells[i];
                        auto& b = group.cells[j];
                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            if (a.geometry->axis_span_exceeds(*b.geometry, axis, options.maximum_span_ratio))
                                require(a, a.factors[axis]);
                            if (b.geometry->axis_span_exceeds(*a.geometry, axis, options.maximum_span_ratio))
                                require(b, b.factors[axis]);
                        }
                        if (!b.traces.empty() && a.traces.size() > (options.contact_budget - pairs) / b.traces.size())
                            throw std::invalid_argument("Closure trace work budget exhausted.");
                        pairs += a.traces.size() * b.traces.size();
                        for (const auto& first : a.traces)
                            for (const auto& second : b.traces)
                            {
                                if (first.stem != second.stem || (!first.terminal && !second.terminal) ||
                                    (first.face && !first.face->contains_point(a.physical_point)) ||
                                    (second.face && !second.face->contains_point(b.physical_point)))
                                    continue;
                                if (first.depth < second.depth &&
                                    second.depth - first.depth > options.maximum_trace_jump)
                                    require(a, std::uint64_t(first.split));
                                if (second.depth < first.depth &&
                                    first.depth - second.depth > options.maximum_trace_jump)
                                    require(b, std::uint64_t(second.split));
                            }
                    }
                for (const auto& cell : group.cells)
                {
                    Writer w;
                    w.word(cell.ordinal);
                    w.bytes(key);
                    w.bytes(encode_adaptive_key(cell.key));
                    w.word(cell.required ? 1 : 0);
                    if (cell.required)
                        w.word(std::uint64_t(*cell.required));
                    budget.packet(w.data.size());
                    replies.push_back({cell.owner, std::move(w.data)});
                }
            }
        });
    auto returned = exchange_owner_packets(replies, comm, options.byte_budget);
    std::vector<AdaptiveMark> result;
    collective(*comm,
        [&]
        {
            Budget budget{options.byte_budget};
            budget.add(expected.size());
            std::vector<bool> seen(expected.size());
            std::map<AdaptiveCellKey, AdaptiveSplit> marks;
            for (const auto& packet : returned.packets)
            {
                Reader r{packet.bytes};
                auto ordinal = r.word();
                auto point = r.bytes();
                auto cell = decode_adaptive_key(r.bytes());
                auto required = r.word();
                if (ordinal >= expected.size() || seen[ordinal] || required > 1 ||
                    expected[ordinal].point_key != point || expected[ordinal].cell != cell ||
                    packet.peer != authority(point, comm->getSize()))
                    throw std::invalid_argument("Invalid closure authority reply.");
                seen[ordinal] = true;
                if (required)
                {
                    auto split = r.word();
                    if (split > 2)
                        throw std::invalid_argument("Invalid closure factor reply.");
                    auto [it, inserted] = marks.emplace(cell, AdaptiveSplit(split));
                    if (!inserted && it->second != AdaptiveSplit(split))
                        it->second = AdaptiveSplit::Isotropic;
                }
                r.done();
            }
            for (bool value : seen)
                if (!value)
                    throw std::invalid_argument("Missing closure authority reply.");
            for (const auto& [key, split] : marks)
            {
                budget.add(sizeof(AdaptiveMark) + encode_adaptive_key(key).size());
                result.push_back({key, split});
            }
        });
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
