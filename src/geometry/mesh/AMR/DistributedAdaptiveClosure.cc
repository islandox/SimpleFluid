/** @file DistributedAdaptiveClosure.cc
 * @brief Point-authority closed-neighbor proofs without global leaf catalogs.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
namespace SimpleFluid::Meshes::AMR
{
using PointIdentity = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
namespace
{
using namespace adaptive_detail;
struct ClosureDiagnostics : AdaptiveClosureStats
{
    int rank;
    AdaptiveClosureStats* output;
    const char* stage = "input";
    ClosureDiagnostics(int rank_, AdaptiveClosureStats* output_) : rank(rank_), output(output_) {}
    ~ClosureDiagnostics()
    {
        if (output)
            *output = *this;
        if (std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            std::fprintf(stderr,
                "ADAPTIVE_CLOSURE rank=%d stage=%s cells=%zu faces=%zu contacts=%zu geometry_records=%zu "
                "trace_records=%zu unique_record_bytes=%zu replicated_record_bytes=%zu "
                "transmitted_geometry_records=%zu decoded_geometry_records=%zu decoded_geometry_bytes=%zu "
                "cumulative_bytes=%zu live_peak_bytes=%zu input_retained_bytes=%zu retained_geometry_bytes=%zu "
                "work=%zu\n",
                rank, stage, cells, faces, contacts, geometry_records, trace_records, unique_record_bytes,
                replicated_record_bytes, transmitted_geometry_records, decoded_geometry_records, decoded_geometry_bytes,
                cumulative_bytes, live_peak_bytes, input_retained_bytes, retained_geometry_bytes, work);
    }
};
struct BudgetObservation
{
    const Budget& budget;
    ClosureDiagnostics& diagnostics;
    ~BudgetObservation()
    {
        diagnostics.cumulative_bytes += budget.cumulative;
        diagnostics.live_peak_bytes = std::max(diagnostics.live_peak_bytes, budget.peak);
    }
};
struct ContactRecord
{
    AdaptiveCellKey key;
    AdaptiveBytes domain;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry;
    std::array<std::uint64_t, 3> factors{};
    std::vector<AdaptiveTraceFace> traces;
};
struct Contact
{
    std::shared_ptr<const ContactRecord> record;
    PointIdentity physical_point;
    int owner = -1;
    std::uint64_t ordinal = 0;
    std::optional<AdaptiveSplit> required;
};
AdaptiveBytes point_key(std::span<const char> domain, const PointIdentity& point)
{
    Writer w;
    w.bytes(domain);
    w.bytes(PLIC::canonical_key(point));
    return std::move(w.data);
}
AdaptiveBytes encode_contact(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, Budget& budget,
    IdentityBudget& identities, std::size_t& trace_records, std::size_t& retained_geometry)
{
    auto shape = kernel.geometry(leaf);
    auto trace = kernel.traces(leaf);
    if (trace.key != leaf.key)
        throw std::invalid_argument("Provider trace identifies another leaf.");
    trace_records += trace.faces.size();
    const auto before_geometry = budget.used;
    identities.retain(budget, shape);
    // All admitted owners and ledger pins survive until the request phase ends,
    // including generated geometry. Never release these charges as leaf scratch.
    retained_geometry = budget.used - before_geometry;
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
            budget.add(face.face->exclusive_storage_bytes());
            const auto before_support = budget.used;
            identities.retain(budget, PLIC::planar_support<ArithmeticMode::Ordinary>(*face.face));
            retained_geometry += budget.used - before_support;
            budget.add(face.face->encode_incidence().size());
            w.bytes(face.face->encode_incidence());
        }
    }
    budget.add(w.data.capacity());
    return std::move(w.data);
}
ContactRecord decode_contact(std::span<const char> bytes)
{
    Reader r{bytes};
    ContactRecord contact;
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
    if (split > 2 || std::all_of(contact.record->factors.begin(), contact.record->factors.end(),
                         [](auto factor) { return factor == 3; }))
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
    ClosureDiagnostics diagnostics{comm->getRank(), options.statistics};
    diagnostics.cells = owned.size();
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
            IdentityBudget identities;
            BudgetObservation observation{budget, diagnostics};
            std::set<const void*> retained_faces;
            for (const auto& source : owned)
            {
                budget.add(sizeof(AdaptiveLeaf));
                if (source.geometry && retained_faces.insert(source.geometry.get()).second)
                {
                    const auto before_payload = identities.payload_bytes();
                    identities.retain(budget, source.geometry);
                    budget.add(sizeof(const void*) + 4 * sizeof(void*));
                    diagnostics.input_retained_bytes += identities.payload_bytes() - before_payload;
                }
                auto leaf = source;
                // Tile authorities consume support faces, not cell shells.
                // Full geometry is still certified by the contact encoder.
                leaf.geometry.reset();
                leaf.surfaces = kernel.surfaces(source);
                diagnostics.faces += leaf.surfaces.size();
                diagnostics.input_retained_bytes += sizeof(AdaptiveLeaf) +
                                                    leaf.surfaces.capacity() * sizeof(AdaptiveSurface) +
                                                    leaf.domain.capacity() + leaf.parent_family.capacity();
                for (const auto& surface : leaf.surfaces)
                {
                    diagnostics.input_retained_bytes +=
                        surface.key.capacity() + surface.boundary.capacity() + surface.authority_image.payload_bytes();
                    if (retained_faces.insert(surface.geometry.get()).second)
                    {
                        budget.add(sizeof(const void*) + 4 * sizeof(void*));
                        const auto before_payload = identities.payload_bytes();
                        identities.face(budget, *surface.geometry);
                        diagnostics.input_retained_bytes +=
                            surface.geometry->exclusive_storage_bytes() + identities.payload_bytes() - before_payload;
                    }
                    budget.add(surface.key.size() + surface.geometry->encode_incidence().size());
                }
                budget.append_capacity(transient);
                transient.push_back(std::move(leaf));
                const auto spare_capacity = (transient.capacity() - transient.size()) * sizeof(AdaptiveLeaf);
                if (diagnostics.input_retained_bytes > options.byte_budget ||
                    spare_capacity > options.byte_budget - diagnostics.input_retained_bytes)
                    throw std::invalid_argument("Closure input retained capacities exceed byte budget.");
            }
            diagnostics.input_retained_bytes += (transient.capacity() - transient.size()) * sizeof(AdaptiveLeaf);
        });
    diagnostics.stage = "canonical_tiles";
    auto tiles = decide_adaptive_tiles(transient, comm, options.byte_budget, options.tile_budget);
    std::vector<AdaptiveLeaf>().swap(transient);
    diagnostics.stage = "contact_requests";
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
            IdentityBudget identities;
            BudgetObservation observation{budget, diagnostics};
            for (const auto& source : owned)
                if (source.geometry)
                    identities.retain(budget, source.geometry);
            std::map<AdaptiveCellKey, std::set<PointIdentity>> points;
            for (const auto& tile : tiles)
            {
                auto record_points =
                    [&](const AdaptiveCellKey& cell, const PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>& face)
                {
                    for (const auto& point : face->vertex_identities())
                    {
                        auto& incidence = points[cell];
                        // Retain the original cumulative point-admission charge,
                        // including repeated observations of an existing point.
                        budget.add(point.payload_bytes() + sizeof(PointIdentity) + 3 * sizeof(void*));
                        if (!incidence.contains(point))
                        {
                            incidence.insert(point);
                        }
                    }
                };
                if (tile.first_owner == comm->getRank())
                    record_points(tile.first, tile.geometry);
                if (tile.second_owner == comm->getRank())
                    record_points(*tile.second,
                        tile.geometry->translated(
                            PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, tile.second_image_identity)));
            }
            std::vector<AdaptiveTile>().swap(tiles);
            // Provider queries consume retained provider records, not the
            // canonical authority copies above. A Mixed retained polyhedron
            // stores original support keys; mapping its already-rewritten
            // transient surfaces again would lose the physical selections.
            for (const auto& leaf : owned)
            {
                auto found = points.find(leaf.key);
                if (found == points.end())
                    throw std::invalid_argument("Closure has no complete owned contact incidence.");
                std::size_t released = 0;
                {
                    const auto before_leaf = budget.used;
                    std::size_t retained_delta = 0, retained_geometry = 0;
                    auto record =
                        encode_contact(leaf, kernel, budget, identities, diagnostics.trace_records, retained_geometry);
                    ++diagnostics.geometry_records;
                    diagnostics.unique_record_bytes += record.size();
                    struct Pending
                    {
                        const PointIdentity* physical;
                        PointIdentity normalized;
                        AdaptiveBytes key;
                        std::uint64_t ordinal;
                    };
                    std::map<int, std::vector<Pending>> destinations;
                    for (const auto& point : found->second)
                    {
                        auto normalized = kernel.contact_point(leaf, point);
                        auto key = point_key(leaf.domain, normalized);
                        ++diagnostics.contacts;
                        budget.add(key.size() + normalized.payload_bytes());
                        budget.add(key.size());
                        retained_delta += key.size();
                        const auto peer = authority(key, comm->getSize());
                        auto& destination = destinations[peer];
                        budget.append_capacity(destination);
                        const auto before_expected = budget.used;
                        budget.append_capacity(expected);
                        retained_delta += budget.used - before_expected;
                        destination.push_back({&point, std::move(normalized), key, expected.size()});
                        expected.push_back({std::move(key), leaf.key});
                    }
                    // One immutable geometry/trace record per leaf and destination,
                    // irrespective of the number of its closed contact points.
                    for (const auto& [peer, contacts] : destinations)
                    {
                        std::size_t bytes = 24 + record.size();
                        for (const auto& contact : contacts)
                            bytes += 32 + contact.key.size() + contact.normalized.payload_bytes() +
                                     contact.physical->payload_bytes();
                        budget.packet(bytes);
                        retained_delta += sizeof(OwnerPacket) + bytes;
                        ++diagnostics.transmitted_geometry_records;
                        diagnostics.replicated_record_bytes += record.size();
                        Writer w;
                        w.data.reserve(bytes);
                        w.word(1); // Interned-contact packet version.
                        w.bytes(record);
                        w.word(contacts.size());
                        for (const auto& contact : contacts)
                        {
                            w.word(contact.ordinal);
                            w.bytes(contact.key);
                            w.bytes(PLIC::encode_identity(contact.normalized));
                            w.bytes(PLIC::encode_identity(*contact.physical));
                        }
                        const auto before_requests = budget.used;
                        budget.append_capacity(requests);
                        retained_delta += budget.used - before_requests;
                        requests.push_back({peer, std::move(w.data)});
                    }
                    diagnostics.retained_geometry_bytes += retained_geometry;
                    released = budget.used - before_leaf - retained_delta - retained_geometry;
                }
                // Record/certificate and destination scratch are dead now.
                // Expected replies, output packets and source-shared geometry
                // keep their charges. Allocation/work totals never reset.
                budget.used -= released;
            }
        });
    diagnostics.stage = "contact_exchange";
    // Canonical tiles and provider-expanded leaves are no longer needed once
    // their complete closed point incidence has been authenticated and packed.
    std::vector<AdaptiveLeaf>().swap(transient);
    std::vector<AdaptiveTile>().swap(tiles);
    auto incoming = consume_owner_packets(std::move(requests), comm, options.byte_budget);
    std::vector<OwnerPacket>().swap(requests);
    diagnostics.stage = "contact_decode_compare";
    std::vector<OwnerPacket> replies;
    collective(*comm,
        [&]
        {
            struct Group
            {
                PointIdentity point;
                std::vector<Contact> cells;
            };
            std::map<AdaptiveBytes, Group> groups;
            Budget budget{options.byte_budget};
            IdentityBudget identities;
            BudgetObservation observation{budget, diagnostics};
            for (const auto& source : owned)
                if (source.geometry)
                    identities.retain(budget, source.geometry);
            budget.add(incoming.packets.capacity() * sizeof(OwnerPacket));
            for (const auto& packet : incoming.packets)
                budget.add(packet.bytes.capacity());
            budget.add(expected.capacity() * sizeof(Expected));
            for (const auto& value : expected)
                budget.add(value.point_key.capacity() + encode_adaptive_key(value.cell).size());
            std::map<AdaptiveCellKey, int> records;
            for (auto& packet : incoming.packets)
            {
                const auto encoded_capacity = packet.bytes.capacity();
                budget.add(sizeof(ContactRecord));
                Reader r{packet.bytes};
                if (r.word() != 1)
                    throw std::invalid_argument("Invalid interned closure packet version.");
                auto record = std::make_shared<const ContactRecord>(decode_contact(r.bytes()));
                ++diagnostics.decoded_geometry_records;
                if (!records.emplace(record->key, packet.peer).second)
                    throw std::invalid_argument("Duplicate closed-contact geometry owner.");
                budget.add(record->domain.capacity() + 2 * encode_adaptive_key(record->key).size() +
                           sizeof(decltype(records)::value_type) + 4 * sizeof(void*));
                const auto before_geometry = identities.payload_bytes();
                identities.retain(budget, record->geometry);
                diagnostics.decoded_geometry_bytes += identities.payload_bytes() - before_geometry;
                budget.add(record->traces.capacity() * sizeof(AdaptiveTraceFace));
                for (const auto& trace : record->traces)
                {
                    budget.add(trace.stem.capacity());
                    if (trace.face)
                        identities.face(budget, *trace.face);
                }
                const auto count = r.word();
                if (!count || count > packet.bytes.size() / 32)
                    throw std::invalid_argument("Invalid interned closure contact count.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto ordinal = r.word();
                    auto key = r.bytes();
                    auto point = PLIC::decode_identity<PointIdentity>(r.bytes());
                    auto physical_point = PLIC::decode_identity<PointIdentity>(r.bytes());
                    const auto point_metadata = key.capacity() + point.payload_bytes();
                    // Preserve the original conservative per-incidence object
                    // charge in addition to admitted vector capacities.
                    budget.add(sizeof(Contact) + sizeof(Group) + 6 * sizeof(void*) + point_metadata +
                               physical_point.payload_bytes());
                    if (key != point_key(record->domain, point) || authority(key, comm->getSize()) != comm->getRank())
                        throw std::invalid_argument("Foreign closed-contact request.");
                    Contact contact{record, std::move(physical_point), packet.peer, ordinal, {}};
                    auto [position, inserted] = groups.try_emplace(std::move(key), Group{point, {}});
                    auto& group = position->second;
                    if (group.point != point)
                        throw std::invalid_argument("Closure point key disagreement.");
                    if (inserted)
                    {
                        budget.add(sizeof(decltype(groups)::value_type) + 4 * sizeof(void*));
                        group.point = std::move(point);
                    }
                    for (const auto& existing : group.cells)
                        if (existing.record->key == record->key)
                            throw std::invalid_argument("Duplicate closed-contact cell owner.");
                    budget.append_capacity(group.cells);
                    group.cells.push_back(std::move(contact));
                    if (!inserted)
                    {
                        // The existing group retains its one key and normalized
                        // point. These reader copies are no longer needed.
                        std::vector<char>().swap(key);
                        budget.used -= point_metadata;
                    }
                }
                r.done();
                std::vector<char>().swap(packet.bytes);
                budget.used -= encoded_capacity;
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
                diagnostics.work = pairs;
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < i; ++j)
                    {
                        auto& a = group.cells[i];
                        auto& b = group.cells[j];
                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            if (a.record->geometry->axis_span_exceeds(
                                    *b.record->geometry, axis, options.maximum_span_ratio))
                                require(a, a.record->factors[axis]);
                            if (b.record->geometry->axis_span_exceeds(
                                    *a.record->geometry, axis, options.maximum_span_ratio))
                                require(b, b.record->factors[axis]);
                        }
                        if (!b.record->traces.empty() &&
                            a.record->traces.size() > (options.contact_budget - pairs) / b.record->traces.size())
                            throw std::invalid_argument("Closure trace work budget exhausted.");
                        pairs += a.record->traces.size() * b.record->traces.size();
                        diagnostics.work = pairs;
                        for (const auto& first : a.record->traces)
                            for (const auto& second : b.record->traces)
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
                    const auto encoded_key = encode_adaptive_key(cell.record->key);
                    const auto size = 32 + key.size() + encoded_key.size() + (cell.required ? 8 : 0);
                    budget.packet(size);
                    Writer w;
                    w.data.reserve(size);
                    w.word(cell.ordinal);
                    w.bytes(key);
                    w.bytes(encoded_key);
                    w.word(cell.required ? 1 : 0);
                    if (cell.required)
                        w.word(std::uint64_t(*cell.required));
                    budget.append_capacity(replies);
                    replies.push_back({cell.owner, std::move(w.data)});
                }
            }
        });
    diagnostics.stage = "contact_reply_exchange";
    // Decoded authority groups have been destroyed by the collective lambda.
    // Release received request storage before packing the authority replies.
    std::vector<OwnerPacket>().swap(incoming.packets);
    auto returned = consume_owner_packets(std::move(replies), comm, options.byte_budget);
    std::vector<OwnerPacket>().swap(replies);
    std::vector<AdaptiveMark> result;
    collective(*comm,
        [&]
        {
            Budget budget{options.byte_budget};
            IdentityBudget identities;
            BudgetObservation observation{budget, diagnostics};
            for (const auto& source : owned)
                if (source.geometry)
                    identities.retain(budget, source.geometry);
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
    diagnostics.stage = "complete";
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
