/** @file DistributedAdaptiveClosure.cc
 * @brief Point-authority closed-neighbor proofs without global leaf catalogs.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/ScalarPacket.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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
    std::size_t point_observations = 0, catalogue_nodes = 0, global_physical_nodes = 0;
    std::size_t tile_input_bytes = 0, request_state_peak_bytes = 0, tile_and_request_peak_bytes = 0;
    bool tile_input_live = false;
    std::size_t planned_reply_records = 0, retired_groups = 0, retired_group_bytes = 0;
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
                "work=%zu point_observations=%zu catalogue_nodes=%zu global_physical_nodes=%zu "
                "tile_input_bytes=%zu request_state_peak_bytes=%zu tile_and_request_peak_bytes=%zu "
                "planned_reply_records=%zu retired_groups=%zu retired_group_bytes=%zu\n",
                rank, stage, cells, faces, contacts, geometry_records, trace_records, unique_record_bytes,
                replicated_record_bytes, transmitted_geometry_records, decoded_geometry_records, decoded_geometry_bytes,
                cumulative_bytes, live_peak_bytes, input_retained_bytes, retained_geometry_bytes, work,
                point_observations, catalogue_nodes, global_physical_nodes, tile_input_bytes, request_state_peak_bytes,
                tile_and_request_peak_bytes, planned_reply_records, retired_groups, retired_group_bytes);
    }
};
struct BudgetObservation
{
    const Budget& budget;
    ClosureDiagnostics& diagnostics;
    bool request_phase = false;
    ~BudgetObservation()
    {
        diagnostics.cumulative_bytes += budget.cumulative;
        diagnostics.live_peak_bytes = std::max(diagnostics.live_peak_bytes, budget.peak);
        if (request_phase)
        {
            diagnostics.request_state_peak_bytes = budget.peak;
            diagnostics.tile_and_request_peak_bytes = std::max(diagnostics.tile_and_request_peak_bytes,
                budget.peak + (diagnostics.tile_input_live ? diagnostics.tile_input_bytes : 0));
        }
    }
};
struct ContactRecord
{
    AdaptiveCellKey key;
    std::size_t identity = 0;
    AdaptiveBytes domain;
    std::array<long double, 3> spans{};
    std::array<std::uint64_t, 3> factors{};
    std::vector<AdaptiveTraceFace> traces;
    bool traces_point_independent = true;
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
std::size_t key_capacity(const AdaptiveCellKey& key)
{
    return std::visit(
        [](const auto& value) -> std::size_t
        {
            if constexpr (requires { value.path; })
                return value.path.capacity() * sizeof(std::uint64_t);
            else if constexpr (requires {
                                   value.base;
                                   value.axial;
                               })
                return (value.base.path.capacity() + value.axial.path.capacity()) * sizeof(std::uint64_t);
            else
                return 0;
        },
        key.ancestry);
}
struct NodeIdentity
{
    std::span<const char> domain, physical;
};
NodeIdentity node_identity(const AdaptiveNode& node)
{
    Reader reader{node.identity};
    auto field = [&]()
    {
        const auto size = reader.word();
        if (size > reader.data.size() - reader.offset)
            throw std::invalid_argument("Invalid closure catalogue identity length.");
        const auto value = reader.data.subspan(reader.offset, size);
        reader.offset += size;
        return value;
    };
    const auto domain = field(), physical = field();
    reader.done();
    if (domain.empty() || physical.empty() ||
        !std::ranges::equal(physical, PLIC::canonical_key(node.coordinate_identity)))
        throw std::invalid_argument("Invalid closure catalogue physical identity.");
    return {domain, physical};
}
AdaptiveBytes encode_contact(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, Budget& budget,
    IdentityBudget& identities, std::size_t& trace_records, std::size_t& retained_geometry)
{
    auto shape = kernel.geometry(leaf);
    auto trace = kernel.traces(leaf);
    if (trace.key != leaf.key)
        throw std::invalid_argument("Provider trace identifies another leaf.");
    trace_records += trace.faces.size();
    const auto before_shape = budget.used;
    identities.retain(budget, shape);
    retained_geometry = budget.used - before_shape;
    std::array<long double, 3> spans;
    {
        // Preserve the original receiver's certificate validation and its
        // decoded coordinate frame before discarding the transient shell.
        // Closure authorities need only these lossless physical axis spans.
        auto certificate = shape->encode_certificate(budget.limit);
        budget.add(certificate.capacity());
        auto decoded =
            PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::decode_certificate(certificate, budget.limit);
        const auto decoded_bytes = decoded->storage_bytes();
        budget.add(decoded_bytes);
        const auto measures = decoded->axis_spans();
        for (std::size_t axis = 0; axis < 3; ++axis)
            spans[axis] = measures[axis].value;
        const auto released = certificate.capacity() + decoded_bytes;
        decoded.reset();
        AdaptiveBytes().swap(certificate);
        budget.used -= released;
    }
    Writer w;
    write_geometry_record_header(w, GeometryRecordKind::ClosureSpanContact);
    w.bytes(encode_adaptive_key(leaf.key));
    w.bytes(leaf.domain);
    w.word(std::numeric_limits<long double>::digits);
    w.word(std::numeric_limits<long double>::max_exponent);
    for (const auto span : spans)
        for (const auto word : PLIC::scalar_packet(span))
            w.word(word);
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
            budget.add(face.face->encode_incidence(budget.limit).size());
            w.bytes(face.face->encode_incidence(budget.limit));
        }
    }
    budget.add(w.data.capacity());
    return std::move(w.data);
}
ContactRecord decode_contact(std::span<const char> bytes, std::size_t limit)
{
    if (bytes.size() > limit)
        throw std::invalid_argument("Closure contact exceeds byte budget.");
    Reader r{bytes};
    read_geometry_record_header(r, GeometryRecordKind::ClosureSpanContact);
    ContactRecord contact;
    contact.key = decode_adaptive_key(r.bytes());
    contact.domain = r.bytes();
    if (contact.domain.empty())
        throw std::invalid_argument("Closure contact requires connected-domain provenance.");
    if (r.word() != std::numeric_limits<long double>::digits ||
        r.word() != std::numeric_limits<long double>::max_exponent)
        throw std::invalid_argument("Incompatible closure span scalar representation.");
    for (auto& span : contact.spans)
    {
        PLIC::ScalarPacketWords words;
        for (auto& word : words)
            word = r.word();
        span = PLIC::scalar_from_packet(words);
        if (!(span > 0))
            throw std::invalid_argument("Invalid closure physical axis span.");
    }
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
        {
            face.face = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes(), limit);
            contact.traces_point_independent = false;
        }
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
    std::vector<AdaptiveNode> native_nodes;
    AdaptiveAuthorityStats authority_statistics;
    auto tiles = decide_adaptive_tiles(
        transient, comm, options.byte_budget, options.tile_budget, &authority_statistics, &native_nodes);
    diagnostics.catalogue_nodes = native_nodes.size();
    diagnostics.global_physical_nodes = authority_statistics.global_nodes;
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
            BudgetObservation observation{budget, diagnostics, true};
            for (const auto& source : owned)
                if (source.geometry)
                    identities.retain(budget, source.geometry);
            // Returned tiles retain the established borrowed-input scope. Report
            // their conservative baseline separately from the admitted builder
            // state, and measure their actual overlap before destroying them.
            diagnostics.tile_input_bytes = tiles.capacity() * sizeof(AdaptiveTile);
            for (const auto& tile : tiles)
            {
                diagnostics.tile_input_bytes +=
                    tile.domain.capacity() + tile.surface.capacity() + tile.identity.capacity() +
                    tile.second_image_identity.payload_bytes() + tile.boundary.capacity() + 1 +
                    (tile.node_gids.capacity() + tile.second_node_gids.capacity()) * sizeof(std::uint64_t) +
                    key_capacity(tile.first) + (tile.second ? key_capacity(*tile.second) : 0);
                if (tile.geometry)
                    diagnostics.tile_input_bytes +=
                        tile.geometry->storage_bytes() + sizeof(*tile.geometry) + 8 * sizeof(void*);
            }
            diagnostics.tile_input_live = true;
            struct CataloguedNode
            {
                const AdaptiveNode* node;
                std::span<const char> domain;
            };
            std::map<std::uint64_t, CataloguedNode> catalogue;
            std::map<AdaptiveCellKey, std::set<std::uint64_t>> points;
            budget.add(native_nodes.capacity() * sizeof(AdaptiveNode) + sizeof(catalogue) + sizeof(points));
            for (const auto& node : native_nodes)
            {
                budget.add(node.identity.capacity() + node.coordinate_identity.payload_bytes());
                const auto identity = node_identity(node);
                budget.add(sizeof(decltype(catalogue)::value_type) + 4 * sizeof(void*));
                if (!catalogue.emplace(node.gid, CataloguedNode{&node, identity.domain}).second)
                    throw std::invalid_argument("Duplicate closure catalogue node GID.");
            }
            for (const auto& tile : tiles)
            {
                if (!tile.geometry)
                    throw std::invalid_argument("Closure tile has no physical geometry.");
                const auto count = tile.geometry->points().size();
                if (tile.node_gids.size() != count || (tile.second && tile.second_node_gids.size() != count) ||
                    (!tile.second && !tile.second_node_gids.empty()))
                    throw std::invalid_argument("Incomplete closure tile node incidence.");
                auto record_points = [&](const AdaptiveCellKey& cell, const std::vector<std::uint64_t>& gids)
                {
                    auto found = points.find(cell);
                    if (found == points.end())
                    {
                        budget.add(sizeof(decltype(points)::value_type) + 4 * sizeof(void*) + key_capacity(cell));
                        found = points.try_emplace(cell).first;
                    }
                    for (const auto gid : gids)
                    {
                        ++diagnostics.point_observations;
                        const auto node = catalogue.find(gid);
                        if (node == catalogue.end() || !std::ranges::equal(node->second.domain, tile.domain))
                            throw std::invalid_argument("Foreign closure tile node identity.");
                        auto& incidence = found->second;
                        if (!incidence.contains(gid))
                        {
                            budget.add(sizeof(std::uint64_t) + 4 * sizeof(void*));
                            incidence.insert(gid);
                        }
                    }
                };
                if (tile.first_owner == comm->getRank())
                    record_points(tile.first, tile.node_gids);
                if (tile.second_owner == comm->getRank())
                {
                    if (!tile.second)
                        throw std::invalid_argument("Closure tile second owner lacks its cell.");
                    record_points(*tile.second, tile.second_node_gids);
                }
            }
            diagnostics.tile_and_request_peak_bytes = diagnostics.tile_input_bytes + budget.peak;
            std::vector<AdaptiveTile>().swap(tiles);
            diagnostics.tile_input_live = false;
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
                    std::vector<const AdaptiveNode*> ordered;
                    for (const auto gid : found->second)
                    {
                        const auto node = catalogue.find(gid);
                        if (node == catalogue.end() || !std::ranges::equal(node->second.domain, leaf.domain))
                            throw std::invalid_argument("Foreign closure cell node identity.");
                        budget.append_capacity(ordered);
                        ordered.push_back(node->second.node);
                    }
                    std::sort(ordered.begin(), ordered.end(), [](const auto* first, const auto* second)
                        { return first->coordinate_identity < second->coordinate_identity; });
                    for (const auto* node : ordered)
                    {
                        const auto& point = node->coordinate_identity;
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
                        std::size_t bytes = 16 + geometry_record_header_bytes + record.size();
                        for (const auto& contact : contacts)
                            bytes += 32 + contact.key.size() + PLIC::encoded_identity_size(contact.normalized) +
                                     PLIC::encoded_identity_size(*contact.physical);
                        budget.packet(bytes);
                        retained_delta += sizeof(OwnerPacket) + bytes;
                        ++diagnostics.transmitted_geometry_records;
                        diagnostics.replicated_record_bytes += record.size();
                        Writer w;
                        w.data.reserve(bytes);
                        write_geometry_record_header(w, GeometryRecordKind::ClosureSpanRequests);
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
    // All Pending references, incidence rows and catalogue lookup nodes are now
    // dead. Destroy the coordinate packet owner before entering the exchange.
    std::vector<AdaptiveNode>().swap(native_nodes);
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
                budget.add(sizeof(ContactRecord) + 2 * sizeof(void*));
                Reader r{packet.bytes};
                read_geometry_record_header(r, GeometryRecordKind::ClosureSpanRequests);
                auto encoded_record = r.bytes();
                budget.add(encoded_record.capacity());
                auto decoded = decode_contact(encoded_record, options.byte_budget);
                const auto record_capacity = encoded_record.capacity();
                decoded.identity = records.size();
                auto record = std::make_shared<const ContactRecord>(std::move(decoded));
                ++diagnostics.decoded_geometry_records;
                if (!records.emplace(record->key, packet.peer).second)
                    throw std::invalid_argument("Duplicate closed-contact geometry owner.");
                budget.add(record->domain.capacity() + 2 * encode_adaptive_key(record->key).size() +
                           sizeof(decltype(records)::value_type) + 4 * sizeof(void*));
                // The geometry record now retains only three decoded scalar
                // spans. Face payloads below retain point-membership geometry.
                diagnostics.decoded_geometry_bytes += sizeof(record->spans);
                budget.add(record->traces.capacity() * sizeof(AdaptiveTraceFace));
                for (const auto& trace : record->traces)
                {
                    budget.add(trace.stem.capacity());
                    if (trace.face)
                        identities.face(budget, *trace.face);
                }
                // Encoded scratch and decoded record capacities coexist until
                // every retained allocation above has been admitted together.
                AdaptiveBytes().swap(encoded_record);
                budget.used -= record_capacity;
                const auto count = r.word();
                if (!count || count > packet.bytes.size() / 32)
                    throw std::invalid_argument("Invalid interned closure contact count.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto ordinal = r.word();
                    auto key = r.bytes();
                    auto point = PLIC::decode_identity<PointIdentity>(r.bytes(), options.byte_budget);
                    auto physical_point = PLIC::decode_identity<PointIdentity>(r.bytes(), options.byte_budget);
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
            // The ownership audit is complete only after every packet's final
            // byte and every incidence have passed validation. Immutable record
            // identities survive independently of these audit-map nodes.
            const auto audit_node_bytes = sizeof(decltype(records)::value_type) + 4 * sizeof(void*);
            if (records.size() > budget.used / audit_node_bytes)
                throw std::invalid_argument("Closure audit lease accounting disagreement.");
            const auto audit_bytes = records.size() * audit_node_bytes;
            records.clear();
            budget.used -= audit_bytes;
            const auto incoming_bytes = incoming.packets.capacity() * sizeof(OwnerPacket);
            std::vector<OwnerPacket>().swap(incoming.packets);
            budget.used -= incoming_bytes;
            std::size_t reply_count = 0;
            for (const auto& [key, group] : groups)
            {
                if (group.cells.size() > std::numeric_limits<std::size_t>::max() - reply_count)
                    throw std::invalid_argument("Closure reply count exceeds its represented range.");
                reply_count += group.cells.size();
            }
            if (reply_count > options.byte_budget / sizeof(OwnerPacket))
                throw std::invalid_argument("Closure reply record capacity exceeds byte budget.");
            diagnostics.planned_reply_records = reply_count;
            // The fresh reply vector has no old allocation. Admit its complete
            // authority-local incidence count once, rather than overlapping
            // growing allocations while every decoded group is still live.
            budget.add(reply_count * sizeof(OwnerPacket));
            replies.reserve(reply_count);
            if (replies.capacity() > reply_count)
                budget.add((replies.capacity() - reply_count) * sizeof(OwnerPacket));
            // Trace comparisons are point-independent only when neither record
            // carries a face membership predicate. Immutable record identities
            // make repeated edge/face contacts safe to memoize in that case.
            struct TraceRequirements
            {
                std::optional<AdaptiveSplit> first, second;
            };
            std::map<std::pair<std::size_t, std::size_t>, TraceRequirements> trace_requirements;
            budget.add(sizeof(trace_requirements));
            std::size_t pairs = 0;
            while (!groups.empty())
            {
                const auto position = groups.begin();
                const auto& key = position->first;
                auto& group = position->second;
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
                            if (a.record->spans[axis] > options.maximum_span_ratio * b.record->spans[axis])
                                require(a, a.record->factors[axis]);
                            if (b.record->spans[axis] > options.maximum_span_ratio * a.record->spans[axis])
                                require(b, b.record->factors[axis]);
                        }
                        const bool point_independent =
                            a.record->traces_point_independent && b.record->traces_point_independent;
                        const bool a_first = a.record->identity < b.record->identity;
                        const auto identities = std::minmax(a.record->identity, b.record->identity);
                        const std::pair<std::size_t, std::size_t> cache_key{identities.first, identities.second};
                        if (point_independent)
                            if (auto cached = trace_requirements.find(cache_key); cached != trace_requirements.end())
                            {
                                const auto& requirements = cached->second;
                                const auto& for_a = a_first ? requirements.first : requirements.second;
                                const auto& for_b = a_first ? requirements.second : requirements.first;
                                if (for_a)
                                    require(a, std::uint64_t(*for_a));
                                if (for_b)
                                    require(b, std::uint64_t(*for_b));
                                continue;
                            }
                        if (!b.record->traces.empty() &&
                            a.record->traces.size() > (options.contact_budget - pairs) / b.record->traces.size())
                            throw std::invalid_argument("Closure trace work budget exhausted.");
                        pairs += a.record->traces.size() * b.record->traces.size();
                        diagnostics.work = pairs;
                        TraceRequirements requirements;
                        auto record_requirement = [](std::optional<AdaptiveSplit>& target, AdaptiveSplit split)
                        {
                            if (!target)
                                target = split;
                            else if (*target != split)
                                target = AdaptiveSplit::Isotropic;
                        };
                        for (const auto& first : a.record->traces)
                            for (const auto& second : b.record->traces)
                            {
                                if (first.stem != second.stem || (!first.terminal && !second.terminal) ||
                                    (first.face && !first.face->contains_point(a.physical_point)) ||
                                    (second.face && !second.face->contains_point(b.physical_point)))
                                    continue;
                                if (first.depth < second.depth &&
                                    second.depth - first.depth > options.maximum_trace_jump)
                                {
                                    require(a, std::uint64_t(first.split));
                                    record_requirement(a_first ? requirements.first : requirements.second, first.split);
                                }
                                if (second.depth < first.depth &&
                                    first.depth - second.depth > options.maximum_trace_jump)
                                {
                                    require(b, std::uint64_t(second.split));
                                    record_requirement(
                                        a_first ? requirements.second : requirements.first, second.split);
                                }
                            }
                        if (point_independent && !a.record->traces.empty() && !b.record->traces.empty())
                        {
                            budget.add(sizeof(decltype(trace_requirements)::value_type) + 4 * sizeof(void*));
                            trace_requirements.emplace(cache_key, requirements);
                        }
                    }
                for (const auto& cell : group.cells)
                {
                    auto encoded_key = encode_adaptive_key(cell.record->key);
                    const auto encoded_key_bytes = encoded_key.capacity();
                    budget.add(encoded_key_bytes);
                    const auto size =
                        32 + geometry_record_header_bytes + key.size() + encoded_key.size() + (cell.required ? 8 : 0);
                    budget.packet(size);
                    Writer w;
                    w.data.reserve(size);
                    write_geometry_record_header(w, GeometryRecordKind::ClosureReply);
                    w.word(cell.ordinal);
                    w.bytes(key);
                    w.bytes(encoded_key);
                    w.word(cell.required ? 1 : 0);
                    if (cell.required)
                        w.word(std::uint64_t(*cell.required));
                    budget.append_capacity(replies);
                    replies.push_back({cell.owner, std::move(w.data)});
                    AdaptiveBytes().swap(encoded_key);
                    budget.used -= encoded_key_bytes;
                }
                // Every comparison and reply for this complete byte-ordered
                // point group has finished. Retire only its own admitted leases;
                // shared immutable record charges stay conservative to the end.
                std::size_t group_bytes = 0;
                auto add_lease = [&](std::size_t count, std::size_t bytes)
                {
                    if (count && bytes > (budget.used - group_bytes) / count)
                        throw std::invalid_argument("Closure point-group lease accounting disagreement.");
                    group_bytes += count * bytes;
                };
                add_lease(1, sizeof(decltype(groups)::value_type) + 4 * sizeof(void*));
                add_lease(1, key.capacity());
                add_lease(1, group.point.payload_bytes());
                add_lease(group.cells.capacity(), sizeof(Contact));
                add_lease(n, sizeof(Contact) + sizeof(Group) + 6 * sizeof(void*));
                for (const auto& cell : group.cells)
                    add_lease(1, cell.physical_point.payload_bytes());
                groups.erase(position);
                budget.used -= group_bytes;
                ++diagnostics.retired_groups;
                diagnostics.retired_group_bytes += group_bytes;
            }
        });
    diagnostics.stage = "contact_reply_exchange";
    // Complete point groups and consumed request storage have been destroyed;
    // only the detached replies and Expected controls cross this handoff.
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
                read_geometry_record_header(r, GeometryRecordKind::ClosureReply);
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
