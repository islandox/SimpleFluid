#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/PLIC/details/NativeGeometryByteWriter.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/AMR/NativeSparseAssembler.hh"
#include <bit>
#include <numeric>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Admission = geometry_detail::NativeConstructionAdmission;
using Point = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
bool occurrence_less(const NativeNumberingOccurrence& a, const NativeNumberingOccurrence& b)
{
    if (a.address != b.address)
        return std::less<const void*>{}(a.address, b.address);
    if (a.control.owner_before(b.control))
        return true;
    if (b.control.owner_before(a.control))
        return false;
    return std::pair{a.slot, a.side} < std::pair{b.slot, b.side};
}
bool occurrence_equal(const NativeNumberingOccurrence& a, const NativeNumberingOccurrence& b)
{
    return !occurrence_less(a, b) && !occurrence_less(b, a);
}
NativeNumberingOccurrence observe(const NativeSubfaceOccurrence& occurrence)
{
    return {occurrence.control.get(), occurrence.control, occurrence.slot, occurrence.side};
}
template<class Pages, class Emit> void rows(const typename Pages::Ptr& root, const Emit& emit)
{
    if (!root)
        return;
    if (root->leaf())
        for (const auto& row : root->rows)
            emit(row);
    else
        for (const auto& child : root->children)
            rows<Pages>(child, emit);
}
std::shared_ptr<Budget> arena(const Admission& a)
{
    return std::static_pointer_cast<Budget>(std::const_pointer_cast<void>(a.state()->counter_owner));
}
Point point(const CanonicalSubfaceView& view, std::size_t slot, bool second)
{
    auto p = view.point(slot);
    if (second && !view.second_image().is_zero())
        p = PLIC::point_translated<ArithmeticMode::Ordinary>(
            p, PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, view.second_image()));
    return p;
}
struct PointKey
{
    std::span<const char> domain;
    Point point;
    std::size_t size() const { return domain.size() + 16 + PLIC::canonical_key_size(point); }
    char at(std::size_t i) const
    {
        if (i < 8)
            return static_cast<char>(domain.size() >> (8 * i));
        i -= 8;
        if (i < domain.size())
            return domain[i];
        i -= domain.size();
        const auto n = PLIC::canonical_key_size(point);
        if (i < 8)
            return static_cast<char>(n >> (8 * i));
        i -= 8;
        return static_cast<char>(
            PLIC::details::GeometryIdentityAccess::canonical_word_at(point, i & ~std::size_t(7)) >> (8 * (i % 8)));
    }
};
int compare(const NativeNumberingPoint& row, const PointKey& wanted)
{
    const auto n = std::min(row.key.size(), wanted.size());
    for (std::size_t i = 0; i < n; ++i)
        if (row.key[i] != wanted.at(i))
            return row.key[i] < wanted.at(i) ? -1 : 1;
    return row.key.size() == wanted.size() ? 0 : (row.key.size() < wanted.size() ? -1 : 1);
}
std::shared_ptr<const NativeCellSourceProof> source(
    NativePublicationSession& session, const AdaptiveCellKey& key, std::span<const NativeRemoteSource> remote)
{
    if (const auto local = session.inputs().layout->find(key))
        return session.source(*local);
    const auto at = std::lower_bound(remote.begin(), remote.end(), key,
        [](const auto& row, const auto& key) { return row.source->source().key < key; });
    if (at == remote.end() || at->source->source().key != key)
        throw std::invalid_argument("Native numbering lacks current remote source.");
    return at->source;
}
std::uint64_t gid(
    const NativePublicationSession& session, const AdaptiveCellKey& key, std::span<const NativeRemoteSource> remote)
{
    if (const auto ordinal = session.inputs().layout->ordinal_of(key))
        return session.inputs().cell_begin + *ordinal;
    const auto at = std::lower_bound(remote.begin(), remote.end(), key,
        [](const auto& row, const auto& key) { return row.source->source().key < key; });
    if (at == remote.end() || at->source->source().key != key)
        throw std::invalid_argument("Native numbering lacks remote GID.");
    return at->gid;
}
NativeIncidenceTicket current_incidence(
    const NativePublicationSession& session, const NativeCellSourceProof& source, int owner)
{
    if (source.ticket().issuer != owner)
        throw std::invalid_argument("Native numbering source differs from its current contact owner.");
    if (owner != session.inputs().comm->getRank())
        return {};
    const auto row = session.indexes()->find(source.source().key);
    if (!row || row->source.get() != &source || !row->incidence || &row->incidence->source() != &source ||
        row->incidence->remote_record() || row->incidence->ticket().issuer != owner)
        throw std::invalid_argument("Native numbering lacks a current owner-local incidence proof.");
    return row->incidence->ticket();
}
const NativeSourceFaceRoute& native_face(const NativeCellSourceProof& proof, const CanonicalSubfaceIncident& incident)
{
    const auto faces = proof.faces();
    const auto at =
        std::find_if(faces.begin(), faces.end(), [&](const auto& f) { return f.surface == incident.source_surface(); });
    if (at == faces.end())
        throw std::invalid_argument("Native numbered piece lost exact source-face binding.");
    return *at;
}
bool bind_native_nodes(const NativeCellSourceProof& source, std::uint64_t face, const CanonicalSubfaceView& piece,
    bool second, geometry_detail::NativeConstructorVector<std::uint64_t>& result)
{
    const auto& fragment = source.fragment()->region();
    return std::visit(
        [&](const auto& region)
        {
            const auto native = region.topology().face_nodes(face);
            result.reserve(piece.point_count());
            std::size_t matched = 0;
            for (std::size_t i = 0; i < piece.point_count(); ++i)
            {
                const auto display = PLIC::point_display<ArithmeticMode::Ordinary>(point(piece, i, second));
                std::uint64_t found = UINT64_MAX;
                for (const auto node : native)
                {
                    const auto p = region.geometry().node_coordinates(node);
                    if (p.x == display[0] && p.y == display[1] && p.z == display[2])
                    {
                        if (found != UINT64_MAX)
                            throw std::invalid_argument("Ambiguous native source point display.");
                        found = node;
                    }
                }
                result.push_back(found);
                matched += found != UINT64_MAX;
            }
            if (matched != native.size() || matched != piece.point_count())
                return false;
            for (std::size_t i = 0; i < result.size(); ++i)
                for (std::size_t j = 0; j < i; ++j)
                    if (result[i] == result[j])
                        return false;
            return true;
        },
        fragment);
}
std::pair<std::uint64_t, std::uint64_t> entity_range(std::uint64_t local, const Teuchos::Comm<int>& comm)
{
    collective(comm,
        [&]
        {
            if (local > UINT64_MAX / std::uint64_t(comm.getSize()))
                throw std::invalid_argument("Native numeric prefix count overflows.");
        });
    return owner_packet_entity_range(comm, local);
}
} // namespace
bool NativeNumberingPieceLess::operator()(const NativeNumberingPieceRow& a, const NativeNumberingPieceRow& b) const
{
    return occurrence_less(a.record->occurrence, b.record->occurrence);
}
void NativePublicationNumbering::visit_current_pieces(
    const AdaptiveSubfaceSnapshot& current, const NativeNumberingPieceEmit& emit) const
{
    if (current.size() != pieces.size())
        throw std::invalid_argument("Native numbering differs from current subface membership.");
    std::size_t slot = 0;
    current.visit_pieces(
        [&](const CanonicalSubfaceView& view)
        {
            if (slot >= pieces.size())
                throw std::invalid_argument("Native numbering has no route for a current subface.");
            const auto& route = pieces[slot];
            const auto live = NativePublicationSession::occurrence(view);
            if (!route.record || route.record->occurrence.slot != live.slot ||
                route.record->occurrence.side != live.side || !route.record->page_observation.matches_live(live.control))
                throw std::invalid_argument("Native numbering route has a foreign current subface occurrence.");
            emit(view, route, slot);
            ++slot;
        });
    if (slot != pieces.size())
        throw std::invalid_argument("Native numbering has an unvisited current subface route.");
}
void NativePublicationNumbering::index_local_incidence(std::span<std::size_t> ordered) const
{
    if (ordered.size() != pieces.size())
        throw std::invalid_argument("Native local incidence index extent differs from current numbered pieces.");
    std::iota(ordered.begin(), ordered.end(), std::size_t{0});
    std::sort(ordered.begin(), ordered.end(),
        [&](auto a, auto b) { return occurrence_less(pieces[a].record->occurrence, pieces[b].record->occurrence); });
}
void NativePublicationNumbering::select_local_incidence(const NativeCellIncidenceProof& incidence,
    std::span<const std::size_t> ordered, geometry_detail::NativeConstructorVector<LocalIncidenceRoute>& selected) const
{
    const auto& source = incidence.source();
    const auto& key = source.source().key;
    selected.clear();
    selected.reserve(incidence.pieces().size());
    for (const auto& view : incidence.pieces())
    {
        const auto live = NativePublicationSession::occurrence(view);
        auto occurrence = observe(live);
        // The global numbering has one canonical first-side record. The
        // independently verified cell proof carries its own incident side.
        occurrence.side = false;
        const auto at = std::lower_bound(ordered.begin(), ordered.end(), occurrence,
            [&](auto slot, const auto& target) { return occurrence_less(pieces[slot].record->occurrence, target); });
        if (at == ordered.end())
            throw std::invalid_argument("Native incidence proof has no current numbered route.");
        const auto& record = *pieces[*at].record;
        const auto incident = live.side ? view.second() : std::optional{view.first()};
        if (!occurrence_equal(record.occurrence, occurrence) || !record.page_observation.matches_live(live.control) ||
            !incident || incident->key() != key ||
            (live.side ? record.second_ticket : record.first_ticket) != source.ticket() ||
            (live.side ? record.second_incidence : record.first_incidence) != incidence.ticket())
            throw std::invalid_argument("Native numbering route has a foreign current incidence proof.");
        selected.push_back({&view, *at});
    }
    std::sort(selected.begin(), selected.end(), [](const auto& a, const auto& b) { return a.slot < b.slot; });
    // A periodic self incidence has two verified sides but one wire piece.
    selected.erase(std::unique(selected.begin(), selected.end(),
                       [](const auto& a, const auto& b)
                       {
                           if (a.slot != b.slot)
                               return false;
                           if (a.source->side_ == b.source->side_)
                               throw std::invalid_argument("Native incidence proof repeats its canonical side.");
                           return true;
                       }),
        selected.end());
}
void NativePublicationNumbering::visit_local_incidence(
    std::span<const LocalIncidenceRoute> selected, const NativeIncidenceEmit& emit) const
{
    const auto nodes = std::span<const std::uint64_t>(loops);
    for (const auto& item : selected)
    {
        const auto& row = pieces[item.slot];
        auto view = *item.source;
        view.side_ = false;
        const NativeIncidenceNumericPiece route{row.face_gid, row.first_gid, row.second_gid,
            nodes.subspan(row.nodes_begin, row.nodes_size), nodes.subspan(row.second_begin, row.second_size)};
        emit(view, route);
    }
}
std::shared_ptr<NativePublicationNumbering> NativeSparseAssembler::number(
    NativePublicationSession& session, std::span<const NativeRemoteSource> remote)
{
    const auto& a = session.admission();
    const auto& inputs = session.inputs();
    const auto old = session.numbering();
    std::shared_ptr<NativePublicationNumbering> result;
    collective(*inputs.comm,
        [&]
        {
            auto created = geometry_detail::NativeAllocationEnvelopeAccess::make<NativePublicationNumbering>(a, a);
            result = created.owner;
            result->allocation = created.envelope;
            if (old)
            {
                result->piece_root = old->piece_root;
                result->point_root = old->point_root;
                result->remote_root = old->remote_root;
                result->boundary_root = old->boundary_root;
                result->fragments = old->fragments;
            }
        });
    NativePublicationNumbering::PiecePages piece_editor(arena(a), nullptr, {}, a.state());
    NativePublicationNumbering::PointPages point_editor(arena(a), nullptr, {}, a.state());
    geometry_detail::NativeConstructorVector<CanonicalSubfaceView> incidence(a.allocator<CanonicalSubfaceView>());
    collective(*inputs.comm,
        [&]
        {
            // Source and incidence are independent. Warm rows only compare exact
            // controls; changed incidence never rebuilds its source convex shell.
            for (const auto& leaf : *inputs.layout)
            {
                auto proof = session.source(leaf);
                incidence.clear();
                for (const auto& face : proof->faces())
                    inputs.subfaces->visit_face_pieces(
                        leaf.key, face.surface, [&](const CanonicalSubfaceView& piece) { incidence.push_back(piece); });
                (void) session.incidence(proof, incidence);
            }
            result->pieces.reserve(inputs.subfaces->size());
            inputs.subfaces->visit_pieces(
                [&](const CanonicalSubfaceView& view)
                {
                    const auto live = NativePublicationSession::occurrence(view);
                    if (live.side)
                        throw std::invalid_argument("Canonical numbering visit returned a second-side occurrence.");
                    const auto occurrence = observe(live);
                    const auto first_contact = view.first();
                    const auto second_contact = view.second();
                    const auto first = source(session, first_contact.key(), remote);
                    const auto second = second_contact ? source(session, second_contact->key(), remote) : nullptr;
                    const auto first_incidence = current_incidence(session, *first, first_contact.owner());
                    const auto second_incidence = second
                        ? current_incidence(session, *second, second_contact->owner())
                        : NativeIncidenceTicket{};
                    NativeNumberingPieceRoute route;
                    route.first_gid = gid(session, first_contact.key(), remote);
                    if (second_contact)
                        route.second_gid = gid(session, second_contact->key(), remote);
                    const auto previous = NativePublicationNumbering::PiecePages::lower_bound(result->piece_root,
                        [&](const auto& row) { return occurrence_less(row.record->occurrence, occurrence); });
                    const bool same = previous && occurrence_equal(previous->record->occurrence, occurrence);
                    if (same && previous->record->page_observation.matches_live(live.control) &&
                        previous->record->first_ticket == first->ticket() &&
                        previous->record->second_ticket == (second ? second->ticket() : NativeSourceTicket{}) &&
                        previous->record->first_incidence == first_incidence &&
                        previous->record->second_incidence == second_incidence)
                    {
                        route.record = previous->record.get();
                        result->pieces.push_back(route);
                        return;
                    }
                    auto made = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeNumberingPiece>(a, a);
                    auto& record = *made.owner;
                    record.allocation = made.envelope;
                    record.occurrence = occurrence;
                    record.page_observation = view.weak_page_observation();
                    if (!record.page_observation.matches_live(live.control))
                        throw std::invalid_argument("Native numbering lacks the actual subface control witness.");
                    record.first_ticket = first->ticket();
                    record.second_ticket = second ? second->ticket() : NativeSourceTicket{};
                    record.first_fragment = first->fragment();
                    record.second_fragment = second ? second->fragment() : nullptr;
                    record.first_incidence = first_incidence;
                    record.second_incidence = second_incidence;
                    record.physical.second_image = view.second_image();
                    const auto& first_face = native_face(*first, first_contact);
                    record.first_native_face = first_face.native_face;
                    record.first_sign = first_face.outward_sign;
                    const bool uncut =
                        bind_native_nodes(*first, record.first_native_face, view, false, record.first_native_nodes);
                    if (second)
                    {
                        const auto& second_face = native_face(*second, *second_contact);
                        record.second_native_face = second_face.native_face;
                        record.second_sign = -second_face.outward_sign;
                        (void) bind_native_nodes(
                            *second, record.second_native_face, view, true, record.second_native_nodes);
                    }
                    record.physical.cut_incidence = !uncut;
                    if (!uncut)
                    {
                        Budget geometry{inputs.byte_budget};
                        IdentityBudget identity_budget;
                        identity_budget.bind_native(a, inputs.byte_budget);
                        record.physical.override_geometry =
                            view.materialize_native(geometry, identity_budget, a, /*detached_support=*/true).geometry;
                    }
                    auto intern = [&](bool image, auto& points)
                    {
                        points.reserve(view.point_count());
                        for (std::size_t i = 0; i < view.point_count(); ++i)
                        {
                            const PointKey wanted{view.domain(), point(view, i, image)};
                            const auto at = NativePublicationNumbering::PointPages::lower_bound(
                                result->point_root, [&](const auto& row) { return compare(*row.record, wanted) < 0; });
                            std::shared_ptr<const NativeNumberingPoint> value;
                            if (at && compare(*at->record, wanted) == 0)
                                value = at->record;
                            else
                            {
                                auto node =
                                    geometry_detail::NativeAllocationEnvelopeAccess::make<NativeNumberingPoint>(a, a);
                                node.owner->allocation = node.envelope;
                                node.owner->point = wanted.point;
                                node.owner->key.reserve(wanted.size());
                                for (std::size_t b = 0; b < wanted.size(); ++b)
                                    node.owner->key.push_back(wanted.at(b));
                                value = std::move(node.owner);
                                NativeNumberingPointRow row{value};
                                result->point_root = point_editor.edit(result->point_root, row, &row, false);
                            }
                            points.push_back(std::move(value));
                        }
                    };
                    intern(false, record.points);
                    if (second)
                        intern(true, record.second_points);
                    NativeNumberingPieceRow row{made.owner};
                    result->piece_root = piece_editor.edit(result->piece_root, row, &row, same);
                    route.record = made.owner.get();
                    result->pieces.push_back(route);
                });
            if (old)
            {
                geometry_detail::NativeConstructorVector<const NativeNumberingPiece*> ordered(
                    a.allocator<const NativeNumberingPiece*>());
                ordered.reserve(result->pieces.size());
                for (const auto& route : result->pieces)
                    ordered.push_back(route.record);
                std::sort(ordered.begin(), ordered.end(),
                    [](const auto* a, const auto* b) { return occurrence_less(a->occurrence, b->occurrence); });
                rows<NativePublicationNumbering::PiecePages>(old->piece_root,
                    [&](const auto& row)
                    {
                        const auto at = std::lower_bound(ordered.begin(), ordered.end(), row.record.get(),
                            [](const auto& a, const auto& b) { return occurrence_less(a->occurrence, b->occurrence); });
                        if (at == ordered.end() || !occurrence_equal((*at)->occurrence, row.record->occurrence))
                            result->piece_root = piece_editor.edit(result->piece_root, row, nullptr, true);
                    });
            }
            std::size_t count = 0;
            for (const auto& route : result->pieces)
                count = persistent_ordered_detail::checked_sum(count,
                    persistent_ordered_detail::checked_sum(
                        route.record->points.size(), route.record->second_points.size()));
            result->points.reserve(count);
            for (const auto& route : result->pieces)
                for (const auto* list : {&route.record->points, &route.record->second_points})
                    for (const auto& point : *list)
                        result->points.push_back({point.get(), 0});
            std::sort(result->points.begin(), result->points.end(),
                [](const auto& a, const auto& b) { return a.record->key < b.record->key; });
            result->points.erase(std::unique(result->points.begin(), result->points.end(),
                                     [](const auto& a, const auto& b) { return a.record->key == b.record->key; }),
                result->points.end());
            if (old)
                rows<NativePublicationNumbering::PointPages>(old->point_root,
                    [&](const auto& row)
                    {
                        const auto at = std::lower_bound(result->points.begin(), result->points.end(), row.record.get(),
                            [](const auto& a, const auto* b) { return a.record->key < b->key; });
                        if (at == result->points.end() || at->record->key != row.record->key)
                            result->point_root = point_editor.edit(result->point_root, row, nullptr, true);
                    });
            result->loops.reserve(count);
            for (auto& route : result->pieces)
            {
                const auto* record = route.record;
                route.nodes_begin = result->loops.size();
                route.nodes_size = record->points.size();
                for (const auto& node : record->points)
                {
                    const auto at = std::lower_bound(result->points.begin(), result->points.end(), node.get(),
                        [](const auto& a, const auto* b) { return a.record->key < b->key; });
                    result->loops.push_back(at - result->points.begin());
                }
                route.second_begin = result->loops.size();
                route.second_size = record->second_points.size();
                for (const auto& node : record->second_points)
                {
                    const auto at = std::lower_bound(result->points.begin(), result->points.end(), node.get(),
                        [](const auto& a, const auto* b) { return a.record->key < b->key; });
                    result->loops.push_back(at - result->points.begin());
                }
            }
        });
    // Complete authority exchange below assigns canonical sorted owner ranges.
    struct FaceRequest
    {
        std::span<const char> key, support, definition;
        int peer = -1, first_owner = -1, second_owner = -1;
        std::uint64_t request = 0, first_native = UINT64_MAX, second_native = UINT64_MAX;
        unsigned mask = 0;
    };
    geometry_detail::NativeConstructorStorageOwner send_storage;
    collective(*inputs.comm, [&] { send_storage = geometry_detail::native_constructor_storage(a); });
    std::vector<OwnerPacket> send;
    geometry_detail::NativeConstructorVector<int> face_authorities(a.allocator<int>());
    collective(*inputs.comm,
        [&]
        {
            send_storage.owner->reserve(send, result->pieces.size());
            face_authorities.reserve(result->pieces.size());
            result->visit_current_pieces(*inputs.subfaces,
                [&](const CanonicalSubfaceView& view, const NativeNumberingPieceRoute& route, std::size_t slot)
                {
                    const auto& record = *route.record;
                    const auto first = view.first();
                    const auto second = view.second();
                    const auto mask = (first.owner() == inputs.comm->getRank() ? 1u : 0u) |
                                      (second && second->owner() == inputs.comm->getRank() ? 2u : 0u);
                    if (!mask)
                        throw std::invalid_argument("Local numbered face has no owner-local incident.");
                    const auto domain = view.domain();
                    const auto boundary = view.boundary();
                    const auto identity = view.identity();
                    const auto support = view.support();
                    const auto& image = record.physical.second_image;
                    const auto image_size = PLIC::canonical_key_size(image);
                    using persistent_ordered_detail::checked_sum;
                    auto definition_size = checked_sum(9 * sizeof(std::uint64_t), domain.size());
                    definition_size = checked_sum(definition_size, encoded_adaptive_key_size(first.key()));
                    if (second)
                        definition_size = checked_sum(definition_size,
                            checked_sum(3 * sizeof(std::uint64_t), encoded_adaptive_key_size(second->key())));
                    definition_size = checked_sum(definition_size, checked_sum(image_size, boundary.size()));
                    const auto extent = checked_sum(9 * sizeof(std::uint64_t),
                        checked_sum(definition_size, checked_sum(identity.size(), support.size())));
                    Writer w;
                    send_storage.owner->reserve(w.data, extent);
                    w.word(slot);
                    w.word(first.owner());
                    w.word(second ? second->owner() + 1 : 0);
                    w.word(mask);
                    w.word(mask & 1 ? record.first_native_face : UINT64_MAX);
                    w.word(mask & 2 ? record.second_native_face : UINT64_MAX);
                    w.bytes(identity);
                    w.bytes(support);
                    w.word(definition_size);
                    w.bytes(domain);
                    NativeAdaptiveKeyCodec::write(w, first.key());
                    w.word(first.revision());
                    w.word(first.source_surface());
                    w.word(second ? 1 : 0);
                    if (second)
                    {
                        NativeAdaptiveKeyCodec::write(w, second->key());
                        w.word(second->revision());
                        w.word(second->source_surface());
                    }
                    w.word(image_size);
                    PLIC::details::NativeGeometryByteWriter destination{&w.data, &a};
                    destination.native_identity(image);
                    w.bytes({boundary.data(), boundary.size()});
                    w.word(view.generation());
                    w.word(view.emission());
                    if (w.data.size() != extent)
                        throw std::invalid_argument("Native numbered face request differs from derived extent.");
                    const auto peer = authority(support, inputs.comm->getSize());
                    face_authorities.push_back(peer);
                    send.push_back({peer, std::move(w.data)});
                });
        });
    auto remaining = [&]
    {
        const auto c = a.counters();
        return c.limit - c.used;
    };
    auto incoming = exchange_adaptive_packets(send, inputs.comm, a);
    collective(*inputs.comm, [&]
    {
        for (auto packet = send.rbegin(); packet != send.rend(); ++packet) send_storage.owner->reset_buffer(packet->bytes);
        send_storage.owner->reset_buffer(send);
    });

    geometry_detail::NativeConstructorVector<FaceRequest> requests(a.allocator<FaceRequest>());
    std::uint64_t local_faces = 0;
    collective(*inputs.comm,
        [&]
        {
            requests.reserve(incoming.packets.size());
            for (const auto& p : incoming.packets)
            {
                Reader r{p.bytes};
                FaceRequest row;
                row.peer = p.peer;
                row.request = r.word();
                const auto first = r.word(), second = r.word(), mask = r.word();
                row.first_native = r.word();
                row.second_native = r.word();
                row.key = r.bytes_view();
                row.support = r.bytes_view();
                row.definition = r.bytes_view();
                r.done();
                if (first >= std::uint64_t(inputs.comm->getSize()) || second > std::uint64_t(inputs.comm->getSize()) ||
                    !mask || mask > 3 || row.key.empty() || row.support.empty() || row.definition.empty() ||
                    authority(row.support, inputs.comm->getSize()) != inputs.comm->getRank())
                    throw std::invalid_argument("Invalid native numbered-face authority request.");
                row.first_owner = static_cast<int>(first);
                row.second_owner = static_cast<int>(second) - 1;
                row.mask = static_cast<unsigned>(mask);
                const auto expected = (row.first_owner == p.peer ? 1u : 0u) | (row.second_owner == p.peer ? 2u : 0u);
                if (mask != expected || (!(mask & 1) && row.first_native != UINT64_MAX) ||
                    (!(mask & 2) && row.second_native != UINT64_MAX))
                    throw std::invalid_argument("Native numbered-face locator lacks its current source owner.");
                requests.push_back(row);
            }
            std::sort(requests.begin(), requests.end(),
                [](const auto& a, const auto& b)
                {
                    if (!std::ranges::equal(a.key, b.key))
                        return std::lexicographical_compare(a.key.begin(), a.key.end(), b.key.begin(), b.key.end());
                    return a.peer < b.peer;
                });
            for (std::size_t i = 0; i < requests.size();)
            {
                auto end = i + 1;
                while (end < requests.size() && std::ranges::equal(requests[i].key, requests[end].key))
                    ++end;
                unsigned mask = 0;
                for (auto j = i; j < end; ++j)
                {
                    const auto& row = requests[j];
                    if ((mask & row.mask) || row.first_owner != requests[i].first_owner ||
                        row.second_owner != requests[i].second_owner ||
                        !std::ranges::equal(row.definition, requests[i].definition) ||
                        !std::ranges::equal(row.support, requests[i].support) ||
                        (j > i && row.peer == requests[j - 1].peer))
                        throw std::invalid_argument(
                            "Native numbered-face replicas have different exact facts or duplicate membership.");
                    mask |= row.mask;
                }
                if (mask != (requests[i].second_owner < 0 ? 1u : 3u))
                    throw std::invalid_argument("Native numbered-face source inventory is incomplete.");
                ++local_faces;
                i = end;
            }
        });
    const auto [face_begin, global_faces] = entity_range(local_faces, *inputs.comm);
    result->global_faces = global_faces;
    geometry_detail::NativeConstructorStorageOwner reply_storage;
    collective(*inputs.comm, [&] { reply_storage = geometry_detail::native_constructor_storage(a); });
    std::vector<OwnerPacket> replies;
    collective(*inputs.comm,
        [&]
        {
            reply_storage.owner->reserve(replies, requests.size());
            std::uint64_t ordinal = 0;
            for (std::size_t i = 0; i < requests.size();)
            {
                auto end = i + 1;
                while (end < requests.size() && std::ranges::equal(requests[i].key, requests[end].key))
                    ++end;
                std::uint64_t first = UINT64_MAX, second = UINT64_MAX;
                for (auto j = i; j < end; ++j)
                {
                    if (requests[j].mask & 1)
                        first = requests[j].first_native;
                    if (requests[j].mask & 2)
                        second = requests[j].second_native;
                }
                for (auto j = i; j < end; ++j)
                {
                    Writer w;
                    reply_storage.owner->reserve(w.data, 4 * sizeof(std::uint64_t));
                    w.word(requests[j].request);
                    w.word(face_begin + ordinal);
                    w.word(first);
                    w.word(second);
                    replies.push_back({requests[j].peer, std::move(w.data)});
                }
                ++ordinal;
                i = end;
            }
        });
    requests.clear();
    decltype(requests)(a.allocator<FaceRequest>()).swap(requests);
    incoming = {};
    auto returned = exchange_adaptive_packets(replies, inputs.comm, a);
    collective(*inputs.comm, [&]
    {
        for (auto packet = replies.rbegin(); packet != replies.rend(); ++packet) reply_storage.owner->reset_buffer(packet->bytes);
        reply_storage.owner->reset_buffer(replies);
    });

    geometry_detail::NativeConstructorVector<unsigned char> seen(a.allocator<unsigned char>());
    collective(*inputs.comm,
        [&]
        {
            seen.resize(result->pieces.size(), 0);
            for (const auto& packet : returned.packets)
            {
                Reader r{packet.bytes};
                const auto slot = r.word(), gid = r.word(), first = r.word(), second = r.word();
                r.done();
                if (slot >= result->pieces.size() || seen[slot] || gid >= global_faces ||
                    packet.peer != face_authorities[slot] ||
                    first != result->pieces[slot].record->first_native_face ||
                    (result->pieces[slot].record->second_fragment &&
                        second != result->pieces[slot].record->second_native_face))
                    throw std::invalid_argument("Foreign or changed native numbered-face authority reply.");
                seen[slot] = 1;
                result->pieces[slot].face_gid = gid;
            }
            if (std::find(seen.begin(), seen.end(), 0) != seen.end())
                throw std::invalid_argument("Missing native numbered-face reply.");
        });
    returned = {};
    decltype(face_authorities)(a.allocator<int>()).swap(face_authorities);
    // Node requests borrow persistent canonical keys. Authorities sort scalar
    // packet views; no map node, key copy or point-loop wrapper is rebuilt.
    geometry_detail::NativeConstructorStorageOwner node_send_storage;
    collective(*inputs.comm, [&] { node_send_storage = geometry_detail::native_constructor_storage(a); });
    std::vector<OwnerPacket> node_send;
    collective(*inputs.comm,
        [&]
        {
            node_send_storage.owner->reserve(node_send, result->points.size());
            for (std::size_t i = 0; i < result->points.size(); ++i)
            {
                const auto& point = *result->points[i].record;
                Writer w;
                node_send_storage.owner->reserve(w.data, 5 * sizeof(std::uint64_t) + point.key.size());
                w.word(i);
                w.bytes(point.key);
                for (auto coordinate : PLIC::point_display<ArithmeticMode::Ordinary>(point.point))
                    w.word(std::bit_cast<std::uint64_t>(coordinate));
                node_send.push_back({authority(point.key, inputs.comm->getSize()), std::move(w.data)});
            }
        });
    auto node_incoming = exchange_adaptive_packets(node_send, inputs.comm, a);
    collective(*inputs.comm, [&]
    {
        for (auto packet = node_send.rbegin(); packet != node_send.rend(); ++packet) node_send_storage.owner->reset_buffer(packet->bytes);
        node_send_storage.owner->reset_buffer(node_send);
    });

    struct NodeRequest
    {
        std::span<const char> key;
        int peer = -1;
        std::uint64_t request = 0;
        PLIC::GeometryPoint coordinate{};
    };
    geometry_detail::NativeConstructorVector<NodeRequest> nodes(a.allocator<NodeRequest>());
    std::uint64_t local_nodes = 0;
    collective(*inputs.comm,
        [&]
        {
            nodes.reserve(node_incoming.packets.size());
            for (const auto& packet : node_incoming.packets)
            {
                Reader r{packet.bytes};
                NodeRequest node;
                node.peer = packet.peer;
                node.request = r.word();
                node.key = r.bytes_view();
                for (auto& x : node.coordinate)
                    x = std::bit_cast<double>(r.word());
                r.done();
                Reader key{node.key};
                const auto domain = key.bytes_view();
                const auto point_key = key.bytes_view();
                key.done();
                const auto p = PLIC::details::GeometryIdentityAccess::decode_key<Point>(point_key, remaining());
                if (domain.empty() || PLIC::point_display<ArithmeticMode::Ordinary>(p) != node.coordinate ||
                    authority(node.key, inputs.comm->getSize()) != inputs.comm->getRank())
                    throw std::invalid_argument("Invalid native canonical-node authority request.");
                nodes.push_back(node);
            }
            std::sort(nodes.begin(), nodes.end(),
                [](const auto& a, const auto& b)
                {
                    if (!std::ranges::equal(a.key, b.key))
                        return std::lexicographical_compare(a.key.begin(), a.key.end(), b.key.begin(), b.key.end());
                    return std::pair{a.peer, a.request} < std::pair{b.peer, b.request};
                });
            for (std::size_t i = 0; i < nodes.size();)
            {
                auto end = i + 1;
                while (end < nodes.size() && std::ranges::equal(nodes[i].key, nodes[end].key))
                    ++end;
                for (auto j = i + 1; j < end; ++j)
                    if (nodes[j].coordinate != nodes[i].coordinate ||
                        (nodes[j].peer == nodes[j - 1].peer && nodes[j].request == nodes[j - 1].request))
                        throw std::invalid_argument(
                            "Native canonical-node replicas disagree or duplicate a subscription.");
                ++local_nodes;
                i = end;
            }
        });
    const auto [node_begin, global_nodes] = entity_range(local_nodes, *inputs.comm);
    result->global_nodes = global_nodes;
    geometry_detail::NativeConstructorStorageOwner node_reply_storage;
    collective(*inputs.comm, [&] { node_reply_storage = geometry_detail::native_constructor_storage(a); });
    std::vector<OwnerPacket> node_replies;
    collective(*inputs.comm,
        [&]
        {
            node_reply_storage.owner->reserve(node_replies, nodes.size());
            std::uint64_t ordinal = 0;
            for (std::size_t i = 0; i < nodes.size();)
            {
                auto end = i + 1;
                while (end < nodes.size() && std::ranges::equal(nodes[i].key, nodes[end].key))
                    ++end;
                for (auto j = i; j < end; ++j)
                {
                    Writer w;
                    node_reply_storage.owner->reserve(w.data, 2 * sizeof(std::uint64_t));
                    w.word(nodes[j].request);
                    w.word(node_begin + ordinal);
                    node_replies.push_back({nodes[j].peer, std::move(w.data)});
                }
                ++ordinal;
                i = end;
            }
        });
    nodes.clear();
    decltype(nodes)(a.allocator<NodeRequest>()).swap(nodes);
    node_incoming = {};
    auto node_returned = exchange_adaptive_packets(node_replies, inputs.comm, a);
    collective(*inputs.comm, [&]
    {
        for (auto packet = node_replies.rbegin(); packet != node_replies.rend(); ++packet) node_reply_storage.owner->reset_buffer(packet->bytes);
        node_reply_storage.owner->reset_buffer(node_replies);
    });

    collective(*inputs.comm,
        [&]
        {
            seen.assign(result->points.size(), 0);
            for (const auto& packet : node_returned.packets)
            {
                Reader r{packet.bytes};
                const auto slot = r.word(), gid = r.word();
                r.done();
                if (slot >= result->points.size() || seen[slot] || gid >= global_nodes ||
                    packet.peer != authority(result->points[slot].record->key, inputs.comm->getSize()))
                    throw std::invalid_argument("Foreign or duplicate native canonical-node authority reply.");
                seen[slot] = 1;
                result->points[slot].gid = gid;
            }
            if (std::find(seen.begin(), seen.end(), 0) != seen.end())
                throw std::invalid_argument("Missing native canonical-node authority reply.");
            for (auto& node : result->loops)
            {
                if (node >= result->points.size())
                    throw std::invalid_argument("Native scalar point route is invalid.");
                node = result->points[node].gid;
            }
        });
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
