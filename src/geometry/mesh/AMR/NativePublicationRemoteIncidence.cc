#include "geometry/mesh/AMR/NativeAdaptiveTileCodec.hh"
#include "geometry/mesh/AMR/NativeImmutableSequenceAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/AMR/AdaptiveBoundaryProof.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveAuthority.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include "geometry/mesh/AMR/NativePublicationExchange.hh"
#include "geometry/mesh/AMR/NativePublicationNumbering.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Face = PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>;
void source_ticket(Writer& writer, NativeSourceTicket value)
{
    writer.word(static_cast<std::uint64_t>(value.issuer + 1));
    writer.word(value.transaction);
    writer.word(value.ordinal);
}
NativeSourceTicket source_ticket(Reader& reader)
{
    const auto rank = reader.word();
    if (rank > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid source-ticket rank.");
    return {static_cast<int>(rank) - 1, reader.word(), reader.word()};
}
void incidence_ticket(Writer& writer, NativeIncidenceTicket value)
{
    writer.word(static_cast<std::uint64_t>(value.issuer + 1));
    writer.word(value.transaction);
    writer.word(value.ordinal);
}
NativeIncidenceTicket incidence_ticket(Reader& reader)
{
    const auto rank = reader.word();
    if (rank > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid incidence-ticket rank.");
    return {static_cast<int>(rank) - 1, reader.word(), reader.word()};
}
void checked_add(std::size_t& count, std::size_t value)
{
    if (value > std::numeric_limits<std::size_t>::max() - count)
        throw std::invalid_argument("Remote incidence extent overflows.");
    count += value;
}
void numbers(Writer& writer, const NativeIncidenceNumericPiece& piece)
{
    writer.word(piece.face_gid);
    writer.word(piece.first_gid);
    writer.word(piece.second_gid);
    writer.word(piece.first_nodes.size());
    for (const auto node : piece.first_nodes)
        writer.word(node);
    writer.word(piece.second_nodes.size());
    for (const auto node : piece.second_nodes)
        writer.word(node);
}
NativeRemoteNumericPiece numbers(Reader& reader, const geometry_detail::NativeConstructorStorageOwner& storage)
{
    NativeRemoteNumericPiece result;
    result.face_gid = reader.word();
    result.first_gid = reader.word();
    result.second_gid = reader.word();
    const auto first = reader.word();
    if (first > (reader.data.size() - reader.offset) / 8)
        throw std::invalid_argument("Remote numeric first-node extent exceeds reply.");
    storage.owner->reserve(result.first_nodes, first);
    for (std::size_t i = 0; i < first; ++i)
        result.first_nodes.push_back(reader.word());
    const auto second = reader.word();
    if (second > (reader.data.size() - reader.offset) / 8)
        throw std::invalid_argument("Remote numeric second-node extent exceeds reply.");
    storage.owner->reserve(result.second_nodes, second);
    for (std::size_t i = 0; i < second; ++i)
        result.second_nodes.push_back(reader.word());
    return result;
}
AdaptiveTile transfer(const CanonicalSubfaceView& view, const NativeIncidenceNumericPiece& numeric, Budget& budget,
    IdentityBudget& identities, const geometry_detail::NativeConstructorStorageOwner& storage)
{
    AdaptiveTile result;
    const auto& admission = storage.owner->admission();
    result.domain = NativeImmutableSequenceAccess::snapshot<char>(view.domain(), admission);
    result.surface = NativeImmutableSequenceAccess::snapshot<char>(view.support(), admission);
    storage.owner->copy(result.identity, view.identity());
    const auto physical = view.materialize_native(budget, identities, admission);
    if (!std::ranges::equal(physical.identity, result.identity))
        throw std::invalid_argument("Captured original subface key differs from full materializer.");
    result.geometry = physical.geometry;
    const auto first = view.first();
    const auto second = view.second();
    result.first = NativeAdaptiveKeyCodec::copy(first.key(), storage);
    result.first_owner = first.owner();
    if (const auto native = first.native_face())
        result.first_native_face = *native;
    if (second)
    {
        result.second = NativeAdaptiveKeyCodec::copy(second->key(), storage);
        result.second_owner = second->owner();
        if (const auto native = second->native_face())
            result.second_native_face = *native;
    }
    result.exterior = !second;
    storage.owner->string(result.boundary, view.boundary());
    result.second_image_identity = view.second_image();
    result.second_image = PLIC::displacement_display<ArithmeticMode::Ordinary>(result.second_image_identity);
    result.face_gid = numeric.face_gid;
    result.node_gids = NativeImmutableSequenceAccess::snapshot<std::uint64_t>(numeric.first_nodes, admission);
    result.second_node_gids = NativeImmutableSequenceAccess::snapshot<std::uint64_t>(numeric.second_nodes, admission);
    return result;
}
} // namespace
geometry_detail::NativeConstructorVector<NativeRemoteIncidence> NativePublicationExchange::fetch_incidence(
    NativePublicationSession& session, const DistributedAdaptiveHierarchy& hierarchy,
    std::span<const NativeRemoteSource> sources, const NativePublicationNumbering& numbering,
    NativePublicationExchangeStatistics* statistics)
{
    const auto& input = session.inputs();
    const auto& admission = session.admission();
    geometry_detail::NativeConstructorStorageOwner scratch;
    std::vector<OwnerPacket> packets;
    geometry_detail::NativeConstructorVector<std::shared_ptr<const NativeCellIncidenceProof>> previous(
        admission.allocator<std::shared_ptr<const NativeCellIncidenceProof>>());
    collective(*input.comm,
        [&]
        {
            scratch = geometry_detail::native_constructor_storage(admission);
            scratch.owner->reserve(packets, sources.size());
            previous.resize(sources.size());
            for (std::size_t index = 0; index < sources.size(); ++index)
            {
                const auto& source = sources[index];
                if (!source.source || source.owner != source.source->ticket().issuer ||
                    source.owner == input.comm->getRank())
                    throw std::invalid_argument("Full remote incidence lacks a current-owner source proof.");
                if (const auto row = session.indexes_->find(source.source->source().key);
                    row && row->source == source.source && row->incidence && row->incidence->remote_)
                    previous[index] = row->incidence;
                Writer request;
                scratch.owner->reserve(request.data, encoded_adaptive_key_size(source.source->source().key) + 10 * 8);
                request.word(session.issuance_);
                request.word(index);
                NativeAdaptiveKeyCodec::write(request, source.source->source().key);
                request.word(source.source->source().revision);
                source_ticket(request, source.source->ticket());
                incidence_ticket(request, previous[index] ? previous[index]->ticket() : NativeIncidenceTicket{});
                packets.push_back({source.owner, std::move(request.data)});
            }
        });
    auto incoming = exchange_adaptive_packets(packets, input.comm, admission);
    collective(*input.comm,
        [&]
        {
            for (auto packet = packets.rbegin(); packet != packets.rend(); ++packet)
                scratch.owner->reset_buffer(packet->bytes);
            scratch.owner->reset_buffer(packets);
        });
    std::vector<OwnerPacket> responses;
    collective(*input.comm,
        [&]
        {
            scratch.owner->reserve(responses, incoming.packets.size());
            geometry_detail::NativeConstructorVector<std::size_t> ordered(admission.allocator<std::size_t>());
            if (!incoming.packets.empty())
            {
                ordered.resize(numbering.local_pieces().size());
                numbering.index_local_incidence(ordered);
            }
            using Route = NativePublicationNumbering::LocalIncidenceRoute;
            geometry_detail::NativeConstructorVector<Route> selected(admission.allocator<Route>());
            for (const auto& packet : incoming.packets)
            {
                Reader request{packet.bytes};
                const auto transaction = request.word(), ordinal = request.word();
                auto key_storage = geometry_detail::native_constructor_storage(admission);
                const auto key = NativeAdaptiveKeyCodec::decode(request.bytes_view(), key_storage);
                const auto revision = request.word();
                const auto offered_source = source_ticket(request);
                const auto offered = incidence_ticket(request);
                request.done();
                const auto* leaf = input.layout->find(key);
                if (!leaf || leaf->revision != revision)
                    throw std::invalid_argument("Remote incidence request targets stale owner membership.");
                const auto source = session.source(*leaf);
                const auto row = session.indexes_->find(key);
                if (source->ticket() != offered_source || !row || !row->incidence ||
                    row->incidence->source_ != source || row->incidence->remote_)
                    throw std::invalid_argument(
                        "Current owner has no exact completed local incidence for the offered source.");
                const auto incidence = row->incidence;
                const bool reuse = incidence->ticket() == offered;
                numbering.select_local_incidence(*incidence, ordered, selected);
                const auto count = selected.size();
                std::size_t scalar_bytes = 0;
                numbering.visit_local_incidence(selected,
                    [&](const CanonicalSubfaceView&, const NativeIncidenceNumericPiece& numeric)
                    {
                        checked_add(scalar_bytes, 5 * 8);
                        checked_add(scalar_bytes, numeric.first_nodes.size() * 8);
                        checked_add(scalar_bytes, numeric.second_nodes.size() * 8);
                    });
                // Numeric routing travels on every confirmation. Rich physical
                // packets are created only after an incidence-ticket miss.
                geometry_detail::NativeConstructorStorageOwner physical_storage;
                std::vector<AdaptiveTile> physical;
                if (!reuse)
                {
                    physical_storage = geometry_detail::native_constructor_storage(admission);
                    physical_storage.owner->reserve(physical, count);
                    Budget geometry_budget{input.byte_budget};
                    IdentityBudget identities;
                    identities.bind_native(admission, input.byte_budget);
                    numbering.visit_local_incidence(selected,
                        [&](const CanonicalSubfaceView& view, const NativeIncidenceNumericPiece& numeric)
                        {
                            physical.push_back(transfer(view, numeric, geometry_budget, identities, physical_storage));
                        });
                }
                std::size_t extent = 11 * 8;
                checked_add(extent, scalar_bytes);
                for (const auto& tile : physical)
                {
                    checked_add(extent, 8);
                    checked_add(extent, NativeAdaptiveTileCodec::size(tile, admission));
                }
                Writer reply;
                scratch.owner->reserve(reply.data, extent);
                reply.word(transaction);
                reply.word(ordinal);
                reply.word(revision);
                reply.word(reuse ? 1 : 0);
                source_ticket(reply, source->ticket());
                incidence_ticket(reply, incidence->ticket());
                reply.word(count);
                std::size_t index = 0;
                numbering.visit_local_incidence(selected,
                    [&](const CanonicalSubfaceView&, const NativeIncidenceNumericPiece& numeric)
                    {
                        numbers(reply, numeric);
                        if (!reuse)
                            NativeAdaptiveTileCodec::write(reply, physical.at(index), admission);
                        ++index;
                    });
                responses.push_back({packet.peer, std::move(reply.data)});
            }
        });
    incoming = {};
    auto replies = exchange_adaptive_packets(responses, input.comm, admission);
    collective(*input.comm,
        [&]
        {
            for (auto packet = responses.rbegin(); packet != responses.rend(); ++packet)
                scratch.owner->reset_buffer(packet->bytes);
            scratch.owner->reset_buffer(responses);
        });
    geometry_detail::NativeConstructorVector<NativeRemoteIncidence> result(
        admission.allocator<NativeRemoteIncidence>());
    geometry_detail::NativeConstructorVector<unsigned char> received(admission.allocator<unsigned char>());
    collective(*input.comm,
        [&]
        {
            result.resize(sources.size());
            received.resize(sources.size(), 0);
            for (const auto& packet : replies.packets)
            {
                Reader reply{packet.bytes};
                const auto transaction = reply.word(), ordinal = reply.word(), revision = reply.word(),
                           reuse = reply.word();
                const auto issued_source = source_ticket(reply);
                const auto issued = incidence_ticket(reply);
                const auto count = reply.word();
                if (transaction != session.issuance_ || ordinal >= sources.size() || received[ordinal] || reuse > 1 ||
                    count > input.byte_budget / sizeof(NativeRemoteNumericPiece))
                    throw std::invalid_argument("Invalid or repeated remote incidence reply.");
                const auto& source = sources[ordinal];
                if (packet.peer != source.owner || issued.issuer != source.owner || !issued.transaction ||
                    issued_source != source.source->ticket() || revision != source.source->source().revision)
                    throw std::invalid_argument("Remote incidence lacks actual current-owner confirmation.");
                auto& output = result[ordinal];
                output.storage = geometry_detail::native_constructor_storage(admission);
                output.storage.owner->reserve(output.numeric, count);
                std::shared_ptr<NativeRemoteIncidenceRecord> record;
                std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> record_allocation;
                if (!reuse)
                {
                    auto frozen = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeRemoteIncidenceRecord>(
                        admission, NativeRemoteIncidenceRecord::ConstructionKey{});
                    record = std::move(frozen.owner);
                    record_allocation = std::move(frozen.envelope);
                    record->metadata_ = geometry_detail::native_constructor_storage(admission);
                    record->metadata_.owner->reserve(record->tiles_, count);
                }
                for (std::size_t piece = 0; piece < count; ++piece)
                {
                    output.numeric.push_back(numbers(reply, output.storage));
                    if (!reuse)
                    {
                        auto tile =
                            NativeAdaptiveTileCodec::decode(reply.bytes_view(), input.byte_budget, record->metadata_);
                        if (tile.first != source.source->source().key &&
                            (!tile.second || *tile.second != source.source->source().key))
                            throw std::invalid_argument("Remote incidence includes a nonincident source key.");
                        record->tiles_.push_back(std::move(tile));
                    }
                }
                if (reuse)
                {
                    output.proof = previous[ordinal];
                    if (!output.proof || output.proof->ticket() != issued || output.proof->source_ != source.source ||
                        !output.proof->remote_ || output.proof->remote_tiles().size() != count ||
                        !output.proof->allocation_ ||
                        !output.proof->allocation_->authentic_owner(output.proof->allocation_) ||
                        !output.proof->allocation_->matches(output.proof))
                        throw std::invalid_argument("Confirmed remote incidence has no exact retained proof.");
                    if (statistics)
                        ++statistics->incidence_reuse;
                }
                else
                {
                    // A private current-owner envelope is separate from the
                    // public decoder. Verify complete source face partitions on
                    // the new rich reply before issuing a reusable capability.
                    const auto outward_storage = geometry_detail::native_constructor_storage(admission);
                    std::vector<Face> outward;
                    const auto& key = source.source->source().key;
                    std::size_t outward_count = 0;
                    for (const auto& tile : record->tiles_)
                    {
                        checked_add(outward_count, tile.first == key);
                        checked_add(outward_count, tile.second && *tile.second == key);
                    }
                    outward_storage.owner->reserve(outward, outward_count);
                    for (const auto& tile : record->tiles_)
                    {
                        if (!tile.geometry)
                            throw std::invalid_argument("New remote incidence tile has no physical geometry.");
                        if (tile.first == key)
                            outward.push_back(tile.geometry);
                        if (tile.second && *tile.second == key)
                        {
                            auto face = PLIC::details::NativePlanarOwnerAccess::reversed(*tile.geometry, admission);
                            if (!tile.second_image_identity.is_zero())
                                face = PLIC::details::NativePlanarOwnerAccess::translated(*face,
                                    PLIC::displacement_difference<ArithmeticMode::Ordinary>(
                                        {}, tile.second_image_identity),
                                    admission);
                            outward.push_back(std::move(face));
                        }
                    }
                    session.verify_incidence(*source.source, outward);
                    session.compact_remote(*source.source, *record);
                    record->allocation_ = std::move(record_allocation);
                    auto proof = geometry_detail::NativeAllocationEnvelopeAccess::make<NativeCellIncidenceProof>(
                        admission, NativeCellIncidenceProof::ConstructionKey{});
                    proof.owner->source_ = source.source;
                    proof.owner->remote_ = std::move(record);
                    proof.owner->ticket_ = issued;
                    proof.owner->allocation_ = proof.envelope;
                    output.proof = std::move(proof.owner);
                    session.replace({source.source, output.proof});
                    if (statistics)
                        ++statistics->incidence_full;
                }
                // The cached rich record's old GIDs are deliberately never used.
                // Every numeric row above belongs to this current reply transaction.
                reply.done();
                received[ordinal] = 1;
            }
            if (std::find(received.begin(), received.end(), 0) != received.end())
                throw std::invalid_argument("Remote full-incidence replies are incomplete.");
        });
    collective(*input.comm, [&] { session.recheck(hierarchy); });
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
