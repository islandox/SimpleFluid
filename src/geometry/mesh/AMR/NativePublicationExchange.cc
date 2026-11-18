#include "geometry/mesh/AMR/NativeAdaptiveKeyCodec.hh"
#include "geometry/mesh/AMR/NativeAdaptiveLeafCodec.hh"
#include "geometry/mesh/AMR/NativeDirectoryAccess.hh"
#include "geometry/mesh/AMR/NativePublicationExchange.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
void ticket(Writer& output, NativeSourceTicket value)
{
    output.word(static_cast<std::uint64_t>(value.issuer + 1));
    output.word(value.transaction);
    output.word(value.ordinal);
}
NativeSourceTicket ticket(Reader& input)
{
    const auto issuer = input.word();
    if (issuer > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid native source ticket issuer.");
    return {static_cast<int>(issuer) - 1, input.word(), input.word()};
}
struct Request
{
    const AdaptiveCellKey* key = nullptr;
    int owner = -1;
    std::uint64_t revision = 0, gid = 0;
    std::shared_ptr<const NativeCellSourceProof> previous, retained_source;
};
std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Native exchange extent overflows.");
    return a + b;
}
} // namespace
geometry_detail::NativeConstructorVector<NativeRemoteSource> NativePublicationExchange::fetch(
    NativePublicationSession& session, const DistributedAdaptiveHierarchy& hierarchy,
    std::span<const AdaptiveCellKey* const> keys, NativePublicationExchangeStatistics* statistics)
{
    const auto& input = session.inputs();
    const auto& admission = session.admission();
    geometry_detail::NativeConstructorStorageOwner scratch;
    collective(*input.comm,
        [&]
        {
            scratch = geometry_detail::native_constructor_storage(admission);
            for (const auto* key : keys)
                if (!key || input.layout->contains(*key))
                    throw std::invalid_argument("Remote publication request is null or owner-local.");
        });
    std::optional<NativeDirectoryLookup> owners;
    owners.emplace(NativePublicationSession::lookup_cells(hierarchy, keys, admission));
    geometry_detail::NativeConstructorVector<Request> requests(admission.allocator<Request>());
    std::vector<OwnerPacket> packets;
    collective(*input.comm,
        [&]
        {
            requests.reserve(keys.size());
            scratch.owner->reserve(packets, keys.size());
            for (std::size_t index = 0; index < keys.size(); ++index)
            {
                const auto& owner = owners->at(index);
                if (owner.owner < 0 || owner.owner >= input.comm->getSize() || owner.owner == input.comm->getRank())
                    throw std::invalid_argument("Native remote directory returned an invalid current owner.");
                Request request{keys[index], owner.owner, owner.revision, owner.address, {}, {}};
                if (const auto row = session.indexes_->find(*keys[index]); row && row->source)
                {
                    request.retained_source = row->source;
                    if (row->source->source().revision == owner.revision && row->source->ticket().issuer == owner.owner)
                        request.previous = row->source;
                }
                requests.push_back(request);
                const auto size = sum(encoded_adaptive_key_size(*keys[index]), 8 * 7);
                Writer output;
                scratch.owner->reserve(output.data, size);
                output.word(session.issuance_);
                output.word(index);
                NativeAdaptiveKeyCodec::write(output, *keys[index]);
                output.word(owner.revision);
                ticket(output, request.previous ? request.previous->ticket() : NativeSourceTicket{});
                packets.push_back({owner.owner, std::move(output.data)});
            }
        });
    owners.reset(); // Actual directory result keys/metadata die before request transport.
    auto incoming = exchange_adaptive_packets(packets, input.comm, admission);
    if (statistics)
        statistics->requests = incoming.stats;
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
            for (const auto& packet : incoming.packets)
            {
                Reader request{packet.bytes};
                const auto transaction = request.word(), index = request.word();
                auto key_storage = geometry_detail::native_constructor_storage(admission);
                const auto key = NativeAdaptiveKeyCodec::decode(request.bytes_view(), key_storage);
                const auto revision = request.word();
                const auto offered = ticket(request);
                request.done();
                const auto* leaf = input.layout->find(key);
                if (!leaf || leaf->revision != revision)
                    throw std::invalid_argument("Native request targets stale owner-local membership.");
                const auto proof = session.source(*leaf);
                const bool reuse = offered == proof->ticket();
                Writer response;
                if (reuse)
                    scratch.owner->reserve(response.data, 8 * 8);
                else
                {
                    const auto bytes = NativeAdaptiveLeafCodec::size(*leaf, admission);
                    scratch.owner->reserve(response.data, sum(bytes, 8 * 9));
                }
                response.word(transaction);
                response.word(index);
                response.word(revision);
                response.word(reuse ? 1 : 0);
                ticket(response, proof->ticket());
                if (!reuse)
                    NativeAdaptiveLeafCodec::write(response, *leaf, admission);
                responses.push_back({packet.peer, std::move(response.data)});
            }
        });
    incoming = {};
    auto replies = exchange_adaptive_packets(responses, input.comm, admission);
    if (statistics)
        statistics->replies = replies.stats;
    collective(*input.comm,
        [&]
        {
            for (auto packet = responses.rbegin(); packet != responses.rend(); ++packet)
                scratch.owner->reset_buffer(packet->bytes);
            scratch.owner->reset_buffer(responses);
        });
    geometry_detail::NativeConstructorVector<NativeRemoteSource> result(admission.allocator<NativeRemoteSource>());
    geometry_detail::NativeConstructorVector<unsigned char> received(admission.allocator<unsigned char>());
    collective(*input.comm,
        [&]
        {
            result.resize(keys.size());
            received.resize(keys.size(), 0);
            for (const auto& packet : replies.packets)
            {
                Reader reply{packet.bytes};
                const auto transaction = reply.word(), index = reply.word(), revision = reply.word(),
                           reuse = reply.word();
                const auto issued = ticket(reply);
                if (transaction != session.issuance_ || index >= requests.size() || received[index] || reuse > 1)
                    throw std::invalid_argument("Foreign, repeated or stale native confirmation reply.");
                const auto& request = requests[index];
                if (packet.peer != request.owner || issued.issuer != request.owner || !issued.transaction ||
                    revision != request.revision)
                    throw std::invalid_argument("Native reply is not from the actual current source owner.");
                std::shared_ptr<const NativeCellSourceProof> source;
                if (reuse)
                {
                    // Wire bytes grant no proof. The receiver must already retain
                    // this exact privately verified source and its authentic owner
                    // just confirmed that same ticket in this fetch transaction.
                    source = request.previous;
                    if (!source || source->ticket() != issued || source->source().key != *request.key ||
                        source->source().revision != revision || !source->allocation_ ||
                        !source->allocation_->authentic_owner(source->allocation_) ||
                        !source->allocation_->matches(source) || source->controls_ != geometry_construction_controls())
                        throw std::invalid_argument("Native confirmation has no exact retained receiver source proof.");
                    if (statistics)
                        ++statistics->confirmed_reuse;
                }
                else
                {
                    auto decoded = NativeAdaptiveLeafCodec::decode(reply.bytes_view(), admission,
                        request.retained_source ? &request.retained_source->source() : nullptr);
                    const auto& leaf = decoded.value;
                    if (leaf.key != *request.key || leaf.revision != revision)
                        throw std::invalid_argument("Native full reply changes the current directory binding.");
                    source = session.source_impl(leaf, issued);
                    if (statistics)
                        ++statistics->full_replies;
                }
                reply.done();
                received[index] = 1;
                result[index] = {std::move(source), request.gid, request.owner};
            }
            if (std::find(received.begin(), received.end(), 0) != received.end())
                throw std::invalid_argument("Native current-owner confirmation is incomplete.");
        });
    collective(*input.comm, [&] { session.recheck(hierarchy); });
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
