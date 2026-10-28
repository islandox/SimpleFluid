namespace SimpleFluid::VoF::general_vof_detail
{
template<class Validate, class Reconstruct, class Serialize>
std::vector<OwnerPacket> owner_plane_replies(const RequestedPlaneMap& local_results,
    std::span<const OwnerPacket> incoming, const PLIC::GeometryPacketContext& context, int rank, int ranks,
    PlaneImportStorage& storage, GeneralVoFOwnerPlaneRequestDiagnostics& diagnostics, Validate&& validate,
    Reconstruct&& reconstruct, Serialize&& serialize)
{
    namespace wire = Meshes::AMR::adaptive_detail;
    const auto mandatory = owner_plane_request_workspace_bytes(incoming);
    storage.charge(mandatory);
    std::vector<OwnerPacket> replies;
    replies.reserve(incoming.size());
    if (replies.capacity() > incoming.size())
        storage.charge((replies.capacity() - incoming.size()) * sizeof(OwnerPacket));
    // Resolve all mandatory capacity before the first optional cache entry.
    for (const auto& packet : incoming)
    {
        OwnerPacket reply;
        reply.bytes.reserve(packet.bytes.size() + 216);
        if (reply.bytes.capacity() > packet.bytes.size() + 216)
            storage.charge(reply.bytes.capacity() - packet.bytes.size() - 216);
        replies.push_back(std::move(reply));
    }
    std::size_t cache_bytes = 0, key_scratch = 0;
    {
        RequestedPlaneMap peer_cache;
        diagnostics.incoming_peer_requests += incoming.size();
        for (std::size_t index = 0; index < incoming.size(); ++index)
        {
            const auto& packet = incoming[index];
            key_scratch = std::max(key_scratch, sizeof(Meshes::AMR::AdaptiveCellKey) + 3 * packet.bytes.size());
            wire::Reader reader{packet.bytes};
            const auto count = reader.word();
            if (count > reader.data.size() - reader.offset)
                throw std::invalid_argument("Invalid sparse plane key length.");
            const auto bytes = reader.data.subspan(reader.offset, static_cast<std::size_t>(count));
            reader.offset += static_cast<std::size_t>(count);
            auto key = Meshes::AMR::decode_adaptive_key(bytes);
            const auto revision = reader.word();
            for (auto word : context.publication)
                if (reader.word() != word)
                    throw std::invalid_argument("General VoF sparse plane context mismatch.");
            reader.done();
            if (rank < 0 || rank >= ranks || packet.peer < 0 || packet.peer >= ranks)
                throw std::invalid_argument("General VoF sparse plane peer is out of range.");
            validate(key, revision, packet.peer);

            wire::Writer writer{std::move(replies[index].bytes)};
            writer.bytes(bytes);
            writer.word(revision);
            for (auto word : context.publication)
                writer.word(word);
            const auto local = local_results.find(key);
            if (local != local_results.end())
            {
                ++diagnostics.local_result_hits;
                serialize(writer, local->second);
            }
            else if (const auto cached = peer_cache.find(key); cached != peer_cache.end())
            {
                ++diagnostics.peer_cache_hits;
                serialize(writer, cached->second);
            }
            else
            {
                auto fitted = reconstruct(key);
                const auto entry_bytes = owner_plane_cache_entry_bytes(key);
                if (storage.try_charge(entry_bytes))
                {
                    cache_bytes += entry_bytes;
                    ++diagnostics.peer_cache_admissions;
                    diagnostics.peer_cache_peak_bytes = std::max(diagnostics.peer_cache_peak_bytes, cache_bytes);
                    const auto inserted = peer_cache.emplace(std::move(key), std::move(fitted));
                    serialize(writer, inserted.first->second);
                }
                else
                {
                    ++diagnostics.cache_admission_denials;
                    serialize(writer, fitted);
                }
            }
            if (writer.data.size() > packet.bytes.size() + 216)
                throw std::invalid_argument("General VoF owner plane exceeds its fixed wire envelope.");
            replies[index] = {packet.peer, std::move(writer.data)};
        }
    } // Destroy interfaces and adaptive keys before releasing their reservation.
    storage.release(cache_bytes + key_scratch);
    return replies;
}
} // namespace SimpleFluid::VoF::general_vof_detail
