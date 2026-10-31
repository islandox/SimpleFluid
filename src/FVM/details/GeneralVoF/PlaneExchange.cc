/** @file PlaneExchange.cc
 * @brief Full and sparse authenticated owner interface-plane imports.
 */
#include "FVM/details/GeneralVoF/OwnerPlaneRequests.hh"
#include "FVM/details/GeneralVoF/Reconstruction.hh"
#include "FVM/details/GeneralVoF/WorkCapture.hh"
#include "geometry/swept_volume/details/FiniteGeometryAccess.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
template<ArithmeticMode Mode> std::map<Key, PLIC::PolyhedralInterface> import_planes(const GeneralVoF::Data& data)
{
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    std::map<Key, PLIC::PolyhedralInterface> result;
    wire::collective(*comm,
        [&]
        {
            current(data);
            result = reconstruct_local<Mode>(data);
        });
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
                if (cell.owner != comm->getRank())
                {
                    auto bytes = AMR::encode_adaptive_key(key);
                    budget.packet(bytes.size() + 16);
                    wire::Writer w;
                    w.bytes(bytes);
                    w.word(cell.revision);
                    requests.push_back({cell.owner, std::move(w.data)});
                }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            for (const auto& packet : incoming.packets)
            {
                wire::Reader r{packet.bytes};
                auto bytes = r.bytes();
                auto key = AMR::decode_adaptive_key(bytes);
                auto revision = r.word();
                r.done();
                const auto& cell = data.cells.at(key);
                if (cell.owner != comm->getRank() || cell.revision != revision)
                    throw std::invalid_argument("General VoF stale plane request.");
                wire::Writer w;
                w.bytes(bytes);
                w.word(revision);
                plane(w, result.at(key));
                budget.packet(w.data.size());
                replies.push_back({packet.peer, std::move(w.data)});
            }
        });
    auto received = exchange_owner_packets(replies, comm, limit);
    wire::collective(*comm,
        [&]
        {
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                auto key = AMR::decode_adaptive_key(r.bytes());
                auto revision = r.word();
                const auto& cell = data.cells.at(key);
                if (cell.owner != packet.peer || cell.owner == comm->getRank() || cell.revision != revision ||
                    !seen.insert(key).second)
                    throw std::invalid_argument("General VoF unauthenticated plane reply.");
                auto fitted =
                    plane(r, data.options.arithmetic_mode == ArithmeticMode::Exact ? std::optional<ArithmeticMode>{}
                                                                                   : data.options.arithmetic_mode);
                if constexpr (Mode != ArithmeticMode::Exact)
                {
                    if (!fitted.finite_geometry)
                        throw std::invalid_argument("Finite VoF imported an exact interface authority.");
                    if (!fitted.pure_gas && !fitted.pure_carrier)
                    {
                        const auto amounts = SimpleFluid::geometry::swept_volume::runtime_phases<Mode>(
                            PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->phases(fitted.gas));
                        const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                        fitted.finite_target_volume_difference =
                            fp::Arithmetic<Mode>::subtract(gas ? amounts.gas : amounts.carrier,
                                {gas ? cell.phase.gas_volume : cell.phase.carrier_volume, 0});
                        fitted.target_volume_residual =
                            static_cast<double>(fitted.finite_target_volume_difference.value);
                    }
                }
                else if (!fitted.pure_gas && !fitted.pure_carrier)
                {
                    const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                    const auto measures = SimpleFluid::geometry::swept_volume::ConvexFragment<
                        SimpleFluid::ArithmeticMode::Exact>::from_geometry(*cell.phase.geometry)
                                              ->phases(fitted.gas);
                    const auto residual = (gas ? measures.gas : measures.carrier) -
                                          Measure::from_value(gas ? cell.phase.gas_volume : cell.phase.carrier_volume);
                    fitted.target_volume_difference = residual.enclosure();
                    fitted.target_volume_residual = double(fitted.target_volume_difference.value);
                }
                result.emplace(key, std::move(fitted));
                r.done();
            }
            if (seen.size() != requests.size())
                throw std::invalid_argument("General VoF incomplete plane replies.");
        });
    return result;
}

std::map<Key, PLIC::PolyhedralInterface> import_requested_planes(const GeneralVoF::Data& data,
    const std::set<Key>& requested, const PLIC::GeometryPacketContext& context, geometry::swept_volume::Work& work,
    GeneralVoFOwnerPlaneRequestDiagnostics& diagnostics, GeneralVoFWorkSample* balance_work)
{
    constexpr auto Mode = ArithmeticMode::Ordinary;
    const auto comm = data.hierarchy->communicator();
    const auto limit = data.options.byte_budget;
    PlaneImportStorage storage(work, diagnostics);
    const auto reconstruct = [&](const Key& key)
    {
        auto* row = balance_cell(balance_work, key);
        auto fitted = reconstruct_observed_ordinary_cell(data, key, &diagnostics, row);
        if (row)
        {
            row->mixed = !fitted.pure_gas && !fitted.pure_carrier;
            row->fit_iterations = fitted.fit_iterations;
            row->request_payload_bytes = owner_plane_key_wire_bytes(key) + 48;
            // Same codec as a real reply; this calibration is outside reconstruction timing.
            storage.charge(216);
            {
                wire::Writer writer;
                writer.data.reserve(216);
                if (writer.data.capacity() > 216)
                    storage.charge(writer.data.capacity() - 216);
                plane(writer, fitted);
                if (writer.data.size() > 216)
                    throw std::invalid_argument("General VoF plane exceeds its capture envelope.");
                row->reply_payload_bytes = row->request_payload_bytes + writer.data.size();
                if (writer.data.capacity() > 216)
                    storage.release(writer.data.capacity() - 216);
            }
            storage.release(216);
        }
        return fitted;
    };
    std::map<Key, PLIC::PolyhedralInterface> result;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (context.publication[0] != data.control ||
                context.publication[1] != data.hierarchy->topology_generation() ||
                context.publication[2] != data.hierarchy->partition_generation())
                throw std::invalid_argument("General VoF sparse plane source context is stale.");
            for (const auto& key : requested)
                if (data.cells.at(key).owner == comm->getRank())
                {
                    ++diagnostics.local_owned_requests;
                    result.emplace(key, reconstruct(key));
                }
        });
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            std::size_t count = 0;
            for (const auto& key : requested)
                count += data.cells.at(key).owner != comm->getRank();
            if (count > std::numeric_limits<std::size_t>::max() / sizeof(OwnerPacket))
                throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "Sparse plane requests overflow.");
            storage.charge(count * sizeof(OwnerPacket));
            requests.reserve(count);
            if (requests.capacity() > count)
                storage.charge((requests.capacity() - count) * sizeof(OwnerPacket));
            for (const auto& key : requested)
                if (const auto& cell = data.cells.at(key); cell.owner != comm->getRank())
                {
                    const auto wire_bytes = owner_plane_key_wire_bytes(key);
                    if (wire_bytes > (std::numeric_limits<std::size_t>::max() - 48) / 3)
                        throw PLIC::GeometryError(
                            PLIC::GeometryErrorCode::ResourceLimit, "Sparse plane key overflows.");
                    const auto scratch = 3 * wire_bytes;
                    storage.charge(scratch);
                    {
                        auto bytes = AMR::encode_adaptive_key(key);
                        if (bytes.size() != wire_bytes)
                            throw std::invalid_argument("Sparse plane key differs from its wire envelope.");
                        const auto payload = wire_bytes + 48;
                        budget.packet(payload);
                        storage.charge(payload);
                        wire::Writer w;
                        w.data.reserve(payload);
                        if (w.data.capacity() > payload)
                            storage.charge(w.data.capacity() - payload);
                        w.bytes(bytes);
                        w.word(cell.revision);
                        for (auto word : context.publication)
                            w.word(word);
                        requests.push_back({cell.owner, std::move(w.data)});
                    }
                    storage.release(scratch);
                }
        });
    const auto remote_requests = requests.size();
    auto incoming = exchange_owner_packets(requests, comm, storage.remaining_bytes());
    wire::collective(*comm,
        [&]
        {
            storage.observe_exchange_peak(incoming.stats.peak_scratch_bytes);
            const auto request_bytes = owner_plane_packet_storage_bytes(requests) +
                                       (requests.capacity() - requests.size()) * sizeof(OwnerPacket);
            std::vector<OwnerPacket>().swap(requests);
            storage.release(request_bytes);
            storage.charge(owner_plane_packet_storage_bytes(incoming.packets) +
                           (incoming.packets.capacity() - incoming.packets.size()) * sizeof(OwnerPacket));
        });
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            replies = owner_plane_replies(
                result, incoming.packets, context, comm->getRank(), comm->getSize(), storage, diagnostics,
                [&](const Key& key, std::uint64_t revision, int)
                {
                    const auto& cell = data.cells.at(key);
                    if (cell.owner != comm->getRank() || cell.revision != revision)
                        throw std::invalid_argument("General VoF stale plane request.");
                },
                [&](const Key& key) { return reconstruct(key); },
                [&](wire::Writer& writer, const PLIC::PolyhedralInterface& fitted)
                {
                    plane(writer, fitted);
                    budget.packet(writer.data.size());
                });
            const auto incoming_bytes = owner_plane_packet_storage_bytes(incoming.packets) +
                                        (incoming.packets.capacity() - incoming.packets.size()) * sizeof(OwnerPacket);
            std::vector<OwnerPacket>().swap(incoming.packets);
            storage.release(incoming_bytes);
        });
    auto received = exchange_owner_packets(replies, comm, storage.remaining_bytes());
    wire::collective(*comm,
        [&]
        {
            storage.observe_exchange_peak(received.stats.peak_scratch_bytes);
            // The peer-only map is already gone. Keep replies charged until
            // exchange and their destruction; account the received owners too.
            storage.charge(owner_plane_packet_storage_bytes(received.packets) +
                           (received.packets.capacity() - received.packets.size()) * sizeof(OwnerPacket));
            std::size_t key_scratch = 0;
            for (const auto& packet : received.packets)
            {
                if (packet.bytes.size() > (std::numeric_limits<std::size_t>::max() - sizeof(Key)) / 3)
                    throw PLIC::GeometryError(
                        PLIC::GeometryErrorCode::ResourceLimit, "Sparse plane reply key overflows.");
                key_scratch = std::max(key_scratch, sizeof(Key) + 3 * packet.bytes.size());
            }
            storage.charge(key_scratch);
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader r{packet.bytes};
                const auto size = r.word();
                if (size > r.data.size() - r.offset)
                    throw std::invalid_argument("Invalid sparse plane reply key length.");
                auto key = AMR::decode_adaptive_key(r.data.subspan(r.offset, static_cast<std::size_t>(size)));
                r.offset += static_cast<std::size_t>(size);
                auto revision = r.word();
                for (auto word : context.publication)
                    if (r.word() != word)
                        throw std::invalid_argument("General VoF sparse plane context mismatch.");
                const auto& cell = data.cells.at(key);
                if (cell.owner != packet.peer || cell.owner == comm->getRank() || cell.revision != revision ||
                    !requested.contains(key) || seen.contains(key))
                    throw std::invalid_argument("General VoF unauthenticated plane reply.");
                storage.charge(
                    sizeof(decltype(seen)::value_type) + 4 * sizeof(void*) + owner_plane_key_storage_bytes(key));
                seen.insert(key);
                auto fitted =
                    plane(r, data.options.arithmetic_mode == ArithmeticMode::Exact ? std::optional<ArithmeticMode>{}
                                                                                   : data.options.arithmetic_mode);
                GeneralVoFWorkReceiver* receiver = nullptr;
                if (balance_work)
                    for (auto& row : balance_work->receivers)
                        if (row.donor.key == key)
                        {
                            receiver = &row;
                            row.mixed = !fitted.pure_gas && !fitted.pure_carrier;
                            row.request_payload_bytes = owner_plane_key_wire_bytes(key) + 48;
                            row.reply_payload_bytes = packet.bytes.size();
                            break;
                        }
                if constexpr (Mode != ArithmeticMode::Exact)
                {
                    if (!fitted.finite_geometry)
                        throw std::invalid_argument("Finite VoF imported an exact interface authority.");
                    if (!fitted.pure_gas && !fitted.pure_carrier)
                    {
                        WorkObservation receiver_observation(receiver ? &receiver->remeasurement : nullptr);
                        const auto amounts = SimpleFluid::geometry::swept_volume::runtime_phases<Mode>(
                            PLIC::ConvexCellGeometry<Mode>::from_dynamic(cell.finite_geometry)->phases(fitted.gas));
                        const bool gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
                        fitted.finite_target_volume_difference =
                            fp::Arithmetic<Mode>::subtract(gas ? amounts.gas : amounts.carrier,
                                {gas ? cell.phase.gas_volume : cell.phase.carrier_volume, 0});
                        fitted.target_volume_residual =
                            static_cast<double>(fitted.finite_target_volume_difference.value);
                    }
                }
                result.emplace(key, std::move(fitted));
                r.done();
            }
            storage.release(key_scratch);
            if (seen.size() != remote_requests)
                throw std::invalid_argument("General VoF incomplete plane replies.");
        });
    if (balance_work)
        balance_work->fully_deduplicated = diagnostics.cache_admission_denials == 0;
    return result;
}

std::map<Key, PLIC::PolyhedralInterface> import_planes(const GeneralVoF::Data& data)
{
    if (data.options.arithmetic_mode == ArithmeticMode::Exact)
        return import_planes<ArithmeticMode::Exact>(data);
    if (data.options.arithmetic_mode == ArithmeticMode::Ordinary)
        return import_planes<ArithmeticMode::Ordinary>(data);
    return import_planes<ArithmeticMode::BoundedCertification>(data);
}
template std::map<Key, PLIC::PolyhedralInterface> import_planes<ArithmeticMode::Ordinary>(const GeneralVoF::Data&);
template std::map<Key, PLIC::PolyhedralInterface> import_planes<ArithmeticMode::BoundedCertification>(
    const GeneralVoF::Data&);
template std::map<Key, PLIC::PolyhedralInterface> import_planes<ArithmeticMode::Exact>(const GeneralVoF::Data&);
} // namespace SimpleFluid::VoF::general_vof_detail
