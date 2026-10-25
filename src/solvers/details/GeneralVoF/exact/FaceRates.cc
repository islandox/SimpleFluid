/** @file FaceRates.cc
 * @brief Compiled general VoF FaceRates operation.
 */
#include "solvers/details/GeneralVoF/exact/FaceRates.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceTransfer.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail::general_vof::exact
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
FaceMeasures exchange_face_measures(
    const MeshHandle<>& mesh, const VoF::GeneralVoF& phases, const FaceMeasures& owned, std::size_t resource_budget)
{
    const Execution execution(mesh);
    using Domain = detail::GeneralVoFStorageDomain;
    detail::GeneralVoFExactStorage storage(resource_budget);
    std::size_t scalar_budget = 0, packet_budget = 0;
    execution.collective(
        [&]
        {
            storage.capacity(Domain::Input, owned.size(), sizeof(FaceMeasures::value_type) + 4 * sizeof(void*));
            for (const auto& [gid, value] : owned)
            {
                storage.measure(Domain::Input, value.first);
                storage.measure(Domain::Input, value.second);
            }
            const auto count = phases.tiles().size();
            storage.capacity(Domain::Scratch, count, 3 * sizeof(GO) + 2 * sizeof(int) + 12 * sizeof(void*));
            storage.capacity(Domain::Result, count, sizeof(FaceMeasures::value_type) + 4 * sizeof(void*));
            const auto pool = storage.remaining() / 2;
            storage.capacity(Domain::Scratch, pool, 1);
            packet_budget = pool / 8;
            if (count > (std::numeric_limits<std::size_t>::max() - 32) / 2)
                throw std::runtime_error("Phi publication scalar count overflow.");
            scalar_budget = storage.remaining() / (2 * count + 32);
            if (scalar_budget < 1024 || packet_budget < 1024)
                throw std::runtime_error("Phi publication cumulative reserve exhausted.");
            storage.capacity(Domain::Result, 2 * count, scalar_budget);
            storage.capacity(Domain::Scratch, 32, scalar_budget);
        });
    std::set<GO> required;
    for (const auto& tile : phases.tiles())
        required.insert(GO(tile.face_gid));
    Teuchos::Array<GO> ids(required.begin(), required.end());
    Teuchos::Array<int> owners(ids.size(), -1);
    (void) mesh.owned_face_map()->getRemoteIndexList(ids(), owners());
    std::map<GO, int> authority;
    std::vector<OwnerPacket> packets;
    execution.collective(
        [&]
        {
            for (std::size_t i = 0; i < std::size_t(ids.size()); ++i)
            {
                if (owners[i] < 0)
                    throw std::runtime_error("Missing canonical Phi authority.");
                authority.emplace(ids[i], owners[i]);
                OwnerPacket p;
                p.peer = owners[i];
                append_face_word(p.bytes, std::uint64_t(ids[i]));
                packets.push_back(std::move(p));
            }
        });
    const auto requests = exchange_owner_packets(packets, mesh.owned_cell_map()->getComm(), packet_budget);
    packets.clear();
    execution.collective(
        [&]
        {
            std::size_t used = 0;
            for (const auto& p : requests.packets)
            {
                if (p.bytes.size() != 8)
                    throw std::runtime_error("Bad Phi certificate request.");
                const GO face = GO(read_face_word(p.bytes, 0));
                const auto& values = owned.at(face);
                if (mesh.owned_face_map()->getLocalElement(face) < 0)
                    throw std::runtime_error("Foreign Phi certificate request.");
                OwnerPacket answer;
                answer.peer = p.peer;
                append_face_word(answer.bytes, std::uint64_t(face));
                for (const auto* value : {&values.first, &values.second})
                {
                    const auto encoded = value->encode(scalar_budget);
                    if (answer.bytes.size() > packet_budget || packet_budget - answer.bytes.size() < 8 ||
                        encoded.size() > packet_budget - answer.bytes.size() - 8)
                        throw std::runtime_error("Phi certificate exceeds budget.");
                    append_face_word(answer.bytes, encoded.size());
                    answer.bytes.insert(answer.bytes.end(), encoded.begin(), encoded.end());
                }
                if (answer.bytes.size() > packet_budget || used > packet_budget - answer.bytes.size() ||
                    sizeof(OwnerPacket) > packet_budget - used - answer.bytes.size())
                    throw std::runtime_error("Phi certificate packets exceed budget.");
                used += sizeof(OwnerPacket) + answer.bytes.size();
                packets.push_back(std::move(answer));
            }
        });
    const auto answers = exchange_owner_packets(packets, mesh.owned_cell_map()->getComm(), packet_budget);
    FaceMeasures result;
    execution.collective(
        [&]
        {
            for (const auto& p : answers.packets)
            {
                if (p.bytes.size() < 24)
                    throw std::runtime_error("Truncated Phi certificate.");
                const GO face = GO(read_face_word(p.bytes, 0));
                if (!authority.contains(face) || authority.at(face) != p.peer)
                    throw std::runtime_error("Wrong Phi certificate authority.");
                std::size_t offset = 8;
                std::array<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>, 2> value;
                for (auto& item : value)
                {
                    if (offset > p.bytes.size() || p.bytes.size() - offset < 8)
                        throw std::runtime_error("Truncated Phi certificate length.");
                    const auto count = read_face_word(p.bytes, offset);
                    offset += 8;
                    if (count > p.bytes.size() - offset)
                        throw std::runtime_error("Invalid Phi certificate length.");
                    item = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::decode(
                        std::span<const char>(p.bytes.data() + offset, std::size_t(count)), scalar_budget);
                    offset += count;
                }
                if (offset != p.bytes.size() || !result.emplace(face, std::pair{value[0], value[1]}).second)
                    throw std::runtime_error("Duplicate Phi certificate.");
            }
            if (result.size() != required.size())
                throw std::runtime_error("Incomplete Phi certificate publication.");
        });
    return result;
}
std::vector<VoF::GeneralVoFFaceVolume> face_volumes(const MeshHandle<>& mesh, const VoF::GeneralVoF& phases,
    const GeneralVoFProjectionProvenance& projection, double dt, std::size_t byte_budget)
{
    const Execution execution(mesh);
    using Domain = detail::GeneralVoFStorageDomain;
    detail::GeneralVoFExactStorage storage(byte_budget);
    std::size_t scalar_budget = 0, publication_budget = 0;
    execution.collective(
        [&]
        {
            const auto nf = mesh.num_owned_faces(), nt = phases.tiles().size();
            const auto& rate = projection.face_rate;
            if (rate.size() != nf)
                throw std::runtime_error("Missing accepted divergence-free face-rate ledger.");
            storage.capacity(Domain::Input, rate.capacity(),
                sizeof(SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>));
            for (const auto& value : rate)
                storage.measure(Domain::Input, value);
            storage.capacity(Domain::Scratch, nf, sizeof(FaceMeasures::value_type) + 4 * sizeof(void*));
            storage.capacity(Domain::Result, nt, sizeof(VoF::GeneralVoFFaceVolume));
            const auto pool = storage.remaining() / 2;
            storage.capacity(Domain::Scratch, pool, 1);
            publication_budget = pool;
            const auto slots = nf + 2 * nt + 16;
            if (slots < nf)
                throw std::runtime_error("Integrated Phi scalar capacity overflow.");
            scalar_budget = storage.remaining() / slots;
            if (scalar_budget < 1024 || publication_budget < 1024)
                throw std::runtime_error("Integrated Phi cumulative reserve exhausted.");
            storage.capacity(Domain::Scratch, nf + 16, scalar_budget);
            storage.capacity(Domain::Result, 2 * nt, scalar_budget);
        });
    FaceMeasures owned;
    execution.collective(
        [&]
        {
            for (std::size_t f = 0; f < mesh.num_owned_faces(); ++f)
                owned.emplace(mesh.face_global_id(LO(f)),
                    std::pair{detail::vof_exact::scale(projection.face_rate[f], dt, scalar_budget),
                        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>{}});
        });
    const auto controls = exchange_face_measures(mesh, phases, owned, publication_budget);
    std::vector<VoF::GeneralVoFFaceVolume> result;
    execution.collective(
        [&]
        {
            result.reserve(phases.tiles().size());
            for (const auto& tile : phases.tiles())
            {
                const auto& exact = controls.at(GO(tile.face_gid)).first;
                const double diagnostic = static_cast<double>(exact.enclosure().value);
                if (!std::isfinite(diagnostic))
                    throw std::runtime_error("Integrated Phi diagnostic overflows.");
                result.push_back({tile.face_gid, diagnostic, exact});
            }
        });
    return result;
}

} // namespace SimpleFluid::detail::general_vof::exact
