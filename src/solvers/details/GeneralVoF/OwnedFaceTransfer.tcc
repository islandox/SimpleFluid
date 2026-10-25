/** @file OwnedFaceTransfer.tcc
 * @brief Compiled transfer kernel with explicit source/target execution boundaries.
 */
#include "FVM/ExactFaceTransfer.hh"
#include "FVM/details/FiniteGeometryAccess.hh"
#include "geometry/PLIC/PlanarFace.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/PhaseArithmetic.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoF/FaceRates.hh"
#include "solvers/details/GeneralVoF/OwnedFaceTransfer.hh"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode>
FaceTransferResult transfer_owned_faces(
    const FaceTransferSource& source_input, const FaceTransferTarget& target, FaceTransferControls controls)
{
    using LO = DefaultTpetraTypes::local_ordinal_type;
    using GO = DefaultTpetraTypes::global_ordinal_type;
    const Execution source_execution(source_input.mesh), target_execution(target.mesh);
    FaceTransferResult transfer;
    using A = VoF::finite_detail::Arithmetic<Mode>;
    constexpr bool bounded_mode = Mode == ArithmeticMode::BoundedCertification;
    source_execution.require(target.arithmetic_mode == source_input.arithmetic_mode &&
                                 target.phases.arithmetic_mode() == source_input.arithmetic_mode,
        "General VoF ownership handoff changed arithmetic mode.");
    namespace fp = VoF::finite_detail;
    const auto comm = source_input.mesh.owned_cell_map()->getComm();
    const auto identity = [](const Meshes::AMR::AdaptiveTile& tile)
    {
        std::vector<char> result;
        for (const auto* key : {&tile.domain, &tile.surface, &tile.identity})
        {
            append_face_word(result, key->size());
            result.insert(result.end(), key->begin(), key->end());
        }
        return result;
    };
    const auto authority = [&](const std::vector<char>& key)
    {
        std::uint64_t hash = 1469598103934665603ULL;
        for (unsigned char c : key)
            hash = (hash ^ c) * 1099511628211ULL;
        return int(hash % comm->getSize()); // Routing only; full identity is compared.
    };
    const auto bytes = [](std::vector<char>& out, const std::vector<char>& value)
    {
        append_face_word(out, value.size());
        out.insert(out.end(), value.begin(), value.end());
    };
    const auto scalar = [](std::vector<char>& out, long double value)
    {
        int exponent = 0;
        const auto mantissa = std::frexp(value, &exponent);
        const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
        if (!std::isfinite(value) || static_cast<long double>(high) + low != mantissa)
            throw std::runtime_error("Finite face handoff scalar exceeds fixed precision.");
        append_face_word(out, std::bit_cast<std::uint64_t>(high));
        append_face_word(out, std::bit_cast<std::uint64_t>(low));
        append_face_word(out, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
    };
    const auto word = [](const std::vector<char>& in, std::size_t& offset)
    {
        const auto value = read_face_word(in, offset);
        offset += 8;
        return value;
    };
    const auto read_bytes = [&](const std::vector<char>& in, std::size_t& offset)
    {
        const auto count = word(in, offset);
        if (count > in.size() - std::min(offset, in.size()))
            throw std::runtime_error("Finite face handoff identity length is malformed.");
        std::vector<char> result(in.begin() + offset, in.begin() + offset + count);
        offset += count;
        return result;
    };
    const auto read_scalar = [&](const std::vector<char>& in, std::size_t& offset)
    {
        const auto high = std::bit_cast<double>(word(in, offset)), low = std::bit_cast<double>(word(in, offset));
        const auto exponent = std::bit_cast<std::int64_t>(word(in, offset));
        if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
            throw std::runtime_error("Finite face handoff exponent is malformed.");
        const auto value = std::scalbn(static_cast<long double>(high) + low, int(exponent));
        if (!std::isfinite(value))
            throw std::runtime_error("Finite face handoff scalar is nonfinite.");
        return value;
    };
    std::map<GO, const Meshes::AMR::AdaptiveTile*> target_tiles;
    std::vector<OwnerPacket> packets;
    source_execution.collective(
        [&]
        {
            if (source_input.projection.face_rate.size() != source_input.mesh.num_owned_faces() ||
                source_input.projection.face_radius.size() != source_input.mesh.num_owned_faces())
                throw std::runtime_error("Finite ownership handoff lacks source rate authority.");
            std::size_t used = 0;
            for (const auto& tile : source_input.phases.tiles())
            {
                const auto face = source_input.mesh.owned_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0)
                    continue;
                auto key = identity(tile);
                OwnerPacket packet;
                packet.peer = authority(key);
                append_face_word(packet.bytes, 0);
                append_face_word(packet.bytes, source_input.phases.accepted_control());
                bytes(packet.bytes, key);
                bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(tile.first));
                bytes(packet.bytes, tile.second ? Meshes::AMR::encode_adaptive_key(*tile.second) : std::vector<char>{});
                scalar(packet.bytes, source_input.projection.face_rate[face]);
                scalar(packet.bytes, source_input.projection.face_radius[face]);
                used += packet.bytes.size() + sizeof(OwnerPacket);
                if (used > controls.byte_budget / 2)
                    throw std::runtime_error("Finite face source directory exceeds budget.");
                packets.push_back(std::move(packet));
            }
            for (const auto& tile : target.phases.tiles())
            {
                const auto face = target.mesh.owned_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0)
                    continue;
                target_tiles.emplace(GO(tile.face_gid), &tile);
                auto key = identity(tile);
                OwnerPacket packet;
                packet.peer = authority(key);
                append_face_word(packet.bytes, 1);
                append_face_word(packet.bytes, source_input.phases.accepted_control());
                bytes(packet.bytes, key);
                append_face_word(packet.bytes, tile.face_gid);
                bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(tile.first));
                used += packet.bytes.size() + sizeof(OwnerPacket);
                if (used > controls.byte_budget / 2)
                    throw std::runtime_error("Finite face target directory exceeds budget.");
                packets.push_back(std::move(packet));
            }
        });
    const auto directory = exchange_owner_packets(packets, comm, controls.byte_budget / 2);
    struct SourceFace
    {
        std::vector<char> first, second;
        long double rate, radius;
    };
    struct Request
    {
        std::vector<char> identity, first;
        std::uint64_t face;
        int owner;
    };
    std::map<std::vector<char>, SourceFace> sources;
    std::vector<Request> requests;
    packets.clear();
    source_execution.collective(
        [&]
        {
            for (const auto& packet : directory.packets)
            {
                std::size_t offset = 0;
                const auto type = word(packet.bytes, offset), control = word(packet.bytes, offset);
                auto key = read_bytes(packet.bytes, offset);
                if (control != source_input.phases.accepted_control() || authority(key) != comm->getRank())
                    throw std::runtime_error("Finite face directory has stale or foreign source authority.");
                if (type == 0)
                {
                    auto first = read_bytes(packet.bytes, offset), second = read_bytes(packet.bytes, offset);
                    const auto rate = read_scalar(packet.bytes, offset), radius = read_scalar(packet.bytes, offset);
                    if (radius < 0 || !sources.emplace(key, SourceFace{first, second, rate, radius}).second)
                        throw std::runtime_error("Finite canonical source face is duplicated.");
                }
                else if (type == 1)
                {
                    const auto face = word(packet.bytes, offset);
                    requests.push_back({key, read_bytes(packet.bytes, offset), face, packet.peer});
                }
                else
                    throw std::runtime_error("Finite face directory type is malformed.");
                if (offset != packet.bytes.size())
                    throw std::runtime_error("Finite face directory has trailing payload.");
            }
            for (const auto& request : requests)
            {
                const auto found = sources.find(request.identity);
                if (found == sources.end())
                    throw std::runtime_error(
                        "Finite adaptive face handoff requires a bounded surface-overlap certificate.");
                const auto& source = found->second;
                const auto sign = request.first == source.first ? 1 : request.first == source.second ? -1 : 0;
                if (!sign)
                    throw std::runtime_error("Finite face handoff incident-cell orientation is foreign.");
                OwnerPacket reply;
                reply.peer = request.owner;
                append_face_word(reply.bytes, request.face);
                bytes(reply.bytes, request.identity);
                scalar(reply.bytes, sign * source.rate);
                scalar(reply.bytes, source.radius);
                packets.push_back(std::move(reply));
            }
        });
    const auto result = exchange_owner_packets(packets, comm, controls.byte_budget / 2);
    target_execution.collective(
        [&]
        {
            transfer.projection.byte_budget = controls.byte_budget;
            transfer.projection.arithmetic_mode = source_input.arithmetic_mode;
            transfer.projection.face_rate.resize(target.mesh.num_owned_faces());
            transfer.projection.face_radius.resize(target.mesh.num_owned_faces());
            transfer.projection.cell_radius.resize(target.volume.size());
            transfer.face_velocity.resize(target.mesh.num_faces());
            std::set<GO> seen;
            for (const auto& packet : result.packets)
            {
                std::size_t offset = 0;
                const auto gid = GO(word(packet.bytes, offset));
                const auto key = read_bytes(packet.bytes, offset);
                const auto rate = read_scalar(packet.bytes, offset), radius = read_scalar(packet.bytes, offset);
                const auto local = target.mesh.owned_face_map()->getLocalElement(gid);
                if (local < 0 || !seen.insert(gid).second || !target_tiles.contains(gid) ||
                    identity(*target_tiles.at(gid)) != key || packet.peer != authority(key) ||
                    offset != packet.bytes.size() || radius < 0)
                    throw std::runtime_error("Finite ownership face handoff has foreign or duplicate target.");
                if (target.mesh.is_geometry_exterior_face(local) && rate != 0)
                    throw std::runtime_error("Finite ownership handoff changes a closed wall.");
                const auto velocity =
                    A::divide(A::make_value(rate, radius), A::make_value(target.mesh.face_area(local), 0));
                transfer.face_velocity[local] = static_cast<double>(velocity.value);
                if (!std::isfinite(transfer.face_velocity[local]))
                    throw std::runtime_error("Finite ownership face velocity diagnostic overflows.");
                transfer.projection.face_rate[local] = rate;
                transfer.projection.face_radius[local] = radius;
            }
            if (seen.size() != target.mesh.num_owned_faces())
                throw std::runtime_error("Finite ownership handoff lacks target face coverage.");
        });
    const auto rates = face_rates<Mode>(target.mesh, target.phases, transfer.projection, target.flow);
    double local_divergence = 0;
    target_execution.collective(
        [&]
        {
            const auto lease = target.mesh.acquire_execution_view();
            for (std::size_t c = 0; c < target.volume.size(); ++c)
            {
                VoF::finite_detail::SumFor<Mode> net(source_input.arithmetic_mode);
                for (const auto face : target.mesh.faces(LO(c)))
                    net.add(target.mesh.owner_cell(face) == LO(c) ? rates[face] : A::negate(rates[face]));
                const auto residual = net.value();
                if (bounded_mode && std::abs(residual.value) > residual.absolute_error)
                    throw std::runtime_error("Finite ownership handoff loses canonical source-free closure.");
                transfer.projection.cell_radius[c] = residual.absolute_error;
                const auto divergence_bound = A::divide(
                    A::make_value(bounded_mode ? fp::bound_add(std::abs(residual.value), residual.absolute_error)
                                               : std::abs(residual.value),
                        0),
                    A::make_value(target.volume[c], 0));
                local_divergence = std::max(local_divergence, fp::upper_double(A::upper(divergence_bound)));
            }
        });
    transfer.face_velocity = target.flow.faces(transfer.face_velocity);
    const auto divergence = target_execution.maximum(local_divergence);
    target_execution.require(divergence <= controls.divergence_tolerance,
        "Finite fixed-clock face handoff exceeds the unchanged physical divergence target.");
    transfer.divergence = divergence;
    return transfer;
}
} // namespace SimpleFluid::detail::general_vof
