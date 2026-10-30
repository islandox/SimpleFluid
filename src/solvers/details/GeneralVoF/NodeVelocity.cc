/** @file NodeVelocity.cc
 * @brief Canonical support union, affine fitting and node-authority exchange.
 */
#include "solvers/details/GeneralVoF/NodeVelocity.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <variant>

namespace SimpleFluid::detail::general_vof
{
namespace
{
using Pack = DefaultTpetraTypes;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
void append_word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned a = 0; a < 8; ++a)
        bytes.push_back(char(value >> (8 * a)));
}
std::uint64_t read_word(const std::vector<char>& bytes, std::size_t offset)
{
    std::uint64_t value = 0;
    for (unsigned a = 0; a < 8; ++a)
        value |= std::uint64_t(static_cast<unsigned char>(bytes.at(offset + a))) << (8 * a);
    return value;
}
} // namespace
std::vector<VoF::GeneralVoFNodeVelocity> node_velocity(const MeshHandle<>& mesh_view, const VoF::GeneralVoF& phases,
    std::span<const VoF::Vector> owned_velocity, std::size_t byte_budget)
{
    const auto* mesh = &mesh_view;
    const auto owned_count = mesh->num_owned_cells();
    const Execution execution(mesh_view);
    const bool finite = phases.arithmetic_mode() != ArithmeticMode::Exact;
    const auto comm = mesh->owned_cell_map()->getComm();
    const auto metadata = phases.support_cells();
    std::map<std::uint64_t, VoF::GeneralVoFSupportCell> cells;
    for (const auto& cell : metadata)
        cells.emplace(cell.cell, cell);
    std::map<std::uint64_t, std::vector<std::uint64_t>> supports;
    std::set<GO> donor_ids;
    execution.collective(
        [&]
        {
            for (const auto& node : phases.nodes())
            {
                auto ids = phases.node_cells(node.gid);
                if (ids.empty())
                    throw std::runtime_error(
                        "General VoF trajectory node has no authenticated owned-incident support.");
                for (auto gid : ids)
                {
                    if (gid > std::uint64_t(std::numeric_limits<GO>::max()) || !cells.contains(gid))
                        throw std::runtime_error("General VoF node support metadata is absent or out of native range.");
                    donor_ids.insert(GO(gid));
                }
                supports.emplace(node.gid, std::move(ids));
            }
        });
    Teuchos::Array<GO> ids(donor_ids.begin(), donor_ids.end());
    auto target =
        Teuchos::rcp(new Pack::map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), GO{}, comm));
    auto marker = execution.import_owned(std::vector<double>(owned_count, 1), target);
    execution.require(std::all_of(marker.begin(), marker.end(), [](double value) { return value == 1; }),
        "General VoF node support references an absent donor.");
    Teuchos::Array<int> owners(ids.size(), -1);
    (void) mesh->owned_cell_map()->getRemoteIndexList(ids(), owners());
    bool owner_valid = true;
    for (std::size_t i = 0; i < static_cast<std::size_t>(ids.size()); ++i)
        owner_valid = owner_valid && owners[i] == cells.at(std::uint64_t(ids[i])).owner;
    execution.require(owner_valid, "General VoF node support owner differs from its native map.");
    std::array<std::vector<double>, 3> velocity;
    for (unsigned a = 0; a < 3; ++a)
    {
        std::vector<double> component(owned_count);
        for (std::size_t c = 0; c < owned_count; ++c)
            component[c] = owned_velocity[c][a];
        velocity[a] = execution.import_owned(component, target);
    }
    // Relays contain source-owner imported values and authority-certified
    // revisions/metrics. The node authority unions all OWNED-incident seed
    // supports, so ghost placement and requester rank never choose a fit.
    std::vector<OwnerPacket> packets;
    execution.collective(
        [&]
        {
            std::size_t used = 0;
            for (const auto& node : phases.nodes())
            {
                OwnerPacket packet;
                packet.peer = int(node.gid % comm->getSize());
                append_word(packet.bytes, node.gid);
                const auto append_identity = [&](const std::vector<char>& identity)
                {
                    append_word(packet.bytes, identity.size());
                    packet.bytes.insert(packet.bytes.end(), identity.begin(), identity.end());
                };
                if (finite)
                    append_identity(node.identity);
                else
                    append_identity(PLIC::encode_identity(node.coordinate_identity));
                for (double x : node.coordinate)
                    append_word(packet.bytes, std::bit_cast<std::uint64_t>(x));
                const auto& support = supports.at(node.gid);
                append_word(packet.bytes, support.size());
                for (auto gid : support)
                {
                    const auto& cell = cells.at(gid);
                    const auto local = target->getLocalElement(GO(gid));
                    append_word(packet.bytes, gid);
                    append_word(packet.bytes, std::uint64_t(cell.owner));
                    append_word(packet.bytes, cell.revision);
                    for (double x : cell.centroid)
                        append_word(packet.bytes, std::bit_cast<std::uint64_t>(x));
                    for (unsigned a = 0; a < 3; ++a)
                        append_word(packet.bytes, std::bit_cast<std::uint64_t>(velocity[a][local]));
                    const auto key = Meshes::AMR::encode_adaptive_key(cell.key);
                    append_word(packet.bytes, key.size());
                    packet.bytes.insert(packet.bytes.end(), key.begin(), key.end());
                }
                std::vector<const Meshes::AMR::AdaptiveTile*> walls;
                for (const auto index : phases.node_tiles(node.gid))
                {
                    const auto& tile = phases.tiles()[index];
                    if (tile.exterior)
                        walls.push_back(&tile);
                }
                append_word(packet.bytes, walls.size());
                for (const auto* wall : walls)
                {
                    append_word(packet.bytes, wall->face_gid);
                    Vector normal{};
                    std::vector<char> certificate;
                    if (finite)
                    {
                        const auto face = mesh->overlap_face_map()->getLocalElement(GO(wall->face_gid));
                        const auto n = mesh->face_normal(face);
                        normal = {n.x, n.y, n.z};
                        certificate = wall->identity;
                    }
                    else
                    {
                        const auto n = wall->geometry->area_vector();
                        normal = {n.x, n.y, n.z};
                        certificate = wall->geometry->encode_incidence();
                    }
                    for (double x : normal)
                        append_word(packet.bytes, std::bit_cast<std::uint64_t>(x));
                    append_word(packet.bytes, certificate.size());
                    packet.bytes.insert(packet.bytes.end(), certificate.begin(), certificate.end());
                }
                if (packet.bytes.size() > byte_budget - used ||
                    sizeof(OwnerPacket) > byte_budget - used - packet.bytes.size())
                    throw std::runtime_error("General VoF node union exceeds its packet budget.");
                used += sizeof(OwnerPacket) + packet.bytes.size();
                packets.push_back(std::move(packet));
            }
        });
    auto gathered = exchange_owner_packets(packets, comm, byte_budget);
    packets.clear();
    using CoordinateIdentity = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
    using NodeIdentity = std::variant<std::vector<char>, CoordinateIdentity>;
    struct NodeUnion
    {
        NodeIdentity identity;
        std::vector<char> coordinate_bytes;
        Vector coordinate{};
        std::map<std::vector<char>, std::vector<char>> cells, walls;
        std::set<int> participants;
    };
    std::map<std::uint64_t, NodeUnion> nodes;
    execution.collective(
        [&]
        {
            for (const auto& packet : gathered.packets)
            {
                std::size_t offset = 0;
                const auto read = [&]()
                {
                    if (packet.bytes.size() - offset < 8)
                        throw std::runtime_error("Truncated general VoF node union.");
                    const auto value = read_word(packet.bytes, offset);
                    offset += 8;
                    return value;
                };
                const auto bytes = [&](std::size_t count)
                {
                    if (count > packet.bytes.size() - offset)
                        throw std::runtime_error("Oversized general VoF node union field.");
                    std::vector<char> result(packet.bytes.begin() + offset, packet.bytes.begin() + offset + count);
                    offset += count;
                    return result;
                };
                const auto gid = read();
                if (int(gid % comm->getSize()) != comm->getRank())
                    throw std::runtime_error("General VoF node union reached a wrong authority.");
                auto& node = nodes[gid];
                if (!node.participants.insert(packet.peer).second)
                    throw std::runtime_error("Duplicate general VoF node union requester.");
                auto encoded_identity = bytes(read());
                NodeIdentity identity = finite
                                            ? NodeIdentity{std::move(encoded_identity)}
                                            : NodeIdentity{PLIC::decode_identity<CoordinateIdentity>(encoded_identity)};
                auto coordinate = bytes(24);
                if (node.participants.size() == 1)
                {
                    node.identity = identity;
                    node.coordinate_bytes = coordinate;
                    for (unsigned a = 0; a < 3; ++a)
                        node.coordinate[a] = std::bit_cast<double>(read_word(coordinate, 8 * a));
                }
                else if (node.identity != identity || node.coordinate_bytes != coordinate)
                    throw std::runtime_error("General VoF node union identity/metric disagreement.");
                const auto count = read();
                if (count > (packet.bytes.size() - offset) / 80)
                    throw std::runtime_error("Invalid general VoF node support count.");
                for (std::uint64_t i = 0; i < count; ++i)
                {
                    auto record = bytes(72);
                    const auto key = bytes(read());
                    if (Meshes::AMR::encode_adaptive_key(Meshes::AMR::decode_adaptive_key(key)) != key)
                        throw std::runtime_error("General VoF node support key is not canonical.");
                    const auto [found, inserted] = node.cells.emplace(key, record);
                    if (!inserted && found->second != record)
                        throw std::runtime_error(
                            "General VoF node source owner/revision/metric/velocity disagreement.");
                }
                const auto walls = read();
                for (std::uint64_t i = 0; i < walls; ++i)
                {
                    const auto start = offset;
                    (void) read();
                    (void) bytes(24);
                    const auto certificate = bytes(read());
                    std::vector<char> record(packet.bytes.begin() + start, packet.bytes.begin() + offset);
                    const auto [found, inserted] = node.walls.emplace(certificate, record);
                    if (!inserted && found->second != record)
                        throw std::runtime_error("General VoF node physical wall certificate disagreement.");
                }
                if (offset != packet.bytes.size())
                    throw std::runtime_error("Trailing general VoF node union data.");
            }
            for (const auto& [gid, node] : nodes)
            {
                struct Sample
                {
                    Vector center, velocity;
                };
                std::vector<Sample> samples;
                for (const auto& [donor, record] : node.cells)
                {
                    Sample sample;
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        sample.center[a] = std::bit_cast<double>(read_word(record, 24 + 8 * a));
                        sample.velocity[a] = std::bit_cast<double>(read_word(record, 48 + 8 * a));
                    }
                    samples.push_back(sample);
                }
                if (samples.empty())
                    throw std::runtime_error("General VoF node authority has no union support.");
                Vector value = samples.front().velocity;
                bool constant = true;
                for (const auto& sample : samples)
                    for (unsigned a = 0; a < 3; ++a)
                        constant = constant && sample.velocity[a] == value[a];
                if (!constant)
                {
                    Vector lower = samples.front().center, upper = lower, span{};
                    for (const auto& sample : samples)
                        for (unsigned a = 0; a < 3; ++a)
                        {
                            lower[a] = std::min(lower[a], sample.center[a]);
                            upper[a] = std::max(upper[a], sample.center[a]);
                        }
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        span[a] = upper[a] - lower[a];
                        if (!(span[a] > 0))
                            throw std::runtime_error("General VoF node union affine support is deficient.");
                    }
                    std::array<std::array<long double, 11>, 4> matrix{};
                    for (const auto& sample : samples)
                    {
                        std::array<long double, 4> row{1, 0, 0, 0};
                        for (unsigned a = 0; a < 3; ++a)
                            row[1 + a] = (static_cast<long double>(sample.center[a]) - node.coordinate[a]) / span[a];
                        const auto weight = 1 / (1 + row[1] * row[1] + row[2] * row[2] + row[3] * row[3]);
                        for (unsigned a = 0; a < 4; ++a)
                        {
                            for (unsigned b = 0; b < 4; ++b)
                                matrix[a][b] += weight * row[a] * row[b];
                            for (unsigned b = 0; b < 3; ++b)
                                matrix[a][4 + b] += weight * row[a] * sample.velocity[b];
                        }
                    }
                    long double norm = 0;
                    for (unsigned a = 0; a < 4; ++a)
                    {
                        long double sum = 0;
                        for (unsigned b = 0; b < 4; ++b)
                            sum += std::abs(matrix[a][b]);
                        norm = std::max(norm, sum);
                        matrix[a][7 + a] = 1;
                    }
                    for (unsigned a = 0; a < 4; ++a)
                    {
                        unsigned pivot = a;
                        for (unsigned b = a + 1; b < 4; ++b)
                            if (std::abs(matrix[b][a]) > std::abs(matrix[pivot][a]))
                                pivot = b;
                        if (!(std::abs(matrix[pivot][a]) > 64 * std::numeric_limits<long double>::epsilon() * norm))
                            throw std::runtime_error("General VoF node union affine support is rank deficient.");
                        std::swap(matrix[a], matrix[pivot]);
                        const auto diagonal = matrix[a][a];
                        for (unsigned b = a; b < 11; ++b)
                            matrix[a][b] /= diagonal;
                        for (unsigned row = 0; row < 4; ++row)
                            if (row != a)
                            {
                                const auto factor = matrix[row][a];
                                for (unsigned b = a; b < 11; ++b)
                                    matrix[row][b] -= factor * matrix[a][b];
                            }
                    }
                    long double inverse_norm = 0;
                    for (unsigned a = 0; a < 4; ++a)
                    {
                        long double sum = 0;
                        for (unsigned b = 7; b < 11; ++b)
                            sum += std::abs(matrix[a][b]);
                        inverse_norm = std::max(inverse_norm, sum);
                    }
                    if (!std::isfinite(norm * inverse_norm) || norm * inverse_norm > 1e8)
                        throw std::runtime_error("General VoF node union affine support is ill-conditioned.");
                    for (unsigned a = 0; a < 3; ++a)
                        value[a] = static_cast<double>(matrix[0][4 + a]);
                }
                // The FVM authority applies the exact wall-nullspace constraint
                // once, using the complete authenticated incident wall catalog.
                // Keep this fitted velocity as the physical model input.
                std::vector<char> result;
                append_word(result, gid);
                for (double component : value)
                {
                    if (!std::isfinite(component))
                        throw std::runtime_error("General VoF node trajectory is nonfinite.");
                    append_word(result, std::bit_cast<std::uint64_t>(component));
                }
                for (int participant : node.participants)
                    packets.push_back({participant, result});
            }
        });
    auto selected = exchange_owner_packets(packets, comm, byte_budget);
    std::vector<VoF::GeneralVoFNodeVelocity> result;
    execution.collective(
        [&]
        {
            std::map<std::uint64_t, Vector> values;
            for (const auto& packet : selected.packets)
            {
                if (packet.bytes.size() != 32)
                    throw std::runtime_error("Invalid general VoF node publication.");
                const auto gid = read_word(packet.bytes, 0);
                Vector velocity{};
                if (packet.peer != int(gid % comm->getSize()))
                    throw std::runtime_error("General VoF node publication has a wrong authority.");
                for (unsigned a = 0; a < 3; ++a)
                    velocity[a] = std::bit_cast<double>(read_word(packet.bytes, 8 + 8 * a));
                if (!supports.contains(gid) || !values.emplace(gid, velocity).second)
                    throw std::runtime_error("General VoF node publication has a duplicate/foreign node.");
            }
            if (values.size() != phases.nodes().size())
                throw std::runtime_error("General VoF node publication is incomplete.");
            for (const auto& node : phases.nodes())
                result.push_back({node.gid, values.at(node.gid)});
        });
    return result;
}
} // namespace SimpleFluid::detail::general_vof
