/** @file ConstructionControls.tcc
 * @brief Trial-pinned native controls used to rebuild authenticated transport
 * roots.
 */
#pragma once
#include "FVM/details/GeneralVoF/Wire.hh"
#include "geometry/PLIC/ConstructionRegistry.hh"

namespace SimpleFluid::VoF::construction_controls
{
using Records = std::map<std::pair<std::uint64_t, std::uint64_t>, AMR::AdaptiveBytes>;
using FaceAuthority = GeneralVoF::Data::FaceAuthority;
constexpr std::uint64_t version = 1;

inline void header(wire::Writer& writer, std::uint64_t kind, std::uint64_t id,
    const PLIC::GeometryPacketContext& context, ArithmeticMode mode)
{
    if (mode != ArithmeticMode::Ordinary && mode != ArithmeticMode::BoundedCertification)
        throw std::invalid_argument("Finite VoF construction control has a foreign arithmetic mode.");
    writer.word(kind);
    writer.word(id);
    writer.word(version);
    writer.word(std::uint64_t(mode));
    for (const auto pin : context.publication)
        writer.word(pin);
}
inline std::pair<std::uint64_t, std::uint64_t> header(
    wire::Reader& reader, const PLIC::GeometryPacketContext& expected, ArithmeticMode mode)
{
    const auto kind = reader.word(), id = reader.word();
    if (reader.word() != version || reader.word() != std::uint64_t(mode))
        throw std::invalid_argument("Finite VoF construction-control protocol or arithmetic mode differs.");
    for (const auto pin : expected.publication)
        if (reader.word() != pin)
            throw std::invalid_argument("Finite VoF construction control has stale publication pins.");
    return {kind, id};
}
inline AMR::AdaptiveBytes algorithm(const GeneralVoFOptions& options)
{
    wire::Writer writer;
    // Byte budgets are local admission limits, not shared algorithm semantics.
    for (const auto value : {std::uint64_t(options.maximum_stencil_cells), std::uint64_t(options.maximum_halo_cells),
             std::uint64_t(options.support_rings), std::uint64_t(options.arithmetic_mode),
             std::uint64_t(options.reconstruction.maximum_offset_iterations),
             std::uint64_t(options.reconstruction.maximum_fit_iterations),
             std::uint64_t(options.reconstruction.maximum_stencil_cells),
             std::uint64_t(options.reconstruction.maximum_clip_evaluations),
             std::uint64_t(options.sweep.maximum_simplices), std::uint64_t(options.sweep.maximum_atoms),
             std::uint64_t(options.sweep.maximum_pair_tests),
             std::uint64_t(SimpleFluid::geometry::swept_volume::TrajectoryLimits{}.maximum_constraints),
             std::uint64_t(SimpleFluid::geometry::swept_volume::TrajectoryLimits{}.maximum_operations)})
        writer.word(value);
    real(writer, options.reconstruction.volume_tolerance);
    real(writer, options.reconstruction.fit_tolerance);
    real(writer, options.reconstruction.maximum_condition);
    real(writer, options.reconstruction.orientation_tolerance);
    real(writer, options.reconstruction.maximum_model_residual);
    real(writer, options.reconstruction.minimum_phase_support_fraction);
    return std::move(writer.data);
}
inline void point(wire::Writer& writer, const PLIC::DynamicPoint& value)
{
    for (unsigned a = 0; a < 3; ++a)
        finite_value(writer, {value.value[a], value.absolute_error[a], value.arithmetic_mode});
}
inline PLIC::DynamicPoint point(wire::Reader& reader, ArithmeticMode mode)
{
    PLIC::DynamicPoint result;
    result.arithmetic_mode = mode;
    for (unsigned a = 0; a < 3; ++a)
    {
        const auto value = finite_value(reader, mode);
        result.value[a] = value.value;
        result.absolute_error[a] = value.absolute_error;
    }
    return result;
}
template<ArithmeticMode Mode>
PLIC::Point<Mode> departure(
    const GeneralVoF::Data& data, std::uint64_t node, const PLIC::GeometryPoint& velocity, double timestep)
{
    std::vector<PLIC::FaceConstraint<Mode>> walls;
    for (const auto& wall : data.topology->node_walls.at(node))
        walls.push_back({PLIC::ConvexCellGeometry<Mode>::from_dynamic(wall.finite_constraint.geometry),
            wall.finite_constraint.face});
    if (walls.size() > SimpleFluid::geometry::swept_volume::TrajectoryLimits{}.maximum_constraints)
        throw std::invalid_argument("Finite VoF construction wall catalog exceeds its constraint cap.");
    PLIC::GeometryLimits limits;
    limits.arithmetic_mode = Mode;
    limits.byte_budget = data.options.byte_budget;
    limits.maximum_operations = SimpleFluid::geometry::swept_volume::TrajectoryLimits{}.maximum_operations;
    return PLIC::backtrace_constrained<Mode>(
        PLIC::Point<Mode>(data.topology->finite_nodes.at(node)), velocity, timestep, walls, limits);
}
template<ArithmeticMode Mode>
AMR::AdaptiveBytes node_record(const GeneralVoF::Data& data, std::uint64_t node, const PLIC::Point<Mode>& source,
    const PLIC::Point<Mode>& departed, const PLIC::GeometryPoint& velocity, double timestep,
    const PLIC::GeometryPacketContext& context)
{
    wire::Writer writer;
    header(writer, 0, node, context, Mode);
    writer.bytes(data.topology->construction_nodes.at(node));
    point(writer, source.dynamic());
    point(writer, departed.dynamic());
    for (const auto component : velocity)
        real(writer, component);
    real(writer, timestep);
    writer.bytes(algorithm(data.options));
    const auto& walls = data.topology->node_walls.at(node);
    writer.word(walls.size());
    for (const auto& wall : walls)
    {
        writer.bytes(wall.identity);
        point(writer, wall.finite_normal);
    }
    return std::move(writer.data);
}
inline AMR::AdaptiveBytes face_record(const GeneralVoF::Data& data, std::uint64_t id,
    PLIC::DynamicGeometryValue control, long double allowance, double timestep,
    const PLIC::GeometryPacketContext& context)
{
    const auto& face = data.topology->construction_faces.at(id);
    wire::Writer writer;
    header(writer, 1, id, context, data.options.arithmetic_mode);
    writer.bytes(face.domain);
    writer.bytes(face.surface);
    writer.bytes(face.identity);
    writer.bytes(AMR::encode_adaptive_key(face.first));
    writer.word(bool(face.second));
    if (face.second)
        writer.bytes(AMR::encode_adaptive_key(*face.second));
    writer.word(face.exterior);
    writer.word(face.node_gids.size());
    for (const auto node : face.node_gids)
        writer.word(node);
    finite_value(writer, control);
    finite_real(writer, allowance);
    real(writer, timestep);
    writer.bytes(algorithm(data.options));
    return std::move(writer.data);
}
template<ArithmeticMode Mode>
std::pair<PLIC::DynamicGeometryValue, long double> accept_face_record(const GeneralVoF::Data& data, std::uint64_t id,
    std::span<const char> bytes, double timestep, const PLIC::GeometryPacketContext& context)
{
    const auto& face = data.topology->construction_faces.at(id);
    wire::Reader reader{bytes};
    if (header(reader, context, Mode) != std::pair<std::uint64_t, std::uint64_t>{1, id} ||
        reader.bytes() != face.domain || reader.bytes() != face.surface || reader.bytes() != face.identity ||
        reader.bytes() != AMR::encode_adaptive_key(face.first))
        throw std::invalid_argument("Finite VoF construction face is outside its "
                                    "retained native authority.");
    const auto second = reader.word();
    if (second != bool(face.second) || (second && reader.bytes() != AMR::encode_adaptive_key(*face.second)) ||
        reader.word() != face.exterior || reader.word() != face.node_gids.size())
        throw std::invalid_argument("Finite VoF construction face incidences "
                                    "differ from native authority.");
    for (const auto node : face.node_gids)
        if (reader.word() != node)
            throw std::invalid_argument("Finite VoF construction face node order "
                                        "differs from native authority.");
    const auto control = finite_value(reader, Mode);
    const auto allowance = finite_real(reader);
    if (real(reader) != timestep || reader.bytes() != algorithm(data.options))
        throw std::invalid_argument("Finite VoF construction face algorithm controls differ.");
    reader.done();
    if ((face.exterior || timestep == 0) && control.value != 0)
        throw std::invalid_argument("Finite VoF construction boundary or stationary control is nonzero.");
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        if (!(allowance > 0) || !std::isfinite(allowance))
            throw std::invalid_argument("Finite VoF construction face has an invalid owner allowance.");
        if (data.cells.contains(face.first) && (!face.second || data.cells.contains(*face.second)))
        {
            auto scale = data.cells.at(face.first).ordinary_quota_scale;
            if (face.second)
                scale = std::min(scale, data.cells.at(*face.second).ordinary_quota_scale);
            const auto expected =
                std::nextafter(static_cast<long double>(data.options.reconstruction.volume_tolerance) * scale, 0.L);
            if (allowance != expected)
                throw std::invalid_argument("Finite VoF construction face allowance "
                                            "differs from retained owner quotas.");
        }
    }
    else if (allowance != 0)
        throw std::invalid_argument("Certified VoF construction face has an ordinary allowance.");
    return {control, allowance};
}
template<ArithmeticMode Mode>
std::pair<PLIC::Point<Mode>, PLIC::Point<Mode>> accept_node_record(const GeneralVoF::Data& data, std::uint64_t id,
    std::span<const char> bytes, double timestep, const PLIC::GeometryPacketContext& context)
{
    wire::Reader reader{bytes};
    if (header(reader, context, Mode) != std::pair<std::uint64_t, std::uint64_t>{0, id} ||
        reader.bytes() != data.topology->construction_nodes.at(id))
        throw std::invalid_argument("Finite VoF construction node is outside retained native authority.");
    (void) point(reader, Mode);
    (void) point(reader, Mode);
    PLIC::GeometryPoint velocity;
    for (auto& value : velocity)
    {
        value = real(reader);
        if (!std::isfinite(value))
            throw std::invalid_argument("Finite VoF construction node velocity is nonfinite.");
    }
    if (real(reader) != timestep)
        throw std::invalid_argument("Finite VoF construction node timestep differs.");
    const PLIC::Point<Mode> source(data.topology->finite_nodes.at(id));
    const auto departed = departure<Mode>(data, id, velocity, timestep);
    const auto expected = node_record<Mode>(data, id, source, departed, velocity, timestep, context);
    if (!std::equal(bytes.begin(), bytes.end(), expected.begin(), expected.end()))
        throw std::invalid_argument("Finite VoF construction node or complete wall "
                                    "recipe differs from local authority.");
    return {source, departed};
}

inline std::span<const char> packet_view(wire::Reader& reader)
{
    const auto size = reader.word();
    if (size > reader.data.size() - reader.offset)
        throw std::invalid_argument("Finite VoF construction packet length is invalid.");
    const auto result = reader.data.subspan(reader.offset, std::size_t(size));
    reader.offset += std::size_t(size);
    return result;
}
inline void admit_root(const GeneralVoF::Data& data, const PLIC::ConstructionRootKey& root, std::uint64_t donor,
    std::uint64_t revision, const Key& recipient)
{
    using Kind = PLIC::ConstructionRootKey::Kind;
    if (!root.identity.empty())
        throw std::invalid_argument("Finite VoF construction dependency has an untrusted source identity.");
    if (root.kind == Kind::Native)
    {
        if (root.entity != donor || root.revision != revision || root.slot != 0 ||
            !data.topology->by_gid.contains(root.entity))
            throw std::invalid_argument("Finite VoF construction native root differs "
                                        "from its authenticated donor.");
        const auto& cell = data.cells.at(data.topology->by_gid.at(root.entity));
        if (cell.revision != root.revision)
            throw std::invalid_argument("Finite VoF construction native root is stale.");
        return;
    }
    if (root.kind != Kind::Sweep || root.revision != 0 || !data.topology->construction_faces.contains(root.entity))
        throw std::invalid_argument("Finite VoF construction root is outside the admitted native halo.");
    const auto& face = data.topology->construction_faces.at(root.entity);
    if (face.first != recipient && (!face.second || *face.second != recipient))
        throw std::invalid_argument("Finite VoF construction sweep is not incident to its authenticated recipient.");
    if (face.node_gids.size() > data.options.sweep.maximum_simplices / 4 || root.slot >= 3 * face.node_gids.size())
        throw std::invalid_argument("Finite VoF construction sweep ordinal exceeds "
                                    "its admitted source face.");
}

template<ArithmeticMode Mode>
void require_donor_ancestry(const PLIC::ConstructionRegistry<Mode>& registry,
    const PLIC::ConvexCellGeometry<Mode>& geometry, std::uint64_t donor, std::uint64_t revision)
{
    const PLIC::ConstructionRootKey root{PLIC::ConstructionRootKey::Kind::Native, donor, revision, 0, {}};
    if (!registry.derived_from(geometry, root))
        throw std::invalid_argument("Finite VoF atom is not constructed from its authenticated native donor.");
}

/** A fixed two-exchange service. Empty owners still retain and serve records
 * assigned to their modulo authority; requested geometry is never serialized.
 */
inline std::map<std::uint64_t, AMR::AdaptiveBytes> request_records(std::uint64_t kind,
    const std::set<std::uint64_t>& requested, const Records& authoritative, const PLIC::GeometryPacketContext& context,
    ArithmeticMode mode, Teuchos::RCP<const Teuchos::Comm<int>> comm, std::size_t limit, std::size_t retained_bytes)
{
    std::vector<OwnerPacket> requests;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(retained_bytes);
            for (const auto id : requested)
            {
                wire::Writer writer;
                header(writer, kind, id, context, mode);
                budget.packet(writer.data.size());
                requests.push_back({int(id % comm->getSize()), std::move(writer.data)});
            }
        });
    auto incoming = exchange_owner_packets(requests, comm, limit);
    std::vector<OwnerPacket>().swap(requests);
    std::vector<OwnerPacket> responses;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(retained_bytes);
            budget.add(incoming.packets.capacity() * sizeof(OwnerPacket));
            for (const auto& packet : incoming.packets)
                budget.add(packet.bytes.capacity());
            for (const auto& request : incoming.packets)
            {
                wire::Reader reader{request.bytes};
                const auto key = header(reader, context, mode);
                reader.done();
                if (key.first != kind || int(key.second % comm->getSize()) != comm->getRank())
                    throw std::invalid_argument("Finite VoF construction request reached a foreign authority.");
                const auto found = authoritative.find(key);
                if (found == authoritative.end())
                    throw std::invalid_argument("Finite VoF construction request has no agreed source control.");
                wire::Reader source{found->second};
                if (header(source, context, mode) != key)
                    throw std::invalid_argument("Finite VoF construction authority retained a stale control.");
                budget.packet(found->second.size());
                responses.push_back({request.peer, found->second});
            }
        });
    std::vector<OwnerPacket>().swap(incoming.packets);
    auto returned = exchange_owner_packets(responses, comm, limit);
    std::vector<OwnerPacket>().swap(responses);
    std::map<std::uint64_t, AMR::AdaptiveBytes> result;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(retained_bytes);
            budget.add(returned.packets.capacity() * sizeof(OwnerPacket));
            for (auto& response : returned.packets)
            {
                wire::Reader reader{response.bytes};
                const auto key = header(reader, context, mode);
                if (key.first != kind || !requested.contains(key.second) ||
                    response.peer != int(key.second % comm->getSize()))
                    throw std::invalid_argument("Finite VoF construction response has a "
                                                "foreign identity or sender.");
                budget.add(response.bytes.size() + sizeof(decltype(result)::value_type) + 4 * sizeof(void*));
                if (!result.emplace(key.second, std::move(response.bytes)).second)
                    throw std::invalid_argument("Finite VoF construction response is duplicated.");
            }
            if (result.size() != requested.size())
                throw std::invalid_argument("Finite VoF construction response is incomplete.");
        });
    return result;
}
} // namespace SimpleFluid::VoF::construction_controls
