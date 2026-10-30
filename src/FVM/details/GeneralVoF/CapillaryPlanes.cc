/** @file CapillaryPlanes.cc
 * @brief Owner-authenticated, independently volume-admitted capillary plane witnesses.
 */
#include "FVM/details/GeneralVoF/CapillaryPlanes.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/PlaneMeasure.hh"

namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
constexpr auto mode = ArithmeticMode::Ordinary;
constexpr auto original_face = std::numeric_limits<std::uint64_t>::max();
constexpr auto full_support_plane = original_face - 1;
constexpr std::uint64_t protocol = 0x5346434150504c03ULL;
using Geometry = PLIC::ConvexCellGeometry<mode>;
using Wide = std::array<long double, 3>;
using Planes = std::map<Key, PLIC::PolyhedralInterface>;
struct Candidate
{
    const AMR::AdaptiveTile* tile = nullptr;
    PLIC::GeometryHalfspace gas;
    std::optional<PLIC::PlaneSupportIdentity<mode>> support;
};
int geometric_phase(const Cell& cell, const PLIC::PolyhedralInterface& original)
{
    if (cell.declaration == GeneralVoFPhaseGeometry::PureGas)
        return 1;
    if (cell.declaration == GeneralVoFPhaseGeometry::PureCarrier)
        return -1;
    if (cell.retained_interface)
    {
        if (cell.retained_interface->pure_gas)
            return 1;
        if (cell.retained_interface->pure_carrier)
            return -1;
        return 0;
    }
    if (original.pure_gas)
        return 1;
    if (original.pure_carrier)
        return -1;
    return 0;
}
long double signed_distance(const Wide& point, const PLIC::GeometryHalfspace& plane)
{
    long double value = -static_cast<long double>(plane.offset);
    for (unsigned axis = 0; axis < 3; ++axis)
        value +=
            static_cast<long double>(plane.normal[axis]) * (point[axis] - static_cast<long double>(plane.anchor[axis]));
    return value;
}
bool complete_stencil(const GeneralVoF::Data& data, const Key& target, std::set<Key>& visited, wire::Budget& budget)
{
    const auto insert = [&](const Key& key)
    {
        budget.add(3 * (sizeof(Key) + AMR::encode_adaptive_key(key).size() + 4 * sizeof(void*)));
        visited.insert(key);
    };
    insert(target);
    std::set<Key> front{target};
    // Authenticate a native-plane witness against the entire configured support
    // neighborhood. This stronger capillary-only check requires full cells even
    // when transport reconstruction selects an earlier complete geometric ring.
    for (unsigned ring = 0; ring < data.options.support_rings; ++ring)
    {
        std::set<Key> next;
        for (const auto& center : front)
        {
            const auto row = data.topology->adjacency.find(center);
            if (row == data.topology->adjacency.end())
                continue;
            for (const auto& neighbor : row->second)
            {
                if (!data.cells.contains(neighbor))
                    return false;
                if (visited.contains(neighbor))
                    continue;
                if (visited.size() - 1 == data.options.maximum_stencil_cells)
                    throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
                        "Capillary plane witness exceeds the reconstruction stencil budget.");
                insert(neighbor);
                next.insert(neighbor);
            }
        }
        front = std::move(next);
    }
    return true;
}
void evaluation(const GeneralVoF::Data& data, std::size_t& evaluations)
{
    if (evaluations++ >= data.options.reconstruction.maximum_clip_evaluations)
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit,
            "Capillary plane witness exceeds the reconstruction evaluation budget.");
}
bool pure_side(const Geometry& geometry, const PLIC::GeometryHalfspace& gas, int phase)
{
    try
    {
        // Structural classification uses the public Ordinary clip's compensated
        // predicates and original support incidence. No rounded support scalar
        // or volume allowance can erase a positive geometric sliver.
        const auto split = geometry.clip(gas, false);
        return phase > 0 ? bool(split.inside) && !split.outside : !split.inside && bool(split.outside);
    }
    catch (const PLIC::GeometryError& error)
    {
        if (error.code() == PLIC::GeometryErrorCode::UnresolvedArithmetic ||
            error.code() == PLIC::GeometryErrorCode::InvalidTopology)
            return false; // Optional candidate cannot establish pure geometry.
        throw;            // Resource exhaustion and metadata/provenance failures retain their evidence.
    }
}
std::optional<Candidate> native_candidate(
    const GeneralVoF::Data& data, const Planes& original, const AMR::AdaptiveTile& tile, std::size_t& evaluations)
{
    if (tile.exterior || !tile.second || !tile.second_image_identity.is_zero() || !data.cells.contains(tile.first) ||
        !data.cells.contains(*tile.second))
        return {};
    const auto& first = data.cells.at(tile.first);
    const auto& second = data.cells.at(*tile.second);
    const int a = geometric_phase(first, original.at(tile.first));
    const int b = geometric_phase(second, original.at(*tile.second));
    if (a * b != -1)
        return {};
    const auto& points = tile.geometry->points();
    if (points.size() < 3 || tile.node_gids.size() != points.size())
        throw std::invalid_argument("Capillary native plane lacks its canonical node loop.");
    const auto area = tile.geometry->area_vector();
    const auto magnitude = std::hypot(area.x, area.y, area.z);
    if (!(magnitude > 0) || !std::isfinite(magnitude))
        return {};
    Candidate candidate;
    candidate.tile = &tile;
    candidate.gas.normal = {area.x / magnitude, area.y / magnitude, area.z / magnitude};
    long double orientation = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
        orientation += static_cast<long double>(candidate.gas.normal[axis]) *
                       (static_cast<long double>(second.centroid[axis]) - first.centroid[axis]);
    if (orientation == 0 || !std::isfinite(orientation))
        return {};
    if ((orientation > 0 ? 1 : -1) != a)
        for (auto& value : candidate.gas.normal)
            value = -value;
    const auto anchor = std::min_element(tile.node_gids.begin(), tile.node_gids.end()) - tile.node_gids.begin();
    for (unsigned axis = 0; axis < 3; ++axis)
        candidate.gas.anchor[axis] = static_cast<double>(points[anchor][axis]);
    long double scale = 0;
    for (const auto& point : points)
        for (unsigned axis = 0; axis < 3; ++axis)
            scale = std::max(scale, std::abs(point[axis] - points[anchor][axis]));
    if (!(scale > 0) || !std::isfinite(scale))
        return {};
    // This existing native-plane representation gate establishes candidate
    // placement only. Pure-cell admission below permits no metric sliver.
    const auto representation = 64 * std::numeric_limits<double>::epsilon() * scale;
    for (const auto& point : points)
        if (!std::isfinite(signed_distance(point, candidate.gas)) ||
            std::abs(signed_distance(point, candidate.gas)) > representation)
            return {};
    // The witness's two original endpoints must themselves lie on their
    // declared sides, even when only one endpoint belongs to the target stencil.
    for (const auto* cell : {&first, &second})
    {
        evaluation(data, evaluations);
        if (!pure_side(*Geometry::from_dynamic(cell->finite_geometry), candidate.gas,
                geometric_phase(*cell, original.at(cell->key))))
            return {};
    }
    candidate.support = PLIC::planar_support<mode>(*tile.geometry);
    return candidate;
}
struct MinorityEvidence
{
    long double difference = 0, numerical_error = 0;
};
std::optional<MinorityEvidence> admit_cell(const GeneralVoF::Data& data, const Planes& original, const Key& key,
    const PLIC::GeometryHalfspace& gas, std::size_t& evaluations, wire::Budget& budget)
{
    const auto& cell = data.cells.at(key);
    const auto geometry = Geometry::from_dynamic(cell.finite_geometry);
    const int phase = geometric_phase(cell, original.at(key));
    evaluation(data, evaluations);
    if (phase)
        return pure_side(*geometry, gas, phase) ? std::optional<MinorityEvidence>{MinorityEvidence{}} : std::nullopt;
    const bool minority_is_gas = cell.phase.gas_volume <= cell.phase.carrier_volume;
    const long double target = minority_is_gas ? cell.phase.gas_volume : cell.phase.carrier_volume;
    if (!(target > 0) || !std::isfinite(target))
        return {};
    // Measure the minority directly, including carrier minorities; subtracting
    // an almost-full gas volume from the bulk would lose independent tails.
    auto minority = gas;
    if (!minority_is_gas)
    {
        for (auto& value : minority.normal)
            value = -value;
        minority.offset = -minority.offset;
    }
    // Account for the temporary boundary query separately from retained maps.
    wire::Budget query{budget.limit};
    query.add(budget.used);
    query.add(geometry->storage_bytes());
    PLIC::PlaneVolumeMeasure measured;
    try
    {
        measured = PLIC::OrdinaryPlaneMeasure(*geometry).below(minority);
    }
    catch (const PLIC::GeometryError& error)
    {
        if (error.code() == PLIC::GeometryErrorCode::UnresolvedArithmetic ||
            error.code() == PLIC::GeometryErrorCode::InvalidTopology)
            return {};
        throw;
    }
    const auto residual = measured.volume - target;
    const auto allowed = data.options.reconstruction.volume_tolerance * target;
    if (measured.status != PLIC::PlaneMeasureStatus::Resolved || !std::isfinite(residual) ||
        !std::isfinite(measured.numerical_error_estimate) || measured.numerical_error_estimate < 0 ||
        std::abs(residual) + measured.numerical_error_estimate > allowed)
        return {};
    return MinorityEvidence{residual, measured.numerical_error_estimate};
}
bool same_plane(const Candidate& a, const Candidate& b)
{
    // Equal retained support is construction evidence, rather than a geometric
    // proximity test. Distinct valid source supports conservatively stay
    // ambiguous, even if a rounded coordinate comparison might coincide.
    return a.support == b.support && a.gas.normal == b.gas.normal;
}
std::pair<std::uint64_t, PLIC::PolyhedralInterface> choose_plane(
    const GeneralVoF::Data& data, const Planes& original, const Key& target, std::size_t retained_bytes)
{
    auto initial = original.at(target);
    const auto& target_cell = data.cells.at(target);
    SimpleFluid::detail::vof_failure::Scope context("capillary_plane_reconstruction");
    context.record.cell = target_cell.gid;
    context.record.cell_owner = target_cell.owner;
    context.record.mark(SimpleFluid::detail::vof_failure::Field::Cell);
    context.record.mark(SimpleFluid::detail::vof_failure::Field::CellOwner);
    if (geometric_phase(target_cell, initial))
        return {original_face, initial};
    wire::Budget budget{data.options.byte_budget};
    budget.add(retained_bytes);
    std::set<Key> stencil;
    if (!complete_stencil(data, target, stencil, budget))
        throw GeneralVoFSurfaceResolutionError(GeneralVoFSurfaceResolutionReason::IncompleteSupport,
            "Capillary plane reconstruction requires its complete physical support.");
    // Curvature and transport have distinct geometric support requirements.
    // Always estimate capillary orientation from the complete configured rings;
    // a locally full-rank transport stencil need not resolve surface bending.
    // An adopted interface capsule remains the immutable geometric authority.
    const auto initial_choice = target_cell.retained_interface ? original_face : full_support_plane;
    if (!target_cell.retained_interface)
    {
        budget.add(2 * (stencil.size() - 1) * sizeof(PLIC::PolyhedralPhaseSample<mode>));
        const auto support = reconstruction_stencil<mode>(data, target, true);
        initial = PLIC::PolyhedralReconstruction<mode>::reconstruct(
            reconstruction_sample<mode>(target_cell), support, data.options.reconstruction);
    }
    std::vector<const AMR::AdaptiveTile*> faces;
    for (const auto& tile : data.topology->surface_tiles)
        if (stencil.contains(tile.first) || (tile.second && stencil.contains(*tile.second)))
        {
            budget.add(2 * sizeof(void*));
            faces.push_back(&tile);
        }
    std::sort(faces.begin(), faces.end(), [](const auto* a, const auto* b) { return a->face_gid < b->face_gid; });
    std::optional<Candidate> selected;
    MinorityEvidence target_evidence;
    long double selected_stencil_residual = 0;
    std::size_t evaluations = 0;
    for (const auto* face : faces)
    {
        const auto candidate = native_candidate(data, original, *face, evaluations);
        if (!candidate)
            continue;
        wire::Budget candidate_budget{budget.limit};
        candidate_budget.add(budget.used);
        candidate_budget.add(candidate->support->payload_bytes());
        bool valid = true;
        MinorityEvidence evidence;
        long double squared_residual = 0;
        for (const auto& key : stencil)
        {
            const auto admitted = admit_cell(data, original, key, candidate->gas, evaluations, candidate_budget);
            if (!admitted)
            {
                valid = false;
                break;
            }
            if (key == target)
                evidence = *admitted;
            else
            {
                const auto bulk = Geometry::from_dynamic(data.cells.at(key).finite_geometry)->volume().value;
                const auto normalized = admitted->difference / bulk;
                squared_residual += normalized * normalized;
            }
        }
        if (!valid)
            continue;
        if (selected && !same_plane(*selected, *candidate))
            return {initial_choice, initial};
        if (!selected)
        {
            budget.add(candidate->support->payload_bytes());
            selected = *candidate;
            target_evidence = evidence;
            selected_stencil_residual = stencil.size() > 1 ? std::sqrt(squared_residual / (stencil.size() - 1)) : 0;
        }
    }
    if (!selected)
        return {initial_choice, initial};
    // A native witness is a separately admitted plane, not the old angular
    // optimizer result. Retain only the unchanged spatial-resolution evidence;
    // optimizer condition/iterations/convergence stay at their neutral defaults.
    PLIC::PolyhedralInterface result;
    result.resolution = initial.resolution;
    result.gas = selected->gas;
    result.stencil_volume_residual = static_cast<double>(selected_stencil_residual);
    result.stencil_volume_residual_measured = true;
    result.finite_geometry = true;
    result.finite_target_volume_difference = {target_evidence.difference, 0, mode};
    result.target_volume_residual = static_cast<double>(target_evidence.difference);
    result.target_volume_numerical_error_estimate = target_evidence.numerical_error;
    return {selected->tile->face_gid, result};
}
void stamp(wire::Writer& writer, const GeneralVoF::Data& data)
{
    writer.word(protocol);
    writer.word(data.control);
    writer.word(data.hierarchy->topology_generation());
    writer.word(data.hierarchy->partition_generation());
}
void check_stamp(wire::Reader& reader, const GeneralVoF::Data& data)
{
    if (reader.word() != protocol || reader.word() != data.control ||
        reader.word() != data.hierarchy->topology_generation() ||
        reader.word() != data.hierarchy->partition_generation())
        throw std::invalid_argument("Capillary plane source/topology/partition context differs.");
}
std::size_t packet_storage(const std::vector<OwnerPacket>& packets, std::size_t limit)
{
    wire::Budget budget{limit};
    if (packets.capacity() > limit / sizeof(OwnerPacket))
        throw PLIC::GeometryError(
            PLIC::GeometryErrorCode::ResourceLimit, "Capillary plane packet capacity exceeds its byte budget.");
    budget.add(packets.capacity() * sizeof(OwnerPacket));
    for (const auto& packet : packets)
        budget.add(packet.bytes.capacity());
    return budget.used;
}
void release(std::vector<OwnerPacket>& packets)
{
    std::vector<OwnerPacket>{}.swap(packets);
}
} // namespace

Planes import_capillary_planes(const GeneralVoF::Data& data)
{
    const auto comm = data.hierarchy->communicator();
    const auto rank = comm->getRank();
    const auto limit = data.options.byte_budget;
    Planes original, result;
    std::map<Key, std::uint64_t> choices;
    std::size_t retained_bytes = 0, requested_bytes = 0;
    wire::collective(*comm,
        [&]
        {
            current(data);
            if (data.options.arithmetic_mode != mode)
                throw std::invalid_argument("Capillary plane source mode or complete plane catalog differs.");
            wire::Budget budget{limit};
            for (const auto& [key, cell] : data.cells)
            {
                PLIC::PolyhedralInterface initial;
                if (cell.retained_interface)
                    initial = *cell.retained_interface;
                else
                {
                    initial.finite_geometry = true;
                    initial.pure_gas =
                        cell.declaration == GeneralVoFPhaseGeometry::PureGas ||
                        (cell.declaration == GeneralVoFPhaseGeometry::Infer && cell.phase.carrier_volume == 0);
                    initial.pure_carrier =
                        cell.declaration == GeneralVoFPhaseGeometry::PureCarrier ||
                        (cell.declaration == GeneralVoFPhaseGeometry::Infer && cell.phase.gas_volume == 0);
                    if (initial.pure_gas || initial.pure_carrier)
                    {
                        const auto minority = initial.pure_gas ? cell.phase.carrier_volume : cell.phase.gas_volume;
                        initial.target_volume_residual = minority;
                        initial.finite_target_volume_difference = {-static_cast<long double>(minority), 0, mode};
                    }
                }
                // Original/result maps and owner choices coexist throughout
                // selection and exchange; include both sets of plane records.
                budget.add(2 * sizeof(Planes::value_type) + sizeof(Key) + sizeof(std::uint64_t) + 12 * sizeof(void*) +
                           3 * AMR::encode_adaptive_key(key).size());
                original.emplace(key, initial);
                result.emplace(key, initial);
                if (cell.owner == rank)
                    choices.emplace(key, original_face);
            }
            retained_bytes = budget.used;
            for (const auto& [key, cell] : data.cells)
                if (cell.owner == rank)
                {
                    auto [face, chosen] = choose_plane(data, original, key, retained_bytes);
                    choices.at(key) = face;
                    result.at(key) = std::move(chosen);
                }
        });
    std::vector<OwnerPacket> requests;
    std::set<Key> requested;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(retained_bytes);
            for (const auto& [key, cell] : data.cells)
                if (cell.owner != rank && !geometric_phase(cell, original.at(key)))
                {
                    wire::Writer writer;
                    stamp(writer, data);
                    writer.bytes(AMR::encode_adaptive_key(key));
                    writer.word(cell.revision);
                    budget.packet(writer.data.capacity());
                    budget.add(sizeof(OwnerPacket)); // Vector growth remains below twice its size.
                    const auto key_bytes = sizeof(Key) + AMR::encode_adaptive_key(key).size() + 4 * sizeof(void*);
                    budget.add(key_bytes);
                    requested_bytes += key_bytes;
                    requested.insert(key);
                    requests.push_back({cell.owner, std::move(writer.data)});
                }
        });
    const auto exchange_limit = [&](const std::vector<OwnerPacket>& outgoing)
    {
        std::size_t remaining = 0;
        wire::collective(*comm,
            [&]
            {
                wire::Budget budget{limit};
                budget.add(retained_bytes);
                budget.add(requested_bytes);
                budget.add(packet_storage(outgoing, limit));
                remaining = limit - budget.used;
            });
        return remaining;
    };
    auto incoming = exchange_owner_packets(requests, comm, exchange_limit(requests));
    release(requests);
    std::vector<OwnerPacket> replies;
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(retained_bytes);
            budget.add(requested_bytes);
            budget.add(packet_storage(incoming.packets, limit));
            std::set<std::pair<int, Key>> seen;
            for (const auto& packet : incoming.packets)
            {
                wire::Reader reader{packet.bytes};
                check_stamp(reader, data);
                const auto bytes = reader.bytes();
                const auto key = AMR::decode_adaptive_key(bytes);
                const auto revision = reader.word();
                reader.done();
                const auto& cell = data.cells.at(key);
                if (cell.owner != rank || cell.revision != revision || geometric_phase(cell, original.at(key)) ||
                    !seen.emplace(packet.peer, key).second)
                    throw std::invalid_argument(
                        "Capillary plane request lacks its current mixed-cell owner authority.");
                budget.add(sizeof(std::pair<int, Key>) + bytes.size() + 4 * sizeof(void*));
                wire::Writer writer;
                stamp(writer, data);
                writer.bytes(bytes);
                writer.word(revision);
                writer.word(choices.at(key));
                plane(writer, result.at(key));
                budget.packet(writer.data.capacity());
                budget.add(sizeof(OwnerPacket));
                replies.push_back({packet.peer, std::move(writer.data)});
            }
        });
    release(incoming.packets);
    auto received = exchange_owner_packets(replies, comm, exchange_limit(replies));
    release(replies);
    wire::collective(*comm,
        [&]
        {
            wire::Budget budget{limit};
            budget.add(retained_bytes);
            budget.add(requested_bytes);
            budget.add(packet_storage(received.packets, limit));
            std::set<Key> seen;
            for (const auto& packet : received.packets)
            {
                wire::Reader reader{packet.bytes};
                check_stamp(reader, data);
                const auto key = AMR::decode_adaptive_key(reader.bytes());
                const auto revision = reader.word();
                const auto face = reader.word();
                const auto& cell = data.cells.at(key);
                if (!requested.contains(key) || cell.owner != packet.peer || cell.owner == rank ||
                    cell.revision != revision || !seen.insert(key).second)
                    throw std::invalid_argument("Capillary plane reply lacks its current source/cell owner authority.");
                budget.add(sizeof(Key) + AMR::encode_adaptive_key(key).size() + 4 * sizeof(void*));
                auto chosen = plane(reader, mode);
                reader.done();
                if (!chosen.finite_geometry || chosen.pure_gas || chosen.pure_carrier)
                    throw std::invalid_argument("Capillary plane reply differs from its mixed Ordinary source.");
                if (face == original_face)
                {
                    const auto& initial = original.at(key);
                    if (chosen.gas.normal != initial.gas.normal || chosen.gas.anchor != initial.gas.anchor ||
                        chosen.gas.offset != initial.gas.offset)
                        throw std::invalid_argument("Capillary original-plane reply changed its source geometry.");
                    continue;
                }
                std::size_t evaluations = 0;
                const auto evidence = admit_cell(data, original, key, chosen.gas, evaluations, budget);
                if (!evidence)
                    throw std::invalid_argument("Imported capillary plane fails its independent minority admission.");
                chosen.target_volume_residual = static_cast<double>(evidence->difference);
                chosen.finite_target_volume_difference = {evidence->difference, 0, mode};
                chosen.target_volume_numerical_error_estimate = evidence->numerical_error;

                result.at(key) = std::move(chosen);
            }
            if (seen.size() != requested.size())
                throw std::invalid_argument("Capillary plane owner replies are incomplete.");
        });
    return result;
}
} // namespace SimpleFluid::VoF::general_vof_detail
