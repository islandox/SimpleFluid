/** @file OrdinaryCurvature.cc
 * @brief Source-pinned connected capillary fits with keyed sparse owner exchange.
 */
#include "solvers/details/GeneralVoF/OrdinaryCurvature.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoFCurvature.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace SimpleFluid::detail::general_vof
{
namespace
{
using Key = VoF::GeneralVoFSurfaceKey;
using Patch = VoF::GeneralVoFSurfacePatch;
using Point = std::array<long double, 3>;
using LO = DefaultTpetraTypes::local_ordinal_type;
using GO = DefaultTpetraTypes::global_ordinal_type;
constexpr auto absent_cell = std::numeric_limits<std::uint64_t>::max();
constexpr long double geometry_tolerance = 64 * std::numeric_limits<double>::epsilon();
constexpr std::uint64_t packet_version = 0x5346435552560001ULL;
constexpr std::size_t stamp_words = 14, request_bytes = stamp_words * 8, response_bytes = request_bytes + 8;

struct Budget
{
    std::size_t limit = 0, used = 0;
    void add(std::size_t count, std::size_t size = 1)
    {
        if (size && count > (limit - used) / size)
            throw std::length_error("Ordinary capillary assembly exceeds its byte budget.");
        used += count * size;
    }
};
template<class Map> constexpr auto map_entry_bytes = sizeof(typename Map::value_type) + 4 * sizeof(void*);
[[noreturn]] void unresolved(VoF::GeneralVoFSurfaceResolutionReason reason, const char* message)
{
    throw VoF::GeneralVoFSurfaceResolutionError(reason, message);
}
Point wide(const PLIC::GeometryPoint& p)
{
    return {p[0], p[1], p[2]};
}
Point difference(Point a, Point b)
{
    for (unsigned axis = 0; axis < 3; ++axis)
        a[axis] -= b[axis];
    return a;
}
long double dot(Point a, Point b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point cross(Point a, Point b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Point unit(Point p)
{
    const auto magnitude = std::hypot(p[0], p[1], p[2]);
    if (!(magnitude > 0) || !std::isfinite(magnitude))
        unresolved(VoF::GeneralVoFSurfaceResolutionReason::DeficientFit, "Capillary patch normal is invalid.");
    for (auto& value : p)
        value /= magnitude;
    return p;
}
std::array<Point, 3> frame(const Patch& patch)
{
    const auto normal = unit(wide(patch.normal));
    unsigned axis = 0;
    for (unsigned a = 1; a < 3; ++a)
        if (std::abs(normal[a]) < std::abs(normal[axis]))
            axis = a;
    Point reference{};
    reference[axis] = 1;
    const auto tangent = unit(cross(reference, normal));
    return {tangent, cross(normal, tangent), normal};
}
GeneralVoFCurvaturePatch numerical_patch(const Patch& patch)
{
    GeneralVoFCurvaturePatch result;
    result.cell = patch.first_cell;
    result.centroid = patch.centroid;
    result.normal = patch.normal;
    result.area = patch.area;
    result.cell_scale = patch.cell_scale;
    result.physical_spans = patch.physical_spans;
    result.key = patch.key;
    return result;
}
// Use the quadratic kernel's tangential radius. Inspect every normal inside
// that radius: dropping a perpendicular stair face cannot prove planarity.
bool radius_admitted(const Patch& target, const Patch& sample, const std::array<Point, 3>& basis)
{
    const auto displacement = difference(wide(sample.centroid), wide(target.centroid));
    std::array<long double, 2> spans{};
    for (unsigned tangent = 0; tangent < 2; ++tangent)
        for (unsigned axis = 0; axis < 3; ++axis)
            spans[tangent] += std::abs(basis[tangent][axis]) * target.physical_spans[axis];
    const auto x = dot(displacement, basis[0]) / (2 * spans[0]);
    const auto y = dot(displacement, basis[1]) / (2 * spans[1]);
    return x * x + y * y <= 1;
}
std::optional<GeneralVoFCurvatureResult> represented_planar(
    const Patch& target, const VoF::GeneralVoFSurfaceCatalog& catalog, const std::vector<std::size_t>& selected)
{
    const auto basis = frame(target);
    const auto origin = wide(target.centroid);
    const auto scale =
        std::max({static_cast<long double>(target.cell_scale), static_cast<long double>(target.physical_spans[0]),
            static_cast<long double>(target.physical_spans[1]), static_cast<long double>(target.physical_spans[2])});
    std::optional<Point> first_direction;
    bool two_directions = false;
    GeneralVoFCurvatureResult result;
    long double residual = 0, area = 0;
    result.condition = 1;
    for (auto index : selected)
    {
        const auto& patch = catalog.patches[index];
        if (!radius_admitted(target, patch, basis))
            continue;
        const auto displacement = difference(wide(patch.centroid), origin);
        const auto normal_error = difference(unit(wide(patch.normal)), basis[2]);
        if (std::hypot(normal_error[0], normal_error[1], normal_error[2]) > geometry_tolerance ||
            std::abs(dot(displacement, basis[2])) > geometry_tolerance * scale)
            return {};
        for (const auto& vertex : patch.vertices)
            if (std::abs(dot(difference(wide(vertex), origin), basis[2])) > geometry_tolerance * scale)
                return {};
        const auto height = dot(displacement, basis[2]) / target.cell_scale;
        residual += static_cast<long double>(patch.area) * height * height;
        area += patch.area;
        ++result.samples;
        if (std::hypot(displacement[0], displacement[1], displacement[2]) <= geometry_tolerance * scale)
            continue;
        if (!first_direction)
            first_direction = displacement;
        else if (std::abs(dot(cross(*first_direction, displacement), basis[2])) > geometry_tolerance * scale * scale)
            two_directions = true;
    }
    if (!two_directions)
        return {};
    result.normalized_residual = static_cast<double>(std::sqrt(residual / area));
    if (!std::isfinite(result.normalized_residual) || result.normalized_residual > .15)
        return {};
    // Zero is the measured represented-plane curvature, not a deficient-fit fallback.
    return result;
}
void append_word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned a = 0; a < 8; ++a)
        bytes.push_back(char(value >> (8 * a)));
}
std::uint64_t read_word(const std::vector<char>& bytes, std::size_t offset)
{
    std::uint64_t result = 0;
    for (unsigned a = 0; a < 8; ++a)
        result |= std::uint64_t(static_cast<unsigned char>(bytes.at(offset + a))) << (8 * a);
    return result;
}
Key read_key(const std::vector<char>& bytes)
{
    const auto kind = read_word(bytes, 5 * 8);
    if (kind > static_cast<std::uint64_t>(VoF::GeneralVoFSurfaceKind::FilteredCell))
        throw std::runtime_error("Capillary owner packet has an invalid patch kind.");
    return {static_cast<VoF::GeneralVoFSurfaceKind>(kind), read_word(bytes, 6 * 8), read_word(bytes, 7 * 8)};
}
} // namespace

std::vector<double> assemble_ordinary_curvature(const MeshHandle<>& mesh, const GeneralVoFFlowWorkspace& workspace,
    const VoF::GeneralVoF& phases, const VoF::GeneralVoFSurfaceOptions& options,
    VoF::GeneralVoFCapillaryDiagnostics* diagnostics, std::size_t byte_budget)
{
    const Execution execution(mesh);
    const auto comm = mesh.owned_cell_map()->getComm();
    const auto rank = comm->getRank();
    const auto epoch = mesh.geometry_epoch();
    const auto control = phases.accepted_control();
    execution.collective(
        [&]
        {
            if (phases.arithmetic_mode() != ArithmeticMode::Ordinary || !phases.accepted_unchanged())
                throw std::runtime_error("Ordinary capillarity requires a current Ordinary source.");
        },
        "ordinary_capillary_source");
    const auto catalog = phases.capillary_surface(options);
    // Keep retained geometry/index/output and temporary fitting/communication in
    // disjoint arenas. Each sparse exchange also bounds its internal framing.
    Budget resident{byte_budget / 2};
    const auto scratch_limit = byte_budget - resident.limit;
    const auto packet_limit = scratch_limit / 3;
    std::map<Key, std::size_t> index;
    std::map<std::uint64_t, std::vector<std::size_t>> incident;
    struct CellStamp
    {
        int owner;
        std::uint64_t revision;
    };
    std::map<std::uint64_t, CellStamp> cells;
    std::vector<double> values, curvature;
    VoF::GeneralVoFCapillaryDiagnostics candidate;
    execution.collective(
        [&]
        {
            if (catalog.source_control != control || catalog.model != options.model)
                throw std::runtime_error("Capillary catalog has stale source/model metadata.");
            resident.add(catalog.patches.capacity(), sizeof(Patch));
            resident.add(catalog.components.capacity(), sizeof(VoF::GeneralVoFSurfaceComponent));
            resident.add(catalog.filtered_source_cells.capacity(), sizeof(std::uint64_t));
            for (const auto& patch : catalog.patches)
                resident.add(patch.vertices.capacity(), sizeof(PLIC::GeometryPoint));
            resident.add(catalog.adjacency.size(), map_entry_bytes<decltype(catalog.adjacency)>);
            for (const auto& [key, row] : catalog.adjacency)
                resident.add(row.capacity(), sizeof(Key));
            resident.add(catalog.patches.size(), map_entry_bytes<decltype(index)> + sizeof(double));
            resident.add(mesh.num_owned_cells(), sizeof(double));
            values.assign(catalog.patches.size(), std::numeric_limits<double>::quiet_NaN());
            curvature.assign(mesh.num_owned_cells(), 0);
            const auto metadata = phases.support_cells();
            Budget scratch{scratch_limit};
            scratch.add(metadata.capacity(), sizeof(VoF::GeneralVoFSupportCell));
            resident.add(metadata.size(), map_entry_bytes<decltype(cells)>);
            for (const auto& cell : metadata)
                if (!cells.emplace(cell.cell, CellStamp{cell.owner, cell.revision}).second)
                    throw std::runtime_error("Capillary source repeats support-cell metadata.");
            for (std::size_t p = 0; p < catalog.patches.size(); ++p)
            {
                const auto& patch = catalog.patches[p];
                if (!index.emplace(patch.key, p).second || patch.owner < 0 || patch.owner >= comm->getSize() ||
                    patch.first_cell > std::uint64_t(std::numeric_limits<GO>::max()) ||
                    cells.at(patch.first_cell).owner != patch.owner ||
                    ((patch.owner == rank) != (mesh.owned_cell_map()->getLocalElement(GO(patch.first_cell)) >= 0)))
                    throw std::runtime_error("Capillary patch key/owner differs from authenticated source metadata.");
                if (!(patch.area > 0) || !std::isfinite(patch.area) || !(patch.cell_scale > 0) ||
                    !std::isfinite(patch.cell_scale) || patch.vertices.size() < 3)
                    unresolved(VoF::GeneralVoFSurfaceResolutionReason::DeficientFit,
                        "Capillary patch lacks measurable physical surface geometry.");
                (void) unit(wide(patch.normal));
                for (const auto& vertex : patch.vertices)
                    for (double coordinate : vertex)
                        if (!std::isfinite(coordinate))
                            unresolved(VoF::GeneralVoFSurfaceResolutionReason::DeficientFit,
                                "Capillary patch has a nonfinite polygon vertex.");
                for (unsigned a = 0; a < 3; ++a)
                    if (!std::isfinite(patch.centroid[a]) || !(patch.physical_spans[a] > 0) ||
                        !std::isfinite(patch.physical_spans[a]))
                        unresolved(VoF::GeneralVoFSurfaceResolutionReason::DeficientFit,
                            "Capillary patch has invalid physical support scales.");
                const auto add_incident = [&](std::uint64_t cell)
                {
                    if (cell == absent_cell)
                        return;
                    if (!cells.contains(cell))
                        throw std::runtime_error("Capillary incidence names an unauthenticated support cell.");
                    auto found = incident.find(cell);
                    if (found == incident.end())
                    {
                        resident.add(1, map_entry_bytes<decltype(incident)>);
                        found = incident.emplace(cell, std::vector<std::size_t>{}).first;
                    }
                    // Reserve explicitly to bound vector capacity, rather than relying
                    // on uncharged geometric growth of incident-patch rows.
                    resident.add(2, sizeof(std::size_t));
                    found->second.reserve(found->second.size() + 1);
                    found->second.push_back(p);
                };
                add_incident(patch.first_cell);
                if (patch.second_cell != patch.first_cell)
                    add_incident(patch.second_cell);
            }
            if (catalog.adjacency.size() != index.size())
                throw std::runtime_error("Capillary adjacency does not cover its patch catalog.");
            for (const auto& [key, neighbors] : catalog.adjacency)
            {
                (void) index.at(key);
                for (std::size_t i = 0; i < neighbors.size(); ++i)
                {
                    (void) index.at(neighbors[i]);
                    const auto& reverse = catalog.adjacency.at(neighbors[i]);
                    if (neighbors[i] == key ||
                        std::find(neighbors.begin(), neighbors.begin() + i, neighbors[i]) != neighbors.begin() + i ||
                        std::find(reverse.begin(), reverse.end(), key) == reverse.end())
                        throw std::runtime_error("Capillary adjacency is not a unique symmetric patch graph.");
                }
            }
            candidate.model = options.model;
            if (diagnostics)
            {
                std::size_t owned_patches = 0, owned_components = 0;
                for (const auto& patch : catalog.patches)
                    owned_patches += patch.owner == rank;
                for (const auto& component : catalog.components)
                    owned_components += component.owner == rank;
                resident.add(owned_patches, sizeof(VoF::GeneralVoFSurfaceFitDiagnostic));
                resident.add(owned_components, sizeof(VoF::GeneralVoFSurfaceComponent));
                candidate.fits.reserve(owned_patches);
                candidate.components.reserve(owned_components);
                for (const auto& component : catalog.components)
                    if (component.owner == rank)
                        candidate.components.push_back(component);
            }
        },
        "ordinary_capillary_catalog");
    // The helper returns only graph-reachable patches incident to the requested
    // owned seed's authenticated cell support. It never connects by proximity.
    const auto connected = [&](std::uint64_t seed_cell, std::optional<std::size_t> seed_patch,
                               std::vector<unsigned char>& admitted, std::vector<unsigned char>& visited)
    {
        const auto support = phases.cell_support(seed_cell);
        Budget scratch{scratch_limit};
        scratch.add(support.capacity(), sizeof(std::uint64_t));
        scratch.add(catalog.patches.size(), 2 * sizeof(unsigned char) + sizeof(std::size_t));
        admitted.assign(catalog.patches.size(), 0);
        visited.assign(catalog.patches.size(), 0);
        for (auto cell : support)
            if (const auto found = incident.find(cell); found != incident.end())
                for (auto patch : found->second)
                    admitted[patch] = 1;
        std::vector<std::size_t> selected;
        selected.reserve(catalog.patches.size());
        if (!seed_patch)
            return selected;
        if (!admitted[*seed_patch])
            throw std::runtime_error("Capillary target patch is absent from its owned seed support.");
        selected.push_back(*seed_patch);
        visited[*seed_patch] = 1;
        for (std::size_t next = 0; next < selected.size(); ++next)
            for (const auto& key : catalog.adjacency.at(catalog.patches[selected[next]].key))
            {
                const auto neighbor = index.at(key);
                if (admitted[neighbor] && !visited[neighbor])
                {
                    visited[neighbor] = 1;
                    selected.push_back(neighbor);
                }
            }
        return selected;
    };
    execution.collective(
        [&]
        {
            for (std::size_t p = 0; p < catalog.patches.size(); ++p)
            {
                const auto& patch = catalog.patches[p];
                if (patch.owner != rank)
                    continue;
                std::vector<unsigned char> admitted, visited;
                auto selected = connected(patch.first_cell, p, admitted, visited);
                if (std::getenv("SIMPLEFLUID_CAPILLARY_REPLAY"))
                {
                    const auto basis = frame(patch);
                    std::size_t absent = 0, disconnected = 0, radius = 0, incoherent = 0, retained = 0;
                    for (std::size_t q = 0; q < catalog.patches.size(); ++q)
                        if (!admitted[q])
                            ++absent;
                        else if (!visited[q])
                            ++disconnected;
                        else if (!radius_admitted(patch, catalog.patches[q], basis))
                            ++radius;
                        else if (dot(unit(wide(catalog.patches[q].normal)), basis[2]) <
                                 GeneralVoFCurvatureOptions{}.minimum_normal_cosine)
                            ++incoherent;
                        else
                            ++retained;
                    std::cerr << "CAP_FIT control=" << control << " rank=" << rank << " cell=" << patch.first_cell
                              << " catalog=" << catalog.patches.size() << " absent=" << absent
                              << " disconnected=" << disconnected << " radius=" << radius
                              << " incoherent=" << incoherent << " retained=" << retained << " connected=";
                    for (auto q : selected)
                        std::cerr << catalog.patches[q].first_cell << ';';
                    std::cerr << '\n';
                }
                vof_failure::Scope context("ordinary_capillary_fit");
                context.record.cell = patch.first_cell;
                context.record.scale = patch.cell_scale;
                context.record.mark(vof_failure::Field::Cell);
                context.record.mark(vof_failure::Field::Scale);
                if (patch.key.kind == VoF::GeneralVoFSurfaceKind::AlignedFace)
                {
                    context.record.face = patch.key.identity;
                    context.record.mark(vof_failure::Field::Face);
                }
                if (patch.key.kind == VoF::GeneralVoFSurfaceKind::FilteredCell)
                {
                    context.record.construction = patch.key.ordinal;
                    context.record.mark(vof_failure::Field::Construction);
                }
                Budget scratch{scratch_limit};
                scratch.add(catalog.patches.size(), 2 * sizeof(unsigned char) + sizeof(std::size_t));
                // Includes the quadratic kernel's rows, QR matrix/reflector, keyed
                // duplicate index and bounded vector-capacity growth.
                scratch.add(selected.size(), 1024 + sizeof(GeneralVoFCurvaturePatch));
                std::sort(selected.begin(), selected.end(),
                    [&](auto a, auto b) { return catalog.patches[a].key < catalog.patches[b].key; });
                const auto result = [&]
                {
                    if (const auto planar = represented_planar(patch, catalog, selected))
                        return *planar;
                    const auto basis = frame(patch);
                    bool raw_binary = patch.key.kind == VoF::GeneralVoFSurfaceKind::AlignedFace;
                    for (auto sample : selected)
                        if (radius_admitted(patch, catalog.patches[sample], basis) &&
                            catalog.patches[sample].key.kind != VoF::GeneralVoFSurfaceKind::AlignedFace &&
                            dot(unit(wide(catalog.patches[sample].normal)), basis[2]) >=
                                GeneralVoFCurvatureOptions{}.minimum_normal_cosine)
                            raw_binary = false;
                    if (raw_binary)
                        unresolved(VoF::GeneralVoFSurfaceResolutionReason::DeficientFit,
                            "A nonplanar binary face surface requires FilteredBinary capillary geometry or "
                            "refinement.");
                    std::vector<GeneralVoFCurvaturePatch> stencil;
                    stencil.reserve(selected.size());
                    for (auto sample : selected)
                        stencil.push_back(numerical_patch(catalog.patches[sample]));
                    try
                    {
                        return GeneralVoFCurvature::fit(numerical_patch(patch), stencil);
                    }
                    catch (const std::runtime_error& error)
                    {
                        throw VoF::GeneralVoFSurfaceResolutionError(
                            VoF::GeneralVoFSurfaceResolutionReason::DeficientFit, error.what());
                    }
                }();
                if (std::getenv("SIMPLEFLUID_CAPILLARY_REPLAY"))
                    std::cerr << "CAP_RESULT control=" << control << " rank=" << rank << " cell=" << patch.first_cell
                              << " curvature=" << result.curvature << " samples=" << result.samples
                              << " condition=" << result.condition << " residual=" << result.normalized_residual
                              << '\n';
                values[p] = result.curvature;
                if (diagnostics)
                    candidate.fits.push_back(
                        {patch.key, result.curvature, result.condition, result.normalized_residual, result.samples});
            }
        },
        "ordinary_capillary_fit");
    const auto stamp = [&](const Patch& patch)
    {
        return std::array<std::uint64_t, stamp_words>{packet_version, control, epoch,
            static_cast<std::uint64_t>(options.model), std::bit_cast<std::uint64_t>(options.filter_width),
            static_cast<std::uint64_t>(patch.key.kind), patch.key.identity, patch.key.ordinal,
            std::uint64_t(patch.owner), patch.first_cell, cells.at(patch.first_cell).revision, patch.second_cell,
            patch.second_cell == absent_cell ? 0 : cells.at(patch.second_cell).revision, patch.component};
    };
    const auto validate_stamp = [&](const std::vector<char>& bytes, const Patch& patch)
    {
        const auto expected = stamp(patch);
        for (std::size_t word = 0; word < expected.size(); ++word)
            if (read_word(bytes, word * 8) != expected[word])
                throw std::runtime_error("Capillary owner packet source/epoch/model/key/revision disagreement.");
    };
    std::vector<OwnerPacket> packets;
    execution.collective(
        [&]
        {
            std::size_t count = 0;
            for (const auto& patch : catalog.patches)
                count += patch.owner != rank;
            Budget packet_budget{packet_limit};
            packet_budget.add(count, sizeof(OwnerPacket) + request_bytes);
            packets.reserve(count);
            for (const auto& patch : catalog.patches)
                if (patch.owner != rank)
                {
                    OwnerPacket packet;
                    packet.peer = patch.owner;
                    packet.bytes.reserve(request_bytes);
                    for (auto word : stamp(patch))
                        append_word(packet.bytes, word);
                    packets.push_back(std::move(packet));
                }
        },
        "ordinary_capillary_requests");
    auto requested = exchange_owner_packets(packets, comm, packet_limit);
    std::vector<OwnerPacket>().swap(packets);
    execution.collective(
        [&]
        {
            Budget packet_budget{packet_limit};
            packet_budget.add(requested.packets.size(), sizeof(OwnerPacket) + response_bytes);
            packets.reserve(requested.packets.size());
            // A requester may import a patch once. The sparse exchange already
            // authenticates sender ranks; this set additionally forbids replay.
            using RequestIndex = std::set<std::pair<int, Key>>;
            Budget scratch{packet_limit};
            scratch.add(requested.packets.size(), sizeof(RequestIndex::value_type) + 4 * sizeof(void*));
            RequestIndex seen;
            for (const auto& request : requested.packets)
            {
                if (request.bytes.size() != request_bytes)
                    throw std::runtime_error("Invalid capillary owner request width.");
                const auto key = read_key(request.bytes);
                const auto p = index.at(key);
                const auto& patch = catalog.patches[p];
                if (patch.owner != rank || request.peer == rank || !std::isfinite(values[p]) ||
                    !seen.emplace(request.peer, key).second)
                    throw std::runtime_error("Capillary request has a wrong owner, duplicate key or absent fit.");
                validate_stamp(request.bytes, patch);
                OwnerPacket response;
                response.peer = request.peer;
                response.bytes.reserve(response_bytes);
                response.bytes.insert(response.bytes.end(), request.bytes.begin(), request.bytes.end());
                append_word(response.bytes, std::bit_cast<std::uint64_t>(values[p]));
                packets.push_back(std::move(response));
            }
        },
        "ordinary_capillary_responses");
    std::vector<OwnerPacket>().swap(requested.packets);
    auto returned = exchange_owner_packets(packets, comm, packet_limit);
    std::vector<OwnerPacket>().swap(packets);
    execution.collective(
        [&]
        {
            for (const auto& response : returned.packets)
            {
                if (response.bytes.size() != response_bytes)
                    throw std::runtime_error("Invalid capillary owner response width.");
                const auto p = index.at(read_key(response.bytes));
                const auto& patch = catalog.patches[p];
                if (response.peer != patch.owner || patch.owner == rank || std::isfinite(values[p]))
                    throw std::runtime_error("Capillary publication has a wrong authority or duplicate patch key.");
                validate_stamp(response.bytes, patch);
                const auto value = std::bit_cast<double>(read_word(response.bytes, request_bytes));
                if (!std::isfinite(value))
                    throw std::runtime_error("Capillary owner published nonfinite curvature.");
                values[p] = value;
            }
            for (double value : values)
                if (!std::isfinite(value))
                    throw std::runtime_error("Capillary owner publication does not cover the local patch catalog.");
        },
        "ordinary_capillary_publication");
    std::vector<OwnerPacket>().swap(returned.packets);
    execution.collective([&] { resident.add(mesh.num_owned_cells(), 2 * sizeof(double) + sizeof(unsigned char)); });
    const auto fraction = phases.geometric_gas_fractions();
    const auto extension_required = workspace.capillary_extension_required(fraction, byte_budget - resident.used);
    execution.collective(
        [&]
        {
            const auto lease = mesh.acquire_execution_view();
            for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
            {
                const auto gid = std::uint64_t(mesh.cell_global_id(LO(c)));
                const auto own = incident.find(gid);
                std::vector<unsigned char> admitted, visited;
                if (own != incident.end())
                {
                    (void) connected(gid, own->second.front(), admitted, visited);
                    long double weighted = 0, area = 0;
                    for (auto p : own->second)
                    {
                        if (!visited[p])
                            unresolved(VoF::GeneralVoFSurfaceResolutionReason::AmbiguousTopology,
                                "One capillary cell contains disconnected surface sheets.");
                        weighted += static_cast<long double>(catalog.patches[p].area) * values[p];
                        area += catalog.patches[p].area;
                    }
                    curvature[c] = static_cast<double>(weighted / area);
                }
                else
                {
                    if (!extension_required[c])
                        continue;
                    // Select the nearest admitted connected sheet deterministically.
                    // Distinct equidistant sheets have no supported extension model.
                    (void) connected(gid, {}, admitted, visited);
                    const auto center = mesh.cell_centroid(LO(c));
                    const Point origin{center.x, center.y, center.z};
                    for (auto coordinate : origin)
                        if (!std::isfinite(coordinate))
                            throw std::runtime_error("Capillary extension has a nonfinite native cell centroid.");
                    long double nearest = std::numeric_limits<long double>::infinity();
                    std::optional<std::size_t> closest;
                    for (std::size_t p = 0; p < catalog.patches.size(); ++p)
                        if (admitted[p])
                        {
                            const auto displacement = difference(wide(catalog.patches[p].centroid), origin);
                            const auto distance = dot(displacement, displacement);
                            if (distance < nearest || (distance == nearest && closest &&
                                                          catalog.patches[p].key < catalog.patches[*closest].key))
                            {
                                nearest = distance;
                                closest = p;
                            }
                        }
                    if (!closest)
                        continue;
                    (void) connected(gid, closest, admitted, visited);
                    for (std::size_t p = 0; p < catalog.patches.size(); ++p)
                        if (admitted[p] && !visited[p])
                        {
                            const auto displacement = difference(wide(catalog.patches[p].centroid), origin);
                            const auto distance = dot(displacement, displacement);
                            const auto scale =
                                std::max(nearest, static_cast<long double>(catalog.patches[*closest].cell_scale) *
                                                      catalog.patches[*closest].cell_scale);
                            if (std::abs(distance - nearest) <= geometry_tolerance * scale)
                                unresolved(VoF::GeneralVoFSurfaceResolutionReason::AmbiguousTopology,
                                    "Pure-cell curvature extension is equidistant from disconnected surfaces.");
                        }
                    curvature[c] = values[*closest];
                }
                if (!std::isfinite(curvature[c]))
                    throw std::runtime_error("Capillary cell extension produced nonfinite curvature.");
            }
            if (!phases.accepted_unchanged() || phases.accepted_control() != control || mesh.geometry_epoch() != epoch)
                throw std::runtime_error("Capillary source/native geometry changed during curvature assembly.");
        },
        "ordinary_capillary_extension");
    if (diagnostics)
        *diagnostics = std::move(candidate);
    return curvature;
}
} // namespace SimpleFluid::detail::general_vof
