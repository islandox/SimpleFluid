/** @file GeneralVoFSolver.cc
 * @brief Transactional native phase/implicit-momentum/physical-flow composition.
 */
#include "solvers/GeneralVoFSolver.hh"
#include "FVM/ExactFaceTransfer.hh"
#include "FVM/ExactPhaseTransfer.hh"
#include "FVM/details/BoundedPhaseArithmetic.hh"
#include <cstring>
#include <tuple>
#include "geometry/ExactPlanarFace.hh"
#include "geometry/BoundedPlanarFace.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/ConservativeVoFTransport.hh"
#include "solvers/details/GeneralVoFCurvature.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "solvers/details/GeneralVoFFlowWorkspace.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
namespace SimpleFluid
{
namespace
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
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
std::shared_ptr<const MeshHandle<>> validated_phase_mesh(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    const auto mesh = phases->mesh();
    int local = phases->hierarchy()->geometry_unchanged(), global = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error("General VoF source geometry is stale before workspace construction.");
    return mesh;
}
std::vector<Vector> finite_spans(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    std::vector<Vector> result;
    if (phases->inventory_policy() == VoF::PhaseInventoryPolicy::Finite)
        for (const auto& geometry : phases->owned_finite_geometry())
        {
            const auto spans = geometry->axis_spans();
            result.push_back({spans[0].as_double(), spans[1].as_double(), spans[2].as_double()});
        }
    return result;
}
} // namespace
#include "solvers/details/GeneralVoFExactSolver.tcc"
struct GeneralVoFSolver::Impl
{
    struct State
    {
        std::shared_ptr<const VoF::GeneralVoF> phases;
        std::vector<double> fraction, density, face_velocity, pressure, curvature;
        std::vector<Vector> velocity;
        // Derived from the source-selected authority. Finite mode always
        // rebuilds from represented phases; experimental exact mode opts into
        // its separately retained authority, never a solver-side reinterpretation.
        std::vector<Meshes::ExactGeometryMeasure> physical_mass;
        std::vector<long double> finite_mass, finite_mass_error;
        detail::GeneralVoFFiniteProjectionProvenance finite_projection;
        detail::GeneralVoFProjectionProvenance projection_arithmetic;
        VoFStepDiagnostics diagnostics;
        std::array<GeneralVoFMomentumDiagnostics, 3> momentum_transport;
    };
    std::shared_ptr<const MeshHandle<>> mesh;
    VoFMaterial material;
    GeneralVoFSolverOptions options;
    detail::GeneralVoFFlowWorkspace flow;
    detail::ConservativeVoFTransport transport;
    std::vector<double> volume, thickness;
    std::shared_ptr<const State> accepted;
    void require(bool condition, const char* message) const
    {
        int local = condition, global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
        if (!global)
            throw std::runtime_error(message);
    }
    template<class Work> void collective(Work&& work) const
    {
        std::exception_ptr error;
        try
        {
            std::forward<Work>(work)();
        }
        catch (...)
        {
            error = std::current_exception();
        }
        int local = bool(error), global = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local, &global);
        if (global)
        {
            if (error)
                std::rethrow_exception(error);
            throw std::runtime_error("General native VoF rejected on another rank.");
        }
    }
    double maximum(double value) const
    {
        double result = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &value, &result);
        return result;
    }
    Impl(std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial m, GeneralVoFSolverOptions o)
        : mesh(validated_phase_mesh(phases)), material(m), options(o), flow(mesh, phases->inventory_policy() == VoF::PhaseInventoryPolicy::Finite, finite_spans(phases)), transport(mesh)
    {
        bool valid = options.byte_budget > 0 && options.maximum_momentum_iterations > 0 &&
                     std::isfinite(options.momentum_relative_tolerance) && options.momentum_relative_tolerance > 0;
        for (double value : {m.gas_density, m.liquid_density})
            valid = valid && std::isfinite(value) && value > 0;
        for (double value : {m.gas_viscosity, m.liquid_viscosity, m.surface_tension})
            valid = valid && std::isfinite(value) && value >= 0;
        for (double value : o.flow.gravity)
            valid = valid && std::isfinite(value);
        for (double value : {o.flow.max_courant, o.flow.capillary_safety, o.flow.viscous_safety,
                 o.flow.pressure_absolute_tolerance, o.flow.divergence_tolerance})
            valid = valid && std::isfinite(value) && value > 0;
        valid = valid && std::isfinite(o.flow.pressure_relative_tolerance) && o.flow.pressure_relative_tolerance >= 0 &&
                o.flow.max_pressure_iterations > 0;
        require(valid, "General VoF flow has invalid material or numerical controls.");
        std::array<std::uint64_t, 17> controls{};
        const std::array<double, 15> real_controls{m.gas_density, m.liquid_density, m.gas_viscosity, m.liquid_viscosity,
            m.surface_tension, o.flow.gravity[0], o.flow.gravity[1], o.flow.gravity[2], o.flow.max_courant,
            o.flow.capillary_safety, o.flow.viscous_safety, o.flow.pressure_relative_tolerance,
            o.flow.pressure_absolute_tolerance, o.flow.divergence_tolerance, o.momentum_relative_tolerance};
        for (std::size_t i = 0; i < real_controls.size(); ++i)
            controls[i] = std::bit_cast<std::uint64_t>(real_controls[i]);
        controls[15] = std::uint64_t(o.maximum_momentum_iterations);
        controls[16] = std::uint64_t(o.flow.max_pressure_iterations);
        auto agreed = controls;
        Teuchos::broadcast(
            *mesh->owned_cell_map()->getComm(), 0, int(sizeof(agreed)), reinterpret_cast<char*>(agreed.data()));
        require(controls == agreed, "General VoF material/numerical controls differ across ranks.");
        require(phases->hierarchy()->geometry_unchanged(), "General VoF source geometry is stale.");
        const auto geometry = phases->owned_geometry();
        const auto bounded_geometry = phases->inventory_policy() == VoF::PhaseInventoryPolicy::Finite ?
            phases->owned_finite_geometry() : std::vector<Meshes::BoundedConvexGeometryPtr>{};
        volume.resize(mesh->num_owned_cells());
        thickness.resize(volume.size());
        collective(
            [&]
            {
                if (geometry.size() != volume.size())
                    throw std::runtime_error("General VoF phase/native geometry order differs.");
                for (std::size_t c = 0; c < volume.size(); ++c)
                {
                    if (!bounded_geometry.empty())
                    {
                        const auto& p = bounded_geometry[c]->points();
                        long double area = 0;
                        for (const auto& face : bounded_geometry[c]->faces())
                        {
                            VoF::finite_detail::Point normal{};
                            for (std::size_t i = 1; i + 1 < face.size(); ++i)
                                normal = VoF::finite_detail::add(normal, VoF::finite_detail::cross(
                                    VoF::finite_detail::subtract(p[face[i]], p[face[0]]),
                                    VoF::finite_detail::subtract(p[face[i + 1]], p[face[0]])));
                            const auto square = VoF::finite_detail::dot(normal, normal);
                            area += std::sqrt(square.value) / 2;
                        }
                        const auto v = bounded_geometry[c]->volume();
                        volume[c] = v.as_double();
                        thickness[c] = static_cast<double>(2 * v.value / area);
                        if (!std::isfinite(thickness[c]) || !(thickness[c] > 0))
                            throw std::runtime_error("Finite VoF cell has no physical thickness.");
                        continue;
                    }
                    volume[c] = geometry[c]->moments().volume.as_double();
                    long double area = 0;
                    for (std::size_t face = 0; face < geometry[c]->num_faces(); ++face)
                        area += geometry[c]->face_geometry(face)->area().value;
                    thickness[c] = static_cast<double>(2 * geometry[c]->moments().volume.value / area);
                    if (!std::isfinite(thickness[c]) || thickness[c] <= 0)
                        throw std::runtime_error("General VoF cell has no positive physical thickness.");
                }
            });
        auto state = std::make_shared<State>();
        state->phases = std::move(phases);
        accepted = std::move(state);
    }
    bool finite(const State& state) const
    {
        return state.phases->inventory_policy() == VoF::PhaseInventoryPolicy::Finite;
    }
    std::vector<double> import(const std::vector<double>& owned, const Teuchos::RCP<const Pack::map_type>& target) const
    {
        require(owned.size() == volume.size(), "General VoF owned import dimensions differ.");
        Pack::vector_type from(mesh->owned_cell_map()), to(target);
        {
            auto values = from.getDataNonConst();
            std::copy(owned.begin(), owned.end(), values.begin());
        }
        Pack::import_type plan(mesh->owned_cell_map(), target);
        to.doImport(from, plan, Tpetra::INSERT);
        const auto values = to.getData();
        return {values.begin(), values.end()};
    }
    void surface_curvature(State& state) const
    {
        state.curvature.assign(volume.size(), 0);
        if (material.surface_tension == 0)
            return;
        const auto patches = state.phases->reconstruction_patches();
        std::map<GO, int> phase_geometry;
        for (const auto& patch : patches)
            phase_geometry.emplace(GO(patch.cell), patch.interface.pure_gas       ? 1
                                                   : patch.interface.pure_carrier ? -1
                                                                                  : 0);
        collective(
            [&]
            {
                const auto lease = mesh->acquire_execution_view();
                for (std::size_t c = 0; c < volume.size(); ++c)
                    for (auto face : mesh->faces(LO(c)))
                    {
                        if (mesh->is_geometry_exterior_face(face))
                            continue;
                        const auto a = phase_geometry.at(mesh->cell_global_id(LO(c)));
                        const auto b = phase_geometry.at(mesh->cell_global_id(mesh->opposite_cell(face, LO(c))));
                        // A binary interface still has capillary physics. Until a
                        // canonical pure-interface face fit is implemented, reject
                        // that unresolved surface rather than return zero curvature.
                        if (a * b == -1)
                            throw std::runtime_error("General VoF capillarity requires an admissible reconstructed "
                                                     "surface stencil at a pure-cell interface.");
                    }
            });
        std::vector<detail::GeneralVoFCurvaturePatch> surface;
        collective(
            [&]
            {
                for (const auto& patch : patches)
                {
                    const auto sample = finite(state) ? detail::GeneralVoFCurvature::patch(
                        patch.cell, *patch.finite_geometry, patch.interface) :
                        detail::Exact::GeneralVoFCurvature::patch(patch.cell, *patch.geometry, patch.interface);
                    if (sample) surface.push_back(*sample);
                }
                constexpr auto index_entry_bytes =
                    sizeof(std::pair<const std::uint64_t, std::size_t>) + 4 * sizeof(void*);
                if (surface.size() > options.byte_budget / index_entry_bytes)
                    throw std::runtime_error("General VoF curvature index exceeds its byte budget.");
                const auto index_bytes = surface.size() * index_entry_bytes;
                std::map<std::uint64_t, std::size_t> surface_index;
                for (std::size_t index = 0; index < surface.size(); ++index)
                    surface_index.emplace(surface[index].cell, index);
                for (const auto& patch : surface)
                {
                    const auto local = mesh->owned_cell_map()->getLocalElement(GO(patch.cell));
                    if (local >= 0)
                    {
                        const auto support = state.phases->cell_support(patch.cell);
                        if (support.size() > (options.byte_budget - index_bytes) / sizeof(std::size_t))
                            throw std::runtime_error("General VoF curvature support index exceeds its byte budget.");
                        std::vector<std::size_t> selected;
                        selected.reserve(support.size());
                        for (const auto cell : support)
                            if (const auto found = surface_index.find(cell); found != surface_index.end())
                                selected.push_back(found->second);
                        // Retain the original surface order: curvature fits sum
                        // floating-point stencil contributions in this order.
                        std::sort(selected.begin(), selected.end());
                        std::vector<detail::GeneralVoFCurvaturePatch> stencil;
                        stencil.reserve(selected.size());
                        for (const auto index : selected)
                            stencil.push_back(surface[index]);
                        state.curvature[local] = detail::GeneralVoFCurvature::fit(patch, stencil).curvature;
                    }
                }
            });
        Teuchos::Array<GO> ids;
        for (const auto& patch : surface)
            ids.push_back(GO(patch.cell));
        auto patch_map = Teuchos::rcp(new Pack::map_type(
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), GO{}, mesh->owned_cell_map()->getComm()));
        const auto k = import(state.curvature, patch_map);
        std::set<GO> mixed_cells;
        for (const auto& patch : surface)
            mixed_cells.insert(GO(patch.cell));
        // Curvature is fitted only at actual mixed PLIC sections. Extend that
        // fitted value to neighboring pure cells for the balanced face force;
        // no manufactured normal or pure-cell curvature fit is substituted.
        collective(
            [&]
            {
                for (std::size_t c = 0; c < volume.size(); ++c)
                {
                    const auto gid = mesh->cell_global_id(LO(c));
                    if (mixed_cells.contains(gid))
                        continue;
                    const auto support = state.phases->cell_support(std::uint64_t(gid));
                    const std::set<std::uint64_t> admitted(support.begin(), support.end());
                    const auto center = mesh->cell_centroid(LO(c));
                    double nearest = std::numeric_limits<double>::infinity();
                    for (std::size_t p = 0; p < surface.size(); ++p)
                    {
                        if (!admitted.contains(surface[p].cell))
                            continue;
                        const auto& point = surface[p].centroid;
                        const double dx = point[0] - center.x, dy = point[1] - center.y, dz = point[2] - center.z;
                        const double distance = dx * dx + dy * dy + dz * dz;
                        if (distance < nearest)
                        {
                            nearest = distance;
                            state.curvature[c] = k[p];
                        }
                    }
                }
            });
    }
    std::vector<VoF::GeneralVoFNodeVelocity> node_velocity(const State& state) const
    {
        const auto comm = mesh->owned_cell_map()->getComm();
        const auto metadata = state.phases->support_cells();
        std::map<std::uint64_t, VoF::GeneralVoFSupportCell> cells;
        for (const auto& cell : metadata)
            cells.emplace(cell.cell, cell);
        std::map<std::uint64_t, std::vector<std::uint64_t>> supports;
        std::set<GO> donor_ids;
        collective(
            [&]
            {
                for (const auto& node : state.phases->nodes())
                {
                    auto ids = state.phases->node_cells(node.gid);
                    if (ids.empty())
                        throw std::runtime_error(
                            "General VoF trajectory node has no authenticated owned-incident support.");
                    for (auto gid : ids)
                    {
                        if (gid > std::uint64_t(std::numeric_limits<GO>::max()) || !cells.contains(gid))
                            throw std::runtime_error(
                                "General VoF node support metadata is absent or out of native range.");
                        donor_ids.insert(GO(gid));
                    }
                    supports.emplace(node.gid, std::move(ids));
                }
            });
        Teuchos::Array<GO> ids(donor_ids.begin(), donor_ids.end());
        auto target = Teuchos::rcp(
            new Pack::map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), GO{}, comm));
        auto marker = import(std::vector<double>(volume.size(), 1), target);
        require(std::all_of(marker.begin(), marker.end(), [](double value) { return value == 1; }),
            "General VoF node support references an absent donor.");
        Teuchos::Array<int> owners(ids.size(), -1);
        (void) mesh->owned_cell_map()->getRemoteIndexList(ids(), owners());
        bool owner_valid = true;
        for (std::size_t i = 0; i < static_cast<std::size_t>(ids.size()); ++i)
            owner_valid = owner_valid && owners[i] == cells.at(std::uint64_t(ids[i])).owner;
        require(owner_valid, "General VoF node support owner differs from its native map.");
        std::array<std::vector<double>, 3> velocity;
        for (unsigned a = 0; a < 3; ++a)
        {
            std::vector<double> component(volume.size());
            for (std::size_t c = 0; c < volume.size(); ++c)
                component[c] = state.velocity[c][a];
            velocity[a] = import(component, target);
        }
        // Relays contain source-owner imported values and authority-certified
        // revisions/metrics. The node authority unions all OWNED-incident seed
        // supports, so ghost placement and requester rank never choose a fit.
        std::vector<OwnerPacket> packets;
        collective(
            [&]
            {
                std::size_t used = 0;
                for (const auto& node : state.phases->nodes())
                {
                    OwnerPacket packet;
                    packet.peer = int(node.gid % comm->getSize());
                    append_word(packet.bytes, node.gid);
                    const auto& node_identity = finite(state) ? node.identity : node.exact_identity;
                    append_word(packet.bytes, node_identity.size());
                    packet.bytes.insert(packet.bytes.end(), node_identity.begin(), node_identity.end());
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
                    for (const auto index : state.phases->node_tiles(node.gid))
                    {
                        const auto& tile = state.phases->tiles()[index];
                        if (tile.exterior)
                            walls.push_back(&tile);
                    }
                    append_word(packet.bytes, walls.size());
                    for (const auto* wall : walls)
                    {
                        append_word(packet.bytes, wall->face_gid);
                        Vector normal{};
                        std::vector<char> certificate;
                        if (finite(state))
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
                    if (packet.bytes.size() > options.byte_budget - used ||
                        sizeof(OwnerPacket) > options.byte_budget - used - packet.bytes.size())
                        throw std::runtime_error("General VoF node union exceeds its packet budget.");
                    used += sizeof(OwnerPacket) + packet.bytes.size();
                    packets.push_back(std::move(packet));
                }
            });
        auto gathered = exchange_owner_packets(packets, comm, options.byte_budget);
        packets.clear();
        struct NodeUnion
        {
            std::vector<char> identity, coordinate_bytes;
            Vector coordinate{};
            std::map<std::vector<char>, std::vector<char>> cells, walls;
            std::set<int> participants;
        };
        std::map<std::uint64_t, NodeUnion> nodes;
        collective(
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
                    auto identity = bytes(read()), coordinate = bytes(24);
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
                                row[1 + a] =
                                    (static_cast<long double>(sample.center[a]) - node.coordinate[a]) / span[a];
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
        auto selected = exchange_owner_packets(packets, comm, options.byte_budget);
        std::vector<VoF::GeneralVoFNodeVelocity> result;
        collective(
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
                if (values.size() != state.phases->nodes().size())
                    throw std::runtime_error("General VoF node publication is incomplete.");
                for (const auto& node : state.phases->nodes())
                    result.push_back({node.gid, values.at(node.gid)});
            });
        return result;
    }
    double stable_timestep(const State& state, const std::vector<VoF::GeneralVoFNodeVelocity>& nodes) const
    {
        const auto bound =
            flow.stable_timestep(state.face_velocity, std::min(material.gas_density, material.liquid_density),
                std::max(material.gas_viscosity, material.liquid_viscosity), material.surface_tension, options.flow);
        double local_thickness = std::numeric_limits<double>::infinity(), global_thickness = 0;
        for (double h : thickness)
            local_thickness = std::min(local_thickness, h);
        Teuchos::reduceAll(
            *mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_thickness, &global_thickness);
        double speed = 0;
        for (const auto& node : nodes)
            speed = std::max(speed, std::hypot(node.velocity[0], node.velocity[1], node.velocity[2]));
        if (speed > 0 && std::isfinite(speed))
            speed = std::nextafter(speed, std::numeric_limits<double>::infinity());
        speed = maximum(speed);
        if (!finite(state))
            return Exact::GeneralVoFSolver(*this).stable_timestep(state, bound, speed, global_thickness);
        namespace fp = VoF::finite_detail;
        const auto rates = finite_rates(state);
        const auto geometry = state.phases->owned_finite_geometry();
        double local_rate = std::numeric_limits<double>::infinity();
        collective([&]
        {
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                fp::Sum sum;
                for (const auto face : mesh->faces(LO(c)))
                    sum.add({std::abs(rates[face].value), rates[face].absolute_error});
                const auto outgoing = sum.value();
                if (outgoing.value + outgoing.absolute_error > 0)
                {
                    const auto lower_volume = geometry[c]->volume().value - geometry[c]->volume().absolute_error;
                    const auto time = lower_volume * options.flow.max_courant / (outgoing.value + outgoing.absolute_error);
                    double value = static_cast<double>(time);
                    if (static_cast<long double>(value) > time) value = std::nextafter(value, 0.);
                    if (!(value > 0) || std::isnan(value))
                        throw std::runtime_error("Finite VoF canonical rates have no positive timestep.");
                    local_rate = std::min(local_rate, value);
                }
            }
        });
        double rate = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_rate, &rate);
        return std::min(rate, speed > 0 ? std::min(bound, options.flow.max_courant * global_thickness / speed) : bound);
    }
    void initialize(const std::vector<Vector>& owned_velocity, const std::vector<double>& face_velocity)
    {
        // Empty owners still participate in the same scalar/face imports. Local
        // vector emptiness cannot select whether a field was supplied globally.
        const int local_fields[2] = {!owned_velocity.empty(), !face_velocity.empty()};
        int supplied_fields[2] = {};
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 2, local_fields, supplied_fields);
        const bool supplied_velocity = supplied_fields[0], supplied_faces = supplied_fields[1];
        require((!supplied_velocity || owned_velocity.size() == volume.size()) &&
                    (!supplied_faces || face_velocity.size() == mesh->num_faces()),
            "General VoF initialization supplied field shape mismatch.");
        auto next = std::make_shared<State>();
        next->phases = accepted->phases;
        next->velocity = supplied_velocity ? owned_velocity : std::vector<Vector>(volume.size());
        if (!supplied_velocity && supplied_faces)
        {
            const auto canonical = flow.faces(face_velocity);
            collective(
                [&]
                {
                    const auto lease = mesh->acquire_execution_view();
                    for (std::size_t c = 0; c < volume.size(); ++c)
                        for (auto face : mesh->faces(LO(c)))
                        {
                            const auto offset = mesh->face_center_vector(face, LO(c));
                            const double amount = (mesh->owner_cell(face) == LO(c) ? 1 : -1) * mesh->face_area(face) *
                                                  canonical[face] / volume[c];
                            next->velocity[c][0] += amount * offset.x;
                            next->velocity[c][1] += amount * offset.y;
                            next->velocity[c][2] += amount * offset.z;
                        }
                });
        }
        bool valid = next->velocity.size() == volume.size();
        for (const auto& velocity : next->velocity)
            for (double value : velocity)
                valid = valid && std::isfinite(value);
        require(valid, "General VoF initialization requires finite owned cell velocities.");
        phase_storage(*next);
        surface_curvature(*next);
        next->face_velocity = supplied_faces ? flow.faces(face_velocity) : flow.interpolate(next->velocity);
        next->projection_arithmetic.byte_budget = options.byte_budget;
        if (finite(*next))
        {
            const auto closure_topology = projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*next->phases);
            next->finite_projection.byte_budget = options.byte_budget;
            flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction,
                next->curvature, 1, 0, options.flow, next->diagnostics, &next->finite_projection,
                std::span<const long double>(next->finite_mass), &closure_topology,
                std::span<const long double>(next->finite_mass_error));
        }
        else
        {
            const auto closure_topology = projection_topology(*next->phases);
            detail::Exact::GeneralVoFFlowWorkspace::project(flow, next->face_velocity, next->velocity, next->pressure,
                next->density, next->fraction, next->curvature, 1, 0, options.flow, next->diagnostics,
                &next->projection_arithmetic, std::span<const Meshes::ExactGeometryMeasure>(next->physical_mass),
                &closure_topology);
        }
        budget(*next);
        accepted = std::move(next);
    }
    template<class Topology = detail::Exact::GeneralVoFProjectionClosureTopology>
    Topology projection_topology(const VoF::GeneralVoF& phases) const
    {
        Topology result;
        collective(
            [&]
            {
                std::map<GO, const Meshes::AMR::AdaptiveTile*> tiles;
                for (const auto& tile : phases.tiles())
                    tiles.emplace(GO(tile.face_gid), &tile);
                const auto lease = mesh->acquire_execution_view();
                for (std::size_t c = 0; c < mesh->num_owned_cells(); ++c)
                    result.cells.push_back(
                        {mesh->cell_global_id(LO(c)), Meshes::AMR::encode_adaptive_key(phases.owned_order()[c]), {}});
                for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
                {
                    const auto face = LO(f);
                    if (mesh->is_geometry_exterior_face(face))
                        continue;
                    const auto& tile = *tiles.at(mesh->face_global_id(face));
                    std::vector<char> identity;
                    for (const auto* bytes : {&tile.domain, &tile.surface, &tile.identity})
                    {
                        append_word(identity, bytes->size());
                        identity.insert(identity.end(), bytes->begin(), bytes->end());
                    }
                    result.faces.push_back({mesh->face_global_id(face), mesh->cell_global_id(mesh->owner_cell(face)),
                        mesh->cell_global_id(mesh->neighbor_cell(face)), std::move(identity), {}, {}});
                }
            });
        return result;
    }
    void phase_storage(State& state) const
    {
        if (!finite(state))
        {
            Exact::GeneralVoFSolver(*this).phase_storage(state);
            return;
        }
        namespace fp = VoF::finite_detail;
        const auto& phases = state.phases->phase_volumes();
        const auto geometry = state.phases->owned_finite_geometry();
        const auto fraction = state.phases->geometric_gas_fractions();
        collective([&]
        {
            if (phases.size() != volume.size() || geometry.size() != volume.size())
                throw std::runtime_error("Finite VoF phase/native dimensions differ.");
            state.fraction = fraction;
            state.density.resize(volume.size());
            state.finite_mass.resize(volume.size());
            state.finite_mass_error.resize(volume.size());
            state.physical_mass.clear();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto mass = fp::add(fp::multiply({phases[c].gas, 0}, {material.gas_density, 0}),
                    fp::multiply({phases[c].carrier, 0}, {material.liquid_density, 0}));
                const auto density = fp::divide(mass, geometry[c]->volume());
                state.finite_mass[c] = mass.value;
                state.finite_mass_error[c] = mass.absolute_error;
                state.density[c] = density.as_double();
                if (!(mass.value > mass.absolute_error) || !std::isfinite(state.density[c]) ||
                    !(state.density[c] > 0) || state.fraction[c] < 0 || state.fraction[c] > 1)
                    throw std::runtime_error("Finite VoF physical mixture storage is unresolved.");
            }
        });
    }
    std::vector<Meshes::BoundedGeometryValue> finite_rates(const State& state) const
    {
        require(state.finite_projection.face_rate.size() == mesh->num_owned_faces() &&
            state.finite_projection.face_radius.size() == mesh->num_owned_faces(),
            "Finite VoF projection rate authority is incomplete.");
        std::array<std::vector<double>, 6> wire;
        collective([&]
        {
            for (auto& v : wire) v.resize(mesh->num_faces());
            for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
                for (unsigned k = 0; k < 2; ++k)
                {
                    const auto value = k == 0 ? state.finite_projection.face_rate[f] : state.finite_projection.face_radius[f];
                    int exponent = 0;
                    const auto mantissa = std::frexp(value, &exponent);
                    const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
                    if (!std::isfinite(value) || (k && value < 0) || static_cast<long double>(high) + low != mantissa)
                        throw std::runtime_error("Finite VoF rate exceeds fixed wire precision.");
                    wire[3 * k][f] = high;
                    wire[3 * k + 1][f] = low;
                    wire[3 * k + 2][f] = exponent;
                }
        });
        for (auto& value : wire) value = flow.faces(value);
        std::vector<Meshes::BoundedGeometryValue> result(mesh->num_faces());
        collective([&]
        {
            for (std::size_t f = 0; f < result.size(); ++f)
            {
                long double values[2]{};
                for (unsigned k = 0; k < 2; ++k)
                {
                    const auto exponent = wire[3 * k + 2][f];
                    if (!std::isfinite(exponent) || exponent < std::numeric_limits<int>::min() ||
                        exponent > std::numeric_limits<int>::max() || exponent != std::trunc(exponent))
                        throw std::runtime_error("Finite VoF canonical wire exponent is malformed.");
                    values[k] = std::scalbn(static_cast<long double>(wire[3 * k][f]) + wire[3 * k + 1][f], int(exponent));
                    if (!std::isfinite(values[k]) || (k && values[k] < 0))
                        throw std::runtime_error("Finite VoF canonical wire rate is nonfinite.");
                }
                result[f] = {values[0], values[1]};
            }
        });
        return result;
    }
    std::vector<VoF::GeneralVoFFaceVolume> finite_face_volumes(const State& state, double dt) const
    {
        const auto rates = finite_rates(state);
        std::vector<VoF::GeneralVoFFaceVolume> result;
        collective([&]
        {
            for (const auto& tile : state.phases->tiles())
            {
                const auto local = mesh->overlap_face_map()->getLocalElement(GO(tile.face_gid));
                if (local < 0) throw std::runtime_error("Finite VoF missing canonical face rate.");
                const auto integrated = VoF::finite_detail::multiply(rates[local], {dt, 0});
                const auto diagnostic = static_cast<double>(integrated.value);
                if (!std::isfinite(diagnostic)) throw std::runtime_error("Finite VoF face diagnostic overflows.");
                result.push_back({tile.face_gid, diagnostic, std::nullopt, integrated});
            }
        });
        return result;
    }
    void verify_phase_publication(const VoF::GeneralVoF::Trial& trial, const VoF::GeneralVoF& candidate) const
    {
        if (candidate.inventory_policy() != VoF::PhaseInventoryPolicy::Finite)
        {
            Exact::GeneralVoFSolver(*this).verify_phase_publication(trial, candidate);
            return;
        }
        namespace fp = VoF::finite_detail;
        collective([&]
        {
            if (candidate.inventory_policy() != accepted->phases->inventory_policy())
                throw std::runtime_error("Finite VoF candidate changed inventory policy.");
            if (candidate.accepted_identity() == accepted->phases->accepted_identity())
                return;
            std::vector<fp::Sum> gas(volume.size()), carrier(volume.size());
            for (const auto& route : trial.finite_transfers())
            {
                const auto row = mesh->owned_cell_map()->getLocalElement(GO(route.recipient));
                if (row < 0) throw std::runtime_error("Finite phase publication has foreign recipient.");
                gas[row].add(route.amount.gas);
                carrier[row].add(route.amount.carrier);
            }
            const auto& phases = candidate.phase_volumes();
            const auto& bounds = trial.phase_roundoff_bounds();
            if (phases.size() != volume.size() || bounds.size() != volume.size())
                throw std::runtime_error("Finite phase publication has incompatible shape.");
            for (std::size_t c = 0; c < volume.size(); ++c)
                for (const auto& item : {std::tuple{phases[c].gas, bounds[c].gas, gas[c].value()},
                         std::tuple{phases[c].carrier, bounds[c].carrier, carrier[c].value()}})
                {
                    const auto expected = std::get<2>(item);
                    const auto difference = std::abs(static_cast<long double>(std::get<0>(item)) - expected.value);
                    if (!std::isfinite(std::get<1>(item)) || std::get<1>(item) < 0 ||
                        difference + expected.absolute_error > std::get<1>(item))
                        throw std::runtime_error("Finite phase publication exceeds its own bounded ledger.");
                }
        });
    }
    void budget(State& state) const
    {
        if (!finite(state))
        {
            Exact::GeneralVoFSolver(*this).budget(state);
            return;
        }
        namespace fp = VoF::finite_detail;
        std::array<fp::Sum, 7> sums;
        collective([&]
        {
            const auto& phases = state.phases->phase_volumes();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                sums[0].add({phases[c].gas, 0});
                sums[1].add(fp::multiply({phases[c].gas, 0}, {material.gas_density, 0}));
                sums[2].add(fp::multiply({phases[c].carrier, 0}, {material.liquid_density, 0}));
                for (unsigned a = 0; a < 3; ++a)
                    sums[3 + a].add(fp::multiply({state.finite_mass[c], 0}, {state.velocity[c][a], 0}));
                const auto square = static_cast<long double>(state.velocity[c][0]) * state.velocity[c][0] +
                    static_cast<long double>(state.velocity[c][1]) * state.velocity[c][1] +
                    static_cast<long double>(state.velocity[c][2]) * state.velocity[c][2];
                sums[6].add(fp::multiply({state.finite_mass[c], 0}, {square / 2, 0}));
            }
        });
        std::array<double, 7> global{};
        for (unsigned a = 0; a < 7; ++a)
        {
            const auto local = sums[a].value();
            int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
            if (local.value != 0) (void) std::frexp(std::abs(local.value), &local_exponent);
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
            const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
            const auto normalized = static_cast<double>(local.value / scale);
            double sum = 0;
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 1, &normalized, &sum);
            collective([&]
            {
                const auto value = static_cast<long double>(sum) * scale;
                global[a] = static_cast<double>(value);
                if (!std::isfinite(global[a]) || (value != 0 && global[a] == 0))
                    throw std::runtime_error("Finite VoF global physical diagnostic is outside representable range.");
            });
        }
        state.diagnostics.gas_volume = global[0];
        state.diagnostics.gas_mass = global[1];
        state.diagnostics.liquid_mass = global[2];
        for (unsigned a = 0; a < 3; ++a) state.diagnostics.momentum[a] = global[3 + a];
        state.diagnostics.kinetic_energy = global[6];
        double speed = 0;
        for (const auto& velocity : state.velocity) speed = std::max(speed, std::sqrt(dot(velocity, velocity)));
        state.diagnostics.maximum_velocity = maximum(speed);
    }
    VoFStepDiagnostics step(double dt)
    {
        if (!finite(*accepted))
            return Exact::GeneralVoFSolver(*this).step(dt);
        namespace fp = VoF::finite_detail;
        require(std::isfinite(dt) && dt > 0, "Finite VoF timestep must be positive and finite.");
        double minimum = 0, maximum_dt = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &dt, &minimum);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &dt, &maximum_dt);
        require(minimum == maximum_dt, "Finite VoF timestep differs across ranks.");
        const auto& old = *accepted;
        const auto nodes = node_velocity(old);
        require(dt <= stable_timestep(old, nodes), "Finite VoF timestep exceeds physical geometry/trajectory CFL.");
        const auto trial = old.phases->plan_transport(nodes, finite_face_volumes(old, dt), dt);
        double local_h = std::numeric_limits<double>::infinity(), global_h = 0;
        for (auto h : thickness) local_h = std::min(local_h, h);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_h, &global_h);
        collective([&]
        {
            if (trial.finite_node_trajectories().size() != nodes.size())
                throw std::runtime_error("Finite VoF trajectory certificate is incomplete.");
            const auto limit = fp::multiply({global_h, 0}, {options.flow.max_courant, 0});
            const auto square_limit = fp::multiply(limit, limit);
            for (const auto& node : trial.finite_node_trajectories())
            {
                const auto displacement = fp::subtract(node.point, node.departure);
                const auto square = fp::dot(displacement, displacement);
                if (square.value + square.absolute_error > square_limit.value - square_limit.absolute_error)
                    throw std::runtime_error("Finite VoF bounded trajectory exceeds physical displacement CFL.");
            }
        });
        auto next = std::make_shared<State>();
        next->phases = trial.candidate();
        require(next->phases->mesh().get() == mesh.get(), "Finite VoF transport changed its immutable native mesh.");
        verify_phase_publication(trial, *next->phases);
        phase_storage(*next);
        std::vector<long double> outgoing(volume.size()), outgoing_error(volume.size()), source(volume.size());
        std::vector<detail::ConservativeVoFTransport::FiniteTransfer> incoming;
        std::vector<detail::ConservativeVoFTransport::FiniteFaceContribution> face_contributions;
        const auto mass = [&](const Meshes::BoundedPhaseMeasures& p)
        {
            return fp::add(fp::multiply(p.gas, {material.gas_density, 0}),
                fp::multiply(p.carrier, {material.liquid_density, 0}));
        };
        collective([&]
        {
            std::set<GO> seen;
            for (const auto& donor : trial.finite_donor_budgets())
            {
                const auto row = mesh->owned_cell_map()->getLocalElement(GO(donor.donor));
                if (row < 0 || !seen.insert(GO(donor.donor)).second)
                    throw std::runtime_error("Finite VoF donor budget is duplicate or foreign.");
                const auto exported_mass = mass(donor.exported);
                outgoing[row] = exported_mass.value;
                outgoing_error[row] = exported_mass.absolute_error;
            }
            if (seen.size() != volume.size()) throw std::runtime_error("Finite VoF donor budgets are incomplete.");
            for (const auto& route : trial.finite_transfers())
                if (route.donor != route.recipient)
                {
                    const auto value = mass(route.amount);
                    incoming.push_back({GO(route.donor), GO(route.recipient), value.value, value.absolute_error});
                }
            for (const auto& face : trial.finite_face_contributions())
            {
                const auto row = mesh->owned_face_map()->getLocalElement(GO(face.face));
                if (row < 0 || (face.direction != 1 && face.direction != -1))
                    throw std::runtime_error("Finite VoF raw stream is not a canonical owned face.");
                const auto value = mass(face.amount);
                face_contributions.push_back({row, GO(face.donor), face.direction * value.value, value.absolute_error});
            }
        });
        next->velocity.resize(volume.size());
        for (unsigned a = 0; a < 3; ++a)
        {
            std::vector<long double> momentum(volume.size()), momentum_error(volume.size());
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto value = fp::multiply({old.finite_mass[c], old.finite_mass_error[c]}, {old.velocity[c][a], 0});
                momentum[c] = value.value;
                momentum_error[c] = value.absolute_error;
            }
            const auto carried = transport.solve(momentum, source, next->finite_mass, outgoing, incoming,
                face_contributions, false, options.maximum_momentum_iterations,
                options.momentum_relative_tolerance, options.byte_budget, momentum_error, {}, next->finite_mass_error,
                outgoing_error);
            next->momentum_transport[a] = {carried.physical_relative_residual, carried.iterations, carried.physical_refinements};
            for (std::size_t c = 0; c < volume.size(); ++c) next->velocity[c][a] = carried.cell_trace[c];
        }
        std::vector<double> viscosity(volume.size());
        collective([&]
        {
            const auto& phases = old.phases->phase_volumes();
            const auto geometry = old.phases->owned_finite_geometry();
            for (std::size_t c = 0; c < volume.size(); ++c)
            {
                const auto weighted = fp::add(fp::multiply({phases[c].gas, 0}, {material.gas_viscosity, 0}),
                    fp::multiply({phases[c].carrier, 0}, {material.liquid_viscosity, 0}));
                viscosity[c] = fp::divide(weighted, geometry[c]->volume()).as_double();
            }
        });
        const auto viscous = flow.viscous_force(old.velocity, viscosity);
        std::vector<Vector> change(volume.size());
        collective([&]
        {
            for (std::size_t c = 0; c < volume.size(); ++c)
                for (unsigned a = 0; a < 3; ++a)
                {
                    const auto force = fp::divide(fp::multiply({viscous[c][a], 0}, {dt, 0}), {next->finite_mass[c], next->finite_mass_error[c]});
                    const auto updated = fp::add(fp::add({next->velocity[c][a], 0},
                        fp::multiply({options.flow.gravity[a], 0}, {dt, 0})), force);
                    next->velocity[c][a] = updated.as_double();
                    change[c][a] = next->velocity[c][a] - old.velocity[c][a];
                }
        });
        next->face_velocity = old.face_velocity;
        const auto face_change = flow.interpolate(change);
        for (std::size_t f = 0; f < next->face_velocity.size(); ++f) next->face_velocity[f] += face_change[f];
        surface_curvature(*next);
        next->finite_projection.byte_budget = options.byte_budget;
        const auto closure_topology = projection_topology<detail::GeneralVoFFiniteProjectionClosureTopology>(*next->phases);
        flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction,
            next->curvature, dt, material.surface_tension, options.flow, next->diagnostics, &next->finite_projection,
            std::span<const long double>(next->finite_mass), &closure_topology,
            std::span<const long double>(next->finite_mass_error));
        next->diagnostics.time = old.diagnostics.time + dt;
        require(std::isfinite(next->diagnostics.time) && next->diagnostics.time > old.diagnostics.time,
            "Finite VoF timestep did not advance the accepted clock.");
        budget(*next);
        next->phases = old.phases->apply(trial);
        accepted = std::move(next);
        return accepted->diagnostics;
    }
    double handoff_finite_adapted_faces(const State& old, Impl& target, State& state,
        const VoF::GeneralVoF::FiniteRemapTrial& trial) const
    {
        namespace fp = VoF::finite_detail;
        using Point = Meshes::BoundedPoint;
        using Value = Meshes::BoundedGeometryValue;
        using Key = VoF::GeneralVoF::Key;
        using Loop = std::vector<Point>;
        using Support = Meshes::BoundedPlaneAuthority;
        const auto comm = mesh->owned_cell_map()->getComm();
        const auto& common = trial.source_common_cells();
        std::vector<Key> requested;
        std::map<Key, std::vector<std::size_t>> columns;
        for (std::size_t i = 0; i < common.size(); ++i)
        {
            columns[common[i].source_key].push_back(i);
            requested.push_back(common[i].target_key);
        }
        const auto draft = Meshes::AMR::fetch_adaptive_native_draft(*state.phases->hierarchy(), requested);
        std::map<Key, Meshes::ConvexGeometryPtr> native_sources;
        std::map<std::uint64_t, std::set<Key>> native_fans;
        collective([&]
        {
            for (const auto* leaves : {&draft.owned, &draft.ghosts})
                for (const auto& leaf : *leaves)
                    native_sources.emplace(leaf.key, state.phases->hierarchy()->geometry(leaf));
            for (const auto& tile : draft.tiles)
                for (auto node : tile.node_gids)
                    for (const auto* key : {&tile.first, tile.second ? &*tile.second : nullptr})
                        if (key && native_sources.contains(*key)) native_fans[node].insert(*key);
        });
        std::map<std::uint64_t, Point> native_points;
        collective([&]
        {
            for (const auto& node : draft.nodes)
            {
                const auto fan = native_fans.find(node.gid);
                if (fan == native_fans.end() || fan->second.empty()) continue;
                std::vector<Meshes::ConvexGeometryPtr> cells;
                for (const auto& key : fan->second) cells.push_back(native_sources.at(key));
                native_points.emplace(node.gid, Meshes::bounded_source_point(cells, node.exact_identity));
            }
        });
        std::map<std::uint64_t, Loop> target_loops;
        std::map<Key, std::vector<const Meshes::AMR::AdaptiveTile*>> target_faces;
        collective([&]
        {
            std::set<Key> required(requested.begin(), requested.end());
            for (const auto& tile : draft.tiles)
                if (required.contains(tile.first))
                {
                    Loop loop;
                    for (auto node : tile.node_gids) loop.push_back(native_points.at(node));
                    target_loops.emplace(tile.face_gid, std::move(loop));
                    target_faces[tile.first].push_back(&tile);
                }
        });
        const auto rates = finite_rates(old);
        std::map<Key, std::vector<const Meshes::AMR::AdaptiveTile*>> source_faces;
        std::map<std::uint64_t, Loop> source_loops;
        collective([&]
        {
            for (const auto& tile : old.phases->tiles())
            {
                if (columns.contains(tile.first)) source_faces[tile.first].push_back(&tile);
                if (tile.second && columns.contains(*tile.second)) source_faces[*tile.second].push_back(&tile);
                source_loops.emplace(tile.face_gid, old.phases->finite_face_points(tile.face_gid));
            }
        });
        Meshes::BoundedGeometryLimits geometry_limits;
        geometry_limits.byte_budget = options.byte_budget / 2;
        std::size_t pairs = 0, patches = 0, retained_bytes = 0;
        const auto charge = [&](std::size_t bytes)
        {
            if (bytes > options.byte_budget - std::min(retained_bytes, options.byte_budget))
                throw std::runtime_error("Finite common-face handoff storage cap exhausted.");
            retained_bytes += bytes;
        };
        const auto shared_support = [](const Loop& first, const Loop& second) -> std::optional<Support>
        {
            const auto a = Meshes::bounded_common_supports(first), b = Meshes::bounded_common_supports(second);
            for (const auto& key : a)
                if (std::find(b.begin(), b.end(), key) != b.end()) return key;
            return {};
        };
        const auto overlap = [&](const Loop& first, const Loop& second)
        {
            if (++pairs > VoF::ExactFaceTransferOptions{}.maximum_pair_tests)
                throw std::runtime_error("Finite common-face pair cap exhausted.");
            const auto support = shared_support(first, second);
            if (!support) return Meshes::BoundedPlanarFaceOverlap{};
            auto result = Meshes::bounded_planar_face_overlap(first, second, *support, geometry_limits);
            if (!result.overlap_vertices.empty())
            {
                if (++patches > VoF::ExactFaceTransferOptions{}.maximum_patches)
                    throw std::runtime_error("Finite common-face patch cap exhausted.");
                charge(sizeof(result) + result.overlap_vertices.capacity() * sizeof(Point));
                for (const auto& point : result.overlap_vertices)
                    charge(Meshes::bounded_point_storage_bytes(point) - sizeof(Point));
            }
            return result;
        };
        const auto nonempty = [](const Meshes::BoundedPlanarFaceOverlap& value)
        {
            if (value.overlap_projected_area.value == 0 && value.overlap_projected_area.absolute_error == 0) return false;
            if (!(value.overlap_projected_area.value > value.overlap_projected_area.absolute_error))
                throw std::runtime_error("Finite common-face overlap positivity is unresolved.");
            return true;
        };
        const auto certify_cover = [&](const Loop& whole, const std::vector<Loop>& pieces)
        {
            fp::Sum fraction;
            for (std::size_t i = 0; i < pieces.size(); ++i)
            {
                const auto part = overlap(whole, pieces[i]);
                if (!nonempty(part)) throw std::runtime_error("Finite common-face cover contains empty piece.");
                fraction.add(part.orientation > 0 ? part.overlap_over_first : fp::negate(part.overlap_over_first));
                for (std::size_t j = 0; j < i; ++j)
                {
                    const auto collision = overlap(pieces[i], pieces[j]);
                    if (collision.overlap_projected_area.value != 0 || collision.overlap_projected_area.absolute_error != 0)
                        throw std::runtime_error("Finite common-face cover has overlapping or unresolved interiors.");
                }
            }
            const auto difference = fp::subtract(fraction.value(), {1, 0});
            if (std::abs(difference.value) > difference.absolute_error ||
                difference.absolute_error > VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
                throw std::runtime_error("Finite common-face cover has a hole or exhausted arithmetic budget.");
        };
        struct Part {Loop loop; Value rate; std::optional<std::size_t> edge; int sign = 1;};
        struct Facet {std::size_t cell; Loop loop; std::vector<Part> parts; bool inherited = false;};
        struct Internal {std::size_t first, second; Value rate;};
        struct Output
        {
            Key source, target;
            std::uint64_t source_revision, face, proof;
            int target_owner;
            Value rate, fraction;
        };
        std::vector<Output> outputs;
        collective([&]
        {
            for (const auto& [source_key, unordered_indices] : columns)
            {
                auto indices = unordered_indices;
                std::sort(indices.begin(), indices.end(), [&](auto a, auto b)
                { return common[a].target_key < common[b].target_key; });
                const auto source_row = mesh->owned_cell_map()->getLocalElement(GO(common[indices.front()].source_gid));
                if (source_row < 0) throw std::runtime_error("Finite common-face source is not locally owned.");
                const auto velocity = old.velocity[source_row];
                std::vector<Facet> facets;
                std::vector<std::vector<std::size_t>> cell_facets(indices.size());
                std::map<std::uint64_t, std::vector<Loop>> old_cover;
                for (std::size_t c = 0; c < indices.size(); ++c)
                {
                    const auto& geometry = *common[indices[c]].geometry;
                    for (const auto& face : geometry.faces())
                    {
                        Facet facet{c, {}, {}};
                        for (auto vertex : face) facet.loop.push_back(geometry.points()[vertex]);
                        for (const auto* tile : source_faces.at(source_key))
                        {
                            const auto part = overlap(facet.loop, source_loops.at(tile->face_gid));
                            if (!nonempty(part)) continue;
                            const auto local = mesh->overlap_face_map()->getLocalElement(GO(tile->face_gid));
                            if (local < 0) throw std::runtime_error("Finite source face rate is absent.");
                            const auto rate = fp::multiply(rates[local], part.overlap_over_second);
                            facet.parts.push_back({part.overlap_vertices, rate, {}, 1});
                            old_cover[tile->face_gid].push_back(part.overlap_vertices);
                        }
                        if (!facet.parts.empty())
                        {
                            std::vector<Loop> cover;
                            for (const auto& part : facet.parts) cover.push_back(part.loop);
                            certify_cover(facet.loop, cover);
                            facet.inherited = true;
                        }
                        charge(sizeof(Facet) + facet.loop.size() * sizeof(Point));
                        if (++patches > VoF::ExactFaceTransferOptions{}.maximum_patches)
                            throw std::runtime_error("Finite common-face patch cap exhausted.");
                        cell_facets[c].push_back(facets.size());
                        facets.push_back(std::move(facet));
                    }
                }
                for (const auto* tile : source_faces.at(source_key))
                {
                    const auto cover = old_cover.find(tile->face_gid);
                    if (cover == old_cover.end()) throw std::runtime_error("Finite old source face is uncovered.");
                    certify_cover(source_loops.at(tile->face_gid), cover->second);
                }
                std::vector<Internal> internal;
                for (std::size_t f = 0; f < facets.size(); ++f)
                {
                    if (facets[f].inherited) continue;
                    for (std::size_t g = f + 1; g < facets.size(); ++g)
                    {
                        if (facets[g].inherited || facets[g].cell == facets[f].cell) continue;
                        const auto piece = overlap(facets[f].loop, facets[g].loop);
                        if (!nonempty(piece)) continue;
                        if (piece.orientation != -1) throw std::runtime_error("Finite interior common faces have equal orientation.");
                        Point area;
                        const auto& loop = piece.overlap_vertices;
                        for (std::size_t i = 1; i + 1 < loop.size(); ++i)
                            area = fp::add(area, fp::cross(fp::subtract(loop[i], loop[0]), fp::subtract(loop[i + 1], loop[0])));
                        const auto rate = fp::divide(fp::dot(area, fp::point(velocity)), {2, 0});
                        const auto index = internal.size();
                        internal.push_back({facets[f].cell, facets[g].cell, rate});
                        facets[f].parts.push_back({loop, {}, index, 1});
                        auto reversed = loop;
                        std::reverse(reversed.begin(), reversed.end());
                        facets[g].parts.push_back({std::move(reversed), {}, index, -1});
                    }
                }
                for (const auto& face : facets)
                {
                    std::vector<Loop> pieces;
                    for (const auto& part : face.parts) pieces.push_back(part.loop);
                    certify_cover(face.loop, pieces);
                }
                const auto divergences = [&]
                {
                    std::vector<fp::Sum> sums(indices.size());
                    for (const auto& face : facets)
                        for (const auto& part : face.parts)
                            sums[face.cell].add(part.edge ? (part.sign > 0 ? internal[*part.edge].rate :
                                fp::negate(internal[*part.edge].rate)) : part.rate);
                    std::vector<Value> result;
                    for (const auto& sum : sums) result.push_back(sum.value());
                    return result;
                };
                std::vector<std::vector<std::size_t>> adjacent(indices.size());
                for (std::size_t e = 0; e < internal.size(); ++e)
                {
                    adjacent[internal[e].first].push_back(e);
                    adjacent[internal[e].second].push_back(e);
                }
                const auto none = std::numeric_limits<std::size_t>::max();
                std::vector<std::size_t> parent(indices.size(), none), parent_edge(indices.size(), none), order{0};
                parent[0] = 0;
                for (std::size_t i = 0; i < order.size(); ++i)
                    for (auto edge : adjacent[order[i]])
                    {
                        const auto other = internal[edge].first == order[i] ? internal[edge].second : internal[edge].first;
                        if (parent[other] == none)
                        {
                            parent[other] = order[i];
                            parent_edge[other] = edge;
                            order.push_back(other);
                        }
                    }
                if (order.size() != indices.size()) throw std::runtime_error("Finite common refinement is not face connected.");
                auto defect = divergences();
                for (std::size_t i = order.size(); i-- > 1;)
                {
                    const auto child = order[i], edge = parent_edge[child];
                    internal[edge].rate = fp::add(internal[edge].rate,
                        internal[edge].first == child ? fp::negate(defect[child]) : defect[child]);
                    defect[parent[child]] = fp::add(defect[parent[child]], defect[child]);
                }
                for (const auto& residual : divergences())
                    if (std::abs(residual.value) > residual.absolute_error)
                        throw std::runtime_error("Finite new common-face extension fails source-free closure.");
                std::uint64_t proof = 0;
                for (const auto& face : facets)
                    for (const auto& part : face.parts)
                    {
                        const auto& cell = common[indices[face.cell]];
                        const auto rate = part.edge ? (part.sign > 0 ? internal[*part.edge].rate :
                            fp::negate(internal[*part.edge].rate)) : part.rate;
                        for (const auto* tile : target_faces.at(cell.target_key))
                        {
                            const auto piece = overlap(part.loop, target_loops.at(tile->face_gid));
                            if (!nonempty(piece)) continue;
                            if (piece.orientation != 1) throw std::runtime_error("Finite target boundary has inconsistent orientation.");
                            outputs.push_back({source_key, cell.target_key,
                                old.phases->hierarchy()->owned().at(source_key).revision, tile->face_gid, proof++,
                                tile->first_owner, fp::multiply(rate, piece.overlap_over_first), piece.overlap_over_second});
                        }
                    }
            }
        });
        // The producer has independently proved every common-cell facet cover,
        // inherited source-face cover and opposite internal incidence. Together
        // with the opaque phase trial's donor/recipient cover, these immutable
        // proofs compose into a target boundary partition. The receiver checks
        // the bounded area closure and authenticates each proof exactly once.
        const auto put_scalar = [](std::vector<char>& out, long double value)
        {
            int exponent = 0;
            const auto m = std::frexp(value, &exponent);
            const auto hi = static_cast<double>(m), lo = static_cast<double>(m - hi);
            if (!std::isfinite(value) || static_cast<long double>(hi) + lo != m)
                throw std::runtime_error("Finite adapted face scalar exceeds wire precision.");
            append_word(out, std::bit_cast<std::uint64_t>(hi));
            append_word(out, std::bit_cast<std::uint64_t>(lo));
            append_word(out, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
        };
        const auto put_bytes = [](std::vector<char>& out, const std::vector<char>& key)
        {
            append_word(out, key.size()); out.insert(out.end(), key.begin(), key.end());
        };
        std::vector<OwnerPacket> packets;
        collective([&]
        {
            std::size_t used = 0;
            for (const auto& item : outputs)
            {
                OwnerPacket packet;
                packet.peer = item.target_owner;
                append_word(packet.bytes, old.phases->accepted_control());
                append_word(packet.bytes, state.phases->accepted_control());
                put_bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(item.source));
                append_word(packet.bytes, item.source_revision);
                put_bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(item.target));
                append_word(packet.bytes, item.face); append_word(packet.bytes, item.proof);
                for (const auto value : {item.rate, item.fraction})
                {
                    put_scalar(packet.bytes, value.value); put_scalar(packet.bytes, value.absolute_error);
                }
                used += sizeof(OwnerPacket) + packet.bytes.size();
                if (used > options.byte_budget / 2) throw std::runtime_error("Finite adapted face packet cap exhausted.");
                packets.push_back(std::move(packet));
            }
        });
        const auto received = exchange_owner_packets(packets, comm, options.byte_budget / 2);
        struct Incoming {Key source, target; std::uint64_t revision, face, proof; int owner; Value rate, fraction;};
        std::vector<Incoming> incoming;
        std::vector<Key> donor_keys;
        target.collective([&]
        {
            for (const auto& packet : received.packets)
            {
                std::size_t offset = 0;
                const auto word = [&]() {const auto value = read_word(packet.bytes, offset); offset += 8; return value;};
                const auto blob = [&]()
                {
                    const auto count = word();
                    if (count > packet.bytes.size() - std::min(offset, packet.bytes.size()))
                        throw std::runtime_error("Finite adapted face identity is malformed.");
                    std::vector<char> value(packet.bytes.begin() + offset, packet.bytes.begin() + offset + count);
                    offset += count; return value;
                };
                const auto scalar = [&]()
                {
                    const auto hi = std::bit_cast<double>(word()), lo = std::bit_cast<double>(word());
                    const auto exponent = std::bit_cast<std::int64_t>(word());
                    if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
                        throw std::runtime_error("Finite adapted face exponent is malformed.");
                    const auto value = std::scalbn(static_cast<long double>(hi) + lo, int(exponent));
                    if (!std::isfinite(value)) throw std::runtime_error("Finite adapted face scalar is nonfinite.");
                    return value;
                };
                if (word() != old.phases->accepted_control() || word() != state.phases->accepted_control())
                    throw std::runtime_error("Finite adapted face proof publication is stale.");
                Incoming item;
                item.source = Meshes::AMR::decode_adaptive_key(blob()); item.revision = word();
                item.target = Meshes::AMR::decode_adaptive_key(blob()); item.face = word(); item.proof = word();
                item.rate = {scalar(), scalar()}; item.fraction = {scalar(), scalar()}; item.owner = packet.peer;
                if (offset != packet.bytes.size() || item.rate.absolute_error < 0 || item.fraction.absolute_error < 0 ||
                    item.fraction.value < item.fraction.absolute_error)
                    throw std::runtime_error("Finite adapted face proof has malformed bounded values.");
                donor_keys.push_back(item.source); incoming.push_back(std::move(item));
            }
        });
        const auto owners = old.phases->hierarchy()->lookup_cells(donor_keys);
        target.collective([&]
        {
            std::map<GO, const Meshes::AMR::AdaptiveTile*> actual;
            for (const auto& tile : state.phases->tiles())
                if (target.mesh->owned_face_map()->getLocalElement(GO(tile.face_gid)) >= 0) actual.emplace(GO(tile.face_gid), &tile);
            std::vector<fp::Sum> flux(target.mesh->num_owned_faces()), coverage(flux.size());
            std::set<std::tuple<Key, std::uint64_t, std::uint64_t>> seen;
            for (std::size_t i = 0; i < incoming.size(); ++i)
            {
                const auto& item = incoming[i];
                const auto local = target.mesh->owned_face_map()->getLocalElement(GO(item.face));
                if (owners[i].owner != item.owner || owners[i].revision != item.revision || local < 0 ||
                    actual.at(GO(item.face))->first != item.target || !seen.emplace(item.source, item.face, item.proof).second)
                    throw std::runtime_error("Finite adapted face proof has foreign or duplicate source/target authority.");
                flux[local].add(item.rate); coverage[local].add(item.fraction);
            }
            state.finite_projection.byte_budget = options.byte_budget;
            state.finite_projection.face_rate.resize(flux.size()); state.finite_projection.face_radius.resize(flux.size());
            state.finite_projection.cell_radius.resize(target.volume.size()); state.face_velocity.resize(target.mesh->num_faces());
            for (std::size_t f = 0; f < flux.size(); ++f)
            {
                const auto cover = fp::subtract(coverage[f].value(), {1, 0});
                if (std::abs(cover.value) > cover.absolute_error || cover.absolute_error >
                    VoF::GeneralVoFFiniteRemapOptions{}.maximum_relative_target_defect)
                    throw std::runtime_error("Finite adapted target face has incomplete bounded coverage.");
                const auto rate = flux[f].value();
                if (target.mesh->is_geometry_exterior_face(LO(f)) && (rate.value != 0 || rate.absolute_error != 0))
                    throw std::runtime_error("Finite adapted handoff changes a physical impermeable wall.");
                state.finite_projection.face_rate[f] = rate.value; state.finite_projection.face_radius[f] = rate.absolute_error;
                const auto diagnostic = fp::divide(rate, {target.mesh->face_area(LO(f)), 0});
                state.face_velocity[f] = static_cast<double>(diagnostic.value);
                if (!std::isfinite(state.face_velocity[f]))
                    throw std::runtime_error("Finite adapted face velocity diagnostic overflows.");
            }
        });
        const auto final_rates = target.finite_rates(state);
        double local_divergence = 0;
        target.collective([&]
        {
            const auto lease = target.mesh->acquire_execution_view();
            for (std::size_t c = 0; c < target.volume.size(); ++c)
            {
                fp::Sum net;
                for (auto face : target.mesh->faces(LO(c))) net.add(target.mesh->owner_cell(face) == LO(c) ?
                    final_rates[face] : fp::negate(final_rates[face]));
                const auto residual = net.value();
                if (std::abs(residual.value) > residual.absolute_error)
                    throw std::runtime_error("Finite adapted handoff fails canonical target closure.");
                state.finite_projection.cell_radius[c] = residual.absolute_error;
                local_divergence = std::max(local_divergence, fp::upper_double(
                    fp::bound_add(std::abs(residual.value), residual.absolute_error) / target.volume[c]));
            }
        });
        const auto divergence = target.maximum(local_divergence);
        target.require(divergence <= options.flow.divergence_tolerance,
            "Finite adapted handoff exceeds the unchanged physical divergence target.");
        state.face_velocity = target.flow.faces(state.face_velocity);
        trial.validate_observations();
        return divergence;
    }
    double handoff_finite_owned_faces(const State& old, Impl& target, State& state) const
    {
        namespace fp = VoF::finite_detail;
        const auto comm = mesh->owned_cell_map()->getComm();
        const auto identity = [](const Meshes::AMR::AdaptiveTile& tile)
        {
            std::vector<char> result;
            for (const auto* key : {&tile.domain, &tile.surface, &tile.identity})
            {
                append_word(result, key->size());
                result.insert(result.end(), key->begin(), key->end());
            }
            return result;
        };
        const auto authority = [&](const std::vector<char>& key)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            for (unsigned char c : key) hash = (hash ^ c) * 1099511628211ULL;
            return int(hash % comm->getSize()); // Routing only; full identity is compared.
        };
        const auto bytes = [](std::vector<char>& out, const std::vector<char>& value)
        {
            append_word(out, value.size());
            out.insert(out.end(), value.begin(), value.end());
        };
        const auto scalar = [](std::vector<char>& out, long double value)
        {
            int exponent = 0;
            const auto mantissa = std::frexp(value, &exponent);
            const auto high = static_cast<double>(mantissa), low = static_cast<double>(mantissa - high);
            if (!std::isfinite(value) || static_cast<long double>(high) + low != mantissa)
                throw std::runtime_error("Finite face handoff scalar exceeds fixed precision.");
            append_word(out, std::bit_cast<std::uint64_t>(high));
            append_word(out, std::bit_cast<std::uint64_t>(low));
            append_word(out, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
        };
        const auto word = [](const std::vector<char>& in, std::size_t& offset)
        {
            const auto value = read_word(in, offset);
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
            if (!std::isfinite(value)) throw std::runtime_error("Finite face handoff scalar is nonfinite.");
            return value;
        };
        std::map<GO, const Meshes::AMR::AdaptiveTile*> target_tiles;
        std::vector<OwnerPacket> packets;
        collective([&]
        {
            if (old.finite_projection.face_rate.size() != mesh->num_owned_faces() ||
                old.finite_projection.face_radius.size() != mesh->num_owned_faces())
                throw std::runtime_error("Finite ownership handoff lacks source rate authority.");
            std::size_t used = 0;
            for (const auto& tile : old.phases->tiles())
            {
                const auto face = mesh->owned_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0) continue;
                auto key = identity(tile);
                OwnerPacket packet;
                packet.peer = authority(key);
                append_word(packet.bytes, 0);
                append_word(packet.bytes, old.phases->accepted_control());
                bytes(packet.bytes, key);
                bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(tile.first));
                bytes(packet.bytes, tile.second ? Meshes::AMR::encode_adaptive_key(*tile.second) : std::vector<char>{});
                scalar(packet.bytes, old.finite_projection.face_rate[face]);
                scalar(packet.bytes, old.finite_projection.face_radius[face]);
                used += packet.bytes.size() + sizeof(OwnerPacket);
                if (used > options.byte_budget / 2) throw std::runtime_error("Finite face source directory exceeds budget.");
                packets.push_back(std::move(packet));
            }
            for (const auto& tile : state.phases->tiles())
            {
                const auto face = target.mesh->owned_face_map()->getLocalElement(GO(tile.face_gid));
                if (face < 0) continue;
                target_tiles.emplace(GO(tile.face_gid), &tile);
                auto key = identity(tile);
                OwnerPacket packet;
                packet.peer = authority(key);
                append_word(packet.bytes, 1);
                append_word(packet.bytes, old.phases->accepted_control());
                bytes(packet.bytes, key);
                append_word(packet.bytes, tile.face_gid);
                bytes(packet.bytes, Meshes::AMR::encode_adaptive_key(tile.first));
                used += packet.bytes.size() + sizeof(OwnerPacket);
                if (used > options.byte_budget / 2) throw std::runtime_error("Finite face target directory exceeds budget.");
                packets.push_back(std::move(packet));
            }
        });
        const auto directory = exchange_owner_packets(packets, comm, options.byte_budget / 2);
        struct SourceFace {std::vector<char> first, second; long double rate, radius;};
        struct Request {std::vector<char> identity, first; std::uint64_t face; int owner;};
        std::map<std::vector<char>, SourceFace> sources;
        std::vector<Request> requests;
        packets.clear();
        collective([&]
        {
            for (const auto& packet : directory.packets)
            {
                std::size_t offset = 0;
                const auto type = word(packet.bytes, offset), control = word(packet.bytes, offset);
                auto key = read_bytes(packet.bytes, offset);
                if (control != old.phases->accepted_control() || authority(key) != comm->getRank())
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
                else throw std::runtime_error("Finite face directory type is malformed.");
                if (offset != packet.bytes.size()) throw std::runtime_error("Finite face directory has trailing payload.");
            }
            for (const auto& request : requests)
            {
                const auto found = sources.find(request.identity);
                if (found == sources.end())
                    throw std::runtime_error("Finite adaptive face handoff requires a bounded surface-overlap certificate.");
                const auto& source = found->second;
                const auto sign = request.first == source.first ? 1 : request.first == source.second ? -1 : 0;
                if (!sign) throw std::runtime_error("Finite face handoff incident-cell orientation is foreign.");
                OwnerPacket reply;
                reply.peer = request.owner;
                append_word(reply.bytes, request.face);
                bytes(reply.bytes, request.identity);
                scalar(reply.bytes, sign * source.rate);
                scalar(reply.bytes, source.radius);
                packets.push_back(std::move(reply));
            }
        });
        const auto result = exchange_owner_packets(packets, comm, options.byte_budget / 2);
        target.collective([&]
        {
            state.finite_projection.byte_budget = options.byte_budget;
            state.finite_projection.face_rate.resize(target.mesh->num_owned_faces());
            state.finite_projection.face_radius.resize(target.mesh->num_owned_faces());
            state.finite_projection.cell_radius.resize(target.volume.size());
            state.face_velocity.resize(target.mesh->num_faces());
            std::set<GO> seen;
            for (const auto& packet : result.packets)
            {
                std::size_t offset = 0;
                const auto gid = GO(word(packet.bytes, offset));
                const auto key = read_bytes(packet.bytes, offset);
                const auto rate = read_scalar(packet.bytes, offset), radius = read_scalar(packet.bytes, offset);
                const auto local = target.mesh->owned_face_map()->getLocalElement(gid);
                if (local < 0 || !seen.insert(gid).second || !target_tiles.contains(gid) ||
                    identity(*target_tiles.at(gid)) != key || packet.peer != authority(key) ||
                    offset != packet.bytes.size() || radius < 0)
                    throw std::runtime_error("Finite ownership face handoff has foreign or duplicate target.");
                if (target.mesh->is_geometry_exterior_face(local) && rate != 0)
                    throw std::runtime_error("Finite ownership handoff changes a closed wall.");
                const auto velocity = fp::divide({rate, radius}, {target.mesh->face_area(local), 0});
                state.face_velocity[local] = static_cast<double>(velocity.value);
                if (!std::isfinite(state.face_velocity[local]))
                    throw std::runtime_error("Finite ownership face velocity diagnostic overflows.");
                state.finite_projection.face_rate[local] = rate;
                state.finite_projection.face_radius[local] = radius;
            }
            if (seen.size() != target.mesh->num_owned_faces())
                throw std::runtime_error("Finite ownership handoff lacks target face coverage.");
        });
        const auto rates = target.finite_rates(state);
        double local_divergence = 0;
        target.collective([&]
        {
            const auto lease = target.mesh->acquire_execution_view();
            for (std::size_t c = 0; c < target.volume.size(); ++c)
            {
                fp::Sum net;
                for (const auto face : target.mesh->faces(LO(c)))
                    net.add(target.mesh->owner_cell(face) == LO(c) ? rates[face] : fp::negate(rates[face]));
                const auto residual = net.value();
                if (std::abs(residual.value) > residual.absolute_error)
                    throw std::runtime_error("Finite ownership handoff loses canonical source-free closure.");
                state.finite_projection.cell_radius[c] = residual.absolute_error;
                local_divergence = std::max(local_divergence, fp::upper_double(
                    fp::bound_add(std::abs(residual.value), residual.absolute_error) / target.volume[c]));
            }
        });
        state.face_velocity = target.flow.faces(state.face_velocity);
        const auto divergence = target.maximum(local_divergence);
        target.require(divergence <= options.flow.divergence_tolerance,
            "Finite fixed-clock face handoff exceeds the unchanged physical divergence target.");
        return divergence;
    }
    std::unique_ptr<Impl> remap(bool migration, std::span<const int> destinations,
        std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const
    {
        if (!finite(*accepted))
            return Exact::GeneralVoFSolver(*this).remap(migration, destinations, refine, coarsen);
        namespace fp = VoF::finite_detail;
        const auto& old = *accepted;
        const auto hierarchy = old.phases->hierarchy();
        std::vector<int> ordered;
        if (migration)
        {
            require(destinations.size() == old.phases->owned_order().size(), "Finite migration destination shape mismatch.");
            collective([&]
            {
                std::map<VoF::GeneralVoF::Key, int> requested;
                for (std::size_t i = 0; i < destinations.size(); ++i) requested.emplace(old.phases->owned_order()[i], destinations[i]);
                for (const auto& [key, leaf] : hierarchy->owned()) ordered.push_back(requested.at(key));
            });
        }
        const auto plan = migration ? hierarchy->plan_migration(ordered) : hierarchy->plan_adaptation(refine, coarsen);
        const auto raw_target = hierarchy->apply(plan);
        const auto native = Meshes::AMR::assemble_adaptive_native(*raw_target, raw_target->configuration_descriptor());
        const auto bound = raw_target->with_partition(native.partition, native.mesh->owned_cell_map());
        const auto target = hierarchy->apply(plan, bound);
        VoF::GeneralVoFFiniteRemapOptions remap_options;
        remap_options.byte_budget = options.byte_budget;
        remap_options.phase_density = {material.gas_density, material.liquid_density};
        remap_options.channels = {{"momentum_x", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
            {"momentum_y", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
            {"momentum_z", VoF::GeneralVoFFiniteRemapCarrier::MixtureMass, false},
            {"pressure_volume", VoF::GeneralVoFFiniteRemapCarrier::BulkVolume, false}};
        std::vector<VoF::GeneralVoFFiniteOwnedChannels> channels;
        collective([&]
        {
            const auto geometry = old.phases->owned_finite_geometry();
            for (std::size_t c = 0; c < old.velocity.size(); ++c)
            {
                VoF::GeneralVoFFiniteOwnedChannels item;
                item.key = old.phases->owned_order()[c];
                for (unsigned a = 0; a < 3; ++a)
                    item.amounts.push_back(fp::multiply({old.finite_mass[c], old.finite_mass_error[c]}, {old.velocity[c][a], 0}));
                item.amounts.push_back(fp::multiply(geometry[c]->volume(), {old.pressure[c], 0}));
                channels.push_back(std::move(item));
            }
        });
        const auto trial = old.phases->plan_remap(plan, target, channels, remap_options);
        trial.validate_observations();
        const auto phases = trial.candidate();
        auto candidate = std::make_unique<Impl>(phases, material, options);
        auto state = std::make_shared<State>();
        state->phases = phases;
        candidate->phase_storage(*state);
        state->velocity.resize(candidate->volume.size());
        state->pressure.resize(candidate->volume.size());
        std::map<VoF::GeneralVoF::Key, const VoF::GeneralVoFFiniteOwnedChannels*> result;
        for (const auto& item : trial.owned_channels()) result.emplace(item.key, &item);
        candidate->collective([&]
        {
            const auto geometry = phases->owned_finite_geometry();
            for (std::size_t c = 0; c < candidate->volume.size(); ++c)
            {
                const auto& item = *result.at(phases->owned_order()[c]);
                if (item.amounts.size() != 4) throw std::runtime_error("Finite remap channel schema differs.");
                for (unsigned a = 0; a < 3; ++a)
                    state->velocity[c][a] = fp::divide(item.amounts[a],
                        {state->finite_mass[c], state->finite_mass_error[c]}).as_double();
                state->pressure[c] = fp::divide(item.amounts[3], geometry[c]->volume()).as_double();
            }
        });
        candidate->surface_curvature(*state);
        const auto handoff_divergence = trial.ownership_only() ? handoff_finite_owned_faces(old, *candidate, *state) :
            handoff_finite_adapted_faces(old, *candidate, *state, trial);
        state->diagnostics = old.diagnostics;
        state->diagnostics.time = old.diagnostics.time;
        state->momentum_transport = old.momentum_transport;
        state->diagnostics.maximum_divergence = handoff_divergence;
        candidate->budget(*state);
        // Recovering a finite velocity must preserve the original physical
        // phase-mass-weighted momentum gate independently of channel transfer.
        for (unsigned a = 0; a < 3; ++a)
        {
            fp::Sum change, reference;
            for (std::size_t c = 0; c < old.velocity.size(); ++c)
            {
                const auto amount = fp::multiply({old.finite_mass[c], old.finite_mass_error[c]}, {old.velocity[c][a], 0});
                change.add(fp::negate(amount));
                reference.add({std::abs(amount.value), amount.absolute_error});
            }
            for (std::size_t c = 0; c < state->velocity.size(); ++c)
            {
                const auto amount = fp::multiply({state->finite_mass[c], state->finite_mass_error[c]}, {state->velocity[c][a], 0});
                change.add(amount);
                reference.add({std::abs(amount.value), amount.absolute_error});
            }
            const auto residual = change.value(), l1 = reference.value();
            int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
            if (l1.value != 0) (void) std::frexp(l1.value + l1.absolute_error, &local_exponent);
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
            const auto scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
            const auto center = residual.value / scale;
            const auto narrowed_center = static_cast<double>(center);
            const auto lower_reference = std::max(0.L, l1.value - l1.absolute_error) / scale;
            auto narrowed_reference = static_cast<double>(lower_reference);
            if (static_cast<long double>(narrowed_reference) > lower_reference)
                narrowed_reference = std::nextafter(narrowed_reference, 0.);
            double local[4] = {narrowed_center,
                fp::upper_double(fp::bound_add(residual.absolute_error / scale, std::abs(center - narrowed_center))),
                narrowed_reference, fp::upper_double(std::abs(center))}, global[4]{};
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_SUM, 4, local, global);
            const auto u = (8.L * mesh->owned_cell_map()->getComm()->getSize() + 16) * std::numeric_limits<double>::epsilon();
            const auto error = (global[1] + u * (global[3] + global[1])) / (1 - u);
            require(std::abs(global[0]) + error <= options.momentum_relative_tolerance * global[2] * (1 - u),
                "Finite fixed-clock remap changed physical phase-mass-weighted momentum.");
        }
        trial.validate_observations();
        candidate->accepted = std::move(state);
        return candidate;
    }
};
GeneralVoFSolver::GeneralVoFSolver(
    std::shared_ptr<const VoF::GeneralVoF> phases, VoFMaterial material, GeneralVoFSolverOptions options)
{
    if (!phases)
        throw std::invalid_argument("General VoF flow requires a phase authority.");
    impl_ = std::make_unique<Impl>(std::move(phases), material, options);
    impl_->initialize({}, {});
}
GeneralVoFSolver::~GeneralVoFSolver() = default;
GeneralVoFSolver::GeneralVoFSolver(GeneralVoFSolver&&) noexcept = default;
GeneralVoFSolver& GeneralVoFSolver::operator=(GeneralVoFSolver&&) noexcept = default;
void GeneralVoFSolver::initialize(const std::vector<VoF::Vector>& velocity, const std::vector<double>& face_velocity)
{
    impl_->initialize(velocity, face_velocity);
}
VoFStepDiagnostics GeneralVoFSolver::step(double dt)
{
    return impl_->step(dt);
}
double GeneralVoFSolver::stable_timestep() const
{
    return impl_->stable_timestep(*impl_->accepted, impl_->node_velocity(*impl_->accepted));
}
void GeneralVoFSolver::regrid(
    std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen)
{
    impl_ = impl_->remap(false, {}, refine, coarsen);
}
void GeneralVoFSolver::repartition(std::span<const int> destinations)
{
    impl_ = impl_->remap(true, destinations, {}, {});
}
double GeneralVoFSolver::time() const noexcept
{
    return impl_->accepted->diagnostics.time;
}
const std::shared_ptr<const VoF::GeneralVoF>& GeneralVoFSolver::phases() const noexcept
{
    return impl_->accepted->phases;
}
const std::shared_ptr<const MeshHandle<>>& GeneralVoFSolver::mesh() const noexcept
{
    return impl_->mesh;
}
const std::vector<double>& GeneralVoFSolver::gas_fraction() const noexcept
{
    return impl_->accepted->fraction;
}
const std::array<GeneralVoFMomentumDiagnostics, 3>& GeneralVoFSolver::momentum_transport_diagnostics() const noexcept
{
    return impl_->accepted->momentum_transport;
}
const std::vector<VoF::Vector>& GeneralVoFSolver::velocity() const noexcept
{
    return impl_->accepted->velocity;
}
const std::vector<double>& GeneralVoFSolver::face_velocity() const noexcept
{
    return impl_->accepted->face_velocity;
}
const std::vector<double>& GeneralVoFSolver::pressure() const noexcept
{
    return impl_->accepted->pressure;
}
const std::vector<double>& GeneralVoFSolver::curvature() const noexcept
{
    return impl_->accepted->curvature;
}
const VoFStepDiagnostics& GeneralVoFSolver::diagnostics() const noexcept
{
    return impl_->accepted->diagnostics;
}
} // namespace SimpleFluid

namespace SimpleFluid
{
void GeneralVoFSolver::enable_exact_geometry_statistics(bool enabled)
{
    Meshes::enable_exact_geometry_statistics(enabled);
}
void GeneralVoFSolver::reset_geometry_statistics()
{
    Meshes::reset_exact_geometry_statistics();
    Meshes::reset_bounded_geometry_statistics();
}
Meshes::GeometryStatisticsSnapshot GeneralVoFSolver::geometry_statistics()
{
    return {Meshes::exact_geometry_statistics_domain(), Meshes::bounded_geometry_statistics_domain(),
        Meshes::exact_geometry_statistics(), Meshes::bounded_geometry_statistics()};
}
}
