/** @file GeneralVoFSolver.cc
 * @brief Transactional native phase/implicit-momentum/physical-flow composition.
 */
#include "solvers/GeneralVoFSolver.hh"
#include "FVM/ExactFaceTransfer.hh"
#include "FVM/ExactPhaseTransfer.hh"
#include "geometry/ExactPlanarFace.hh"
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
} // namespace
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
        : mesh(validated_phase_mesh(phases)), material(m), options(o), flow(mesh), transport(mesh)
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
        volume.resize(mesh->num_owned_cells());
        thickness.resize(volume.size());
        collective(
            [&]
            {
                if (geometry.size() != volume.size())
                    throw std::runtime_error("General VoF phase/native geometry order differs.");
                for (std::size_t c = 0; c < volume.size(); ++c)
                {
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
    void phase_storage(State& state) const
    {
        const auto& gas = state.phases->gas_volumes();
        const auto& carrier = state.phases->carrier_volumes();
        const auto geometric_fraction = state.phases->geometric_gas_fractions();
        std::vector<Meshes::ExactPhaseMeasures> authority;
        require(gas.size() == volume.size() && carrier.size() == volume.size(),
            "General VoF phase/native storage dimensions differ.");
        std::vector<Meshes::ConvexGeometryPtr> geometry;
        collective(
            [&]
            {
                authority = current_inventory(*state.phases);
                geometry = state.phases->owned_geometry();
                state.fraction.resize(volume.size());
                state.density.resize(volume.size());
                state.physical_mass.resize(volume.size());
            });
        collective(
            [&]
            {
                for (std::size_t c = 0; c < volume.size(); ++c)
                {
                    state.fraction[c] = geometric_fraction[c];
                    state.physical_mass[c] = phase_mass_measure(authority[c]);
                    const auto& mass = state.physical_mass[c];
                    const auto density =
                        mass.divided_by(Meshes::ExactGeometryMeasure::volume(*geometry[c]), options.byte_budget);
                    state.density[c] = density.sign() ? density.certified().as_double() : 0;
                    if (mass.sign() <= 0 || !std::isfinite(state.density[c]) || !(state.density[c] > 0) ||
                        !std::isfinite(state.fraction[c]) || state.fraction[c] < 0 || state.fraction[c] > 1)
                        throw std::runtime_error("General VoF final extensive mixture mass/fraction is invalid.");
                }
            });
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
                    if (auto sample = detail::GeneralVoFCurvature::patch(patch.cell, *patch.geometry, patch.interface))
                        surface.push_back(*sample);
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
    void budget(State& state) const
    {
        std::array<Meshes::ExactGeometryMeasure, 7> local;
        std::vector<Meshes::ExactPhaseMeasures> authority;
        bool valid = true;
        collective(
            [&]
            {
                authority = current_inventory(*state.phases);
                for (std::size_t c = 0; c < volume.size(); ++c)
                {
                    const auto& gas = authority[c].gas;
                    const auto& carrier = authority[c].carrier;
                    const auto& mass = state.physical_mass[c];
                    const auto& velocity = state.velocity[c];
                    local[0] = detail::vof_exact::add(local[0], gas, options.byte_budget);
                    local[1] = detail::vof_exact::add(local[1], gas.scaled(material.gas_density), options.byte_budget);
                    local[2] =
                        detail::vof_exact::add(local[2], carrier.scaled(material.liquid_density), options.byte_budget);
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        const auto momentum = mass.scaled(velocity[a]);
                        local[3 + a] = detail::vof_exact::add(local[3 + a], momentum, options.byte_budget);
                        valid = valid && std::isfinite(velocity[a]);
                    }
                    local[6] = detail::vof_exact::add(
                        local[6], mass.scaled(.5 * dot(velocity, velocity)), options.byte_budget);
                    valid = valid && std::isfinite(state.density[c]) && state.density[c] > 0;
                }
            });
        require(valid, "General VoF accepted candidate has nonfinite velocity/storage.");
        std::array<double, 7> global{};
        for (unsigned a = 0; a < 7; ++a)
        {
            const auto exact = global_sum({local[a]});
            collective([&] { global[a] = exact.sign() ? exact.certified().as_double() : 0; });
        }
        state.diagnostics.gas_volume = global[0];
        state.diagnostics.gas_mass = global[1];
        state.diagnostics.liquid_mass = global[2];
        for (unsigned a = 0; a < 3; ++a)
            state.diagnostics.momentum[a] = global[3 + a];
        state.diagnostics.kinetic_energy = global[6];
        double speed = 0;
        for (const auto& velocity : state.velocity)
            speed = std::max(speed, std::sqrt(dot(velocity, velocity)));
        state.diagnostics.maximum_velocity = maximum(speed);
        require(std::all_of(global.begin(), global.end(), [](double value) { return std::isfinite(value); }),
            "General VoF global accepted budget is nonfinite.");
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
                    append_word(packet.bytes, node.exact_identity.size());
                    packet.bytes.insert(packet.bytes.end(), node.exact_identity.begin(), node.exact_identity.end());
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
                        const auto normal = wall->geometry->area_vector();
                        for (double x : {normal.x, normal.y, normal.z})
                            append_word(packet.bytes, std::bit_cast<std::uint64_t>(x));
                        const auto certificate = wall->geometry->encode_incidence();
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
        // Supplement rounded face CFL with the authoritative exact rate, including
        // nonzero streams whose individual binary64 product is zero.
        FaceMeasures owned;
        collective(
            [&]
            {
                if (state.projection_arithmetic.face_rate.size() != mesh->num_owned_faces())
                    throw std::runtime_error("Missing exact rate for timestep admission.");
                for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
                    owned.emplace(mesh->face_global_id(LO(f)),
                        std::pair{state.projection_arithmetic.face_rate[f], Meshes::ExactGeometryMeasure{}});
            });
        const auto rates = exchange_face_measures(state, owned, options.byte_budget / 2);
        double local_rate_bound = std::numeric_limits<double>::infinity();
        collective(
            [&]
            {
                using M = Meshes::ExactGeometryMeasure;
                const auto geometry = state.phases->owned_geometry();
                const auto slots = geometry.size() + 32;
                const auto scalar_budget = options.byte_budget / 2 / slots;
                if (scalar_budget < 1024)
                    throw std::runtime_error("Exact rate CFL cumulative reserve exhausted.");
                std::vector<M> sums(geometry.size());
                const auto lease = mesh->acquire_execution_view();
                for (const auto& tile : state.phases->tiles())
                {
                    const auto face = mesh->overlap_face_map()->getLocalElement(GO(tile.face_gid));
                    if (face < 0)
                        throw std::runtime_error("Missing local exact CFL face.");
                    const auto rate = rates.at(GO(tile.face_gid)).first;
                    const auto magnitude = rate.sign() < 0 ? rate.scaled(-1) : rate;
                    for (const auto cell : {mesh->owner_cell(face), mesh->neighbor_cell(face)})
                        if (cell >= 0 && std::size_t(cell) < sums.size())
                            sums[cell] = detail::vof_exact::add(sums[cell], magnitude, scalar_budget);
                }
                for (std::size_t c = 0; c < sums.size(); ++c)
                    if (sums[c].sign())
                    {
                        const auto exact =
                            M::volume(*geometry[c]).scaled(options.flow.max_courant).divided_by(sums[c], scalar_budget);
                        if (exact.compare(M::from_value(std::numeric_limits<double>::max())) > 0)
                            continue;
                        double value = exact.certified().as_double();
                        if (M::from_value(value).compare(exact) > 0)
                            value = std::nextafter(value, 0.0);
                        if (!(value > 0))
                            throw std::runtime_error("Exact rate has no positive representable timestep.");
                        local_rate_bound = std::min(local_rate_bound, value);
                    }
            });
        double rate_bound = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_rate_bound, &rate_bound);
        // A trajectory displacement bound supplements net-face CFL: opposing
        // swept streams can exist on a face whose projected net Phi is zero.
        return std::min(
            rate_bound, speed > 0 ? std::min(bound, options.flow.max_courant * global_thickness / speed) : bound);
    }
    Meshes::ExactGeometryMeasure phase_mass_measure(const Meshes::ExactPhaseMeasures& phase) const
    {
        return phase.gas.scaled(material.gas_density) + phase.carrier.scaled(material.liquid_density);
    }
    std::vector<Meshes::ExactPhaseMeasures> current_inventory(const VoF::GeneralVoF& phases) const
    {
        // The immutable FVM source selects the policy. Finite is the default;
        // experimental exact authority is explicit and remains unqualified.
        return phases.inventory_measures();
    }
    void verify_phase_publication(const VoF::GeneralVoF::Trial& trial, const VoF::GeneralVoF& candidate) const
    {
        collective(
            [&]
            {
                using M = Meshes::ExactGeometryMeasure;
                using Domain = detail::GeneralVoFStorageDomain;
                detail::GeneralVoFExactStorage storage(options.byte_budget);
                // Only finite centers/bounds enter these small interval sums.
                // Keep their pending values and one operation disjoint from
                // the variable-size recipient aggregation below.
                storage.capacity(Domain::Scratch, 16 * 1024, 1);
                const auto n = volume.size();
                storage.capacity(Domain::Scratch, n, 4 * sizeof(M));
                M zero;
                storage.measure(Domain::Input, zero);
                for (const auto& route : trial.transfers())
                {
                    storage.measure(Domain::Input, route.exact.gas);
                    storage.measure(Domain::Input, route.exact.carrier);
                }
                std::vector<M> gas(n, zero), carrier(n, zero);
                if (gas.capacity() > 2 * n || carrier.capacity() > 2 * n)
                    throw std::runtime_error("General VoF publication vector capacity exceeds admission.");
                for (std::size_t c = 0; c < n; ++c)
                {
                    storage.measure(Domain::Scratch, gas[c]);
                    storage.measure(Domain::Scratch, carrier[c]);
                }
                const auto add = [&](M& value, const M& part)
                {
                    const auto remaining = storage.remaining();
                    if (remaining <= 2048)
                        throw std::runtime_error("General VoF finite publication proof exhausted its operation lease.");
                    auto sum = detail::vof_exact::add(value, part, remaining - 1024);
                    storage.measure(Domain::Scratch, sum);
                    storage.release(Domain::Scratch, value);
                    value = std::move(sum);
                };
                for (const auto& route : trial.transfers())
                {
                    const auto local = mesh->owned_cell_map()->getLocalElement(GO(route.recipient));
                    if (local < 0)
                        throw std::runtime_error("General VoF publication proof has a foreign recipient.");
                    add(gas[local], route.exact.gas);
                    add(carrier[local], route.exact.carrier);
                }
                if (candidate.inventory_policy() != accepted->phases->inventory_policy())
                    throw std::runtime_error("General VoF trial changed inventory authority policy.");
                if (candidate.inventory_policy() == VoF::PhaseInventoryPolicy::ExperimentalExact)
                {
                    const auto& exact = candidate.experimental_phases();
                    if (exact.size() != n)
                        throw std::runtime_error("Experimental exact phase publication shape differs.");
                    for (std::size_t c = 0; c < n; ++c)
                        if (gas[c].compare(exact[c].gas) || carrier[c].compare(exact[c].carrier))
                            throw std::runtime_error("Experimental exact phase publication differs from its routes.");
                    return;
                }
                const auto& phases = candidate.phase_volumes();
                // Identity trials retain the accepted object and its historical
                // publication metadata; only THIS trial's bounds may admit error.
                const auto& bounds = trial.phase_roundoff_bounds();
                if (phases.size() != n || bounds.size() != n)
                    throw std::runtime_error("General VoF finite publication bound shape differs.");
                const auto check = [&](double value, double bound, const M& expected)
                {
                    if (!std::isfinite(value) || !std::isfinite(bound) || bound < 0)
                        throw std::runtime_error("General VoF finite publication bound is invalid.");
                    const auto center = M::from_value(value);
                    const auto lower = detail::vof_exact::add(center, M::from_value(-bound), 8192);
                    const auto upper = detail::vof_exact::add(center, M::from_value(bound), 8192);
                    if (expected.compare(lower) < 0 || expected.compare(upper) > 0)
                        throw std::runtime_error("General VoF phase publication exceeds its own conversion bound.");
                };
                for (std::size_t c = 0; c < n; ++c)
                {
                    check(phases[c].gas, bounds[c].gas, gas[c]);
                    check(phases[c].carrier, bounds[c].carrier, carrier[c]);
                }
                // Positive densities give |Mnew-Mrouted| <= rhoG*boundG+
                // rhoC*boundC. This is arithmetic provenance only: momentum
                // still solves its original equation with represented Mnew.
            });
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
        const auto closure_topology = projection_topology(*next->phases);
        flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction,
            next->curvature, 1, 0, options.flow, next->diagnostics, &next->projection_arithmetic,
            std::span<const Meshes::ExactGeometryMeasure>(next->physical_mass), &closure_topology);
        budget(*next);
        accepted = std::move(next);
    }
    using FaceMeasures = std::map<GO, std::pair<Meshes::ExactGeometryMeasure, Meshes::ExactGeometryMeasure>>;
    FaceMeasures exchange_face_measures(
        const State& state, const FaceMeasures& owned, std::size_t resource_budget) const
    {
        using Domain = detail::GeneralVoFStorageDomain;
        detail::GeneralVoFExactStorage storage(resource_budget);
        std::size_t scalar_budget = 0, packet_budget = 0;
        collective(
            [&]
            {
                storage.capacity(Domain::Input, owned.size(), sizeof(FaceMeasures::value_type) + 4 * sizeof(void*));
                for (const auto& [gid, value] : owned)
                {
                    storage.measure(Domain::Input, value.first);
                    storage.measure(Domain::Input, value.second);
                }
                const auto count = state.phases->tiles().size();
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
        for (const auto& tile : state.phases->tiles())
            required.insert(GO(tile.face_gid));
        Teuchos::Array<GO> ids(required.begin(), required.end());
        Teuchos::Array<int> owners(ids.size(), -1);
        (void) mesh->owned_face_map()->getRemoteIndexList(ids(), owners());
        std::map<GO, int> authority;
        std::vector<OwnerPacket> packets;
        collective(
            [&]
            {
                for (std::size_t i = 0; i < std::size_t(ids.size()); ++i)
                {
                    if (owners[i] < 0)
                        throw std::runtime_error("Missing canonical Phi authority.");
                    authority.emplace(ids[i], owners[i]);
                    OwnerPacket p;
                    p.peer = owners[i];
                    append_word(p.bytes, std::uint64_t(ids[i]));
                    packets.push_back(std::move(p));
                }
            });
        const auto requests = exchange_owner_packets(packets, mesh->owned_cell_map()->getComm(), packet_budget);
        packets.clear();
        collective(
            [&]
            {
                std::size_t used = 0;
                for (const auto& p : requests.packets)
                {
                    if (p.bytes.size() != 8)
                        throw std::runtime_error("Bad Phi certificate request.");
                    const GO face = GO(read_word(p.bytes, 0));
                    const auto& values = owned.at(face);
                    if (mesh->owned_face_map()->getLocalElement(face) < 0)
                        throw std::runtime_error("Foreign Phi certificate request.");
                    OwnerPacket answer;
                    answer.peer = p.peer;
                    append_word(answer.bytes, std::uint64_t(face));
                    for (const auto* value : {&values.first, &values.second})
                    {
                        const auto encoded = value->encode(scalar_budget);
                        if (answer.bytes.size() > packet_budget || packet_budget - answer.bytes.size() < 8 ||
                            encoded.size() > packet_budget - answer.bytes.size() - 8)
                            throw std::runtime_error("Phi certificate exceeds budget.");
                        append_word(answer.bytes, encoded.size());
                        answer.bytes.insert(answer.bytes.end(), encoded.begin(), encoded.end());
                    }
                    if (answer.bytes.size() > packet_budget || used > packet_budget - answer.bytes.size() ||
                        sizeof(OwnerPacket) > packet_budget - used - answer.bytes.size())
                        throw std::runtime_error("Phi certificate packets exceed budget.");
                    used += sizeof(OwnerPacket) + answer.bytes.size();
                    packets.push_back(std::move(answer));
                }
            });
        const auto answers = exchange_owner_packets(packets, mesh->owned_cell_map()->getComm(), packet_budget);
        FaceMeasures result;
        collective(
            [&]
            {
                for (const auto& p : answers.packets)
                {
                    if (p.bytes.size() < 24)
                        throw std::runtime_error("Truncated Phi certificate.");
                    const GO face = GO(read_word(p.bytes, 0));
                    if (!authority.contains(face) || authority.at(face) != p.peer)
                        throw std::runtime_error("Wrong Phi certificate authority.");
                    std::size_t offset = 8;
                    std::array<Meshes::ExactGeometryMeasure, 2> value;
                    for (auto& item : value)
                    {
                        if (offset > p.bytes.size() || p.bytes.size() - offset < 8)
                            throw std::runtime_error("Truncated Phi certificate length.");
                        const auto count = read_word(p.bytes, offset);
                        offset += 8;
                        if (count > p.bytes.size() - offset)
                            throw std::runtime_error("Invalid Phi certificate length.");
                        item = Meshes::ExactGeometryMeasure::decode(
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
    detail::GeneralVoFProjectionClosureTopology projection_topology(const VoF::GeneralVoF& phases) const
    {
        detail::GeneralVoFProjectionClosureTopology result;
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
    std::vector<VoF::GeneralVoFFaceVolume> closed_face_volumes(const State& state, double dt) const
    {
        using Domain = detail::GeneralVoFStorageDomain;
        detail::GeneralVoFExactStorage storage(options.byte_budget);
        std::size_t scalar_budget = 0, publication_budget = 0;
        collective(
            [&]
            {
                const auto nf = mesh->num_owned_faces(), nt = state.phases->tiles().size();
                const auto& rate = state.projection_arithmetic.face_rate;
                if (rate.size() != nf)
                    throw std::runtime_error("Missing accepted divergence-free face-rate ledger.");
                storage.capacity(Domain::Input, rate.capacity(), sizeof(Meshes::ExactGeometryMeasure));
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
        collective(
            [&]
            {
                for (std::size_t f = 0; f < mesh->num_owned_faces(); ++f)
                    owned.emplace(mesh->face_global_id(LO(f)),
                        std::pair{detail::vof_exact::scale(state.projection_arithmetic.face_rate[f], dt, scalar_budget),
                            Meshes::ExactGeometryMeasure{}});
            });
        const auto controls = exchange_face_measures(state, owned, publication_budget);
        std::vector<VoF::GeneralVoFFaceVolume> result;
        collective(
            [&]
            {
                result.reserve(state.phases->tiles().size());
                for (const auto& tile : state.phases->tiles())
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
    VoFStepDiagnostics step(double dt)
    {
        require(std::isfinite(dt) && dt > 0, "General VoF timestep must be positive and finite.");
        double minimum = 0, maximum_dt = 0;
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &dt, &minimum);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &dt, &maximum_dt);
        require(minimum == maximum_dt, "General VoF timestep differs across ranks.");
        const auto& old = *accepted;
        const auto nodes = node_velocity(old);
        require(
            dt <= stable_timestep(old, nodes), "General VoF timestep exceeds its physical geometry/trajectory limit.");
        const auto controls = closed_face_volumes(old, dt);
        const auto trial = old.phases->plan_transport(nodes, controls, dt);
        double local_h = std::numeric_limits<double>::infinity(), global_h = 0;
        for (double h : thickness)
            local_h = std::min(local_h, h);
        Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local_h, &global_h);
        collective(
            [&]
            {
                const auto scalar_budget = options.byte_budget / 32;
                Meshes::ExactTrajectoryLimits limits;
                limits.byte_budget = scalar_budget;
                const auto displacement_limit = Meshes::ExactGeometryMeasure::from_value(global_h)
                                                    .scaled(options.flow.max_courant)
                                                    .squared(scalar_budget);
                if (trial.node_trajectories().size() != nodes.size())
                    throw std::runtime_error("Incomplete exact node trajectory CFL certificate.");
                for (const auto& node : trial.node_trajectories())
                    if (node.trajectory.displacement_squared(limits).compare(displacement_limit) > 0)
                        throw std::runtime_error("Exact node trajectory exceeds physical displacement CFL.");
            });
        auto next = std::make_shared<State>();
        next->phases = trial.candidate();
        require(next->phases->mesh().get() == mesh.get(), "General VoF transport changed its immutable native mesh.");
        verify_phase_publication(trial, *next->phases);
        phase_storage(*next);
        std::vector<Meshes::ExactGeometryMeasure> outgoing(volume.size()), source(volume.size()),
            capacity(volume.size());
        std::vector<detail::ConservativeVoFTransport::ExactTransfer> incoming;
        std::vector<detail::ConservativeVoFTransport::ExactFaceContribution> face_contributions;
        collective(
            [&]
            {
                capacity = next->physical_mass;
                std::set<GO> seen;
                for (const auto& donor : trial.donor_budgets())
                {
                    const auto local = mesh->owned_cell_map()->getLocalElement(GO(donor.donor));
                    if (local < 0 || !seen.insert(GO(donor.donor)).second)
                        throw std::runtime_error("General VoF donor budget is duplicate or foreign.");
                    outgoing[local] = phase_mass_measure(donor.exact_exported);
                }
                if (seen.size() != volume.size())
                    throw std::runtime_error("General VoF donor budgets do not cover native owned cells.");
                for (const auto& route : trial.transfers())
                    if (route.donor != route.recipient)
                        incoming.push_back({GO(route.donor), GO(route.recipient), phase_mass_measure(route.exact)});
                for (const auto& face : trial.face_contributions())
                {
                    const auto local = mesh->owned_face_map()->getLocalElement(GO(face.face));
                    if (local < 0 || (face.direction != 1 && face.direction != -1))
                        throw std::runtime_error("General VoF raw donor stream is not a canonical owned face.");
                    const auto mass = phase_mass_measure(face.exact);
                    face_contributions.push_back({local, GO(face.donor), mass.scaled(face.direction)});
                }
            });
        next->velocity.resize(volume.size());
        for (unsigned a = 0; a < 3; ++a)
        {
            std::vector<Meshes::ExactGeometryMeasure> momentum(volume.size());
            collective(
                [&]
                {
                    for (std::size_t c = 0; c < volume.size(); ++c)
                    {
                        momentum[c] = old.physical_mass[c].scaled(old.velocity[c][a]);
                    }
                });
            const auto carried =
                transport.solve_exact(momentum, source, capacity, outgoing, incoming, face_contributions, false,
                    options.maximum_momentum_iterations, options.momentum_relative_tolerance, options.byte_budget);
            next->momentum_transport[a] = {
                carried.physical_relative_residual, carried.iterations, carried.physical_refinements};
            for (std::size_t c = 0; c < volume.size(); ++c)
                next->velocity[c][a] = carried.cell_trace[c];
        }
        std::vector<double> viscosity(volume.size());
        const auto physical_geometry = old.phases->owned_geometry();
        std::vector<Meshes::ExactPhaseMeasures> old_inventory;
        collective(
            [&]
            {
                old_inventory = current_inventory(*old.phases);
                for (std::size_t c = 0; c < volume.size(); ++c)
                {
                    const auto weighted = old_inventory[c].gas.scaled(material.gas_viscosity) +
                                          old_inventory[c].carrier.scaled(material.liquid_viscosity);
                    const auto coefficient = weighted.divided_by(
                        Meshes::ExactGeometryMeasure::volume(*physical_geometry[c]), options.byte_budget);
                    viscosity[c] = coefficient.sign() ? coefficient.certified().as_double() : 0;
                }
            });
        const auto viscous = flow.viscous_force(old.velocity, viscosity);
        std::vector<Vector> change(volume.size());
        collective(
            [&]
            {
                for (std::size_t c = 0; c < volume.size(); ++c)
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        const auto mass = next->physical_mass[c];
                        const auto viscous_change = Meshes::ExactGeometryMeasure::from_value(viscous[c][a])
                                                        .scaled(dt)
                                                        .divided_by(mass, options.byte_budget);
                        const auto updated =
                            Meshes::ExactGeometryMeasure::from_value(next->velocity[c][a]) +
                            Meshes::ExactGeometryMeasure::from_value(options.flow.gravity[a]).scaled(dt) +
                            viscous_change;
                        next->velocity[c][a] = updated.sign() ? updated.certified().as_double() : 0;
                        change[c][a] = next->velocity[c][a] - old.velocity[c][a];
                    }
            });
        next->face_velocity = old.face_velocity;
        const auto face_change = flow.interpolate(change);
        for (std::size_t f = 0; f < next->face_velocity.size(); ++f)
            next->face_velocity[f] += face_change[f];
        surface_curvature(*next);
        next->projection_arithmetic.byte_budget = options.byte_budget;
        const auto closure_topology = projection_topology(*next->phases);
        flow.project(next->face_velocity, next->velocity, next->pressure, next->density, next->fraction,
            next->curvature, dt, material.surface_tension, options.flow, next->diagnostics,
            &next->projection_arithmetic, std::span<const Meshes::ExactGeometryMeasure>(next->physical_mass),
            &closure_topology);
        next->diagnostics.time = old.diagnostics.time + dt;
        require(std::isfinite(next->diagnostics.time) && next->diagnostics.time > old.diagnostics.time,
            "General VoF timestep did not advance its finite clock.");
        budget(*next);
        // The authority verifies source identity again only after every flow
        // gate passes. Publishing this one immutable snapshot is the commit.
        next->phases = old.phases->apply(trial);
        accepted = std::move(next);
        return accepted->diagnostics;
    }
    Meshes::ExactGeometryMeasure global_sum(const std::vector<Meshes::ExactGeometryMeasure>& values) const
    {
        Meshes::ExactGeometryMeasure local;
        std::vector<OwnerPacket> packets;
        collective(
            [&]
            {
                for (const auto& value : values)
                    local = detail::vof_exact::add(local, value, options.byte_budget);
                packets.push_back({0, local.encode(options.byte_budget)});
            });
        const auto received = exchange_owner_packets(packets, mesh->owned_cell_map()->getComm(), options.byte_budget);
        Meshes::ExactGeometryMeasure sum;
        collective(
            [&]
            {
                if (mesh->owned_cell_map()->getComm()->getRank() == 0)
                {
                    std::set<int> senders;
                    for (const auto& p : received.packets)
                    {
                        if (!senders.insert(p.peer).second)
                            throw std::runtime_error("Duplicate exact global inventory sender.");
                        sum = detail::vof_exact::add(sum,
                            Meshes::ExactGeometryMeasure::decode(p.bytes, options.byte_budget), options.byte_budget);
                    }
                    if (senders.size() != std::size_t(mesh->owned_cell_map()->getComm()->getSize()))
                        throw std::runtime_error("Incomplete exact global inventory reduction.");
                }
            });
        std::vector<char> encoded;
        collective(
            [&]
            {
                if (mesh->owned_cell_map()->getComm()->getRank() == 0)
                    encoded = sum.encode(options.byte_budget);
            });
        std::uint64_t count = encoded.size();
        Teuchos::broadcast(*mesh->owned_cell_map()->getComm(), 0, 1, &count);
        collective(
            [&]
            {
                if (count > options.byte_budget || count > std::uint64_t(std::numeric_limits<int>::max()))
                    throw std::runtime_error("Exact global inventory broadcast exceeds budget.");
                encoded.resize(std::size_t(count));
            });
        Teuchos::broadcast(*mesh->owned_cell_map()->getComm(), 0, int(count), encoded.data());
        collective([&] { sum = Meshes::ExactGeometryMeasure::decode(encoded, options.byte_budget); });
        return sum;
    }
    std::unique_ptr<Impl> remap(bool migration, std::span<const int> destinations,
        std::span<const Meshes::AMR::AdaptiveMark> refine, std::span<const Meshes::AMR::AdaptiveBytes> coarsen) const
    {
        using M = Meshes::ExactGeometryMeasure;
        const auto& old = *accepted;
        const auto hierarchy = old.phases->hierarchy();
        auto source = old.phases->export_transfer_source(!migration);
        std::vector<VoF::ExactTransferChannel> schema = {{"momentum_x", VoF::ExactTransferCarrier::MixtureMass, false},
            {"momentum_y", VoF::ExactTransferCarrier::MixtureMass, false},
            {"momentum_z", VoF::ExactTransferCarrier::MixtureMass, false},
            {"pressure_volume", VoF::ExactTransferCarrier::BulkVolume, false}};
        std::vector<VoF::ExactTransferOwnedChannels> channels;
        const auto source_geometry = old.phases->owned_geometry();
        std::vector<Meshes::ExactPhaseMeasures> source_inventory;
        collective(
            [&]
            {
                source_inventory = current_inventory(*old.phases);
                for (std::size_t c = 0; c < old.phases->owned_order().size(); ++c)
                {
                    const auto& gas = source_inventory[c].gas;
                    const auto& carrier = source_inventory[c].carrier;
                    VoF::ExactTransferOwnedChannels item;
                    item.key = old.phases->owned_order()[c];
                    item.phase_mass = VoF::ExactTransferPhaseMass{
                        gas.scaled(material.gas_density), carrier.scaled(material.liquid_density)};
                    const auto mass = item.phase_mass->gas + item.phase_mass->carrier;
                    for (unsigned a = 0; a < 3; ++a)
                        item.amounts.push_back(mass.scaled(old.velocity[c][a]));
                    item.amounts.push_back(M::volume(*source_geometry[c]).scaled(old.pressure[c]));
                    channels.push_back(std::move(item));
                }
            });
        VoF::ExactPhaseTransferOptions transfer_options;
        transfer_options.byte_budget = options.byte_budget;
        source = source.with_channels(std::move(schema), std::move(channels), transfer_options);
        std::vector<int> ordered_destinations;
        if (migration)
        {
            require(destinations.size() == old.phases->owned_order().size(),
                "General VoF repartition destination count differs.");
            collective(
                [&]
                {
                    std::map<VoF::GeneralVoF::Key, int> requested;
                    for (std::size_t i = 0; i < destinations.size(); ++i)
                        requested.emplace(old.phases->owned_order()[i], destinations[i]);
                    for (const auto& [key, leaf] : hierarchy->owned())
                        ordered_destinations.push_back(requested.at(key));
                });
        }
        const auto plan =
            migration ? hierarchy->plan_migration(ordered_destinations) : hierarchy->plan_adaptation(refine, coarsen);
        const auto raw_target = hierarchy->apply(plan);
        const auto native_target =
            Meshes::AMR::assemble_adaptive_native(*raw_target, raw_target->configuration_descriptor());
        const auto bound_target =
            raw_target->with_partition(native_target.partition, native_target.mesh->owned_cell_map());
        const auto target = hierarchy->apply(plan, bound_target);
        const auto transferred = VoF::ExactPhaseTransfer::prepare(source, plan, target, transfer_options);
        const auto candidate_phase = old.phases->adopt_transfer(transferred);
        require(candidate_phase->inventory_policy() == old.phases->inventory_policy(),
            "General VoF remap changed inventory authority policy.");
        auto candidate = std::make_unique<Impl>(candidate_phase, material, options);
        auto state = std::make_shared<State>();
        state->phases = candidate_phase;
        candidate->phase_storage(*state);
        state->velocity.resize(candidate->volume.size());
        std::vector<double> mapped_pressure(candidate->volume.size());
        std::map<VoF::GeneralVoF::Key, const VoF::ExactPhaseTransferCapsule*> result;
        for (const auto& item : transferred.owned())
            result.emplace(item.key, &item);
        const auto target_geometry = candidate_phase->owned_geometry();
        candidate->collective(
            [&]
            {
                for (std::size_t c = 0; c < candidate_phase->owned_order().size(); ++c)
                {
                    const auto& item = *result.at(candidate_phase->owned_order()[c]);
                    if (item.amounts.size() != 4 || !item.phase_mass)
                        throw std::runtime_error("Exact general flow transfer schema differs.");
                    const auto mass = state->physical_mass[c];
                    if (!std::isfinite(item.roundoff.gas) || !std::isfinite(item.roundoff.carrier) ||
                        item.roundoff.gas < 0 || item.roundoff.carrier < 0)
                        throw std::runtime_error("Remapped phase publication bound is invalid.");
                    const auto mass_difference = item.phase_mass->gas + item.phase_mass->carrier - mass;
                    const auto mass_error = mass_difference.sign() < 0 ? mass_difference.scaled(-1) : mass_difference;
                    // Ownership copies perform no new publication. Their
                    // historical phase metadata cannot admit a mass mismatch.
                    const bool exact_policy =
                        candidate_phase->inventory_policy() == VoF::PhaseInventoryPolicy::ExperimentalExact;
                    const auto mass_bound =
                        migration || exact_policy
                            ? M{}
                            : M::from_value(item.roundoff.gas).scaled(material.gas_density) +
                                  M::from_value(item.roundoff.carrier).scaled(material.liquid_density);
                    if (mass_error.compare(mass_bound) > 0)
                        throw std::runtime_error("Remapped phase mass exceeds its finite publication bound.");
                    for (unsigned a = 0; a < 3; ++a)
                    {
                        const auto value = item.amounts[a].divided_by(mass, options.byte_budget);
                        state->velocity[c][a] = value.sign() ? value.certified().as_double() : 0;
                    }
                    const auto pressure =
                        item.amounts[3].divided_by(M::volume(*target_geometry[c]), options.byte_budget);
                    mapped_pressure[c] = pressure.sign() ? pressure.certified().as_double() : 0;
                }
            });
        candidate->surface_curvature(*state);
        // Cell momentum/pV and face q are independent accepted degrees of
        // freedom. A fixed-clock handoff must not apply a physical dt=1 impulse.
        using Domain = detail::GeneralVoFStorageDomain;
        detail::GeneralVoFExactStorage handoff_storage(options.byte_budget);
        std::size_t face_budget = 0, certificate_budget = 0, scalar_budget = 0;
        std::vector<VoF::ExactOwnedFaceRate> old_rates;
        std::vector<VoF::ExactOwnedCellVelocity> seeds;
        candidate->collective(
            [&]
            {
                const auto source_faces = mesh->num_owned_faces(), target_faces = candidate->mesh->num_owned_faces();
                if (old.projection_arithmetic.face_rate.size() != source_faces)
                    throw std::runtime_error("Missing authoritative source face rates for fixed-clock handoff.");
                handoff_storage.capacity(Domain::Input, old.projection_arithmetic.face_rate.capacity(), sizeof(M));
                for (const auto& q : old.projection_arithmetic.face_rate)
                    handoff_storage.measure(Domain::Input, q);
                handoff_storage.capacity(Domain::Scratch, source_faces, sizeof(VoF::ExactOwnedFaceRate));
                handoff_storage.capacity(
                    Domain::Scratch, migration ? 0 : old.velocity.size(), sizeof(VoF::ExactOwnedCellVelocity));
                if (!migration)
                    for (const auto& key : old.phases->owned_order())
                        handoff_storage.capacity(Domain::Scratch, Meshes::AMR::encode_adaptive_key(key).size(), 1);
                handoff_storage.capacity(
                    Domain::Result, target_faces, sizeof(M) + sizeof(detail::GeneralVoFRoundedValue));
                handoff_storage.capacity(Domain::Scratch, target_faces, sizeof(M) + sizeof(unsigned char));
                handoff_storage.capacity(Domain::Result, candidate->mesh->num_faces(), sizeof(double));
                // Preserve one admitted half for the face-transfer result throughout
                // validation; the other covers caller views and exact closure proof.
                face_budget = handoff_storage.remaining() / 2;
                handoff_storage.capacity(Domain::Scratch, face_budget, 1);
                certificate_budget = handoff_storage.remaining() / 2;
                handoff_storage.capacity(Domain::Scratch, certificate_budget, 1);
                if (target_faces > (std::numeric_limits<std::size_t>::max() - 32) / 4)
                    throw std::runtime_error("Fixed-clock face handoff scalar count overflow.");
                const auto slots = 4 * target_faces + 32;
                scalar_budget = handoff_storage.remaining() / slots;
                if (face_budget < 1024 || certificate_budget < 1024 || scalar_budget < 1024)
                    throw std::runtime_error("Fixed-clock face handoff cumulative reserve exhausted.");
                handoff_storage.capacity(Domain::Scratch, slots, scalar_budget);
                old_rates.reserve(source_faces);
                const auto lease = mesh->acquire_execution_view();
                for (std::size_t f = 0; f < source_faces; ++f)
                    old_rates.push_back(
                        {std::uint64_t(mesh->face_global_id(LO(f))), old.projection_arithmetic.face_rate[f]});
                if (!migration)
                {
                    seeds.reserve(old.velocity.size());
                    for (std::size_t c = 0; c < old.velocity.size(); ++c)
                        seeds.push_back({old.phases->owned_order()[c], old.velocity[c]});
                }
            });
        VoF::ExactFaceTransferOptions face_options;
        face_options.byte_budget = face_budget;
        const auto transferred_faces =
            VoF::ExactFaceTransfer::prepare(*old.phases, *candidate_phase, transferred, old_rates, seeds, face_options);
        transferred_faces.validate_observations(*old.phases, *candidate_phase);
        std::vector<M> zero_radius;
        candidate->collective(
            [&]
            {
                const auto count = candidate->mesh->num_owned_faces();
                state->projection_arithmetic.byte_budget = options.byte_budget;
                state->projection_arithmetic.face_rate.resize(count);
                state->projection_arithmetic.face_velocity.resize(count);
                state->face_velocity.resize(candidate->mesh->num_faces());
                zero_radius.resize(count);
                std::vector<unsigned char> seen(count);
                detail::GeneralVoFProjectionArithmetic arithmetic(scalar_budget);
                if (transferred_faces.owned_rates().size() != count)
                    throw std::runtime_error("Fixed-clock face handoff has incomplete actual target coverage.");
                const auto lease = candidate->mesh->acquire_execution_view();
                for (const auto& item : transferred_faces.owned_rates())
                {
                    const auto f = candidate->mesh->owned_face_map()->getLocalElement(GO(item.face));
                    if (f < 0 || seen[f]++)
                        throw std::runtime_error("Foreign/duplicate target face in fixed-clock handoff.");
                    if (candidate->mesh->is_geometry_exterior_face(f) && item.rate.sign())
                        throw std::runtime_error("Fixed-clock handoff changes a closed physical wall rate.");
                    const auto view = detail::vof_exact::leased_divide(
                        item.rate, M::from_value(candidate->mesh->face_area(f)), scalar_budget)
                                          .enclosure();
                    const double velocity = static_cast<double>(view.value);
                    if (!std::isfinite(velocity))
                        throw std::runtime_error("Fixed-clock face diagnostic overflows.");
                    state->projection_arithmetic.face_rate[f] = item.rate;
                    // This leaf describes the diagnostic view only. Radius zero
                    // belongs to retained EXACT q, not to a rounded q/area product.
                    state->projection_arithmetic.face_velocity[f] = arithmetic.leaf(velocity);
                    state->face_velocity[f] = velocity;
                }
                state->pressure = std::move(mapped_pressure);
            });
        const auto proof =
            detail::certify_general_vof_projection_rate(candidate->mesh, state->projection_arithmetic.face_velocity,
                certificate_budget, &state->projection_arithmetic.face_rate, &zero_radius);
        candidate->require(
            proof.arithmetic_admitted, "Transferred exact face rates do not close the actual target topology.");
        candidate->collective(
            [&]
            {
                for (const auto& q : proof.owned_net_rate)
                    if (q.sign())
                        throw std::runtime_error("Fixed-clock face handoff has nonzero exact recipient divergence.");
            });
        state->face_velocity = candidate->flow.faces(state->face_velocity);
        state->diagnostics = old.diagnostics;
        state->momentum_transport = old.momentum_transport;
        state->diagnostics.time = old.diagnostics.time;
        state->diagnostics.maximum_divergence = 0; // Proven on actual target faces.
        candidate->budget(*state);
        // The opaque remap proves exact channel columns. Final represented
        // velocity recovery retains its independent physical momentum gate.
        for (unsigned a = 0; a < 3; ++a)
        {
            std::vector<M> before, after, reference;
            collective(
                [&]
                {
                    for (std::size_t c = 0; c < old.velocity.size(); ++c)
                    {
                        const auto value = old.physical_mass[c].scaled(old.velocity[c][a]);
                        before.push_back(value);
                        reference.push_back(value.sign() < 0 ? value.scaled(-1) : value);
                    }
                    for (std::size_t c = 0; c < state->velocity.size(); ++c)
                    {
                        const auto value = state->physical_mass[c].scaled(state->velocity[c][a]);
                        after.push_back(value);
                        reference.push_back(value.sign() < 0 ? value.scaled(-1) : value);
                    }
                });
            const auto difference = global_sum(after) - global_sum(before);
            const auto magnitude = difference.sign() < 0 ? difference.scaled(-1) : difference;
            const auto limit = global_sum(reference).scaled(options.momentum_relative_tolerance);
            // This is a residual-vs-L1 bound, not a value-vs-reference relative
            // error. An exactly zero L1 reference requires an exact zero residual.
            require(magnitude.compare(limit) <= 0,
                "General VoF fixed-clock remap changed physical phase-mass-weighted momentum.");
        }
        transferred_faces.validate_observations(*old.phases, *candidate_phase);
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
