/** @file Curvature.tcc
 * @brief Compiled mode-selected distributed curvature assembly.
 */
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoFCurvature.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace SimpleFluid::detail::general_vof
{
namespace curvature_face_detail
{
using Point = PLIC::GeometryPoint;
Point difference(Point a, Point b)
{
    for (unsigned axis = 0; axis < 3; ++axis)
        a[axis] -= b[axis];
    return a;
}
double dot(Point a, Point b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Point cross(Point a, Point b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
// This admits a represented planar face surface only. It does not fit a
// curved binary staircase or manufacture a zero-curvature mixed section.
void admit_planar(const MeshHandle<>& mesh, const VoF::GeneralVoF& phases,
    std::span<const VoF::GeneralVoFFaceInterface> faces, std::size_t byte_budget)
{
    using LO = DefaultTpetraTypes::local_ordinal_type;
    constexpr auto tolerance = 64 * std::numeric_limits<double>::epsilon();
    constexpr auto entry_bytes =
        sizeof(std::pair<const std::uint64_t, std::vector<std::size_t>>) + 4 * sizeof(void*) + 2 * sizeof(std::size_t);
    if (faces.size() > byte_budget / (2 * entry_bytes))
        throw std::runtime_error("General VoF face-surface index exceeds its byte budget.");
    // The catalog includes authenticated halo/halo faces. Index it once;
    // each owned target selects only its bounded topological support below.
    std::map<std::uint64_t, std::vector<std::size_t>> incident;
    for (std::size_t i = 0; i < faces.size(); ++i)
    {
        incident[faces[i].first_cell].push_back(i);
        incident[faces[i].second_cell].push_back(i);
    }
    for (std::size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto gid = std::uint64_t(mesh.cell_global_id(LO(c)));
        const auto targets = incident.find(gid);
        if (targets == incident.end())
            continue;
        const auto support = phases.cell_support(gid);
        std::set<std::size_t> selected;
        for (const auto cell : support)
            if (const auto found = incident.find(cell); found != incident.end())
                selected.insert(found->second.begin(), found->second.end());
        if (selected.size() > phases.options().maximum_stencil_cells ||
            selected.size() > byte_budget / (sizeof(std::size_t) + 4 * sizeof(void*)))
            throw std::runtime_error("General VoF face-surface support exceeds its byte budget.");
        for (const auto target : targets->second)
        {
            const auto& face = faces[target];
            const auto scale = std::sqrt(face.area);
            if (!(scale > 0) || !std::isfinite(scale))
                throw std::runtime_error("General VoF face-surface scale is invalid.");
            std::vector<Point> tangents;
            tangents.reserve(selected.size());
            for (const auto index : selected)
            {
                const auto& sample = faces[index];
                const auto displacement = difference(sample.centroid, face.centroid);
                const auto distance = std::hypot(displacement[0], displacement[1], displacement[2]);
                if (distance > 6 * scale)
                    continue;
                const auto normal_error = difference(sample.normal, face.normal);
                if (std::hypot(normal_error[0], normal_error[1], normal_error[2]) > tolerance ||
                    std::abs(dot(face.normal, displacement)) > tolerance * scale)
                    throw std::runtime_error("General VoF capillarity cannot admit a curved or stepped binary "
                                             "face-interface surface as a planar zero-curvature surface.");
                if (distance > tolerance * scale)
                    tangents.push_back(displacement);
            }
            bool supported = false;
            for (std::size_t i = 0; i < tangents.size() && !supported; ++i)
                for (std::size_t j = i + 1; j < tangents.size(); ++j)
                    if (std::abs(dot(face.normal, cross(tangents[i], tangents[j]))) > tolerance * scale * scale)
                    {
                        supported = true;
                        break;
                    }
            if (!supported)
                throw std::runtime_error(
                    "General VoF planar face-interface surface lacks two-dimensional local support.");
        }
    }
}
} // namespace curvature_face_detail
template<ArithmeticMode Mode>
std::vector<double> Curvature<Mode>::assemble(
    const MeshHandle<>& mesh_view, const VoF::GeneralVoF& phases, double surface_tension, std::size_t byte_budget)
{
    using Pack = DefaultTpetraTypes;
    using LO = Pack::local_ordinal_type;
    using GO = Pack::global_ordinal_type;
    const auto* mesh = &mesh_view;
    const auto owned_count = mesh->num_owned_cells();
    const Execution execution(mesh_view);
    std::vector<double> curvature;
    curvature.assign(owned_count, 0);
    if (surface_tension == 0)
        return curvature;
    if constexpr (Mode == ArithmeticMode::Ordinary)
    {
        if (phases.options().transport_backend != VoF::GeneralVoFTransportBackend::SweptPartition)
        {
            int local_mixed = 0, global_mixed = 0;
            execution.collective(
                [&]
                {
                    const auto declarations = phases.phase_geometry_declarations();
                    const auto& amounts = phases.phase_volumes();
                    for (std::size_t c = 0; c < amounts.size(); ++c)
                        if (declarations[c] == VoF::GeneralVoFPhaseGeometry::Infer && amounts[c].gas > 0 &&
                            amounts[c].carrier > 0)
                            local_mixed = 1;
                });
            Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MAX, 1, &local_mixed, &global_mixed);
            if (!global_mixed)
            {
                const auto faces = phases.face_interfaces();
                execution.collective([&] { curvature_face_detail::admit_planar(*mesh, phases, faces, byte_budget); });
                return curvature;
            }
        }
    }
    const auto patches = phases.reconstruction_patches();
    std::map<GO, int> phase_geometry;
    for (const auto& patch : patches)
        phase_geometry.emplace(GO(patch.cell), patch.interface.pure_gas ? 1 : patch.interface.pure_carrier ? -1 : 0);
    execution.collective(
        [&]
        {
            const auto lease = mesh->acquire_execution_view();
            for (std::size_t c = 0; c < owned_count; ++c)
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
    std::vector<GeneralVoFCurvaturePatch> surface;
    execution.collective(
        [&]
        {
            for (const auto& patch : patches)
            {
                const auto sample = [&]
                {
                    if constexpr (Mode == ArithmeticMode::Exact)
                        return GeneralVoFCurvature::patch<ArithmeticMode::Exact>(patch.cell,
                            *PLIC::ConvexCellGeometry<ArithmeticMode::Exact>::from_geometry(*patch.geometry),
                            patch.interface);
                    else
                        return GeneralVoFCurvature::patch<Mode>(patch.cell,
                            *PLIC::ConvexCellGeometry<Mode>::from_dynamic(patch.finite_geometry), patch.interface);
                }();
                if (sample)
                    surface.push_back(*sample);
            }
            constexpr auto index_entry_bytes = sizeof(std::pair<const std::uint64_t, std::size_t>) + 4 * sizeof(void*);
            if (surface.size() > byte_budget / index_entry_bytes)
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
                    const auto support = phases.cell_support(patch.cell);
                    if (support.size() > (byte_budget - index_bytes) / sizeof(std::size_t))
                        throw std::runtime_error("General VoF curvature support index exceeds its byte budget.");
                    std::vector<std::size_t> selected;
                    selected.reserve(support.size());
                    for (const auto cell : support)
                        if (const auto found = surface_index.find(cell); found != surface_index.end())
                            selected.push_back(found->second);
                    // Retain the original surface order: curvature fits sum
                    // floating-point stencil contributions in this order.
                    std::sort(selected.begin(), selected.end());
                    std::vector<GeneralVoFCurvaturePatch> stencil;
                    stencil.reserve(selected.size());
                    for (const auto index : selected)
                        stencil.push_back(surface[index]);
                    curvature[local] = GeneralVoFCurvature::fit(patch, stencil).curvature;
                }
            }
        });
    Teuchos::Array<GO> ids;
    for (const auto& patch : surface)
        ids.push_back(GO(patch.cell));
    auto patch_map = Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), GO{}, mesh->owned_cell_map()->getComm()));
    const auto k = execution.import_owned(curvature, patch_map);
    std::set<GO> mixed_cells;
    for (const auto& patch : surface)
        mixed_cells.insert(GO(patch.cell));
    // Curvature is fitted only at actual mixed PLIC sections. Extend that
    // fitted value to neighboring pure cells for the balanced face force;
    // no manufactured normal or pure-cell curvature fit is substituted.
    execution.collective(
        [&]
        {
            for (std::size_t c = 0; c < owned_count; ++c)
            {
                const auto gid = mesh->cell_global_id(LO(c));
                if (mixed_cells.contains(gid))
                    continue;
                const auto support = phases.cell_support(std::uint64_t(gid));
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
                        curvature[c] = k[p];
                    }
                }
            }
        });
    return curvature;
}
} // namespace SimpleFluid::detail::general_vof
