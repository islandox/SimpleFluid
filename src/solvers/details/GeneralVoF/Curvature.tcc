/** @file Curvature.tcc
 * @brief Compiled mode-selected distributed curvature assembly.
 */
#include "solvers/details/GeneralVoF/Curvature.hh"
#include "solvers/details/GeneralVoF/Execution.hh"
#include "solvers/details/GeneralVoFCurvature.hh"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace SimpleFluid::detail::general_vof
{
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
