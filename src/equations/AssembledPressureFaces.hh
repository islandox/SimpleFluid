/** @file AssembledPressureFaces.hh
 * @brief Cell matrix of compact pressure and signed gradient face contributions.
 */
#pragma once

#include "FVM/details/OperatorDetails.hh"
#include "equations/BoundaryConditions.hh"
#include "equations/CollectiveValidation.hh"

#include <Kokkos_Core.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Distributor.hpp>
#include <Tpetra_Import.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace SimpleFluid::detail
{

/** @brief Assemble K for [p, grad_x, grad_y, grad_z] with columns 4*cell_gid+component.
 * The caller validates the encoded GID range and supplies current face metrics.
 */
template<class Pack, class Mesh, class Topology, class Metrics>
auto assembled_pressure_face_matrix(const Mesh& mesh, const Topology& topology, const Metrics& metrics,
    const BoundaryConditionMap& correction_bc, const std::optional<typename Pack::global_ordinal_type>& gauge,
    const typename Pack::matrix_type& compact, const Teuchos::RCP<const typename Pack::map_type>& channel_map,
    const typename Pack::import_type& importer)
    -> Teuchos::RCP<typename Pack::matrix_type>
{
    using local_type = typename Pack::local_ordinal_type;
    using global_type = typename Pack::global_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using matrix_type = typename Pack::matrix_type;

    constexpr auto invalid_gid = std::numeric_limits<global_type>::max();
    struct FaceRow
    {
        global_type owner = std::numeric_limits<global_type>::max();
        global_type neighbor = std::numeric_limits<global_type>::max();
        std::array<scalar_type, 3> owner_weights{}, neighbor_weights{};
    };
    static_assert(std::is_trivially_copyable_v<FaceRow>);

    const auto owned_cells = mesh.owned_cell_map();
    const auto overlap_cells = mesh.overlap_cell_map();
    const auto owned_faces = mesh.owned_face_map();
    const auto overlap_faces = mesh.overlap_face_map();
    const auto compact_rows = compact.getRowMap();
    const auto compact_columns = compact.getColMap();
    std::vector<FaceRow> owned;

    collective_detail::collective_local_validation(mesh, "Assembled pressure face rows", [&]
    {
        if (channel_map.is_null() || !compact.isFillComplete() || compact_rows.is_null() || compact_columns.is_null() ||
            compact.getLocalNumRows() != mesh.num_owned_cells() ||
            owned_cells->getLocalNumElements() != mesh.num_owned_cells() ||
            overlap_cells->getLocalNumElements() != mesh.num_cells() ||
            owned_faces->getLocalNumElements() != mesh.num_owned_faces() ||
            overlap_faces->getLocalNumElements() != mesh.num_faces())
            throw std::invalid_argument("Assembled pressure rows require matching compact and mesh maps.");
        if constexpr (requires { topology.require_valid(mesh); })
            topology.require_valid(mesh);
        if (topology.interior.size() != metrics.interior.size() ||
            topology.boundary.size() != metrics.boundary.size() ||
            topology.zero_flux_faces.size() != metrics.zero_flux_faces.size())
            throw std::invalid_argument("Assembled pressure face topology and metrics have different sizes.");

        owned.resize(owned_faces->getLocalNumElements());
        const auto cell_gid = [&](local_type lid)
        {
            if (!overlap_cells->isNodeLocalElement(lid))
                throw std::out_of_range("Assembled pressure face cell is outside the overlap map.");
            return overlap_cells->getGlobalElement(lid);
        };
        const auto add_face = [&](local_type face, local_type owner) -> FaceRow&
        {
            auto& row = owned.at(static_cast<std::size_t>(face));
            if (row.owner != invalid_gid)
                throw std::invalid_argument("Assembled pressure face topology has a duplicate owned face.");
            row.owner = cell_gid(owner);
            return row;
        };

        for (std::size_t index = 0; index < topology.interior.size(); ++index)
        {
            const auto& face = topology.interior[index];
            const auto& metric = metrics.interior[index];
            auto& row = add_face(face.face, face.owner);
            row.neighbor = cell_gid(face.neighbor);
            if (!face.owned_owner || metric.distance_squared <= scalar_type{})
                continue;
            const auto alpha = metric.center_projection / metric.distance_squared;
            const auto tangent = metric.area_vector - metric.center_delta * alpha;
            for (std::size_t component = 0; component < 3; ++component)
            {
                row.owner_weights[component] = metric.owner_weight * tangent.component(component);
                row.neighbor_weights[component] = metric.neighbor_weight * tangent.component(component);
            }
        }
        for (const auto& batch : topology.boundary_groups)
        {
            if (batch.begin > topology.boundary.size() || batch.count > topology.boundary.size() - batch.begin)
                throw std::out_of_range("Assembled pressure boundary batch is outside the face topology.");
            const auto condition = correction_bc.find(batch.name);
            const bool dirichlet = condition != correction_bc.end() &&
                                   condition->second.type == BoundaryConditionType::Dirichlet;
            for (std::size_t index = batch.begin; index < batch.begin + batch.count; ++index)
            {
                const auto& face = topology.boundary[index];
                const auto& metric = metrics.boundary[index];
                auto& row = add_face(face.face, face.owner);
                if (!dirichlet || !face.owned_owner)
                    continue;
                const auto tangent = metric.area_vector - metric.center_delta * metric.boundary_diffusion;
                for (std::size_t component = 0; component < 3; ++component)
                    row.owner_weights[component] = tangent.component(component);
            }
        }
        for (const auto& face : topology.zero_flux_faces)
            add_face(face.face, face.owner);
        for (const auto& row : owned)
            if (row.owner == invalid_gid)
                throw std::invalid_argument("Assembled pressure face topology is missing an owned face.");
    });

    const bool distributed = owned_faces->isDistributed();
    using byte_view = Kokkos::View<char*, Kokkos::HostSpace>;
    byte_view send, receive;
    std::vector<FaceRow> imports;
    std::vector<const FaceRow*> overlap;
    collective_detail::collective_local_validation(mesh, "Assembled pressure face import preparation", [&]
    {
        if (!importer.isLocallyComplete() || (!distributed && importer.getNumRemoteIDs() != 0))
            throw std::invalid_argument("Assembled pressure overlap face has no publishing rank.");
        constexpr auto max_packets = static_cast<std::size_t>(std::numeric_limits<int>::max()) / sizeof(FaceRow);
        if (importer.getNumExportIDs() > max_packets || importer.getNumRemoteIDs() > max_packets)
            throw std::overflow_error("Assembled pressure face import exceeds MPI byte counts or offsets.");

        overlap.resize(overlap_faces->getLocalNumElements());
        for (std::size_t same = 0; same < importer.getNumSameIDs(); ++same)
            overlap.at(same) = &owned.at(same);
        const auto from = importer.getPermuteFromLIDs();
        const auto to = importer.getPermuteToLIDs();
        for (std::size_t slot = 0; slot < importer.getNumPermuteIDs(); ++slot)
            overlap.at(to[slot]) = &owned.at(from[slot]);

        send = byte_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "pressure face exports"),
            importer.getNumExportIDs() * sizeof(FaceRow));
        receive = byte_view(Kokkos::view_alloc(Kokkos::WithoutInitializing, "pressure face imports"),
            importer.getNumRemoteIDs() * sizeof(FaceRow));
        std::size_t offset = 0;
        for (const auto lid : importer.getExportLIDs())
        {
            std::memcpy(send.data() + offset, &owned.at(lid), sizeof(FaceRow));
            offset += sizeof(FaceRow);
        }
        imports.resize(importer.getNumRemoteIDs());
        const auto remote = importer.getRemoteLIDs();
        for (std::size_t slot = 0; slot < imports.size(); ++slot)
            overlap.at(remote[slot]) = &imports[slot];
        for (const auto* row : overlap)
            if (row == nullptr)
                throw std::invalid_argument("Assembled pressure overlap face has no publishing rank.");
    });

    if (distributed)
    {
        importer.getDistributor().doPostsAndWaits(send, sizeof(FaceRow), receive);
        if (!imports.empty())
            std::memcpy(imports.data(), receive.data(), receive.extent(0));
    }

    send = byte_view{};
    receive = byte_view{};

    Teuchos::RCP<matrix_type> matrix;
    collective_detail::collective_local_validation(mesh, "Assembled pressure cell accumulation", [&]
    {
        const auto execution = acquire_mesh_execution(mesh);
        const auto compact_row = [&](global_type gid)
        {
            const auto lid = compact_rows->getLocalElement(gid);
            if (!compact_rows->isNodeLocalElement(lid))
                throw std::invalid_argument("Assembled pressure compact matrix is missing an owned cell row.");
            return lid;
        };
        const auto visit_faces = [&](local_type cell_lid, const auto& visitor)
        {
            const auto cell_id = FVM::detail::query_cell_id(mesh, cell_lid);
            for (const auto face_id : mesh.faces(cell_id))
            {
                const auto face_lid = FVM::detail::packed_face_local_id(mesh, face_id);
                visitor(face_id, *overlap.at(face_lid));
            }
        };

        Teuchos::Array<std::size_t> widths(mesh.num_owned_cells());
        std::vector<global_type> gradient_cells;
        for (std::size_t index = 0; index < mesh.num_owned_cells(); ++index)
        {
            const auto cell_lid = static_cast<local_type>(index);
            const auto gid = owned_cells->getGlobalElement(cell_lid);
            auto& width = widths[index];
            width = compact.getNumEntriesInLocalRow(compact_row(gid));
            if (gauge && gid == *gauge)
                continue;

            gradient_cells.clear();
            visit_faces(cell_lid, [&](const auto&, const FaceRow& face)
            {
                gradient_cells.push_back(face.owner);
                if (face.neighbor != invalid_gid)
                    gradient_cells.push_back(face.neighbor);
            });
            std::sort(gradient_cells.begin(), gradient_cells.end());
            const auto unique_end = std::unique(gradient_cells.begin(), gradient_cells.end());
            const auto count = static_cast<std::size_t>(unique_end - gradient_cells.begin());
            if (count > (std::numeric_limits<std::size_t>::max() - width) / 3)
                throw std::overflow_error("Assembled pressure row width overflows size_t.");
            width += 3 * count;
        }
        matrix = Teuchos::rcp(new matrix_type(owned_cells, widths()));

        std::map<global_type, scalar_type> row;
        Teuchos::Array<global_type> columns;
        Teuchos::Array<scalar_type> values;
        for (std::size_t index = 0; index < mesh.num_owned_cells(); ++index)
        {
            const auto cell_lid = static_cast<local_type>(index);
            const auto cell_gid = owned_cells->getGlobalElement(cell_lid);
            typename matrix_type::local_inds_host_view_type compact_indices;
            typename matrix_type::values_host_view_type compact_values;
            compact.getLocalRowView(compact_row(cell_gid), compact_indices, compact_values);
            row.clear();
            for (std::size_t entry = 0; entry < compact_indices.extent(0); ++entry)
            {
                const auto column = compact_indices(entry);
                if (!compact_columns->isNodeLocalElement(column))
                    throw std::out_of_range("Assembled pressure compact column is outside its column map.");
                row.emplace(global_type{4} * compact_columns->getGlobalElement(column), compact_values(entry));
            }
            if (!gauge || cell_gid != *gauge)
            {
                visit_faces(cell_lid, [&](const auto& face_id, const FaceRow& face)
                {
                    const auto owner = FVM::detail::packed_cell_local_id(mesh, mesh.owner_cell(face_id));
                    const auto sign = owner == cell_lid ? scalar_type{-1} : scalar_type{1};
                    const auto add_gradient = [&](global_type gid, const auto& weights)
                    {
                        const auto base = global_type{4} * gid + global_type{1};
                        for (std::size_t component = 0; component < weights.size(); ++component)
                            row[base + static_cast<global_type>(component)] += sign * weights[component];
                    };
                    add_gradient(face.owner, face.owner_weights);
                    if (face.neighbor != invalid_gid)
                        add_gradient(face.neighbor, face.neighbor_weights);
                });
            }

            columns.resize(row.size());
            values.resize(row.size());
            std::size_t slot = 0;
            for (const auto& [column, value] : row)
            {
                columns[slot] = column;
                values[slot] = value;
                ++slot;
            }
            if (!row.empty())
                matrix->insertGlobalValues(cell_gid, columns(), values());
        }
    });

    decltype(overlap){}.swap(overlap);
    decltype(imports){}.swap(imports);
    decltype(owned){}.swap(owned);
    matrix->fillComplete(channel_map, owned_cells);
    return matrix;
}

} // namespace SimpleFluid::detail
