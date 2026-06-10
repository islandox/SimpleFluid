/**
 * @file MeshHandle.hh
 * @brief Variant-backed runtime handle for all supported mesh families.
 */

#pragma once

#include "dataclass/TpetraTypes.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/OrthogonalCylindrial3D.hh"
#include "geometry/mesh/OrthoMeshPartitioner.hh"
#include "geometry/mesh/STKMeshAdapter.hh"
#include "geometry/mesh/SemiStructuredXY_Z.hh"
#include "io/VTUWriter.hh"
#include "utils/debug_check.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Runtime-polymorphic distributed view of supported mesh families.
 *
 * MeshHandle normalizes structured, semi-structured, and legacy STK meshes
 * into one FVM-facing API. It builds owned/overlap maps and translates compact
 * local IDs to each concrete mesh's geometry IDs.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class MeshHandle
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using map_type = typename Pack::map_type;
    using Vec3 = MeshUtils::Vec3;

    using Cartesian = Meshes::OrthogonalCartesian3D;
    using Cylindrical = Meshes::OrthogonalCylindrial3D;
    using SemiStructured = Meshes::SemiStructuredXY_Z;
    using STKAdapter = Meshes::STKMeshAdapter<Pack>;

    using CartesianPtr = SP<const Cartesian>;
    using CylindricalPtr = SP<const Cylindrical>;
    using SemiStructuredPtr = SP<const SemiStructured>;
    using STKAdapterPtr = SP<const STKAdapter>;
    using variant_type = std::variant<CartesianPtr,
                                      CylindricalPtr,
                                      SemiStructuredPtr,
                                      STKAdapterPtr>;

    /** @brief Locally visible faces belonging to one boundary patch. */
    struct BoundaryFacePatch
    {
        int id = -1;
        std::vector<local_ordinal_type> face_lids;
    };

    /** @brief Partition selection and ghost depth for structured meshes. */
    struct DistributionOptions
    {
        size_t ghost_layers = 1;
        std::optional<size_t> partition;
        std::optional<size_t> partitions;
    };

    static constexpr int invalid_boundary_id = -1;

    /** @brief Build a distributed handle for a Cartesian mesh. */
    explicit MeshHandle(CartesianPtr mesh,
                        DistributionOptions options = {})
        : d_mesh(require_mesh(std::move(mesh)))
    {
        initialize_orthogonal(std::get<CartesianPtr>(d_mesh), options);
    }

    /** @brief Build a distributed handle for a cylindrical mesh. */
    explicit MeshHandle(CylindricalPtr mesh,
                        DistributionOptions options = {})
        : d_mesh(require_mesh(std::move(mesh)))
    {
        initialize_orthogonal(std::get<CylindricalPtr>(d_mesh), options);
    }

    /** @brief Build a handle for a semi-structured mesh. */
    explicit MeshHandle(SemiStructuredPtr mesh)
        : d_mesh(require_mesh(std::move(mesh)))
    {
        initialize_semi_structured(std::get<SemiStructuredPtr>(d_mesh));
    }

    /** @brief Build a handle around a legacy STK adapter. */
    explicit MeshHandle(STKAdapterPtr mesh)
        : d_mesh(require_mesh(std::move(mesh)))
    {
        initialize_stk(std::get<STKAdapterPtr>(d_mesh));
    }

    /** @brief Adapt a legacy distributed mesh and build a runtime handle. */
    explicit MeshHandle(SP<const SimpleFluid::Mesh<Pack>> mesh)
        : MeshHandle(std::make_shared<STKAdapter>(std::move(mesh)))
    {
    }

    const variant_type& variant() const noexcept { return d_mesh; }

    /**
     * @brief Invoke a visitor with the concrete mesh object.
     *
     * Pointer alternatives are dereferenced before dispatch.
     */
    template<class Visitor>
    decltype(auto) visit(Visitor&& visitor) const
    {
        return std::visit(
            [&](const auto& mesh) -> decltype(auto)
            {
                return std::forward<Visitor>(visitor)(*mesh);
            },
            d_mesh);
    }

    bool is_stk() const noexcept
    {
        return std::holds_alternative<STKAdapterPtr>(d_mesh);
    }

    SP<const SimpleFluid::Mesh<Pack>> legacy_mesh() const noexcept
    {
        if (const auto* adapter = std::get_if<STKAdapterPtr>(&d_mesh))
        {
            return (*adapter)->mesh_ptr();
        }
        return {};
    }

    size_t spatial_dimension() const noexcept { return 3; }
    size_t num_owned_cells() const noexcept { return d_num_owned_cells; }
    size_t num_local_cells() const noexcept { return d_cell_geometry_lids.size(); }
    size_t num_cells() const noexcept { return num_local_cells(); }
    size_t num_owned_faces() const noexcept { return d_num_owned_faces; }
    size_t num_faces() const noexcept { return d_face_geometry_lids.size(); }

    bool is_owned_cell(local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        return static_cast<size_t>(cell_lid) < d_num_owned_cells;
    }

    bool is_owned_face(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        return static_cast<size_t>(face_lid) < d_num_owned_faces;
    }

    global_ordinal_type cell_global_id(local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        if (const auto legacy = legacy_mesh())
        {
            return legacy->mesh_gid_to_tpetra_gid(
                legacy->cell_global_id(geometry_cell_lid(cell_lid)));
        }
        return static_cast<global_ordinal_type>(
            d_cell_geometry_lids[static_cast<size_t>(cell_lid)]);
    }

    global_ordinal_type face_global_id(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        if (const auto legacy = legacy_mesh())
        {
            return legacy->face_global_id(
                geometry_face_lid(face_lid));
        }
        return static_cast<global_ordinal_type>(
            d_face_geometry_lids[static_cast<size_t>(face_lid)]);
    }

    real_t cell_volume(local_ordinal_type cell_lid) const
    {
        return visit_geometry_cell(
            cell_lid,
            [](const auto& mesh, const auto id)
            {
                return mesh.cell_volume(id);
            });
    }

    Vec3 cell_centroid(local_ordinal_type cell_lid) const
    {
        return visit_geometry_cell(
            cell_lid,
            [](const auto& mesh, const auto id)
            {
                return mesh.cell_centroid(id);
            });
    }

    std::span<const local_ordinal_type> faces(
        local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        const auto local = static_cast<size_t>(cell_lid);
        const auto begin = d_cell_face_offsets[local];
        const auto end = d_cell_face_offsets[local + 1];
        return std::span<const local_ordinal_type>(d_cell_face_lids)
            .subspan(begin, end - begin);
    }

    local_ordinal_type owner_cell(local_ordinal_type face_lid) const
    {
        return adjacent_cell(face_lid, true);
    }

    local_ordinal_type neighbor_cell(local_ordinal_type face_lid) const
    {
        return adjacent_cell(face_lid, false);
    }

    local_ordinal_type opposite_cell(local_ordinal_type face_lid,
                                     local_ordinal_type cell_lid) const
    {
        const auto owner = owner_cell(face_lid);
        const auto neighbor = neighbor_cell(face_lid);
        if (cell_lid == owner)
        {
            return neighbor;
        }
        if (neighbor != invalid_local_id() && cell_lid == neighbor)
        {
            return owner;
        }
        throw std::invalid_argument(
            "Cell is not adjacent to requested face.");
    }

    local_ordinal_type opposite_or_periodic_neighbor_cell(
        local_ordinal_type face_lid,
        local_ordinal_type cell_lid) const
    {
        return opposite_cell(face_lid, cell_lid);
    }

    real_t face_area(local_ordinal_type face_lid) const
    {
        return visit_geometry_face(
            face_lid,
            [](const auto& mesh, const auto id)
            {
                return mesh.face_area(id);
            });
    }

    Vec3 face_centroid(local_ordinal_type face_lid) const
    {
        return visit_geometry_face(
            face_lid,
            [](const auto& mesh, const auto id)
            {
                return mesh.face_centroid(id);
            });
    }

    Vec3 face_normal(local_ordinal_type face_lid) const
    {
        return visit_geometry_face(
            face_lid,
            [](const auto& mesh, const auto id)
            {
                return mesh.face_normal(id);
            });
    }

    Vec3 face_area_vector(local_ordinal_type face_lid) const
    {
        return face_normal(face_lid) * face_area(face_lid);
    }

    Vec3 face_normal_outward(local_ordinal_type face_lid,
                             local_ordinal_type cell_lid) const
    {
        const auto owner = owner_cell(face_lid);
        if (cell_lid == owner)
        {
            return face_normal(face_lid);
        }
        if (cell_lid == neighbor_cell(face_lid))
        {
            return face_normal(face_lid) * -1.0;
        }
        throw std::invalid_argument(
            "Cell is not adjacent to requested face.");
    }

    Vec3 face_area_vector_outward(local_ordinal_type face_lid,
                                  local_ordinal_type cell_lid) const
    {
        return face_normal_outward(face_lid, cell_lid)
             * face_area(face_lid);
    }

    real_t face_cell_center_distance(local_ordinal_type face_lid) const
    {
        const auto neighbor = neighbor_cell(face_lid);
        if (neighbor == invalid_local_id())
        {
            return 0.0;
        }
        return (cell_centroid(neighbor)
              - cell_centroid(owner_cell(face_lid))).norm();
    }

    Vec3 cell_center_vector(local_ordinal_type face_lid,
                            local_ordinal_type cell_lid) const
    {
        const auto other = opposite_cell(face_lid, cell_lid);
        if (other == invalid_local_id())
        {
            throw std::invalid_argument(
                "Exterior face does not have an opposite cell.");
        }
        return cell_centroid(other) - cell_centroid(cell_lid);
    }

    real_t cell_to_face_distance(local_ordinal_type face_lid,
                                 local_ordinal_type cell_lid) const
    {
        return (face_centroid(face_lid) - cell_centroid(cell_lid)).norm();
    }

    bool is_exterior_face(local_ordinal_type face_lid) const
    {
        return neighbor_cell(face_lid) == invalid_local_id();
    }

    bool is_interior_face(local_ordinal_type face_lid) const
    {
        return !is_exterior_face(face_lid);
    }

    int boundary_id(local_ordinal_type face_lid) const
    {
        return visit_geometry_face(
            face_lid,
            [](const auto& mesh, const auto id)
            {
                return mesh.boundary_id(id);
            });
    }

    bool is_boundary_face(local_ordinal_type face_lid) const
    {
        return is_exterior_face(face_lid)
            && boundary_id(face_lid) != invalid_boundary_id;
    }

    const std::string& boundary_patch_name(int patch_id) const
    {
        const auto iter = d_boundary_names.find(patch_id);
        if (iter == d_boundary_names.end())
        {
            throw std::out_of_range("Unknown boundary patch ID.");
        }
        return iter->second;
    }

    const BoundaryFacePatch& boundary_face_patch(int patch_id) const
    {
        return d_boundary_patches.at(patch_id);
    }

    const std::unordered_map<int, BoundaryFacePatch>&
    boundary_patches() const noexcept
    {
        return d_boundary_patches;
    }

    Teuchos::RCP<const map_type> owned_cell_map() const
    {
        return d_owned_cell_map;
    }

    Teuchos::RCP<const map_type> overlap_cell_map() const
    {
        return d_overlap_cell_map;
    }

    Teuchos::RCP<const map_type> owned_face_map() const
    {
        return d_owned_face_map;
    }

    Teuchos::RCP<const map_type> overlap_face_map() const
    {
        return d_overlap_face_map;
    }

    Teuchos::RCP<const map_type> boundary_face_map() const
    {
        return d_boundary_face_map;
    }

    static constexpr local_ordinal_type invalid_local_id() noexcept
    {
        return Teuchos::OrdinalTraits<local_ordinal_type>::invalid();
    }

    template<class StoredField>
    void sync_periodic_boundaries(StoredField& field) const
    {
        field.sync_ghosts();
    }

    /**
     * @brief Export owned cells to a rank-specific VTU file.
     *
     * Legacy STK meshes delegate to their native exporter; other mesh families
     * are converted through VTUWriter.
     */
    void export_vtu(const std::string& filename) const
    {
        visit(
            [&](const auto& mesh)
            {
                using mesh_type = std::decay_t<decltype(mesh)>;
                if constexpr (std::is_same_v<mesh_type, STKAdapter>)
                {
                    mesh.export_vtu(filename);
                }
                else if constexpr (
                    std::is_same_v<mesh_type, SemiStructured>)
                {
                    export_semi_structured_vtu(mesh, filename);
                }
                else
                {
                    export_orthogonal_vtu(mesh, filename);
                }
            });
    }

private:
    template<class Pointer>
    static Pointer require_mesh(Pointer mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "MeshHandle requires a non-null mesh.");
        }
        return mesh;
    }

    static local_ordinal_type checked_local(size_t value)
    {
        if (value > static_cast<size_t>(
                std::numeric_limits<local_ordinal_type>::max()))
        {
            throw std::overflow_error(
                "MeshHandle local ordinal overflow.");
        }
        return static_cast<local_ordinal_type>(value);
    }

    std::string local_output_filename(
        const std::string& filename) const
    {
        const auto comm = Tpetra::getDefaultComm();
        if (comm->getSize() <= 1)
        {
            return filename;
        }

        const auto suffix =
            "_rank" + std::to_string(comm->getRank());
        const auto dot = filename.rfind('.');
        return dot == std::string::npos
             ? filename + suffix
             : filename.substr(0, dot) + suffix
                 + filename.substr(dot);
    }

    void add_geometry_cell_data(VTUWriter& writer) const
    {
        VTUWriter::Int64Data cell_ids;
        VTUWriter::ScalarData cell_volumes;
        VTUWriter::VectorData cell_centroids;
        cell_ids.reserve(d_num_owned_cells);
        cell_volumes.reserve(d_num_owned_cells);
        cell_centroids.reserve(d_num_owned_cells);
        for (size_t lid = 0; lid < d_num_owned_cells; ++lid)
        {
            const auto local_id = checked_local(lid);
            cell_ids.push_back(static_cast<global_index_t>(
                cell_global_id(local_id)));
            cell_volumes.push_back(cell_volume(local_id));
            cell_centroids.push_back(cell_centroid(local_id));
        }

        writer.add_int64_cell_data(
            "cell_gid", std::move(cell_ids));
        writer.add_scalar_cell_data(
            "cell_volume", std::move(cell_volumes));
        writer.add_vector_cell_data(
            "cell_centroid", std::move(cell_centroids));
    }

    template<class MeshType>
    VTUWriter::VectorData collect_vtu_points(
        const MeshType& mesh) const
    {
        VTUWriter::VectorData points;
        points.reserve(mesh.num_nodes());
        for (size_t lid = 0; lid < mesh.num_nodes(); ++lid)
        {
            points.push_back(mesh.node_coordinates(lid));
        }
        return points;
    }

    void write_vtu(
        const std::string& filename,
        VTUWriter::VectorData points,
        VTUWriter::Int64Data connectivity,
        VTUWriter::Int64Data offsets,
        VTUWriter::UInt8Data cell_types) const
    {
        VTUWriter writer;
        writer.set_points(std::move(points));
        writer.set_cells(
            std::move(connectivity),
            std::move(offsets),
            std::move(cell_types));
        add_geometry_cell_data(writer);
        writer.write(local_output_filename(filename));
    }

    template<class MeshType>
    void export_orthogonal_vtu(
        const MeshType& mesh,
        const std::string& filename) const
    {
        VTUWriter::Int64Data connectivity;
        VTUWriter::Int64Data offsets;
        VTUWriter::UInt8Data cell_types;
        connectivity.reserve(d_num_owned_cells * 8);
        offsets.reserve(d_num_owned_cells);
        cell_types.reserve(d_num_owned_cells);

        constexpr std::array<std::array<unsigned, 3>, 8> corners{{
            {{0, 0, 0}},
            {{1, 0, 0}},
            {{1, 1, 0}},
            {{0, 1, 0}},
            {{0, 0, 1}},
            {{1, 0, 1}},
            {{1, 1, 1}},
            {{0, 1, 1}}
        }};
        const auto& indexer = mesh.indexer();
        for (size_t lid = 0; lid < d_num_owned_cells; ++lid)
        {
            const auto cell = mesh.cell_id(
                d_cell_geometry_lids[lid]);
            for (const auto& corner : corners)
            {
                const typename MeshType::NodeID node{
                    static_cast<unsigned>(
                        (cell.i + corner[0])
                        % indexer.num_nodes_per_dim[0]),
                    static_cast<unsigned>(
                        (cell.j + corner[1])
                        % indexer.num_nodes_per_dim[1]),
                    static_cast<unsigned>(
                        (cell.k + corner[2])
                        % indexer.num_nodes_per_dim[2])};
                connectivity.push_back(static_cast<global_index_t>(
                    mesh.node_local_id(node)));
            }
            offsets.push_back(static_cast<global_index_t>(
                connectivity.size()));
            cell_types.push_back(12); // VTK_HEXAHEDRON
        }

        write_vtu(
            filename,
            collect_vtu_points(mesh),
            std::move(connectivity),
            std::move(offsets),
            std::move(cell_types));
    }

    void export_semi_structured_vtu(
        const SemiStructured& mesh,
        const std::string& filename) const
    {
        VTUWriter::Int64Data connectivity;
        VTUWriter::Int64Data offsets;
        VTUWriter::UInt8Data cell_types;
        offsets.reserve(d_num_owned_cells);
        cell_types.reserve(d_num_owned_cells);
        const auto& indexer = mesh.indexer();
        for (size_t lid = 0; lid < d_num_owned_cells; ++lid)
        {
            const auto cell = mesh.cell_id(
                d_cell_geometry_lids[lid]);
            const auto& xy_nodes =
                mesh.xy_cell_nodes()[cell.ij];
            const auto top_layer =
                (cell.k + 1) % indexer.num_node_layers;
            for (const auto xy_node : xy_nodes)
            {
                connectivity.push_back(static_cast<global_index_t>(
                    mesh.node_local_id(
                        SemiStructured::NodeID{xy_node, cell.k})));
            }
            for (const auto xy_node : xy_nodes)
            {
                connectivity.push_back(static_cast<global_index_t>(
                    mesh.node_local_id(
                        SemiStructured::NodeID{xy_node, top_layer})));
            }

            offsets.push_back(static_cast<global_index_t>(
                connectivity.size()));
            if (xy_nodes.size() == 3)
            {
                cell_types.push_back(13); // VTK_WEDGE
            }
            else if (xy_nodes.size() == 4)
            {
                cell_types.push_back(12); // VTK_HEXAHEDRON
            }
            else
            {
                cell_types.push_back(41); // VTK_CONVEX_POINT_SET
            }
        }

        write_vtu(
            filename,
            collect_vtu_points(mesh),
            std::move(connectivity),
            std::move(offsets),
            std::move(cell_types));
    }

    local_ordinal_type geometry_cell_lid(
        local_ordinal_type local_id) const
    {
        check_cell(local_id);
        return checked_local(
            d_cell_geometry_lids[static_cast<size_t>(local_id)]);
    }

    local_ordinal_type geometry_face_lid(
        local_ordinal_type local_id) const
    {
        check_face(local_id);
        return checked_local(
            d_face_geometry_lids[static_cast<size_t>(local_id)]);
    }

    template<class Function>
    decltype(auto) visit_geometry_cell(
        local_ordinal_type cell_lid,
        Function&& function) const
    {
        const auto geometry_lid = geometry_cell_lid(cell_lid);
        return visit(
            [&](const auto& mesh) -> decltype(auto)
            {
                return std::forward<Function>(function)(
                    mesh,
                    mesh.cell_id(static_cast<size_t>(geometry_lid)));
            });
    }

    template<class Function>
    decltype(auto) visit_geometry_face(
        local_ordinal_type face_lid,
        Function&& function) const
    {
        const auto geometry_lid = geometry_face_lid(face_lid);
        return visit(
            [&](const auto& mesh) -> decltype(auto)
            {
                return std::forward<Function>(function)(
                    mesh,
                    mesh.face_id(static_cast<size_t>(geometry_lid)));
            });
    }

    local_ordinal_type adjacent_cell(local_ordinal_type face_lid,
                                     bool owner) const
    {
        const auto geometry_lid = geometry_face_lid(face_lid);
        const auto geometry_cell = visit(
            [&](const auto& mesh) -> size_t
            {
                const auto face =
                    mesh.face_id(static_cast<size_t>(geometry_lid));
                const auto cell = owner
                    ? mesh.owner_cell(face)
                    : mesh.neighbor_cell(face);
                if constexpr (std::is_same_v<
                                  std::decay_t<decltype(mesh)>,
                                  STKAdapter>)
                {
                    if (cell == invalid_id<local_ordinal_type>())
                    {
                        return std::numeric_limits<size_t>::max();
                    }
                }
                else if (cell == std::decay_t<decltype(mesh)>::invalid_cell_id())
                {
                    return std::numeric_limits<size_t>::max();
                }
                return static_cast<size_t>(mesh.cell_local_id(cell));
            });

        if (geometry_cell == std::numeric_limits<size_t>::max())
        {
            return invalid_local_id();
        }
        const auto iter = d_cell_local_by_geometry.find(geometry_cell);
        return iter == d_cell_local_by_geometry.end()
             ? invalid_local_id()
             : iter->second;
    }

    template<class MeshType>
    void initialize_orthogonal(SP<const MeshType> mesh,
                               DistributionOptions options)
    {
        const auto comm = Tpetra::getDefaultComm();
        const auto rank = options.partition.value_or(
            static_cast<size_t>(comm->getRank()));
        const auto ranks = options.partitions.value_or(
            static_cast<size_t>(comm->getSize()));
        if (options.partition.has_value()
            != options.partitions.has_value())
        {
            throw std::invalid_argument(
                "Mesh distribution override requires both partition "
                "and partition count.");
        }
        if (rank >= ranks)
        {
            throw std::invalid_argument(
                "Mesh distribution partition index is out of range.");
        }

        if (ranks == 1)
        {
            initialize_serial(*mesh);
            return;
        }

        const auto& dimensions =
            mesh->indexer().num_cells_per_dim;
        size_t coordinate = 0;
        for (size_t candidate = 1; candidate < 3; ++candidate)
        {
            if (dimensions[candidate] > dimensions[coordinate])
            {
                coordinate = candidate;
            }
        }
        if (ranks > dimensions[coordinate])
        {
            throw std::invalid_argument(
                "Orthogonal mesh has fewer cells than MPI ranks along "
                "its largest coordinate.");
        }

        Meshes::OrthoMeshPartitioner partitioner(
            mesh->topology(),
            static_cast<typename Meshes::OrthoMeshPartitioner::Dimension>(
                coordinate),
            ranks,
            static_cast<Meshes::OrthoMeshPartitioner::Ordinal>(
                options.ghost_layers));

        std::vector<size_t> owned_cells;
        std::vector<size_t> ghost_cells;
        for (const auto id : partitioner.owned_cells(rank))
        {
            owned_cells.push_back(mesh->cell_local_id(id));
        }
        for (const auto id : partitioner.ghost_cells(rank))
        {
            ghost_cells.push_back(mesh->cell_local_id(id));
        }
        initialize_cells(std::move(owned_cells), std::move(ghost_cells));

        std::unordered_set<size_t> seen_faces;
        std::vector<size_t> owned_faces;
        std::vector<size_t> overlap_faces;
        for (const auto cell_geometry_lid : d_cell_geometry_lids)
        {
            const typename MeshType::cell_id_t typed_cell =
                mesh->indexer().cell_id(cell_geometry_lid);
            const auto typed_faces = mesh->cell_faces(typed_cell);
            for (const auto face : typed_faces)
            {
                const auto face_geometry_lid =
                    mesh->indexer().face_local_id(face);
                if (!seen_faces.insert(face_geometry_lid).second)
                {
                    continue;
                }
                if (partitioner.is_owned_face(rank, face))
                {
                    owned_faces.push_back(face_geometry_lid);
                }
                else
                {
                    overlap_faces.push_back(face_geometry_lid);
                }
            }
        }
        initialize_faces(
            std::move(owned_faces), std::move(overlap_faces));
        initialize_cell_faces();
        initialize_boundary_patches(*mesh);
        create_maps(comm);
    }

    void initialize_semi_structured(SemiStructuredPtr mesh)
    {
        const auto comm = Tpetra::getDefaultComm();
        if (comm->getSize() != 1)
        {
            throw std::runtime_error(
                "SemiStructuredXY_Z does not yet support multi-rank "
                "distribution.");
        }
        initialize_serial(*mesh);
    }

    void initialize_stk(STKAdapterPtr adapter)
    {
        const auto& mesh = adapter->mesh();
        std::vector<size_t> owned_cells;
        std::vector<size_t> ghost_cells;
        for (size_t lid = 0; lid < mesh.num_local_cells(); ++lid)
        {
            if (mesh.is_owned_cell(checked_local(lid)))
            {
                owned_cells.push_back(lid);
            }
            else
            {
                ghost_cells.push_back(lid);
            }
        }
        initialize_cells(std::move(owned_cells), std::move(ghost_cells));

        std::vector<size_t> owned_faces;
        std::vector<size_t> overlap_faces;
        for (size_t lid = 0; lid < mesh.num_faces(); ++lid)
        {
            if (mesh.is_owned_face(checked_local(lid)))
            {
                owned_faces.push_back(lid);
            }
            else
            {
                overlap_faces.push_back(lid);
            }
        }
        initialize_faces(
            std::move(owned_faces), std::move(overlap_faces));
        initialize_cell_faces();
        initialize_boundary_patches(*adapter);

        d_owned_cell_map = mesh.owned_cell_map();
        d_overlap_cell_map = mesh.overlap_cell_map();
        d_owned_face_map = mesh.owned_face_map();
        d_boundary_face_map = mesh.boundary_face_map();
        std::vector<global_ordinal_type> overlap_face_gids;
        overlap_face_gids.reserve(d_face_geometry_lids.size());
        for (const auto face_lid : d_face_geometry_lids)
        {
            overlap_face_gids.push_back(
                mesh.face_global_id(checked_local(face_lid)));
        }
        d_overlap_face_map = make_map(
            Tpetra::getDefaultComm(), overlap_face_gids);
    }

    template<class MeshType>
    void initialize_serial(const MeshType& mesh)
    {
        std::vector<size_t> cells(mesh.num_cells());
        for (size_t lid = 0; lid < cells.size(); ++lid)
        {
            cells[lid] = lid;
        }
        initialize_cells(std::move(cells), {});

        std::vector<size_t> faces(mesh.num_faces());
        for (size_t lid = 0; lid < faces.size(); ++lid)
        {
            faces[lid] = lid;
        }
        initialize_faces(std::move(faces), {});
        initialize_cell_faces();
        initialize_boundary_patches(mesh);
        create_maps(Tpetra::getDefaultComm());
    }

    void initialize_cells(std::vector<size_t> owned,
                          std::vector<size_t> ghost)
    {
        d_num_owned_cells = owned.size();
        d_cell_geometry_lids = std::move(owned);
        d_cell_geometry_lids.insert(
            d_cell_geometry_lids.end(), ghost.begin(), ghost.end());
        for (size_t local = 0; local < d_cell_geometry_lids.size(); ++local)
        {
            d_cell_local_by_geometry.emplace(
                d_cell_geometry_lids[local], checked_local(local));
        }
    }

    void initialize_faces(std::vector<size_t> owned,
                          std::vector<size_t> overlap)
    {
        d_num_owned_faces = owned.size();
        d_face_geometry_lids = std::move(owned);
        d_face_geometry_lids.insert(
            d_face_geometry_lids.end(), overlap.begin(), overlap.end());
        for (size_t local = 0; local < d_face_geometry_lids.size(); ++local)
        {
            d_face_local_by_geometry.emplace(
                d_face_geometry_lids[local], checked_local(local));
        }
    }

    void initialize_cell_faces()
    {
        d_cell_face_offsets.clear();
        d_cell_face_lids.clear();
        d_cell_face_offsets.reserve(d_cell_geometry_lids.size() + 1);
        d_cell_face_offsets.push_back(0);

        visit(
            [&](const auto& mesh)
            {
                for (const auto geometry_lid : d_cell_geometry_lids)
                {
                    const auto geometry_faces =
                        mesh.faces(mesh.cell_id(geometry_lid));
                    d_cell_face_lids.reserve(
                        d_cell_face_lids.size() + geometry_faces.size());
                    for (const auto geometry_face : geometry_faces)
                    {
                        const auto face_geometry_lid =
                            static_cast<size_t>(
                                mesh.face_local_id(geometry_face));
                        const auto iter =
                            d_face_local_by_geometry.find(face_geometry_lid);
                        if (iter != d_face_local_by_geometry.end())
                        {
                            d_cell_face_lids.push_back(iter->second);
                        }
                    }
                    d_cell_face_offsets.push_back(d_cell_face_lids.size());
                }
            });
    }

    template<class MeshType>
    void initialize_boundary_patches(const MeshType& mesh)
    {
        auto materialize_patch =
            [&](int patch_id, const auto& source_faces)
        {
            BoundaryFacePatch patch;
            patch.id = patch_id;
            for (const auto face : source_faces)
            {
                const auto geometry_lid =
                    static_cast<size_t>(mesh.face_local_id(face));
                const auto iter =
                    d_face_local_by_geometry.find(geometry_lid);
                if (iter != d_face_local_by_geometry.end())
                {
                    patch.face_lids.push_back(iter->second);
                }
            }
            if (!patch.face_lids.empty())
            {
                d_boundary_names.emplace(
                    patch_id, mesh.boundary_patch_name(patch_id));
                d_boundary_patches.emplace(
                    patch_id, std::move(patch));
            }
        };

        if constexpr (std::ranges::range<
                          decltype(mesh.boundary_face_patch(0))>)
        {
            // New view-based API: boundary_face_patch() returns a view
            for (int patch_id : mesh.boundary_patch_ids())
            {
                materialize_patch(
                    patch_id, mesh.boundary_face_patch(patch_id));
            }
        }
        else
        {
            // Legacy materialized-map API
            for (const auto& [patch_id, source_patch] :
                 mesh.boundary_patches())
            {
                materialize_patch(patch_id, source_patch.face_lids);
            }
        }
    }

    template<class CommPtr, class Id>
    Teuchos::RCP<const map_type> make_map(
        const CommPtr& comm,
        const std::vector<Id>& ids) const
    {
        std::vector<global_ordinal_type> gids;
        gids.reserve(ids.size());
        for (const auto id : ids)
        {
            gids.push_back(static_cast<global_ordinal_type>(id));
        }
        const auto invalid_size =
            Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
        return Teuchos::rcp(new map_type(
            invalid_size,
            gids.data(),
            checked_local(gids.size()),
            global_ordinal_type{},
            comm));
    }

    template<class CommPtr>
    void create_maps(const CommPtr& comm)
    {
        std::vector<size_t> owned_cells(
            d_cell_geometry_lids.begin(),
            d_cell_geometry_lids.begin()
                + static_cast<std::ptrdiff_t>(d_num_owned_cells));
        std::vector<size_t> owned_faces(
            d_face_geometry_lids.begin(),
            d_face_geometry_lids.begin()
                + static_cast<std::ptrdiff_t>(d_num_owned_faces));
        std::vector<size_t> boundary_faces;
        for (const auto& [patch_id, patch] : d_boundary_patches)
        {
            (void)patch_id;
            for (const auto face_lid : patch.face_lids)
            {
                if (is_owned_face(face_lid))
                {
                    boundary_faces.push_back(
                        d_face_geometry_lids[
                            static_cast<size_t>(face_lid)]);
                }
            }
        }

        d_owned_cell_map = make_map(comm, owned_cells);
        d_overlap_cell_map = make_map(comm, d_cell_geometry_lids);
        d_owned_face_map = make_map(comm, owned_faces);
        d_overlap_face_map = make_map(comm, d_face_geometry_lids);
        d_boundary_face_map = make_map(comm, boundary_faces);
    }

    void check_cell(local_ordinal_type cell_lid) const
    {
        CHECK_BOUNDS(cell_lid, 0, num_local_cells());
    }

    void check_face(local_ordinal_type face_lid) const
    {
        CHECK_BOUNDS(face_lid, 0, num_faces());
    }

    variant_type d_mesh;
    size_t d_num_owned_cells = 0;
    size_t d_num_owned_faces = 0;
    std::vector<size_t> d_cell_geometry_lids;
    std::vector<size_t> d_face_geometry_lids;
    std::vector<size_t> d_cell_face_offsets;
    std::vector<local_ordinal_type> d_cell_face_lids;
    std::unordered_map<size_t, local_ordinal_type>
        d_cell_local_by_geometry;
    std::unordered_map<size_t, local_ordinal_type>
        d_face_local_by_geometry;
    std::unordered_map<int, std::string> d_boundary_names;
    std::unordered_map<int, BoundaryFacePatch> d_boundary_patches;
    Teuchos::RCP<const map_type> d_owned_cell_map;
    Teuchos::RCP<const map_type> d_overlap_cell_map;
    Teuchos::RCP<const map_type> d_owned_face_map;
    Teuchos::RCP<const map_type> d_overlap_face_map;
    Teuchos::RCP<const map_type> d_boundary_face_map;
};

} // namespace SimpleFluid
