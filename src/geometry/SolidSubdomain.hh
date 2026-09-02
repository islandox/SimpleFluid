/**
 * @file SolidSubdomain.hh
 * @brief Compact finite-volume view of selected solid cells in a MeshHandle.
 */

#pragma once

#include "SimpleFluidExport.hh"
#include "geometry/MeshHandle.hh"
#include "geometry/MeshReorderingFactory.hh"

#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief A compact distributed mesh view over selected cells of a parent mesh.
 *
 * SolidSubdomain keeps the parent geometry alive but gives selected cells and
 * their faces independent, owned-first local ordinals and Tpetra maps. Faces
 * separating a selected cell from an unselected cell are exposed as a named
 * exterior boundary. Such cut faces are reoriented so their owner and normal
 * are always the selected-cell side.
 *
 * Selector-based construction first asks MeshReorderingFactory to put selected
 * owned cells and selected ghost cells at the fronts of their respective parent
 * ranges.  The subdomain then translates those two compact ranges
 * arithmetically. Ghost membership therefore remains authoritative without a
 * persistent parent-sized mask.
 *
 * @tparam Pack Tpetra scalar, ordinal, communicator, and map types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes> class SIMPLEFLUID_PUBLIC_TYPE SolidSubdomain
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using map_type = typename Pack::map_type;
    using mesh_type = MeshHandle<Pack>;
    using reordering_factory_type = MeshReorderingFactory<Pack>;
    using selected_cell_layout_type = typename reordering_factory_type::SelectedCellLayout;
    using Vec3 = typename mesh_type::Vec3;

    /** @brief Locally visible faces belonging to one boundary batch. */
    struct BoundaryFaceBatch
    {
        int id = -1;
        std::vector<local_ordinal_type> face_lids;
    };

    /** @brief Parent-side identity for one selected/unselected cut face. */
    struct InterfaceFace
    {
        local_ordinal_type face_lid = invalid_local_id();
        local_ordinal_type solid_cell_lid = invalid_local_id();
        local_ordinal_type parent_face_lid = invalid_local_id();
        local_ordinal_type outside_parent_cell_lid = invalid_local_id();
        global_ordinal_type outside_cell_geometry_global_id{};
    };

    /**
     * @brief Decide whether a parent-owned cell belongs to the solid.
     *
     * The callback receives the partition-independent geometry ID and cell
     * centroid. It must not perform communication.
     */
    using cell_selector_type = std::function<bool(global_ordinal_type, const Vec3&)>;

    static constexpr int invalid_boundary_id = -1;

    /**
     * @brief Build a compact solid-cell view of @p parent.
     * @param parent Uniquely owned parent runtime mesh; legacy meshes are
     *        supported by first wrapping them in MeshHandle. Ownership is
     *        consumed so local cells can be reordered in place.
     * @param selector Pure local predicate evaluated for parent-owned cells.
     *        Construct mesh-backed fields and caches only after this operation.
     * @param interface_name Name assigned to selected/unselected cut faces.
     * @throws std::invalid_argument for null inputs, an empty name, a boundary
     *         name collision, rank-inconsistent interface names, a globally
     *         empty selection, an overlap cell without a communicator owner,
     *         shared mesh-handle ownership when reordering is required, or
     *         insufficient parent overlap around a selected owned cell.
     * @throws std::runtime_error on ranks where another rank's selector fails.
     */
    SolidSubdomain(SP<mesh_type>&& parent, cell_selector_type selector, std::string interface_name = "solid_interface");

    /**
     * @brief Build from a parent mesh whose selected cells have already been
     *        reordered to owned and ghost prefixes.
     *
     * The layout is normally produced by
     * MeshReorderingFactory::selected_cells_first().  Cell translations are
     * represented arithmetically from its two selected ranges; no
     * parent-sized membership or reverse-index arrays are retained.
     */
    explicit SolidSubdomain(selected_cell_layout_type layout, std::string interface_name = "solid_interface");

    /** @brief Treat every parent-mesh cell as part of the solid region. */
    explicit SolidSubdomain(SP<const mesh_type> parent, std::string interface_name = "solid_interface");

    const mesh_type& parent_mesh() const noexcept { return *d_parent; }
    SP<const mesh_type> parent_mesh_ptr() const noexcept { return d_parent; }

    size_t spatial_dimension() const noexcept { return d_parent->spatial_dimension(); }

    size_t num_owned_cells() const noexcept { return d_num_owned_cells; }
    size_t num_local_cells() const noexcept { return d_num_owned_cells + d_num_ghost_cells; }
    size_t num_cells() const noexcept { return num_local_cells(); }

    size_t num_owned_faces() const noexcept { return d_num_owned_faces; }
    size_t num_faces() const noexcept { return d_faces.size(); }

    static constexpr local_ordinal_type invalid_local_id() noexcept
    {
        if constexpr (std::is_signed_v<local_ordinal_type>)
        {
            return static_cast<local_ordinal_type>(-1);
        }
        return std::numeric_limits<local_ordinal_type>::max();
    }

    /** @brief Identity conversion for the subdomain's packed cell IDs. */
    local_ordinal_type cell_id(local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        return cell_lid;
    }

    /** @brief Identity conversion for the subdomain's packed face IDs. */
    local_ordinal_type face_id(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        return face_lid;
    }

    local_ordinal_type cell_local_id(local_ordinal_type cell_id) const
    {
        check_cell(cell_id);
        return cell_id;
    }

    local_ordinal_type face_local_id(local_ordinal_type face_id) const
    {
        check_face(face_id);
        return face_id;
    }

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
        return d_parent->cell_global_id(parent_cell_lid(cell_lid));
    }

    /** @brief Stable geometry ID inherited from the parent mesh. */
    global_ordinal_type cell_geometry_global_id(local_ordinal_type cell_lid) const
    {
        return d_parent->cell_geometry_global_id(parent_cell_lid(cell_lid));
    }

    global_ordinal_type face_global_id(local_ordinal_type face_lid) const
    {
        return d_parent->face_global_id(parent_face_lid(face_lid));
    }

    local_ordinal_type parent_cell_lid(local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        const auto local = static_cast<size_t>(cell_lid);
        const auto parent = local < d_num_owned_cells ? local : d_parent->num_owned_cells() + local - d_num_owned_cells;
        return checked_local(parent);
    }

    local_ordinal_type parent_face_lid(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        return d_faces[static_cast<size_t>(face_lid)].parent_lid;
    }

    /** @return Packed subdomain LID, or invalid_local_id() if unselected. */
    local_ordinal_type subdomain_cell_lid(local_ordinal_type parent_cell_lid) const
    {
        check_parent_cell(parent_cell_lid);
        const auto parent = static_cast<size_t>(parent_cell_lid);
        if (parent < d_num_owned_cells)
        {
            return parent_cell_lid;
        }

        const auto first_parent_ghost = d_parent->num_owned_cells();
        if (parent >= first_parent_ghost && parent - first_parent_ghost < d_num_ghost_cells)
        {
            return checked_local(d_num_owned_cells + parent - first_parent_ghost);
        }
        return invalid_local_id();
    }

    /** @return Packed subdomain LID, or invalid_local_id() if unavailable. */
    local_ordinal_type subdomain_face_lid(local_ordinal_type parent_face_lid) const
    {
        check_parent_face(parent_face_lid);
        const auto find_in = [&](size_t begin, size_t end) -> local_ordinal_type
        {
            const auto first = d_faces.begin() + static_cast<std::ptrdiff_t>(begin);
            const auto last = d_faces.begin() + static_cast<std::ptrdiff_t>(end);
            const auto found = std::lower_bound(first, last, parent_face_lid,
                [](const FaceRecord& face, local_ordinal_type parent_lid) { return face.parent_lid < parent_lid; });
            if (found == last || found->parent_lid != parent_face_lid)
            {
                return invalid_local_id();
            }
            return checked_local(static_cast<size_t>(std::distance(d_faces.begin(), found)));
        };

        const auto owned = find_in(0, d_num_owned_faces);
        return owned != invalid_local_id() ? owned : find_in(d_num_owned_faces, d_faces.size());
    }

    bool contains_parent_cell(local_ordinal_type parent_cell_lid) const
    {
        return subdomain_cell_lid(parent_cell_lid) != invalid_local_id();
    }

    real_t cell_volume(local_ordinal_type cell_lid) const { return d_parent->cell_volume(parent_cell_lid(cell_lid)); }

    Vec3 cell_centroid(local_ordinal_type cell_lid) const { return d_parent->cell_centroid(parent_cell_lid(cell_lid)); }

    std::span<const local_ordinal_type> faces(local_ordinal_type cell_lid) const
    {
        check_cell(cell_lid);
        const auto local = static_cast<size_t>(cell_lid);
        const auto begin = d_cell_face_offsets[local];
        const auto end = d_cell_face_offsets[local + 1];
        return std::span<const local_ordinal_type>(d_cell_face_lids).subspan(begin, end - begin);
    }

    local_ordinal_type owner_cell(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        return d_faces[static_cast<size_t>(face_lid)].owner;
    }

    local_ordinal_type neighbor_cell(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        return d_faces[static_cast<size_t>(face_lid)].neighbor;
    }

    local_ordinal_type opposite_cell(local_ordinal_type face_lid, local_ordinal_type cell_lid) const
    {
        check_face(face_lid);
        check_cell(cell_lid);
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
        throw std::invalid_argument("Cell is not adjacent to requested solid-subdomain face.");
    }

    local_ordinal_type opposite_or_periodic_neighbor_cell(
        local_ordinal_type face_lid, local_ordinal_type cell_lid) const
    {
        return opposite_cell(face_lid, cell_lid);
    }

    real_t face_area(local_ordinal_type face_lid) const { return d_parent->face_area(parent_face_lid(face_lid)); }

    Vec3 face_centroid(local_ordinal_type face_lid) const { return d_parent->face_centroid(parent_face_lid(face_lid)); }

    Vec3 face_normal(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        const auto& face = d_faces[static_cast<size_t>(face_lid)];
        const auto normal = d_parent->face_normal(face.parent_lid);
        return face.reverse_normal ? normal * -1.0 : normal;
    }

    Vec3 face_area_vector(local_ordinal_type face_lid) const { return face_normal(face_lid) * face_area(face_lid); }

    Vec3 face_normal_outward(local_ordinal_type face_lid, local_ordinal_type cell_lid) const
    {
        const auto owner = owner_cell(face_lid);
        const auto neighbor = neighbor_cell(face_lid);
        if (cell_lid == owner)
        {
            return face_normal(face_lid);
        }
        if (neighbor != invalid_local_id() && cell_lid == neighbor)
        {
            return face_normal(face_lid) * -1.0;
        }
        check_cell(cell_lid);
        throw std::invalid_argument("Cell is not adjacent to requested solid-subdomain face.");
    }

    Vec3 face_area_vector_outward(local_ordinal_type face_lid, local_ordinal_type cell_lid) const
    {
        return face_normal_outward(face_lid, cell_lid) * face_area(face_lid);
    }

    real_t face_cell_center_distance(local_ordinal_type face_lid) const
    {
        const auto neighbor = neighbor_cell(face_lid);
        if (neighbor == invalid_local_id())
        {
            return 0.0;
        }
        return (cell_centroid(neighbor) - cell_centroid(owner_cell(face_lid))).norm();
    }

    Vec3 cell_center_vector(local_ordinal_type face_lid, local_ordinal_type cell_lid) const
    {
        const auto other = opposite_cell(face_lid, cell_lid);
        if (other == invalid_local_id())
        {
            throw std::invalid_argument("Exterior solid-subdomain face has no opposite cell.");
        }
        return cell_centroid(other) - cell_centroid(cell_lid);
    }

    real_t cell_to_face_distance(local_ordinal_type face_lid, local_ordinal_type cell_lid) const
    {
        check_face(face_lid);
        check_cell(cell_lid);
        if (cell_lid != owner_cell(face_lid) && cell_lid != neighbor_cell(face_lid))
        {
            throw std::invalid_argument("Cell is not adjacent to requested solid-subdomain face.");
        }
        return d_parent->cell_to_face_distance(parent_face_lid(face_lid), parent_cell_lid(cell_lid));
    }

    bool is_exterior_face(local_ordinal_type face_lid) const { return neighbor_cell(face_lid) == invalid_local_id(); }

    bool is_interior_face(local_ordinal_type face_lid) const { return !is_exterior_face(face_lid); }

    int boundary_id(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        return d_faces[static_cast<size_t>(face_lid)].boundary_id;
    }

    bool is_boundary_face(local_ordinal_type face_lid) const
    {
        return is_exterior_face(face_lid) && boundary_id(face_lid) != invalid_boundary_id;
    }

    const std::string& boundary_batch_name(int batch_id) const
    {
        const auto iter = d_boundary_names.find(batch_id);
        if (iter == d_boundary_names.end())
        {
            throw std::out_of_range("Unknown solid-subdomain boundary batch ID.");
        }
        return iter->second;
    }

    const BoundaryFaceBatch& boundary_face_batch(int batch_id) const { return d_boundary_batches.at(batch_id); }

    const std::unordered_map<int, BoundaryFaceBatch>& boundary_batches() const noexcept { return d_boundary_batches; }

    int interface_boundary_id() const noexcept { return d_interface_boundary_id; }

    const std::string& interface_boundary_name() const noexcept { return d_interface_boundary_name; }

    bool is_interface_face(local_ordinal_type face_lid) const
    {
        check_face(face_lid);
        return d_faces[static_cast<size_t>(face_lid)].interface;
    }

    const std::vector<InterfaceFace>& interface_faces() const noexcept { return d_interface_faces; }

    Teuchos::RCP<const map_type> owned_cell_map() const { return d_owned_cell_map; }

    Teuchos::RCP<const map_type> overlap_cell_map() const { return d_overlap_cell_map; }

    Teuchos::RCP<const map_type> owned_face_map() const { return d_owned_face_map; }

    Teuchos::RCP<const map_type> overlap_face_map() const { return d_overlap_face_map; }

    Teuchos::RCP<const map_type> boundary_face_map() const { return d_boundary_face_map; }

    template<class StoredField> void sync_periodic_boundaries(StoredField& field) const { field.sync_ghosts(); }

private:
    struct FaceRecord
    {
        local_ordinal_type parent_lid = invalid_local_id();
        local_ordinal_type owner = invalid_local_id();
        local_ordinal_type neighbor = invalid_local_id();
        local_ordinal_type outside_parent_cell = invalid_local_id();
        int boundary_id = invalid_boundary_id;
        bool reverse_normal = false;
        bool interface = false;
    };

    static SP<const mesh_type> require_parent(SP<const mesh_type> parent)
    {
        if (!parent)
        {
            throw std::invalid_argument("SolidSubdomain requires a non-null parent mesh.");
        }
        return parent;
    }

    static selected_cell_layout_type all_cells_layout(SP<const mesh_type> parent)
    {
        parent = require_parent(std::move(parent));
        const auto selected_owned = parent->num_owned_cells();
        const auto selected_ghost = parent->num_local_cells() - selected_owned;
        return {std::move(parent), selected_owned, selected_ghost};
    }

    static bool valid_local(local_ordinal_type local_id, size_t count)
    {
        if constexpr (std::is_signed_v<local_ordinal_type>)
        {
            if (local_id < 0)
            {
                return false;
            }
        }
        return static_cast<size_t>(local_id) < count;
    }

    static local_ordinal_type checked_local(size_t value)
    {
        if (value > static_cast<size_t>(std::numeric_limits<local_ordinal_type>::max()))
        {
            throw std::overflow_error("SolidSubdomain local ordinal overflow.");
        }
        return static_cast<local_ordinal_type>(value);
    }

    void check_cell(local_ordinal_type cell_lid) const
    {
        if (!valid_local(cell_lid, num_local_cells()))
        {
            throw std::out_of_range("SolidSubdomain cell local ID is out of range.");
        }
    }

    void check_face(local_ordinal_type face_lid) const
    {
        if (!valid_local(face_lid, num_faces()))
        {
            throw std::out_of_range("SolidSubdomain face local ID is out of range.");
        }
    }

    void check_parent_cell(local_ordinal_type cell_lid) const
    {
        if (!valid_local(cell_lid, d_parent->num_local_cells()))
        {
            throw std::out_of_range("Parent cell local ID is out of range.");
        }
    }

    void check_parent_face(local_ordinal_type face_lid) const
    {
        if (!valid_local(face_lid, d_parent->num_faces()))
        {
            throw std::out_of_range("Parent face local ID is out of range.");
        }
    }

    Teuchos::RCP<const map_type> make_map(std::span<const global_ordinal_type> gids) const
    {
        const auto invalid_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
        return Teuchos::rcp(new map_type(invalid_size, gids.data(), checked_local(gids.size()), global_ordinal_type{},
            d_parent->owned_cell_map()->getComm()));
    }

    void initialize_layout(size_t selected_owned_cells, size_t selected_ghost_cells);
    void validate_owned_adjacency() const;
    void initialize_interface_boundary();
    void initialize_faces();
    void initialize_cell_faces();
    void initialize_boundary_batches();
    void initialize_maps();

    SP<const mesh_type> d_parent;
    std::string d_interface_boundary_name;
    int d_interface_boundary_id = invalid_boundary_id;

    size_t d_num_owned_cells = 0;
    size_t d_num_ghost_cells = 0;
    size_t d_num_owned_faces = 0;
    std::vector<FaceRecord> d_faces;
    std::vector<size_t> d_cell_face_offsets;
    std::vector<local_ordinal_type> d_cell_face_lids;
    std::unordered_map<int, std::string> d_boundary_names;
    std::unordered_map<int, BoundaryFaceBatch> d_boundary_batches;
    std::vector<InterfaceFace> d_interface_faces;

    Teuchos::RCP<const map_type> d_owned_cell_map;
    Teuchos::RCP<const map_type> d_overlap_cell_map;
    Teuchos::RCP<const map_type> d_owned_face_map;
    Teuchos::RCP<const map_type> d_overlap_face_map;
    Teuchos::RCP<const map_type> d_boundary_face_map;
};

} // namespace SimpleFluid
