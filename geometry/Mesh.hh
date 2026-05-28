/**
 * @file Mesh.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Abstract finite-volume mesh base class with CRS connectivity and Kokkos device views.
 * @version 0.1
 * @date 2026-05-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include "dataclass/TpetraTypes.hh"
#include "dataclass/vec3.hh"
#include "dataclass/RandomAccessView.hh"
#include "utils/debug_check.hh"

#include "MeshUtils.hh"

#include <Teuchos_RCP.hpp>
#include <Kokkos_Core.hpp>

#include <string>
#include <memory>
#include <optional>
#include <limits>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace SimpleFluid
{

template <typename ID>
constexpr ID invalid_id() noexcept
{
    if constexpr (std::is_signed_v<ID>)
    {
        return static_cast<ID>(-1);
    }
    else
    {
        return std::numeric_limits<ID>::max();
    }
}

/**
 * @brief finite-volume mesh for a hybrid triangular-prism / hexahedral mesh.
 *
 * @details
 * Designed for Trilinos/Tpetra assembly:
 *
 *   - owned cells correspond to Tpetra rows;
 *   - ghost cells provide off-rank stencil columns;
 *   - connectivity is stored in CRS-like arrays;
 *   - geometry is stored in Kokkos::View for CPU/GPU kernels.
 *
 * Typical FVM stencil:
 *
 *   A(P,P) and A(P,N) are assembled by looping over faces of owned cell P.
 * 
 * @tparam Pack Tpetra type pack used for map and vector types.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class Mesh
{
public:
    using map_type = typename Pack::map_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using comm_type = typename Pack::comm_type;

    using ViewLO = RandomAccessView<local_ordinal_type>;
    using ViewGO = RandomAccessView<global_ordinal_type>;
    using ViewReal = RandomAccessView<real_t>;

    static constexpr int invalid_boundary_id = invalid_id<int>();

    using Vec3 = MeshUtils::Vec3;
    using CellType = MeshUtils::CellType;
    using FaceType = MeshUtils::FaceType;
    using ArrVec3 = std::vector<Vec3>;
    using ArrLO = std::vector<local_ordinal_type>;
    using ArrGO = std::vector<global_ordinal_type>;

    using GO2LOMap = std::unordered_map<global_ordinal_type, local_ordinal_type>;

    template <class T>
    using kokkos_1dview = Kokkos::View<T*, typename Pack::device_type>;

    template <class T>
    using kokkos_vec3view = Kokkos::View<T*[3], typename Pack::device_type>;

    template <class T>
    using RCP = Teuchos::RCP<T>;

    /**
     * @brief Information about a cell, accessible by local cell ID.
     * 
     */
    struct CellInfo {
        bool owned = false;                  // Whether this cell is owned by the local process (true) or is a ghost cell (false).
        CellType type = CellType::INVALID;   // Cell type (e.g., TETRAHEDRON, HEXAHEDRON, TRIPRISM).
        Vec3 center;                         // Cell centroid coordinates.
        double volume = 0.0;                 // Cell volume.
        ViewLO faces;                        // Local face IDs of the faces that bound this cell, in the local face order from the first element that introduced this cell.
        ViewReal face_distances;             // Distances from this cell centroid to each face centroid, parallel to faces.
        ViewGO node_gids;                     // Global node IDs of the nodes that define this cell, in the local node order from the first element that introduced this cell.
    };

    /**
     * @brief Information about a face, accessible by local face ID.
     * 
     */
    struct FaceInfo {
        FaceType type = FaceType::INVALID;
        /*
         * Physical boundary ID from Exodus sideset/STK side part.
         * invalid_boundary_id means no physical boundary part was detected.
         */
        int boundary_id = invalid_boundary_id;

        /*
         * owner and neighbor are local cell indices in this StkFvmMesh object.
         *
         * owner is preferred to be a locally owned cell when exactly one side
         * is owned and the other side is ghost/aura.
         */
        local_ordinal_type owner = invalid_id<local_ordinal_type>();
        local_ordinal_type neighbor = invalid_id<local_ordinal_type>();

        /*
         * Node IDs in the local side order from the first element that
         * introduced this face. Sorted copies are used internally for matching.
         */
        ViewGO node_gids;

        MeshUtils::Vec3 center;

        /*
         * Unit normal pointing outward from owner.
         * For a specific cell c, use face_normal_outward(c, f).
         */
        MeshUtils::Vec3 unit_normal_from_owner;
        MeshUtils::Vec3 unit_normal_from_neighbor;

        double area = 0.0;
        double owner_to_face_distance = 0.0;
        double neighbor_to_face_distance = 0.0;
        double cell_center_distance = 0.0;
    };


    struct DeviceViews;

    Mesh();

//-------------------------------- assemble ----------------------------------//
public:
    virtual void assemble() = 0;
    virtual void export_vtu(const std::string& filename) const = 0;

protected:
    void create_maps();
    void create_cell_face_distances();
    void create_device_views();

    void prefer_owned_face_owners();

    inline static std::string make_face_key(ViewGO node_ids);
    inline static std::string make_face_key(ArrGO node_ids);

//------------------------- one-by-one setting -------------------------------//
public:


//------------------------------- checks -------------------------------------//
protected:
    void check_cell(local_ordinal_type lid) const;
    void check_face(local_ordinal_type lid) const;

    void check_connectivity() const;

//----------------------------- accessors ------------------------------------//
public:
    DeviceViews device_views() const noexcept { return d_device_views; }

    const ArrLO& owned_cell_ids() const noexcept { return d_owned_cell_ids; }
    const ArrGO& owned_cell_global_ids() const noexcept { return d_owned_cell_global_ids; }

    RCP<const map_type> owned_cell_map() const { return d_owned_cell_map; }
    RCP<const map_type> overlap_cell_map() const { return d_overlap_cell_map; }

//------------------------------ random access -------------------------------//
    inline const CellInfo& cell(local_ordinal_type lid) const;
    inline const FaceInfo& face(local_ordinal_type lid) const;
    inline const global_ordinal_type& cell_global_id(local_ordinal_type lid) const;
    inline const Vec3& node_coord(global_ordinal_type node_gid) const;

//-------------------------------- queries -----------------------------------//

    size_t spatial_dimension() const noexcept { return d_spatial_dim; }

    size_t num_local_cells() const noexcept { return d_cells.size(); }
    size_t num_owned_cells() const noexcept { return d_owned_cell_ids.size(); }
    size_t num_faces() const noexcept { return d_faces.size(); }

    inline bool is_owned_cell(local_ordinal_type lid) const;

    inline const ViewLO& faces(local_ordinal_type cell_lid) const;
    inline const ViewReal& face_distances(local_ordinal_type cell_lid) const;

    inline real_t cell_volume(local_ordinal_type lid) const;
    inline const Vec3& cell_centroid(local_ordinal_type lid) const;

    inline local_ordinal_type owner_cell(local_ordinal_type fid) const;
    inline local_ordinal_type neighbor_cell(local_ordinal_type fid) const;
    inline local_ordinal_type opposite_cell(local_ordinal_type fid, local_ordinal_type cell_lid) const;

    inline real_t             face_area(local_ordinal_type fid) const;
    inline real_t             face_cell_center_distance(local_ordinal_type fid) const;
    inline real_t             cell_to_face_distance(local_ordinal_type fid, local_ordinal_type cell_lid) const;
    inline const Vec3&        face_normal(local_ordinal_type fid) const;
    inline const Vec3&        face_centroid(local_ordinal_type fid) const;
    inline const Vec3&        face_normal_outward(local_ordinal_type fid, local_ordinal_type cell_lid) const;

    inline bool is_exterior_face(local_ordinal_type fid) const;
    inline bool is_interior_face(local_ordinal_type fid) const;
    inline bool is_boundary_face(local_ordinal_type fid) const;

    inline int boundary_id(local_ordinal_type fid) const;
    inline const std::string& boundary_name(local_ordinal_type fid) const;
    inline const ArrLO& face_patch(int patch_id) const;

    inline local_ordinal_type global_to_local_cell(global_ordinal_type gid) const;

//----------------------------- device views ---------------------------------//
protected:
    template <class T>
    static inline kokkos_1dview<const T> 
    make_vector_view(const std::string& name, const std::vector<T>& data);

    static inline kokkos_vec3view<const real_t> 
    make_vectorV3D_view(const std::string& name, 
                        const std::vector<Vec3>& data);

//---------------------------------- Data ------------------------------------//
protected:
    int d_spatial_dim = 0;

    //!<@brief Cell information stored on host for easy random access.
    //!<@note ghost cells are always stored after owned cells, so owned cells are contiguous at the beginning of d_cells.
    Arr<CellInfo> d_cells;
    Arr<FaceInfo> d_faces;   //!< Face information stored on host for easy random access.

    ArrLO d_owned_cell_ids;
    ArrGO d_owned_cell_global_ids;
    ArrGO d_ghost_cell_global_ids;

    ArrLO d_cell_owned_face_ids;
    ArrReal d_cell_face_distances;
    ArrGO d_cell_owned_node_global_ids;
    ArrGO d_face_owned_node_global_ids;
    ArrVec3 d_node_coords;

    GO2LOMap d_cell_gid_to_lid;
    GO2LOMap d_node_gid_to_lid;

    /*
     * @brief Internal face lookup: sorted node IDs encoded as a string key.
     * This is only used at setup time, so simplicity is preferred over speed.
     */
    std::unordered_map<std::string, local_ordinal_type> d_face_key_to_face;

    std::unordered_map<int, std::string> d_boundary_id_to_name;
    std::unordered_map<std::string, int> d_boundary_name_to_id;
    std::unordered_map<int, ArrLO> d_boundary_id_to_faces;
    int d_next_boundary_id = 1; // Start from 1 since 0 is reserved for invalid_boundary_id

    RCP<const map_type> d_owned_cell_map;
    RCP<const map_type> d_overlap_cell_map;

    DeviceViews d_device_views;
};

/**
 * @brief Kokkos device-side views for mesh geometry and connectivity data.
 */
template<TpetraTypePack Pack>
struct Mesh<Pack>::DeviceViews
{
    kokkos_1dview<const global_ordinal_type>  cell_gid;
    kokkos_1dview<const int>                  cell_type;

    kokkos_1dview<const real_t>          cell_volume;
    kokkos_vec3view<const real_t>        cell_centroid;

    kokkos_1dview<const local_ordinal_type>   cell_face_offset;
    kokkos_1dview<const local_ordinal_type>   cell_face_ids;
    kokkos_1dview<const real_t>               cell_face_distance;

    kokkos_1dview<const local_ordinal_type>   face_owner;
    kokkos_1dview<const local_ordinal_type>   face_neighbor;
    kokkos_1dview<const int>                  face_type;
    kokkos_1dview<const int>                  face_patch;

    kokkos_1dview<const real_t>          face_area;
    kokkos_vec3view<const real_t>        face_area_vector;
    kokkos_vec3view<const real_t>        face_centroid;

    kokkos_vec3view<const real_t>        node_coord;

    kokkos_1dview<const local_ordinal_type>   cell_node_offset;
    kokkos_1dview<const local_ordinal_type>   cell_node_ids;

    kokkos_1dview<const local_ordinal_type>   face_node_offset;
    kokkos_1dview<const local_ordinal_type>   face_node_ids;
};

}

// Include inline function definitions
#include "geometry/Mesh.ipp"
