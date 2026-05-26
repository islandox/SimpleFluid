/**
 * @file Mesh.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#pragma once

#include "dataclass/TpetraTypes.hh"
#include "dataclass/vec3.hh"
#include "utils/debug_check.hh"

#include <Teuchos_RCP.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>

#include <string>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace SimpleFluid
{

/**
 * Cell-centered finite-volume mesh for a hybrid triangular-prism / hexahedral mesh.
 *
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
 */
template<TpetraTypePack Pack>
class Mesh
{
public:
    using map_type = typename Pack::map_type;
    using global_ordinal_type = typename Pack::global_ordinal_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using scalar_type = typename Pack::scalar_type;
    using comm_type = typename Pack::comm_type;

    using Entity = stk::mesh::Entity;
    using EntityId = stk::mesh::EntityId;

    using BulkData = stk::mesh::BulkData;
    using MetaData = stk::mesh::MetaData;

    using Vec3 = vec3<real_t>;

    static constexpr int invalid_boundary_id = -1;

    struct Options {
        /*
         * Optional mapping from STK sideset/side-part names to your boundary IDs.
         *
         * Example:
         *   boundary_name_to_id["wall"] = 1;
         *   boundary_name_to_id["inlet"] = 2;
         */
        std::unordered_map<std::string, int> boundary_name_to_id;

        /*
         * If true, side-rank STK parts not listed in boundary_name_to_id
         * receive automatically generated IDs.
         */
        bool auto_assign_boundary_ids = true;
    };

    struct CellInfo {
        EntityId global_id = 0;
        stk::topology topology;
        Vec3 center;
        double volume = 0.0;
        bool owned = false;
        std::vector<local_ordinal_type> faces;
    };

    struct FaceInfo {
        /*
         * Node IDs in the local side order from the first element that
         * introduced this face. Sorted copies are used internally for matching.
         */
        std::vector<EntityId> node_ids;

        /*
         * owner and neighbor are local cell indices in this StkFvmMesh object.
         *
         * owner is preferred to be a locally owned cell when exactly one side
         * is owned and the other side is ghost/aura.
         */
        local_ordinal_type owner = 0;
        std::optional<local_ordinal_type> neighbor;

        Vec3 center;

        /*
         * Unit normal pointing outward from owner.
         * For a specific cell c, use face_normal_outward(c, f).
         */
        Vec3 unit_normal_from_owner;
        Vec3 unit_normal_from_neighbor;

        double area = 0.0;

        /*
         * Physical boundary ID from Exodus sideset/STK side part.
         * invalid_boundary_id means no physical boundary part was detected.
         */
        int boundary_id = invalid_boundary_id;
        std::string boundary_name;
    };

    using ArrVec3 = std::vector<Vec3>;
    using ArrLO = std::vector<local_ordinal_type>;
    using ArrGO = std::vector<global_ordinal_type>;

    template <class T>
    using kokkos_1dview = Kokkos::View<T*, typename Pack::device_type>;

    template <class T>
    using kokkos_vec3view = Kokkos::View<T*[3], typename Pack::device_type>;

    template <class T>
    using RCP = Teuchos::RCP<T>;

    enum class CellType : uint8_t 
    {
        TRIPRISM = 3,
        HEXAHEDRON = 4
    };
    enum class FaceType : uint8_t 
    {
        TRIANGLE = 3,
        QUAD = 4
    };

    struct DeviceViews;

    Mesh();

//-------------------------------- assemble ----------------------------------//
public:
    void assemble();
    void export_vtu(const std::string& filename) const;

private:
    void create_maps();
    void create_device_views();

    void build_cell_list();
    void compute_cell_geometry();

    void build_face_table();
    void prefer_owned_face_owners();
    void compute_face_geometry();

    void assign_boundary_ids_from_stk_side_parts();

    void initialize_boundary_id_maps();
    int get_or_create_boundary_id(const std::string& name);

    static bool is_supported_volume_topology(stk::topology topo);
    static std::vector<unsigned> side_node_ordinals(stk::topology topo,
                                                    unsigned side_ordinal);

    static std::string make_face_key(std::vector<EntityId> node_ids);

    bool is_candidate_boundary_part(const stk::mesh::Part& part,
                                    stk::mesh::EntityRank side_rank) const;

    const stk::mesh::Part* choose_boundary_part(
        const stk::mesh::Bucket& bucket,
        stk::mesh::EntityRank side_rank) const;

//------------------------- one-by-one setting -------------------------------//
public:


//------------------------------- checks -------------------------------------//
private:
    void check_cell(local_ordinal_type lid) const {CHECK(lid < num_local_cells());};
    void check_face(local_ordinal_type lid) const {CHECK(lid < num_faces());};

    void check_connectivity() const;

//----------------------------- accessors ------------------------------------//
public:
    DeviceViews device_views() const noexcept { return d_device_views; }

    const MetaData& meta() const noexcept { return d_meta; }
    const BulkData& bulk() const noexcept { return d_bulk; }
    MetaData& meta() noexcept { return d_meta; }
    BulkData& bulk() noexcept { return d_bulk; }

    const ArrLO& owned_cell_ids() const noexcept { return d_owned_cell_ids; }
    const Arr<EntityId>& owned_cell_global_ids() const noexcept { return d_owned_cell_global_ids; }

    RCP<const map_type> owned_cell_map() const { return d_owned_cell_map; }
    RCP<const map_type> overlap_cell_map() const { return d_overlap_cell_map; }

//------------------------------ random access -------------------------------//
    inline const CellInfo& cell(local_ordinal_type lid) const;
    inline const FaceInfo& face(local_ordinal_type lid) const;
    inline const EntityId& cell_global_id(local_ordinal_type lid) const;

//-------------------------------- queries -----------------------------------//

    size_t spatial_dimension() const noexcept { return d_meta.spatial_dimension(); }

    size_t num_local_cells() const noexcept { return d_cells.size(); }
    size_t num_owned_cells() const noexcept { return d_owned_cell_ids.size(); }
    size_t num_faces() const noexcept { return d_faces.size(); }

    inline bool is_owned_cell(local_ordinal_type lid) const;

    inline const ArrLO& faces(local_ordinal_type cell_lid) const;

    inline Vec3 node_coord(stk::mesh::Entity node) const;
    inline Vec3 node_coord_by_id(EntityId node_id) const;
    inline Arr<Vec3> element_node_coords(stk::mesh::Entity elem) const;

    inline real_t cell_volume(local_ordinal_type lid) const;
    inline const Vec3& cell_centroid(local_ordinal_type lid) const;

    inline local_ordinal_type owner_cell(local_ordinal_type fid) const;
    inline local_ordinal_type neighbor_cell(local_ordinal_type fid) const;
    inline local_ordinal_type opposite_cell(local_ordinal_type fid, local_ordinal_type cell_lid) const;

    inline real_t             face_area(local_ordinal_type fid) const;
    inline const Vec3&        face_normal(local_ordinal_type fid) const;
    inline const Vec3&        face_centroid(local_ordinal_type fid) const;
    inline const Vec3&        face_normal_outward(local_ordinal_type fid, local_ordinal_type cell_lid) const;

    inline bool is_exterior_face(local_ordinal_type fid) const;
    inline bool is_interior_face(local_ordinal_type fid) const;
    inline bool is_boundary_face(local_ordinal_type fid) const;

    inline int boundary_id(local_ordinal_type fid) const;
    inline const std::string& boundary_name(local_ordinal_type fid) const;
    inline const ArrLO& face_patch(int patch_id) const;

    inline local_ordinal_type find_local_cell(EntityId gid) const;

//----------------------------- device views ---------------------------------//
private:
    template <class T>
    static inline kokkos_1dview<const T> 
    make_vector_view(const std::string& name, const std::vector<T>& data);

    static inline kokkos_vec3view<const real_t> 
    make_vectorV3D_view(const std::string& name, 
                        const std::vector<Vec3>& data);

//---------------------------------- Data ------------------------------------//
private:
    Options d_options;
    std::shared_ptr<MetaData> d_meta_storage;
    std::unique_ptr<BulkData> d_bulk_storage;
    stk::io::StkMeshIoBroker d_io;

    MetaData& d_meta;
    BulkData& d_bulk;

    stk::mesh::Field<double>* d_coord_field = nullptr;

    Arr<Entity> d_cell_entities;
    Arr<CellInfo> d_cells;
    Arr<FaceInfo> d_faces;

    ArrLO d_owned_cell_ids;
    Arr<EntityId> d_owned_cell_global_ids;

    std::unordered_map<EntityId, local_ordinal_type> d_cell_gid_to_lid;

    /*
     * Internal face lookup: sorted node IDs encoded as a string key.
     * This is only used at setup time, so simplicity is preferred over speed.
     */
    std::unordered_map<std::string, local_ordinal_type> face_key_to_face_;

    std::unordered_map<int, std::string> d_boundary_id_to_name;
    std::unordered_map<std::string, int> d_boundary_name_to_id;
    std::unordered_map<int, ArrLO> d_boundary_id_to_faces;
    int d_next_boundary_id = 1; // Start from 1 since 0 is reserved for invalid_boundary_id

    RCP<const map_type> d_owned_cell_map;
    RCP<const map_type> d_overlap_cell_map;

    DeviceViews d_device_views;
};

template<TpetraTypePack Pack>
struct Mesh<Pack>::DeviceViews
{
    kokkos_1dview<const global_ordinal_type>  cell_gid;
    kokkos_1dview<const int>                  cell_type;

    kokkos_1dview<const real_t>          cell_volume;
    kokkos_vec3view<const real_t>        cell_centroid;

    kokkos_1dview<const local_ordinal_type>   cell_face_offset;
    kokkos_1dview<const local_ordinal_type>   cell_face_ids;

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
#include "geometry/Mesh_inline_functions.hh"
