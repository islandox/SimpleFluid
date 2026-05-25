/**
 * @file Mesh.hh
 * @author your name (you@domain.com)
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

    using Vec3 = vec3<real_t>;
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

    Mesh() = default;

//-------------------------------- assemble ----------------------------------//
public:
    void assemble();

private:
    void create_maps();
    void create_device_views();

//------------------------- one-by-one setting -------------------------------//
public:
    inline local_ordinal_type add_cell(global_ordinal_type gid, CellType type, 
                                real_t volume, const Vec3& centroid);
    inline local_ordinal_type add_face(local_ordinal_type owner, local_ordinal_type neighbor, 
                                FaceType type, int patch_id,
                                real_t area, const Vec3& area_vector, const Vec3& centroid);
    inline local_ordinal_type add_node(const Vec3& coord);

    inline void set_cell_faces(local_ordinal_type cell_lid, const ArrLO& face_ids);
    inline void set_cell_nodes(local_ordinal_type cell_lid, const ArrLO& node_ids);
    inline void set_face_nodes(local_ordinal_type face_lid, const ArrLO& node_ids);

//------------------------------- checks -------------------------------------//
private:
    void check_cell(local_ordinal_type lid) const {CHECK(lid < d_num_local_cells);};
    void check_face(local_ordinal_type lid) const {CHECK(lid < d_num_faces);};
    void check_node(local_ordinal_type lid) const {CHECK(lid < d_num_nodes);};

    void check_connectivity() const;

//----------------------------- accessors ------------------------------------//
public:
    DeviceViews device_views() const { return d_device_views; }

    RCP<const map_type> owned_cell_map() const { return d_owned_cell_map; }
    RCP<const map_type> overlap_cell_map() const { return d_overlap_cell_map; }

    local_ordinal_type num_owned_cells() const { return d_num_owned_cells; }
    local_ordinal_type num_local_cells() const { return d_num_local_cells; }
    local_ordinal_type num_faces() const { return d_num_faces; }
    local_ordinal_type num_nodes() const { return d_num_nodes; }

//-------------------------------- queries -----------------------------------//
    inline global_ordinal_type cell_global_id(local_ordinal_type lid) const;
    inline local_ordinal_type cell_local_id(global_ordinal_type gid) const;

    inline CellType        cell_type(local_ordinal_type lid) const;
    inline real_t          cell_volume(local_ordinal_type lid) const;
    inline const Vec3&     cell_centroid(local_ordinal_type lid) const;

    inline local_ordinal_type face_owner(local_ordinal_type fid) const;
    inline local_ordinal_type face_neighbor(local_ordinal_type fid) const;
    inline FaceType           face_type(local_ordinal_type fid) const;
    inline int                face_patch(local_ordinal_type fid) const;
    inline real_t             face_area(local_ordinal_type fid) const;
    inline const Vec3&        face_area_vector(local_ordinal_type fid) const;
    inline const Vec3&        face_centroid(local_ordinal_type fid) const;

    inline local_ordinal_type cell_face_begin(local_ordinal_type lid) const;
    inline local_ordinal_type cell_face_end(local_ordinal_type lid) const;

    inline local_ordinal_type cell_face(local_ordinal_type cell_lid, local_ordinal_type i) const;

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
    global_ordinal_type d_num_global_cells = 0;
    local_ordinal_type d_num_owned_cells = 0;
    local_ordinal_type d_num_local_cells = 0;

    local_ordinal_type d_num_faces = 0;
    local_ordinal_type d_num_nodes = 0;

    RCP<const comm_type> d_comm;

    RCP<const map_type> d_owned_cell_map;
    RCP<const map_type> d_overlap_cell_map;

    /*
     * Host-side data used during construction and host assembly.
     *
     * Owned cells must be first:
     *
     *   local cell lid: [0, num_owned_cells)
     *
     * Ghost cells come after:
     *
     *   local ghost lid: [num_owned_cells, num_local_cells)
     */
    ArrGO   d_cell_gid;

    ArrInt  d_cell_type;
    ArrReal d_cell_volume;
    ArrVec3 d_cell_centroid;

    ArrLO   d_cell_face_offset = {0};
    ArrLO   d_cell_face_ids;

    ArrLO   d_face_owner;
    ArrLO   d_face_neighbor;

    ArrInt  d_face_type;
    ArrInt  d_face_patch;

    ArrReal d_face_area;
    ArrVec3 d_face_area_vector;
    ArrVec3 d_face_centroid;

    ArrVec3 d_node_coord;

    ArrLO   d_cell_node_offset = {0};
    ArrLO   d_cell_node_ids;

    ArrLO   d_face_node_offset = {0};
    ArrLO   d_face_node_ids;

    std::unordered_map<global_ordinal_type, local_ordinal_type> d_cell_gid_to_lid;

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