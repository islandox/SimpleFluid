/**
 * @file STKMesh.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Mesh.hh"

#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MetaData.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_topology/topology.hpp>

namespace SimpleFluid
{

/**
 * @brief Container for STK mesh objects and I/O state.
 *
 * The container stores STK metadata, bulk data, coordinate field handles,
 * and the mesh I/O broker used to read external meshes.
 */
struct STKMeshContainer
{
    using Entity = stk::mesh::Entity;
    using EntityId = stk::mesh::EntityId;

    using BulkData = stk::mesh::BulkData;
    using MetaData = stk::mesh::MetaData;

    /**
     * @brief STK mesh construction options.
     *
     * Provides boundary part mapping and auto-assignment behavior for
     * external meshes loaded through STK.
     */
    struct Options {
        /**
         * Optional mapping from STK sideset/side-part names to your boundary IDs.
         *
         * Example:
         *   boundary_name_to_id["wall"] = 1;
         *   boundary_name_to_id["inlet"] = 2;
         */
        std::unordered_map<std::string, int> boundary_name_to_id;

        /**
         * If true, side-rank STK parts not listed in boundary_name_to_id
         * receive automatically generated IDs.
         */
        bool auto_assign_boundary_ids = true;
    };
    
    Arr<Entity> cell_entities;
    
    Options options;
    SP<MetaData> meta;
    SP<BulkData> bulk;
    
    stk::io::StkMeshIoBroker io;

    stk::mesh::Field<double>* coord_field = nullptr;
};

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
class STKMesh : public Mesh<Pack>
{
public:
    using Base = Mesh<Pack>;
    using typename Base::map_type;
    using typename Base::global_ordinal_type;
    using typename Base::local_ordinal_type;
    using typename Base::scalar_type;
    using typename Base::comm_type;


    using typename Base::Vec3;
    using typename Base::ViewLO;
    using typename Base::ViewGO;
    using typename Base::CellType;
    using typename Base::FaceType;

    using Options = STKMeshContainer::Options;
    using typename Base::CellInfo;
    using typename Base::FaceInfo;
    using typename Base::DeviceViews;
    using typename Base::ArrVec3;
    using typename Base::ArrLO;
    using typename Base::ArrGO;

    using Entity = stk::mesh::Entity;
    using EntityId = stk::mesh::EntityId;

    using BulkData = stk::mesh::BulkData;
    using MetaData = stk::mesh::MetaData;

    static constexpr int invalid_boundary_id = Base::invalid_boundary_id;
    
public:
    STKMesh();

    STKMesh(const std::string& mesh_filename, const Options& options = Options());

    void assemble() override;

    void export_vtu(const std::string& filename) const override;

    SP<const MetaData> meta() const noexcept { return d_stk.meta; }
    SP<const BulkData> bulk() const noexcept { return d_stk.bulk; }
    SP<MetaData> meta() noexcept { return d_stk.meta; }
    SP<BulkData> bulk() noexcept { return d_stk.bulk; }

private:
    void build_face_table();
    void compute_face_geometry();

    void build_cell_list();
    void compute_cell_geometry();

    void create_device_views();

    int get_or_create_boundary_id(const std::string& name);
    void initialize_boundary_id_maps();
    void assign_boundary_ids_from_stk_side_parts();
    bool is_candidate_boundary_part(const stk::mesh::Part& part,
                                    stk::mesh::EntityRank side_rank) const;

    static bool is_supported_volume_topology(stk::topology topo);
    static std::vector<unsigned> side_node_ordinals(stk::topology topo,
                                                    unsigned side_ordinal);


    const stk::mesh::Part* choose_boundary_part(
        const stk::mesh::Bucket& bucket,
        stk::mesh::EntityRank side_rank) const;

    inline Vec3 node_coord(stk::mesh::Entity node) const;
    inline Vec3 node_coord_by_id(EntityId node_id) const;
    inline Arr<Vec3> element_node_coords(stk::mesh::Entity elem) const;

private:
    using Base::check_connectivity;
    using Base::create_maps;
    using Base::d_boundary_id_to_faces;
    using Base::d_boundary_id_to_name;
    using Base::d_boundary_name_to_id;
    using Base::d_cell_gid_to_lid;
    using Base::d_cell_owned_face_ids;
    using Base::d_cell_owned_node_global_ids;
    using Base::d_cells;
    using Base::d_device_views;
    using Base::d_face_key_to_face;
    using Base::d_face_owned_node_global_ids;
    using Base::d_faces;
    using Base::d_next_boundary_id;
    using Base::d_node_gid_to_lid;
    using Base::d_owned_cell_global_ids;
    using Base::d_owned_cell_ids;
    using Base::d_spatial_dim;
    using Base::make_face_key;
    using Base::make_vectorV3D_view;
    using Base::make_vector_view;
    using Base::prefer_owned_face_owners;

    STKMeshContainer d_stk;
};

} // namespace SimpleFluid

//----------------------------- inline functions ---------------------------------//
#include "STKMesh.ipp"
