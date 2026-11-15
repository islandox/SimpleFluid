/** @file MeshFactory.hh
 * @brief Native mesh construction and STK/Ioss file input.
 */
#pragma once

#include "dataclass/Database.hh"
#include "dataclass/TpetraTypes.hh"
#include "geometry/BoundaryLayerMeshFactory.hh"
#include "geometry/Mesh.hh"
#include "geometry/STKMesh.hh"
#include "geometry/MeshHandle.hh"
#include "io/STKMeshInput.hh"

namespace SimpleFluid
{
/**
 * @brief Construct native geometry from configuration or a mesh input file.
 * Programmatic source geometry is generated only on rank zero. STK is used
 * transiently by the file reader; returned handles own native geometry/maps.
 */
class MeshFactory
{
public:
    enum class DomainType : uint8_t
    {
        BOX = 0,
        CYLINDER = 1,
        SPHERE = 2,
        EXTERNAL = 3,
        ANNULUS = 4
    };
    MeshFactory(SP<const Database> db);

    /** Explicit legacy materialization retained for compatibility callers. */
    template<TpetraTypePack Pack = DefaultTpetraTypes> SP<Mesh<Pack>> build();

    template<TpetraTypePack Pack = DefaultTpetraTypes> SP<MeshHandle<Pack>> build_handle();

    template<TpetraTypePack Pack = DefaultTpetraTypes>
    SP<MeshHandle<Pack>> build_handle(Teuchos::RCP<const typename Pack::comm_type> comm);

    /** Collective input on comm; omitted comm uses the default Tpetra communicator. */
    template<TpetraTypePack Pack = DefaultTpetraTypes>
    static SP<MeshHandle<Pack>> from_stk_file(const std::string& filename, const STKMeshInputOptions& options = {},
        Teuchos::RCP<const typename Pack::comm_type> comm = Teuchos::null);

    template<TpetraTypePack Pack = DefaultTpetraTypes>
    static SP<Mesh<Pack>> build_empty_mesh();

private:
    using BoundaryLayerSpec = BoundaryLayerMeshFactory::BoundaryLayerSpec;
    const BoundaryLayerSpec* boundary_layer_spec(const std::string& boundary_name) const;
    void validate_boundary_layer_names() const;
    Meshes::UnstructuredMeshData build_source(const Teuchos::RCP<const Teuchos::Comm<int>>& comm) const;

    DomainType d_domain_type = DomainType::BOX;
    int d_dimension = 3;
    Vec3D<ArrReal> d_box_cell_edges;
    Vec3D<ArrReal> d_annulus_cell_edges;
    real_t d_mesh_size = 1.0;
    real_t d_radius = 1.0;
    real_t d_cylinder_height = 1.0;
    real_t d_cylinder_circumferential_mesh_size = 1.0;
    std::string d_external_mesh_file;
    std::string d_external_decomposition_method = "auto";
    ArrString d_domain_exterior_face_types;
    Arr<BoundaryLayerSpec> d_boundary_layer_specs;
};
} // namespace SimpleFluid
