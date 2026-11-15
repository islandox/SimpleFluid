/** @file MeshFactory.cc
 * @brief Native factory publication and explicit legacy compatibility.
 */
#include "SimpleFluidExport.hh"
SIMPLEFLUID_LEGACY_API_BEGIN

#include "geometry/MeshFactory.hh"
#include "geometry/details/LegacyMeshSnapshot.hh"
#include "geometry/details/NativeMeshGenerators.hh"
#include "parallel/CollectiveConstruction.hh"
#include "parallel/MeshPartitioner.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <exception>
#include <stdexcept>

namespace SimpleFluid
{
MeshFactory::MeshFactory(SP<const Database> db)
{
    if (!db)
        throw std::invalid_argument("MeshFactory requires a non-null database.");
    d_dimension = db->get<int>("dimension");
    d_mesh_size = db->get<real_t>("mesh_size");

    d_domain_type = static_cast<DomainType>(db->get<int>("domain_type"));
    if (d_domain_type == DomainType::BOX)
    {
        d_box_cell_edges = {db->get<ArrReal>("X"), db->get<ArrReal>("Y"), db->get<ArrReal>("Z")};
    }
    else if (d_domain_type == DomainType::CYLINDER)
    {
        d_radius = db->get<real_t>("radius");
        d_cylinder_height = db->get<real_t>("height");
        d_cylinder_circumferential_mesh_size = db->contains("cylinder_circumferential_mesh_size")
                                                   ? db->get<real_t>("cylinder_circumferential_mesh_size")
                                                   : d_mesh_size;
    }
    else if (d_domain_type == DomainType::ANNULUS)
    {
        d_annulus_cell_edges = {db->get<ArrReal>("R"), db->get<ArrReal>("Theta"), db->get<ArrReal>("Z")};
    }
    else if (d_domain_type == DomainType::SPHERE)
    {
        d_radius = db->get<real_t>("radius");
    }
    else if (d_domain_type == DomainType::EXTERNAL)
    {
        d_external_mesh_file = db->get<std::string>("external_mesh_file");
        if (db->contains("external_mesh_decomposition_method"))
            d_external_decomposition_method = db->get<std::string>("external_mesh_decomposition_method");
    }

    d_domain_exterior_face_types = db->get<ArrString>("domain_exterior_face_types");

    d_boundary_layer_specs = BoundaryLayerMeshFactory::read_specs(db);
    if (!d_boundary_layer_specs.empty())
    {
        validate_boundary_layer_names();
    }
}

/**
 * @brief Look up the boundary-layer specification for a given boundary name.
 *
 * @param boundary_name Name of the mesh boundary.
 * @return Pointer to the matching BoundaryLayerSpec, or nullptr if not found.
 */
const MeshFactory::BoundaryLayerSpec* MeshFactory::boundary_layer_spec(const std::string& boundary_name) const
{
    const auto iter = std::find_if(d_boundary_layer_specs.begin(), d_boundary_layer_specs.end(),
        [&](const BoundaryLayerSpec& spec) { return spec.boundary_name == boundary_name; });
    return iter == d_boundary_layer_specs.end() ? nullptr : &*iter;
}

/**
 * @brief Validate that all boundary-layer names correspond to domain exterior face types.
 *
 * @throws std::runtime_error If a boundary-layer name is not found in the domain exterior face types.
 */
void MeshFactory::validate_boundary_layer_names() const
{
    for (const auto& spec : d_boundary_layer_specs)
    {
        const auto matches =
            std::find(d_domain_exterior_face_types.begin(), d_domain_exterior_face_types.end(), spec.boundary_name);
        if (matches == d_domain_exterior_face_types.end())
        {
            throw std::runtime_error("Boundary-layer boundary name is not part of this domain: " + spec.boundary_name);
        }
    }
}

Meshes::UnstructuredMeshData MeshFactory::build_source(const Teuchos::RCP<const Teuchos::Comm<int>>& comm) const
{
    if (comm.is_null())
        throw std::invalid_argument("MeshFactory requires a communicator.");
    const int domain = static_cast<int>(d_domain_type);
    int minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &domain, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &domain, &maximum);
    if (minimum != maximum)
        throw std::invalid_argument("MeshFactory domain differs between ranks.");
    if (d_domain_type == DomainType::EXTERNAL)
    {
        STKMeshInputOptions options;
        std::exception_ptr error;
        try
        {
            options.decomposition_method = d_external_decomposition_method;
            int next = 1;
            for (const auto& name : d_domain_exterior_face_types)
                if (!options.boundary_name_to_id.contains(name))
                    options.boundary_name_to_id.emplace(name, next++);
        }
        catch (...)
        {
            error = std::current_exception();
        }
        const int local = error ? 1 : 0;
        int failed = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &failed);
        if (failed)
        {
            if (error)
                std::rethrow_exception(error);
            throw std::runtime_error("MeshFactory input option preparation failed on another rank.");
        }
        return detail::read_stk_mesh_input(d_external_mesh_file, options, comm);
    }
    Meshes::UnstructuredMeshData source;
    std::exception_ptr error;
    try
    {
        // Metadata is available even on ranks which hold no source cells.
        int next = 1;
        std::unordered_map<std::string, int> ids;
        for (const auto& name : d_domain_exterior_face_types)
            if (!ids.contains(name))
            {
                ids.emplace(name, next);
                source.boundary_names.emplace(next++, name);
            }
        if (comm->getRank() == 0)
        {
            detail::NativeMeshGeneratorOptions options;
            options.domain_type = static_cast<uint8_t>(d_domain_type);
            options.dimension = d_dimension;
            options.box_cell_edges = d_box_cell_edges;
            options.annulus_cell_edges = d_annulus_cell_edges;
            options.mesh_size = d_mesh_size;
            options.radius = d_radius;
            options.cylinder_height = d_cylinder_height;
            options.cylinder_circumferential_mesh_size = d_cylinder_circumferential_mesh_size;
            options.domain_exterior_face_types = d_domain_exterior_face_types;
            options.boundary_layer_specs = d_boundary_layer_specs;
            source = detail::generate_native_mesh(options);
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    const int local = error ? 1 : 0;
    int failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &failed);
    if (failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error("MeshFactory source construction failed on another rank.");
    }
    return source;
}

template<TpetraTypePack Pack> SP<Mesh<Pack>> MeshFactory::build()
{
    return detail::make_legacy_mesh_snapshot<Pack>(build_source(Tpetra::getDefaultComm()));
}

template<TpetraTypePack Pack> SP<MeshHandle<Pack>> MeshFactory::build_handle()
{
    return build_handle<Pack>(Tpetra::getDefaultComm());
}

template<TpetraTypePack Pack>
SP<MeshHandle<Pack>> MeshFactory::build_handle(Teuchos::RCP<const typename Pack::comm_type> comm)
{
    auto partition = MeshPartitioner<Pack>::partition_native(build_source(comm), comm, 1, true);
    return detail::collective_make_shared<MeshHandle<Pack>>(
        *comm, std::move(partition.geometry), partition.indexer, comm);
}

template<TpetraTypePack Pack>
SP<MeshHandle<Pack>> MeshFactory::from_stk_file(
    const std::string& filename, const STKMeshInputOptions& options, Teuchos::RCP<const typename Pack::comm_type> comm)
{
    if (comm.is_null())
        comm = Tpetra::getDefaultComm();
    auto source = detail::read_stk_mesh_input(filename, options, comm);
    auto partition = MeshPartitioner<Pack>::partition_native(std::move(source), comm, 1, true);
    return detail::collective_make_shared<MeshHandle<Pack>>(
        *comm, std::move(partition.geometry), partition.indexer, comm);
}

template<TpetraTypePack Pack> SP<Mesh<Pack>> MeshFactory::build_empty_mesh()
{
    return detail::make_legacy_mesh_snapshot<Pack>({});
}

template SP<Mesh<DefaultTpetraTypes>> MeshFactory::build<DefaultTpetraTypes>();
template SP<MeshHandle<DefaultTpetraTypes>> MeshFactory::build_handle<DefaultTpetraTypes>();
template SP<MeshHandle<DefaultTpetraTypes>> MeshFactory::build_handle<DefaultTpetraTypes>(
    Teuchos::RCP<const DefaultTpetraTypes::comm_type>);
template SP<MeshHandle<DefaultTpetraTypes>> MeshFactory::from_stk_file<DefaultTpetraTypes>(
    const std::string&, const STKMeshInputOptions&, Teuchos::RCP<const DefaultTpetraTypes::comm_type>);
template SP<Mesh<DefaultTpetraTypes>> MeshFactory::build_empty_mesh<DefaultTpetraTypes>();
} // namespace SimpleFluid

SIMPLEFLUID_LEGACY_API_END
