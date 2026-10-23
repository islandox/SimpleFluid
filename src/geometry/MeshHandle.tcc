/**
 * @file MeshHandle.tcc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Template implementations for MeshHandle construction and VTU output.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "MeshHandle.hh"
#include "geometry/details/SemiStructuredPartition.hh"
#include "geometry/MeshReorderingFactory.hh"

#include <array>
#include <cmath>
#include <Teuchos_CommHelpers.hpp>

namespace SimpleFluid
{

/**
 * @brief Construct and distribute a Cartesian mesh handle.
 *
 * @param mesh Cartesian mesh to wrap.
 * @param options Partition override and ghost-layer configuration.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(MultiRegionPtr mesh) : MeshHandle(std::move(mesh), DistributionOptions{}) {}

template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(MutableMultiRegionPtr mesh) : MeshHandle(std::move(mesh),DistributionOptions{}) {}

template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(MutableMultiRegionPtr mesh, DistributionOptions options)
    : MeshHandle(MultiRegionPtr(mesh),options)
{
    d_mutable_mesh.emplace(std::move(mesh));
}

template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(MultiRegionPtr mesh, DistributionOptions options) : d_mesh(std::move(mesh))
{
    initialize_composite(std::get<MultiRegionPtr>(d_mesh), options);
}

/** @brief Partition compact global descriptors using arithmetic cell ownership and local Tpetra maps. */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_composite(MultiRegionPtr mesh, DistributionOptions options)
{
    const auto comm = mesh ? mesh->communicator() : Tpetra::getDefaultComm();
    int invalid = !mesh || mesh->partition_resident() || options.partition.has_value() || options.partitions.has_value() || options.ghost_layers == 0;
    int any_invalid = 0;
    std::exception_ptr geometry_error;
    try { if (mesh) mesh->validate_static(); }
    catch (...) { invalid=1; geometry_error=std::current_exception(); }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &invalid, &any_invalid);
    if (any_invalid)
    {
        if (geometry_error) std::rethrow_exception(geometry_error);
        throw std::invalid_argument("Composite handles require live unchanged geometry, positive halos and communicator-defined ownership on every rank.");
    }
    const auto execution = mesh->acquire_execution_view();
    if (comm->getSize() == 1) { initialize_serial(*mesh); return; }
    const size_t rank = static_cast<size_t>(comm->getRank()), ranks = static_cast<size_t>(comm->getSize());
    const size_t cells = mesh->num_cells();
    if (cells > static_cast<size_t>(std::numeric_limits<global_ordinal_type>::max())
        || mesh->num_faces() > static_cast<size_t>(std::numeric_limits<global_ordinal_type>::max())
        || mesh->num_nodes() > static_cast<size_t>(std::numeric_limits<global_ordinal_type>::max()))
        throw std::overflow_error("Composite global entity IDs exceed the Tpetra ordinal type.");
    const auto begin = (cells / ranks) * rank + std::min(cells % ranks, rank);
    const auto count = cells / ranks + (rank < cells % ranks ? 1 : 0);
    const auto end = begin + count;
    std::vector<size_t> owned(count); std::iota(owned.begin(), owned.end(), begin);
    std::vector<size_t> ghosts, frontier = owned;
    std::unordered_set<size_t> seen(owned.begin(), owned.end());
    for (size_t layer = 0; layer < options.ghost_layers && !frontier.empty(); ++layer)
    {
        std::vector<size_t> next;
        for (const auto c : frontier) for (const auto f : mesh->cell_faces(c))
        {
            const auto other = mesh->opposite_cell(f, c);
            if (other != MultiRegion::invalid_cell_id() && seen.insert(other).second)
            { ghosts.push_back(other); next.push_back(other); }
        }
        frontier = std::move(next);
    }
    std::sort(ghosts.begin(), ghosts.end());
    initialize_cells(std::move(owned), std::move(ghosts));
    std::unordered_set<size_t> faces, nodes;
    for (const auto c : d_indexer.cell_global_ids())
    {
        for (const auto f : mesh->cell_faces(c)) faces.insert(f);
        for (const auto n : mesh->cell_nodes(c)) nodes.insert(n);
    }
    std::vector<size_t> owned_faces, overlap_faces;
    for (const auto f : faces)
    {
        const auto owner = mesh->owner_cell(f);
        (owner >= begin && owner < end ? owned_faces : overlap_faces).push_back(f);
    }
    std::sort(owned_faces.begin(), owned_faces.end()); std::sort(overlap_faces.begin(), overlap_faces.end());
    initialize_faces(std::move(owned_faces), std::move(overlap_faces));
    std::vector<size_t> local_nodes(nodes.begin(), nodes.end()); std::sort(local_nodes.begin(), local_nodes.end());
    d_indexer.set_nodes(checked_global_ids(std::move(local_nodes)));
    initialize_boundary_batches(*mesh);
    create_maps(comm);
}

template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(const CompositePartition<Pack>& partition)
    : d_mesh(MultiRegionPtr(partition.geometry()))
{
    d_mutable_mesh.emplace(partition.geometry());
    const auto& plan = partition.plan();
    const auto& comm = plan.communicator();
    std::exception_ptr error;
    try
    {
        if (!partition.geometry() || comm.is_null())
            throw std::invalid_argument("Composite partition requires live geometry and communicator.");
        partition.geometry()->validate_static();
        const auto copy = [](auto values) { return std::vector<size_t>(values.begin(), values.end()); };
        initialize_cells(copy(plan.owned_cells()), copy(plan.ghost_cells()));
        initialize_faces(copy(plan.owned_faces()), copy(plan.overlap_faces()));
        d_indexer.set_nodes(checked_global_ids(copy(plan.node_ids())));
        initialize_boundary_batches(*partition.geometry());
    }
    catch (...) { error = std::current_exception(); }
    if (comm.is_null()) std::rethrow_exception(error);
    int failed = error ? 1 : 0, any_failed = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
    if (any_failed)
    {
        if (error) std::rethrow_exception(error);
        throw std::invalid_argument("Composite partition handle failed on another rank.");
    }
    create_maps(comm);
}

template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    CartesianPtr mesh,
    DistributionOptions options)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_orthogonal(std::get<CartesianPtr>(d_mesh), options);
}

/**
 * @brief Construct and distribute a mutable Cartesian mesh handle.
 *
 * @param mesh Mutable Cartesian mesh to retain.
 * @param options Partition override and ghost-layer configuration.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    MutableCartesianPtr mesh,
    DistributionOptions options)
    : MeshHandle(CartesianPtr(mesh), options)
{
    d_mutable_mesh.emplace(std::move(mesh));
}

/**
 * @brief Construct and distribute a cylindrical mesh handle.
 *
 * @param mesh Cylindrical mesh to wrap.
 * @param options Partition override and ghost-layer configuration.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    CylindricalPtr mesh,
    DistributionOptions options)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_orthogonal(std::get<CylindricalPtr>(d_mesh), options);
}

/**
 * @brief Construct and distribute a mutable cylindrical mesh handle.
 *
 * @param mesh Mutable cylindrical mesh to retain.
 * @param options Partition override and ghost-layer configuration.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    MutableCylindricalPtr mesh,
    DistributionOptions options)
    : MeshHandle(CylindricalPtr(mesh), options)
{
    d_mutable_mesh.emplace(std::move(mesh));
}

/**
 * @brief Construct a compact semi-structured mesh handle with default distribution.
 *
 * @param mesh Semi-structured mesh to wrap.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(SemiStructuredPtr mesh) : MeshHandle(std::move(mesh), DistributionOptions{})
{
}

/** @brief Preserve the one-argument mutable constructor and geometry identity. */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(MutableSemiStructuredPtr mesh) : MeshHandle(std::move(mesh), DistributionOptions{})
{
}

/** @brief Observe compact XY/Z geometry and construct communicator-owned maps. */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(SemiStructuredPtr mesh, DistributionOptions options) : d_mesh(std::move(mesh))
{
    initialize_semi_structured(std::get<SemiStructuredPtr>(d_mesh), options);
}

/** @brief Retain the original mutable geometry after collective initialization. */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(MutableSemiStructuredPtr mesh, DistributionOptions options)
    : MeshHandle(SemiStructuredPtr(mesh), options)
{
    d_mutable_mesh.emplace(std::move(mesh));
}

/**
 * @brief Construct a serial unstructured mesh handle.
 *
 * @param mesh Unstructured mesh to wrap.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(UnstructuredPtr mesh)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_unstructured(std::get<UnstructuredPtr>(d_mesh));
}

/** @brief Construct a mutable serial unstructured mesh handle. */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(MutableUnstructuredPtr mesh)
    : MeshHandle(UnstructuredPtr(mesh))
{
    d_mutable_mesh.emplace(std::move(mesh));
}

/**
 * @brief Construct a handle from an already partitioned unstructured mesh.
 *
 * @param mesh Rank-local unstructured mesh geometry.
 * @param indexer Mapping from compact local IDs to original global IDs.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    UnstructuredPtr mesh,
    const unstructured_indexer_type& indexer)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_unstructured(
        std::get<UnstructuredPtr>(d_mesh), indexer,
        Tpetra::getDefaultComm());
}

/** @brief Construct a mutable handle from an indexed unstructured mesh. */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(
    MutableUnstructuredPtr mesh,
    const unstructured_indexer_type& indexer)
    : MeshHandle(UnstructuredPtr(mesh), indexer)
{
    d_mutable_mesh.emplace(std::move(mesh));
}

/**
 * @brief Construct a handle from an existing STK mesh adapter.
 *
 * @param mesh STK adapter to wrap.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(STKAdapterPtr mesh)
    : d_mesh(require_mesh(std::move(mesh)))
{
    initialize_stk(std::get<STKAdapterPtr>(d_mesh));
}

/** @brief Construct a mutable handle around a legacy STK adapter. */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(MutableSTKAdapterPtr mesh)
    : MeshHandle(STKAdapterPtr(mesh))
{
    if (!mesh->has_mutable_mesh())
    {
        throw std::invalid_argument(
            "MeshHandle requires mutable legacy geometry when constructed "
            "from a mutable STK adapter pointer.");
    }
    d_mutable_mesh.emplace(std::move(mesh));
}

/**
 * @brief Adapt a legacy distributed mesh and construct its handle.
 *
 * @param mesh Legacy mesh to adapt.
 */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(SP<const SimpleFluid::Mesh<Pack>> mesh)
    : MeshHandle(STKAdapterPtr(
          std::make_shared<STKAdapter>(std::move(mesh))))
{
}

/** @brief Adapt a mutable legacy distributed mesh and construct its handle. */
template<TpetraTypePack Pack>
MeshHandle<Pack>::MeshHandle(SP<SimpleFluid::Mesh<Pack>> mesh)
    : MeshHandle(std::make_shared<STKAdapter>(std::move(mesh)))
{
}

/**
 * @brief Initialize ownership, overlap, connectivity, and maps for an
 * orthogonal mesh.
 *
 * The mesh is partitioned along its largest coordinate when more than one
 * partition is requested.
 *
 * @tparam MeshType Cartesian or cylindrical orthogonal mesh type.
 * @param mesh Orthogonal mesh to initialize from.
 * @param options Partition override and ghost-layer configuration.
 * @throws std::invalid_argument if partition options are inconsistent or the
 * requested partitioning cannot be represented by the mesh.
 */
template<TpetraTypePack Pack>
template<class MeshType>
void MeshHandle<Pack>::initialize_orthogonal(
    SP<const MeshType> mesh,
    DistributionOptions options)
{
    const auto comm = Tpetra::getDefaultComm();
    const int allow_empty = options.allow_empty_partitions ? 1 : 0;
    int empty_options_count = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &allow_empty, &empty_options_count);
    if (empty_options_count != 0 && empty_options_count != comm->getSize())
    {
        throw std::invalid_argument(
            "Orthogonal allow_empty_partitions must agree on every rank.");
    }
    const auto rank = options.partition.value_or(
        static_cast<size_t>(comm->getRank()));
    const auto ranks = options.partitions.value_or(
        static_cast<size_t>(comm->getSize()));
    if (options.partition.has_value() != options.partitions.has_value())
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

    const auto& dimensions = mesh->indexer().num_cells_per_dim;
    size_t coordinate = 0;
    for (size_t candidate = 1; candidate < 3; ++candidate)
    {
        if (dimensions[candidate] > dimensions[coordinate])
        {
            coordinate = candidate;
        }
    }
    if (ranks > dimensions[coordinate] && !options.allow_empty_partitions)
    {
        throw std::invalid_argument(
            "Orthogonal mesh has fewer cells than MPI ranks along "
            "its largest coordinate.");
    }

    // Keep the partitioner's nonempty-slab contract, while allowing callers to
    // retain the full communicator for transfers involving a very coarse mesh.
    const auto active_partitions =
        std::min(ranks, static_cast<size_t>(dimensions[coordinate]));
    Meshes::OrthoMeshPartitioner partitioner(
        mesh->topology(),
        static_cast<typename Meshes::OrthoMeshPartitioner::Dimension>(
            coordinate),
        active_partitions,
        static_cast<Meshes::OrthoMeshPartitioner::Ordinal>(
            options.ghost_layers));

    std::vector<size_t> owned_cells;
    std::vector<size_t> ghost_cells;
    if (rank < active_partitions)
    {
        for (const auto id : partitioner.owned_cells(rank))
        {
            owned_cells.push_back(mesh->cell_local_id(id));
        }
        for (const auto id : partitioner.ghost_cells(rank))
        {
            ghost_cells.push_back(mesh->cell_local_id(id));
        }
    }
    initialize_cells(std::move(owned_cells), std::move(ghost_cells));

    std::unordered_set<size_t> seen_faces;
    std::vector<size_t> owned_faces;
    std::vector<size_t> overlap_faces;
    for (const auto cell_geometry_lid : d_indexer.cell_global_ids())
    {
        const typename MeshType::cell_id_t typed_cell =
            mesh->indexer().cell_id(cell_geometry_lid);
        const auto typed_faces = mesh->cell_faces(typed_cell);
        for (const auto face : typed_faces)
        {
            const auto face_geometry_lid =
                mesh->indexer().face_ordinal(face);
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
    initialize_faces(std::move(owned_faces), std::move(overlap_faces));

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
    std::unordered_set<size_t> seen_nodes;
    std::vector<size_t> local_nodes;
    for (const auto cell_geometry_lid : d_indexer.cell_global_ids())
    {
        const auto cell = mesh->cell_id(
            static_cast<size_t>(cell_geometry_lid));
        for (const auto& corner : corners)
        {
            const typename MeshType::NodeID node{
                static_cast<unsigned>(
                    (cell.i + corner[0])
                    % mesh->indexer().num_nodes_per_dim[0]),
                static_cast<unsigned>(
                    (cell.j + corner[1])
                    % mesh->indexer().num_nodes_per_dim[1]),
                static_cast<unsigned>(
                    (cell.k + corner[2])
                    % mesh->indexer().num_nodes_per_dim[2])};
            const auto global_node = mesh->node_local_id(node);
            if (seen_nodes.insert(global_node).second)
            {
                local_nodes.push_back(global_node);
            }
        }
    }
    d_indexer.set_nodes(checked_global_ids(std::move(local_nodes)));
    initialize_boundary_batches(*mesh);
    create_maps(comm);
}

/**
 * @brief Build local ownership/halos from compact XY/Z topology without expanded incidence.
 * @param mesh Shared immutable geometry, identical on every rank.
 * @param options Communicator-defined ownership, positive halo depth and empty-owner policy.
 * Geometry/configuration errors are fenced before local layout allocation and
 * before Tpetra map collectives, including when a rank owns no entities.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_semi_structured(SemiStructuredPtr mesh, DistributionOptions options)
{
    const auto comm = Tpetra::getDefaultComm();
    const size_t rank = static_cast<size_t>(comm->getRank()), ranks = static_cast<size_t>(comm->getSize());
    const auto collective_error = [&](const std::exception_ptr& error, const char* message)
    {
        const int failed = error ? 1 : 0;
        int any_failed = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
        if (any_failed)
        {
            if (error)
                std::rethrow_exception(error);
            throw std::invalid_argument(message);
        }
    };
    std::exception_ptr error;
    try
    {
        if (!mesh || options.ghost_layers == 0 || options.partition.has_value() != options.partitions.has_value() ||
            (options.partition && (*options.partition != rank || *options.partitions != ranks)))
            throw std::invalid_argument(
                "SemiStructuredXY_Z requires live geometry, positive halos and communicator-matching partitions.");
        if (mesh->num_cells() > static_cast<size_t>(std::numeric_limits<global_ordinal_type>::max()) ||
            mesh->num_faces() > static_cast<size_t>(std::numeric_limits<global_ordinal_type>::max()) ||
            mesh->num_nodes() > static_cast<size_t>(std::numeric_limits<global_ordinal_type>::max()))
            throw std::overflow_error("SemiStructuredXY_Z global IDs exceed the Tpetra ordinal type.");
        if ((options.xy_partitions && (*options.xy_partitions == 0 || ranks % *options.xy_partitions != 0)) ||
            (options.z_partitions && (*options.z_partitions == 0 || ranks % *options.z_partitions != 0)) ||
            (options.xy_partitions && options.z_partitions && *options.xy_partitions != ranks / *options.z_partitions))
            throw std::invalid_argument(
                "SemiStructuredXY_Z XY/Z factors must be positive and multiply to communicator size.");
        if (mesh->num_cells() < ranks && !options.allow_empty_partitions)
            throw std::invalid_argument(
                "SemiStructuredXY_Z has fewer cells than ranks; empty partitions are disabled.");
    }
    catch (...)
    {
        error = std::current_exception();
    }
    collective_error(error, "SemiStructuredXY_Z configuration failed on another rank.");
    const auto execution = mesh->acquire_geometry_read();

    // Hash only compact descriptors, never expanded 3D incidence. Partition
    // presence is uniform, but the validated rank value is intentionally not.
    unsigned long long fingerprint = 14695981039346656037ull;
    const auto bytes = [&](const void* source, size_t count)
    {
        const auto* data = static_cast<const unsigned char*>(source);
        for (size_t i = 0; i < count; ++i)
        {
            fingerprint ^= data[i];
            fingerprint *= 1099511628211ull;
        }
    };
    const auto ordinal = [&](size_t value)
    {
        const auto encoded = static_cast<unsigned long long>(value);
        bytes(&encoded, sizeof(encoded));
    };
    const auto text = [&](const std::string& value)
    {
        ordinal(value.size());
        bytes(value.data(), value.size());
    };
    error = {};
    try
    {
        ordinal(options.ghost_layers);
        ordinal(options.allow_empty_partitions);
        ordinal(options.partition.has_value());
        ordinal(options.xy_partitions.has_value());
        if (options.xy_partitions)
            ordinal(*options.xy_partitions);
        ordinal(options.z_partitions.has_value());
        if (options.z_partitions)
            ordinal(*options.z_partitions);
        ordinal(ranks);
        ordinal(mesh->geometry_epoch());
        const auto& indexer = mesh->indexer();
        ordinal(indexer.num_cells_per_layer);
        ordinal(indexer.num_side_faces_per_layer);
        ordinal(indexer.num_nodes_per_layer);
        ordinal(indexer.num_layers);
        ordinal(indexer.num_node_layers);
        ordinal(indexer.axial_periodic);
        ordinal(mesh->xy_nodes().size());
        for (const auto& node : mesh->xy_nodes())
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const auto value = node.component(axis);
                if (!std::isfinite(value))
                    throw std::invalid_argument("SemiStructuredXY_Z coordinates are nonfinite.");
                bytes(&value, sizeof(value));
            }
        ordinal(mesh->xy_cell_nodes().size());
        for (const auto& row : mesh->xy_cell_nodes())
        {
            ordinal(row.size());
            for (const auto node : row)
                ordinal(node);
        }
        ordinal(mesh->z_edges().size());
        for (const auto value : mesh->z_edges())
        {
            if (!std::isfinite(value))
                throw std::invalid_argument("SemiStructuredXY_Z axial edges are nonfinite.");
            bytes(&value, sizeof(value));
        }
        for (const auto& face : mesh->topology().side_faces())
        {
            ordinal(face.nodes[0]);
            ordinal(face.nodes[1]);
            ordinal(face.owner);
            ordinal(face.neighbor);
            bytes(&face.boundary_id, sizeof(face.boundary_id));
        }
        auto batches = mesh->boundary_batch_ids();
        std::sort(batches.begin(), batches.end());
        ordinal(batches.size());
        for (const auto id : batches)
        {
            bytes(&id, sizeof(id));
            text(mesh->boundary_batch_name(id));
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    collective_error(error, "SemiStructuredXY_Z descriptor validation failed on another rank.");
    unsigned long long minimum = 0, maximum = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &fingerprint, &minimum);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &fingerprint, &maximum);
    if (minimum != maximum)
        throw std::invalid_argument("SemiStructuredXY_Z geometry, topology or distribution differs across ranks.");
    const size_t xy_cells = mesh->indexer().num_cells_per_layer, layers = mesh->indexer().num_layers;
    size_t xy_parts = 1, z_parts = ranks;
    if (options.xy_partitions || options.z_partitions)
    {
        xy_parts = options.xy_partitions.value_or(ranks / options.z_partitions.value_or(ranks));
        z_parts = ranks / xy_parts;
    }
    else
    {
        // Score divisors without constructing any extruded graph. Prefer using
        // all possible owners, then genuine XY/Z coupling when both are useful.
        size_t best_active = 0, best_load = std::numeric_limits<size_t>::max();
        bool best_hybrid = false;
        double best_interface = std::numeric_limits<double>::infinity();
        for (size_t px = 1; px <= ranks; ++px)
        {
            if (ranks % px != 0)
                continue;
            const size_t pz = ranks / px;
            if (!options.allow_empty_partitions && (px > xy_cells || pz > layers))
                continue;
            const size_t active_xy = std::min(px, xy_cells), active_z = std::min(pz, layers);
            const size_t active = active_xy * active_z;
            const bool hybrid = active == ranks && px > 1 && pz > 1;
            const size_t load =
                (xy_cells / active_xy + (xy_cells % active_xy != 0)) * (layers / active_z + (layers % active_z != 0));
            const double interface = static_cast<double>(layers) * std::sqrt(static_cast<double>(xy_cells)) *
                                         (std::sqrt(static_cast<double>(active_xy)) - 1.0) +
                                     static_cast<double>(xy_cells) * static_cast<double>(active_z - 1);
            if (active > best_active || (active == best_active && hybrid > best_hybrid) ||
                (active == best_active && hybrid == best_hybrid && load < best_load) ||
                (active == best_active && hybrid == best_hybrid && load == best_load && interface < best_interface))
            {
                best_active = active;
                best_hybrid = hybrid;
                best_load = load;
                best_interface = interface;
                xy_parts = px;
                z_parts = pz;
            }
        }
        if (best_active == 0)
            throw std::invalid_argument("SemiStructuredXY_Z has no nonempty XY/Z factorization for this communicator.");
    }
    if (!options.allow_empty_partitions && (xy_parts > xy_cells || z_parts > layers))
        throw std::invalid_argument("SemiStructuredXY_Z XY/Z factorization creates disabled empty owners.");
    const std::array<unsigned long long, 2> local_factors{xy_parts, z_parts};
    std::array<unsigned long long, 2> minimum_factors{}, maximum_factors{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 2, local_factors.data(), minimum_factors.data());
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 2, local_factors.data(), maximum_factors.data());
    if (minimum_factors != maximum_factors)
        throw std::invalid_argument("SemiStructuredXY_Z selected XY/Z factors differ across ranks.");
    d_semi_structured_decomposition = {xy_parts, z_parts};
    if (ranks == 1)
    {
        initialize_serial(*mesh);
        return;
    }
    const auto xy_owners =
        detail::partition_semi_structured_xy(mesh->topology(), xy_parts, options.allow_empty_partitions, comm);
    std::vector<size_t> xy_order, xy_position;
    error = {};
    try
    {
        xy_order = MeshReorderingFactory<Pack>::partitioned_xy_order(mesh->topology(), xy_owners);
        if (xy_order.size() != xy_cells)
            throw std::logic_error("SemiStructuredXY_Z graph reordering has the wrong size.");
        xy_position.assign(xy_cells, xy_cells);
        for (size_t position = 0; position < xy_cells; ++position)
        {
            const auto original = xy_order[position];
            if (original >= xy_cells || xy_position[original] != xy_cells)
                throw std::logic_error("SemiStructuredXY_Z graph reordering is not a permutation.");
            xy_position[original] = position;
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    collective_error(error, "SemiStructuredXY_Z graph reordering failed on another rank.");
    error = {};
    try
    {
        auto source = std::shared_ptr<const Meshes::SemiStructMeshTopo>(mesh, &mesh->topology());
        d_semi_structured_topology = std::make_shared<Meshes::DecomposedSemiStructuredTopology>(
            std::move(source), xy_owners, xy_order, rank, xy_parts, z_parts, options.ghost_layers);
        checked_local(d_semi_structured_topology->num_local_cells());
        checked_local(d_semi_structured_topology->num_local_faces());
        checked_local(d_semi_structured_topology->num_local_nodes());
        for (const auto& band : d_semi_structured_topology->cell_bands())
        {
            const auto cells = d_semi_structured_topology->view_source_ids(band.view);
            d_cells_reordered = d_cells_reordered || !std::is_sorted(cells.begin(), cells.end());
        }
        initialize_boundary_batches(*mesh);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    collective_error(error, "SemiStructuredXY_Z local compact topology failed on another rank.");
    initialize_contiguous_semi_structured_maps(comm);
}

/** Assign algebraic IDs only after the shared XY order and compact topology are fixed.
 * Canonical owned maps exist transiently for integer owner/local-row queries;
 * no canonical 3D map or expanded owner table is retained by the handle.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_contiguous_semi_structured_maps(
    const Teuchos::RCP<const typename Pack::comm_type>& comm)
{
    const auto collective_error = [&](const std::exception_ptr& error)
    {
        const int failed = error ? 1 : 0;
        int any_failed = 0;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &failed, &any_failed);
        if (any_failed)
        {
            if (error)
                std::rethrow_exception(error);
            throw std::invalid_argument("Contiguous XY/Z algebraic map setup failed on another rank.");
        }
    };
    const auto invalid_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
    const auto build = [&](size_t owned, size_t local, auto geometry_id)
    {
        std::vector<global_ordinal_type> owned_geometry, ghost_geometry, overlap_algebraic, rank_counts, rank_offsets;
        std::vector<int> owner_ranks;
        std::vector<local_ordinal_type> owner_rows;
        std::exception_ptr error;
        try
        {
            checked_local(owned);
            checked_local(local);
            owned_geometry.resize(owned);
            ghost_geometry.resize(local - owned);
            overlap_algebraic.resize(local);
            owner_ranks.resize(local - owned);
            owner_rows.resize(local - owned);
            rank_counts.resize(static_cast<size_t>(comm->getSize()));
            rank_offsets.resize(rank_counts.size());
            for (size_t row = 0; row < owned; ++row)
                owned_geometry[row] = geometry_id(row);
            for (size_t row = owned; row < local; ++row)
                ghost_geometry[row - owned] = geometry_id(row);
        }
        catch (...)
        {
            error = std::current_exception();
        }
        collective_error(error);
        const global_ordinal_type owned_count = static_cast<global_ordinal_type>(owned);
        Teuchos::gatherAll(*comm, 1, &owned_count, comm->getSize(), rank_counts.data());
        global_ordinal_type total = 0;
        for (size_t rank = 0; rank < rank_counts.size(); ++rank)
        {
            if (rank_counts[rank] < 0 || rank_counts[rank] > std::numeric_limits<global_ordinal_type>::max() - total)
                throw std::overflow_error("XY/Z algebraic global IDs exceed the ordinal range.");
            rank_offsets[rank] = total;
            total += rank_counts[rank];
        }
        const auto canonical_owned = Teuchos::rcp(
            new map_type(invalid_size, owned_geometry.data(), checked_local(owned), global_ordinal_type{}, comm));
        Teuchos::RCP<const map_type> algebraic_owned =
            Teuchos::rcp(new map_type(invalid_size, owned, global_ordinal_type{}, comm));
        const auto status = canonical_owned->getRemoteIndexList(
            Teuchos::ArrayView<const global_ordinal_type>(ghost_geometry.data(), checked_local(ghost_geometry.size())),
            Teuchos::ArrayView<int>(owner_ranks.data(), checked_local(owner_ranks.size())),
            Teuchos::ArrayView<local_ordinal_type>(owner_rows.data(), checked_local(owner_rows.size())));
        error = {};
        try
        {
            if (status == Tpetra::IDNotPresent)
                throw std::invalid_argument("A canonical XY/Z halo entity has no owner.");
            for (size_t row = 0; row < owned; ++row)
                overlap_algebraic[row] = algebraic_owned->getGlobalElement(checked_local(row));
            for (size_t row = 0; row < ghost_geometry.size(); ++row)
            {
                const auto rank = owner_ranks[row];
                const auto owner_row = owner_rows[row];
                if (rank < 0 || rank >= comm->getSize() || owner_row < 0 ||
                    static_cast<global_ordinal_type>(owner_row) >= rank_counts[static_cast<size_t>(rank)])
                    throw std::invalid_argument("Invalid canonical XY/Z owner/local-row lookup.");
                overlap_algebraic[owned + row] =
                    rank_offsets[static_cast<size_t>(rank)] + static_cast<global_ordinal_type>(owner_row);
            }
        }
        catch (...)
        {
            error = std::current_exception();
        }
        collective_error(error);
        Teuchos::RCP<const map_type> algebraic_overlap = Teuchos::rcp(
            new map_type(invalid_size, overlap_algebraic.data(), checked_local(local), global_ordinal_type{}, comm));
        return std::pair{std::move(algebraic_owned), std::move(algebraic_overlap)};
    };
    const auto cells =
        build(num_owned_cells(), num_local_cells(), [&](size_t row) { return geometry_cell_lid(checked_local(row)); });
    const auto faces =
        build(num_owned_faces(), num_faces(), [&](size_t row) { return geometry_face_lid(checked_local(row)); });
    d_owned_cell_map = cells.first;
    d_overlap_cell_map = cells.second;
    d_owned_face_map = faces.first;
    d_overlap_face_map = faces.second;
    d_map_indexing = true;
    std::vector<global_ordinal_type> boundary;
    std::exception_ptr error;
    try
    {
        for (const auto& [id, batch] : d_boundary_batches)
        {
            static_cast<void>(id);
            for (const auto face : batch.face_lids)
                if (is_owned_face(face))
                    boundary.push_back(face_global_id(face));
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    collective_error(error);
    d_boundary_face_map = make_map(comm, boundary);
    d_indexer = indexer_type{};
}

/**
 * @brief Initialize a serial unstructured mesh handle.
 *
 * @param mesh Unstructured mesh to initialize from.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_unstructured(
    UnstructuredPtr mesh)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 1)
    {
        throw std::invalid_argument(
            "Distributed UnstructuredMesh must be partitioned before "
            "constructing MeshHandle.");
    }
    initialize_serial(*mesh);
}

/**
 * @brief Initialize from rank-local geometry and its global indexer.
 *
 * @param mesh Previously partitioned unstructured mesh.
 * @param indexer Mapping from mesh-local IDs to original global IDs.
 * @param comm Communicator that owns the partition.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_unstructured(
    UnstructuredPtr mesh,
    const unstructured_indexer_type& indexer,
    Teuchos::RCP<const typename Pack::comm_type> comm)
{
    if (mesh->num_cells() != indexer.num_local_cells()
        || mesh->num_owned_cells() != indexer.num_owned_cells()
        || mesh->num_faces() != indexer.num_local_faces()
        || mesh->num_owned_faces() != indexer.num_owned_faces()
        || mesh->num_nodes() != indexer.num_local_nodes())
    {
        throw std::invalid_argument(
            "Partitioned UnstructuredMesh does not match its indexer.");
    }

    auto global_ids = [&](const auto& ids)
    {
        std::vector<size_t> flattened;
        flattened.reserve(ids.size());
        for (const auto id : ids)
        {
            using id_type = std::remove_cvref_t<decltype(id)>;
            if constexpr (
                std::numeric_limits<id_type>::digits
                > std::numeric_limits<size_t>::digits)
            {
                if (id > static_cast<id_type>(
                        std::numeric_limits<size_t>::max()))
                {
                    throw std::overflow_error(
                        "MeshHandle unstructured global ID overflow.");
                }
            }
            flattened.push_back(static_cast<size_t>(id));
        }
        return checked_global_ids(std::move(flattened));
    };

    initialize_indexer(indexer_type(
        global_ids(indexer.owned_cell_global_ids()),
        global_ids(indexer.ghost_cell_global_ids()),
        global_ids(indexer.owned_face_global_ids()),
        global_ids(indexer.overlap_face_global_ids()),
        global_ids(indexer.node_global_ids())));
    initialize_boundary_batches(*mesh);
    create_maps(comm ? std::move(comm) : Tpetra::getDefaultComm());
}

/**
 * @brief Import ownership, overlap, connectivity, and maps from an STK
 * adapter.
 *
 * @param adapter Adapter providing the legacy mesh interface.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_stk(STKAdapterPtr adapter)
{
    const auto& mesh = adapter->mesh();
    bool identity_order = true;
    bool saw_overlap_face = false;
    for (size_t lid = 0; lid < mesh.num_faces(); ++lid)
    {
        if (mesh.is_owned_face(checked_local(lid)))
        {
            if (saw_overlap_face)
            {
                identity_order = false;
                break;
            }
        }
        else
        {
            saw_overlap_face = true;
        }
    }
    if (!identity_order)
    {
        d_legacy_face_geometry_lids.reserve(mesh.num_faces());
        for (size_t lid = 0; lid < mesh.num_faces(); ++lid)
        {
            if (mesh.is_owned_face(checked_local(lid)))
            {
                d_legacy_face_geometry_lids.push_back(
                    checked_local(lid));
            }
        }
        for (size_t lid = 0; lid < mesh.num_faces(); ++lid)
        {
            if (!mesh.is_owned_face(checked_local(lid)))
            {
                d_legacy_face_geometry_lids.push_back(
                    checked_local(lid));
            }
        }
        d_legacy_face_local_lids.resize(
            d_legacy_face_geometry_lids.size());
        for (size_t local = 0;
             local < d_legacy_face_geometry_lids.size();
             ++local)
        {
            d_legacy_face_local_lids[
                static_cast<size_t>(
                    d_legacy_face_geometry_lids[local])] =
                checked_local(local);
        }
        // Interleaved ownership is uncommon. Preserve the established
        // owned-first handle ordering with a compact permutation and
        // materialize cell connectivity only for this fallback.
        }
    initialize_boundary_batches(*adapter);

    d_owned_cell_map = mesh.owned_cell_map();
    d_overlap_cell_map = mesh.overlap_cell_map();
    d_owned_face_map = mesh.owned_face_map();
    d_boundary_face_map = mesh.boundary_face_map();
    std::vector<global_ordinal_type> overlap_face_gids;
    overlap_face_gids.reserve(num_faces());
    for (size_t face_lid = 0; face_lid < num_faces(); ++face_lid)
    {
        const auto geometry_lid =
            geometry_face_lid(checked_local(face_lid));
        overlap_face_gids.push_back(
            mesh.face_global_id(
                checked_local(static_cast<size_t>(geometry_lid))));
    }
    d_overlap_face_map = make_map(
        Tpetra::getDefaultComm(), overlap_face_gids);
}

/**
 * @brief Initialize all cells and faces as locally owned on one rank.
 *
 * @tparam MeshType Concrete mesh type.
 * @param mesh Mesh whose complete geometry is locally available.
 */
template<TpetraTypePack Pack>
template<class MeshType>
void MeshHandle<Pack>::initialize_serial(const MeshType& mesh)
{
    checked_local(mesh.num_cells());
    checked_local(mesh.num_faces());
    checked_local(mesh.num_nodes());
    d_serial_identity = true;
    d_serial_counts = {mesh.num_cells(), mesh.num_faces(), mesh.num_nodes()};
    initialize_boundary_batches(mesh);
    if constexpr (std::same_as<MeshType, MultiRegion>) create_maps(mesh.communicator());
    else create_maps(Tpetra::getDefaultComm());
}

/**
 * @brief Store owned and ghost cell global IDs in the local/global indexer.
 *
 * Owned cells are placed before ghost cells and reverse lookup is populated.
 *
 * @param owned Geometry-local IDs of owned cells.
 * @param ghost Geometry-local IDs of ghost cells.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_cells(
    std::vector<size_t> owned,
    std::vector<size_t> ghost)
{
    d_indexer.set_cells(
        checked_global_ids(std::move(owned)),
        checked_global_ids(std::move(ghost)));
}

/**
 * @brief Store owned and overlap face global IDs in the local/global indexer.
 *
 * Owned faces are placed before overlap faces and reverse lookup is populated.
 *
 * @param owned Geometry-local IDs of owned faces.
 * @param overlap Geometry-local IDs of non-owned locally visible faces.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_faces(
    std::vector<size_t> owned,
    std::vector<size_t> overlap)
{
    d_indexer.set_faces(
        checked_global_ids(std::move(owned)),
        checked_global_ids(std::move(overlap)));
}

/**
 * @brief Replace the handle's local/global indexer with a prepared layout.
 * @tparam Pack Tpetra type pack.
 * @param indexer Complete owned-first indexer to store.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_indexer(indexer_type indexer)
{
    d_indexer = std::move(indexer);
}

/**
 * @brief Lazily materialize legacy identity/permutation lookup tables.
 *
 * Production geometry and field paths use the wrapped mesh arrays directly;
 * this preserves the historical indexer() API only for callers that request
 * it explicitly.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::materialize_legacy_indexer() const
{
    if (d_legacy_indexer_materialized)
    {
        return;
    }
    const auto legacy = legacy_mesh();
    if (!legacy)
    {
        return;
    }

    std::vector<size_t> owned_cells;
    std::vector<size_t> ghost_cells;
    owned_cells.reserve(legacy->num_owned_cells());
    ghost_cells.reserve(
        legacy->num_local_cells() - legacy->num_owned_cells());
    for (size_t local = 0; local < legacy->num_local_cells(); ++local)
    {
        (local < legacy->num_owned_cells()
             ? owned_cells
             : ghost_cells)
            .push_back(static_cast<size_t>(geometry_cell_lid(
                checked_local(local))));
    }

    std::vector<size_t> owned_faces;
    std::vector<size_t> overlap_faces;
    const auto owned_face_count = num_owned_faces();
    owned_faces.reserve(owned_face_count);
    overlap_faces.reserve(num_faces() - owned_face_count);
    for (size_t local = 0; local < num_faces(); ++local)
    {
        const auto geometry = static_cast<size_t>(
            geometry_face_lid(checked_local(local)));
        (local < owned_face_count
             ? owned_faces
             : overlap_faces)
            .push_back(geometry);
    }

    d_indexer = indexer_type(
        checked_global_ids(std::move(owned_cells)),
        checked_global_ids(std::move(ghost_cells)),
        checked_global_ids(std::move(owned_faces)),
        checked_global_ids(std::move(overlap_faces)),
        std::vector<global_ordinal_type>{});
    d_legacy_indexer_materialized = true;
}

/**
 * @brief Materialize cell-to-face connectivity in MeshHandle local IDs.
 *
 * Faces not present in the local overlap are omitted.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::initialize_cell_faces()
{
    const auto execution = acquire_execution_view();
    d_cell_face_offsets.clear();
    d_cell_face_lids.clear();
    d_cell_face_offsets.reserve(num_local_cells() + 1);
    d_cell_face_offsets.push_back(0);

    visit(
        [&](const auto& mesh)
        {
            for (size_t local_lid = 0;
                 local_lid < num_local_cells();
                 ++local_lid)
            {
                const auto geometry_lid = geometry_cell_lid(
                    checked_local(local_lid));
                const auto& geometry_faces =
                    mesh.faces(mesh.cell_id(
                        static_cast<size_t>(geometry_lid)));
                // Retain vector's amortized growth. Reserving only the next
                // cell's entries here repeatedly copies the complete prefix;
                // logical coarse/fine faces also rule out assuming six faces
                // per cell for the allocation.
                for (const auto geometry_face : geometry_faces)
                {
                    const auto face_geometry_lid =
                        static_cast<size_t>(
                            mesh.face_local_id(geometry_face));
                    const auto local_face =
                        geometry_to_local_face(face_geometry_lid);
                    if (local_face != invalid_local_id())
                    {
                        d_cell_face_lids.push_back(local_face);
                    }
                }
                d_cell_face_offsets.push_back(d_cell_face_lids.size());
            }
        });
}

/**
 * @brief Materialize locally visible boundary batches.
 *
 * Supports both view-based and legacy materialized boundary-batch APIs.
 *
 * @tparam MeshType Concrete mesh or adapter type.
 * @param mesh Mesh providing boundary batch metadata and face IDs.
 */
template<TpetraTypePack Pack>
template<class MeshType>
void MeshHandle<Pack>::initialize_boundary_batches(
    const MeshType& mesh)
{
    if constexpr (std::same_as<MeshType, MultiRegion> || std::same_as<MeshType, SemiStructured>)
    {
        if constexpr (std::same_as<MeshType, SemiStructured>)
        {
            // Boundary configuration is global even when this rank has no
            // incident boundary faces (or owns no cells at all).
            for (const auto id : mesh.boundary_batch_ids())
            {
                d_boundary_names.emplace(id, mesh.boundary_batch_name(id));
                d_boundary_batches[id].id = id;
            }
        }
        for (size_t f = 0; f < num_faces(); ++f)
        {
            const auto local = checked_local(f);
            const auto geometry = static_cast<size_t>(geometry_face_lid(local));
            const int id = mesh.boundary_id(mesh.face_id(geometry));
            if (id == invalid_boundary_id) continue;
            d_boundary_names.emplace(id, mesh.boundary_batch_name(id));
            auto& batch = d_boundary_batches[id]; batch.id = id; batch.face_lids.push_back(local);
        }
        return;
    }

    auto materialize_batch =
        [&](int batch_id, auto&& source_faces)
    {
        BoundaryFaceBatch batch;
        batch.id = batch_id;
        for (const auto face : source_faces)
        {
            const auto geometry_lid =
                static_cast<size_t>(mesh.face_local_id(face));
            const auto local_face = geometry_to_local_face(geometry_lid);
            if (local_face != invalid_local_id())
            {
                batch.face_lids.push_back(local_face);
            }
        }
        if (!batch.face_lids.empty())
        {
            d_boundary_names.emplace(
                batch_id, mesh.boundary_batch_name(batch_id));
            d_boundary_batches.emplace(batch_id, std::move(batch));
        }
    };

    if constexpr (std::ranges::range<
                      decltype(mesh.boundary_face_batch(0))>)
    {
        // New view-based API: boundary_face_batch() returns a view
        for (int batch_id : mesh.boundary_batch_ids())
        {
            materialize_batch(
                batch_id, mesh.boundary_face_batch(batch_id));
        }
    }
    else
    {
        // Legacy materialized-map API
        for (const auto& [batch_id, source_batch] :
             mesh.boundary_batches())
        {
            materialize_batch(batch_id, source_batch.face_lids);
        }
    }
}

/**
 * @brief Export owned mesh cells to VTU.
 *
 * STK meshes delegate to their native exporter; other mesh families are
 * converted to VTU connectivity by this handle.
 *
 * @param filename Requested output filename.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::export_vtu(const std::string& filename) const
{
    visit(
        [&](const auto& mesh)
        {
            using mesh_type = std::decay_t<decltype(mesh)>;
            if constexpr (std::is_same_v<mesh_type, STKAdapter>)
            {
                mesh.export_vtu(filename);
            }
            else if constexpr (std::is_same_v<mesh_type, SemiStructured>)
            {
                write_vtu(filename, semi_structured_vtu_topology(mesh));
            }
            else if constexpr (std::is_same_v<mesh_type, Unstructured>)
            {
                write_vtu(filename, unstructured_vtu_topology(mesh));
            }
            else if constexpr (std::is_same_v<mesh_type, MultiRegion>)
            {
                write_vtu(filename, composite_vtu_topology(mesh));
            }
            else
            {
                write_vtu(filename, orthogonal_vtu_topology(mesh));
            }
        });
}

/**
 * @brief Build reusable VTU topology for the owned cells.
 *
 * @return Immutable topology in the same owned-cell order used by fields.
 */
template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::composite_vtu_topology(const MultiRegion& mesh) const
{
    return mesh.vtu_topology({this, 0, num_owned_cells(), [](const void* source, size_t, size_t cell) -> uint64_t
    {
        return static_cast<const MeshHandle*>(source)->geometry_cell_lid(checked_local(cell));
    }});
}

template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::vtu_topology() const
{
    return visit(
        [&](const auto& mesh) -> VTUWriter::TopologyHandle
        {
            using mesh_type = std::decay_t<decltype(mesh)>;
            if constexpr (std::is_same_v<mesh_type, STKAdapter>)
            {
                return legacy_vtu_topology(mesh);
            }
            else if constexpr (std::is_same_v<mesh_type, SemiStructured>)
            {
                return semi_structured_vtu_topology(mesh);
            }
            else if constexpr (std::is_same_v<mesh_type, Unstructured>)
            {
                return unstructured_vtu_topology(mesh);
            }
            else if constexpr (std::is_same_v<mesh_type, MultiRegion>)
            {
                return composite_vtu_topology(mesh);
            }
            else
            {
                return orthogonal_vtu_topology(mesh);
            }
        });
}

/**
 * @brief Add the MPI rank suffix required for per-rank VTU output.
 *
 * @param filename Requested output filename.
 * @return The unchanged serial filename or a rank-specific parallel filename.
 */
template<TpetraTypePack Pack>
std::string MeshHandle<Pack>::local_output_filename(
    const std::string& filename) const
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() <= 1)
    {
        return filename;
    }

    const auto suffix = "_rank" + std::to_string(comm->getRank());
    const auto dot = filename.rfind('.');
    return dot == std::string::npos
         ? filename + suffix
         : filename.substr(0, dot) + suffix + filename.substr(dot);
}

/**
 * @brief Add global IDs, volumes, and centroids for owned VTU cells.
 *
 * @param writer Writer receiving the geometry cell-data arrays.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::add_geometry_cell_data(VTUWriter& writer) const
{
    const auto execution = acquire_execution_view();
    VTUWriter::Int64Data cell_ids;
    VTUWriter::ScalarData cell_volumes;
    VTUWriter::VectorData cell_centroids;
    cell_ids.reserve(num_owned_cells());
    cell_volumes.reserve(num_owned_cells());
    cell_centroids.reserve(num_owned_cells());
    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto local_id = checked_local(lid);
        cell_ids.push_back(static_cast<global_index_t>(
            d_semi_structured_topology ? cell_geometry_global_id(local_id) : cell_global_id(local_id)));
        cell_volumes.push_back(cell_volume(local_id));
        cell_centroids.push_back(cell_centroid(local_id));
    }

    if (const auto* composite = std::get_if<MultiRegionPtr>(&d_mesh))
    {
        VTUWriter::Int64Data regions;
        regions.reserve(num_owned_cells());
        for (size_t c = 0; c < num_owned_cells(); ++c)
            regions.push_back((*composite)->native_cell(geometry_cell_lid(checked_local(c))).first);
        writer.add_int64_cell_data("topology_region", std::move(regions));
    }
    writer.add_int64_cell_data("cell_gid", std::move(cell_ids));
    writer.add_scalar_cell_data("cell_volume", std::move(cell_volumes));
    writer.add_vector_cell_data("cell_centroid", std::move(cell_centroids));
}

/**
 * @brief Collect all mesh node coordinates as VTU points.
 *
 * @tparam MeshType Concrete mesh type.
 * @param mesh Mesh providing node coordinates.
 * @return Point coordinates in mesh-local node order.
 */
template<TpetraTypePack Pack>
template<class MeshType>
VTUWriter::VectorData MeshHandle<Pack>::collect_vtu_points(
    const MeshType& mesh) const
{
    VTUWriter::VectorData points;
    points.reserve(mesh.num_nodes());
    for (size_t lid = 0; lid < mesh.num_nodes(); ++lid)
    {
        points.push_back(mesh.node_coordinates(mesh.node_id(lid)));
    }
    return points;
}

/**
 * @brief Write assembled geometry and owned-cell data to a VTU file.
 *
 * @param filename Requested output filename.
 * @param points VTU point coordinates.
 * @param connectivity Flattened cell-to-point connectivity.
 * @param offsets End offsets for each cell in the connectivity array.
 * @param cell_types VTK cell type identifiers.
 */
template<TpetraTypePack Pack>
void MeshHandle<Pack>::write_vtu(
    const std::string& filename,
    VTUWriter::TopologyHandle topology) const
{
    VTUWriter writer(std::move(topology));
    add_geometry_cell_data(writer);
    writer.write(local_output_filename(filename));
}

/**
 * @brief Build topology for the exact legacy mesh wrapped by this handle.
 */
template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::legacy_vtu_topology(
    const STKAdapter& mesh) const
{
    std::unordered_map<global_ordinal_type, global_index_t> node_lids;
    VTUWriter::VectorData points;
    VTUWriter::Int64Data connectivity;
    VTUWriter::Int64Data offsets;
    VTUWriter::UInt8Data cell_types;
    offsets.reserve(num_owned_cells());
    cell_types.reserve(num_owned_cells());

    const auto& legacy = mesh.mesh();
    auto append_node =
        [&](global_ordinal_type node_gid) -> global_index_t
    {
        const auto [position, inserted] = node_lids.emplace(
            node_gid, static_cast<global_index_t>(points.size()));
        if (inserted)
        {
            points.push_back(legacy.node_coord(node_gid));
        }
        return position->second;
    };

    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto geometry_lid = geometry_cell_lid(checked_local(lid));
        const auto& cell = legacy.cell(checked_local(
            static_cast<size_t>(geometry_lid)));
        for (const auto node_gid : cell.node_gids)
        {
            connectivity.push_back(append_node(node_gid));
        }
        offsets.push_back(static_cast<global_index_t>(connectivity.size()));
        cell_types.push_back(static_cast<std::uint8_t>(
            MeshUtils::vtu_cell_type_code(cell.type)));
    }

    return VTUWriter::make_topology(
        std::move(points),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types));
}

/**
 * @brief Export owned orthogonal cells as VTK hexahedra.
 *
 * @tparam MeshType Cartesian or cylindrical orthogonal mesh type.
 * @param mesh Mesh providing node and cell indexing.
 * @param filename Requested output filename.
 */
template<TpetraTypePack Pack>
template<class MeshType>
VTUWriter::TopologyHandle MeshHandle<Pack>::orthogonal_vtu_topology(
    const MeshType& mesh) const
{
    VTUWriter::Int64Data connectivity;
    VTUWriter::Int64Data offsets;
    VTUWriter::UInt8Data cell_types;
    connectivity.reserve(num_owned_cells() * 8);
    offsets.reserve(num_owned_cells());
    cell_types.reserve(num_owned_cells());

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
    if constexpr (std::is_same_v<MeshType, Cartesian>)
    {
        const auto& indexer = mesh.indexer();
        if (std::any_of(indexer.periodic_dimensions.begin(), indexer.periodic_dimensions.end(), [](bool p) { return p; }))
        {
            // Periodic topological vertices have distinct physical endpoint images.
            const auto& edges = mesh.cell_edges();
            VTUWriter::VectorData points;
            for (auto z : edges[2]) for (auto y : edges[1]) for (auto x : edges[0]) points.push_back({x,y,z});
            for (size_t lid=0; lid<num_owned_cells(); ++lid)
            {
                const auto cell=mesh.cell_id(static_cast<size_t>(geometry_cell_lid(checked_local(lid))));
                for (const auto& corner : corners)
                    connectivity.push_back((cell.i+corner[0])+edges[0].size()
                        *((cell.j+corner[1])+edges[1].size()*(cell.k+corner[2])));
                offsets.push_back(connectivity.size()); cell_types.push_back(12);
            }
            return VTUWriter::make_topology(std::move(points),std::move(connectivity),std::move(offsets),std::move(cell_types));
        }
    }
    const auto& indexer = mesh.indexer();
    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto cell = mesh.cell_id(static_cast<size_t>(
            geometry_cell_lid(checked_local(lid))));
        for (const auto& corner : corners)
        {
            const typename MeshType::NodeID node{
                static_cast<unsigned>(
                    (cell.i + corner[0]) % indexer.num_nodes_per_dim[0]),
                static_cast<unsigned>(
                    (cell.j + corner[1]) % indexer.num_nodes_per_dim[1]),
                static_cast<unsigned>(
                    (cell.k + corner[2]) % indexer.num_nodes_per_dim[2])};
            connectivity.push_back(static_cast<global_index_t>(
                mesh.node_local_id(node)));
        }
        offsets.push_back(static_cast<global_index_t>(connectivity.size()));
        cell_types.push_back(12); // VTK_HEXAHEDRON
    }

    return VTUWriter::make_topology(
        collect_vtu_points(mesh),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types));
}

/**
 * @brief Export owned semi-structured cells using their extruded topology.
 *
 * Triangular base cells become wedges, quadrilateral base cells become
 * hexahedra, and other polygons retain their face loops as VTK polyhedra.
 *
 * @param mesh Semi-structured mesh providing layered node connectivity.
 * @param filename Requested output filename.
 */
template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::semi_structured_vtu_topology(
    const SemiStructured& mesh) const
{
    const auto execution = acquire_execution_view();
    VTUWriter::VectorData points;
    std::unordered_map<size_t, global_index_t> point_lids;
    if (d_serial_identity) points = collect_vtu_points(mesh);
    const auto local_node = [&](SemiStructured::NodeID node) -> global_index_t
    {
        const auto global = mesh.node_local_id(node);
        if (d_serial_identity) return static_cast<global_index_t>(global);
        const auto [position, inserted] = point_lids.emplace(global, static_cast<global_index_t>(points.size()));
        if (inserted) points.push_back(mesh.node_coordinates(node));
        return position->second;
    };
    VTUWriter::Int64Data connectivity;
    VTUWriter::Int64Data offsets;
    VTUWriter::Int64Data faces;
    VTUWriter::Int64Data face_offsets;
    VTUWriter::UInt8Data cell_types;
    offsets.reserve(num_owned_cells());
    cell_types.reserve(num_owned_cells());
    const auto& indexer = mesh.indexer();
    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto cell = mesh.cell_id(static_cast<size_t>(
            geometry_cell_lid(checked_local(lid))));
        const auto& xy_nodes = mesh.xy_cell_nodes()[cell.ij];
        const auto top_layer = (cell.k + 1) % indexer.num_node_layers;
        for (const auto xy_node : xy_nodes)
        {
            connectivity.push_back(local_node(SemiStructured::NodeID{xy_node, cell.k}));
        }
        for (const auto xy_node : xy_nodes)
        {
            connectivity.push_back(local_node(SemiStructured::NodeID{xy_node, top_layer}));
        }

        offsets.push_back(static_cast<global_index_t>(connectivity.size()));
        if (xy_nodes.size() == 3)
        {
            cell_types.push_back(13); // VTK_WEDGE
            face_offsets.push_back(-1);
        }
        else if (xy_nodes.size() == 4)
        {
            cell_types.push_back(12); // VTK_HEXAHEDRON
            face_offsets.push_back(-1);
        }
        else
        {
            cell_types.push_back(42); // VTK_POLYHEDRON
            const auto node = [&](size_t i, auto layer)
            {
                return local_node(SemiStructured::NodeID{xy_nodes[i], layer});
            };
            faces.push_back(static_cast<global_index_t>(xy_nodes.size() + 2));
            // Base loops are counter-clockwise; the lower cap points down.
            faces.push_back(static_cast<global_index_t>(xy_nodes.size()));
            for (size_t i = xy_nodes.size(); i > 0; --i) faces.push_back(node(i - 1, cell.k));
            faces.push_back(static_cast<global_index_t>(xy_nodes.size()));
            for (size_t i = 0; i < xy_nodes.size(); ++i) faces.push_back(node(i, top_layer));
            for (size_t i = 0; i < xy_nodes.size(); ++i)
            {
                const auto next = (i + 1) % xy_nodes.size();
                faces.insert(faces.end(), {4, node(i, cell.k), node(next, cell.k),
                    node(next, top_layer), node(i, top_layer)});
            }
            face_offsets.push_back(static_cast<global_index_t>(faces.size()));
        }
    }
    if (faces.empty()) face_offsets.clear();

    return VTUWriter::make_topology(
        std::move(points),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types),
        std::move(faces),
        std::move(face_offsets));
}

/**
 * @brief Export owned unstructured cells using their explicit node lists.
 *
 * @param mesh Unstructured mesh providing node connectivity.
 * @param filename Requested output filename.
 */
template<TpetraTypePack Pack>
VTUWriter::TopologyHandle MeshHandle<Pack>::unstructured_vtu_topology(
    const Unstructured& mesh) const
{
    VTUWriter::Int64Data connectivity;
    VTUWriter::Int64Data offsets;
    VTUWriter::Int64Data faces;
    VTUWriter::Int64Data face_offsets;
    VTUWriter::UInt8Data cell_types;
    offsets.reserve(num_owned_cells());
    cell_types.reserve(num_owned_cells());

    for (size_t lid = 0; lid < num_owned_cells(); ++lid)
    {
        const auto cell = mesh.cell_id(static_cast<size_t>(
            geometry_cell_lid(checked_local(lid))));
        for (const auto node : mesh.cell_nodes(cell))
        {
            connectivity.push_back(static_cast<global_index_t>(
                mesh.node_local_id(node)));
        }
        offsets.push_back(static_cast<global_index_t>(
            connectivity.size()));
        const auto type = mesh.cell_type(cell);
        cell_types.push_back(static_cast<std::uint8_t>(MeshUtils::vtu_cell_type_code(type)));
        if (type == MeshUtils::CellType::POLYHEDRON)
        {
            const auto& cell_faces = mesh.faces(cell);
            faces.push_back(static_cast<global_index_t>(cell_faces.size()));
            for (const auto face : cell_faces)
            {
                const auto& nodes = mesh.face_nodes(face);
                faces.push_back(static_cast<global_index_t>(nodes.size()));
                for (size_t i = 0; i < nodes.size(); ++i)
                {
                    const auto index = mesh.owner_cell(face) == cell ? i : nodes.size() - 1 - i;
                    faces.push_back(static_cast<global_index_t>(mesh.node_local_id(nodes[index])));
                }
            }
            face_offsets.push_back(static_cast<global_index_t>(faces.size()));
        }
        else face_offsets.push_back(-1);
    }
    if (faces.empty()) face_offsets.clear();

    return VTUWriter::make_topology(
        collect_vtu_points(mesh),
        std::move(connectivity),
        std::move(offsets),
        std::move(cell_types),
        std::move(faces),
        std::move(face_offsets));
}

} // namespace SimpleFluid
