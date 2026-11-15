/** @file STKMeshInput.cc
 * @brief Transient STK/Ioss extraction into native construction records.
 */
#include "io/STKMeshInput.hh"
#include "geometry/details/NativeMeshAssembly.hh"
#include "io/details/STKBackendBridge.hh"
#include "io/details/STKFaceAuthority.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <stk_mesh/base/Part.hpp>
#include <stk_topology/topology.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail
{
namespace
{
using Data = Meshes::UnstructuredMeshData;
using ID = Data::ID;
using Key = std::vector<ID>;

void agree(const Teuchos::Comm<int>& comm, const std::exception_ptr& error, const char* message)
{
    const int failed = error ? 1 : 0;
    int any = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &failed, &any);
    if (any)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::runtime_error(message);
    }
}

Key key_for(const stk::mesh::BulkData& bulk, stk::mesh::Entity entity)
{
    Key key;
    for (unsigned i = 0; i < bulk.num_nodes(entity); ++i)
        key.push_back(bulk.identifier(bulk.begin_nodes(entity)[i]));
    std::sort(key.begin(), key.end());
    return key;
}

bool candidate(const stk::mesh::Part& part, stk::mesh::EntityRank side_rank, const STKMeshInputOptions& options)
{
    return part.primary_entity_rank() == side_rank && !stk::mesh::is_auto_declared_part(part) &&
           !stk::mesh::is_topology_root_part(part) &&
           (options.boundary_name_to_id.contains(part.name()) || options.auto_assign_boundary_ids);
}

const stk::mesh::Part* choose_part(
    const stk::mesh::Bucket& bucket, stk::mesh::EntityRank rank, const STKMeshInputOptions& options)
{
    const stk::mesh::Part* first = nullptr;
    const stk::mesh::Part* surface = nullptr;
    for (const auto* part : bucket.supersets())
    {
        if (!part || !candidate(*part, rank, options))
            continue;
        if (options.boundary_name_to_id.contains(part->name()))
            return part;
        if (!surface && stk::io::is_part_surface_io_part(*part))
            surface = part;
        if (!first)
            first = part;
    }
    return surface ? surface : first;
}

std::vector<std::string> gather_names(const Teuchos::Comm<int>& comm, const std::vector<std::string>& local)
{
    std::vector<char> bytes;
    std::vector<int> counts;
    std::exception_ptr error;
    try
    {
        counts.resize(comm.getSize());
        for (const auto& name : local)
        {
            if (name.find('\0') != std::string::npos)
                throw std::invalid_argument("Boundary name contains a null byte.");
            bytes.insert(bytes.end(), name.begin(), name.end());
            bytes.push_back('\0');
        }
        if (bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("STK boundary metadata exceeds collective count.");
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(comm, error, "STK boundary metadata could not be encoded.");
    const int count = static_cast<int>(bytes.size());
    Teuchos::gatherAll(comm, 1, &count, comm.getSize(), counts.data());
    const int width = *std::max_element(counts.begin(), counts.end());
    std::vector<char> all;
    error = {};
    try
    {
        if (width && comm.getSize() > std::numeric_limits<int>::max() / width)
            throw std::overflow_error("STK boundary catalog exceeds collective count.");
        bytes.resize(width);
        all.resize(static_cast<size_t>(width) * comm.getSize());
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(comm, error, "STK boundary catalog allocation failed.");
    if (width)
        Teuchos::gatherAll(comm, width, bytes.data(), width * comm.getSize(), all.data());
    std::vector<std::string> result;
    error = {};
    try
    {
        std::set<std::string> seen;
        for (int rank = 0; rank < comm.getSize(); ++rank)
        {
            size_t first = static_cast<size_t>(rank) * width;
            const size_t end = first + counts[rank];
            while (first < end)
            {
                const auto last = std::find(all.begin() + first, all.begin() + end, '\0');
                if (last == all.begin() + end)
                    throw std::runtime_error("Truncated STK boundary metadata.");
                std::string name(all.begin() + first, last);
                if (seen.insert(name).second)
                    result.push_back(std::move(name));
                first = static_cast<size_t>(last - all.begin()) + 1;
            }
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(comm, error, "STK boundary catalog decoding failed on another rank.");
    return result;
}
} // namespace

Data read_stk_mesh_input(
    const std::string& filename, const STKMeshInputOptions& options, const Teuchos::RCP<const Teuchos::Comm<int>>& comm)
{
    if (comm.is_null())
        throw std::invalid_argument("STK input requires a communicator.");
    std::exception_ptr error;
    try
    {
        if (filename.empty() || filename.find('\0') != std::string::npos)
            throw std::invalid_argument("STK input requires a nonempty filename without null bytes.");
        if (options.decomposition_method.find('\0') != std::string::npos)
            throw std::invalid_argument("STK decomposition method contains a null byte.");
        std::map<int, std::string> names;
        for (const auto& [name, id] : options.boundary_name_to_id)
        {
            if (name.empty() || name.find('\0') != std::string::npos || id == -1 || !names.emplace(id, name).second)
                throw std::invalid_argument("STK boundary IDs/names must be nonempty and unique.");
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "Invalid STK input options on another rank.");

    // All ranks must enter Ioss with the same file and boundary policy.
    std::string signature;
    unsigned long long hash = 14695981039346656037ull;
    error = {};
    try
    {
        signature = filename + (options.auto_assign_boundary_ids ? "\1" : "\2") + options.decomposition_method;
        std::map<std::string, int> ordered(options.boundary_name_to_id.begin(), options.boundary_name_to_id.end());
        for (const auto& [name, id] : ordered)
            signature += '\0' + name + '\0' + std::to_string(id);
        for (unsigned char c : signature)
        {
            hash ^= c;
            hash *= 1099511628211ull;
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK input signature preparation failed on another rank.");
    unsigned long long low = 0, high = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &hash, &low);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &hash, &high);
    if (low != high)
        throw std::invalid_argument("STK input filename/options differ between ranks.");

    MPI_Comm raw = MPI_COMM_SELF;
    if (const auto* mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(comm.getRawPtr()))
        raw = *mpi->getRawMpiComm();
    else if (comm->getSize() != 1)
        throw std::invalid_argument("STK input requires an MPI or serial communicator.");

    std::string input_decomposition;
    int single_file = 0, local_piece = 0;
    error = {};
    try
    {
        if (options.decomposition_method == "auto")
        {
            if (comm->getRank() == 0)
                single_file = std::filesystem::is_regular_file(filename) ? 1 : 0;
            if (comm->getSize() > 1)
                local_piece = std::filesystem::is_regular_file(
                                  stk_decomposed_filename(filename, comm->getRank(), comm->getSize()))
                                  ? 1
                                  : 0;
        }
        else
            input_decomposition = options.decomposition_method;
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK input file policy preparation failed on another rank.");
    int all_pieces = 0, any_piece = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_piece, &all_pieces);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_piece, &any_piece);
    Teuchos::broadcast(*comm, 0, 1, &single_file);
    error = {};
    try
    {
        if (options.decomposition_method == "auto")
        {
            if (any_piece && !all_pieces)
                throw std::invalid_argument("STK decomposed input set is incomplete.");
            if (!all_pieces && !single_file)
                throw std::invalid_argument("STK input file does not exist.");
            if (!all_pieces && single_file && comm->getSize() > 1)
                input_decomposition = "LINEAR";
        }
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK input decomposition preparation failed on another rank.");

    Data data;
    std::vector<std::string> declared_names;
    std::optional<stk::mesh::MeshBuilder> builder;
    std::shared_ptr<stk::mesh::MetaData> meta;
    std::unique_ptr<stk::mesh::BulkData> bulk;
    std::unique_ptr<stk::io::StkMeshIoBroker> io;
    error = {};
    try
    {
        builder.emplace(raw);
        builder->set_spatial_dimension(3);
        meta = builder->create_meta_data();
        bulk = builder->create(meta);
        io = std::make_unique<stk::io::StkMeshIoBroker>(raw);
        io->set_bulk_data(*bulk);
        if (!input_decomposition.empty())
            simplefluid_trilinos_stk_io_set_string_property(
                io.get(), "DECOMPOSITION_METHOD", input_decomposition.c_str());
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK reader setup failed on another rank.");
    error = {};
    try
    {
        io->add_mesh_database(filename, "exodus", stk::io::READ_MESH);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK input database could not be opened on another rank.");
    error = {};
    try
    {
        io->create_input_mesh();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK input metadata could not be read on another rank.");
    error = {};
    try
    {
        io->populate_bulk_data();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK bulk input failed on another rank.");
    error = {};
    try
    {
        const auto side_rank = meta->side_rank();
        for (const auto* part : meta->get_parts())
            if (part && candidate(*part, side_rank, options))
                declared_names.push_back(part->name());
        auto* coordinates =
            dynamic_cast<stk::mesh::Field<double>*>(const_cast<stk::mesh::FieldBase*>(meta->coordinate_field()));
        if (!coordinates && !meta->coordinate_field_name().empty())
            coordinates = meta->get_field<double>(stk::topology::NODE_RANK, meta->coordinate_field_name());
        if (!coordinates)
            coordinates = meta->get_field<double>(stk::topology::NODE_RANK, "coordinates");
        if (!coordinates)
            throw std::runtime_error("STK input has no double coordinates field.");

        std::map<Key, std::string> labels;
        for (const auto* bucket : bulk->buckets(side_rank))
        {
            const auto* part = choose_part(*bucket, side_rank, options);
            if (!part)
                continue;
            for (auto entity : *bucket)
                labels.emplace(key_for(*bulk, entity), part->name());
        }
        std::vector<Data::Node> nodes;
        std::vector<Data::Cell> cells;
        std::vector<NativeBoundarySide> boundaries;
        std::set<ID> seen_nodes;
        size_t owned = 0;
        for (bool take_owned : {true, false})
        {
            for (const auto* bucket : bulk->buckets(stk::topology::ELEMENT_RANK))
            {
                if (bucket->owned() != take_owned)
                    continue;
                const auto topology = bucket->topology();
                if (bucket->size() && topology != stk::topology::HEX_8 && topology != stk::topology::WEDGE_6)
                    throw std::invalid_argument("STK input supports only HEX_8 and WEDGE_6 volume elements.");
                for (auto element : *bucket)
                {
                    Data::Cell cell;
                    cell.id = bulk->identifier(element);
                    cell.type = topology == stk::topology::HEX_8 ? MeshUtils::CellType::HEXAHEDRON
                                                                 : MeshUtils::CellType::TRIPRISM;
                    const auto* element_nodes = bulk->begin_nodes(element);
                    for (unsigned i = 0; i < bulk->num_nodes(element); ++i)
                    {
                        const auto node = element_nodes[i];
                        const ID id = bulk->identifier(node);
                        cell.node_ids.push_back(id);
                        if (seen_nodes.insert(id).second)
                        {
                            const auto* x = stk::mesh::field_data(*coordinates, node);
                            if (!x)
                                throw std::runtime_error("STK node has no coordinate data.");
                            const auto dim = meta->spatial_dimension();
                            nodes.push_back({id, {x[0], dim > 1 ? x[1] : 0.0, dim > 2 ? x[2] : 0.0}});
                        }
                    }
                    for (unsigned side = 0; side < topology.num_sides(); ++side)
                    {
                        std::vector<unsigned> ordinals(topology.side_topology(side).num_nodes());
                        topology.side_node_ordinals(side, ordinals.begin());
                        Key key;
                        for (auto ordinal : ordinals)
                            key.push_back(cell.node_ids.at(ordinal));
                        std::sort(key.begin(), key.end());
                        if (auto found = labels.find(key); found != labels.end())
                            boundaries.push_back({cell.id, side, found->second});
                    }
                    cells.push_back(std::move(cell));
                    if (take_owned)
                        ++owned;
                }
            }
        }
        data = build_native_mesh_data(std::move(nodes), std::move(cells), boundaries, declared_names);
        data.num_owned_cells = owned;
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK extraction failed on another rank.");
    io.reset();
    bulk.reset();
    meta.reset();
    builder.reset();

    std::vector<std::string> local_names;
    error = {};
    try
    {
        std::vector<std::pair<int, std::string>> local_order(data.boundary_names.begin(), data.boundary_names.end());
        std::sort(local_order.begin(), local_order.end());
        for (const auto& [id, name] : local_order)
            local_names.push_back(name);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK boundary catalog preparation failed on another rank.");
    const auto catalog = gather_names(*comm, local_names);
    error = {};
    try
    {
        std::unordered_map<std::string, int> ids = options.boundary_name_to_id;
        std::map<int, std::string> names;
        std::int64_t next = 1;
        for (const auto& [name, id] : ids)
        {
            names.emplace(id, name);
            next = std::max(next, static_cast<std::int64_t>(id) + 1);
        }
        for (const auto& name : catalog)
            if (!ids.contains(name))
            {
                if (!options.auto_assign_boundary_ids)
                    continue;
                if (next > std::numeric_limits<int>::max())
                    throw std::overflow_error("STK boundary ID overflow.");
                ids.emplace(name, static_cast<int>(next));
                names.emplace(static_cast<int>(next++), name);
            }
        for (auto& face : data.faces)
            if (face.boundary_id != -1)
                face.boundary_id = ids.at(data.boundary_names.at(face.boundary_id));
        data.boundary_names.clear();
        for (const auto& [id, name] : names)
            data.boundary_names.emplace(id, name);
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(*comm, error, "STK boundary catalog publication failed on another rank.");
    synchronize_stk_face_authority(data, comm);
    return data;
}
} // namespace SimpleFluid::detail
