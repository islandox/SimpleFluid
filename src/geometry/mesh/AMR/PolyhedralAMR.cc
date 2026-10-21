/** @file PolyhedralAMR.cc
 * @brief Collective convex split/coarsen transactions and bounded canonical topology.
 */
#include "geometry/mesh/AMR/PolyhedralAMR.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

namespace SimpleFluid::Meshes::AMR
{
namespace
{
using ID = std::uint64_t;
struct Node
{
    ConvexGeometryPtr geometry;
    std::vector<PolyhedralSurfaceUse> surfaces;
    std::vector<AdaptiveCellKey> children;
    ID material = 0;
};
template<class Function> void collectively(Function&& function)
{
    std::exception_ptr error;
    try
    {
        function();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    int local = bool(error), failed = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_MAX, 1, &local, &failed);
    if (failed)
    {
        if (error)
            std::rethrow_exception(error);
        throw std::invalid_argument("Polyhedral AMR request failed on another rank.");
    }
}
void same(const std::string& text)
{
    const auto comm = Tpetra::getDefaultComm();
    collectively(
        [&]
        {
            if (text.size() > std::size_t(std::numeric_limits<int>::max()))
                throw std::overflow_error("Polyhedral descriptor too large.");
        });
    int length = comm->getRank() == 0 ? static_cast<int>(text.size()) : 0;
    Teuchos::broadcast(*comm, 0, 1, &length);
    std::string source = comm->getRank() == 0 ? text : std::string(length, '\0');
    Teuchos::broadcast(*comm, 0, length, source.data());
    collectively(
        [&]
        {
            if (text != source)
                throw std::invalid_argument("Polyhedral AMR descriptions disagree.");
        });
}
ID identity()
{
    static std::atomic<ID> counter{0};
    const auto comm = Tpetra::getDefaultComm();
    ID result = 0;
    if (comm->getRank() == 0)
    {
        auto previous = counter.load();
        while (previous != std::numeric_limits<ID>::max())
            if (counter.compare_exchange_weak(previous, previous + 1))
            {
                result = previous + 1;
                break;
            }
    }
    Teuchos::broadcast(*comm, 0, 1, &result);
    if (!result)
        throw std::overflow_error("Polyhedral AMR collective identity exhausted.");
    return result;
}
ID add(ID a, ID b)
{
    if (b > std::numeric_limits<ID>::max() - a)
        throw std::overflow_error("Polyhedral bootstrap growth overflow.");
    return a + b;
}
ID multiply(ID a, ID b)
{
    if (b && a > std::numeric_limits<ID>::max() / b)
        throw std::overflow_error("Polyhedral bootstrap growth overflow.");
    return a * b;
}
std::string key_text(const AdaptiveCellKey& key)
{
    const auto& ancestry = std::get<ConvexCellAncestry>(key.ancestry);
    std::ostringstream out;
    out << key.region << ' ' << ancestry.root << ' ' << ancestry.path.size() << ' ';
    for (const auto octant : ancestry.path)
        out << octant << ' ';
    return out.str();
}
} // namespace
struct PolyhedralAMR::Data
{
    std::shared_ptr<const PolyhedralAMRGrid> grid;
    std::map<AdaptiveCellKey, Node> nodes;
    std::set<AdaptiveCellKey> leaves;
    std::vector<AdaptiveCellKey> keys;
    std::shared_ptr<const ExactPolyhedralComplex> complex;
    SP<const MultiRegionMesh> geometry;
    std::shared_ptr<const CompositePartition<>> partition;
    std::shared_ptr<const void> root_identity, instance, topology, ownership;
    ID collective = 0, generation = 0, partition_generation = 0, geometry_epoch = 0;
    std::map<int, std::string> boundaries;
};
namespace
{
void build(PolyhedralAMR::Data& data)
{
    data.keys.assign(data.leaves.begin(), data.leaves.end());
    if (data.keys.empty() || data.keys.size() > data.grid->maximum_cells)
        throw std::invalid_argument("Polyhedral bootstrap cell budget exhausted.");
    std::vector<PolyhedralComplexCell> cells;
    std::map<PolyhedralSurfaceKey, std::array<ID, 3>> groups;
    for (ID gid = 0; gid < data.keys.size(); ++gid)
    {
        const auto& key = data.keys[gid];
        const auto& node = data.nodes.at(key);
        cells.push_back({key, gid, node.geometry, node.surfaces});
        for (const auto& use : node.surfaces)
        {
            auto& counts = groups[use.key];
            auto& count = counts[use.exterior ? 2 : use.orientation > 0 ? 0 : 1];
            count = add(count, 1);
        }
    }
    ID upper_faces = 0;
    for (const auto& [key, counts] : groups)
        upper_faces = add(upper_faces, add(counts[2], multiply(counts[0], counts[1])));
    if (upper_faces > data.grid->maximum_face_tiles)
        throw std::invalid_argument("Polyhedral bootstrap face-overlay budget exhausted before expansion.");
    data.complex = ExactPolyhedralComplex::make(std::move(cells));
    if (data.complex->global_faces() > data.grid->maximum_face_tiles)
        throw std::invalid_argument("Polyhedral bootstrap actual face budget exhausted.");
    const auto provider = ExplicitRegionProvider::from_exact_complex(data.complex, data.boundaries);
    data.geometry = std::make_shared<MultiRegionMesh>(
        std::vector<MultiRegionMesh::Region>{DistributedExplicitRegion("polyhedral", provider, provider)},
        std::vector<MultiRegionMesh::Interface>{}, Teuchos::rcp(new Teuchos::SerialComm<int>()), InterfaceTolerance{},
        MultiRegionMesh::BoundaryNamePolicy::MergeMatchingNames);
    data.geometry_epoch = data.geometry->geometry_epoch();
    data.partition.reset();
}
} // namespace

PolyhedralAMR::PolyhedralAMR(PolyhedralAMRGrid grid)
{
    auto data = std::make_shared<Data>();
    collectively(
        [&]
        {
            if (!grid.mesh || !grid.maximum_cells || !grid.maximum_face_tiles || !grid.max_depth ||
                grid.max_depth > 64 || !std::isfinite(grid.minimum_scaled_volume) || grid.minimum_scaled_volume < 0 ||
                grid.mesh->num_cells() > grid.maximum_cells || grid.mesh->num_faces() > grid.maximum_face_tiles ||
                grid.mesh->regions().size() != 1 || !grid.mesh->interfaces().empty() ||
                (!grid.physical_regions.empty() && grid.physical_regions.size() != grid.mesh->num_cells()) ||
                (!grid.materials.empty() && grid.materials.size() != grid.mesh->num_cells()))
                throw std::invalid_argument("Invalid polyhedral input or bootstrap budget.");
            if (!grid.physical_regions.empty() &&
                !std::all_of(grid.physical_regions.begin(), grid.physical_regions.end(),
                    [&](ID region) { return region == grid.physical_regions.front(); }))
                throw std::invalid_argument(
                    "Polyhedral Stage 4 requires one physical region; mixed-region closure is Stage 5.");
        });
    std::string description;
    collectively(
        [&]
        {
            std::ostringstream signature;
            signature << grid.mesh->configuration_signature() << ' ' << grid.max_depth << ' ' << std::hexfloat
                      << grid.minimum_scaled_volume << ' ' << grid.maximum_cells << ' ' << grid.maximum_face_tiles
                      << ' ';
            for (const auto region : grid.physical_regions)
                signature << region << ' ';
            signature << " materials ";
            for (const auto material : grid.materials)
                signature << material << ' ';
            description = signature.str();
        });
    same(description);
    data->grid = std::make_shared<const PolyhedralAMRGrid>(std::move(grid));
    collectively(
        [&]
        {
            const auto& mesh = *data->grid->mesh;
            for (ID c = 0; c < mesh.num_cells(); ++c)
            {
                const ID region = data->grid->physical_regions.empty() ? 0 : data->grid->physical_regions[c];
                const ID material = data->grid->materials.empty() ? 0 : data->grid->materials[c];
                const AdaptiveCellKey key{region, ConvexCellAncestry{c, {}}};
                Node node;
                node.geometry = ConvexCellGeometry::from_native_cell_atomic(mesh, c);
                node.material = material;
                if (!node.geometry->quality_above(data->grid->minimum_scaled_volume))
                    throw std::invalid_argument("Polyhedral root fails configured exact quality admission.");
                for (const auto face : mesh.cell_faces(c))
                {
                    PolyhedralSurfaceUse use;
                    use.key = {PolyhedralSurfaceKey::Kind::Root, region, ConvexCellAncestry{face, {}}, 0};
                    use.orientation = mesh.owner_cell(face) == c ? 1 : -1;
                    use.exterior = mesh.is_exterior_face(face);
                    if (use.exterior && mesh.boundary_id(face) != UnstructuredMesh::invalid_boundary_id)
                    {
                        use.boundary = mesh.boundary_name(face);
                        data->boundaries.emplace(mesh.boundary_id(face), use.boundary);
                    }
                    node.surfaces.push_back(std::move(use));
                }
                if (node.surfaces.size() != node.geometry->num_faces())
                    throw std::invalid_argument(
                        "Native canonical cell faces require an explicit authoritative surface mapping.");
                data->nodes.emplace(key, std::move(node));
                data->leaves.insert(key);
            }
            build(*data);
        });
    std::string exact_descriptor;
    collectively(
        [&]
        {
            const auto packet = data->complex->encode();
            exact_descriptor.assign(packet.begin(), packet.end());
        });
    same(exact_descriptor);
    data->root_identity = std::make_shared<const char>(0);
    data->instance = data->root_identity;
    data->topology = data->instance;
    data->ownership = data->instance;
    data->collective = identity();
    data_ = std::move(data);
}
HierarchyObservation PolyhedralAMR::observation() const noexcept
{
    const auto& geometry = data_->partition ? data_->partition->geometry() : data_->geometry;
    ID epoch = std::numeric_limits<ID>::max();
    try
    {
        epoch = geometry->geometry_epoch();
    }
    catch (...)
    {
    }
    return {data_->instance, data_->topology, data_->ownership, data_->generation, data_->partition_generation, epoch,
        geometry.get(), Tpetra::getDefaultComm().getRawPtr()};
}
bool PolyhedralAMR::geometry_unchanged() const noexcept
{
    try
    {
        return data_->geometry->geometry_epoch() == data_->geometry_epoch &&
               (!data_->partition || data_->partition->geometry_unchanged());
    }
    catch (...)
    {
        return false;
    }
}
const SP<const MultiRegionMesh>& PolyhedralAMR::native_geometry() const noexcept
{
    return data_->geometry;
}
const SP<const MultiRegionMesh>& PolyhedralAMR::geometry() const noexcept
{
    return data_->geometry;
}
const std::shared_ptr<const CompositePartition<>>& PolyhedralAMR::native_partition() const noexcept
{
    return data_->partition;
}
const std::shared_ptr<const ExactPolyhedralComplex>& PolyhedralAMR::complex() const noexcept
{
    return data_->complex;
}
AdaptiveCellKey PolyhedralAMR::cell_key(ID cell) const
{
    return data_->keys.at(cell);
}
ID PolyhedralAMR::cell_material(ID cell) const
{
    return data_->nodes.at(data_->keys.at(cell)).material;
}
std::size_t PolyhedralAMR::cell_fragment(ID cell) const
{
    (void) data_->keys.at(cell);
    return 0;
}
ConvexGeometryPtr PolyhedralAMR::cell_geometry(ID cell) const
{
    return data_->nodes.at(data_->keys.at(cell)).geometry;
}

std::shared_ptr<const PolyhedralAMR> PolyhedralAMR::with_partition(
    std::shared_ptr<const CompositePartition<>> partition) const
{
    collectively(
        [&]
        {
            const auto comm = Tpetra::getDefaultComm();
            if (!geometry_unchanged() || !partition || !partition->geometry_unchanged() ||
                partition->plan().communicator().getRawPtr() != comm.getRawPtr() ||
                partition->plan().global_cells() != data_->keys.size() ||
                (comm->getRank() == partition->plan().source_rank() &&
                    !partition->plan().matches_source(data_->geometry)))
                throw std::invalid_argument("Polyhedral partition has foreign source, geometry or communicator.");
            if (data_->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Polyhedral ownership generation exhausted.");
        });
    same(std::to_string(data_->collective) + " " + std::to_string(partition->plan().collective_identity()));
    same(data_->partition == partition ? "unchanged" : "replace");
    if (data_->partition == partition)
        return std::shared_ptr<const PolyhedralAMR>(new PolyhedralAMR(data_));
    auto next = std::make_shared<Data>(*data_);
    next->partition = std::move(partition);
    ++next->partition_generation;
    next->instance = std::make_shared<const char>(0);
    next->ownership = next->instance;
    next->collective = identity();
    return std::shared_ptr<const PolyhedralAMR>(new PolyhedralAMR(std::move(next)));
}

std::shared_ptr<const PolyhedralAMR> PolyhedralAMR::adapt(
    const std::vector<std::size_t>& refine, const std::vector<AdaptiveCellKey>& coarsen) const
{
    collectively(
        [&]
        {
            if (!geometry_unchanged())
                throw std::invalid_argument("Polyhedral hierarchy geometry moved.");
        });
    std::string description;
    collectively(
        [&]
        {
            std::ostringstream request;
            request << data_->collective << " refine ";
            for (const auto ordinal : refine)
                request << ordinal << ' ';
            request << " coarsen ";
            for (const auto& key : coarsen)
                request << key_text(key) << ';';
            description = request.str();
        });
    same(description);
    std::set<AdaptiveCellKey> refined, coarsened;
    collectively(
        [&]
        {
            if (data_->generation == std::numeric_limits<ID>::max() ||
                data_->partition_generation == std::numeric_limits<ID>::max())
                throw std::overflow_error("Polyhedral topology generation exhausted.");
            for (const auto ordinal : refine)
            {
                const auto& key = data_->keys.at(ordinal);
                if (!refined.insert(key).second ||
                    std::get<ConvexCellAncestry>(key.ancestry).path.size() >= data_->grid->max_depth)
                    throw std::invalid_argument("Invalid, repeated or depth-limited polyhedral refine request.");
            }
            for (const auto& key : coarsen)
            {
                const auto found = data_->nodes.find(key);
                if (!coarsened.insert(key).second || found == data_->nodes.end() || found->second.children.empty() ||
                    !std::all_of(found->second.children.begin(), found->second.children.end(),
                        [&](const auto& child) { return data_->leaves.contains(child) && !refined.contains(child); }))
                    throw std::invalid_argument(
                        "Polyhedral coarsening requires its unchanged complete immediate source family.");
            }
            ID upper = data_->leaves.size();
            for (const auto& key : coarsened)
                upper -= data_->nodes.at(key).children.size() - 1;
            upper = add(upper, multiply(7, refined.size()));
            if (upper > data_->grid->maximum_cells)
                throw std::invalid_argument("Polyhedral upper cell-growth bootstrap budget exhausted.");
        });
    if (refined.empty() && coarsened.empty())
        return std::shared_ptr<const PolyhedralAMR>(new PolyhedralAMR(data_));
    auto next = std::make_shared<Data>(*data_);
    collectively(
        [&]
        {
            for (const auto& key : coarsened)
            {
                auto& parent = next->nodes.at(key);
                for (const auto& child : parent.children)
                {
                    next->leaves.erase(child);
                    next->nodes.erase(child);
                }
                parent.children.clear();
                next->leaves.insert(key);
            }
            for (const auto& key : refined)
            {
                auto& parent = next->nodes.at(key);
                const auto pieces = parent.geometry->split_aabb_midpoints(data_->grid->minimum_scaled_volume);
                if (pieces.empty())
                    throw std::invalid_argument("Polyhedral cut produced no occupied child.");
                for (const auto& piece : pieces)
                {
                    AdaptiveCellKey child_key = key;
                    auto& ancestry = std::get<ConvexCellAncestry>(child_key.ancestry);
                    ancestry.path.push_back(piece.octant);
                    Node child;
                    child.geometry = piece.geometry;
                    child.material = parent.material;
                    for (const auto& origin : piece.face_origins)
                    {
                        if (origin.kind == ConvexFaceOrigin::Kind::ParentFace)
                            child.surfaces.push_back(parent.surfaces.at(origin.index));
                        else
                        {
                            PolyhedralSurfaceUse use;
                            use.key = {PolyhedralSurfaceKey::Kind::Cut, 0, std::get<ConvexCellAncestry>(key.ancestry),
                                origin.index};
                            // Physical regions are independent of root IDs; include the
                            // cell region in cut surfaces to avoid creator collisions.
                            use.key.region = key.region;
                            use.orientation = (piece.octant & (1u << origin.index)) ? -1 : 1;
                            use.exterior = false;
                            child.surfaces.push_back(std::move(use));
                        }
                    }
                    if (!next->nodes.emplace(child_key, std::move(child)).second)
                        throw std::invalid_argument("Polyhedral child ancestry already exists.");
                    next->leaves.insert(child_key);
                    parent.children.push_back(child_key);
                }
                next->leaves.erase(key);
            }
            build(*next);
        });
    std::string exact_descriptor;
    collectively(
        [&]
        {
            const auto packet = next->complex->encode();
            exact_descriptor.assign(packet.begin(), packet.end());
        });
    same(exact_descriptor);
    ++next->generation;
    ++next->partition_generation;
    next->instance = std::make_shared<const char>(0);
    next->topology = next->instance;
    next->ownership = next->instance;
    next->collective = identity();
    return std::shared_ptr<const PolyhedralAMR>(new PolyhedralAMR(std::move(next)));
}
PolyhedralAMR::AdaptationPlan PolyhedralAMR::plan_adaptation(
    const std::vector<std::size_t>& refine, const std::vector<AdaptiveCellKey>& coarsen) const
{
    AdaptationPlan result;
    result.source_ = observation();
    result.candidate_ = adapt(refine, coarsen);
    result.collective_id_ = identity();
    return result;
}
std::shared_ptr<const PolyhedralAMR> PolyhedralAMR::apply_adaptation(const AdaptationPlan& plan) const
{
    collectively(
        [&]
        {
            if (!geometry_unchanged() || !plan.candidate_ || plan.source_ != observation())
                throw std::invalid_argument("Polyhedral plan has foreign or stale source observation.");
        });
    same(std::to_string(data_->collective) + " " + std::to_string(plan.collective_id_));
    return plan.candidate_;
}
std::vector<AdaptiveAncestryRelation> PolyhedralAMR::overlap_relations(const PolyhedralAMR& target) const
{
    std::vector<ID> targets(target.data_->keys.size());
    std::iota(targets.begin(), targets.end(), ID{});
    return overlap_relations(target, targets);
}
std::vector<AdaptiveAncestryRelation> PolyhedralAMR::overlap_relations(
    const PolyhedralAMR& target, const std::vector<ID>& targets) const
{
    if (data_->root_identity != target.data_->root_identity || !geometry_unchanged() || !target.geometry_unchanged())
        throw std::invalid_argument("Polyhedral overlap snapshots have different roots or stale geometry.");
    std::map<AdaptiveCellKey, std::vector<ID>> descendants;
    for (ID source = 0; source < data_->keys.size(); ++source)
    {
        auto key = data_->keys[source];
        auto& ancestry = std::get<ConvexCellAncestry>(key.ancestry);
        for (;;)
        {
            descendants[key].push_back(source);
            if (ancestry.path.empty())
                break;
            ancestry.path.pop_back();
        }
    }
    std::map<AdaptiveCellKey, ID> active;
    for (ID source = 0; source < data_->keys.size(); ++source)
        active.emplace(data_->keys[source], source);
    std::vector<AdaptiveAncestryRelation> result;
    std::set<ID> seen;
    for (const auto destination : targets)
    {
        if (!seen.insert(destination).second)
            throw std::invalid_argument("Repeated polyhedral overlap target.");
        auto key = target.data_->keys.at(destination);
        const auto original = key;
        auto& ancestry = std::get<ConvexCellAncestry>(key.ancestry);
        bool found = false;
        for (;;)
        {
            if (const auto source = active.find(key); source != active.end())
            {
                result.push_back({key == original ? AdaptiveAncestryRelation::Kind::Unchanged
                                                  : AdaptiveAncestryRelation::Kind::Refinement,
                    source->second, destination});
                found = true;
                break;
            }
            if (ancestry.path.empty())
                break;
            ancestry.path.pop_back();
        }
        if (!found)
        {
            const auto children = descendants.find(original);
            if (children == descendants.end())
                throw std::invalid_argument("Polyhedral snapshots have incompatible active coverage.");
            for (const auto source : children->second)
                result.push_back({AdaptiveAncestryRelation::Kind::Coarsening, source, destination});
        }
    }
    return result;
}
} // namespace SimpleFluid::Meshes::AMR
