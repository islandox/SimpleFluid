/**
 * @file MultiRegionMesh.hh
 * @author islandox
 * @brief Static conforming composition without flattening native interiors.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#pragma once

#include "geometry/mesh/RegionProviders.hh"
#include "io/VTUWriter.hh"
#include <optional>
#include <variant>

namespace SimpleFluid::Meshes
{
/** @brief A standalone native face ordinal qualified by topology-region ordinal. */
struct RegionFace
{
    size_t region = 0;
    uint64_t face = 0;
    auto operator<=>(const RegionFace&) const = default;
};

/**
 * @brief Rectangular subset of a Cartesian exterior patch, parametrized by two logical axes.
 *
 * boundary is 2*normal_axis + upper_side. begin/extent use the increasing
 * native tangential axes. Zero extents select the whole patch. axes permutes
 * that parameterization; reversed reverses each parametrized axis.
 */
struct StructuredPatch
{
    size_t region = 0;
    int boundary = 0;
    std::array<size_t, 2> begin{};
    std::array<size_t, 2> extent{};
    std::array<unsigned, 2> axes{0, 1};
    std::array<bool, 2> reversed{};
};

/** @brief One-to-one signed permutation between equal rectangular parameter domains. */
struct StructuredPatchInterface
{
    StructuredPatch first, second;
    std::array<unsigned, 2> permutation{0, 1};
    std::array<bool, 2> reversed{};
    std::array<size_t, 2> map(std::array<size_t, 2> p) const;
    std::array<size_t, 2> inverse(std::array<size_t, 2> p) const;
    void validate_mapping() const;
};

/** @brief Complete one-to-one pairs between two named native physical boundary patches. */
struct ExplicitConformingInterface
{
    size_t first_region = 0, second_region = 0;
    int first_boundary = 0, second_boundary = 0;
    std::vector<std::pair<uint64_t, uint64_t>> faces;
};

/** @brief Scale-aware geometric matching tolerances, in coordinate units and relative units. */
struct InterfaceTolerance
{
    real_t absolute_length = 1e-12;
    real_t relative = 1e-10;
};

/**
 * @brief One serial FVM mesh with procedural region interiors and canonical seam faces.
 *
 * Each interface keeps its first face, oriented out of that face's native
 * owner, and removes its second face from logical iteration and field storage.
 * Regions retain providers, never initialized MeshHandles. Topology and all
 * geometry are static; a changed native child causes a clear exception through
 * the existing geometry-epoch observation path. Nodes are region-qualified
 * for output, while cells/faces have compact contiguous logical ordinals.
 */
class MultiRegionMesh : public MeshBase<MultiRegionMesh, UnstructuredMeshIndexTypes>
{
public:
    using Base = MeshBase<MultiRegionMesh, UnstructuredMeshIndexTypes>;
    using Indexer = UnstructuredMesh::Indexer;
    using ID = uint64_t;
    using Region = std::variant<CartesianRegion, NativeIsoRegion<OrthogonalCartesian3D>,
        NativeIsoRegion<SemiStructuredXY_Z>, NativeIsoRegion<UnstructuredMesh>>;
    using Interface = std::variant<StructuredPatchInterface, ExplicitConformingInterface>;
    enum class BoundaryNamePolicy { NamespaceRegions, MergeMatchingNames };
    static constexpr ID invalid_cell_id() noexcept { return UnstructuredMesh::invalid_ordinal; }

    MultiRegionMesh(std::vector<Region> regions, std::vector<Interface> interfaces,
        InterfaceTolerance tolerance = {}, BoundaryNamePolicy names = BoundaryNamePolicy::NamespaceRegions);
    MultiRegionMesh(const MultiRegionMesh&) = delete;
    MultiRegionMesh& operator=(const MultiRegionMesh&) = delete;
    MultiRegionMesh(MultiRegionMesh&&) = delete;
    MultiRegionMesh& operator=(MultiRegionMesh&&) = delete;

    const Indexer& indexer() const { return d_indexer; }
    const std::vector<Region>& regions() const { return d_regions; }
    const std::vector<Interface>& interfaces() const { return d_interfaces; }
    const RegionLayout& region_layout(size_t r) const;
    const std::string& region_name(size_t r) const;
    ID canonical_face(RegionFace f) const;
    RegionFace native_face(ID f) const;
    std::pair<size_t, ID> native_cell(ID c) const;
    ID composite_cell(size_t r, ID c) const;
    ID patch_face(const StructuredPatch& p, std::array<size_t, 2> coordinate) const;
    void validate_static() const;
    std::uint64_t geometry_epoch() const { validate_static(); return 0; }
    MeshStorageReport storage_report() const;
    VTUWriter::TopologyHandle vtu_topology() const;
    MeshUtils::CellType cell_type(ID c) const;
    EntityRange<ID> cell_nodes(ID c) const;
    EntityRange<ID> face_nodes(ID f) const;

    /** @brief Dispatch once per region for future batched native kernels. */
    template<class Visitor> void visit_regions(Visitor&& visitor) const
    {
        validate_static();
        for (size_t r = 0; r < d_regions.size(); ++r)
            std::visit([&](const auto& region) { visitor(r, d_cells[r], region); }, d_regions[r]);
    }

private:
    friend Base;
    struct Boundary { size_t region; int native_id; int id; std::string name; };
    // One sorted list per explicit slave patch. Regular patches use no face lists.
    struct ExplicitLookup
    {
        std::vector<std::pair<ID, ID>> first_to_second, second_to_first;
    };
    const Region& region(size_t r) const { return d_regions.at(r); }
    template<class F> decltype(auto) topology(size_t r, F&& f) const
    {
        return std::visit([&](const auto& region) -> decltype(auto) { return f(region.topology()); }, region(r));
    }
    template<class F> decltype(auto) geometry(size_t r, F&& f) const
    {
        return std::visit([&](const auto& region) -> decltype(auto) { return f(region.geometry()); }, region(r));
    }
    void check_cell_id(ID c) const;
    void check_face_id(ID f) const;
    void check_node_id(ID n) const;
    bool is_owned_cell_impl(ID) const { return true; }
    bool is_owned_face_impl(ID) const { return true; }
    real_t cell_volume_impl(ID c) const;
    Vec3 cell_centroid_impl(ID c) const;
    EntityRange<ID> cell_faces_impl(ID c) const;
    ID owner_cell_impl(ID f) const;
    ID neighbor_cell_impl(ID f) const;
    real_t face_area_impl(ID f) const;
    Vec3 face_centroid_impl(ID f) const;
    Vec3 face_normal_impl(ID f) const;
    Vec3 node_coordinates_impl(ID n) const;
    int boundary_id_impl(ID f) const;
    const std::string& boundary_batch_name_impl(int b) const;
    auto boundary_face_batch_impl(int b) const
    {
        return std::views::iota(ID{0}, static_cast<ID>(num_faces()))
            | std::views::filter([this, b](ID f) { return boundary_id_impl(f) == b; });
    }
    std::vector<int> boundary_batch_ids_impl() const;
    int num_boundary_batches_impl() const noexcept;
    void normalize(StructuredPatch& patch) const;
    std::optional<std::array<size_t, 2>> patch_coordinate(const StructuredPatch& p, ID face) const;
    size_t patch_count_before(const StructuredPatch& p, ID face) const;
    size_t removed_before(size_t r, ID face) const;
    std::optional<RegionFace> partner(RegionFace f, bool first) const;
    bool stitched(RegionFace f) const;
    void validate_pair(RegionFace a, RegionFace b) const;
    void validate_regions() const;

    std::vector<Region> d_regions;
    std::vector<Interface> d_interfaces;
    std::vector<ExplicitLookup> d_explicit;
    std::vector<ID> d_cells, d_faces, d_nodes;
    std::vector<Boundary> d_boundaries;
    InterfaceTolerance d_tolerance;
    Indexer d_indexer;
};
static_assert(MeshClass<MultiRegionMesh>);
} // namespace SimpleFluid::Meshes
