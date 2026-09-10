/**
 * @file MultiRegionMesh.hh
 * @author islandox
 * @brief Compact distributed region composition with canonical interfaces and controlled axial motion.
 * @version 0.1
 * @date 2026-09-09
 * @copyright Copyright (c) 2026
 */
#pragma once

#include "geometry/mesh/RegionProviders.hh"
#include "geometry/mesh/ConservativeInterface.hh"
#include "geometry/mesh/ExtrudedRegionProviders.hh"
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
 * @brief Rectangular subset of a structured exterior patch, parametrized by two logical axes.
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
    /** Explicit periodic identification: first-face point + translation = second-face point. */
    std::optional<MeshUtils::Vec3> periodic_translation;
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
 * @brief One FVM domain with procedural interiors, collective compact descriptions and canonical seam faces.
 *
 * Each interface keeps its first face, oriented out of that face's native
 * owner, and removes its second face from logical iteration and field storage.
 * Regions retain providers, never initialized MeshHandles. Child topology and
 * geometry stay frozen; the existing planar ALE controller can change a common
 * composite axial affine map. Changed children fail through the geometry-epoch
 * observation path. Construction is collective on the default communicator;
 * MeshHandle supplies local ownership/maps. Explicit global regions are serial.
 * Nodes are region-qualified, while cells/faces have contiguous logical ordinals.
 */
class MultiRegionMesh : public GeometryExecutionGuard, public MeshBase<MultiRegionMesh, UnstructuredMeshIndexTypes>
{
public:
    using Base = MeshBase<MultiRegionMesh, UnstructuredMeshIndexTypes>;
    using Indexer = UnstructuredMesh::Indexer;
    using ID = uint64_t;
    using Region = std::variant<CartesianRegion, NativeIsoRegion<OrthogonalCartesian3D>,
        NativeIsoRegion<SemiStructuredXY_Z>, NativeIsoRegion<UnstructuredMesh>, NativeIsoRegion<OrthogonalCylindrial3D>, ExtrudedRegion>;
    using Interface = std::variant<StructuredPatchInterface, ExplicitConformingInterface, NonconformingInterface>;
    enum class BoundaryNamePolicy { NamespaceRegions, MergeMatchingNames };
    static constexpr ID invalid_cell_id() noexcept { return UnstructuredMesh::invalid_ordinal; }

    /**
     * @brief Operation-local canonical incidence and transformed geometry snapshot.
     *
     * The centroid members retain physical coordinates. Displacements include
     * the incident periodic image; kernels must use those rather than subtract
     * the centroids themselves. No provider pointers or connectivity arrays
     * are retained.
     */
    struct ResolvedFace
    {
        ID face = 0, owner = invalid_cell_id(), neighbor = invalid_cell_id();
        Vec3 area_vector{}, centroid{}, owner_centroid{}, neighbor_centroid{};
        Vec3 owner_to_neighbor{}, neighbor_to_owner{}, owner_to_face{}, neighbor_to_face{};
        bool interior() const noexcept { return neighbor != invalid_cell_id(); }
        real_t area() const { return area_vector.norm(); }
        Vec3 normal() const { return area_vector / area(); }
        ID opposite_cell(ID cell) const
        {
            if (cell == owner) return neighbor;
            if (interior() && cell == neighbor) return owner;
            throw std::invalid_argument("Cell is not incident to resolved composite face.");
        }
        Vec3 face_normal_outward(ID cell) const
        {
            const auto other = opposite_cell(cell);
            (void)other;
            return cell == owner ? normal() : normal() * -1.0;
        }
        // Keep the checked MeshBase normal-times-area arithmetic order.
        Vec3 face_area_vector_outward(ID cell) const { return face_normal_outward(cell) * area(); }
        Vec3 face_center_vector(ID cell) const
        {
            const auto other = opposite_cell(cell);
            (void)other;
            return cell == owner ? owner_to_face : neighbor_to_face;
        }
        Vec3 cell_center_vector(ID cell) const
        {
            if (opposite_cell(cell) == invalid_cell_id())
                throw std::invalid_argument("Exterior face has no adjacent cell.");
            return cell == owner ? owner_to_neighbor : neighbor_to_owner;
        }
    };

    class ExecutionView;
    /** @brief Typed region geometry borrowed for the enclosing execution callback. */
    template<class R> class RegionGeometryView
    {
    public:
        RegionGeometryView(const RegionGeometryView&) = delete;
        RegionGeometryView& operator=(const RegionGeometryView&) = delete;
        const RegionLayout& layout() const noexcept { return d_region.layout(); }
        real_t cell_volume(ID cell) const
        {
            check_cell(cell);
            return d_mesh.axial_scale() * d_region.geometry().cell_volume(cell);
        }
        Vec3 cell_centroid(ID cell) const
        {
            check_cell(cell);
            return d_mesh.transformed_point(d_region.geometry().cell_centroid(cell));
        }
        /** @brief Traverse native incidence once, preserving canonical logical face order. */
        template<class Visitor> void visit_cell_faces(ID cell, Visitor&& visitor) const
        {
            check_cell(cell);
            const auto visit = [&](ID face)
            {
                const auto adjacent = d_region.topology().neighbor_cell(face);
                if (adjacent != invalid_cell_id())
                {
                    visitor(d_mesh.resolve_native_face(d_index, d_region, face, adjacent));
                    return;
                }
                const RegionFace native{d_index, face};
                if (const auto refined = d_mesh.refinement(native))
                {
                    // The retained fine faces are already known native identities.
                    for (const auto fine : refined->faces)
                        visitor(d_mesh.resolve_retained_face({refined->region, fine}));
                }
                else if (const auto retained = d_mesh.partner(native, false))
                    visitor(d_mesh.resolve_retained_face(*retained));
                else
                    visitor(d_mesh.resolve_native_face(d_index, d_region, face, adjacent));
            };
            const auto& topology = d_region.topology();
            if constexpr (requires { topology.native(); })
            {
                const auto& native = topology.native();
                // Do not use the provider's ordinal range here: its element
                // accessor would recreate the native range for every entry.
                for (const auto face : native.cell_faces(native.cell_id(cell)))
                    visit(native.face_local_id(face));
            }
            else if constexpr (requires { topology.native_topology().cell_faces(topology.indexer().cell_id(cell)); })
            {
                for (const auto face : topology.native_topology().cell_faces(topology.indexer().cell_id(cell)))
                    visit(topology.indexer().face_ordinal(face));
            }
            else
                for (const auto face : topology.cell_faces(cell)) visit(face);
        }
    private:
        friend class ExecutionView;
        RegionGeometryView(const MultiRegionMesh& mesh, size_t index, const R& region)
            : d_mesh(mesh), d_index(index), d_region(region) {}
        void check_cell(ID cell) const
        {
            if (cell >= layout().cells) throw std::out_of_range("Native cell out of bounds.");
        }
        const MultiRegionMesh& d_mesh;
        size_t d_index;
        const R& d_region;
    };

    /**
     * @brief Validate once and pin constituent geometry for an existing kernel.
     *
     * Queries through this mesh (including MeshHandle aliases) keep bounds
     * checks but avoid whole-composite validation on this thread while the
     * view lives. Public queries outside a view remain fully checked. Views
     * borrow the mesh, are thread-affine, and must be destroyed in scope order.
     * Nested operations reuse an enclosing view of the same mesh. Geometry
     * motion and constituent/provider replacement are rejected before mutation.
     */
    class ExecutionView
    {
    public:
        explicit ExecutionView(const MultiRegionMesh* mesh);
        ExecutionView(const ExecutionView&) = delete;
        ExecutionView& operator=(const ExecutionView&) = delete;
        ExecutionView(ExecutionView&&) = delete;
        ExecutionView& operator=(ExecutionView&&) = delete;
        ~ExecutionView();
        /** @brief Visit raw constituent coordinates; composite affine motion is applied by canonical mesh queries. */
        template<class Visitor> void visit_regions(Visitor&& visitor) const
        {
            if (!d_mesh) throw std::logic_error("Execution view has no composite regions.");
            for (size_t r = 0; r < d_mesh->d_regions.size(); ++r)
                std::visit([&](const auto& region) { visitor(r, d_mesh->d_cells[r], region); }, d_mesh->d_regions[r]);
        }
        /**
         * @brief Dispatch once per region with composite motion and periodic images.
         *
         * Geometry views borrow this execution lease, are thread-affine, and
         * must only be used within the callback. Resolved faces are value
         * snapshots and do not extend the lease or follow later mesh motion.
         */
        template<class Visitor> void visit_region_geometry(Visitor&& visitor) const
        {
            if (!d_mesh) throw std::logic_error("Execution view has no composite regions.");
            for (size_t r = 0; r < d_mesh->d_regions.size(); ++r)
                std::visit([&](const auto& region)
                {
                    const RegionGeometryView<std::remove_cvref_t<decltype(region)>> geometry(*d_mesh, r, region);
                    visitor(r, d_mesh->d_cells[r], geometry);
                }, d_mesh->d_regions[r]);
        }
        ResolvedFace resolve_face(ID face) const;
    private:
        friend MultiRegionMesh;
        static thread_local const ExecutionView* d_current;
        const MultiRegionMesh* d_mesh;
        const ExecutionView* d_previous;
        std::vector<GeometryExecutionGuard::ReadLease> d_leases;
    };
    [[nodiscard]] ExecutionView acquire_execution_view() const { return ExecutionView(this); }

    MultiRegionMesh(std::vector<Region> regions, std::vector<Interface> interfaces,
        InterfaceTolerance tolerance = {}, BoundaryNamePolicy names = BoundaryNamePolicy::NamespaceRegions);
    MultiRegionMesh(const MultiRegionMesh&) = delete;
    MultiRegionMesh& operator=(const MultiRegionMesh&) = delete;
    MultiRegionMesh(MultiRegionMesh&&) = delete;
    MultiRegionMesh& operator=(MultiRegionMesh&&) = delete;

    const Indexer& indexer() const { return d_indexer; }
    const std::vector<Region>& regions() const { return d_regions; }
    const std::vector<Interface>& interfaces() const { return d_interfaces; }
    /** @brief Descriptor counts for storage/scaling diagnostics; a self seam has two sides. */
    size_t region_interface_side_count(size_t r) const
    {
        const auto& offsets = d_region_interfaces.at(r).offsets;
        return offsets.back() - offsets.front();
    }
    ID removed_native_face_count(size_t r) const { return d_region_interfaces.at(r).removed_faces; }
    const RegionLayout& region_layout(size_t r) const;
    const std::string& region_name(size_t r) const;
    ID canonical_face(RegionFace f) const;
    EntityRange<ID> logical_faces(RegionFace native) const;
    EntityRange<FaceFluxWeight> face_flux_weights(RegionFace native) const;
    /** @brief Sum canonical integrated fluxes in a native face's outward orientation. */
    template<class Values> real_t restrict_face_flux(RegionFace native, Values&& value) const
    {
        real_t sum=0;
        for(const auto weight:face_flux_weights(native)) sum+=(weight.coefficient>0?1.0:-1.0)*value(weight.face);
        return sum;
    }
    RegionFace native_face(ID f) const;
    std::pair<size_t, ID> native_cell(ID c) const;
    ID composite_cell(size_t r, ID c) const;
    ID patch_face(const StructuredPatch& p, std::array<size_t, 2> coordinate) const;
    void validate_static() const;
    std::uint64_t geometry_epoch() const { validate_query(); return d_geometry_state.epoch; }
    const ArrReal& axial_edges() const noexcept { return d_axial_edges; }
    bool supports_axial_motion() const noexcept;
    Vec3 cell_center_vector(ID face,ID cell) const;
    Vec3 face_center_vector(ID face,ID cell) const;
    real_t cell_to_face_distance(ID face,ID cell) const { return face_center_vector(face,cell).norm(); }
    real_t face_cell_center_distance(ID face) const
    { return is_exterior_face(face)?0.0:cell_center_vector(face,owner_cell(face)).norm(); }
    MeshStorageReport storage_report() const;
    VTUWriter::TopologyHandle vtu_topology() const;
    VTUWriter::TopologyHandle vtu_topology(const EntityRange<ID>& owned_cells) const;
    /** @brief Exact compact geometry/topology configuration, excluding process-local pointer identities. */
    std::string configuration_signature() const;
    MeshUtils::CellType cell_type(ID c) const;
    EntityRange<ID> cell_nodes(ID c) const;
    EntityRange<ID> face_nodes(ID f) const;

    /** @brief Dispatch once per region under a validated geometry read lease. */
    template<class Visitor> void visit_regions(Visitor&& visitor) const
    {
        auto execution = acquire_execution_view();
        execution.visit_regions(std::forward<Visitor>(visitor));
    }

private:
    friend Base;
    friend class PlanarALEGeometryAccess;
    void replace_axial_edges_fixed_topology(ArrReal edges);
    Vec3 transformed_point(Vec3 point) const;
    Vec3 transformed_area_vector(Vec3 vector) const;
    real_t axial_scale() const noexcept;
    struct Boundary { size_t region; int native_id; int id; std::string name; };
    // One sorted list per explicit slave patch. Regular patches use no face lists.
    struct ExplicitLookup
    {
        std::vector<std::pair<ID, ID>> first_to_second, second_to_first;
    };
    // Two side descriptors per interface. Structured sides are grouped by
    // native boundary 0..5; only irregular correspondence uses bucket 6.
    struct InterfaceSide { size_t interface; bool first; };
    struct RegionInterfaceDirectory
    {
        std::array<size_t, 8> offsets{};
        ID removed_faces = 0;
        size_t boundary_begin = 0, boundary_end = 0;
    };
    void initialize_interface_directory();
    std::span<const InterfaceSide> interface_sides(size_t region, size_t bucket) const;
    std::span<const InterfaceSide> structured_interface_sides(RegionFace face) const;
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
    bool executing_on_this_thread() const noexcept;
    void validate_query() const;
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
    std::optional<size_t> structured_boundary_size(size_t region, int boundary) const;
    std::optional<std::array<size_t, 2>> patch_coordinate(const StructuredPatch& p, ID face) const;
    size_t patch_count_before(const StructuredPatch& p, ID face) const;
    size_t removed_before(size_t r, ID face) const;
    std::optional<RegionFace> partner(RegionFace f, bool first) const;
    struct RefinedFaceView { size_t region; std::span<const ID> faces; };
    std::optional<RefinedFaceView> refinement(RegionFace f) const;
    ID encode_native_face(RegionFace f) const;
    RegionFace decode_native_face(ID key) const;
    InterfaceFacePolygon interface_polygon(RegionFace f) const;
    size_t interface_uses(RegionFace f) const;
    bool stitched(RegionFace f) const;
    void validate_pair(RegionFace a, RegionFace b, std::optional<Vec3> translation = {}) const;
    Vec3 periodic_translation(ID face) const;
    Vec3 periodic_translation(RegionFace native) const;
    ResolvedFace resolve_retained_face(RegionFace native) const;
    template<class R> ResolvedFace resolve_native_face(size_t r, const R& region, ID face, ID adjacent) const
    {
        ResolvedFace resolved;
        resolved.face = d_faces[r] + face - removed_before(r, face);
        const auto owner = region.topology().owner_cell(face);
        resolved.owner = d_cells[r] + owner;
        const auto& geometry = region.geometry();
        resolved.area_vector = transformed_area_vector(geometry.face_area_vector(face));
        resolved.centroid = transformed_point(geometry.face_centroid(face));
        resolved.owner_centroid = transformed_point(geometry.cell_centroid(owner));
        Vec3 shift{};
        if (adjacent != invalid_cell_id())
        {
            resolved.neighbor = d_cells[r] + adjacent;
            resolved.neighbor_centroid = transformed_point(geometry.cell_centroid(adjacent));
        }
        else if (const auto other = partner({r, face}, true))
        {
            std::visit([&](const auto& neighbor_region)
            {
                const auto cell = neighbor_region.topology().owner_cell(other->face);
                resolved.neighbor = d_cells[other->region] + cell;
                resolved.neighbor_centroid = transformed_point(neighbor_region.geometry().cell_centroid(cell));
            }, d_regions[other->region]);
            shift = periodic_translation(RegionFace{r, face});
        }
        resolved.owner_to_face = resolved.centroid - resolved.owner_centroid;
        if (resolved.interior())
        {
            resolved.owner_to_neighbor = resolved.neighbor_centroid - resolved.owner_centroid + shift * -1.0;
            resolved.neighbor_to_owner = resolved.owner_centroid - resolved.neighbor_centroid + shift * 1.0;
            resolved.neighbor_to_face = resolved.centroid + shift - resolved.neighbor_centroid;
        }
        return resolved;
    }
    void validate_regions() const;
    void initialize_regions(InterfaceTolerance tolerance, BoundaryNamePolicy names);

    std::vector<Region> d_regions;
    std::vector<Interface> d_interfaces;
    std::vector<ExplicitLookup> d_explicit;
    std::vector<RegionInterfaceDirectory> d_region_interfaces;
    std::vector<InterfaceSide> d_interface_sides;
    std::vector<ID> d_cells, d_faces, d_nodes, d_native_faces;
    std::vector<Boundary> d_boundaries;
    InterfaceTolerance d_tolerance;
    Indexer d_indexer;
    ArrReal d_reference_axial_edges, d_axial_edges;
    GeometryEpochState d_geometry_state;
};
static_assert(MeshClass<MultiRegionMesh>);
} // namespace SimpleFluid::Meshes
