/** @file MeshTraverse.hh
 * @brief Reusable, ordered cell and face selections with typed numerical dispatch.
 */
#pragma once

#include "SimpleFluidExport.hh"
#include "geometry/GeometryExecutionGuard.hh"
#include "geometry/details/MeshTraverseAccess.hh"

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid
{

enum class MeshTraverseSelection
{
    All,
    Interior,
    Boundary
};
enum class MeshTraverseOwnership
{
    Owned,
    Local
};
enum class MeshTraverseEntity
{
    Cell,
    Face
};
/** Incidence storage is opt-in; ordinary traverses retain only ranges. */
enum class MeshTraverseEntities
{
    Cells,
    Faces,
    All,
    AllWithCellFaces
};

template<class Mesh> class MeshTraversePlan;
template<class Mesh> class MeshTraverseExecution;
template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection = MeshTraverseSelection::All,
    MeshTraverseOwnership Ownership = MeshTraverseOwnership::Owned, bool Incidence = false>
class MeshBatchProxy;

template<class Mesh, MeshTraverseSelection Selection = MeshTraverseSelection::All,
    MeshTraverseOwnership Ownership = MeshTraverseOwnership::Owned>
using CellBatchProxy = MeshBatchProxy<Mesh, MeshTraverseEntity::Cell, Selection, Ownership>;
template<class Mesh, MeshTraverseSelection Selection = MeshTraverseSelection::All,
    MeshTraverseOwnership Ownership = MeshTraverseOwnership::Owned>
using FaceBatchProxy = MeshBatchProxy<Mesh, MeshTraverseEntity::Face, Selection, Ownership>;
template<class Mesh, MeshTraverseSelection Selection = MeshTraverseSelection::All>
using CellFaceBatchProxy =
    MeshBatchProxy<Mesh, MeshTraverseEntity::Face, Selection, MeshTraverseOwnership::Local, true>;

namespace detail
{
/** Half-open ordinal range. Classification is constant over the entire run. */
struct MeshTraverseRun
{
    size_t begin{}, end{};
    bool interior = false;
    int batch = -1;
};

struct MeshTraverseSchedule
{
    std::vector<MeshTraverseRun> ordered, interior, boundary;
    std::map<int, std::vector<MeshTraverseRun>> batches;

    static void append(std::vector<MeshTraverseRun>& runs, size_t ordinal, bool inner, int batch);

    void append(size_t ordinal, bool inner, int batch, bool face);

    size_t storage_bytes() const noexcept
    {
        size_t bytes = (ordered.capacity() + interior.capacity() + boundary.capacity()) * sizeof(MeshTraverseRun);
        for (const auto& [batch, runs] : batches)
            bytes += sizeof(batch) + sizeof(runs) + runs.capacity() * sizeof(MeshTraverseRun);
        return bytes;
    }
};

template<class Local> struct MeshTraverseIncidence
{
    Local face{};
    int orientation = 1;
};

/** Native guarded meshes also retain their read lease for the execution scope. */
template<class Mesh> auto acquire_traverse_geometry(const Mesh& mesh)
{
    if constexpr (requires { mesh.acquire_execution_view(); })
        return mesh.acquire_execution_view();
    else if constexpr (requires { mesh.acquire_geometry_read(); })
        return mesh.acquire_geometry_read();
    else
        return acquire_mesh_execution(mesh);
}

/** Keep a replaced handle's providers alive until its borrowed leases release. */
template<class Mesh> auto retain_traverse_backend(const Mesh& mesh)
{
    if constexpr (requires { mesh.variant(); })
        return mesh.variant();
    else if constexpr (requires { mesh.parent_mesh(); })
        return retain_traverse_backend(mesh.parent_mesh());
    else
    {
        struct BorrowedMesh
        {
        };
        return BorrowedMesh{};
    }
}

/** Cheap binding checks. Same-count in-place topology edits require rebuilding. */
template<class Mesh> auto mesh_traverse_binding(const Mesh& mesh);

template<class Mesh> auto mesh_traverse_counts(const Mesh& mesh);
template<bool Cell> struct TraverseCellOrigin
{
    explicit TraverseCellOrigin(const void* = nullptr) {}
};
template<> struct TraverseCellOrigin<true>
{
    const void* plan;
    explicit TraverseCellOrigin(const void* source = nullptr) : plan(source) {}
};
} // namespace detail

/**
 * @brief Borrowed entity with a compile-time physical category.
 *
 * IDs are field-local ordinals, including for native structured providers.
 * A topology-interior face can lack a locally visible adjacent cell; neighbor()
 * and owner() return the mesh's invalid sentinel in that case. Kernels needing
 * both values must call require_local_adjacency() before acquiring field views.
 */
template<class Access, MeshTraverseEntity Entity, bool Interior, bool Incidence = false> class MeshTraverseRef
{
public:
    using local_ordinal_type = typename Access::local_ordinal_type;
    static constexpr bool is_interior = Interior;
    static constexpr MeshTraverseEntity entity = Entity;

    MeshTraverseRef(const Access& access, local_ordinal_type id, int batch = -1, local_ordinal_type cell = {},
        int orientation = 1, const void* origin_plan = nullptr);

    local_ordinal_type id() const noexcept { return d_id; }
    auto centroid() const
    {
        if constexpr (Entity == MeshTraverseEntity::Cell)
            return d_access->cell_centroid(d_id);
        else
            return d_access->face_centroid(d_id);
    }
    auto volume() const
        requires(Entity == MeshTraverseEntity::Cell)
    {
        return d_access->cell_volume(d_id);
    }
    auto area() const
        requires(Entity == MeshTraverseEntity::Face)
    {
        return d_access->face_area(d_id);
    }
    auto normal() const
        requires(Entity == MeshTraverseEntity::Face)
    {
        return d_access->face_normal(d_id);
    }
    auto owner() const
        requires(Entity == MeshTraverseEntity::Face)
    {
        return d_access->owner_cell(d_id);
    }
    auto neighbor() const
        requires(Entity == MeshTraverseEntity::Face && Interior)
    {
        return d_access->neighbor_cell(d_id);
    }
    int boundary_id() const
        requires(Entity == MeshTraverseEntity::Face && !Interior)
    {
        return d_batch;
    }
    local_ordinal_type cell_id() const
        requires Incidence
    {
        return d_cell;
    }
    int orientation() const
        requires Incidence
    {
        return d_orientation;
    }
    auto outward_normal() const
        requires Incidence
    {
        return normal() * d_orientation;
    }
    const Access& geometry() const noexcept { return *d_access; }
    const void* origin_plan() const noexcept
        requires(Entity == MeshTraverseEntity::Cell)
    {
        return d_origin.plan;
    }

private:
    const Access* d_access;
    local_ordinal_type d_id;
    int d_batch;
    local_ordinal_type d_cell;
    int d_orientation;
    [[no_unique_address]] detail::TraverseCellOrigin<Entity == MeshTraverseEntity::Cell> d_origin;
};

/** Layered incidence with resolved topology IDs and live provider geometry. */
template<class Access, bool Interior>
class LayeredIncidenceRef : public MeshTraverseRef<Access, MeshTraverseEntity::Face, Interior, true>
{
    using Base = MeshTraverseRef<Access, MeshTraverseEntity::Face, Interior, true>;

public:
    using local_ordinal_type = typename Access::local_ordinal_type;
    using CellID = detail::LayeredTraverseLayout::Indexer::CellID;
    LayeredIncidenceRef(
        const Access& access, const detail::LayeredTraverseLayout::ResolvedIncidence& face, local_ordinal_type cell);
    local_ordinal_type owner() const noexcept { return d_owner; }
    local_ordinal_type neighbor() const noexcept
        requires Interior
    {
        return d_neighbor;
    }
    auto cell_center_vector() const
        requires Interior
    {
        if (d_owner == Access::invalid_local_id() || d_neighbor == Access::invalid_local_id())
            throw std::invalid_argument("Exterior face does not have an opposite cell.");
        return this->geometry().resolved_cell_center_vector(d_cell, d_other);
    }

private:
    static local_ordinal_type local(size_t ordinal)
    {
        return ordinal == detail::LayeredTraverseLayout::invalid_ordinal ? Access::invalid_local_id()
                                                                         : static_cast<local_ordinal_type>(ordinal);
    }
    local_ordinal_type d_owner, d_neighbor;
    CellID d_cell, d_other;
};

/**
 * @brief Homogeneous borrowed ranges. No selection predicate runs per entity.
 *
 * The batch and its entity references expire with the visitor callback. The
 * visitor may retain numerical results, but must not retain these views.
 */
template<class Access, MeshTraverseEntity Entity, bool Interior, bool Incidence = false> class MeshTraverseBatch
{
public:
    using local_ordinal_type = typename Access::local_ordinal_type;
    using reference = MeshTraverseRef<Access, Entity, Interior, Incidence>;
    static constexpr bool is_interior = Interior;
    static constexpr MeshTraverseEntity entity = Entity;

    MeshTraverseBatch(const Access& access, std::span<const detail::MeshTraverseRun> runs,
        std::span<const detail::MeshTraverseIncidence<local_ordinal_type>> incidence = {}, local_ordinal_type cell = {},
        const void* origin_plan = nullptr);

    size_t size() const noexcept
    {
        size_t count = 0;
        for (const auto& run : d_runs)
            count += run.end - run.begin;
        return count;
    }
    bool empty() const noexcept { return d_runs.empty(); }

    template<class Operation> void apply(Operation&& operation) const;

private:
    const Access* d_access;
    std::span<const detail::MeshTraverseRun> d_runs;
    std::span<const detail::MeshTraverseIncidence<local_ordinal_type>> d_incidence;
    local_ordinal_type d_cell;
    [[no_unique_address]] detail::TraverseCellOrigin<Entity == MeshTraverseEntity::Cell> d_origin;
};

/**
 * @brief Immutable reusable selection metadata borrowing a fixed-topology mesh.
 *
 * Boundary means physical exterior, including untagged exterior faces. Periodic
 * and stitched faces are interior even if they retain a batch identifier. Cell
 * classification uses full geometric incidence, independently of halo depth.
 *
 * Geometry-only motion is allowed between execution scopes. Rebuild this plan
 * after topology/periodic-pairing, partition, or ordering edits; geometry_epoch
 * is not a topology revision. Count, backend and map replacement are checked
 * at acquire(), but same-count in-place topology mutation is the caller's
 * responsibility. The mesh must outlive the plan; the plan must outlive views.
 */
template<class Mesh> class SIMPLEFLUID_PUBLIC_TYPE MeshTraversePlan
{
public:
    using local_ordinal_type = typename Mesh::local_ordinal_type;

    explicit MeshTraversePlan(const Mesh& mesh, MeshTraverseEntities entities = MeshTraverseEntities::All);

    MeshTraversePlan(const MeshTraversePlan&) = delete;
    MeshTraversePlan& operator=(const MeshTraversePlan&) = delete;
    MeshTraversePlan(MeshTraversePlan&&) = delete;
    MeshTraversePlan& operator=(MeshTraversePlan&&) = delete;
    MeshTraversePlan(Mesh&&, MeshTraverseEntities = MeshTraverseEntities::All) = delete;
    MeshTraversePlan(const Mesh&&, MeshTraverseEntities = MeshTraverseEntities::All) = delete;

    [[nodiscard]] MeshTraverseExecution<Mesh> acquire() const&;
    MeshTraverseExecution<Mesh> acquire() const&& = delete;
    const Mesh& mesh() const noexcept { return *d_mesh; }
    MeshTraverseEntities entities() const noexcept { return d_entities; }
    bool uses_structured_ranges() const noexcept { return d_structured.has_value(); }

    bool uses_layered_ranges() const noexcept { return bool(d_layered); }
    const std::shared_ptr<const detail::LayeredTraverseLayout>& layered_layout() const noexcept { return d_layered; }

    /** Explicitly reject further use before editing topology in place. */
    void invalidate() noexcept;
    void require_valid() const
    {
        if (!d_valid || d_counts != detail::mesh_traverse_counts(*d_mesh) ||
            d_binding != detail::mesh_traverse_binding(*d_mesh))
            throw std::logic_error("Mesh traverse plan is stale; rebuild it after topology, map, or mesh replacement.");
    }

    /** Selection storage estimate excluding allocator bookkeeping. */
    size_t storage_bytes() const noexcept;

private:
    friend class MeshTraverseExecution<Mesh>;
    template<class, MeshTraverseEntity, MeshTraverseSelection, MeshTraverseOwnership, bool> friend class MeshBatchProxy;

    const Mesh* d_mesh;
    MeshTraverseEntities d_entities;
    bool d_valid = true;
    std::array<size_t, 4> d_counts{};
    std::array<const void*, 5> d_binding{};
    std::optional<detail::StructuredTraverseLayout> d_structured;
    std::shared_ptr<const detail::LayeredTraverseLayout> d_layered;
    std::array<detail::MeshTraverseSchedule, 2> d_cells, d_faces;
    std::vector<detail::MeshTraverseSchedule> d_cell_faces;
    std::vector<detail::MeshTraverseIncidence<local_ordinal_type>> d_incidence;
};

/** Direct six-face incidences from a same-plan structured interior cell proof. */
template<class Mesh, class Access, MeshTraverseSelection Selection = MeshTraverseSelection::All>
class StructuredInteriorCellFaces
{
public:
    using local_ordinal_type = typename Mesh::local_ordinal_type;
    using reference = MeshTraverseRef<Access, MeshTraverseEntity::Face, true, true>;
    static constexpr bool is_structured_interior_range = true;
    static constexpr bool is_interior = true;
    static constexpr MeshTraverseEntity entity = MeshTraverseEntity::Face;
    auto all() const { return select<MeshTraverseSelection::All>(); }
    auto interior() const { return select<MeshTraverseSelection::Interior>(); }
    auto boundary() const { return select<MeshTraverseSelection::Boundary>(); }
    auto boundary(int) const { return boundary(); }
    auto independent() const { return *this; }
    size_t size() const
    {
        d_plan->require_valid();
        if constexpr (Selection == MeshTraverseSelection::Boundary)
            return 0;
        else
            return 6;
    }
    bool empty() const { return size() == 0; }
    void require_local_adjacency() const
    {
        // Canonical structured plans cover the full provider, so the typed
        // interior-cell proof also guarantees both adjacent local ordinals.
        validate(*d_access);
    }
    template<class Operation> void apply(Operation&& operation) const;
    template<class Operation> void apply(const Access& access, Operation&& operation) const;
    template<class Visitor> void visit_batches(Visitor&& visitor) const;
    template<class Visitor> void visit_batches(const Access& access, Visitor&& visitor) const;

private:
    friend class MeshTraverseExecution<Mesh>;
    template<class, class, MeshTraverseSelection> friend class StructuredInteriorCellFaces;
    StructuredInteriorCellFaces(const MeshTraversePlan<Mesh>& plan, const Access& access,
        local_ordinal_type cell, const detail::StructuredTraverseLayout& layout);
    StructuredInteriorCellFaces(const MeshTraversePlan<Mesh>* plan, const Access* access, local_ordinal_type cell,
        const detail::StructuredTraverseLayout* layout);
    template<MeshTraverseSelection Next> auto select() const
    {
        return StructuredInteriorCellFaces<Mesh, Access, Next>(d_plan, d_access, d_cell, d_layout);
    }
    void validate(const Access& access) const
    {
        d_plan->require_valid();
        if (&access.mesh() != &d_plan->mesh())
            throw std::invalid_argument("Mesh traverse access belongs to another mesh.");
    }
    const MeshTraversePlan<Mesh>* d_plan;
    const Access* d_access;
    const detail::StructuredTraverseLayout* d_layout;
    local_ordinal_type d_cell;
};

/** Direct interior incidences from a same-plan typed cell proof. */
template<class Mesh, class Access, MeshTraverseSelection Selection = MeshTraverseSelection::All>
class SIMPLEFLUID_LOCAL LayeredInteriorCellFaces
{
public:
    using local_ordinal_type = typename Mesh::local_ordinal_type;
    using reference = LayeredIncidenceRef<Access, true>;
    static constexpr bool is_layered_interior_range = true;
    static constexpr bool is_interior = true;
    static constexpr MeshTraverseEntity entity = MeshTraverseEntity::Face;
    auto all() const { return select<MeshTraverseSelection::All>(); }
    auto interior() const { return select<MeshTraverseSelection::Interior>(); }
    auto boundary() const { return select<MeshTraverseSelection::Boundary>(); }
    auto boundary(int) const { return boundary(); }
    auto independent() const { return *this; }
    size_t size() const
    {
        d_plan->require_valid();
        if constexpr (Selection == MeshTraverseSelection::Boundary)
            return 0;
        else
            return d_faces.size();
    }
    bool empty() const { return size() == 0; }
    void require_local_adjacency() const
    {
        // This proof is available only for a canonical, fully local layout.
        // Physical interior classification therefore proves both adjacent IDs.
        validate(*d_access);
    }
    template<class Operation> void apply(Operation&& operation) const;
    template<class Operation> [[gnu::always_inline]] void apply(const Access& access, Operation&& operation) const;
    template<class Visitor> void visit_batches(Visitor&& visitor) const;
    template<class Visitor> void visit_batches(const Access& access, Visitor&& visitor) const;

private:
    friend class MeshTraverseExecution<Mesh>;
    template<class, class, MeshTraverseSelection> friend class LayeredInteriorCellFaces;
    LayeredInteriorCellFaces(const MeshTraversePlan<Mesh>& plan, const Access& access, local_ordinal_type cell);
    LayeredInteriorCellFaces(const MeshTraversePlan<Mesh>* plan, const Access* access, local_ordinal_type cell,
        detail::LayeredTraverseLayout::Incidences faces);
    template<MeshTraverseSelection Next> auto select() const
    {
        return LayeredInteriorCellFaces<Mesh, Access, Next>(d_plan, d_access, d_cell, d_faces);
    }
    void validate(const Access& access) const
    {
        d_plan->require_valid();
        if (&access.mesh() != &d_plan->mesh())
            throw std::invalid_argument("Mesh traverse access belongs to another mesh.");
    }
    const MeshTraversePlan<Mesh>* d_plan;
    const Access* d_access;
    local_ordinal_type d_cell;
    detail::LayeredTraverseLayout::Incidences d_faces;
};

/** A synchronous, nonmovable geometry lease; no field synchronization is implicit. */
template<class Mesh> class MeshTraverseExecution
{
public:
    MeshTraverseExecution(const MeshTraverseExecution&) = delete;
    MeshTraverseExecution& operator=(const MeshTraverseExecution&) = delete;
    MeshTraverseExecution(MeshTraverseExecution&&) = delete;
    MeshTraverseExecution& operator=(MeshTraverseExecution&&) = delete;

    CellBatchProxy<Mesh> cells() const&
    {
        d_plan->require_valid();
        if (d_plan->d_entities == MeshTraverseEntities::Faces)
            throw std::logic_error("This mesh traverse plan contains only faces.");
        return CellBatchProxy<Mesh>(*this);
    }
    FaceBatchProxy<Mesh> faces() const&
    {
        d_plan->require_valid();
        if (d_plan->d_entities == MeshTraverseEntities::Cells)
            throw std::logic_error("This mesh traverse plan contains only cells.");
        return FaceBatchProxy<Mesh>(*this);
    }
    CellFaceBatchProxy<Mesh> cell_faces(typename Mesh::local_ordinal_type cell) const&
    {
        d_plan->require_valid();
        require_cell_faces_id(cell);
        return CellFaceBatchProxy<Mesh>(*this, cell);
    }
    template<class Access, bool Interior>
        requires std::is_same_v<typename Access::mesh_type, Mesh>
    auto cell_faces(const MeshTraverseRef<Access, MeshTraverseEntity::Cell, Interior>& cell) const&;
    template<class Access, bool Interior>
    auto cell_faces(const MeshTraverseRef<Access, MeshTraverseEntity::Cell, Interior>&) const&& = delete;
    CellBatchProxy<Mesh> cells() const&& = delete;
    FaceBatchProxy<Mesh> faces() const&& = delete;
    CellFaceBatchProxy<Mesh> cell_faces(typename Mesh::local_ordinal_type) const&& = delete;

private:
    void require_cell_faces_id(typename Mesh::local_ordinal_type cell) const
    {
        if (d_plan->d_entities != MeshTraverseEntities::AllWithCellFaces)
            throw std::logic_error("Cell-face traverses require an AllWithCellFaces plan.");
        if (static_cast<size_t>(cell) >= d_plan->d_counts[0])
            throw std::out_of_range("Mesh traverse cell ID is out of range.");
    }
    friend class MeshTraversePlan<Mesh>;
    template<class, MeshTraverseEntity, MeshTraverseSelection, MeshTraverseOwnership, bool> friend class MeshBatchProxy;
    explicit MeshTraverseExecution(const MeshTraversePlan<Mesh>& plan);
    static const Mesh& checked_mesh(const MeshTraversePlan<Mesh>& plan)
    {
        plan.require_valid();
        return plan.mesh();
    }
    const MeshTraversePlan<Mesh>* d_plan;
    [[maybe_unused]] decltype(detail::retain_traverse_backend(std::declval<const Mesh&>())) d_backend;
    decltype(detail::acquire_traverse_geometry(std::declval<const Mesh&>())) d_lease;
};

/**
 * @brief Cheap selection proxy. Category dispatch occurs outside numerical loops.
 *
 * Default order is the selected subsequence of existing local ordinals (native
 * incidence order for cell-face views). independent() permits regrouping all
 * interior entries followed by boundaries grouped by batch. It is an explicit
 * assertion by the caller that aliasing, callbacks and accumulation allow that
 * reordering; it does not enable threads or concurrent scatter writes.
 */
template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
class MeshBatchProxy
{
public:
    using local_ordinal_type = typename Mesh::local_ordinal_type;

    auto all() const { return select<MeshTraverseSelection::All>(); }
    auto interior() const { return select<MeshTraverseSelection::Interior>(); }
    auto boundary() const { return select<MeshTraverseSelection::Boundary>(); }
    auto boundary(int batch) const
        requires(Entity == MeshTraverseEntity::Face);
    auto owned() const
        requires(!Incidence)
    {
        return ownership<MeshTraverseOwnership::Owned>();
    }
    auto local() const
        requires(!Incidence)
    {
        return ownership<MeshTraverseOwnership::Local>();
    }
    auto independent() const;

    size_t size() const;
    bool empty() const { return size() == 0; }

    // Caller-instantiated callback adapters are not compiled binary entry points.
    template<class Operation> SIMPLEFLUID_LOCAL void apply(Operation&& operation) const;

    /**
     * Reuse access borrowed from an enclosing typed traversal of this mesh.
     * The enclosing callback keeps the geometry lease alive; neither this
     * operation nor its entity references may outlive that callback.
     */
    template<class Access, class Operation>
        requires std::is_same_v<typename Access::mesh_type, Mesh>
    SIMPLEFLUID_LOCAL void apply(const Access& access, Operation&& operation) const;

    template<class Visitor> SIMPLEFLUID_LOCAL void visit_batches(Visitor&& visitor) const;

    /** Borrow an enclosing traversal's same-mesh access without redispatch. */
    template<class Access, class Visitor>
        requires std::is_same_v<typename Access::mesh_type, Mesh>
    SIMPLEFLUID_LOCAL void visit_batches(const Access& access, Visitor&& visitor) const;

    /** Local preflight only; the enclosing numerical operation coordinates MPI failure. */
    void require_local_adjacency() const
        requires(Entity == MeshTraverseEntity::Face);

private:
    friend class MeshTraverseExecution<Mesh>;
    template<class, MeshTraverseEntity, MeshTraverseSelection, MeshTraverseOwnership, bool> friend class MeshBatchProxy;
    explicit MeshBatchProxy(const MeshTraverseExecution<Mesh>& execution, local_ordinal_type cell = {});

    static constexpr detail::StructuredTraverseSelection structured_selection()
    {
        if constexpr (Selection == MeshTraverseSelection::Interior)
            return detail::StructuredTraverseSelection::Interior;
        else if constexpr (Selection == MeshTraverseSelection::Boundary)
            return detail::StructuredTraverseSelection::Boundary;
        else
            return detail::StructuredTraverseSelection::All;
    }
    std::optional<int> structured_batch() const { return d_has_batch ? std::optional<int>(d_batch) : std::nullopt; }
    bool structured_selected(bool interior, int batch) const
    {
        if constexpr (Selection == MeshTraverseSelection::Interior)
            return interior;
        else if constexpr (Selection == MeshTraverseSelection::Boundary)
            return !interior && (!d_has_batch || batch == d_batch);
        else
            return true;
    }

    template<class Access, class Visitor>
    SIMPLEFLUID_LOCAL void visit_structured_batches(
        const Access& access, Visitor&& visitor, const detail::StructuredTraverseLayout& layout) const;
    template<class Access, bool Interior> class SIMPLEFLUID_LOCAL LayeredIncidenceBatch
    {
    public:
        using reference = LayeredIncidenceRef<Access, Interior>;
        static constexpr bool is_interior = Interior;
        static constexpr MeshTraverseEntity entity = Entity;
        LayeredIncidenceBatch(const Access& access, detail::LayeredTraverseLayout::Incidences faces, size_t begin,
            size_t end, local_ordinal_type cell);
        size_t size() const noexcept { return d_end - d_begin; }
        bool empty() const noexcept { return size() == 0; }
        template<class Operation> void apply(Operation&& operation) const;

    private:
        const Access* d_access;
        detail::LayeredTraverseLayout::Incidences d_faces;
        size_t d_begin, d_end;
        local_ordinal_type d_cell;
    };
    template<class Access, class Visitor>
    SIMPLEFLUID_LOCAL void visit_layered_batches(
        const Access& access, Visitor&& visitor, const detail::LayeredTraverseLayout& layout) const;
    template<MeshTraverseSelection Next> auto select() const;
    template<MeshTraverseOwnership Next> auto ownership() const;
    const detail::MeshTraverseSchedule& selected_schedule() const
    {
        const auto& plan = *d_execution->d_plan;
        if constexpr (Incidence)
            return plan.d_cell_faces[static_cast<size_t>(d_cell)];
        else if constexpr (Entity == MeshTraverseEntity::Cell)
            return plan.d_cells[Ownership == MeshTraverseOwnership::Owned ? 0 : 1];
        else
            return plan.d_faces[Ownership == MeshTraverseOwnership::Owned ? 0 : 1];
    }
    std::span<const detail::MeshTraverseRun> selected_runs() const;
    const MeshTraverseExecution<Mesh>* d_execution;
    local_ordinal_type d_cell;
    bool d_independent = false, d_has_batch = false;
    int d_batch = -1;
};

template<class Mesh>
[[nodiscard]] MeshTraversePlan<Mesh> make_mesh_traverse_plan(
    const Mesh& mesh, MeshTraverseEntities entities = MeshTraverseEntities::All);

template<class Mesh>
    requires(!std::is_lvalue_reference_v<Mesh>)
void make_mesh_traverse_plan(Mesh&&, MeshTraverseEntities = MeshTraverseEntities::All) = delete;

/** Compiled plans for the supported native meshes and default Tpetra pack. */
extern template class MeshTraversePlan<MeshHandle<DefaultTpetraTypes>>;
extern template class MeshTraversePlan<Mesh<DefaultTpetraTypes>>;
extern template class MeshTraversePlan<Meshes::OrthogonalCartesian3D>;
extern template class MeshTraversePlan<Meshes::OrthogonalCylindrial3D>;
extern template class MeshTraversePlan<Meshes::SemiStructuredXY_Z>;
extern template class MeshTraversePlan<Meshes::UnstructuredMesh>;
extern template class MeshTraversePlan<Meshes::MultiRegionMesh>;
extern template class MeshTraversePlan<Meshes::STKMeshAdapter<DefaultTpetraTypes>>;
extern template class MeshTraversePlan<Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, DefaultTpetraTypes>>;
extern template class MeshTraversePlan<Meshes::PartitionedMesh<Meshes::UnstructuredMesh, DefaultTpetraTypes>>;

} // namespace SimpleFluid

#include "geometry/MeshTraverse.ipp"
