/** @file MeshTraverse.hh
 * @brief Reusable, ordered cell and face selections with typed numerical dispatch.
 */
#pragma once

#include "geometry/GeometryExecutionGuard.hh"
#include "geometry/details/MeshTraverseAccess.hh"

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace SimpleFluid
{

enum class MeshTraverseSelection { All, Interior, Boundary };
enum class MeshTraverseOwnership { Owned, Local };
enum class MeshTraverseEntity { Cell, Face };
/** Incidence storage is opt-in; ordinary traverses retain only ranges. */
enum class MeshTraverseEntities { Cells, Faces, All, AllWithCellFaces };

template<class Mesh> class MeshTraversePlan;
template<class Mesh> class MeshTraverseExecution;
template<class Mesh, MeshTraverseEntity Entity,
         MeshTraverseSelection Selection = MeshTraverseSelection::All,
         MeshTraverseOwnership Ownership = MeshTraverseOwnership::Owned,
         bool Incidence = false>
class MeshBatchProxy;

template<class Mesh, MeshTraverseSelection Selection = MeshTraverseSelection::All,
         MeshTraverseOwnership Ownership = MeshTraverseOwnership::Owned>
using CellBatchProxy = MeshBatchProxy<Mesh, MeshTraverseEntity::Cell, Selection, Ownership>;
template<class Mesh, MeshTraverseSelection Selection = MeshTraverseSelection::All,
         MeshTraverseOwnership Ownership = MeshTraverseOwnership::Owned>
using FaceBatchProxy = MeshBatchProxy<Mesh, MeshTraverseEntity::Face, Selection, Ownership>;
template<class Mesh, MeshTraverseSelection Selection = MeshTraverseSelection::All>
using CellFaceBatchProxy = MeshBatchProxy<Mesh, MeshTraverseEntity::Face, Selection,
                                        MeshTraverseOwnership::Local, true>;

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

    static void append(std::vector<MeshTraverseRun>& runs, size_t ordinal, bool inner, int batch)
    {
        if (!runs.empty() && runs.back().end == ordinal
            && runs.back().interior == inner && runs.back().batch == batch)
            ++runs.back().end;
        else
            runs.push_back({ordinal, ordinal + 1, inner, batch});
    }

    void append(size_t ordinal, bool inner, int batch, bool face)
    {
        append(ordered, ordinal, inner, batch);
        append(inner ? interior : boundary, ordinal, inner, batch);
        if (face && !inner) append(batches[batch], ordinal, false, batch);
    }

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
    if constexpr (requires { mesh.variant(); }) return mesh.variant();
    else if constexpr (requires { mesh.parent_mesh(); }) return retain_traverse_backend(mesh.parent_mesh());
    else
    {
        struct BorrowedMesh {};
        return BorrowedMesh{};
    }
}

/** Cheap binding checks. Same-count in-place topology edits require rebuilding. */
template<class Mesh> auto mesh_traverse_binding(const Mesh& mesh)
{
    std::array<const void*, 5> binding{};
    if constexpr (requires { mesh.variant(); })
        mesh.visit([&](const auto& native) { binding[0] = &native; });
    else
        binding[0] = &mesh;
    if constexpr (requires { mesh.owned_cell_map().getRawPtr(); })
        binding[1] = mesh.owned_cell_map().getRawPtr();
    if constexpr (requires { mesh.overlap_cell_map().getRawPtr(); })
        binding[2] = mesh.overlap_cell_map().getRawPtr();
    if constexpr (requires { mesh.owned_face_map().getRawPtr(); })
        binding[3] = mesh.owned_face_map().getRawPtr();
    if constexpr (requires { mesh.overlap_face_map().getRawPtr(); })
        binding[4] = mesh.overlap_face_map().getRawPtr();
    return binding;
}

template<class Mesh> auto mesh_traverse_counts(const Mesh& mesh)
{
    const auto owned_faces = [&]() -> size_t
    {
        if constexpr (requires { mesh.num_owned_faces(); }) return mesh.num_owned_faces();
        else return mesh.owned_face_map()->getLocalNumElements();
    }();
    return std::array<size_t, 4>{mesh.num_local_cells(), mesh.num_faces(),
                                mesh.num_owned_cells(), owned_faces};
}
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
template<class Access, MeshTraverseEntity Entity, bool Interior, bool Incidence = false>
class MeshTraverseRef
{
public:
    using local_ordinal_type = typename Access::local_ordinal_type;
    static constexpr bool is_interior = Interior;
    static constexpr MeshTraverseEntity entity = Entity;

    MeshTraverseRef(const Access& access, local_ordinal_type id, int batch = -1,
                 local_ordinal_type cell = {}, int orientation = 1, const void* origin_plan = nullptr)
        : d_access(&access), d_id(id), d_batch(batch), d_cell(cell), d_orientation(orientation), d_origin(origin_plan) {}

    local_ordinal_type id() const noexcept { return d_id; }
    auto centroid() const
    {
        if constexpr (Entity == MeshTraverseEntity::Cell) return d_access->cell_centroid(d_id);
        else return d_access->face_centroid(d_id);
    }
    auto volume() const requires (Entity == MeshTraverseEntity::Cell) { return d_access->cell_volume(d_id); }
    auto area() const requires (Entity == MeshTraverseEntity::Face) { return d_access->face_area(d_id); }
    auto normal() const requires (Entity == MeshTraverseEntity::Face) { return d_access->face_normal(d_id); }
    auto owner() const requires (Entity == MeshTraverseEntity::Face) { return d_access->owner_cell(d_id); }
    auto neighbor() const requires (Entity == MeshTraverseEntity::Face && Interior)
    { return d_access->neighbor_cell(d_id); }
    int boundary_id() const requires (Entity == MeshTraverseEntity::Face && !Interior) { return d_batch; }
    local_ordinal_type cell_id() const requires Incidence { return d_cell; }
    int orientation() const requires Incidence { return d_orientation; }
    auto outward_normal() const requires Incidence { return normal() * d_orientation; }
    const Access& geometry() const noexcept { return *d_access; }
    const void* origin_plan() const noexcept requires (Entity == MeshTraverseEntity::Cell) { return d_origin.plan; }

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
    LayeredIncidenceRef(const Access& access, const detail::LayeredTraverseLayout::ResolvedIncidence& face,
        local_ordinal_type cell)
        : Base(access, static_cast<local_ordinal_type>(face.incidence.face), face.incidence.batch, cell, face.incidence.orientation),
          d_owner(local(face.owner)), d_neighbor(local(face.neighbor)), d_cell(face.cell), d_other(face.other) {}
    local_ordinal_type owner() const noexcept { return d_owner; }
    local_ordinal_type neighbor() const noexcept requires Interior { return d_neighbor; }
    auto cell_center_vector() const requires Interior
    {
        if (d_owner == Access::invalid_local_id() || d_neighbor == Access::invalid_local_id())
            throw std::invalid_argument("Exterior face does not have an opposite cell.");
        return this->geometry().resolved_cell_center_vector(d_cell, d_other);
    }
private:
    static local_ordinal_type local(size_t ordinal)
    { return ordinal == detail::LayeredTraverseLayout::invalid_ordinal ? Access::invalid_local_id() : static_cast<local_ordinal_type>(ordinal); }
    local_ordinal_type d_owner, d_neighbor;
    CellID d_cell, d_other;
};

/**
 * @brief Homogeneous borrowed ranges. No selection predicate runs per entity.
 *
 * The batch and its entity references expire with the visitor callback. The
 * visitor may retain numerical results, but must not retain these views.
 */
template<class Access, MeshTraverseEntity Entity, bool Interior, bool Incidence = false>
class MeshTraverseBatch
{
public:
    using local_ordinal_type = typename Access::local_ordinal_type;
    using reference = MeshTraverseRef<Access, Entity, Interior, Incidence>;
    static constexpr bool is_interior = Interior;
    static constexpr MeshTraverseEntity entity = Entity;

    MeshTraverseBatch(const Access& access, std::span<const detail::MeshTraverseRun> runs,
        std::span<const detail::MeshTraverseIncidence<local_ordinal_type>> incidence = {},
        local_ordinal_type cell = {}, const void* origin_plan = nullptr)
        : d_access(&access), d_runs(runs), d_incidence(incidence), d_cell(cell), d_origin(origin_plan) {}

    size_t size() const noexcept
    {
        size_t count = 0;
        for (const auto& run : d_runs) count += run.end - run.begin;
        return count;
    }
    bool empty() const noexcept { return d_runs.empty(); }

    template<class Operation> void apply(Operation&& operation) const
    {
        for (const auto& run : d_runs)
            for (size_t ordinal = run.begin; ordinal < run.end; ++ordinal)
            {
                if constexpr (Incidence)
                {
                    const auto& face = d_incidence[ordinal];
                    if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Operation>>)
                        std::invoke(operation, reference(*d_access, face.face, run.batch, d_cell, face.orientation));
                    else
                        operation(reference(*d_access, face.face, run.batch, d_cell, face.orientation));
                }
                else if constexpr (Entity == MeshTraverseEntity::Cell)
                {
                    if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Operation>>)
                        std::invoke(operation, reference(*d_access, static_cast<local_ordinal_type>(ordinal), run.batch, {}, 1, d_origin.plan));
                    else operation(reference(*d_access, static_cast<local_ordinal_type>(ordinal), run.batch, {}, 1, d_origin.plan));
                }
                else
                {
                    if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Operation>>)
                        std::invoke(operation, reference(*d_access, static_cast<local_ordinal_type>(ordinal), run.batch));
                    else operation(reference(*d_access, static_cast<local_ordinal_type>(ordinal), run.batch));
                }
            }
    }

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
template<class Mesh> class MeshTraversePlan
{
public:
    using local_ordinal_type = typename Mesh::local_ordinal_type;

    explicit MeshTraversePlan(const Mesh& mesh, MeshTraverseEntities entities = MeshTraverseEntities::All)
        : d_mesh(&mesh), d_entities(entities)
    {
        const auto lease = detail::acquire_traverse_geometry(mesh);
        d_counts = detail::mesh_traverse_counts(mesh);
        d_binding = detail::mesh_traverse_binding(mesh);
        if ((d_counts[0] && d_counts[0] - 1 > static_cast<size_t>(std::numeric_limits<local_ordinal_type>::max()))
            || (d_counts[1] && d_counts[1] - 1 > static_cast<size_t>(std::numeric_limits<local_ordinal_type>::max())))
            throw std::overflow_error("Mesh traverse local ordinal overflow.");

        detail::with_traverse_access(mesh, [&](const auto& access)
        {
            d_structured = access.structured_layout();
            if (d_structured) return;
            d_layered = access.layered_layout();
            if (d_layered) return;
            if (entities != MeshTraverseEntities::Faces)
                for (size_t cell = 0; cell < d_counts[0]; ++cell)
                {
                    const bool interior = !access.is_exterior_cell(cell);
                    d_cells[1].append(cell, interior, -1, false);
                    if (access.is_owned_cell(cell)) d_cells[0].append(cell, interior, -1, false);
                }
            if (entities != MeshTraverseEntities::Cells)
                for (size_t face = 0; face < d_counts[1]; ++face)
                {
                    const bool interior = !access.is_exterior_face(face);
                    const int batch = interior ? -1 : access.boundary_id(face);
                    d_faces[1].append(face, interior, batch, true);
                    if (access.is_owned_face(face)) d_faces[0].append(face, interior, batch, true);
                }
            if (entities == MeshTraverseEntities::AllWithCellFaces)
            {
                d_cell_faces.resize(d_counts[0]);
                for (size_t cell = 0; cell < d_counts[0]; ++cell)
                    access.face_ids(cell, [&](auto face)
                    {
                        const bool interior = !access.is_exterior_face(face);
                        const int batch = interior ? -1 : access.boundary_id(face);
                        const auto position = d_incidence.size();
                        d_incidence.push_back({static_cast<local_ordinal_type>(face),
                            access.owner_cell(face) == static_cast<local_ordinal_type>(cell) ? 1 : -1});
                        d_cell_faces[cell].append(position, interior, batch, true);
                    });
            }
        });
    }

    MeshTraversePlan(const MeshTraversePlan&) = delete;
    MeshTraversePlan& operator=(const MeshTraversePlan&) = delete;
    MeshTraversePlan(MeshTraversePlan&&) = delete;
    MeshTraversePlan& operator=(MeshTraversePlan&&) = delete;
    MeshTraversePlan(Mesh&&, MeshTraverseEntities = MeshTraverseEntities::All) = delete;
    MeshTraversePlan(const Mesh&&, MeshTraverseEntities = MeshTraverseEntities::All) = delete;

    [[nodiscard]] MeshTraverseExecution<Mesh> acquire() const &;
    MeshTraverseExecution<Mesh> acquire() const && = delete;
    const Mesh& mesh() const noexcept { return *d_mesh; }
    MeshTraverseEntities entities() const noexcept { return d_entities; }
    bool uses_structured_ranges() const noexcept { return d_structured.has_value(); }

    bool uses_layered_ranges() const noexcept { return bool(d_layered); }
    const std::shared_ptr<const detail::LayeredTraverseLayout>& layered_layout() const noexcept { return d_layered; }

    /** Explicitly reject further use before editing topology in place. */
    void invalidate() noexcept { d_valid = false; }
    void require_valid() const
    {
        if (!d_valid || d_counts != detail::mesh_traverse_counts(*d_mesh)
            || d_binding != detail::mesh_traverse_binding(*d_mesh))
            throw std::logic_error("Mesh traverse plan is stale; rebuild it after topology, map, or mesh replacement.");
    }

    /** Selection storage estimate excluding allocator bookkeeping. */
    size_t storage_bytes() const noexcept
    {
        size_t bytes = sizeof(*this) + d_incidence.capacity() * sizeof(detail::MeshTraverseIncidence<local_ordinal_type>)
                     + d_cell_faces.capacity() * sizeof(detail::MeshTraverseSchedule);
        for (const auto& schedule : d_cells) bytes += schedule.storage_bytes();
        for (const auto& schedule : d_faces) bytes += schedule.storage_bytes();
        for (const auto& schedule : d_cell_faces) bytes += schedule.storage_bytes();
        if (d_layered) bytes += d_layered->storage_bytes();
        return bytes;
    }

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

/** Direct interior incidences from a same-plan typed cell proof. */
template<class Mesh, class Access, MeshTraverseSelection Selection = MeshTraverseSelection::All>
class LayeredInteriorCellFaces
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
    size_t size() const { d_plan->require_valid(); if constexpr (Selection == MeshTraverseSelection::Boundary) return 0; else return d_faces.size(); }
    bool empty() const { return size() == 0; }
    void require_local_adjacency() const
    {
        // This proof is available only for a canonical, fully local layout.
        // Physical interior classification therefore proves both adjacent IDs.
        validate(*d_access);
    }
    template<class Operation> void apply(Operation&& operation) const
    { apply(*d_access, std::forward<Operation>(operation)); }
    template<class Operation>
    [[gnu::always_inline]] void apply(const Access& access, Operation&& operation) const
    {
        validate(access);
        if constexpr (Selection != MeshTraverseSelection::Boundary)
            for (size_t slot = 0; slot < d_faces.size(); ++slot)
            {
                const auto face = access.resolve_incidence(d_faces, slot);
                if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Operation>>)
                    std::invoke(operation, reference(access, face, d_cell));
                else operation(reference(access, face, d_cell));
            }
    }
    template<class Visitor> void visit_batches(Visitor&& visitor) const
    { visit_batches(*d_access, std::forward<Visitor>(visitor)); }
    template<class Visitor> void visit_batches(const Access& access, Visitor&& visitor) const
    {
        validate(access);
        if constexpr (Selection != MeshTraverseSelection::Boundary)
        {
            if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Visitor>>)
                std::invoke(visitor, LayeredInteriorCellFaces(d_plan, &access, d_cell, d_faces));
            else visitor(LayeredInteriorCellFaces(d_plan, &access, d_cell, d_faces));
        }
    }
private:
    friend class MeshTraverseExecution<Mesh>;
    template<class, class, MeshTraverseSelection> friend class LayeredInteriorCellFaces;
    LayeredInteriorCellFaces(const MeshTraversePlan<Mesh>& plan, const Access& access, local_ordinal_type cell)
        : d_plan(&plan), d_access(&access), d_cell(cell), d_faces(plan.layered_layout()->cell_faces(cell)) {}
    LayeredInteriorCellFaces(const MeshTraversePlan<Mesh>* plan, const Access* access,
        local_ordinal_type cell, detail::LayeredTraverseLayout::Incidences faces)
        : d_plan(plan), d_access(access), d_cell(cell), d_faces(faces) {}
    template<MeshTraverseSelection Next> auto select() const
    { return LayeredInteriorCellFaces<Mesh, Access, Next>(d_plan, d_access, d_cell, d_faces); }
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

    CellBatchProxy<Mesh> cells() const &
    {
        d_plan->require_valid();
        if (d_plan->d_entities == MeshTraverseEntities::Faces)
            throw std::logic_error("This mesh traverse plan contains only faces.");
        return CellBatchProxy<Mesh>(*this);
    }
    FaceBatchProxy<Mesh> faces() const &
    {
        d_plan->require_valid();
        if (d_plan->d_entities == MeshTraverseEntities::Cells)
            throw std::logic_error("This mesh traverse plan contains only cells.");
        return FaceBatchProxy<Mesh>(*this);
    }
    CellFaceBatchProxy<Mesh> cell_faces(typename Mesh::local_ordinal_type cell) const &
    {
        d_plan->require_valid();
        require_cell_faces_id(cell);
        return CellFaceBatchProxy<Mesh>(*this, cell);
    }
    template<class Access, bool Interior>
        requires std::is_same_v<typename Access::mesh_type, Mesh>
    auto cell_faces(const MeshTraverseRef<Access, MeshTraverseEntity::Cell, Interior>& cell) const &
    {
        d_plan->require_valid();
        if (&cell.geometry().mesh() != &d_plan->mesh() || cell.origin_plan() != d_plan)
            throw std::invalid_argument("Mesh traverse cell proof belongs to another plan or mesh.");
        require_cell_faces_id(cell.id());
        if constexpr (Interior && detail::is_layered_traverse_access<Access>)
        {
            if (!d_plan->d_layered) throw std::invalid_argument("Layered cell proof requires a layered plan.");
            return LayeredInteriorCellFaces<Mesh, Access>(*d_plan, cell.geometry(), cell.id());
        }
        else return CellFaceBatchProxy<Mesh>(*this, cell.id());
    }
    template<class Access, bool Interior>
    auto cell_faces(const MeshTraverseRef<Access, MeshTraverseEntity::Cell, Interior>&) const && = delete;
    CellBatchProxy<Mesh> cells() const && = delete;
    FaceBatchProxy<Mesh> faces() const && = delete;
    CellFaceBatchProxy<Mesh> cell_faces(typename Mesh::local_ordinal_type) const && = delete;

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
    explicit MeshTraverseExecution(const MeshTraversePlan<Mesh>& plan)
        : d_plan(&plan), d_backend(detail::retain_traverse_backend(checked_mesh(plan))),
          d_lease(detail::acquire_traverse_geometry(plan.mesh())) {}
    static const Mesh& checked_mesh(const MeshTraversePlan<Mesh>& plan)
    { plan.require_valid(); return plan.mesh(); }
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
template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection,
         MeshTraverseOwnership Ownership, bool Incidence>
class MeshBatchProxy
{
public:
    using local_ordinal_type = typename Mesh::local_ordinal_type;

    auto all() const { return select<MeshTraverseSelection::All>(); }
    auto interior() const { return select<MeshTraverseSelection::Interior>(); }
    auto boundary() const { return select<MeshTraverseSelection::Boundary>(); }
    auto boundary(int batch) const requires (Entity == MeshTraverseEntity::Face)
    {
        auto result = select<MeshTraverseSelection::Boundary>();
        result.d_batch = batch;
        result.d_has_batch = true;
        return result;
    }
    auto owned() const requires (!Incidence) { return ownership<MeshTraverseOwnership::Owned>(); }
    auto local() const requires (!Incidence) { return ownership<MeshTraverseOwnership::Local>(); }
    auto independent() const { auto result = *this; result.d_independent = true; return result; }

    size_t size() const
    {
        const auto& plan = *d_execution->d_plan;
        plan.require_valid();
        if (plan.d_structured)
        {
            const auto& layout = *plan.d_structured;
            if constexpr (Incidence)
            {
                size_t count = 0;
                for (const auto& face : layout.cell_faces(static_cast<size_t>(d_cell)))
                    if (structured_selected(face.interior, face.batch)) ++count;
                return count;
            }
            else if constexpr (Entity == MeshTraverseEntity::Cell)
                return layout.cell_count(structured_selection());
            else
                return layout.face_count(structured_selection(), structured_batch());
        }
        if (plan.d_layered)
        {
            const auto& layout = *plan.d_layered;
            if constexpr (Incidence)
            {
                size_t count = 0;
                for (const auto face : layout.cell_faces(static_cast<size_t>(d_cell)))
                    if (structured_selected(face.interior, face.batch)) ++count;
                return count;
            }
            else if constexpr (Entity == MeshTraverseEntity::Cell) return layout.cell_count(structured_selection());
            else return layout.face_count(structured_selection(), structured_batch());
        }
        size_t count = 0;
        for (const auto& run : selected_runs()) count += run.end - run.begin;
        return count;
    }
    bool empty() const { return size() == 0; }

    template<class Operation> void apply(Operation&& operation) const
    {
        visit_batches([&](const auto& batch) { batch.apply(operation); });
    }

    /**
     * Reuse access borrowed from an enclosing typed traversal of this mesh.
     * The enclosing callback keeps the geometry lease alive; neither this
     * operation nor its entity references may outlive that callback.
     */
    template<class Access, class Operation>
        requires std::is_same_v<typename Access::mesh_type, Mesh>
    void apply(const Access& access, Operation&& operation) const
    {
        visit_batches(access, [&](const auto& batch) { batch.apply(operation); });
    }

    template<class Visitor> void visit_batches(Visitor&& visitor) const
    {
        const auto& plan = *d_execution->d_plan;
        plan.require_valid();
        detail::with_traverse_access(plan.mesh(), [&](const auto& access)
        {
            visit_batches(access, visitor);
        });
    }

    /** Borrow an enclosing traversal's same-mesh access without redispatch. */
    template<class Access, class Visitor>
        requires std::is_same_v<typename Access::mesh_type, Mesh>
    void visit_batches(const Access& access, Visitor&& visitor) const
    {
        const auto& plan = *d_execution->d_plan;
        plan.require_valid();
        if (&access.mesh() != &plan.mesh())
            throw std::invalid_argument("Mesh traverse access belongs to another mesh.");
        if constexpr (detail::is_layered_traverse_access<Access>)
        {
            if (!plan.d_layered) throw std::invalid_argument("Layered access requires a layered plan.");
            visit_layered_batches(access, visitor, *plan.d_layered);
        }
        else if constexpr (detail::is_structured_traverse_access<Access>)
        {
            if (!plan.d_structured)
                throw std::invalid_argument("Structured traverse access requires a structured plan.");
            visit_structured_batches(access, visitor, *plan.d_structured);
        }
        else
        {
            if constexpr (detail::has_structured_traverse_layout<typename Access::geometry_type>
                && (detail::traverse_is_handle<typename Access::mesh_type>
                    || std::is_same_v<typename Access::mesh_type, typename Access::geometry_type>))
            {
                if (plan.d_structured)
                {
                    const detail::StructuredTraverseAccess<Access> structured(access);
                    visit_structured_batches(structured, visitor, *plan.d_structured);
                    return;
                }
            }
            if constexpr (detail::has_layered_traverse_layout<typename Access::geometry_type>
                && (detail::traverse_is_handle<typename Access::mesh_type>
                    || std::is_same_v<typename Access::mesh_type, typename Access::geometry_type>))
            {
                if (plan.d_layered)
                {
                    const detail::LayeredTraverseAccess<Access> layered(access);
                    visit_layered_batches(layered, visitor, *plan.d_layered);
                    return;
                }
            }
            const auto emit = [&]<bool Interior>(std::span<const detail::MeshTraverseRun> runs)
            {
                if (!runs.empty())
                    std::invoke(visitor, MeshTraverseBatch<Access, Entity, Interior, Incidence>(
                        access, runs, plan.d_incidence, d_cell, &plan));
            };
            const auto& schedule = selected_schedule();
            if constexpr (Selection == MeshTraverseSelection::Interior)
                emit.template operator()<true>(schedule.interior);
            else if constexpr (Selection == MeshTraverseSelection::Boundary)
            {
                if (d_independent && !d_has_batch && Entity == MeshTraverseEntity::Face)
                    for (const auto& [batch, runs] : schedule.batches) emit.template operator()<false>(runs);
                else
                    emit.template operator()<false>(selected_runs());
            }
            else
            {
                if (d_independent)
                {
                    emit.template operator()<true>(schedule.interior);
                    if constexpr (Entity == MeshTraverseEntity::Face)
                        for (const auto& [batch, runs] : schedule.batches) emit.template operator()<false>(runs);
                    else
                        emit.template operator()<false>(schedule.boundary);
                }
                else
                    for (const auto& run : schedule.ordered)
                    {
                        if (run.interior) emit.template operator()<true>({&run, 1});
                        else emit.template operator()<false>({&run, 1});
                    }
            }
        }
    }

    /** Local preflight only; the enclosing numerical operation coordinates MPI failure. */
    void require_local_adjacency() const requires (Entity == MeshTraverseEntity::Face)
    {
        visit_batches([&](const auto& batch)
        {
            batch.apply([](const auto& face)
            {
                using Access = std::remove_cvref_t<decltype(face.geometry())>;
                if (face.owner() == Access::invalid_local_id())
                    throw std::invalid_argument("Mesh traverse requires the face owner in the local cell map.");
                if constexpr (std::remove_cvref_t<decltype(face)>::is_interior)
                    if (face.neighbor() == Access::invalid_local_id())
                        throw std::invalid_argument("Mesh traverse requires the face neighbor in the local cell map.");
            });
        });
    }

private:
    friend class MeshTraverseExecution<Mesh>;
    template<class, MeshTraverseEntity, MeshTraverseSelection, MeshTraverseOwnership, bool> friend class MeshBatchProxy;
    explicit MeshBatchProxy(const MeshTraverseExecution<Mesh>& execution, local_ordinal_type cell = {})
        : d_execution(&execution), d_cell(cell) {}

    static constexpr detail::StructuredTraverseSelection structured_selection()
    {
        if constexpr (Selection == MeshTraverseSelection::Interior)
            return detail::StructuredTraverseSelection::Interior;
        else if constexpr (Selection == MeshTraverseSelection::Boundary)
            return detail::StructuredTraverseSelection::Boundary;
        else
            return detail::StructuredTraverseSelection::All;
    }
    std::optional<int> structured_batch() const
    {
        return d_has_batch ? std::optional<int>(d_batch) : std::nullopt;
    }
    bool structured_selected(bool interior, int batch) const
    {
        if constexpr (Selection == MeshTraverseSelection::Interior) return interior;
        else if constexpr (Selection == MeshTraverseSelection::Boundary)
            return !interior && (!d_has_batch || batch == d_batch);
        else return true;
    }

    template<class Access, class Visitor>
    void visit_structured_batches(const Access& access, Visitor&& visitor,
        const detail::StructuredTraverseLayout& layout) const
    {
        if constexpr (Incidence)
        {
            // A cell has exactly six canonical incidences, including distinct
            // periodic faces sharing a neighbour. Scratch expires with this
            // callback and never becomes per-cell retained heap storage.
            const auto faces = layout.cell_faces(static_cast<size_t>(d_cell));
            std::array<detail::MeshTraverseIncidence<local_ordinal_type>, 6> incidence;
            bool all_interior = true;
            for (size_t i = 0; i < faces.size(); ++i)
            {
                incidence[i] = {static_cast<local_ordinal_type>(faces[i].face), faces[i].orientation};
                all_interior = all_interior && faces[i].interior;
            }
            // Most structured cells need one six-face range. Avoid rebuilding
            // and filtering six general run descriptors on this hot path.
            if (all_interior)
            {
                if constexpr (Selection != MeshTraverseSelection::Boundary)
                {
                    const detail::MeshTraverseRun run{0, faces.size(), true, -1};
                    std::invoke(visitor, MeshTraverseBatch<Access, Entity, true, Incidence>(
                        access, {&run, 1}, incidence, d_cell, d_execution->d_plan));
                }
                return;
            }
            const auto emit_selected = [&](auto&& selected)
            {
                std::array<detail::MeshTraverseRun, 6> runs;
                size_t count = 0;
                for (size_t i = 0; i < faces.size(); ++i)
                {
                    const auto& face = faces[i];
                    if (!selected(face)) continue;
                    if (count && runs[count - 1].end == i
                        && runs[count - 1].interior == face.interior
                        && runs[count - 1].batch == face.batch)
                        runs[count - 1].end = i + 1;
                    else
                        runs[count++] = {i, i + 1, face.interior, face.batch};
                }
                for (size_t i = 0; i < count; ++i)
                {
                    const auto emit = [&]<bool Interior>()
                    {
                        std::invoke(visitor, MeshTraverseBatch<Access, Entity, Interior, Incidence>(
                            access, {&runs[i], 1}, incidence, d_cell, d_execution->d_plan));
                    };
                    if constexpr (Selection == MeshTraverseSelection::Interior)
                        emit.template operator()<true>();
                    else if constexpr (Selection == MeshTraverseSelection::Boundary)
                        emit.template operator()<false>();
                    else if (runs[i].interior) emit.template operator()<true>();
                    else emit.template operator()<false>();
                }
            };
            if constexpr (Selection == MeshTraverseSelection::All)
            {
                if (d_independent)
                {
                    emit_selected([](const auto& face) { return face.interior; });
                    // Orthogonal incidence order is lower/upper per axis, so
                    // its boundary subsequence is already sorted by batch ID.
                    emit_selected([](const auto& face) { return !face.interior; });
                }
                else emit_selected([](const auto&) { return true; });
            }
            else emit_selected([&](const auto& face) { return structured_selected(face.interior, face.batch); });
        }
        else
        {
            const auto emit_run = [&](const detail::StructuredTraverseRun& structured)
            {
                const detail::MeshTraverseRun run{
                    structured.begin, structured.end, structured.interior, structured.batch};
                const auto emit = [&]<bool Interior>()
                {
                    std::invoke(visitor, MeshTraverseBatch<Access, Entity, Interior, Incidence>(access, {&run, 1}, {}, {}, d_execution->d_plan));
                };
                if constexpr (Selection == MeshTraverseSelection::Interior)
                    emit.template operator()<true>();
                else if constexpr (Selection == MeshTraverseSelection::Boundary)
                    emit.template operator()<false>();
                else if (run.interior) emit.template operator()<true>();
                else emit.template operator()<false>();
            };
            if constexpr (Entity == MeshTraverseEntity::Cell)
                layout.visit_cells(structured_selection(), d_independent, emit_run);
            else
                layout.visit_faces(structured_selection(), d_independent, structured_batch(), emit_run);
        }
    }
    template<class Access, bool Interior> class LayeredIncidenceBatch
    {
    public:
        using reference = LayeredIncidenceRef<Access, Interior>;
        static constexpr bool is_interior = Interior;
        static constexpr MeshTraverseEntity entity = Entity;
        LayeredIncidenceBatch(const Access& access, detail::LayeredTraverseLayout::Incidences faces,
            size_t begin, size_t end, local_ordinal_type cell)
            : d_access(&access), d_faces(faces), d_begin(begin), d_end(end), d_cell(cell) {}
        size_t size() const noexcept { return d_end-d_begin; }
        bool empty() const noexcept { return size()==0; }
        template<class Operation> void apply(Operation&& operation) const
        {
            for (size_t i=d_begin; i<d_end; ++i)
            {
                const auto face=d_access->resolve_incidence(d_faces, i);
                if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Operation>>)
                    std::invoke(operation, reference(*d_access, face, d_cell));
                else operation(reference(*d_access, face, d_cell));
            }
        }
    private:
        const Access* d_access;
        detail::LayeredTraverseLayout::Incidences d_faces;
        size_t d_begin,d_end;
        local_ordinal_type d_cell;
    };
    template<class Access, class Visitor>
    void visit_layered_batches(const Access& access, Visitor&& visitor,
        const detail::LayeredTraverseLayout& layout) const
    {
        if constexpr (Incidence)
        {
            const auto faces=layout.cell_faces(static_cast<size_t>(d_cell));
            faces.visit_runs(structured_selection(), d_independent, structured_batch(),
                [&](const detail::StructuredTraverseRun& run)
                {
                    if constexpr (Selection == MeshTraverseSelection::Interior)
                        std::invoke(visitor, LayeredIncidenceBatch<Access, true>(access, faces, run.begin, run.end, d_cell));
                    else if constexpr (Selection == MeshTraverseSelection::Boundary)
                        std::invoke(visitor, LayeredIncidenceBatch<Access, false>(access, faces, run.begin, run.end, d_cell));
                    else if (run.interior)
                        std::invoke(visitor, LayeredIncidenceBatch<Access, true>(access, faces, run.begin, run.end, d_cell));
                    else
                        std::invoke(visitor, LayeredIncidenceBatch<Access, false>(access, faces, run.begin, run.end, d_cell));
                });
        }
        else
        {
            const auto emit_run=[&](const detail::StructuredTraverseRun& source)
            {
                const detail::MeshTraverseRun run{source.begin,source.end,source.interior,source.batch};
                if constexpr (Selection == MeshTraverseSelection::Interior)
                    std::invoke(visitor,MeshTraverseBatch<Access,Entity,true>(access,{&run,1},{},{},d_execution->d_plan));
                else if constexpr (Selection == MeshTraverseSelection::Boundary)
                    std::invoke(visitor,MeshTraverseBatch<Access,Entity,false>(access,{&run,1},{},{},d_execution->d_plan));
                else if (source.interior) std::invoke(visitor,MeshTraverseBatch<Access,Entity,true>(access,{&run,1},{},{},d_execution->d_plan));
                else std::invoke(visitor,MeshTraverseBatch<Access,Entity,false>(access,{&run,1},{},{},d_execution->d_plan));
            };
            if constexpr (Entity == MeshTraverseEntity::Cell) layout.visit_cells(structured_selection(),d_independent,emit_run);
            else layout.visit_faces(structured_selection(),d_independent,structured_batch(),emit_run);
        }
    }
    template<MeshTraverseSelection Next> auto select() const
    {
        MeshBatchProxy<Mesh, Entity, Next, Ownership, Incidence> result(*d_execution, d_cell);
        result.d_independent = d_independent;
        return result;
    }
    template<MeshTraverseOwnership Next> auto ownership() const
    {
        MeshBatchProxy<Mesh, Entity, Selection, Next, Incidence> result(*d_execution, d_cell);
        result.d_independent = d_independent;
        result.d_has_batch = d_has_batch;
        result.d_batch = d_batch;
        return result;
    }
    const detail::MeshTraverseSchedule& selected_schedule() const
    {
        const auto& plan = *d_execution->d_plan;
        if constexpr (Incidence) return plan.d_cell_faces[static_cast<size_t>(d_cell)];
        else if constexpr (Entity == MeshTraverseEntity::Cell)
            return plan.d_cells[Ownership == MeshTraverseOwnership::Owned ? 0 : 1];
        else
            return plan.d_faces[Ownership == MeshTraverseOwnership::Owned ? 0 : 1];
    }
    std::span<const detail::MeshTraverseRun> selected_runs() const
    {
        const auto& schedule = selected_schedule();
        if constexpr (Selection == MeshTraverseSelection::Interior) return schedule.interior;
        else if constexpr (Selection == MeshTraverseSelection::Boundary)
        {
            if (d_has_batch)
            {
                const auto iter = schedule.batches.find(d_batch);
                return iter == schedule.batches.end() ? std::span<const detail::MeshTraverseRun>{} : iter->second;
            }
            return schedule.boundary;
        }
        else return schedule.ordered;
    }
    const MeshTraverseExecution<Mesh>* d_execution;
    local_ordinal_type d_cell;
    bool d_independent = false, d_has_batch = false;
    int d_batch = -1;
};

template<class Mesh> MeshTraverseExecution<Mesh> MeshTraversePlan<Mesh>::acquire() const &
{ return MeshTraverseExecution<Mesh>(*this); }

template<class Mesh>
[[nodiscard]] MeshTraversePlan<Mesh> make_mesh_traverse_plan(
    const Mesh& mesh, MeshTraverseEntities entities = MeshTraverseEntities::All)
{ return MeshTraversePlan<Mesh>(mesh, entities); }

template<class Mesh> requires (!std::is_lvalue_reference_v<Mesh>)
void make_mesh_traverse_plan(Mesh&&, MeshTraverseEntities = MeshTraverseEntities::All) = delete;

} // namespace SimpleFluid
