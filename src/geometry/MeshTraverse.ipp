/** @file MeshTraverse.ipp
 * @brief Inline accessors and numerical loops for MeshTraverse.
 */
#pragma once

#include "geometry/MeshTraverse.hh"

#include <functional>
#include <stdexcept>

namespace SimpleFluid
{

inline void detail::MeshTraverseSchedule::append(
    std::vector<MeshTraverseRun>& runs, size_t ordinal, bool inner, int batch)
{
    if (!runs.empty() && runs.back().end == ordinal && runs.back().interior == inner && runs.back().batch == batch)
        ++runs.back().end;
    else
        runs.push_back({ordinal, ordinal + 1, inner, batch});
}

inline void detail::MeshTraverseSchedule::append(size_t ordinal, bool inner, int batch, bool face)
{
    append(ordered, ordinal, inner, batch);
    append(inner ? interior : boundary, ordinal, inner, batch);
    if (face && !inner)
        append(batches[batch], ordinal, false, batch);
}

template<class Access, MeshTraverseEntity Entity, bool Interior, bool Incidence>
inline MeshTraverseRef<Access, Entity, Interior, Incidence>::MeshTraverseRef(const Access& access,
    local_ordinal_type id, int batch, local_ordinal_type cell, int orientation, const void* origin_plan)
    : d_access(&access), d_id(id), d_batch(batch), d_cell(cell), d_orientation(orientation), d_origin(origin_plan)
{
}

template<class Access, bool Interior>
inline LayeredIncidenceRef<Access, Interior>::LayeredIncidenceRef(
    const Access& access, const detail::LayeredTraverseLayout::ResolvedIncidence& face, local_ordinal_type cell)
    : Base(access, static_cast<local_ordinal_type>(face.incidence.face), face.incidence.batch, cell,
          face.incidence.orientation),
      d_owner(local(face.owner)), d_neighbor(local(face.neighbor)), d_cell(face.cell), d_other(face.other)
{
}

template<class Access, MeshTraverseEntity Entity, bool Interior, bool Incidence>
inline MeshTraverseBatch<Access, Entity, Interior, Incidence>::MeshTraverseBatch(const Access& access,
    std::span<const detail::MeshTraverseRun> runs,
    std::span<const detail::MeshTraverseIncidence<local_ordinal_type>> incidence, local_ordinal_type cell,
    const void* origin_plan)
    : d_access(&access), d_runs(runs), d_incidence(incidence), d_cell(cell), d_origin(origin_plan)
{
}

template<class Access, MeshTraverseEntity Entity, bool Interior, bool Incidence>
template<class Operation>
inline void MeshTraverseBatch<Access, Entity, Interior, Incidence>::apply(Operation&& operation) const
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
                    std::invoke(operation, reference(*d_access, static_cast<local_ordinal_type>(ordinal), run.batch, {},
                                               1, d_origin.plan));
                else
                    operation(reference(
                        *d_access, static_cast<local_ordinal_type>(ordinal), run.batch, {}, 1, d_origin.plan));
            }
            else
            {
                if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Operation>>)
                    std::invoke(operation, reference(*d_access, static_cast<local_ordinal_type>(ordinal), run.batch));
                else
                    operation(reference(*d_access, static_cast<local_ordinal_type>(ordinal), run.batch));
            }
        }
}

template<class Mesh> inline void MeshTraversePlan<Mesh>::invalidate() noexcept
{
    d_valid = false;
}

template<class Mesh, class Access, MeshTraverseSelection Selection>
template<class Operation>
inline void LayeredInteriorCellFaces<Mesh, Access, Selection>::apply(Operation&& operation) const
{
    apply(*d_access, std::forward<Operation>(operation));
}

template<class Mesh, class Access, MeshTraverseSelection Selection>
template<class Operation>
[[gnu::always_inline]] inline void LayeredInteriorCellFaces<Mesh, Access, Selection>::apply(
    const Access& access, Operation&& operation) const
{
    validate(access);
    if constexpr (Selection != MeshTraverseSelection::Boundary)
        for (size_t slot = 0; slot < d_faces.size(); ++slot)
        {
            const auto face = access.resolve_incidence(d_faces, slot);
            if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Operation>>)
                std::invoke(operation, reference(access, face, d_cell));
            else
                operation(reference(access, face, d_cell));
        }
}

template<class Mesh, class Access, MeshTraverseSelection Selection>
template<class Visitor>
inline void LayeredInteriorCellFaces<Mesh, Access, Selection>::visit_batches(Visitor&& visitor) const
{
    visit_batches(*d_access, std::forward<Visitor>(visitor));
}

template<class Mesh, class Access, MeshTraverseSelection Selection>
template<class Visitor>
inline void LayeredInteriorCellFaces<Mesh, Access, Selection>::visit_batches(
    const Access& access, Visitor&& visitor) const
{
    validate(access);
    if constexpr (Selection != MeshTraverseSelection::Boundary)
    {
        if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Visitor>>)
            std::invoke(visitor, LayeredInteriorCellFaces(d_plan, &access, d_cell, d_faces));
        else
            visitor(LayeredInteriorCellFaces(d_plan, &access, d_cell, d_faces));
    }
}

template<class Mesh, class Access, MeshTraverseSelection Selection>
inline LayeredInteriorCellFaces<Mesh, Access, Selection>::LayeredInteriorCellFaces(
    const MeshTraversePlan<Mesh>& plan, const Access& access, local_ordinal_type cell)
    : d_plan(&plan), d_access(&access), d_cell(cell), d_faces(plan.layered_layout()->cell_faces(cell))
{
}

template<class Mesh, class Access, MeshTraverseSelection Selection>
inline LayeredInteriorCellFaces<Mesh, Access, Selection>::LayeredInteriorCellFaces(const MeshTraversePlan<Mesh>* plan,
    const Access* access, local_ordinal_type cell, detail::LayeredTraverseLayout::Incidences faces)
    : d_plan(plan), d_access(access), d_cell(cell), d_faces(faces)
{
}

template<class Mesh>
template<class Access, bool Interior>
    requires std::is_same_v<typename Access::mesh_type, Mesh>
inline auto MeshTraverseExecution<Mesh>::cell_faces(
    const MeshTraverseRef<Access, MeshTraverseEntity::Cell, Interior>& cell) const&
{
    d_plan->require_valid();
    if (&cell.geometry().mesh() != &d_plan->mesh() || cell.origin_plan() != d_plan)
        throw std::invalid_argument("Mesh traverse cell proof belongs to another plan or mesh.");
    require_cell_faces_id(cell.id());
    if constexpr (Interior && detail::is_layered_traverse_access<Access>)
    {
        if (!d_plan->d_layered)
            throw std::invalid_argument("Layered cell proof requires a layered plan.");
        return LayeredInteriorCellFaces<Mesh, Access>(*d_plan, cell.geometry(), cell.id());
    }
    else
        return CellFaceBatchProxy<Mesh>(*this, cell.id());
}

template<class Mesh>
inline MeshTraverseExecution<Mesh>::MeshTraverseExecution(const MeshTraversePlan<Mesh>& plan)
    : d_plan(&plan), d_backend(detail::retain_traverse_backend(checked_mesh(plan))),
      d_lease(detail::acquire_traverse_geometry(plan.mesh()))
{
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
inline auto MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::boundary(int batch) const
    requires(Entity == MeshTraverseEntity::Face)
{
    auto result = select<MeshTraverseSelection::Boundary>();
    result.d_batch = batch;
    result.d_has_batch = true;
    return result;
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
inline auto MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::independent() const
{
    auto result = *this;
    result.d_independent = true;
    return result;
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<class Operation>
inline void MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::apply(Operation&& operation) const
{
    visit_batches([&](const auto& batch) { batch.apply(operation); });
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<class Access, class Operation>
    requires std::is_same_v<typename Access::mesh_type, Mesh>
inline void MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::apply(
    const Access& access, Operation&& operation) const
{
    visit_batches(access, [&](const auto& batch) { batch.apply(operation); });
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
inline MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::MeshBatchProxy(
    const MeshTraverseExecution<Mesh>& execution, local_ordinal_type cell)
    : d_execution(&execution), d_cell(cell)
{
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<MeshTraverseSelection Next>
inline auto MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::select() const
{
    MeshBatchProxy<Mesh, Entity, Next, Ownership, Incidence> result(*d_execution, d_cell);
    result.d_independent = d_independent;
    return result;
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<MeshTraverseOwnership Next>
inline auto MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::ownership() const
{
    MeshBatchProxy<Mesh, Entity, Selection, Next, Incidence> result(*d_execution, d_cell);
    result.d_independent = d_independent;
    result.d_has_batch = d_has_batch;
    result.d_batch = d_batch;
    return result;
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
inline std::span<const detail::MeshTraverseRun>
MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::selected_runs() const
{
    const auto& schedule = selected_schedule();
    if constexpr (Selection == MeshTraverseSelection::Interior)
        return schedule.interior;
    else if constexpr (Selection == MeshTraverseSelection::Boundary)
    {
        if (d_has_batch)
        {
            const auto iter = schedule.batches.find(d_batch);
            return iter == schedule.batches.end() ? std::span<const detail::MeshTraverseRun>{} : iter->second;
        }
        return schedule.boundary;
    }
    else
        return schedule.ordered;
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<class Access, bool Interior>
inline MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::LayeredIncidenceBatch<Access,
    Interior>::LayeredIncidenceBatch(const Access& access, detail::LayeredTraverseLayout::Incidences faces,
    size_t begin, size_t end, local_ordinal_type cell)
    : d_access(&access), d_faces(faces), d_begin(begin), d_end(end), d_cell(cell)
{
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<class Access, bool Interior>
template<class Operation>
inline void
MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::LayeredIncidenceBatch<Access, Interior>::apply(
    Operation&& operation) const
{
    for (size_t i = d_begin; i < d_end; ++i)
    {
        const auto face = d_access->resolve_incidence(d_faces, i);
        if constexpr (std::is_member_pointer_v<std::remove_cvref_t<Operation>>)
            std::invoke(operation, reference(*d_access, face, d_cell));
        else
            operation(reference(*d_access, face, d_cell));
    }
}

template<class Mesh> MeshTraverseExecution<Mesh> MeshTraversePlan<Mesh>::acquire() const&
{
    return MeshTraverseExecution<Mesh>(*this);
}

template<class Mesh>
[[nodiscard]] MeshTraversePlan<Mesh> make_mesh_traverse_plan(const Mesh& mesh, MeshTraverseEntities entities)
{
    return MeshTraversePlan<Mesh>(mesh, entities);
}

namespace detail
{
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
        if constexpr (requires { mesh.num_owned_faces(); })
            return mesh.num_owned_faces();
        else
            return mesh.owned_face_map()->getLocalNumElements();
    }();
    return std::array<size_t, 4>{mesh.num_local_cells(), mesh.num_faces(), mesh.num_owned_cells(), owned_faces};
}

} // namespace detail

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
inline size_t MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::size() const
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
                if (structured_selected(face.interior, face.batch))
                    ++count;
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
                if (structured_selected(face.interior, face.batch))
                    ++count;
            return count;
        }
        else if constexpr (Entity == MeshTraverseEntity::Cell)
            return layout.cell_count(structured_selection());
        else
            return layout.face_count(structured_selection(), structured_batch());
    }
    size_t count = 0;
    for (const auto& run : selected_runs())
        count += run.end - run.begin;
    return count;
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<class Visitor>
inline void MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::visit_batches(Visitor&& visitor) const
{
    const auto& plan = *d_execution->d_plan;
    plan.require_valid();
    detail::with_traverse_access(plan.mesh(), [&](const auto& access) { visit_batches(access, visitor); });
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<class Access, class Visitor>
    requires std::is_same_v<typename Access::mesh_type, Mesh>
inline void MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::visit_batches(
    const Access& access, Visitor&& visitor) const
{
    const auto& plan = *d_execution->d_plan;
    plan.require_valid();
    if (&access.mesh() != &plan.mesh())
        throw std::invalid_argument("Mesh traverse access belongs to another mesh.");
    if constexpr (detail::is_layered_traverse_access<Access>)
    {
        if (!plan.d_layered)
            throw std::invalid_argument("Layered access requires a layered plan.");
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
        if constexpr (detail::has_structured_traverse_layout<typename Access::geometry_type> &&
                      (detail::traverse_is_handle<typename Access::mesh_type> ||
                          std::is_same_v<typename Access::mesh_type, typename Access::geometry_type>) )
        {
            if (plan.d_structured)
            {
                const detail::StructuredTraverseAccess<Access> structured(access);
                visit_structured_batches(structured, visitor, *plan.d_structured);
                return;
            }
        }
        if constexpr (detail::has_layered_traverse_layout<typename Access::geometry_type> &&
                      (detail::traverse_is_handle<typename Access::mesh_type> ||
                          std::is_same_v<typename Access::mesh_type, typename Access::geometry_type>) )
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
                for (const auto& [batch, runs] : schedule.batches)
                    emit.template operator()<false>(runs);
            else
                emit.template operator()<false>(selected_runs());
        }
        else
        {
            if (d_independent)
            {
                emit.template operator()<true>(schedule.interior);
                if constexpr (Entity == MeshTraverseEntity::Face)
                    for (const auto& [batch, runs] : schedule.batches)
                        emit.template operator()<false>(runs);
                else
                    emit.template operator()<false>(schedule.boundary);
            }
            else
                for (const auto& run : schedule.ordered)
                {
                    if (run.interior)
                        emit.template operator()<true>({&run, 1});
                    else
                        emit.template operator()<false>({&run, 1});
                }
        }
    }
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
inline void MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::require_local_adjacency() const
    requires(Entity == MeshTraverseEntity::Face)
{
    visit_batches(
        [&](const auto& batch)
        {
            batch.apply(
                [](const auto& face)
                {
                    using Access = std::remove_cvref_t<decltype(face.geometry())>;
                    if (face.owner() == Access::invalid_local_id())
                        throw std::invalid_argument("Mesh traverse requires the face owner in the local cell map.");
                    if constexpr (std::remove_cvref_t<decltype(face)>::is_interior)
                        if (face.neighbor() == Access::invalid_local_id())
                            throw std::invalid_argument(
                                "Mesh traverse requires the face neighbor in the local cell map.");
                });
        });
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<class Access, class Visitor>
inline void MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::visit_structured_batches(
    const Access& access, Visitor&& visitor, const detail::StructuredTraverseLayout& layout) const
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
                if (!selected(face))
                    continue;
                if (count && runs[count - 1].end == i && runs[count - 1].interior == face.interior &&
                    runs[count - 1].batch == face.batch)
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
                else if (runs[i].interior)
                    emit.template operator()<true>();
                else
                    emit.template operator()<false>();
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
            else
                emit_selected([](const auto&) { return true; });
        }
        else
            emit_selected([&](const auto& face) { return structured_selected(face.interior, face.batch); });
    }
    else
    {
        const auto emit_run = [&](const detail::StructuredTraverseRun& structured)
        {
            const detail::MeshTraverseRun run{structured.begin, structured.end, structured.interior, structured.batch};
            const auto emit = [&]<bool Interior>()
            {
                std::invoke(visitor, MeshTraverseBatch<Access, Entity, Interior, Incidence>(
                                         access, {&run, 1}, {}, {}, d_execution->d_plan));
            };
            if constexpr (Selection == MeshTraverseSelection::Interior)
                emit.template operator()<true>();
            else if constexpr (Selection == MeshTraverseSelection::Boundary)
                emit.template operator()<false>();
            else if (run.interior)
                emit.template operator()<true>();
            else
                emit.template operator()<false>();
        };
        if constexpr (Entity == MeshTraverseEntity::Cell)
            layout.visit_cells(structured_selection(), d_independent, emit_run);
        else
            layout.visit_faces(structured_selection(), d_independent, structured_batch(), emit_run);
    }
}

template<class Mesh, MeshTraverseEntity Entity, MeshTraverseSelection Selection, MeshTraverseOwnership Ownership,
    bool Incidence>
template<class Access, class Visitor>
inline void MeshBatchProxy<Mesh, Entity, Selection, Ownership, Incidence>::visit_layered_batches(
    const Access& access, Visitor&& visitor, const detail::LayeredTraverseLayout& layout) const
{
    if constexpr (Incidence)
    {
        const auto faces = layout.cell_faces(static_cast<size_t>(d_cell));
        faces.visit_runs(structured_selection(), d_independent, structured_batch(),
            [&](const detail::StructuredTraverseRun& run)
            {
                if constexpr (Selection == MeshTraverseSelection::Interior)
                    std::invoke(
                        visitor, LayeredIncidenceBatch<Access, true>(access, faces, run.begin, run.end, d_cell));
                else if constexpr (Selection == MeshTraverseSelection::Boundary)
                    std::invoke(
                        visitor, LayeredIncidenceBatch<Access, false>(access, faces, run.begin, run.end, d_cell));
                else if (run.interior)
                    std::invoke(
                        visitor, LayeredIncidenceBatch<Access, true>(access, faces, run.begin, run.end, d_cell));
                else
                    std::invoke(
                        visitor, LayeredIncidenceBatch<Access, false>(access, faces, run.begin, run.end, d_cell));
            });
    }
    else
    {
        const auto emit_run = [&](const detail::StructuredTraverseRun& source)
        {
            const detail::MeshTraverseRun run{source.begin, source.end, source.interior, source.batch};
            if constexpr (Selection == MeshTraverseSelection::Interior)
                std::invoke(
                    visitor, MeshTraverseBatch<Access, Entity, true>(access, {&run, 1}, {}, {}, d_execution->d_plan));
            else if constexpr (Selection == MeshTraverseSelection::Boundary)
                std::invoke(
                    visitor, MeshTraverseBatch<Access, Entity, false>(access, {&run, 1}, {}, {}, d_execution->d_plan));
            else if (source.interior)
                std::invoke(
                    visitor, MeshTraverseBatch<Access, Entity, true>(access, {&run, 1}, {}, {}, d_execution->d_plan));
            else
                std::invoke(
                    visitor, MeshTraverseBatch<Access, Entity, false>(access, {&run, 1}, {}, {}, d_execution->d_plan));
        };
        if constexpr (Entity == MeshTraverseEntity::Cell)
            layout.visit_cells(structured_selection(), d_independent, emit_run);
        else
            layout.visit_faces(structured_selection(), d_independent, structured_batch(), emit_run);
    }
}

} // namespace SimpleFluid
