#include "test_helpers.hh"

namespace
{
using IncidenceRunKey = std::tuple<size_t, size_t, bool, int>;

// Build the oracle only from public point queries. Retained XY runs and the
// arithmetic layout are deliberately absent from this classification.
std::vector<IncidenceRunKey> reference_incidence_runs(const Handle& reference,
    Local cell, detail::StructuredTraverseSelection selection, bool independent,
    std::optional<int> patch)
{
    struct Entry { size_t slot; bool interior; int patch; };
    std::vector<Entry> entries;
    size_t slot = 0;
    for (auto face : reference.faces(cell))
    {
        const bool interior = !reference.is_geometry_exterior_face(face);
        const int id = interior ? -1 : reference.boundary_id(face);
        const bool selected = selection == detail::StructuredTraverseSelection::All
            || (selection == detail::StructuredTraverseSelection::Interior) == interior;
        if (selected && (!patch || selection == detail::StructuredTraverseSelection::Interior || id == *patch))
            entries.push_back({slot, interior, id});
        ++slot;
    }
    if (independent)
        std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b)
        {
            if (a.interior != b.interior) return a.interior;
            return a.patch < b.patch;
        });
    std::vector<IncidenceRunKey> runs;
    for (const auto& entry : entries)
    {
        if (!runs.empty() && std::get<1>(runs.back()) == entry.slot
            && std::get<2>(runs.back()) == entry.interior && std::get<3>(runs.back()) == entry.patch)
            ++std::get<1>(runs.back());
        else runs.emplace_back(entry.slot, entry.slot + 1, entry.interior, entry.patch);
    }
    return runs;
}

template<class Incidences, class Proxy>
void expect_layered_incidence_runs(const Incidences& indexed, const Proxy& proxy,
    const Handle& reference, Local cell)
{
    using Selection = detail::StructuredTraverseSelection;
    std::vector<Local> face_ids;
    std::vector<std::optional<int>> patches{std::nullopt, 987654};
    for (auto face : reference.faces(cell))
    {
        face_ids.push_back(face);
        if (reference.is_geometry_exterior_face(face)) patches.push_back(reference.boundary_id(face));
    }
    std::sort(patches.begin(), patches.end());
    patches.erase(std::unique(patches.begin(), patches.end()), patches.end());
    auto check = [&](Selection selection, bool independent, std::optional<int> patch, const auto& selected_proxy)
    {
        const auto expected = reference_incidence_runs(reference, cell, selection, independent, patch);
        std::vector<IncidenceRunKey> actual;
        indexed.visit_runs(selection, independent, patch, [&](detail::StructuredTraverseRun& run)
        {
            actual.emplace_back(run.begin, run.end, run.interior, run.batch);
            // A nested visit must leave the outer descriptor unchanged.
            std::vector<IncidenceRunKey> nested;
            indexed.visit_runs(selection, independent, patch, [&](const auto& inner)
            { nested.emplace_back(inner.begin, inner.end, inner.interior, inner.batch); });
            EXPECT_EQ(nested, expected);
        });
        EXPECT_EQ(actual, expected);
        std::vector<std::vector<Local>> expected_batches, actual_batches;
        for (const auto& [begin, end, interior, batch] : expected)
            expected_batches.emplace_back(face_ids.begin() + begin, face_ids.begin() + end);
        selected_proxy.visit_batches([&](const auto& batch)
        {
            std::vector<Local> ids;
            batch.apply([&](const auto& face) { ids.push_back(face.id()); });
            EXPECT_EQ(batch.size(), ids.size());
            actual_batches.push_back(std::move(ids));
            using Ref = typename std::remove_cvref_t<decltype(batch)>::reference;
            batch.apply(&Ref::id); // Preserve member-pointer invocation.
        });
        EXPECT_EQ(actual_batches, expected_batches);
        std::vector<Local> move_only_ids;
        selected_proxy.apply([state = std::make_unique<unsigned>(0), &move_only_ids](auto&& face) mutable
        { ++*state; move_only_ids.push_back(face.id()); });
        std::vector<Local> flattened;
        for (const auto& batch : expected_batches) flattened.insert(flattened.end(), batch.begin(), batch.end());
        EXPECT_EQ(move_only_ids, flattened);
    };
    for (bool independent : {false, true})
    {
        const auto selected_proxy = independent ? proxy.independent() : proxy;
        check(Selection::All, independent, {}, selected_proxy);
        check(Selection::Interior, independent, {}, selected_proxy.interior());
        check(Selection::Boundary, independent, {}, selected_proxy.boundary());
        for (const auto patch : patches)
            if (patch) check(Selection::Boundary, independent, patch, selected_proxy.boundary(*patch));
    }
}

template<class Mesh>
void expect_layered_plan(const Mesh& mesh, const Handle& reference)
{
    const auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    ASSERT_TRUE(plan.uses_layered_ranges());
    EXPECT_FALSE(plan.uses_structured_ranges());
    ASSERT_TRUE(plan.layered_layout());
    const auto traverse = plan.acquire();
    std::vector<Local> interior_cells, boundary_cells, interior_faces, boundary_faces;
    std::map<int,std::vector<Local>> patches;
    size_t interior_incidences = 0, boundary_incidences = 0;
    for (const auto cell : ordinals(reference.num_owned_cells()))
    {
        bool boundary=false;
        size_t incidences = 0;
        for (auto face : reference.faces(cell))
        {
            boundary = boundary || reference.is_geometry_exterior_face(face);
            ++incidences;
        }
        (boundary ? boundary_cells : interior_cells).push_back(cell);
        (boundary ? boundary_incidences : interior_incidences) += incidences;
    }
    using Selection = detail::StructuredTraverseSelection;
    EXPECT_EQ(plan.layered_layout()->cell_incidence_count(Selection::All), interior_incidences + boundary_incidences);
    EXPECT_EQ(plan.layered_layout()->cell_incidence_count(Selection::Interior), interior_incidences);
    EXPECT_EQ(plan.layered_layout()->cell_incidence_count(Selection::Boundary), boundary_incidences);
    for (const auto face : ordinals(reference.num_owned_faces()))
    {
        const bool boundary=reference.is_geometry_exterior_face(face);
        (boundary ? boundary_faces : interior_faces).push_back(face);
        if (boundary) patches[reference.boundary_id(face)].push_back(face);
    }
    EXPECT_EQ(selected_ids(traverse.cells()),ordinals(reference.num_owned_cells()));
    EXPECT_EQ(selected_ids(traverse.cells().interior()),interior_cells);
    EXPECT_EQ(selected_ids(traverse.cells().boundary()),boundary_cells);
    EXPECT_EQ(traverse.cells().interior().size(),interior_cells.size());
    EXPECT_EQ(selected_ids(traverse.faces()),ordinals(reference.num_owned_faces()));
    EXPECT_EQ(selected_ids(traverse.faces().interior()),interior_faces);
    EXPECT_EQ(selected_ids(traverse.faces().boundary()),boundary_faces);
    std::vector<Local> grouped=interior_faces;
    for (const auto& [id, faces] : patches)
    {
        EXPECT_EQ(selected_ids(traverse.faces().boundary(id)),faces);
        EXPECT_EQ(traverse.faces().boundary(id).size(),faces.size());
        grouped.insert(grouped.end(),faces.begin(),faces.end());
    }
    EXPECT_EQ(selected_ids(traverse.faces().independent()),grouped);
    traverse.cells().apply([&](const auto& cell)
    {
        EXPECT_DOUBLE_EQ(cell.volume(),reference.cell_volume(cell.id()));
        expect_vector_equal(cell.centroid(),reference.cell_centroid(cell.id()));
        std::vector<Local> expected,inner;
        std::map<int,std::vector<Local>> boundary;
        for (auto face : reference.faces(cell.id()))
        {
            expected.push_back(face);
            if (reference.is_geometry_exterior_face(face)) boundary[reference.boundary_id(face)].push_back(face);
            else inner.push_back(face);
        }
        const auto incidence=traverse.cell_faces(cell.id());
        const bool fully_interior = boundary.empty();
        EXPECT_EQ(plan.layered_layout()->cell_faces(cell.id()).all_interior(), fully_interior);
        if (fully_interior)
        {
            size_t batches = 0;
            incidence.interior().independent().visit_batches([&](const auto& batch)
            { ++batches; EXPECT_EQ(batch.size(), expected.size()); });
            EXPECT_EQ(batches, 1U);
            EXPECT_TRUE(selected_ids(incidence.boundary()).empty());
            EXPECT_TRUE(selected_ids(incidence.boundary(987654).independent()).empty());
        }
        // Existing batch apply passes a temporary reference. A constrained
        // rvalue-only callable catches accidental lvalue dispatch for both
        // native and canonical Handle layered incidence.
        std::vector<Local> rvalue_faces;
        const auto rvalue_only = [&]<class Ref>(Ref&& face)
            requires (!std::is_lvalue_reference_v<Ref>)
        { rvalue_faces.push_back(face.id()); };
        incidence.apply(rvalue_only);
        EXPECT_EQ(rvalue_faces, expected);
        rvalue_faces.clear();
        incidence.apply(cell.geometry(), rvalue_only);
        EXPECT_EQ(rvalue_faces, expected);
        EXPECT_EQ(selected_ids(incidence),expected);
        EXPECT_EQ(incidence.size(),expected.size());
        EXPECT_EQ(plan.layered_layout()->cell_faces(cell.id()).size(),expected.size());
        const auto indexed_faces = plan.layered_layout()->cell_faces(cell.id());
        expect_layered_incidence_runs(indexed_faces, incidence, reference, static_cast<Local>(cell.id()));
        for (size_t slot = 0; slot < expected.size(); ++slot)
        {
            EXPECT_EQ(indexed_faces.face_ordinal(slot), static_cast<size_t>(expected[slot]));
            using Access = std::remove_cvref_t<decltype(cell.geometry())>;
            if constexpr (detail::is_layered_traverse_access<Access>)
            {
                const auto resolved = cell.geometry().resolve_incidence(indexed_faces, slot);
                const auto face = expected[slot];
                const auto owner = reference.owner_cell(face);
                const auto neighbor = reference.neighbor_cell(face);
                const auto& indexer = plan.layered_layout()->indexer();
                EXPECT_EQ(resolved.incidence.face, static_cast<size_t>(face));
                EXPECT_EQ(resolved.incidence.orientation, owner == cell.id() ? 1 : -1);
                EXPECT_EQ(resolved.incidence.interior, !reference.is_geometry_exterior_face(face));
                EXPECT_EQ(resolved.incidence.batch, resolved.incidence.interior ? -1 : reference.boundary_id(face));
                EXPECT_EQ(resolved.owner, static_cast<size_t>(owner));
                EXPECT_EQ(resolved.neighbor, neighbor == Handle::invalid_local_id()
                    ? std::numeric_limits<size_t>::max() : static_cast<size_t>(neighbor));
                EXPECT_EQ(resolved.cell, indexer.cell_id(cell.id()));
                const auto other = owner == cell.id() ? neighbor : owner;
                EXPECT_EQ(resolved.other, other == Handle::invalid_local_id()
                    ? Meshes::SemiStructuredIndexer::CellID{} : indexer.cell_id(other));
            }
        }
        incidence.apply(cell.geometry(),[&](const auto& face)
        {
            EXPECT_EQ(face.orientation(),reference.owner_cell(face.id())==cell.id() ? 1 : -1);
            EXPECT_EQ(face.owner(),reference.owner_cell(face.id()));
            using Ref = std::remove_cvref_t<decltype(face)>;
            using Access = std::remove_cvref_t<decltype(face.geometry())>;
            if constexpr (Ref::is_interior)
            {
                EXPECT_EQ(face.neighbor(),reference.neighbor_cell(face.id()));
                if constexpr (detail::is_layered_traverse_access<Access>)
                {
                    static_assert(requires { face.cell_center_vector(); });
                    expect_vector_equal(face.cell_center_vector(),reference.cell_center_vector(face.id(),cell.id()));
                }
            }
            EXPECT_DOUBLE_EQ(face.area(),reference.face_area(face.id()));
            expect_vector_equal(face.centroid(),reference.face_centroid(face.id()));
            expect_vector_equal(face.normal(),reference.face_normal(face.id()));
            // Nested borrowed incidence must not overwrite shared scratch.
            EXPECT_EQ(selected_ids(traverse.cell_faces(cell.id())),expected);
        });
        for (const auto& [id,faces] : boundary) inner.insert(inner.end(),faces.begin(),faces.end());
        EXPECT_EQ(selected_ids(incidence.independent()),inner);
        std::vector<Local> grouped_boundary;
        for (const auto& [id,faces] : boundary) grouped_boundary.insert(grouped_boundary.end(),faces.begin(),faces.end());
        EXPECT_EQ(selected_ids(incidence.boundary().independent()),grouped_boundary);
    });
}
}

TEST(MeshTraverseTest, LayeredPolygonIncidenceAndPatchOrder)
{
    using Layered=Meshes::SemiStructuredXY_Z;
    for (unsigned degree : {3U,4U,7U,300U})
        for (unsigned layers : {1U,2U,5U})
        {
            Arr<Layered::Vec3> nodes; Arr<unsigned> polygon; Arr<Layered::BoundaryEdge> edges;
            for (unsigned i=0;i<degree;++i)
            {
                const double angle=2*std::acos(-1.0)*i/degree;
                nodes.push_back({std::cos(angle),std::sin(angle),0}); polygon.push_back(i);
                edges.push_back({i,(i+1)%degree,"patch"+std::to_string((i/2)%11)});
            }
            ArrReal z{0}; for (unsigned k=0;k<layers;++k) z.push_back(z.back()+0.5+k*0.125);
            auto native=std::make_shared<Layered>(nodes,Arr<Arr<unsigned>>{polygon},z,edges);
            const Handle handle(native);
            expect_layered_plan(*native,handle); expect_layered_plan(handle,handle);
        }
}

TEST(MeshTraverseTest, LayeredMixedCellsAndStorageIndependentOfLayers)
{
    using Layered=Meshes::SemiStructuredXY_Z;
    const Arr<Layered::Vec3> nodes{{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0}};
    const Arr<Arr<unsigned>> cells{{0,1,4,3},{1,2,4},{2,5,4}};
    size_t storage=0;
    for (unsigned layers : {1U,2U,8U,100U})
    {
        ArrReal z{0}; for (unsigned k=0;k<layers;++k) z.push_back(z.back()+1.0+k*0.01);
        auto native=std::make_shared<Layered>(nodes,cells,z);
        const Handle handle(native);
        const auto plan=make_mesh_traverse_plan(*native,MeshTraverseEntities::AllWithCellFaces);
        if (storage==0) storage=plan.storage_bytes();
        EXPECT_EQ(plan.storage_bytes(),storage);
        if (layers<100) { expect_layered_plan(*native,handle); expect_layered_plan(handle,handle); }
    }
    // A three-by-three XY grid has a physical interior prism on inner Z layers.
    Arr<Layered::Vec3> grid; Arr<Arr<unsigned>> quads;
    for (unsigned j=0;j<4;++j) for (unsigned i=0;i<4;++i) grid.push_back({double(i),double(j),0});
    for (unsigned j=0;j<3;++j) for (unsigned i=0;i<3;++i) { auto b=i+4*j; quads.push_back({b,b+1,b+5,b+4}); }
    auto native=std::make_shared<Layered>(grid,quads,ArrReal{0,0.5,2,4});
    const Handle handle(native); expect_layered_plan(*native,handle); expect_layered_plan(handle,handle);
}

TEST(MeshTraverseTest, ReorderedLayeredHandleRetainsCanonicalFallback)
{
    using Layered=Meshes::SemiStructuredXY_Z;
    auto native=std::make_shared<Layered>(
        Arr<Layered::Vec3>{{0,0,0},{1,0,0},{1,1,0},{0,1,0}},
        Arr<Arr<unsigned>>{{0,1,3},{1,2,3}},ArrReal{0,1,3});
    auto handle=std::make_shared<Handle>(native);
    auto reordered=MeshReorderingFactory<Pack>::selected_cells_first(
        std::move(handle),[](auto,const auto& center){return center.z>1;});
    const auto plan=make_mesh_traverse_plan(*reordered.mesh,MeshTraverseEntities::AllWithCellFaces);
    EXPECT_FALSE(plan.uses_layered_ranges()); EXPECT_FALSE(plan.uses_structured_ranges());
    expect_handle_selections(*reordered.mesh);
    const auto traverse=plan.acquire();
    traverse.cells().apply([&](const auto& cell)
    {
        std::vector<Local> expected;
        for (auto face : reordered.mesh->faces(cell.id())) expected.push_back(face);
        EXPECT_EQ(selected_ids(traverse.cell_faces(cell.id())),expected);
    });
}

TEST(MeshTraverseTest, LayeredIncidenceRunsMergeAxialPrefixWithFirstSideRun)
{
    using Layered = Meshes::SemiStructuredXY_Z;
    // Start the second polygon at its shared edge. On the bottom Z layer,
    // upper axial incidence and first side incidence form one interior run.
    const Arr<Layered::Vec3> nodes{{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0}};
    const Arr<Arr<unsigned>> cells{{0,1,4,3},{4,1,2,5}};
    size_t native_storage = 0, handle_storage = 0;
    for (unsigned layers : {1U,2U,7U,40U})
    {
        ArrReal z{0};
        for (unsigned k = 0; k < layers; ++k) z.push_back(z.back() + 0.25 + k*0.03);
        auto native = std::make_shared<Layered>(nodes, cells, z);
        const Handle reference(native);
        const auto native_plan = make_mesh_traverse_plan(*native, MeshTraverseEntities::AllWithCellFaces);
        const auto handle_plan = make_mesh_traverse_plan(reference, MeshTraverseEntities::AllWithCellFaces);
        if (native_storage == 0)
        { native_storage = native_plan.storage_bytes(); handle_storage = handle_plan.storage_bytes(); }
        EXPECT_EQ(native_plan.storage_bytes(), native_storage);
        EXPECT_EQ(handle_plan.storage_bytes(), handle_storage);
        expect_layered_plan(*native, reference);
        expect_layered_plan(reference, reference);
        if (layers > 1)
        {
            const auto expected = reference_incidence_runs(reference, 1,
                detail::StructuredTraverseSelection::All, false, {});
            ASSERT_GE(expected.size(), 2U);
            EXPECT_EQ(expected[1], (IncidenceRunKey{1, 3, true, -1}));
        }
    }
}

TEST(MeshTraverseTest, LayeredTopologyPeriodicIncidenceRuns)
{
    using Topology = Meshes::SemiStructMeshTopo;
    using Selection = detail::StructuredTraverseSelection;
    const Arr<Arr<unsigned>> cells{{0,1,4,3},{4,1,2,5}};
    size_t storage = 0;
    for (unsigned layers : {1U,2U,5U})
    {
        const Topology topology(6, cells, layers, {}, true);
        const detail::LayeredTraverseLayout layout(topology);
        if (storage == 0) storage = layout.storage_bytes();
        EXPECT_EQ(layout.storage_bytes(), storage);
        const auto& indexer = topology.indexer();
        for (size_t ordinal = 0; ordinal < indexer.total_cells(); ++ordinal)
        {
            const auto cell = indexer.cell_id(ordinal);
            const auto incidences = layout.cell_faces(ordinal);
            struct Entry { size_t slot; bool interior; int batch; };
            std::vector<Entry> entries;
            std::vector<std::optional<int>> patches{std::nullopt, 987654};
            size_t slot = 0;
            for (auto face : topology.cell_faces(cell))
            {
                const bool interior = topology.neighbor_cell(face) != Topology::CellID{};
                const int batch = interior ? -1 : topology.boundary_id(face);
                const auto actual = incidences[slot];
                const auto resolved = incidences.resolved(slot, topology);
                const auto owner = topology.owner_cell(face), neighbor = topology.neighbor_cell(face);
                EXPECT_EQ(resolved.incidence.face, actual.face);
                EXPECT_EQ(resolved.incidence.orientation, actual.orientation);
                EXPECT_EQ(resolved.incidence.interior, actual.interior);
                EXPECT_EQ(resolved.incidence.batch, actual.batch);
                EXPECT_EQ(resolved.owner, indexer.cell_ordinal(owner));
                EXPECT_EQ(resolved.neighbor, neighbor == Topology::CellID{}
                    ? std::numeric_limits<size_t>::max() : indexer.cell_ordinal(neighbor));
                EXPECT_EQ(resolved.cell, cell);
                EXPECT_EQ(resolved.other, owner == cell ? neighbor : owner);
                EXPECT_EQ(actual.face, indexer.face_ordinal(face));
                EXPECT_EQ(incidences.face_ordinal(slot), indexer.face_ordinal(face));
                EXPECT_EQ(actual.orientation, topology.owner_cell(face) == cell ? 1 : -1);
                EXPECT_EQ(actual.interior, interior);
                EXPECT_EQ(actual.interior, !topology.is_boundary_face(face));
                EXPECT_EQ(actual.batch, batch);
                entries.push_back({slot++, interior, batch});
                if (!interior) patches.push_back(batch);
            }
            EXPECT_EQ(incidences.size(), entries.size());
            if (layers == 1)
            {
                EXPECT_EQ(incidences[0].face, incidences[1].face);
                EXPECT_EQ(incidences[0].orientation, 1);
                EXPECT_EQ(incidences[1].orientation, 1);
            }
            std::sort(patches.begin(), patches.end());
            patches.erase(std::unique(patches.begin(), patches.end()), patches.end());
            for (Selection selection : {Selection::All, Selection::Interior, Selection::Boundary})
                for (bool independent : {false, true})
                    for (auto patch : patches)
                    {
                        if (selection != Selection::Boundary && patch) continue;
                        std::vector<Entry> selected;
                        for (const auto entry : entries)
                            if ((selection == Selection::All || (selection == Selection::Interior) == entry.interior)
                                && (!patch || entry.batch == *patch)) selected.push_back(entry);
                        if (independent)
                            std::stable_sort(selected.begin(), selected.end(), [](const Entry& a, const Entry& b)
                            {
                                if (a.interior != b.interior) return a.interior;
                                return a.batch < b.batch;
                            });
                        std::vector<IncidenceRunKey> expected;
                        for (const auto entry : selected)
                        {
                            if (!expected.empty() && std::get<1>(expected.back()) == entry.slot
                                && std::get<2>(expected.back()) == entry.interior && std::get<3>(expected.back()) == entry.batch)
                                ++std::get<1>(expected.back());
                            else expected.emplace_back(entry.slot, entry.slot + 1, entry.interior, entry.batch);
                        }
                        std::vector<IncidenceRunKey> actual;
                        incidences.visit_runs(selection, independent, patch, [&](const auto& run)
                        { actual.emplace_back(run.begin, run.end, run.interior, run.batch); });
                        EXPECT_EQ(actual, expected);
                    }
        }
    }
}

TEST(MeshTraverseTest, LayeredFullyInteriorPeriodicRowsProduceOneRun)
{
    using Topology = Meshes::SemiStructMeshTopo;
    using Selection = detail::StructuredTraverseSelection;
    Arr<Arr<unsigned>> cells;
    for (unsigned y = 0; y < 3; ++y)
        for (unsigned x = 0; x < 3; ++x)
        { const auto p = x + 4*y; cells.push_back({p, p+1, p+5, p+4}); }
    for (unsigned layers : {1U, 2U, 5U})
    {
        const Topology topology(16, cells, layers, {}, true);
        const detail::LayeredTraverseLayout layout(topology);
        for (unsigned layer = 0; layer < layers; ++layer)
        {
            const auto cell = layout.indexer().cell_ordinal({4, layer});
            const auto faces = layout.cell_faces(cell);
            ASSERT_TRUE(faces.all_interior());
            for (bool independent : {false, true})
                for (auto selection : {Selection::All, Selection::Interior})
                {
                    std::vector<IncidenceRunKey> runs;
                    faces.visit_runs(selection, independent, {}, [&](const auto& run)
                    { runs.emplace_back(run.begin, run.end, run.interior, run.batch); });
                    EXPECT_EQ(runs, (std::vector<IncidenceRunKey>{{0, 6, true, -1}}));
                }
            for (auto patch : {std::optional<int>{}, std::optional<int>{987654}})
            {
                size_t batches = 0;
                faces.visit_runs(Selection::Boundary, true, patch, [&](const auto&) { ++batches; });
                EXPECT_EQ(batches, 0U);
            }
            if (layers == 1)
            {
                EXPECT_EQ(faces[0].face, faces[1].face);
                EXPECT_EQ(faces[0].orientation, 1);
                EXPECT_EQ(faces[1].orientation, 1);
            }
        }
    }
}

