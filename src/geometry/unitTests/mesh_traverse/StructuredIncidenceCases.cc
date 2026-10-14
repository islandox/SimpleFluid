#include "test_helpers.hh"

namespace
{
template<class Mesh>
void expect_structured_typed_cells(const Mesh& mesh, const Handle& reference)
{
    const auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    const auto foreign_plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    ASSERT_TRUE(plan.uses_structured_ranges());
    const auto traversal = plan.acquire();
    const auto another_execution = plan.acquire();
    const auto foreign = foreign_plan.acquire();
    size_t direct_cells = 0;
    traversal.cells().apply([&](const auto& cell)
    {
        using Cell = std::remove_cvref_t<decltype(cell)>;
        using Access = std::remove_cvref_t<decltype(cell.geometry())>;
        const auto typed = traversal.cell_faces(cell);
        const auto by_id = traversal.cell_faces(cell.id());
        EXPECT_NO_THROW(typed.require_local_adjacency());
        std::vector<Local> expected;
        for (const auto face : reference.faces(cell.id())) expected.push_back(face);
        EXPECT_EQ(selected_ids(typed), expected);
        if constexpr (Cell::is_interior && detail::is_structured_traverse_access<Access>)
        {
            ++direct_cells;
            static_assert(std::remove_cvref_t<decltype(typed)>::is_structured_interior_range);
            EXPECT_EQ(typed.size(), 6U);
            EXPECT_FALSE(typed.empty());
            EXPECT_EQ(selected_ids(typed.all()), expected);
            EXPECT_EQ(selected_ids(typed.interior()), expected);
            EXPECT_EQ(selected_ids(typed.independent()), expected);
            EXPECT_EQ(selected_ids(by_id), expected);
            EXPECT_EQ(selected_ids(another_execution.cell_faces(cell)), expected);
            EXPECT_EQ(typed.boundary().size(), 0U);
            EXPECT_TRUE(typed.boundary(987654).empty());
            EXPECT_NO_THROW(typed.boundary().require_local_adjacency());
            EXPECT_NO_THROW(typed.boundary(987654).require_local_adjacency());
            size_t callbacks = 0;
            typed.boundary().apply([&](const auto&) { ++callbacks; });
            typed.boundary(987654).visit_batches([&](const auto&) { ++callbacks; });
            typed.boundary().apply(cell.geometry(), [&](const auto&) { ++callbacks; });
            typed.boundary(987654).visit_batches(cell.geometry(), [&](const auto&) { ++callbacks; });
            EXPECT_EQ(callbacks, 0U);
            size_t batches = 0;
            typed.visit_batches([&]<class Batch>(Batch&& batch) requires (!std::is_lvalue_reference_v<Batch>)
            {
                using Range = std::remove_cvref_t<Batch>;
                static_assert(Range::is_interior);
                using Ref = typename Range::reference;
                ++batches;
                EXPECT_EQ(batch.size(), 6U);
                EXPECT_EQ(selected_ids(batch), expected);
                batch.apply(&Ref::id);
            });
            EXPECT_EQ(batches, 1U);
            using Range = std::remove_cvref_t<decltype(typed)>;
            typed.visit_batches(&Range::size);
            std::vector<Local> ids;
            typed.apply([&]<class Ref>(Ref&& face) requires (!std::is_lvalue_reference_v<Ref>)
            { ids.push_back(face.id()); });
            EXPECT_EQ(ids, expected);
            ids.clear();
            typed.apply([state = std::make_unique<unsigned>(0), &ids](auto&& face) mutable
            { ++*state; ids.push_back(face.id()); });
            EXPECT_EQ(ids, expected);
            typed.apply(cell.geometry(), [&](const auto& face)
            {
                static_assert(std::remove_cvref_t<decltype(face)>::is_interior);
                EXPECT_EQ(face.cell_id(), cell.id());
                EXPECT_EQ(face.owner(), reference.owner_cell(face.id()));
                EXPECT_EQ(face.neighbor(), reference.neighbor_cell(face.id()));
                EXPECT_EQ(face.orientation(), reference.owner_cell(face.id()) == cell.id() ? 1 : -1);
                EXPECT_DOUBLE_EQ(face.area(), reference.face_area(face.id()));
                expect_vector_equal(face.centroid(), reference.face_centroid(face.id()));
                expect_vector_equal(face.outward_normal(), reference.face_normal_outward(face.id(), cell.id()));
                expect_vector_equal(face.geometry().cell_center_vector(face.id(), cell.id()),
                    reference.cell_center_vector(face.id(), cell.id()));
            });
        }
        EXPECT_THROW(static_cast<void>(foreign.cell_faces(cell)), std::invalid_argument);
        const Cell forged(cell.geometry(), cell.id());
        EXPECT_THROW(static_cast<void>(traversal.cell_faces(forged)), std::invalid_argument);
    });
    EXPECT_EQ(direct_cells, traversal.cells().interior().size());
    const auto cells_only = make_mesh_traverse_plan(mesh, MeshTraverseEntities::Cells);
    const auto cell_execution = cells_only.acquire();
    cell_execution.cells().apply([&](const auto& cell)
    { EXPECT_THROW(static_cast<void>(cell_execution.cell_faces(cell)), std::logic_error); });
}
}

TEST(MeshTraverseTest, StructuredTypedInteriorCellsPreservePeriodicThinAndSingleCellIncidences)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    for (const auto shape : {std::array<size_t, 3>{3, 4, 3}, {1, 1, 1}, {1, 3, 3},
                            {3, 1, 3}, {3, 3, 1}, {2, 3, 3}})
        for (unsigned mask = 0; mask < 8; ++mask)
        {
            SCOPED_TRACE(::testing::PrintToString(shape));
            SCOPED_TRACE(mask);
            Vec3D<ArrReal> edges;
            for (size_t axis = 0; axis < 3; ++axis)
                for (size_t i = 0; i <= shape[axis]; ++i)
                    edges[axis].push_back(real_t(i) + real_t(0.1) * real_t(i * i));
            const Vec3D<bool> periodic{{bool(mask & 1U), bool(mask & 2U), bool(mask & 4U)}};
            const bool unsupported = (shape[0] == 1 && periodic[0]) || (shape[1] == 1 && periodic[1])
                || (shape[2] == 1 && periodic[2]);
            if (unsupported)
            {
                EXPECT_THROW(static_cast<void>(std::make_shared<Cartesian>(edges, periodic)), std::invalid_argument);
                continue;
            }
            const auto geometry = std::make_shared<Cartesian>(edges, periodic);
            const Handle mesh(geometry);
            expect_structured_typed_cells(*geometry, mesh);
            expect_structured_typed_cells(mesh, mesh);
        }
}

TEST(MeshTraverseTest, StructuredLayoutSyntheticSingleCellPeriodicIncidenceArithmetic)
{
    // The indexer/layout permit this topology for arithmetic checks; the public
    // Cartesian geometry provider requires at least two cells per periodic axis.
    const Meshes::OrthogonalIndexer indexer(1, 1, 1, true, true, true);
    const detail::StructuredTraverseLayout layout(indexer);
    EXPECT_EQ(layout.cell_count(detail::StructuredTraverseSelection::Interior), 1U);
    EXPECT_EQ(layout.face_count(detail::StructuredTraverseSelection::Interior), 3U);
    const auto faces = layout.cell_faces(0);
    for (size_t axis = 0; axis < 3; ++axis)
        for (size_t side = 0; side < 2; ++side)
        {
            const auto& face = faces[2 * axis + side];
            EXPECT_EQ(face.face, axis);
            EXPECT_EQ(face.orientation, 1);
            EXPECT_TRUE(face.interior);
            EXPECT_EQ(face.batch, -1);
        }
}

TEST(MeshTraverseTest, StructuredTypedInteriorCellsPreserveCylindricalSectorAndPeriodicGeometry)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    const auto pi = std::numbers::pi_v<real_t>;
    for (const auto& theta : {ArrReal{0, 0.2, 0.7, 1.5}, ArrReal{0, pi, 2 * pi},
                             ArrReal{0, pi / 2, pi, 3 * pi / 2, 2 * pi}})
    {
        const auto geometry = std::make_shared<Cylindrical>(
            Vec3D<ArrReal>{{{1, 2, 4, 7}, theta, {0, 0.5, 2, 3}}});
        const Handle mesh(geometry);
        expect_structured_typed_cells(*geometry, mesh);
        expect_structured_typed_cells(mesh, mesh);
    }
}

TEST(MeshTraverseTest, StructuredTypedInteriorCellProofRejectsForeignMeshAndBoundAccess)
{
    const auto native = cartesian(3, 3, 3), other_native = cartesian(3, 3, 3);
    const auto plan = make_mesh_traverse_plan(*native, MeshTraverseEntities::AllWithCellFaces);
    const auto other_plan = make_mesh_traverse_plan(*other_native, MeshTraverseEntities::AllWithCellFaces);
    const auto traversal = plan.acquire(), other = other_plan.acquire();
    traversal.cells().interior().apply([&](const auto& cell)
    {
        const auto range = traversal.cell_faces(cell);
        other.cells().interior().apply([&](const auto& other_cell)
        {
            EXPECT_THROW(static_cast<void>(traversal.cell_faces(other_cell)), std::invalid_argument);
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(cell.geometry())>,
                              std::remove_cvref_t<decltype(other_cell.geometry())>>)
            {
                size_t callbacks = 0;
                EXPECT_THROW(range.apply(other_cell.geometry(), [&](const auto&) { ++callbacks; }),
                    std::invalid_argument);
                EXPECT_THROW(range.visit_batches(other_cell.geometry(), [&](const auto&) { ++callbacks; }),
                    std::invalid_argument);
                EXPECT_THROW(range.boundary().apply(other_cell.geometry(), [&](const auto&) { ++callbacks; }),
                    std::invalid_argument);
                EXPECT_THROW(range.boundary().visit_batches(other_cell.geometry(), [&](const auto&) { ++callbacks; }),
                    std::invalid_argument);
                EXPECT_EQ(callbacks, 0U);
            }
        });
    });
}

TEST(MeshTraverseTest, StructuredTypedInteriorRangeRejectsInvalidationAndBackendReplacement)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Requires serial identity maps.";
    for (const bool replace_backend : {false, true})
    {
        Handle mesh(cartesian(3, 3, 3));
        auto plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
        const auto traversal = plan.acquire();
        size_t checked = 0;
        traversal.cells().interior().apply([&](const auto& cell)
        {
            using Access = std::remove_cvref_t<decltype(cell.geometry())>;
            if constexpr (detail::is_structured_traverse_access<Access>)
            {
                ++checked;
                const auto range = traversal.cell_faces(cell);
                if (replace_backend) mesh = Handle(cartesian(3, 3, 3));
                else plan.invalidate();
                EXPECT_THROW(static_cast<void>(range.size()), std::logic_error);
                EXPECT_THROW(static_cast<void>(range.empty()), std::logic_error);
                EXPECT_THROW(range.require_local_adjacency(), std::logic_error);
                EXPECT_THROW(range.boundary().require_local_adjacency(), std::logic_error);
                EXPECT_THROW(static_cast<void>(range.boundary(987654).size()), std::logic_error);
                size_t callbacks = 0;
                EXPECT_THROW(range.apply([&](const auto&) { ++callbacks; }), std::logic_error);
                EXPECT_THROW(range.visit_batches([&](const auto&) { ++callbacks; }), std::logic_error);
                EXPECT_THROW(range.boundary().apply([&](const auto&) { ++callbacks; }), std::logic_error);
                EXPECT_THROW(range.boundary().visit_batches([&](const auto&) { ++callbacks; }), std::logic_error);
                EXPECT_EQ(callbacks, 0U);
            }
        });
        EXPECT_EQ(checked, 1U);
    }
}
