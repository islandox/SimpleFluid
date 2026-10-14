/** @file LayeredIncidenceCases.cc
 * @brief Layered cell-face incidence geometry and orientation checks.
 */
#include "test_helpers.hh"

namespace
{
auto layered_interior_fixture(unsigned layers)
{
    using Layered = Meshes::SemiStructuredXY_Z;
    Arr<Layered::Vec3> nodes;
    Arr<Arr<unsigned>> cells;
    for (unsigned j=0;j<4;++j) for (unsigned i=0;i<4;++i) nodes.push_back({double(i),double(j),0});
    for (unsigned j=0;j<3;++j) for (unsigned i=0;i<3;++i)
    { const auto b=i+4*j; cells.push_back({b,b+1,b+5,b+4}); }
    ArrReal z{0};
    for (unsigned k=0;k<layers;++k) z.push_back(z.back()+0.2+k*0.07);
    return std::make_shared<Layered>(nodes,cells,z);
}

template<class Mesh>
void expect_layered_typed_cells(const Mesh& mesh)
{
    const auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto foreign_plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto traversal=plan.acquire();
    const auto another_execution=plan.acquire();
    const auto foreign=foreign_plan.acquire();
    traversal.cells().apply([&](const auto& cell)
    {
        using Cell=std::remove_cvref_t<decltype(cell)>;
        using Access=std::remove_cvref_t<decltype(cell.geometry())>;
        const auto typed=traversal.cell_faces(cell);
        const auto by_id=traversal.cell_faces(cell.id());
        EXPECT_NO_THROW(typed.require_local_adjacency());
        EXPECT_NO_THROW(by_id.require_local_adjacency());
        if constexpr (Cell::is_interior && detail::is_layered_traverse_access<Access>)
        {
            static_assert(std::remove_cvref_t<decltype(typed)>::is_layered_interior_range);
            EXPECT_EQ(selected_ids(typed),selected_ids(by_id));
            EXPECT_EQ(selected_ids(typed.interior()),selected_ids(by_id));
            EXPECT_EQ(selected_ids(typed.independent()),selected_ids(by_id));
            EXPECT_EQ(typed.size(),by_id.size());
            EXPECT_EQ(typed.boundary().size(),0U);
            EXPECT_EQ(typed.boundary(987654).size(),0U);
            EXPECT_NO_THROW(typed.boundary().require_local_adjacency());
            EXPECT_NO_THROW(typed.boundary(987654).require_local_adjacency());
            size_t empty_callbacks=0;
            typed.boundary().apply([&](const auto&){++empty_callbacks;});
            typed.boundary(987654).visit_batches([&](const auto&){++empty_callbacks;});
            EXPECT_EQ(empty_callbacks,0U);
            size_t batches=0;
            typed.visit_batches([&](const auto& batch)
            {
                using Batch=std::remove_cvref_t<decltype(batch)>;
                static_assert(Batch::is_interior);
                using Ref=typename Batch::reference;
                ++batches;
                EXPECT_EQ(batch.size(),by_id.size());
                batch.apply(&Ref::id);
            });
            EXPECT_EQ(batches,1U);
            std::vector<Local> ids;
            typed.apply([&]<class Ref>(Ref&& face) requires (!std::is_lvalue_reference_v<Ref>)
            { ids.push_back(face.id()); });
            EXPECT_EQ(ids,selected_ids(by_id));
            ids.clear();
            typed.apply([state=std::make_unique<unsigned>(0),&ids](auto&& face) mutable
            { ++*state; ids.push_back(face.id()); });
            EXPECT_EQ(ids,selected_ids(by_id));
            std::vector<std::pair<Local,int>> expected,actual;
            by_id.apply([&](const auto& face){expected.emplace_back(face.id(),face.orientation());});
            typed.apply([&](const auto& face)
            {
                actual.emplace_back(face.id(),face.orientation());
                using Ref = std::remove_cvref_t<decltype(face)>;
                static_assert(Ref::is_interior);
                static_assert(requires { face.cell_center_vector(); });
                EXPECT_EQ(face.owner(),face.geometry().owner_cell(face.id()));
                EXPECT_EQ(face.neighbor(),face.geometry().neighbor_cell(face.id()));
                expect_vector_equal(face.cell_center_vector(),face.geometry().cell_center_vector(face.id(),cell.id()));
                EXPECT_EQ(selected_ids(another_execution.cell_faces(cell)),selected_ids(by_id));
                EXPECT_EQ(selected_ids(traversal.cell_faces(cell)),selected_ids(by_id));
            });
            EXPECT_EQ(actual,expected);
        }
        else EXPECT_EQ(selected_ids(typed),selected_ids(by_id));
        EXPECT_THROW(static_cast<void>(foreign.cell_faces(cell)),std::invalid_argument);
        const Cell forged(cell.geometry(),cell.id());
        EXPECT_THROW(static_cast<void>(traversal.cell_faces(forged)),std::invalid_argument);
    });
    const auto cells_only=make_mesh_traverse_plan(mesh,MeshTraverseEntities::Cells);
    const auto cell_execution=cells_only.acquire();
    cell_execution.cells().apply([&](const auto& cell)
    { EXPECT_THROW(static_cast<void>(cell_execution.cell_faces(cell)),std::logic_error); });
}
}

TEST(MeshTraverseTest, LayeredTypedInteriorCellProofPreservesIncidenceContracts)
{
    for (unsigned layers : {1U,2U,3U,7U})
    {
        const auto native=layered_interior_fixture(layers);
        const Handle handle(native);
        expect_layered_typed_cells(*native);
        expect_layered_typed_cells(handle);
    }
}

TEST(MeshTraverseTest, LayeredTypedCellProofRejectsForeignMesh)
{
    const auto native=layered_interior_fixture(3);
    const auto other_native=layered_interior_fixture(3);
    const Handle mesh(native),other_mesh(other_native);
    const auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto other_plan=make_mesh_traverse_plan(other_mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto traversal=plan.acquire(),other=other_plan.acquire();
    other.cells().apply([&](const auto& cell)
    { EXPECT_THROW(static_cast<void>(traversal.cell_faces(cell)),std::invalid_argument); });
}

TEST(MeshTraverseTest, LayeredTypedInteriorRangeRejectsInvalidatedPlan)
{
    const auto native=layered_interior_fixture(3);
    const Handle mesh(native);
    auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto traversal=plan.acquire();
    traversal.cells().interior().apply([&](const auto& cell)
    {
        using Access=std::remove_cvref_t<decltype(cell.geometry())>;
        if constexpr (detail::is_layered_traverse_access<Access>)
        {
            const auto range=traversal.cell_faces(cell);
            plan.invalidate();
            EXPECT_THROW(static_cast<void>(range.size()),std::logic_error);
            EXPECT_THROW(range.require_local_adjacency(),std::logic_error);
            EXPECT_THROW(range.boundary().require_local_adjacency(),std::logic_error);
            EXPECT_THROW(range.boundary(987654).require_local_adjacency(),std::logic_error);
            EXPECT_THROW(static_cast<void>(range.empty()),std::logic_error);
            size_t callbacks=0;
            EXPECT_THROW(range.apply([&](const auto&){++callbacks;}),std::logic_error);
            EXPECT_THROW(range.visit_batches([&](const auto&){++callbacks;}),std::logic_error);
            EXPECT_THROW(static_cast<void>(range.boundary().size()),std::logic_error);
            EXPECT_EQ(callbacks,0U);
        }
    });
}

TEST(MeshTraverseTest, LayeredTypedInteriorRangeRejectsBackendReplacement)
{
    const auto native=layered_interior_fixture(3);
    Handle mesh(native);
    const auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto traversal=plan.acquire();
    traversal.cells().interior().apply([&](const auto& cell)
    {
        using Access=std::remove_cvref_t<decltype(cell.geometry())>;
        if constexpr (detail::is_layered_traverse_access<Access>)
        {
            const auto range=traversal.cell_faces(cell);
            mesh=Handle(layered_interior_fixture(3));
            EXPECT_THROW(static_cast<void>(range.size()),std::logic_error);
            EXPECT_THROW(range.require_local_adjacency(),std::logic_error);
            EXPECT_THROW(range.boundary().require_local_adjacency(),std::logic_error);
            EXPECT_THROW(range.boundary(987654).require_local_adjacency(),std::logic_error);
            size_t callbacks=0;
            EXPECT_THROW(range.apply([&](const auto&){++callbacks;}),std::logic_error);
            EXPECT_THROW(range.visit_batches([&](const auto&){++callbacks;}),std::logic_error);
            EXPECT_THROW(static_cast<void>(range.boundary(987654).size()),std::logic_error);
            EXPECT_EQ(callbacks,0U);
        }
    });
}

TEST(MeshTraverseTest, LayeredResolvedAdjacencyRetainsLiveGeometryAcrossAxialMotion)
{
    auto native=layered_interior_fixture(3);
    const Handle mesh(native);
    const auto plan=make_mesh_traverse_plan(mesh,MeshTraverseEntities::AllWithCellFaces);
    const auto layout=plan.layered_layout();
    const auto check=[&]
    {
        const auto traversal=plan.acquire();
        traversal.cells().apply([&](const auto& cell)
        {
            const auto check_faces=[&](const auto& range)
            {
                range.interior().apply([&](const auto& face)
                {
                    using Access=std::remove_cvref_t<decltype(face.geometry())>;
                    if constexpr (detail::is_layered_traverse_access<Access>)
                    {
                        EXPECT_EQ(face.owner(),mesh.owner_cell(face.id()));
                        EXPECT_EQ(face.neighbor(),mesh.neighbor_cell(face.id()));
                        expect_vector_equal(face.cell_center_vector(),mesh.cell_center_vector(face.id(),cell.id()));
                    }
                });
            };
            check_faces(traversal.cell_faces(cell.id()));
            check_faces(traversal.cell_faces(cell));
        });
    };
    check();
    PlanarALEMeshMotion<Pack> motion(std::make_shared<Handle>(mesh));
    motion.begin_trial(1.2, 0.2);
    EXPECT_EQ(plan.layered_layout().get(),layout.get());
    check();
    motion.rollback_trial();
    check();
}

