/** @file MpiCases.cc
 * @brief Distributed traversal ownership and partition checks.
 */
#include "test_helpers.hh"

TEST(MeshTraverseMultiRankTest, ZeroHaloDoesNotTurnPartitionInterfacesIntoPhysicalBoundaries)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two ranks.";
    Handle mesh(cartesian(4, 3, 3), Handle::DistributionOptions{.ghost_layers = 0});
    expect_handle_selections(mesh);
    const auto plan = make_mesh_traverse_plan(mesh);
    const auto traverse = plan.acquire();
    int absent_adjacency = 0;
    EXPECT_FALSE(plan.uses_structured_ranges());
    traverse.faces().interior().local().apply([&](const auto& face)
    {
        EXPECT_FALSE(mesh.is_geometry_exterior_face(face.id()));
        if (face.owner() == Handle::invalid_local_id() || face.neighbor() == Handle::invalid_local_id())
            ++absent_adjacency;
    });
    int global_absent = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &absent_adjacency, &global_absent);
    EXPECT_GT(global_absent, 0);
    EXPECT_THROW(traverse.faces().interior().local().require_local_adjacency(), std::invalid_argument);
    EXPECT_NO_THROW(traverse.faces().boundary().local().require_local_adjacency());
    int interior = static_cast<int>(selected_ids(traverse.cells().interior()).size());
    int global_interior = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &interior, &global_interior);
    EXPECT_EQ(global_interior, 2);

    const auto incidence_plan = make_mesh_traverse_plan(mesh, MeshTraverseEntities::AllWithCellFaces);
    const auto incidence = incidence_plan.acquire();
    int unowned_incidence = 0;
    for (size_t c = 0; c < mesh.num_owned_cells(); ++c)
    {
        const auto cell = static_cast<Local>(c);
        std::vector<Local> expected;
        for (const auto face : mesh.faces(cell))
        {
            expected.push_back(face);
            if (!mesh.is_owned_face(face)) ++unowned_incidence;
        }
        EXPECT_EQ(selected_ids(incidence.cell_faces(cell).all()), expected);
    }
    int global_unowned = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &unowned_incidence, &global_unowned);
    EXPECT_GT(global_unowned, 0);
}

TEST(MeshTraverseMultiRankTest, OwnedAndLocalSelectionsHandleGhostsAndEmptyRanks)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two ranks.";
    const auto geometry = cartesian(4, 3, 3);
    Handle with_ghosts(geometry);
    expect_handle_selections(with_ghosts);
    EXPECT_GT(with_ghosts.num_local_cells(), with_ghosts.num_owned_cells());
    {
        const auto plan = make_mesh_traverse_plan(with_ghosts);
        const auto traverse = plan.acquire();
        EXPECT_NO_THROW(traverse.faces().all().require_local_adjacency());
        std::vector<Local> interior, boundary;
        for (size_t c = 0; c < with_ghosts.num_local_cells(); ++c)
        {
            const auto cell = static_cast<Local>(c);
            const auto native = geometry->cell_id(with_ghosts.cell_geometry_global_id(cell));
            bool exterior = false;
            for (const auto face : geometry->faces(native))
                exterior = exterior || geometry->is_exterior_face(face);
            (exterior ? boundary : interior).push_back(cell);
        }
        EXPECT_EQ(selected_ids(traverse.cells().interior().local()), interior);
        EXPECT_EQ(selected_ids(traverse.cells().boundary().local()), boundary);
    }

    Handle sparse(cartesian(1, 1, 1),
                  Handle::DistributionOptions{.ghost_layers = 0, .allow_empty_partitions = true});
    expect_handle_selections(sparse);
    int empty = sparse.num_owned_cells() == 0 ? 1 : 0;
    int empty_ranks = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &empty_ranks);
    EXPECT_EQ(empty_ranks, 1);
}

