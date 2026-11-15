/** @file testSTKFaceAuthority.cc
 *  @brief Pure-data face authority agreement without STK objects.
 */
#include "io/details/STKFaceAuthority.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_DefaultSerialComm.hpp>
#include <gtest/gtest.h>
#include <vector>

namespace
{
using namespace SimpleFluid;
using MeshData = Meshes::UnstructuredMeshData;
using ID = MeshData::ID;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

MeshData::Face canonical_face()
{
    return {1007, {11, 12, 13, 14}, 4001, 9001, -1, {0.5, 0.5, 0.0625}, {0, 0, 1}, 1.125, 77};
}

MeshData input(int rank)
{
    MeshData data;
    data.num_owned_cells = 0;
    data.boundary_names = {{7, "wall"}, {19, "empty_patch"}};
    if (rank > 1)
        return data;
    data.nodes = {{11, {0, 0, 0}}, {12, {1, 0, 0}}, {13, {1, 1, 0.25}}, {14, {0, 1, 0}}};
    auto face = canonical_face();
    if (rank == 1)
    {
        // Same cyclic topology and unordered incidence, but a different
        // triangulation/owner has already produced different metrics.
        face.node_ids = {13, 12, 11, 14};
        face.owner = 9001;
        face.neighbor = 4001;
        face.center = {0.625, 0.375, 0.125};
        face.normal_from_owner = {0, 0, -1};
        face.area = 1.25;
    }
    data.faces.push_back(std::move(face));
    data.cells.push_back({rank == 0 ? ID{4001} : ID{9001}, MeshUtils::CellType::POLYHEDRON, {11, 12, 13, 14}, {1007},
        {0.5, 0.5, rank == 0 ? -0.5 : 0.5}, 1});
    data.num_owned_cells = 1;
    return data;
}

void expect_face(const MeshData::Face& actual, const MeshData::Face& expected)
{
    EXPECT_EQ(actual.id, expected.id);
    EXPECT_EQ(actual.node_ids, expected.node_ids);
    EXPECT_EQ(actual.owner, expected.owner);
    EXPECT_EQ(actual.neighbor, expected.neighbor);
    EXPECT_EQ(actual.boundary_id, expected.boundary_id);
    EXPECT_EQ(actual.algebraic_id, expected.algebraic_id);
    EXPECT_EQ(actual.center, expected.center);
    EXPECT_EQ(actual.normal_from_owner, expected.normal_from_owner);
    EXPECT_EQ(actual.area, expected.area);
}
} // namespace

TEST(STKFaceAuthorityTest, CopiesWholeMinimumOwnedCellRecordWithoutRecomputingMetrics)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    const auto nodes = data.nodes;
    const auto cells = data.cells;
    detail::synchronize_stk_face_authority(data, comm);
    if (comm->getRank() < 2)
        expect_face(data.faces[0], canonical_face());
    EXPECT_EQ(data.cells.size(), cells.size());
    EXPECT_EQ(data.nodes.size(), nodes.size());
    for (size_t c = 0; c < cells.size(); ++c)
        EXPECT_EQ(data.cells[c].face_ids, cells[c].face_ids);
    for (size_t n = 0; n < nodes.size(); ++n)
        EXPECT_EQ(data.nodes[n].coordinates, nodes[n].coordinates);
    EXPECT_EQ(data.boundary_names.at(19), "empty_patch");
}

TEST(STKFaceAuthorityTest, RankBreaksEqualProducerCellPriority)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    if (comm->getRank() == 1)
        data.cells[0].id = 4001;
    detail::synchronize_stk_face_authority(data, comm);
    if (comm->getRank() < 2)
        expect_face(data.faces[0], canonical_face());
}

TEST(STKFaceAuthorityTest, AuraOnlyPartialRecordsRemainUnchanged)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    MeshData::Face aura;
    aura.id = 987654;
    aura.boundary_id = 999;
    data.faces.push_back(aura);
    detail::synchronize_stk_face_authority(data, comm);
    expect_face(data.faces.back(), aura);
}

TEST(STKFaceAuthorityTest, CrossedQuadLoopFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    if (comm->getRank() == 1)
        data.faces[0].node_ids = {11, 13, 12, 14};
    EXPECT_ANY_THROW(detail::synchronize_stk_face_authority(data, comm));
}

TEST(STKFaceAuthorityTest, SharedNodeCoordinateDisagreementFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    if (comm->getRank() == 1)
        data.nodes[0].coordinates.x = 0.25;
    EXPECT_ANY_THROW(detail::synchronize_stk_face_authority(data, comm));
}

TEST(STKFaceAuthorityTest, BoundaryCatalogDisagreementFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    if (comm->getRank() == 1)
        data.boundary_names[19] = "conflicting_patch";
    EXPECT_ANY_THROW(detail::synchronize_stk_face_authority(data, comm));
}

TEST(STKFaceAuthorityTest, AuthoritativePhysicalFaceIdsMustAgree)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    if (comm->getRank() == 1)
    {
        data.faces[0].id = 1009;
        data.cells[0].face_ids[0] = 1009;
    }
    EXPECT_ANY_THROW(detail::synchronize_stk_face_authority(data, comm));
}

TEST(STKFaceAuthorityTest, EffectiveAlgebraicOrdinalsMustAgree)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    if (comm->getRank() == 1)
        data.faces[0].algebraic_id = 78;
    EXPECT_ANY_THROW(detail::synchronize_stk_face_authority(data, comm));
}

TEST(STKFaceAuthorityTest, OneRankMissingRequiredSourceFaceFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    if (comm->getRank() == 1)
        data.faces.clear();
    EXPECT_ANY_THROW(detail::synchronize_stk_face_authority(data, comm));
}

TEST(STKFaceAuthorityTest, SerialInputRetainsItsOriginalRecord)
{
    const auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>);
    auto data = input(1);
    const auto original = data.faces[0];
    detail::synchronize_stk_face_authority(data, comm);
    expect_face(data.faces[0], original);
}

TEST(STKFaceAuthorityTest, UsesSuppliedSubcommunicator)
{
    const auto world = Tpetra::getDefaultComm();
    if (world->getSize() < 2)
        GTEST_SKIP();
    const auto comm = world->split(world->getRank() % 2, world->getRank());
    auto data = input(comm->getRank());
    detail::synchronize_stk_face_authority(data, comm);
    if (comm->getRank() < 2)
        expect_face(data.faces[0], canonical_face());
}

TEST(STKFaceAuthorityTest, SharedNodeSignedZeroRepresentationDisagreementFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() < 2)
        GTEST_SKIP();
    auto data = input(comm->getRank());
    if (comm->getRank() == 1)
        data.nodes[0].coordinates.x = -0.0;
    EXPECT_ANY_THROW(detail::synchronize_stk_face_authority(data, comm));
}
