/**
 * @file testMeshQualityAccess.cc
 * @brief Scoped mesh-quality access, collective failures, and lease release.
 */
#include "geometry/MeshQuality.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>
#include <Teuchos_CommHelpers.hpp>

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Cartesian = Meshes::OrthogonalCartesian3D;
using Composite = Meshes::MultiRegionMesh;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

struct Fixture
{
    std::shared_ptr<Cartesian> child;
    std::shared_ptr<Composite> geometry;
    std::shared_ptr<Handle> mesh;
};

Fixture single_cell()
{
    Fixture result;
    result.child = std::make_shared<Cartesian>(Vec3D<ArrReal>{{{0, 1}, {0, 1}, {0, 1}}});
    result.geometry = std::make_shared<Composite>(std::vector<Composite::Region>{
        Meshes::native_region("single", result.child)}, std::vector<Composite::Interface>{});
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    result.mesh = std::make_shared<Handle>(result.geometry, options);
    return result;
}

Fixture two_cells()
{
    Fixture result;
    result.child = std::make_shared<Cartesian>(Vec3D<ArrReal>{{{0, 1}, {0, 1}, {0, 1}}});
    auto right = std::make_shared<Cartesian>(Vec3D<ArrReal>{{{1, 3}, {0, 1}, {0, 1}}});
    result.geometry = std::make_shared<Composite>(std::vector<Composite::Region>{
        Meshes::native_region("left", result.child), Meshes::native_region("right", right)},
        std::vector<Composite::Interface>{Meshes::StructuredPatchInterface{{0, 1}, {1, 0}}});
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    result.mesh = std::make_shared<Handle>(result.geometry, options);
    return result;
}

void expect_same_metrics(const MeshQualityMetrics& actual, const MeshQualityMetrics& expected)
{
    EXPECT_EQ(actual.global_cell_count, expected.global_cell_count);
    EXPECT_EQ(actual.global_face_count, expected.global_face_count);
    EXPECT_DOUBLE_EQ(actual.minimum_cell_volume, expected.minimum_cell_volume);
    EXPECT_DOUBLE_EQ(actual.minimum_face_area, expected.minimum_face_area);
    EXPECT_DOUBLE_EQ(actual.minimum_normal_distance, expected.minimum_normal_distance);
    EXPECT_DOUBLE_EQ(actual.maximum_growth_ratio, expected.maximum_growth_ratio);
    EXPECT_NEAR(actual.maximum_non_orthogonality_degrees, expected.maximum_non_orthogonality_degrees, 1.0e-12);
    EXPECT_NEAR(actual.maximum_skewness, expected.maximum_skewness, 1.0e-12);
    EXPECT_DOUBLE_EQ(actual.maximum_aspect_ratio, expected.maximum_aspect_ratio);
}

void expect_stale_child_failure_and_release(Fixture& fixture, bool mutate_here)
{
    const auto comm = fixture.mesh->owned_cell_map()->getComm();
    if (mutate_here)
        *fixture.child = Cartesian(Vec3D<ArrReal>{{{0, 1}, {0, 1}, {0, 1}}});
    int caught = 0;
    try { (void)evaluate_mesh_quality(*fixture.mesh); }
    catch (const std::exception& error)
    {
        caught = typeid(error) == typeid(std::runtime_error)
            && std::string(error.what()) == "Mesh-quality evaluation encountered invalid local geometry.";
        EXPECT_TRUE(typeid(error) == typeid(std::runtime_error));
        EXPECT_STREQ(error.what(), "Mesh-quality evaluation encountered invalid local geometry.");
    }
    EXPECT_EQ(caught, 1);
    int globally_caught = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &caught, &globally_caught);
    EXPECT_EQ(globally_caught, comm->getSize());
    // ExecutionView takes parent and child leases before validate_static().
    // A throwing constructor must release even its partially built lease list.
    EXPECT_NO_THROW(fixture.geometry->require_geometry_writable());
    EXPECT_NO_THROW(fixture.child->require_geometry_writable());
}
} // namespace

TEST(QualityAccessTest, NativeAndCompositeMatchAnalyticalTwoCellMetrics)
{
    auto fixture = two_cells();
    auto native_geometry = std::make_shared<Cartesian>(Vec3D<ArrReal>{{{0, 1, 3}, {0, 1}, {0, 1}}});
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    const Handle native(native_geometry, options);
    MeshQualityMetrics expected;
    expected.global_cell_count = 2;
    expected.global_face_count = 11;
    expected.minimum_cell_volume = 1;
    expected.minimum_face_area = 1;
    expected.minimum_normal_distance = 0.5;
    expected.maximum_growth_ratio = 2;
    expected.maximum_non_orthogonality_degrees = 0;
    expected.maximum_skewness = 0;
    expected.maximum_aspect_ratio = 2;
    const auto native_quality = evaluate_mesh_quality(native);
    const auto composite_quality = evaluate_mesh_quality(*fixture.mesh);
    expect_same_metrics(native_quality, expected);
    expect_same_metrics(composite_quality, expected);
    expect_same_metrics(composite_quality, native_quality);
    EXPECT_NO_THROW(fixture.geometry->require_geometry_writable());
    EXPECT_NO_THROW(fixture.child->require_geometry_writable());
}

TEST(QualityAccessTest, RankLocalChildReplacementFailsCollectivelyAndReleasesAcquisition)
{
    auto fixture = single_cell();
    expect_stale_child_failure_and_release(fixture, fixture.mesh->owned_cell_map()->getComm()->getRank() == 0);
}

TEST(QualityAccessMultiRankTest, EmptyRankChildReplacementFailsCollectivelyAndReleasesAcquisition)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Register this case only for two MPI ranks.";
    auto fixture = single_cell();
    const bool empty = fixture.mesh->num_owned_cells() == 0;
    const int empty_count = empty ? 1 : 0;
    int global_empty_count = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty_count, &global_empty_count);
    EXPECT_EQ(global_empty_count, 1);
    expect_stale_child_failure_and_release(fixture, empty);
}

TEST(QualityAccessTest, EvaluationReleasesBeforeMotionQualityRollbackAndRetry)
{
    auto fixture = two_cells();
    const auto original = evaluate_mesh_quality(*fixture.mesh);
    const auto original_epoch = fixture.mesh->geometry_epoch();
    const auto owned_map = fixture.mesh->owned_cell_map();
    const auto overlap_map = fixture.mesh->overlap_cell_map();
    const auto owned_faces = fixture.mesh->owned_face_map();
    PlanarALEMeshMotionOptions options;
    options.quality_limits.maximum_aspect_ratio = 4;
    PlanarALEMeshMotion<> motion(fixture.mesh, options);
    EXPECT_THROW(motion.begin_trial(100, 1), std::runtime_error);
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(fixture.mesh->geometry_epoch(), original_epoch + 2);
    EXPECT_EQ(fixture.mesh->owned_cell_map(), owned_map);
    EXPECT_EQ(fixture.mesh->overlap_cell_map(), overlap_map);
    EXPECT_EQ(fixture.mesh->owned_face_map(), owned_faces);
    expect_same_metrics(evaluate_mesh_quality(*fixture.mesh), original);
    EXPECT_NO_THROW(fixture.geometry->require_geometry_writable());
    EXPECT_NO_THROW(fixture.child->require_geometry_writable());
    EXPECT_NO_THROW(motion.begin_trial(1.25, 0.2));
    EXPECT_TRUE(motion.has_active_trial());
    EXPECT_NO_THROW(MeshQualityGate(options.quality_limits).require(evaluate_mesh_quality(*fixture.mesh)));
    EXPECT_NO_THROW(motion.rollback_trial());
    EXPECT_FALSE(motion.has_active_trial());
    EXPECT_EQ(fixture.mesh->geometry_epoch(), original_epoch + 4);
    expect_same_metrics(evaluate_mesh_quality(*fixture.mesh), original);
    EXPECT_NO_THROW(fixture.geometry->require_geometry_writable());
    EXPECT_NO_THROW(fixture.child->require_geometry_writable());
}
