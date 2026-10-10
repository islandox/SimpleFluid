/** @file testPressureGeometryAccess.cc
 * @brief Pressure geometry leases, exact row parity, and collective failures.
 */
#include "FVM/MatrixOperators.hh"
#include "geometry/MeshReorderingFactory.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "utils/testing_environment.hh"

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Matrix = Pack::matrix_type;
using Gauge = std::optional<Pack::global_ordinal_type>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

/** Hide the lease entry to exercise the legacy fallback and per-query validation. */
class UnleasedHandle : public Handle
{
public:
    using Handle::Handle;
    void acquire_execution_view() const = delete;
};

class ThrowingHandle : public Handle
{
public:
    using Handle::Handle;
    bool fail_acquisition = false, fail_faces = false, generic_capacity = false;
    auto acquire_execution_view() const
    {
        if (fail_acquisition) throw std::logic_error("Injected pressure lease failure.");
        return Handle::acquire_execution_view();
    }
    CellFaceRange faces(local_ordinal_type cell) const
    {
        if (fail_faces) throw std::out_of_range("Injected pressure face failure.");
        return Handle::faces(cell);
    }
    bool supports_region_execution() const noexcept
    {
        return !generic_capacity && Handle::supports_region_execution();
    }
};

struct Row
{
    std::vector<Pack::local_ordinal_type> columns;
    std::vector<Pack::scalar_type> values;
    bool operator==(const Row&) const = default;
};

std::vector<Row> snapshot(const Matrix& matrix)
{
    std::vector<Row> result(matrix.getLocalNumRows());
    for (size_t row = 0; row < result.size(); ++row)
    {
        Matrix::local_inds_host_view_type columns;
        Matrix::values_host_view_type values;
        matrix.getLocalRowView(static_cast<Pack::local_ordinal_type>(row), columns, values);
        for (size_t entry = 0; entry < columns.extent(0); ++entry)
        {
            result[row].columns.push_back(columns(entry));
            result[row].values.push_back(values(entry));
        }
    }
    return result;
}

BoundaryCondition mixed_boundary(int batch, size_t index)
{
    return {(batch + index) % 2 ? BoundaryConditionType::Dirichlet : BoundaryConditionType::Neumann, 0.0};
}

BoundaryCondition dirichlet(int, size_t) { return {BoundaryConditionType::Dirichlet, 0.0}; }
BoundaryCondition neumann(int, size_t) { return {}; }

void eliminate_gauge_column(Matrix& matrix, Gauge gauge)
{
    if (!gauge) return;
    matrix.resumeFill();
    const auto column = matrix.getColMap()->getLocalElement(*gauge);
    if (column != Teuchos::OrdinalTraits<Pack::local_ordinal_type>::invalid())
    {
        const Pack::scalar_type zero = 0;
        for (size_t row = 0; row < matrix.getLocalNumRows(); ++row)
            if (matrix.getRowMap()->getGlobalElement(row) != *gauge)
                matrix.replaceLocalValues(row, 1, &zero, &column);
    }
    matrix.fillComplete();
}

template<class MeshType>
void check_refresh(const MeshType& mesh, Matrix& matrix, Gauge gauge, bool symmetric, bool expect_changed)
{
    bool changed = !expect_changed;
    EXPECT_TRUE(FVM::detail::refresh_pressure_poisson_matrix_values<Pack>(
        mesh, gauge, mixed_boundary, symmetric, matrix, changed));
    EXPECT_EQ(changed, expect_changed);
    const auto fresh = FVM::pressure_poisson_matrix<Pack>(mesh, gauge, mixed_boundary);
    if (symmetric) eliminate_gauge_column(*fresh, gauge);
    EXPECT_EQ(snapshot(matrix), snapshot(*fresh));
    EXPECT_TRUE(matrix.isFillComplete());
}

template<class Action>
void expect_collective_failure(const Teuchos::Comm<int>& comm, Action&& action,
    bool local_origin, const char* local_message, const std::type_info& local_type = typeid(std::logic_error))
{
    int caught = 0;
    try { action(); }
    catch (const std::exception& error)
    {
        caught = 1;
        if (local_origin)
        {
            EXPECT_STREQ(error.what(), local_message);
            EXPECT_TRUE(typeid(error) == local_type) << typeid(error).name();
        }
        else
        {
            EXPECT_NE(std::string(error.what()).find("another rank"), std::string::npos);
            EXPECT_TRUE(typeid(error) == typeid(std::runtime_error)) << typeid(error).name();
        }
    }
    EXPECT_EQ(caught, 1);
    int total_caught = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &caught, &total_caught);
    EXPECT_EQ(total_caught, comm.getSize());
}

template<class MeshType, class Boundary>
void check_failed_assembly_and_refresh(const MeshType& mesh, Boundary boundary,
    Matrix& matrix, bool local_origin, const char* local_message,
    const std::type_info& local_type = typeid(std::logic_error))
{
    const auto before = snapshot(matrix);
    const auto comm = mesh.owned_cell_map()->getComm();
    expect_collective_failure(*comm,
        [&] { (void)FVM::pressure_poisson_matrix<Pack>(mesh, Gauge{}, boundary); },
        local_origin, local_message, local_type);
    bool changed = true;
    expect_collective_failure(*comm, [&]
    {
        (void)FVM::detail::refresh_pressure_poisson_matrix_values<Pack>(
            mesh, Gauge{}, boundary, false, matrix, changed);
    }, local_origin, local_message, local_type);
    EXPECT_FALSE(changed);
    EXPECT_TRUE(matrix.isFillComplete());
    EXPECT_EQ(snapshot(matrix), before);
}
} // namespace

TEST(PressureGeometryAccessTest, LeasedRowsMatchIndividuallyValidatedGeometryExactly)
{
    for (const auto& geometry : {test::two_regions(), test::mixed_regions(), test::coarse_fine_regions(),
             test::periodic_regions(), test::cylindrical_regions(), test::independent_extruded_regions()})
    {
        const Handle mesh(geometry);
        const UnleasedHandle reference(geometry);
        for (const auto gauge : {Gauge{}, Gauge{mesh.owned_cell_map()->getMinAllGlobalIndex()}})
        {
            std::vector<std::pair<int, size_t>> calls, expected_calls;
            const auto assembled = FVM::pressure_poisson_matrix<Pack>(mesh, gauge,
                [&](int batch, size_t index)
                {
                    EXPECT_THROW(geometry->require_geometry_writable(), std::logic_error);
                    calls.emplace_back(batch, index);
                    return mixed_boundary(batch, index);
                });
            const auto expected = FVM::pressure_poisson_matrix<Pack>(reference, gauge,
                [&](int batch, size_t index)
                {
                    EXPECT_NO_THROW(geometry->require_geometry_writable());
                    expected_calls.emplace_back(batch, index);
                    return mixed_boundary(batch, index);
                });
            EXPECT_EQ(calls, expected_calls);
            EXPECT_EQ(snapshot(*assembled), snapshot(*expected));
            EXPECT_EQ(assembled->getGlobalNumEntries(), expected->getGlobalNumEntries());
            check_refresh(mesh, *assembled, gauge, false, false);
            EXPECT_NO_THROW(geometry->require_geometry_writable());
        }
    }
}

TEST(PressureGeometryAccessTest, MotionRollbackAndIdenticalReplayReacquireCurrentGeometry)
{
    for (const auto& geometry : {test::coarse_fine_regions(), test::periodic_regions()})
        for (const bool symmetric : {false, true})
        {
            auto mesh = std::make_shared<Handle>(geometry);
            const Gauge gauge = mesh->owned_cell_map()->getMinAllGlobalIndex();
            auto matrix = FVM::pressure_poisson_matrix<Pack>(*mesh, gauge, mixed_boundary);
            if (symmetric) eliminate_gauge_column(*matrix, gauge);
            const auto original = snapshot(*matrix);
            PlanarALEMeshMotion<> motion(mesh);
            motion.begin_trial(1.3, 0.2);
            check_refresh(*mesh, *matrix, gauge, symmetric, true);
            const auto trial = snapshot(*matrix);
            check_refresh(*mesh, *matrix, gauge, symmetric, false);
            motion.rollback_trial();
            check_refresh(*mesh, *matrix, gauge, symmetric, true);
            EXPECT_EQ(snapshot(*matrix), original);
            motion.begin_trial(1.3, 0.2);
            check_refresh(*mesh, *matrix, gauge, symmetric, true);
            EXPECT_EQ(snapshot(*matrix), trial);
            motion.accept_trial();
            check_refresh(*mesh, *matrix, gauge, symmetric, false);
            motion.begin_trial(1.0, 0.2);
            motion.accept_trial();
        }
}

TEST(PressureGeometryAccessTest, AcquisitionCapacityAndRowFailuresAreCollectiveAndReleaseLeases)
{
    auto mesh = std::make_shared<ThrowingHandle>(test::two_regions());
    const auto comm = mesh->owned_cell_map()->getComm();
    const bool origin = comm->getRank() == 0;
    auto matrix = FVM::pressure_poisson_matrix<Pack>(*mesh, Gauge{}, dirichlet);
    mesh->fail_acquisition = origin;
    check_failed_assembly_and_refresh(*mesh, dirichlet, *matrix, origin, "Injected pressure lease failure.");
    mesh->fail_acquisition = false;
    mesh->fail_faces = origin;
    check_failed_assembly_and_refresh(*mesh, dirichlet, *matrix, origin,
        "Injected pressure face failure.", typeid(std::out_of_range));
    mesh->generic_capacity = true;
    check_failed_assembly_and_refresh(*mesh, dirichlet, *matrix, origin,
        "Injected pressure face failure.", typeid(std::out_of_range));
    mesh->fail_faces = false;
    mesh->generic_capacity = false;
    PlanarALEMeshMotion<> motion(mesh);
    EXPECT_NO_THROW(motion.begin_trial(1.1, 0.2));
    EXPECT_NO_THROW(motion.rollback_trial());
    bool changed = true;
    EXPECT_TRUE(FVM::detail::refresh_pressure_poisson_matrix_values<Pack>(
        *mesh, Gauge{}, dirichlet, false, *matrix, changed));
    EXPECT_FALSE(changed);
}

TEST(PressureGeometryAccessTest, RankDivergentAcquisitionAndTraversalFailuresStayCollective)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    ThrowingHandle mesh(test::two_regions());
    auto matrix = FVM::pressure_poisson_matrix<Pack>(mesh, Gauge{}, dirichlet);
    mesh.fail_acquisition = comm->getRank() == 0;
    mesh.fail_faces = comm->getRank() == 1;
    const auto before = snapshot(*matrix);
    bool changed = true;
    expect_collective_failure(*comm, [&]
    {
        (void)FVM::detail::refresh_pressure_poisson_matrix_values<Pack>(
            mesh, Gauge{}, dirichlet, false, *matrix, changed);
    }, true, comm->getRank() == 0 ? "Injected pressure lease failure." : "Injected pressure face failure.",
        comm->getRank() == 0 ? typeid(std::logic_error) : typeid(std::out_of_range));
    EXPECT_FALSE(changed);
    EXPECT_EQ(snapshot(*matrix), before);
    EXPECT_TRUE(matrix->isFillComplete());
}

TEST(PressureGeometryAccessTest, BoundaryFailuresKeepOriginalExceptionAndMatrix)
{
    auto mesh = std::make_shared<Handle>(test::two_regions());
    const bool origin = mesh->owned_cell_map()->getComm()->getRank() == 0;
    auto matrix = FVM::pressure_poisson_matrix<Pack>(*mesh, Gauge{}, dirichlet);
    auto unsupported = [&](int, size_t)
    {
        return BoundaryCondition{origin ? BoundaryConditionType::Robin : BoundaryConditionType::Dirichlet, 0};
    };
    check_failed_assembly_and_refresh(*mesh, unsupported, *matrix, origin,
        "pressure_poisson_matrix supports only Dirichlet and Neumann pressure boundary conditions.",
        typeid(std::invalid_argument));
    auto throws = [&](int, size_t)
    {
        if (origin) throw std::invalid_argument("Injected pressure boundary failure.");
        return dirichlet(0, 0);
    };
    check_failed_assembly_and_refresh(*mesh, throws, *matrix, origin,
        "Injected pressure boundary failure.", typeid(std::invalid_argument));
    PlanarALEMeshMotion<> motion(mesh);
    EXPECT_NO_THROW(motion.begin_trial(1.2, 0.2));
    EXPECT_NO_THROW(motion.rollback_trial());
}

TEST(PressureGeometryAccessTest, StaticChildMutationFailsBeforeFreshOrRefreshCollectives)
{
    const Vec3D<ArrReal> left_edges{{{0,0.5,1}, {0,0.5,1}, {0,0.5,1}}};
    const Vec3D<ArrReal> right_edges{{{1,1.5,2}, {0,0.5,1}, {0,0.5,1}}};
    auto left = std::make_shared<Meshes::OrthogonalCartesian3D>(left_edges);
    auto right = std::make_shared<Meshes::OrthogonalCartesian3D>(right_edges);
    auto geometry = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{
            Meshes::native_region("left", left), Meshes::native_region("right", right)},
        std::vector<Meshes::MultiRegionMesh::Interface>{Meshes::StructuredPatchInterface{{0,1}, {1,0}}});
    Handle mesh(geometry);
    auto matrix = FVM::pressure_poisson_matrix<Pack>(mesh, Gauge{}, dirichlet);
    const bool origin = mesh.owned_cell_map()->getComm()->getRank() == 0;
    if (origin) *left = Meshes::OrthogonalCartesian3D(left_edges);
    check_failed_assembly_and_refresh(mesh, dirichlet, *matrix, origin,
        "Static region constituent changed; rebuild MultiRegionMesh and its fields/operators.");
    EXPECT_NO_THROW(geometry->require_geometry_writable());
    EXPECT_NO_THROW(left->require_geometry_writable());
}

TEST(PressureGeometryAccessTest, IncompatibleGraphsAndMapsLeaveMatrixUntouched)
{
    const auto geometry = test::two_regions();
    Handle mesh(geometry);
    const Gauge gauge = mesh.owned_cell_map()->getMinAllGlobalIndex();
    // Removing the gauge requires off-diagonal entries absent from the gauge row.
    auto matrix = FVM::pressure_poisson_matrix<Pack>(mesh, gauge, neumann);
    auto before = snapshot(*matrix);
    bool changed = true;
    EXPECT_FALSE(FVM::detail::refresh_pressure_poisson_matrix_values<Pack>(
        mesh, Gauge{}, neumann, false, *matrix, changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(snapshot(*matrix), before);
    EXPECT_TRUE(matrix->isFillComplete());
    EXPECT_NO_THROW(geometry->require_geometry_writable());

    Handle other(test::two_regions(3));
    auto foreign = FVM::pressure_poisson_matrix<Pack>(other, Gauge{}, dirichlet);
    before = snapshot(*foreign);
    changed = true;
    EXPECT_FALSE(FVM::detail::refresh_pressure_poisson_matrix_values<Pack>(
        mesh, Gauge{}, dirichlet, false, *foreign, changed));
    EXPECT_FALSE(changed);
    EXPECT_EQ(snapshot(*foreign), before);
    EXPECT_TRUE(foreign->isFillComplete());
    EXPECT_NO_THROW(geometry->require_geometry_writable());
}

TEST(PressureGeometryAccessTest, EmptyRanksParticipateInSuccessAndFailure)
{
    auto geometry = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{Meshes::cartesian_region("single", {{{0,1}, {0,1}, {0,1}}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    ThrowingHandle mesh(geometry, options);
    const auto comm = mesh.owned_cell_map()->getComm();
    const int empty = mesh.num_owned_cells() == 0;
    int total_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(total_empty, comm->getSize() - 1);
    auto matrix = FVM::pressure_poisson_matrix<Pack>(mesh, Gauge{}, mixed_boundary);
    check_refresh(mesh, *matrix, Gauge{}, false, false);
    // An empty rank must still acquire and validate geometry on both paths.
    mesh.fail_acquisition = comm->getSize() > 1 ? bool(empty) : true;
    check_failed_assembly_and_refresh(mesh, mixed_boundary, *matrix, mesh.fail_acquisition,
        "Injected pressure lease failure.");
    mesh.fail_acquisition = false;
    const bool origin = !empty;
    auto invalid_boundary = [&](int, size_t)
    {
        if (origin) throw std::invalid_argument("Injected pressure boundary failure.");
        return dirichlet(0, 0);
    };
    check_failed_assembly_and_refresh(mesh, invalid_boundary, *matrix, origin,
        "Injected pressure boundary failure.", typeid(std::invalid_argument));
}

TEST(PressureGeometryAccessTest, ReorderedNativeHandleKeepsGenericPressureAssembly)
{
    if (Tpetra::getDefaultComm()->getSize() != 1) GTEST_SKIP() << "Serial reordered reference.";
    auto mesh = std::make_shared<Handle>(std::make_shared<Meshes::OrthogonalCartesian3D>(
        Vec3D<ArrReal>{{{0,0.5,1,1.5,2}, {0,0.5,1}, {0,0.5,1}}}));
    auto reordered = MeshReorderingFactory<>::selected_cells_first(std::move(mesh),
        [](auto, const MeshUtils::Vec3& center) { return center.x > 1; });
    EXPECT_FALSE(reordered.mesh->supports_region_execution());
    const Gauge gauge = reordered.mesh->owned_cell_map()->getMinAllGlobalIndex();
    auto matrix = FVM::pressure_poisson_matrix<Pack>(*reordered.mesh, gauge, mixed_boundary);
    check_refresh(*reordered.mesh, *matrix, gauge, false, false);
}
