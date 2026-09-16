/**
 * @file testMeshToMeshTransferMultiRank.cc
 * @brief Distributed mesh transfer, halo refresh and collective validation.
 */

#include <gtest/gtest.h>

#include "fields/MeshToMeshTransfer.hh"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <vector>

namespace
{

using Pack = SimpleFluid::DefaultTpetraTypes;
using Handle = SimpleFluid::MeshHandle<Pack>;
using Transfer = SimpleFluid::MeshToMeshTransfer<Pack>;
using Options = SimpleFluid::MeshToMeshTransferOptions;
using Method = SimpleFluid::MeshToMeshTransferMethod;
using CoverageMode = SimpleFluid::MeshToMeshCoverageMode;
using Quantity = SimpleFluid::MeshToMeshQuantity;
using ScalarField = SimpleFluid::ScalarCellFieldStored<Pack>;
using VectorField = SimpleFluid::VectorCellFieldStored<Pack>;
using LO = Pack::local_ordinal_type;

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

SimpleFluid::SP<Handle> make_cartesian(bool reversed = false,
                                      bool coarse = false)
{
    const auto comm = Tpetra::getDefaultComm();
    SimpleFluid::ArrReal x = coarse
        ? SimpleFluid::ArrReal{0.0, 2.0, 4.0, 6.0, 8.0}
        : SimpleFluid::ArrReal{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto geometry = std::make_shared<Handle::Cartesian>(
        SimpleFluid::Vec3D<SimpleFluid::ArrReal>{{
            std::move(x), {0.0, 1.0}, {0.0, 1.0}}});
    Handle::DistributionOptions distribution;
    distribution.partition = static_cast<size_t>(
        reversed ? comm->getSize() - 1 - comm->getRank() : comm->getRank());
    distribution.partitions = static_cast<size_t>(comm->getSize());
    return std::make_shared<Handle>(std::move(geometry), distribution);
}

double global_integral(const ScalarField& field)
{
    double local = 0.0;
    for (size_t row = 0; row < field.num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        local += field.value(cell) * field.mesh().cell_volume(cell);
    }
    double global = 0.0;
    Teuchos::reduceAll(*field.map()->getComm(), Teuchos::REDUCE_SUM,
                       1, &local, &global);
    return global;
}

SimpleFluid::SP<Handle> make_cylindrical(
    SimpleFluid::Vec3D<SimpleFluid::ArrReal> edges, bool reversed = false,
    bool allow_empty_partitions = false)
{
    const auto comm = Tpetra::getDefaultComm();
    auto geometry = std::make_shared<Handle::Cylindrical>(std::move(edges));
    Handle::DistributionOptions distribution;
    distribution.partition = static_cast<size_t>(
        reversed ? comm->getSize() - 1 - comm->getRank() : comm->getRank());
    distribution.partitions = static_cast<size_t>(comm->getSize());
    distribution.allow_empty_partitions = allow_empty_partitions;
    return std::make_shared<Handle>(std::move(geometry), distribution);
}

auto cylindrical_id(const Handle& mesh, LO local)
{
    return std::get<Handle::CylindricalPtr>(mesh.variant())->cell_id(
        static_cast<size_t>(mesh.cell_geometry_global_id(local)));
}

SimpleFluid::SP<Handle> make_unstructured_on_comm(
    Teuchos::RCP<const Pack::comm_type> comm, bool reversed = false)
{
    using Mesh = Handle::Unstructured;
    using Partition = SimpleFluid::Meshes::PartitionedMesh<Mesh, Pack>;
    using Indexer = Handle::unstructured_indexer_type;
    const auto global = static_cast<Indexer::cell_id_t>(
        reversed ? comm->getSize() - 1 - comm->getRank() : comm->getRank());
    const double x = 2.0 * static_cast<double>(global);
    SimpleFluid::Arr<Mesh::Vec3> vertices{
        {x, 0.0, 0.0}, {x + 1.0, 0.0, 0.0},
        {x + 1.0, 1.0, 0.0}, {x, 1.0, 0.0},
        {x, 0.0, 1.0}, {x + 1.0, 0.0, 1.0},
        {x + 1.0, 1.0, 1.0}, {x, 1.0, 1.0}};
    SimpleFluid::Arr<Mesh::CellDefinition> cells{
        {SimpleFluid::MeshUtils::CellType::HEXAHEDRON, {0, 1, 2, 3, 4, 5, 6, 7}}};
    auto geometry = std::make_shared<Mesh>(vertices, cells);
    std::vector<Indexer::cell_id_t> owned_cells{global};
    std::vector<Indexer::face_id_t> owned_faces;
    std::vector<Indexer::node_id_t> owned_nodes;
    for (size_t face = 0; face < geometry->num_faces(); ++face)
        owned_faces.push_back(static_cast<Indexer::face_id_t>(
            6 * global + face));
    for (size_t node = 0; node < geometry->num_nodes(); ++node)
        owned_nodes.push_back(static_cast<Indexer::node_id_t>(
            8 * global + node));
    auto partition = std::make_shared<Partition>(geometry,
        Indexer(std::move(owned_cells), {}, std::move(owned_faces), {},
                std::move(owned_nodes)), std::move(comm));
    return std::make_shared<Handle>(std::move(partition));
}

void expect_scalar_report(
    const SimpleFluid::MeshToMeshConservationReport& report,
    double source, double transferred, double uncovered)
{
    ASSERT_EQ(report.source_integral.size(), 1U);
    ASSERT_EQ(report.transferred_integral.size(), 1U);
    ASSERT_EQ(report.uncovered_source_integral.size(), 1U);
    ASSERT_EQ(report.target_integral.size(), 1U);
    ASSERT_EQ(report.conservation_error.size(), 1U);
    ASSERT_EQ(report.transfer_error.size(), 1U);
    const double tolerance = 2.0e-12 * std::max(1.0, std::abs(source));
    EXPECT_NEAR(report.source_integral[0], source, tolerance);
    EXPECT_NEAR(report.transferred_integral[0], transferred, tolerance);
    EXPECT_NEAR(report.uncovered_source_integral[0], uncovered, tolerance);
    EXPECT_NEAR(report.target_integral[0], transferred, tolerance);
    EXPECT_NEAR(report.conservation_error[0], 0.0, tolerance);
    EXPECT_NEAR(report.transfer_error[0], 0.0, tolerance);
}

} // namespace

TEST(MeshToMeshTransferMultiRankTest, ImportsRemoteOwnersAndRefreshesScalarAndVectorHalos)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    auto source_mesh = make_cartesian(true);
    auto target_mesh = make_cartesian();
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    VectorField source_vector(source_mesh, "source_vector");
    VectorField target_vector(target_mesh, "target_vector");
    ASSERT_GT(target_mesh->num_local_cells(), target_mesh->num_owned_cells());
    int local_remote = 0;
    for (size_t row = 0; row < target_mesh->num_owned_cells(); ++row)
    {
        const auto gid = target_mesh->cell_global_id(static_cast<LO>(row));
        if (!source_mesh->owned_cell_map()->isNodeGlobalElement(gid))
        {
            ++local_remote;
        }
    }
    int global_remote = 0;
    Teuchos::reduceAll(*target.map()->getComm(), Teuchos::REDUCE_SUM,
                       1, &local_remote, &global_remote);
    ASSERT_GT(global_remote, 0);

    Transfer transfer(source_mesh, target_mesh);
    for (const double offset : {0.0, 100.0})
    {
        for (size_t row = 0; row < source_mesh->num_owned_cells(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const double x = source_mesh->cell_centroid(cell).x;
            source.set_owned_value(cell, 2.0 * x + offset);
            source_vector.set_owned_value(cell, {x + offset, -2.0 * x, 3.0 * x});
        }
        target.put_scalar(-999.0);
        target_vector.put_value({-999.0, -999.0, -999.0});
        transfer.apply(source, target);
        transfer.apply(source_vector, target_vector);
        for (size_t row = 0; row < target_mesh->num_local_cells(); ++row)
        {
            const auto cell = static_cast<LO>(row);
            const double x = target_mesh->cell_centroid(cell).x;
            EXPECT_DOUBLE_EQ(target.local_value(cell), 2.0 * x + offset);
            const auto vector = target_vector.local_value(cell);
            EXPECT_DOUBLE_EQ(vector.x, x + offset);
            EXPECT_DOUBLE_EQ(vector.y, -2.0 * x);
            EXPECT_DOUBLE_EQ(vector.z, 3.0 * x);
        }
    }
}

TEST(MeshToMeshTransferMultiRankTest, ConservativeProjectionPreservesGlobalIntegral)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    auto source_mesh = make_cartesian(true);
    auto target_mesh = make_cartesian(false, true);
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    for (size_t row = 0; row < source_mesh->num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        source.set_owned_value(cell, source_mesh->cell_centroid(cell).x + 0.5);
    }
    Options options;
    options.method = Method::ConservativeCellAverage;
    Transfer transfer(source_mesh, target_mesh, options);
    transfer.apply(source, target);
    for (size_t row = 0; row < target_mesh->num_local_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        EXPECT_NEAR(target.local_value(cell),
                    target_mesh->cell_centroid(cell).x + 0.5, 1.0e-14);
    }
    EXPECT_NEAR(global_integral(target), global_integral(source), 1.0e-13);
    EXPECT_NEAR(global_integral(target), 36.0, 1.0e-13);

    Transfer reverse(target_mesh, source_mesh, options);
    reverse.apply(target, source);
    EXPECT_NEAR(global_integral(source), 36.0, 1.0e-13);
}

TEST(MeshToMeshTransferMultiRankTest, RankDependentOptionsFailCollectively)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    auto mesh = make_cartesian();
    const int rank = mesh->owned_cell_map()->getComm()->getRank();
    Options options;
    options.neighbors = rank == 0 ? 1 : 2;
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    options = Options{};
    options.method = rank == 0 ? Method::NearestCell : Method::InverseDistance;
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    options = Options{};
    options.max_distance = rank == 0 ? 1.0 : 2.0;
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    options = Options{};
    options.coverage_tolerance = rank == 0 ? 1.0e-10 : 1.0e-8;
    EXPECT_THROW((Transfer(mesh, mesh, options)), std::invalid_argument);
    SimpleFluid::SP<const Handle> maybe_source = rank == 0 ? nullptr : mesh;
    SimpleFluid::SP<const Handle> maybe_target = rank == 0 ? nullptr : mesh;
    EXPECT_THROW((Transfer(maybe_source, mesh)), std::invalid_argument);
    EXPECT_THROW((Transfer(mesh, maybe_target)), std::invalid_argument);
}

TEST(MeshToMeshTransferMultiRankTest, HandlesRanksWithNoOwnedSourceOrTargetCells)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    using Composite = SimpleFluid::Meshes::MultiRegionMesh;
    auto geometry = std::make_shared<Composite>(
        std::vector<Composite::Region>{SimpleFluid::Meshes::cartesian_region(
            "single", {{{0.0, 8.0}, {0.0, 1.0}, {0.0, 1.0}}})},
        std::vector<Composite::Interface>{});
    auto target_mesh = std::make_shared<Handle>(geometry);
    auto source_mesh = make_cartesian();
    const int rank = target_mesh->owned_cell_map()->getComm()->getRank();
    EXPECT_EQ(target_mesh->num_owned_cells(), rank == 0 ? 1U : 0U);
    ScalarField source(source_mesh, "source");
    ScalarField target(target_mesh, "target");
    for (size_t row = 0; row < source.num_owned_cells(); ++row)
    {
        const auto cell = static_cast<LO>(row);
        source.set_owned_value(cell, source_mesh->cell_centroid(cell).x + 0.5);
    }
    Transfer forward(source_mesh, target_mesh);
    forward.apply(source, target);
    // Symmetric inverse-distance pairs average this linear fixture to 4.5.
    EXPECT_NEAR(global_integral(target), 36.0, 1.0e-13);
    EXPECT_NEAR(global_integral(target), global_integral(source), 1.0e-13);

    Transfer reverse(target_mesh, source_mesh);
    reverse.apply(target, source);
    for (size_t row = 0; row < source.num_local_cells(); ++row)
    {
        EXPECT_NEAR(source.local_value(static_cast<LO>(row)), 4.5, 1.0e-13);
    }
}

TEST(MeshToMeshTransferMultiRankTest, RankLocalRawMapMismatchFailsCollectively)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    auto mesh = make_cartesian();
    const auto map = mesh->owned_cell_map();
    const int rank = map->getComm()->getRank();
    Teuchos::Array<Pack::global_ordinal_type> reversed_ids;
    for (const auto gid : map->getLocalElementList())
    {
        reversed_ids.push_back(gid);
    }
    std::reverse(reversed_ids.begin(), reversed_ids.end());
    auto reversed_map = Teuchos::rcp(new Pack::map_type(
        map->getGlobalNumElements(), reversed_ids(), map->getIndexBase(),
        map->getComm()));
    Pack::multi_vector_type source(map, 1);
    Pack::multi_vector_type target(map, 1);
    Pack::multi_vector_type wrong_source(reversed_map, 1);
    source.putScalar(2.0);
    wrong_source.putScalar(3.0);
    target.putScalar(17.0);
    Transfer transfer(mesh, mesh);
    const auto& selected_source = rank == 0 ? wrong_source : source;
    EXPECT_THROW(transfer.apply(selected_source, target), std::invalid_argument);
    const auto values = target.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t row = 0; row < mesh->num_owned_cells(); ++row)
    {
        EXPECT_DOUBLE_EQ(values(row, 0), 17.0);
    }

    const size_t columns = rank == 0 ? 1 : 2;
    Pack::multi_vector_type rank_dependent_source(map, columns);
    Pack::multi_vector_type rank_dependent_target(map, columns);
    EXPECT_THROW(transfer.apply(rank_dependent_source, rank_dependent_target),
                 std::invalid_argument);
}

TEST(MeshToMeshTransferMultiRankTest, RankLocalBadFieldAndValueFailBeforeTargetChanges)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    auto mesh = make_cartesian();
    auto unrelated_mesh = make_cartesian();
    const int rank = mesh->owned_cell_map()->getComm()->getRank();
    ScalarField source(mesh, "source");
    ScalarField target(mesh, "target");
    ScalarField unrelated(unrelated_mesh, "unrelated");
    target.put_scalar(17.0);
    Transfer transfer(mesh, mesh);
    const ScalarField& selected_source = rank == 0 ? unrelated : source;
    EXPECT_THROW(transfer.apply(selected_source, target), std::invalid_argument);
    ScalarField& selected_target = rank == 0 ? unrelated : target;
    EXPECT_THROW(transfer.apply(source, selected_target), std::invalid_argument);

    source.put_scalar(2.0);
    if (rank == 0)
    {
        source.set_owned_value(0, std::numeric_limits<double>::quiet_NaN());
    }
    EXPECT_THROW(transfer.apply(source, target), std::invalid_argument);
    for (size_t row = 0; row < target.num_local_cells(); ++row)
    {
        EXPECT_DOUBLE_EQ(target.local_value(static_cast<LO>(row)), 17.0);
    }
    source.put_scalar(2.0);
    transfer.apply(source, target);
    for (size_t row = 0; row < target.num_local_cells(); ++row)
    {
        EXPECT_DOUBLE_EQ(target.local_value(static_cast<LO>(row)), 2.0);
    }
}

TEST(MeshToMeshTransferMultiRankTest, AnnularProjectionMatchesAnalyticSerialValuesAcrossPartitions)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    constexpr double pi = std::numbers::pi;
    Options options;
    options.method = Method::ConservativeCellAverage;
    // Equal r-squared source intervals give equal annular areas. The target's
    // middle radial strip and shifted theta sectors each average donor pairs.
    // This closed-form oracle is independent of the distributed overlap code.
    const std::array<double, 3> radial{0.0, 5.0, 10.0};
    const std::array<double, 4> angular{1.0, 3.0, 5.0, 3.0};
    const std::array<double, 3> axial{0.0, 2.0, 3.0};
    for (const bool reversed_source : {false, true})
    {
        SCOPED_TRACE(reversed_source);
        auto source_mesh = make_cylindrical({{
            {1.0, std::sqrt(5.0), 3.0},
            {0.0, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi},
            {0.0, 1.0, 3.0}}}, reversed_source);
        auto target_mesh = make_cylindrical({{
            {1.0, std::sqrt(3.0), std::sqrt(7.0), 3.0},
            {pi / 4.0, 3.0 * pi / 4.0, 5.0 * pi / 4.0,
             7.0 * pi / 4.0, 9.0 * pi / 4.0},
            {0.0, 0.5, 2.0, 3.0}}});
        Transfer transfer(source_mesh, target_mesh, options);
        ScalarField source(source_mesh, "annular_source");
        ScalarField target(target_mesh, "annular_target");
        for (const auto quantity : {Quantity::Intensive, Quantity::Extensive})
        {
            SCOPED_TRACE(static_cast<int>(quantity));
            for (size_t row = 0; row < source.num_owned_cells(); ++row)
            {
                const auto lid = static_cast<LO>(row);
                const auto id = cylindrical_id(*source_mesh, lid);
                double value = 1.0 + 10.0 * id.i + 2.0 * id.j + 3.0 * id.k;
                if (quantity == Quantity::Extensive)
                    value *= source_mesh->cell_volume(lid);
                source.set_owned_value(lid, value);
            }
            target.put_scalar(-999.0);
            const auto report = transfer.project(source, target, quantity);
            for (size_t row = 0; row < target.num_local_cells(); ++row)
            {
                const auto lid = static_cast<LO>(row);
                const auto id = cylindrical_id(*target_mesh, lid);
                double expected = 1.0 + radial[id.i] + angular[id.j] + axial[id.k];
                if (quantity == Quantity::Extensive)
                    expected *= target_mesh->cell_volume(lid);
                EXPECT_NEAR(target.local_value(lid), expected,
                            2.0e-12 * std::max(1.0, std::abs(expected)));
            }
            expect_scalar_report(report, 264.0 * pi, 264.0 * pi, 0.0);
        }

        const auto& coverage = transfer.coverage();
        EXPECT_NEAR(coverage.source_volume, 24.0 * pi, 2.0e-12);
        EXPECT_NEAR(coverage.target_volume, 24.0 * pi, 2.0e-12);
        EXPECT_NEAR(coverage.overlap_volume, 24.0 * pi, 2.0e-12);
        EXPECT_NEAR(coverage.uncovered_source_volume, 0.0, 2.0e-12);
        EXPECT_NEAR(coverage.uncovered_target_volume, 0.0, 2.0e-12);
        ASSERT_EQ(coverage.source_cell_volumes.size(), source.num_owned_cells());
        ASSERT_EQ(coverage.source_covered_volumes.size(), source.num_owned_cells());
        ASSERT_EQ(coverage.source_uncovered_volumes.size(), source.num_owned_cells());
        ASSERT_EQ(coverage.target_cell_volumes.size(), target.num_owned_cells());
        ASSERT_EQ(coverage.target_covered_volumes.size(), target.num_owned_cells());
        ASSERT_EQ(coverage.target_uncovered_volumes.size(), target.num_owned_cells());
        for (size_t row = 0; row < source.num_owned_cells(); ++row)
        {
            const auto volume = source_mesh->cell_volume(static_cast<LO>(row));
            EXPECT_NEAR(coverage.source_cell_volumes[row], volume, 1.0e-13);
            EXPECT_NEAR(coverage.source_covered_volumes[row], volume, 1.0e-13);
            EXPECT_NEAR(coverage.source_uncovered_volumes[row], 0.0, 1.0e-13);
        }
        for (size_t row = 0; row < target.num_owned_cells(); ++row)
        {
            const auto volume = target_mesh->cell_volume(static_cast<LO>(row));
            EXPECT_NEAR(coverage.target_cell_volumes[row], volume, 1.0e-13);
            EXPECT_NEAR(coverage.target_covered_volumes[row], volume, 1.0e-13);
            EXPECT_NEAR(coverage.target_uncovered_volumes[row], 0.0, 1.0e-13);
        }

        source.put_scalar(7.25);
        const auto constant_report = transfer.project(source, target, Quantity::Intensive);
        for (size_t row = 0; row < target.num_local_cells(); ++row)
            EXPECT_NEAR(target.local_value(static_cast<LO>(row)), 7.25, 1.0e-13);
        expect_scalar_report(constant_report, 174.0 * pi, 174.0 * pi, 0.0);
    }
}

TEST(MeshToMeshTransferMultiRankTest, PartialAnnularTopExpansionAndContractionReportInventory)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    constexpr double pi = std::numbers::pi;
    constexpr double area = 3.0 * pi;
    Options options;
    options.method = Method::ConservativeCellAverage;
    options.coverage_mode = CoverageMode::AllowPartial;
    for (const bool expansion : {true, false})
    {
        SCOPED_TRACE(expansion);
        const auto source_z = expansion
            ? SimpleFluid::ArrReal{0.0, 1.0, 2.0}
            : SimpleFluid::ArrReal{0.0, 1.0, 2.0, 3.0};
        const auto target_z = expansion
            ? SimpleFluid::ArrReal{0.0, 0.5, 1.5, 3.0}
            : SimpleFluid::ArrReal{0.0, 0.5, 1.5, 2.0};
        auto source_mesh = make_cylindrical(
            {{{1.0, 2.0}, {0.0, pi, 2.0 * pi}, source_z}}, true);
        auto target_mesh = make_cylindrical(
            {{{1.0, 2.0}, {0.0, pi, 2.0 * pi}, target_z}});
        Options strict = options;
        strict.coverage_mode = CoverageMode::RequireFull;
        EXPECT_THROW((Transfer(source_mesh, target_mesh, strict)), std::invalid_argument);
        Transfer transfer(source_mesh, target_mesh, options);
        ScalarField source(source_mesh, "moving_top_source");
        ScalarField target(target_mesh, "moving_top_target");
        source.put_scalar(1.0);
        target.put_scalar(17.0);
        EXPECT_THROW(transfer.apply(source, target), std::invalid_argument);
        for (size_t row = 0; row < target.num_local_cells(); ++row)
            EXPECT_DOUBLE_EQ(target.local_value(static_cast<LO>(row)), 17.0);

        for (const auto quantity : {Quantity::Intensive, Quantity::Extensive})
        {
            SCOPED_TRACE(static_cast<int>(quantity));
            for (size_t row = 0; row < source.num_owned_cells(); ++row)
            {
                const auto lid = static_cast<LO>(row);
                const auto id = cylindrical_id(*source_mesh, lid);
                double value = 2.0 + 4.0 * id.k;
                if (quantity == Quantity::Extensive)
                    value *= source_mesh->cell_volume(lid);
                source.set_owned_value(lid, value);
            }
            const auto report = transfer.project(source, target, quantity);
            const std::array<double, 3> expected_density{
                2.0, 4.0, expansion ? 2.0 : 6.0};
            for (size_t row = 0; row < target.num_local_cells(); ++row)
            {
                const auto lid = static_cast<LO>(row);
                double expected = expected_density[cylindrical_id(*target_mesh, lid).k];
                if (quantity == Quantity::Extensive)
                    expected *= target_mesh->cell_volume(lid);
                EXPECT_NEAR(target.local_value(lid), expected,
                            1.0e-12 * std::max(1.0, std::abs(expected)));
            }
            expect_scalar_report(report, (expansion ? 8.0 : 18.0) * area,
                                 8.0 * area, expansion ? 0.0 : 10.0 * area);
        }

        const auto& coverage = transfer.coverage();
        EXPECT_NEAR(coverage.source_volume, (expansion ? 2.0 : 3.0) * area, 1.0e-12);
        EXPECT_NEAR(coverage.target_volume, (expansion ? 3.0 : 2.0) * area, 1.0e-12);
        EXPECT_NEAR(coverage.overlap_volume, 2.0 * area, 1.0e-12);
        EXPECT_NEAR(coverage.uncovered_source_volume, expansion ? 0.0 : area, 1.0e-12);
        EXPECT_NEAR(coverage.uncovered_target_volume, expansion ? area : 0.0, 1.0e-12);
        ASSERT_EQ(coverage.source_cell_volumes.size(), source.num_owned_cells());
        ASSERT_EQ(coverage.source_covered_volumes.size(), source.num_owned_cells());
        ASSERT_EQ(coverage.source_uncovered_volumes.size(), source.num_owned_cells());
        ASSERT_EQ(coverage.target_cell_volumes.size(), target.num_owned_cells());
        ASSERT_EQ(coverage.target_covered_volumes.size(), target.num_owned_cells());
        ASSERT_EQ(coverage.target_uncovered_volumes.size(), target.num_owned_cells());
        for (size_t row = 0; row < source.num_owned_cells(); ++row)
        {
            const auto lid = static_cast<LO>(row);
            const auto volume = source_mesh->cell_volume(lid);
            const bool outside = !expansion && cylindrical_id(*source_mesh, lid).k == 2;
            EXPECT_NEAR(coverage.source_cell_volumes[row], volume, 1.0e-13);
            EXPECT_NEAR(coverage.source_covered_volumes[row], outside ? 0.0 : volume, 1.0e-13);
            EXPECT_NEAR(coverage.source_uncovered_volumes[row], outside ? volume : 0.0, 1.0e-13);
        }
        for (size_t row = 0; row < target.num_owned_cells(); ++row)
        {
            const auto lid = static_cast<LO>(row);
            const auto volume = target_mesh->cell_volume(lid);
            const bool cut = expansion && cylindrical_id(*target_mesh, lid).k == 2;
            EXPECT_NEAR(coverage.target_cell_volumes[row], volume, 1.0e-13);
            EXPECT_NEAR(coverage.target_covered_volumes[row], cut ? volume / 3.0 : volume, 1.0e-13);
            EXPECT_NEAR(coverage.target_uncovered_volumes[row], cut ? 2.0 * volume / 3.0 : 0.0, 1.0e-13);
        }
    }
}

TEST(MeshToMeshTransferMultiRankTest, ProjectionRejectsRankLocalInvalidValuesAndQuantityFlags)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    constexpr double pi = std::numbers::pi;
    auto source_mesh = make_cylindrical(
        {{{1.0, 2.0}, {0.0, pi, 2.0 * pi}, {0.0, 1.0, 2.0}}}, true);
    auto target_mesh = make_cylindrical(
        {{{1.0, 2.0}, {0.0, pi, 2.0 * pi}, {0.0, 0.5, 2.0}}});
    const int rank = source_mesh->owned_cell_map()->getComm()->getRank();
    Options options;
    options.method = Method::ConservativeCellAverage;
    options.coverage_mode = rank == 0 ? CoverageMode::AllowPartial : CoverageMode::RequireFull;
    EXPECT_THROW((Transfer(source_mesh, target_mesh, options)), std::invalid_argument);
    options.coverage_mode = rank == 0 ? static_cast<CoverageMode>(-1) : CoverageMode::RequireFull;
    EXPECT_THROW((Transfer(source_mesh, target_mesh, options)), std::invalid_argument);
    options.coverage_mode = CoverageMode::RequireFull;
    Transfer transfer(source_mesh, target_mesh, options);
    ScalarField source(source_mesh, "valid_source");
    ScalarField target(target_mesh, "unchanged_target");
    source.put_scalar(2.0);
    target.put_scalar(17.0);
    const auto invalid_quantity = rank == 0 ? static_cast<Quantity>(-1) : Quantity::Intensive;
    EXPECT_THROW((void)transfer.project(source, target, invalid_quantity), std::invalid_argument);
    const auto mismatched_quantity = rank == 0 ? Quantity::Extensive : Quantity::Intensive;
    EXPECT_THROW((void)transfer.project(source, target, mismatched_quantity), std::invalid_argument);
    if (rank == 0)
        source.set_owned_value(0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW((void)transfer.project(source, target, Quantity::Intensive), std::invalid_argument);
    for (size_t row = 0; row < target.num_local_cells(); ++row)
        EXPECT_DOUBLE_EQ(target.local_value(static_cast<LO>(row)), 17.0);
    source.put_scalar(2.0);
    const auto report = transfer.project(source, target, Quantity::Intensive);
    for (size_t row = 0; row < target.num_local_cells(); ++row)
        EXPECT_NEAR(target.local_value(static_cast<LO>(row)), 2.0, 1.0e-13);
    expect_scalar_report(report, 12.0 * pi, 12.0 * pi, 0.0);

    // A same-sized replacement can retain equivalent cell maps and epoch zero
    // while changing the native geometry behind the original handle object.
    auto alternate = make_cylindrical(
        {{{2.0, 3.0}, {0.0, pi, 2.0 * pi}, {0.0, 0.75, 2.0}}}, true);
    target.put_scalar(17.0);
    if (rank == 0)
        *source_mesh = *alternate;
    EXPECT_THROW((void)transfer.coverage(), std::runtime_error);
    EXPECT_THROW((void)transfer.project(source, target, Quantity::Intensive),
                 std::runtime_error);
    for (size_t row = 0; row < target.num_local_cells(); ++row)
        EXPECT_DOUBLE_EQ(target.local_value(static_cast<LO>(row)), 17.0);
}

TEST(MeshToMeshTransferMultiRankTest, NativeEmptyPartitionOptInMustAgreeAcrossRanks)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    constexpr double pi = std::numbers::pi;
    const int rank = Tpetra::getDefaultComm()->getRank();
    EXPECT_THROW((make_cylindrical(
        {{{1.0, 3.0}, {0.0, pi / 2.0}, {0.0, 2.0}}},
        false, rank == 0)), std::invalid_argument);
}

TEST(MeshToMeshTransferMultiRankTest, NativeSectorProjectionHandlesEmptyOwnersInBothDirections)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    constexpr double pi = std::numbers::pi;
    constexpr double volume = 4.0 * pi;
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();
    Options options;
    options.method = Method::ConservativeCellAverage;
    for (const bool refined_target : {false, true})
    {
        SCOPED_TRACE(refined_target);
        auto source_mesh = make_cylindrical(
            {{{1.0, 3.0}, {0.0, pi / 2.0}, {0.0, 2.0}}}, false, true);
        const auto target_r = refined_target
            ? SimpleFluid::ArrReal{1.0, std::sqrt(5.0), 3.0}
            : SimpleFluid::ArrReal{1.0, 3.0};
        auto target_mesh = make_cylindrical(
            {{target_r, {0.0, pi / 2.0}, {0.0, 2.0}}}, true, true);
        EXPECT_EQ(source_mesh->num_owned_cells(), rank == 0 ? 1U : 0U);
        const bool target_active = rank >= comm->getSize() - (refined_target ? 2 : 1);
        EXPECT_EQ(target_mesh->num_owned_cells(), target_active ? 1U : 0U);
        Transfer forward(source_mesh, target_mesh, options);
        Transfer reverse(target_mesh, source_mesh, options);
        const auto& coverage = forward.coverage();
        EXPECT_NEAR(coverage.source_volume, volume, 1.0e-12);
        EXPECT_NEAR(coverage.target_volume, volume, 1.0e-12);
        EXPECT_NEAR(coverage.overlap_volume, volume, 1.0e-12);
        EXPECT_NEAR(coverage.uncovered_source_volume, 0.0, 1.0e-12);
        EXPECT_NEAR(coverage.uncovered_target_volume, 0.0, 1.0e-12);
        EXPECT_EQ(coverage.source_cell_volumes.size(), source_mesh->num_owned_cells());
        EXPECT_EQ(coverage.source_covered_volumes.size(), source_mesh->num_owned_cells());
        EXPECT_EQ(coverage.source_uncovered_volumes.size(), source_mesh->num_owned_cells());
        EXPECT_EQ(coverage.target_cell_volumes.size(), target_mesh->num_owned_cells());
        EXPECT_EQ(coverage.target_covered_volumes.size(), target_mesh->num_owned_cells());
        EXPECT_EQ(coverage.target_uncovered_volumes.size(), target_mesh->num_owned_cells());
        const auto& reverse_coverage = reverse.coverage();
        EXPECT_EQ(reverse_coverage.source_cell_volumes.size(), target_mesh->num_owned_cells());
        EXPECT_EQ(reverse_coverage.target_cell_volumes.size(), source_mesh->num_owned_cells());

        ScalarField source(source_mesh, "small_source");
        ScalarField target(target_mesh, "small_target");
        for (const auto quantity : {Quantity::Intensive, Quantity::Extensive})
        {
            for (size_t row = 0; row < source.num_owned_cells(); ++row)
            {
                const auto lid = static_cast<LO>(row);
                source.set_owned_value(lid, quantity == Quantity::Intensive
                    ? 4.0 : 4.0 * source_mesh->cell_volume(lid));
            }
            const auto forward_report = forward.project(source, target, quantity);
            expect_scalar_report(forward_report, 4.0 * volume, 4.0 * volume, 0.0);
            for (size_t row = 0; row < target.num_local_cells(); ++row)
            {
                const auto lid = static_cast<LO>(row);
                const double expected = quantity == Quantity::Intensive
                    ? 4.0 : 4.0 * target_mesh->cell_volume(lid);
                EXPECT_NEAR(target.local_value(lid), expected, 1.0e-12);
            }
            source.put_scalar(-17.0);
            const auto reverse_report = reverse.project(target, source, quantity);
            expect_scalar_report(reverse_report, 4.0 * volume, 4.0 * volume, 0.0);
            for (size_t row = 0; row < source.num_local_cells(); ++row)
            {
                const auto lid = static_cast<LO>(row);
                EXPECT_NEAR(source.local_value(lid), quantity == Quantity::Intensive
                    ? 4.0 : 4.0 * volume, 1.0e-12);
            }
        }
    }
}

TEST(MeshToMeshTransferMultiRankTest, ProjectionRejectsNullRawMapsOnExcludedRanksBeforeMutation)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    constexpr double pi = std::numbers::pi;
    auto mesh = make_cylindrical(
        {{{1.0, 3.0}, {0.0, pi / 2.0}, {0.0, 2.0}}}, false, true);
    const auto map = mesh->owned_cell_map();
    const int rank = map->getComm()->getRank();
    Options options;
    options.method = Method::ConservativeCellAverage;
    Transfer transfer(mesh, mesh, options);
    Options interpolation_options;
    interpolation_options.method = Method::NearestCell;
    Transfer interpolation(mesh, mesh, interpolation_options);
    const auto reduced_map = map->removeEmptyProcesses();
    EXPECT_EQ(reduced_map.is_null(), rank != 0);
    for (const bool invalid_source : {true, false})
    {
        SCOPED_TRACE(invalid_source);
        Pack::multi_vector_type source(map, 1);
        Pack::multi_vector_type target(map, 1);
        source.putScalar(3.0);
        target.putScalar(17.0);
        auto& excluded = invalid_source ? source : target;
        excluded.replaceMap(reduced_map);
        EXPECT_EQ(excluded.getMap().is_null(), rank != 0);
        EXPECT_THROW(interpolation.apply(source, target), std::invalid_argument);
        EXPECT_THROW(transfer.apply(source, target), std::invalid_argument);
        EXPECT_THROW((void)transfer.project(source, target, Quantity::Extensive),
                     std::invalid_argument);
        if (mesh->num_owned_cells() != 0)
        {
            const auto values = target.getData(0);
            EXPECT_DOUBLE_EQ(values[0], 17.0);
        }
    }
}

TEST(MeshToMeshTransferMultiRankTest, CongruentEndpointContextsRejectRankLocalNullHandlesCollectively)
{
#ifndef HAVE_TEUCHOS_MPI
    GTEST_SKIP() << "Requires a Trilinos build with MPI.";
#else
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    const auto world = Tpetra::getDefaultComm();
    const auto source_comm = world->duplicate();
    const auto target_comm = world->duplicate();
    const auto* source_mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(source_comm.get());
    const auto* target_mpi = dynamic_cast<const Teuchos::MpiComm<int>*>(target_comm.get());
    ASSERT_NE(source_mpi, nullptr);
    ASSERT_NE(target_mpi, nullptr);
    int comparison = MPI_UNEQUAL;
    MPI_Comm_compare(*source_mpi->getRawMpiComm(), *target_mpi->getRawMpiComm(),
                     &comparison);
    ASSERT_EQ(comparison, MPI_CONGRUENT);
    auto source_mesh = make_unstructured_on_comm(source_comm);
    auto target_mesh = make_unstructured_on_comm(target_comm, true);
    const int rank = world->getRank();
    Options options;
    options.method = Method::NearestCell;
    // Every endpoint, when present, is collectively consistent on its own
    // context. Missing pointers must not select a different validation context.
    for (int missing = 0; missing < 3; ++missing)
    {
        SCOPED_TRACE(missing);
        SimpleFluid::SP<const Handle> source = rank == 0 && missing != 1
            ? nullptr : source_mesh;
        SimpleFluid::SP<const Handle> target = rank == 0 && missing != 0
            ? nullptr : target_mesh;
        EXPECT_THROW((Transfer(source, target, options)), std::invalid_argument);
        EXPECT_THROW((Transfer(*source_comm, source, target, options)),
                     std::invalid_argument);
    }

    ScalarField source(source_mesh, "duplicate_context_source");
    ScalarField target(target_mesh, "duplicate_context_target");
    const double source_x = source_mesh->cell_centroid(0).x;
    const double target_x = target_mesh->cell_centroid(0).x;
    source.set_owned_value(0, 100.0 + 3.0 * source_x);
    Transfer default_context(source_mesh, target_mesh, options);
    target.put_scalar(-999.0);
    default_context.apply(source, target);
    EXPECT_DOUBLE_EQ(target.value(0), 100.0 + 3.0 * target_x);

    Transfer explicit_context(*source_comm, source_mesh, target_mesh, options);
    source.set_owned_value(0, 131.0 + 3.0 * source_x);
    target.put_scalar(-999.0);
    explicit_context.apply(source, target);
    EXPECT_DOUBLE_EQ(target.value(0), 131.0 + 3.0 * target_x);
#endif
}

TEST(MeshToMeshTransferMultiRankTest, ExplicitConstructionContextSupportsASubgroup)
{
    SKIP_SINGLE_RANK(MeshToMeshTransferMultiRankTest);
    const auto world = Tpetra::getDefaultComm();
    const auto subgroup = world->split(world->getRank() == 0 ? 0 : -1, 0);
    if (subgroup.is_null()) return;
    const auto target_comm = subgroup->duplicate();
    auto source_mesh = make_unstructured_on_comm(subgroup);
    auto target_mesh = make_unstructured_on_comm(target_comm);
    ScalarField source(source_mesh, "subgroup_source");
    ScalarField target(target_mesh, "subgroup_target");
    source.put_scalar(29.0);
    target.put_scalar(-999.0);
    Transfer transfer(*subgroup, source_mesh, target_mesh);
    transfer.apply(source, target);
    EXPECT_DOUBLE_EQ(target.value(0), 29.0);
}
