/**
 * @file testDICPreconditionerMultiRank.cc
 * @brief Distributed diagonal incomplete-Cholesky factors and Belos integration.
 */

#include <gtest/gtest.h>

#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::detail
{

template<TpetraTypePack Pack>
struct BelosLinearSolverTestAccess
{
    static std::size_t preconditioner_setup_count(
        const BelosLinearSolver<Pack>& solver) noexcept
    {
        return solver.d_preconditioner_setup_count;
    }
};

template<TpetraTypePack Pack>
struct DICPreconditionerTestAccess
{
    static std::size_t stage_count(const DICPreconditioner<Pack>& inverse)
    {
        return inverse.d_stage_offsets.size() - 1;
    }

    static std::array<std::size_t, 2> transfer_rows(const DICPreconditioner<Pack>& inverse)
    {
        std::array<std::size_t, 2> result{};
        for (const auto* transfers : {&inverse.d_forward_transfers, &inverse.d_backward_transfers})
            for (const auto& transfer : *transfers)
            {
                result[0] += transfer.importer->getNumRemoteIDs();
                result[1] += transfer.importer->getNumSameIDs()
                    + transfer.importer->getNumPermuteIDs();
            }
        return result;
    }
};

} // namespace SimpleFluid::detail

namespace
{

using Pack = SimpleFluid::TpetraTypes<>;
using GO = Pack::global_ordinal_type;
using LO = Pack::local_ordinal_type;
using Comm = Teuchos::Comm<int>;
using Inverse = SimpleFluid::detail::DICPreconditioner<Pack>;
constexpr std::size_t row_count = 8;
constexpr std::array<GO, row_count> global_ids{11, 28, 53, 92, 117, 148, 203, 246};
using DenseMatrix = std::array<std::array<double, row_count>, row_count>;
enum class Partition { Interleaved, Contiguous };

testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

/**
 * @brief Prescribe D and L, then form both the DIC input A and factor product M.
 *
 * A retains L and its transpose, with diagonal entries chosen to produce the
 * prescribed positive pivots. M is an independent dense matrix multiplication
 * of (D+L) D^-1 (D+L)^T. Shared predecessors create nonzero fill in M, so an
 * exact inverse of A, ILU factors, or rank-local DIC cannot satisfy this oracle.
 */
struct FactorFixture
{
    DenseMatrix matrix{};
    DenseMatrix product{};

    explicit FactorFixture(double scale = 1.0)
    {
        const std::array<double, row_count> diagonal{2, 3, 4, 5, 6, 7, 8, 9};
        DenseMatrix factor{};
        for (std::size_t row = 0; row < row_count; ++row)
        {
            factor[row][row] = diagonal[row];
            if (row >= 1)
                factor[row][row - 1] = -1.0;
            if (row >= 2)
                factor[row][row - 2] = -0.25;
        }
        factor[3][0] = -0.5;
        factor[7][0] = -0.125;
        for (std::size_t row = 0; row < row_count; ++row)
        {
            matrix[row][row] = diagonal[row];
            for (std::size_t column = 0; column < row; ++column)
            {
                matrix[row][column] = matrix[column][row] = factor[row][column];
                matrix[row][row] += factor[row][column] * factor[row][column]
                    / diagonal[column];
            }
            for (std::size_t column = 0; column < row_count; ++column)
                for (std::size_t inner = 0; inner < row_count; ++inner)
                    product[row][column] += factor[row][inner]
                        * factor[column][inner] / diagonal[inner];
        }
        for (std::size_t row = 0; row < row_count; ++row)
            for (std::size_t column = 0; column < row_count; ++column)
            {
                matrix[row][column] *= scale;
                product[row][column] *= scale;
            }
    }
};

std::size_t global_index(GO gid)
{
    return std::lower_bound(global_ids.begin(), global_ids.end(), gid)
        - global_ids.begin();
}

/** @brief Interleave global dependencies across ranks and reverse local order. */
Teuchos::RCP<const Pack::map_type> make_map(
    const Teuchos::RCP<const Comm>& comm, bool empty_last_rank = false,
    bool permute_rank_zero = false, Partition partition = Partition::Interleaved)
{
    const int owners = comm->getSize() - (empty_last_rank ? 1 : 0);
    Teuchos::Array<GO> local_ids;
    for (std::size_t row = 0; row < row_count; ++row)
    {
        const auto owner = partition == Partition::Contiguous
            ? static_cast<int>(row * owners / row_count)
            : static_cast<int>(row % owners);
        if (owner == comm->getRank())
            local_ids.push_back(global_ids[row]);
    }
    if (!permute_rank_zero || comm->getRank() != 0)
        std::reverse(local_ids.begin(), local_ids.end());
    return Teuchos::rcp(new Pack::map_type(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        local_ids(), global_ids.front(), comm));
}

Teuchos::RCP<Pack::matrix_type> make_matrix(
    const Teuchos::RCP<const Pack::map_type>& map, const DenseMatrix& entries,
    const Teuchos::RCP<const Pack::map_type>& domain = Teuchos::null)
{
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, row_count));
    for (std::size_t local_row = 0; local_row < map->getLocalNumElements(); ++local_row)
    {
        const auto gid = map->getGlobalElement(static_cast<LO>(local_row));
        const auto row = global_index(gid);
        Teuchos::Array<GO> columns;
        Teuchos::Array<double> values;
        for (std::size_t column = 0; column < row_count; ++column)
            if (entries[row][column] != 0.0 || row == column)
            {
                columns.push_back(global_ids[column]);
                values.push_back(entries[row][column]);
            }
        matrix->insertGlobalValues(gid, columns(), values());
    }
    matrix->fillComplete(domain.is_null() ? map : domain, map);
    return matrix;
}

double exact_value(std::size_t row, std::size_t column)
{
    const double index = static_cast<double>(row + 1);
    switch (column % 3)
    {
        case 0: return index;
        case 1: return index * index - 0.5;
        default: return row % 2 == 0 ? 2.0 : -3.0;
    }
}

void fill_exact(Pack::multi_vector_type& vector)
{
    for (std::size_t column = 0; column < vector.getNumVectors(); ++column)
    {
        auto values = vector.getDataNonConst(column);
        for (std::size_t row = 0; row < vector.getLocalLength(); ++row)
            values[row] = exact_value(global_index(
                vector.getMap()->getGlobalElement(static_cast<LO>(row))), column);
    }
}

/** @brief Form a known RHS using dense multiplication, without a triangular solve. */
void fill_product(Pack::multi_vector_type& vector, const DenseMatrix& matrix)
{
    for (std::size_t column = 0; column < vector.getNumVectors(); ++column)
    {
        auto values = vector.getDataNonConst(column);
        for (std::size_t local_row = 0; local_row < vector.getLocalLength(); ++local_row)
        {
            const auto row = global_index(
                vector.getMap()->getGlobalElement(static_cast<LO>(local_row)));
            values[local_row] = 0.0;
            for (std::size_t inner = 0; inner < row_count; ++inner)
                values[local_row] += matrix[row][inner] * exact_value(inner, column);
        }
    }
}

void expect_exact(const Pack::multi_vector_type& vector, double scale = 1.0,
                  double tolerance = 1.0e-11)
{
    for (std::size_t column = 0; column < vector.getNumVectors(); ++column)
    {
        const auto values = vector.getData(column);
        for (std::size_t local_row = 0; local_row < vector.getLocalLength(); ++local_row)
        {
            const auto row = global_index(
                vector.getMap()->getGlobalElement(static_cast<LO>(local_row)));
            EXPECT_NEAR(values[local_row], scale * exact_value(row, column), tolerance)
                << "rank=" << vector.getMap()->getComm()->getRank()
                << " global row=" << global_ids[row] << " RHS=" << column;
        }
    }
}

void check_factor_application(const Teuchos::RCP<const Pack::map_type>& map)
{
    const FactorFixture fixture;
    const auto matrix = make_matrix(map, fixture.matrix);
    Inverse inverse(*matrix);
    Pack::multi_vector_type rhs(map, 3, true);
    Pack::multi_vector_type result(map, 3, true);
    fill_product(rhs, fixture.product);
    result.putScalar(std::numeric_limits<double>::quiet_NaN());
    inverse.apply(rhs, result);
    expect_exact(result);
    inverse.apply(rhs, result, Teuchos::NO_TRANS, 2.0, 3.0);
    expect_exact(result, 5.0);
    inverse.apply(rhs, rhs);
    expect_exact(rhs);

    rhs.putScalar(std::numeric_limits<double>::quiet_NaN());
    inverse.apply(rhs, result, Teuchos::NO_TRANS, 0.0, 0.0);
    for (std::size_t column = 0; column < result.getNumVectors(); ++column)
        for (const auto value : result.getData(column))
            EXPECT_DOUBLE_EQ(value, 0.0);

    fill_exact(result);
    inverse.apply(rhs, result, Teuchos::NO_TRANS, 0.0, -2.0);
    expect_exact(result, -2.0);

    // Recreate workspace for a different RHS count, then return to three RHS.
    Pack::multi_vector_type single_rhs(map, 1, true);
    fill_product(single_rhs, fixture.product);
    inverse.apply(single_rhs, single_rhs);
    expect_exact(single_rhs);
    fill_product(rhs, fixture.product);
    inverse.apply(rhs, result);
    expect_exact(result);

    // Differently ordered overlapping column views require staging every RHS
    // before writing any output column, not just copying each column in turn.
    Pack::multi_vector_type storage(map, 4, true);
    const Teuchos::Array<std::size_t> input_columns{3, 0, 2};
    const Teuchos::Array<std::size_t> output_columns{2, 3, 0};
    auto input_view = storage.subViewNonConst(input_columns());
    auto output_view = storage.subViewNonConst(output_columns());
    fill_product(*input_view, fixture.product);
    inverse.apply(*input_view, *output_view, Teuchos::NO_TRANS, 2.0);
    expect_exact(*output_view, 2.0);
}

} // namespace

TEST(DICPreconditionerMultiRankTest, CrossRankFactorMatchesGlobalOrderWithPermutedLocalIds)
{
    SKIP_SINGLE_RANK(CrossRankFactorMatchesGlobalOrderWithPermutedLocalIds);
    check_factor_application(make_map(Tpetra::getDefaultComm()));
}

TEST(DICPreconditionerMultiRankTest, ContiguousPartitionCombinesLocalAndRemoteDependencies)
{
    SKIP_SINGLE_RANK(ContiguousPartitionCombinesLocalAndRemoteDependencies);
    check_factor_application(make_map(
        Tpetra::getDefaultComm(), false, false, Partition::Contiguous));
}

TEST(DICPreconditionerMultiRankTest, StageTransfersSendOnlyConsumedRemoteDependencies)
{
    SKIP_SINGLE_RANK(StageTransfersSendOnlyConsumedRemoteDependencies);
    const auto comm = Tpetra::getDefaultComm();
    const auto map = make_map(comm);
    const FactorFixture fixture;
    const auto matrix = make_matrix(map, fixture.matrix);
    Inverse inverse(*matrix);
    using Access = SimpleFluid::detail::DICPreconditionerTestAccess<Pack>;
    const auto transferred = Access::transfer_rows(inverse);
    EXPECT_EQ(transferred[1], 0U);

    // The interleaved chain has one ready global row per stage. Every directed
    // edge crossing the partition must therefore be received once; no other
    // column is needed. This oracle comes from the input graph and ownership.
    std::size_t expected_remote = 0;
    for (const auto gid : map->getLocalElementList())
    {
        const auto row = global_index(gid);
        for (std::size_t column = 0; column < row_count; ++column)
            if (fixture.matrix[row][column] != 0.0
                && map->getLocalElement(global_ids[column])
                    == Teuchos::OrdinalTraits<LO>::invalid())
                ++expected_remote;
    }
    EXPECT_EQ(transferred[0], expected_remote);

    const Pack::import_type full_import(map, matrix->getColMap());
    const auto old_remote = full_import.getNumRemoteIDs()
        * 2 * (Access::stage_count(inverse) - 1);
    const std::array<unsigned long long, 2> local{
        static_cast<unsigned long long>(transferred[0]),
        static_cast<unsigned long long>(old_remote)};
    std::array<unsigned long long, 2> global{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM,
        static_cast<int>(local.size()), local.data(), global.data());
    EXPECT_LT(global[0], global[1]);
}

TEST(DICPreconditionerMultiRankTest, DirectionalApplicationsRemainSymmetricPositiveDefinite)
{
    SKIP_SINGLE_RANK(DirectionalApplicationsRemainSymmetricPositiveDefinite);
    const auto map = make_map(Tpetra::getDefaultComm());
    const auto matrix = make_matrix(map, FactorFixture{}.matrix);
    Inverse inverse(*matrix);
    Pack::multi_vector_type x(map, 1, true), y(map, 1, true);
    Pack::multi_vector_type px(map, 1, true), py(map, 1, true);
    fill_exact(x);
    {
        const auto values = y.getDataNonConst(0);
        for (std::size_t row = 0; row < map->getLocalNumElements(); ++row)
            values[row] = exact_value(global_index(map->getGlobalElement(static_cast<LO>(row))), 2);
    }
    inverse.apply(x, px);
    inverse.apply(y, py);
    Teuchos::Array<double> x_py(1), y_px(1), x_px(1), y_py(1);
    x.dot(py, x_py());
    y.dot(px, y_px());
    x.dot(px, x_px());
    y.dot(py, y_py());
    EXPECT_NEAR(x_py[0], y_px[0], 1.0e-12);
    EXPECT_GT(x_px[0], 0.0);
    EXPECT_GT(y_py[0], 0.0);
}

TEST(DICPreconditionerMultiRankTest, EmptyRankParticipatesInSetupAndApplication)
{
    SKIP_SINGLE_RANK(EmptyRankParticipatesInSetupAndApplication);
    const auto comm = Tpetra::getDefaultComm();
    const auto map = make_map(comm, true);
    if (comm->getRank() == comm->getSize() - 1)
        EXPECT_EQ(map->getLocalNumElements(), 0U);
    check_factor_application(map);
}

TEST(DICPreconditionerMultiRankTest, UsesMatrixSubcommunicator)
{
    const auto world = Tpetra::getDefaultComm();
    if (world->getSize() < 4)
        GTEST_SKIP() << "Four ranks provide a proper distributed subcommunicator.";
    Teuchos::Array<int> members;
    for (int rank = 0; rank < world->getSize(); rank += 2)
        members.push_back(rank);
    const auto comm = world->createSubcommunicator(members());
    if (!comm.is_null())
        check_factor_application(make_map(comm));
    // Nonmembers perform no DIC collectives; accidental world use cannot pass.
    world->barrier();
}

TEST(DICPreconditionerMultiRankTest, PCGSolvesMultipleRhsAndReusesReplacesResetsFactors)
{
    SKIP_SINGLE_RANK(PCGSolvesMultipleRhsAndReusesReplacesResetsFactors);
    const auto map = make_map(Tpetra::getDefaultComm());
    auto first = make_matrix(map, FactorFixture{}.matrix);
    auto second = make_matrix(map, FactorFixture{2.0}.matrix);
    Pack::multi_vector_type rhs(map, 3, true);
    Pack::multi_vector_type solution(map, 3, true);
    SimpleFluid::LinearSolverOptions options;
    options.backend = SimpleFluid::LinearSolverBackend::Cg;
    options.preconditioner = SimpleFluid::LinearPreconditioner::DIC;
    options.reuse_preconditioner = true;
    options.max_iterations = 20;
    options.tolerance = 1.0e-12;
    SimpleFluid::BelosLinearSolver<Pack> solver;
    using Access = SimpleFluid::detail::BelosLinearSolverTestAccess<Pack>;
    for (int step = 0; step < 5; ++step)
    {
        SCOPED_TRACE(step);
        const auto matrix = step < 2 ? first : second;
        if (step == 3)
            solver.reset();
        if (step == 4)
            options.reuse_preconditioner = false;
        fill_product(rhs, FactorFixture{step < 2 ? 1.0 : 2.0}.matrix);
        solution.putScalar(0.0);
        const auto statistics = solver.solve_with_statistics(matrix, rhs, solution, options);
        EXPECT_TRUE(statistics.converged);
        EXPECT_LE(statistics.achieved_tolerance, options.tolerance);
        expect_exact(solution, 1.0, 1.0e-9);
        EXPECT_EQ(Access::preconditioner_setup_count(solver),
                  static_cast<std::size_t>(step < 2 ? 1 : step));
    }
}

TEST(DICPreconditionerMultiRankTest, RejectsRankLocalInvalidEntriesAndPivotsCollectively)
{
    SKIP_SINGLE_RANK(RejectsRankLocalInvalidEntriesAndPivotsCollectively);
    const auto map = make_map(Tpetra::getDefaultComm());
    for (int invalid_kind = 0; invalid_kind < 6; ++invalid_kind)
    {
        SCOPED_TRACE(invalid_kind);
        auto entries = FactorFixture{}.matrix;
        switch (invalid_kind)
        {
            case 0: entries[1][1] = std::numeric_limits<double>::quiet_NaN(); break;
            case 1: entries[1][0] = -1.5; break;
            case 2: entries[1][1] = 0.0; break;
            case 3: entries[1][1] = 0.25; break;
            case 4: entries[7][7] = 0.01; break;
            default:
                entries[0][0] = std::numeric_limits<double>::infinity();
                entries[1][1] = -1.0;
                break;
        }
        const auto matrix = make_matrix(map, entries);
        EXPECT_THROW((Inverse(*matrix)), std::invalid_argument);
    }
}

TEST(DICPreconditionerMultiRankTest, RejectsRankLocalFillStateAndDomainMapMismatchCollectively)
{
    SKIP_SINGLE_RANK(RejectsRankLocalFillStateAndDomainMapMismatchCollectively);
    const auto comm = Tpetra::getDefaultComm();
    const auto map = make_map(comm);
    const auto wrong_domain = make_map(comm, false, true);
    const auto mismatched = make_matrix(map, FactorFixture{}.matrix, wrong_domain);
    EXPECT_THROW((Inverse(*mismatched)), std::invalid_argument);

    const auto unfinished = make_matrix(map, FactorFixture{}.matrix);
    if (comm->getRank() == 0)
        unfinished->resumeFill();
    EXPECT_THROW((Inverse(*unfinished)), std::invalid_argument);
}

TEST(DICPreconditionerMultiRankTest, ApplyRejectsRankLocalMapCountAndModeMismatchCollectively)
{
    SKIP_SINGLE_RANK(ApplyRejectsRankLocalMapCountAndModeMismatchCollectively);
    const auto comm = Tpetra::getDefaultComm();
    const auto map = make_map(comm);
    const auto wrong_map = make_map(comm, false, true);
    const auto matrix = make_matrix(map, FactorFixture{}.matrix);
    Inverse inverse(*matrix);
    Pack::multi_vector_type rhs(map, 3, true);
    Pack::multi_vector_type result(map, 3, true);
    Pack::multi_vector_type wrong_rhs(wrong_map, 3, true);
    Pack::multi_vector_type wrong_result(wrong_map, 3, true);
    EXPECT_THROW(inverse.apply(wrong_rhs, result), std::invalid_argument);
    EXPECT_THROW(inverse.apply(rhs, wrong_result), std::invalid_argument);

    Pack::multi_vector_type wrong_count(map, comm->getRank() == 0 ? 2 : 3, true);
    EXPECT_THROW(inverse.apply(rhs, wrong_count), std::invalid_argument);
    const auto mode = comm->getRank() == 0 ? Teuchos::TRANS : Teuchos::NO_TRANS;
    EXPECT_THROW(inverse.apply(rhs, result, mode), std::invalid_argument);
}

TEST(DICPreconditionerMultiRankTest, ApplyRejectsDivergentCommunicationShapeCollectively)
{
    SKIP_SINGLE_RANK(ApplyRejectsDivergentCommunicationShapeCollectively);
    const auto comm = Tpetra::getDefaultComm();
    const auto map = make_map(comm);
    const auto matrix = make_matrix(map, FactorFixture{}.matrix);
    Inverse inverse(*matrix);
    const auto columns = comm->getRank() == 0 ? 2 : 3;
    Pack::multi_vector_type varying_rhs(map, columns, true);
    Pack::multi_vector_type varying_result(map, columns, true);
    EXPECT_THROW(inverse.apply(varying_rhs, varying_result), std::invalid_argument);
    Pack::multi_vector_type rhs(map, 3, true);
    Pack::multi_vector_type result(map, 3, true);
    EXPECT_THROW(inverse.apply(rhs, result, Teuchos::NO_TRANS,
                              comm->getRank() == 0 ? 0.0 : 1.0),
                 std::invalid_argument);
}

TEST(DICPreconditionerMultiRankTest,
     ApplyRejectsForeignCommunicatorsAndAcceptsCongruentDuplicates)
{
    const auto world = Tpetra::getDefaultComm();
    if (world->getSize() != 4)
        GTEST_SKIP() << "Four ranks provide different communicators of equal size.";
    const auto comm = world->split(world->getRank() / 2, world->getRank());
    const auto alternate = world->split(world->getRank() % 2, world->getRank());
    const auto map = make_map(comm);
    const FactorFixture fixture;
    const auto matrix = make_matrix(map, fixture.matrix);
    Inverse inverse(*matrix);

    // Preserve local IDs, global size, and index base while changing group
    // membership. On world rank zero even communicator rank and size match.
    const auto foreign_map = Teuchos::rcp(new Pack::map_type(
        map->getGlobalNumElements(), map->getLocalElementList(),
        map->getIndexBase(), alternate));
    Pack::multi_vector_type rhs(map, 3, true);
    Pack::multi_vector_type foreign_rhs(foreign_map, 3, true);
    Pack::multi_vector_type result(map, 3, true);
    fill_product(rhs, fixture.product);
    fill_product(foreign_rhs, fixture.product);
    const auto& selected_rhs = comm->getRank() == 0 ? foreign_rhs : rhs;
    EXPECT_THROW(inverse.apply(selected_rhs, result), std::invalid_argument);

    // MPI communicator duplication changes the handle but retains membership
    // and ordering, so otherwise identical maps must remain valid inputs.
    const auto duplicate = comm->duplicate();
    const auto duplicate_map = Teuchos::rcp(new Pack::map_type(
        map->getGlobalNumElements(), map->getLocalElementList(),
        map->getIndexBase(), duplicate));
    Pack::multi_vector_type duplicate_rhs(duplicate_map, 3, true);
    Pack::multi_vector_type duplicate_result(duplicate_map, 3, true);
    fill_product(duplicate_rhs, fixture.product);
    inverse.apply(duplicate_rhs, duplicate_result);
    expect_exact(duplicate_result);
}
