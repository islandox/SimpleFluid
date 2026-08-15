/**
 * @file testMPIInterface.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Multi-rank tests for the lightweight MPI wrapper.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <gtest/gtest.h>

#include "parallel/MPI_interface.hh"
#include "utils/testing_environment.hh"

#include <array>
#include <complex>
#include <vector>

namespace
{

using utils_test::KokkosEnvironment;
testing::Environment* const kokkos_environment =
    testing::AddGlobalTestEnvironment(new KokkosEnvironment);

std::pair<int, int> rank_and_size()
{
    int rank = -1;
    int size = 0;
    EXPECT_EQ(my_mpi::comm_rank(rank), MPI_SUCCESS);
    EXPECT_EQ(my_mpi::comm_size(size), MPI_SUCCESS);
    return {rank, size};
}

} // namespace

/**
 * @brief Verifies communicator management, rank/size queries, datatype
 * mappings, and MPI timing helpers.
 */
TEST(MPIInterfaceTest, ExposesCommunicatorAndDatatypeHelpers)
{
    const auto [rank, size] = rank_and_size();
    EXPECT_GE(rank, 0);
    EXPECT_GE(size, 1);
    EXPECT_EQ(my_mpi::type_trait<int>(), MPI_INT32_T);
    EXPECT_EQ(my_mpi::type_trait<double>(), MPI_DOUBLE);
    EXPECT_EQ(my_mpi::type_trait<long double>(), MPI_LONG_DOUBLE);
    EXPECT_EQ(
        my_mpi::type_trait<std::complex<double>>(),
        MPI_CXX_DOUBLE_COMPLEX);
    EXPECT_EQ(my_mpi::type_trait<bool>(), MPI_CXX_BOOL);

    my_mpi::Comm duplicate = MPI_COMM_NULL;
    ASSERT_EQ(my_mpi::comm_dup(duplicate), MPI_SUCCESS);
    my_mpi::set_comm(duplicate);
    int duplicate_size = 0;
    EXPECT_EQ(my_mpi::comm_size(duplicate_size), MPI_SUCCESS);
    EXPECT_EQ(duplicate_size, size);
    my_mpi::set_comm(MPI_COMM_WORLD);
    EXPECT_EQ(my_mpi::comm_free(duplicate), MPI_SUCCESS);
    EXPECT_EQ(duplicate, MPI_COMM_NULL);
    EXPECT_GT(my_mpi::wtick(), 0.0);
    EXPECT_GE(my_mpi::wtime(), 0.0);
}

/**
 * @brief Verifies blocking and nonblocking point-to-point transfers between
 * two ranks, including request completion.
 */
TEST(MPIInterfaceTest, SupportsBlockingAndNonBlockingPointToPoint)
{
    SKIP_SINGLE_RANK(SupportsBlockingAndNonBlockingPointToPoint);

    const auto [rank, size] = rank_and_size();
    ASSERT_GE(size, 2);
    constexpr int blocking_tag = 41;
    constexpr int nonblocking_tag = 42;

    if (rank == 0)
    {
        const int value = 17;
        EXPECT_EQ(
            my_mpi::send(&value, 1, 1, blocking_tag),
            MPI_SUCCESS);
    }
    else if (rank == 1)
    {
        int value = 0;
        my_mpi::Status status{};
        EXPECT_EQ(
            my_mpi::recv(
                &value, 1, 0, blocking_tag, status),
            MPI_SUCCESS);
        EXPECT_EQ(value, 17);
    }
    ASSERT_EQ(my_mpi::barrier(), MPI_SUCCESS);

    if (rank == 0)
    {
        const int value = 23;
        my_mpi::Request request = MPI_REQUEST_NULL;
        my_mpi::Status status{};
        ASSERT_EQ(
            my_mpi::isend(
                &value, 1, 1, nonblocking_tag, request),
            MPI_SUCCESS);
        EXPECT_EQ(my_mpi::wait(request, status), MPI_SUCCESS);
    }
    else if (rank == 1)
    {
        int value = 0;
        int complete = 0;
        my_mpi::Request request = MPI_REQUEST_NULL;
        my_mpi::Status status{};
        ASSERT_EQ(
            my_mpi::irecv(
                &value, 1, 0, nonblocking_tag, request),
            MPI_SUCCESS);
        while (!complete)
        {
            ASSERT_EQ(
                my_mpi::test(request, complete, status),
                MPI_SUCCESS);
        }
        EXPECT_EQ(value, 23);
    }
    EXPECT_EQ(my_mpi::barrier(), MPI_SUCCESS);
}

/**
 * @brief Verifies broadcast, gather, variable gather, barrier, and scalar
 * reduction wrappers across the test communicator.
 */
TEST(MPIInterfaceTest, SupportsCollectivesAndReductions)
{
    SKIP_SINGLE_RANK(SupportsCollectivesAndReductions);

    const auto [rank, size] = rank_and_size();
    ASSERT_GE(size, 2);

    int broadcast_value = rank == 0 ? 9 : 0;
    EXPECT_EQ(
        my_mpi::broadcast(&broadcast_value, 1, 0),
        MPI_SUCCESS);
    EXPECT_EQ(broadcast_value, 9);

    std::vector<int> gathered(static_cast<size_t>(size), -1);
    EXPECT_EQ(
        my_mpi::gather(
            &rank, 1, gathered.data(), 1, 0),
        MPI_SUCCESS);
    if (rank == 0)
    {
        for (int source = 0; source < size; ++source)
        {
            EXPECT_EQ(gathered[static_cast<size_t>(source)], source);
        }
    }

    std::vector<int> counts(static_cast<size_t>(size), 1);
    std::vector<int> displacements(static_cast<size_t>(size));
    for (int source = 0; source < size; ++source)
    {
        displacements[static_cast<size_t>(source)] = source;
    }
    std::vector<int> all_gathered(static_cast<size_t>(size), -1);
    EXPECT_EQ(
        my_mpi::allgatherv(
            &rank, 1, all_gathered.data(),
            counts.data(), displacements.data()),
        MPI_SUCCESS);
    for (int source = 0; source < size; ++source)
    {
        EXPECT_EQ(
            all_gathered[static_cast<size_t>(source)],
            source);
    }

    std::vector<int> send(static_cast<size_t>(size));
    std::vector<int> receive(static_cast<size_t>(size), -1);
    for (int destination = 0; destination < size; ++destination)
    {
        send[static_cast<size_t>(destination)] =
            rank * 100 + destination;
    }
    EXPECT_EQ(
        my_mpi::alltoall(
            send.data(), 1, receive.data(), 1),
        MPI_SUCCESS);
    for (int source = 0; source < size; ++source)
    {
        EXPECT_EQ(
            receive[static_cast<size_t>(source)],
            source * 100 + rank);
    }

    const int local = rank + 1;
    int sum = 0;
    int maximum = 0;
    int minimum = 0;
    EXPECT_EQ(my_mpi::global_sum(local, sum), MPI_SUCCESS);
    EXPECT_EQ(my_mpi::global_max(local, maximum), MPI_SUCCESS);
    EXPECT_EQ(my_mpi::global_min(local, minimum), MPI_SUCCESS);
    EXPECT_EQ(sum, size * (size + 1) / 2);
    EXPECT_EQ(maximum, size);
    EXPECT_EQ(minimum, 1);
}
