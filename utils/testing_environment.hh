/**
 * @file testing_environment.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace utils_test
{


/**
 * @brief Global Google Test environment to initialize MPI and Kokkos.
 */
class KokkosEnvironment : public testing::Environment
{
public:
    void SetUp() override
    {
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized)
        {
            MPI_Init(nullptr, nullptr);
            d_initialized_mpi = true;
        }

        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize();
            d_initialized_kokkos = true;
        }
    }

    void TearDown() override
    {
        if (d_initialized_kokkos && Kokkos::is_initialized())
        {
            Kokkos::finalize();
        }

        int mpi_finalized = 0;
        MPI_Finalized(&mpi_finalized);
        if (d_initialized_mpi && !mpi_finalized)
        {
            MPI_Finalize();
        }
    }

private:
    bool d_initialized_mpi = false;
    bool d_initialized_kokkos = false;
};

}