/**
 * @file testing_environment.hh
 * @author islandox (59904740+islandox@users.noreply.github.com)
 * @brief Google Test environment that initializes MPI and Kokkos for unit tests.
 * @version 0.1
 * @date 2026-05-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Tpetra_Core.hpp>
#include "parallel/MPI_interface.hh"

namespace utils_test
{

my_mpi::ErrorCode error_code;

/**
 * @brief Global Google Test environment to initialize MPI and Kokkos.
 */
class KokkosEnvironment : public testing::Environment
{
public:
    void SetUp() override
    {
        if (!my_mpi::initialized(error_code))
        {
            my_mpi::init(nullptr, nullptr);
            d_initialized_mpi = true;
        }

        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize();
            d_initialized_kokkos = true;
        }

        if (!Tpetra::isInitialized())
        {
            Tpetra::initialize(nullptr, nullptr);
            d_initialized_tpetra = true;
        }
    }

    void TearDown() override
    {
        if (d_initialized_kokkos && Kokkos::is_initialized())
        {
            Kokkos::finalize();
        }

        if (d_initialized_tpetra && Tpetra::isInitialized())
        {
            Tpetra::finalize();
        }

        if (d_initialized_mpi && !my_mpi::finalized(error_code))
        {
            my_mpi::finalize();
        }
    }

private:
    bool d_initialized_mpi = false;
    bool d_initialized_kokkos = false;
    bool d_initialized_tpetra = false;
};

}