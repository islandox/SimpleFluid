/** @file testGeneralVoFFailures.cc
 * @brief Independent collective rejection records, bounded context, and local causes.
 */
#include "FVM/details/VoFFailure.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <gtest/gtest.h>
#include <string>

namespace
{
using namespace SimpleFluid;
namespace failure = SimpleFluid::detail::vof_failure;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(GeneralVoFFailures, NonzeroEmptyOwnerSuppliesIdenticalRecordAndOriginalCause)
{
    const auto comm = Tpetra::getDefaultComm();
    const int origin = comm->getSize() > 1 ? 1 : 0;
    // Only rank zero owns a row. The failing rank has no work in the MPI runs.
    Tpetra::Map<> owned(1, comm->getRank() == 0 ? std::size_t(1) : std::size_t(0), 0, comm);
    if (comm->getSize() > 1 && comm->getRank() == origin)
        EXPECT_EQ(owned.getLocalNumElements(), 0u);
    bool caught = false;
    try
    {
        failure::collective(*comm,
            [&]
            {
                if (comm->getRank() != origin)
                    return;
                failure::Scope outer("sweep_partition");
                outer.record.face = 91;
                outer.record.mark(failure::Field::Face);
                outer.record.arithmetic_mode = ArithmeticMode::Ordinary;
                outer.record.mark(failure::Field::Arithmetic);
                failure::Scope inner("recipient_overlay");
                inner.record.donor = 17;
                inner.record.recipient = 23;
                inner.record.mark(failure::Field::Donor);
                inner.record.mark(failure::Field::Recipient);
                inner.record.volume_budget = 0.125L;
                inner.record.mark(failure::Field::VolumeBudget);
                throw PLIC::GeometryError(
                    PLIC::GeometryErrorCode::UnresolvedArithmetic, "independent arithmetic cause");
            });
    }
    catch (const failure::CollectiveFailure<std::invalid_argument>& error)
    {
        caught = true;
        const auto& record = error.record();
        EXPECT_EQ(record.origin_rank, origin);
        EXPECT_EQ(record.category, failure::Category::Arithmetic);
        EXPECT_EQ(record.geometry_code, int(PLIC::GeometryErrorCode::UnresolvedArithmetic));
        EXPECT_STREQ(record.stage.data(), "recipient_overlay");
        EXPECT_STREQ(record.message.data(), "independent arithmetic cause");
        EXPECT_EQ(record.face, 91u);
        EXPECT_EQ(record.donor, 17u);
        EXPECT_EQ(record.recipient, 23u);
        EXPECT_EQ(record.volume_budget, 0.125L);
        EXPECT_TRUE(record.has(failure::Field::Arithmetic));
        EXPECT_FALSE(record.has(failure::Field::Cell));
        EXPECT_FALSE(record.has(failure::Field::Construction));
        EXPECT_FALSE(record.has(failure::Field::Retry));
        EXPECT_EQ(bool(error.local_cause()), comm->getRank() == origin);
        if (error.local_cause())
        {
            try
            {
                std::rethrow_exception(error.local_cause());
            }
            catch (const PLIC::GeometryError& original)
            {
                EXPECT_EQ(original.code(), PLIC::GeometryErrorCode::UnresolvedArithmetic);
                EXPECT_STREQ(original.what(), "independent arithmetic cause");
            }
            catch (...)
            {
                ADD_FAILURE() << "Original geometry cause was not retained.";
            }
        }
    }
    EXPECT_TRUE(caught);
}

TEST(GeneralVoFFailures, SimultaneousErrorsChooseLowestFailingRankAndKeepEveryLocalCause)
{
    const auto comm = Tpetra::getDefaultComm();
    const int first = comm->getSize() > 1 ? 1 : 0;
    try
    {
        failure::collective<std::runtime_error>(*comm,
            [&]
            {
                if (comm->getRank() < first)
                    return;
                failure::Scope context("momentum_prepare");
                context.record.cell = 100 + comm->getRank();
                context.record.mark(failure::Field::Cell);
                if (comm->getRank() == first)
                    throw std::bad_alloc();
                throw std::invalid_argument("different metadata cause on a higher rank");
            });
        ADD_FAILURE() << "Expected a collective failure.";
    }
    catch (const failure::CollectiveFailure<std::runtime_error>& error)
    {
        EXPECT_EQ(error.record().origin_rank, first);
        EXPECT_EQ(error.record().category, failure::Category::Allocation);
        EXPECT_EQ(error.record().cell, 100u + first);
        EXPECT_STREQ(error.record().stage.data(), "momentum_prepare");
        EXPECT_EQ(bool(error.local_cause()), comm->getRank() >= first);
        if (error.local_cause())
        {
            try
            {
                std::rethrow_exception(error.local_cause());
            }
            catch (const std::bad_alloc&)
            {
                EXPECT_EQ(comm->getRank(), first);
            }
            catch (const std::invalid_argument&)
            {
                EXPECT_GT(comm->getRank(), first);
            }
            catch (...)
            {
                ADD_FAILURE() << "Unexpected original cause.";
            }
        }
    }
}

TEST(GeneralVoFFailures, CategoriesAndMissingFieldsDoNotLeakAcrossBoundaries)
{
    const auto comm = Tpetra::getDefaultComm();
    const int origin = comm->getSize() - 1;
    const std::array expected{failure::Category::ResourceLimit, failure::Category::Metadata,
        failure::Category::Arithmetic, failure::Category::ResourceLimit};
    for (unsigned which = 0; which < expected.size(); ++which)
    {
        try
        {
            failure::collective(
                *comm,
                [&]
                {
                    if (comm->getRank() != origin)
                        return;
                    switch (which)
                    {
                        case 0:
                            throw std::length_error("capacity limit");
                        case 1:
                            throw std::invalid_argument("foreign identity");
                        case 2:
                            throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "uncertain sign");
                        default:
                            throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, "explicit byte budget");
                    }
                },
                "category_test");
            ADD_FAILURE() << "Expected a collective failure.";
        }
        catch (const failure::Failure& error)
        {
            EXPECT_EQ(error.record().origin_rank, origin);
            EXPECT_EQ(error.record().category, expected[which]);
            EXPECT_FALSE(error.record().has(failure::Field::Face));
            EXPECT_FALSE(error.record().has(failure::Field::Donor));
            EXPECT_FALSE(error.record().has(failure::Field::Arithmetic));
            EXPECT_FALSE(error.record().has(failure::Field::ByteBudget));
        }
    }
    EXPECT_NO_THROW(failure::collective(*comm, [] {}, "successful_empty_work"));
}

TEST(GeneralVoFFailures, RecoveredLocalTrialDoesNotMaskNextScopedFailure)
{
    const auto comm = Tpetra::getDefaultComm();
    try
    {
        failure::collective(*comm,
            [&]
            {
                try
                {
                    failure::Scope trial("recovered_trial");
                    trial.record.face = 999;
                    trial.record.mark(failure::Field::Face);
                    throw PLIC::GeometryError(PLIC::GeometryErrorCode::UnresolvedArithmetic, "recovered trial");
                }
                catch (const PLIC::GeometryError&)
                {
                }
                failure::Scope actual("actual_failure");
                actual.record.donor = 42;
                actual.record.mark(failure::Field::Donor);
                throw std::invalid_argument("actual error");
            });
        ADD_FAILURE() << "Expected a collective failure.";
    }
    catch (const failure::Failure& error)
    {
        EXPECT_EQ(error.record().origin_rank, 0);
        EXPECT_EQ(error.record().category, failure::Category::Metadata);
        EXPECT_STREQ(error.record().stage.data(), "actual_failure");
        EXPECT_EQ(error.record().donor, 42u);
        EXPECT_FALSE(error.record().has(failure::Field::Face));
        EXPECT_STREQ(error.record().message.data(), "actual error");
    }
}

TEST(GeneralVoFFailures, NestedPropagationRetainsOriginAndBoundedConstructionPayload)
{
    const auto comm = Tpetra::getDefaultComm();
    const int origin = comm->getSize() - 1;
    try
    {
        failure::collective<std::runtime_error>(*comm,
            [&]
            {
                failure::collective(*comm,
                    [&]
                    {
                        if (comm->getRank() != origin)
                            return;
                        failure::Scope context("simplex_input");
                        context.record.face = 7;
                        context.record.mark(failure::Field::Face);
                        context.record.construction = 3;
                        context.record.mark(failure::Field::Construction);
                        context.record.simplex = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
                        context.record.mark(failure::Field::Simplex);
                        std::array<char, 1024> long_message;
                        long_message.fill('x');
                        long_message.back() = 0;
                        throw std::invalid_argument(long_message.data());
                    });
            });
        ADD_FAILURE() << "Expected a collective failure.";
    }
    catch (const failure::CollectiveFailure<std::runtime_error>& error)
    {
        const auto& record = error.record();
        EXPECT_EQ(record.origin_rank, origin);
        EXPECT_STREQ(record.stage.data(), "simplex_input");
        EXPECT_EQ(record.face, 7u);
        EXPECT_EQ(record.construction, 3u);
        EXPECT_EQ(record.simplex, (std::array<long double, 12>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}));
        EXPECT_EQ(record.message.back(), 0);
        EXPECT_EQ(std::string(record.message.data()).size(), record.message.size() - 1);
        EXPECT_NE(std::string(error.what()).find("simplex=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]"), std::string::npos);
        EXPECT_TRUE(error.local_cause());
    }
}
} // namespace
