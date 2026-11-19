#include "SolversExceptionProvider.hh"
#include "equations/PressureConditionedThermal.hh"

#include <gtest/gtest.h>
#include <memory>

namespace
{
using namespace SimpleFluid::abi_fixture;
using SimpleFluid::Reacting::CaloricDomainUnresolved;
using SimpleFluid::Reacting::ThermalCaloricRangeError;

template<class Exception, class Other> void check_boundary(CaloricExceptionKind kind, const char* message)
{
    EXPECT_EQ(&caloric_exception_type(kind), &typeid(Exception));
    EXPECT_THROW(throw_caloric_exception(kind), Exception);

    try
    {
        throw_caloric_exception(kind);
    }
    catch (const std::domain_error& error)
    {
        EXPECT_STREQ(error.what(), message);
        EXPECT_EQ(&typeid(error), &typeid(Exception));
        EXPECT_NE(dynamic_cast<const Exception*>(&error), nullptr);
        EXPECT_EQ(dynamic_cast<const Other*>(&error), nullptr);
    }

    // The producer constructs the exception and the consumer deletes it through its base.
    const std::unique_ptr<std::domain_error> owned(make_caloric_exception(kind));
    ASSERT_NE(owned, nullptr);
    EXPECT_STREQ(owned->what(), message);
    EXPECT_EQ(&typeid(*owned), &typeid(Exception));
    EXPECT_NE(dynamic_cast<Exception*>(owned.get()), nullptr);
    EXPECT_EQ(dynamic_cast<Other*>(owned.get()), nullptr);
}

TEST(SolversExceptionBoundary, CaloricDomainUnresolved)
{
    check_boundary<CaloricDomainUnresolved, ThermalCaloricRangeError>(
        CaloricExceptionKind::Domain, "caloric domain boundary");
}

TEST(SolversExceptionBoundary, ThermalCaloricRangeError)
{
    check_boundary<ThermalCaloricRangeError, CaloricDomainUnresolved>(
        CaloricExceptionKind::Range, "thermal range boundary");
}
} // namespace
