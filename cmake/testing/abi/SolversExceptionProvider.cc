/**
 * @file SolversExceptionProvider.cc
 * @brief Produce public caloric exceptions in a shared image separate from their consumer.
 */
#include "SolversExceptionProvider.hh"

#include "equations/PressureConditionedThermal.hh"

namespace SimpleFluid::abi_fixture
{
void throw_caloric_exception(CaloricExceptionKind kind)
{
    if (kind == CaloricExceptionKind::Range)
        throw Reacting::ThermalCaloricRangeError("thermal range boundary");
    throw Reacting::CaloricDomainUnresolved("caloric domain boundary");
}

std::domain_error* make_caloric_exception(CaloricExceptionKind kind)
{
    if (kind == CaloricExceptionKind::Range)
        return new Reacting::ThermalCaloricRangeError("thermal range boundary");
    return new Reacting::CaloricDomainUnresolved("caloric domain boundary");
}

const std::type_info& caloric_exception_type(CaloricExceptionKind kind)
{
    if (kind == CaloricExceptionKind::Range)
        return typeid(Reacting::ThermalCaloricRangeError);
    return typeid(Reacting::CaloricDomainUnresolved);
}
} // namespace SimpleFluid::abi_fixture
