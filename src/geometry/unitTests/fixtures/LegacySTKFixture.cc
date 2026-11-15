/** @file LegacySTKFixture.cc
 * @brief Test-only provider for the frozen STK geometry oracle.
 */
#include "LegacySTKFixture.hh"
#include "LegacySTKFixture.tcc"

namespace SimpleFluid::test
{
template class LegacySTKFixture<DefaultTpetraTypes>;
} // namespace SimpleFluid::test
