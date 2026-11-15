/** @brief Test-owned ALE validator for injected mesh traversal failures. */
#include "ALEFixtures.hh"
#include "FVM/ALEControlVolumeState.tcc"

namespace SimpleFluid::FVM
{
template void ALEControlVolumeState::validate(const test::ale::ThrowingHandle&) const;
} // namespace SimpleFluid::FVM
