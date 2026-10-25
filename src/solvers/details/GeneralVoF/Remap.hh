#pragma once
#include "solvers/details/GeneralVoF/Resources.hh"
#include "solvers/details/GeneralVoF/State.hh"
namespace SimpleFluid::detail::general_vof
{
struct SIMPLEFLUID_SOLVERS_LOCAL PreparedRemap
{
    std::unique_ptr<Resources> resources;
    std::shared_ptr<State> state;
};
template<ArithmeticMode Mode> struct SIMPLEFLUID_SOLVERS_LOCAL Remap
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    [[nodiscard]] static PreparedRemap prepare(const Resources&, const State&, bool migration,
        std::span<const int> destinations, std::span<const Meshes::AMR::AdaptiveMark> refine,
        std::span<const Meshes::AMR::AdaptiveBytes> coarsen);
};
template<> struct SIMPLEFLUID_SOLVERS_LOCAL Remap<ArithmeticMode::Exact>
{
    [[nodiscard]] static PreparedRemap prepare(const Resources&, const State&, bool migration,
        std::span<const int> destinations, std::span<const Meshes::AMR::AdaptiveMark> refine,
        std::span<const Meshes::AMR::AdaptiveBytes> coarsen);
};
extern template struct Remap<ArithmeticMode::Ordinary>;
extern template struct Remap<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::detail::general_vof
