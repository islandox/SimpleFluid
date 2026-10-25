#pragma once
#include "solvers/details/GeneralVoF/Resources.hh"
#include "solvers/details/GeneralVoF/State.hh"
namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode> struct SIMPLEFLUID_SOLVERS_LOCAL Advance
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);
    [[nodiscard]] static double stable_timestep(
        const Resources&, const State&, const std::vector<VoF::GeneralVoFNodeVelocity>&);
    [[nodiscard]] static std::shared_ptr<State> prepare(Resources&, const State&, double dt);
};
template<> struct SIMPLEFLUID_SOLVERS_LOCAL Advance<ArithmeticMode::Exact>
{
    [[nodiscard]] static double stable_timestep(
        const Resources&, const State&, const std::vector<VoF::GeneralVoFNodeVelocity>&);
    [[nodiscard]] static std::shared_ptr<State> prepare(Resources&, const State&, double dt);
};
extern template struct Advance<ArithmeticMode::Ordinary>;
extern template struct Advance<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::detail::general_vof
