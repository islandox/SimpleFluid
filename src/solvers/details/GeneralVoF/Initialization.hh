#pragma once
#include "solvers/details/GeneralVoF/Resources.hh"
#include "solvers/details/GeneralVoF/State.hh"
namespace SimpleFluid::detail::general_vof
{
template<ArithmeticMode Mode> struct SIMPLEFLUID_SOLVERS_LOCAL Initialization
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification ||
                  Mode == ArithmeticMode::Exact);
    [[nodiscard]] static std::shared_ptr<State> prepare(Resources&, std::shared_ptr<const VoF::GeneralVoF> phases,
        const std::vector<VoF::Vector>& owned_velocity, const std::vector<double>& face_velocity);
};
extern template struct Initialization<ArithmeticMode::Ordinary>;
extern template struct Initialization<ArithmeticMode::BoundedCertification>;
extern template struct Initialization<ArithmeticMode::Exact>;
} // namespace SimpleFluid::detail::general_vof
