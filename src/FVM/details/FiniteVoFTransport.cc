#include "FVM/details/FiniteVoFTransport.tcc"
namespace SimpleFluid::VoF::finite_detail
{
template Sweep<ArithmeticMode::Ordinary> prepare_sweep<ArithmeticMode::Ordinary>(std::uint64_t,
    std::span<const std::uint64_t>,
    const std::map<std::uint64_t,
        std::pair<PLIC::Point<ArithmeticMode::Ordinary>, PLIC::Point<ArithmeticMode::Ordinary>>>&,
    Value, Work&, long double);
template Sweep<ArithmeticMode::Ordinary> materialize_sweep<ArithmeticMode::Ordinary>(
    Sweep<ArithmeticMode::Ordinary>, std::span<const std::uint64_t>, Work&);
template Sweep<ArithmeticMode::Ordinary> make_sweep<ArithmeticMode::Ordinary>(std::uint64_t,
    std::span<const std::uint64_t>,
    const std::map<std::uint64_t,
        std::pair<PLIC::Point<ArithmeticMode::Ordinary>, PLIC::Point<ArithmeticMode::Ordinary>>>&,
    Value, Work&, long double);
template std::vector<Atom<ArithmeticMode::Ordinary>> recipient_partition<ArithmeticMode::Ordinary>(std::uint64_t,
    const std::map<std::uint64_t, GeometryPtr<ArithmeticMode::Ordinary>>&,
    const std::vector<std::pair<const Sweep<ArithmeticMode::Ordinary>*, int>>&, Work&, double);
template Sweep<ArithmeticMode::BoundedCertification> prepare_sweep<ArithmeticMode::BoundedCertification>(std::uint64_t,
    std::span<const std::uint64_t>,
    const std::map<std::uint64_t, std::pair<PLIC::Point<ArithmeticMode::BoundedCertification>,
                                      PLIC::Point<ArithmeticMode::BoundedCertification>>>&,
    Value, Work&, long double);
template Sweep<ArithmeticMode::BoundedCertification> materialize_sweep<ArithmeticMode::BoundedCertification>(
    Sweep<ArithmeticMode::BoundedCertification>, std::span<const std::uint64_t>, Work&);
template Sweep<ArithmeticMode::BoundedCertification> make_sweep<ArithmeticMode::BoundedCertification>(std::uint64_t,
    std::span<const std::uint64_t>,
    const std::map<std::uint64_t, std::pair<PLIC::Point<ArithmeticMode::BoundedCertification>,
                                      PLIC::Point<ArithmeticMode::BoundedCertification>>>&,
    Value, Work&, long double);
template std::vector<Atom<ArithmeticMode::BoundedCertification>>
recipient_partition<ArithmeticMode::BoundedCertification>(std::uint64_t,
    const std::map<std::uint64_t, GeometryPtr<ArithmeticMode::BoundedCertification>>&,
    const std::vector<std::pair<const Sweep<ArithmeticMode::BoundedCertification>*, int>>&, Work&, double);
} // namespace SimpleFluid::VoF::finite_detail
