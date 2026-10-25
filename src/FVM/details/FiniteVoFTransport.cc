#include "FVM/details/FiniteVoFTransport.tcc"
namespace SimpleFluid::VoF::finite_detail
{
template Sweep make_sweep<ArithmeticMode::Ordinary>(std::uint64_t, std::span<const std::uint64_t>,
    const std::map<std::uint64_t, std::pair<Point, Point>>&, Value, Work&, long double);
template std::vector<Atom> recipient_partition<ArithmeticMode::Ordinary>(std::uint64_t,
    const std::map<std::uint64_t, GeometryPtr>&, const std::vector<std::pair<const Sweep*, int>>&, Work&, double);
template Sweep make_sweep<ArithmeticMode::BoundedCertification>(std::uint64_t, std::span<const std::uint64_t>,
    const std::map<std::uint64_t, std::pair<Point, Point>>&, Value, Work&, long double);
template std::vector<Atom> recipient_partition<ArithmeticMode::BoundedCertification>(std::uint64_t,
    const std::map<std::uint64_t, GeometryPtr>&, const std::vector<std::pair<const Sweep*, int>>&, Work&, double);
} // namespace SimpleFluid::VoF::finite_detail
