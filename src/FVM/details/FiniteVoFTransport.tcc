#pragma once
#include "FVM/details/FiniteVoFTransport.hh"
namespace SimpleFluid::VoF::finite_detail
{
template<ArithmeticMode Mode>
Sweep<Mode> make_sweep(std::uint64_t face, std::span<const std::uint64_t> nodes,
    const std::map<std::uint64_t, std::pair<PLIC::Point<Mode>, PLIC::Point<Mode>>>& traces, Value control, Work& work,
    long double ordinary_volume_budget)
{
    using Math = Arithmetic<Mode>;
    using Points = PointArithmetic<Mode>;
    if (work.arithmetic_mode != Mode)
        throw std::invalid_argument("Finite VoF kernel arithmetic mode differs from its work scope.");
    if (nodes.size() < 3 || nodes.size() > work.limits.maximum_simplices / 4)
        throw std::invalid_argument("Finite VoF face simplex limit is exhausted.");
    if (!std::isfinite(work.volume_tolerance) || !(work.volume_tolerance > 0))
        throw std::invalid_argument("Finite VoF sweep volume tolerance is invalid.");
    validate(control);
    if (control.arithmetic_mode != work.arithmetic_mode)
        throw std::invalid_argument("Finite VoF sweep control arithmetic mode differs.");
    if constexpr (Mode == ArithmeticMode::Ordinary)
        if (!std::isfinite(ordinary_volume_budget) || !(ordinary_volume_budget > 0))
            throw std::invalid_argument("Ordinary VoF sweep needs a positive cell-derived volume budget.");
    Sweep<Mode> result;
    result.face = face;
    result.control = control;
    result.ordinary_volume_budget = ordinary_volume_budget;
    std::set<std::uint64_t> unique;
    for (auto node : nodes)
    {
        if (!unique.insert(node).second)
            throw std::invalid_argument("Finite VoF canonical face repeats a node.");
        const auto& trace = traces.at(node);
        if (trace.first.arithmetic_mode != work.arithmetic_mode ||
            trace.second.arithmetic_mode != work.arithmetic_mode)
            throw std::invalid_argument("Finite VoF sweep trajectory arithmetic mode differs.");
        result.current.push_back(trace.first);
        result.departure.push_back(trace.second);
        result.center = Points::add(result.center, trace.first);
        result.backcenter = Points::add(result.backcenter, trace.second);
    }
    result.center = PLIC::point_average<Mode>(result.current);
    result.backcenter = PLIC::point_average<Mode>(result.departure);
    PLIC::Point<Mode> normal{};
    for (std::size_t i = 0; i < nodes.size(); ++i)
        normal = Points::add(normal, Points::cross(Points::subtract(result.current[i], result.center),
                                         Points::subtract(result.current[(i + 1) % nodes.size()], result.center)));
    const auto triangles = sweep_boundary(nodes, result.current, result.departure, result.center, result.backcenter);
    SumFor<Mode> original;
    for (const auto& triangle : triangles)
        original.add(cone_determinant(result.center, triangle));
    // Only the departure cap depends on the private center. Its linear
    // coefficient follows directly from the determinant's multilinearity.
    SumFor<Mode> coefficient;
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        const auto j = (i + 1) % nodes.size();
        coefficient.add(Points::dot(normal, Points::cross(Points::subtract(result.departure[j], result.center),
                                                Points::subtract(result.departure[i], result.center))));
    }
    const auto difference = Math::subtract(Math::multiply(control, {6, 0}), original.value());
    if (difference.value != 0 || difference.absolute_error != 0)
        result.backcenter = PLIC::point_displaced<Mode>(
            result.backcenter, Points::scale(normal, Math::divide(difference, coefficient.value())));
    SumFor<Mode> check;
    SumFor<Mode> constructed(work.arithmetic_mode);
    const auto corrected = sweep_boundary(nodes, result.current, result.departure, result.center, result.backcenter);
    for (const auto& triangle : corrected)
    {
        const auto volume = cone_determinant(result.center, triangle);
        if (volume.value == 0 && volume.absolute_error == 0)
            continue;
        if (!(std::abs(volume.value) > volume.absolute_error))
            throw std::invalid_argument("Finite VoF swept simplex orientation is arithmetically unresolved.");
        PLIC::GeometryLimits geometry_limits;
        geometry_limits.arithmetic_mode = work.arithmetic_mode;
        geometry_limits.byte_budget = work.limits.byte_budget;
        auto geometry =
            Geometry<Mode>::tetrahedron({result.center, triangle[0], triangle[1], triangle[2]}, geometry_limits);
        if (!geometry)
            throw std::invalid_argument("Finite VoF positive swept simplex disappeared.");
        work.atom(geometry);
        const auto measured = runtime_value<Mode>(geometry->volume());
        constructed.add(volume.value > 0 ? measured : Math::negate(measured));
        result.pieces.push_back({std::move(geometry), volume.value > 0 ? 1 : -1});
        check.add(volume);
    }
    const auto residual = Math::subtract(Math::divide(check.value(), {6, 0}), control);
    result.constructed_volume = constructed.value();
    result.signed_control_difference = Math::subtract(result.constructed_volume, control);
    const auto mismatch = [&]
    {
        if constexpr (Mode == ArithmeticMode::Ordinary)
            return std::abs(result.signed_control_difference.value) > ordinary_volume_budget;
        else return std::abs(residual.value) > residual.absolute_error;
    }();
    if (mismatch)
        throw std::invalid_argument("Finite VoF corrected face chain does not match its canonical control.");
    return result;
}
template<ArithmeticMode Mode>
std::vector<Atom<Mode>> recipient_partition(std::uint64_t recipient,
    const std::map<std::uint64_t, GeometryPtr<Mode>>& donors,
    const std::vector<std::pair<const Sweep<Mode>*, int>>& faces, Work& work, double physical_budget)
{
    using Math = Arithmetic<Mode>;
    using Points = PointArithmetic<Mode>;
    if (work.arithmetic_mode != Mode)
        throw std::invalid_argument("Finite VoF kernel arithmetic mode differs from its work scope.");
    std::vector<Atom<Mode>> result;
    SumFor<Mode> volume;
    for (const auto& [donor, geometry] : donors)
    {
        std::vector<OverlayAtom<Mode>> atoms{{geometry, donor == recipient ? 1 : 0}};
        for (const auto& [sweep, incidence] : faces)
            for (const auto& piece : sweep->pieces)
                atoms = overlay(std::move(atoms), piece, -incidence, work);
        for (const auto& atom : atoms)
        {
            if (atom.coefficient != 0 && atom.coefficient != 1)
                throw std::invalid_argument("Finite VoF closed-cell pullback has invalid signed multiplicity.");
            if (atom.coefficient)
            {
                volume.add(runtime_value<Mode>(atom.geometry->volume()));
                work.push(result, Atom<Mode>{donor, recipient, atom.geometry});
            }
        }
    }
    const auto target = runtime_value<Mode>(donors.at(recipient)->volume());
    const auto difference = Math::subtract(volume.value(), target);
    if (!Math::closure_within(difference, std::abs(target.value) * physical_budget))
        throw std::invalid_argument("Finite VoF recipient closure exceeds its original arithmetic/physical budget.");
    return result;
}
}
