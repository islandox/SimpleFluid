/** @file LimiterReplay.hh
 * @brief Donor-local arithmetic replay of rounded limiter exports.
 */
#pragma once
#include "FVM/GeneralVoF.hh"
#include "utils/BoundedFloatingPoint.hh"
#include "utils/CompensatedSum.hh"
#include <array>
#include <cfenv>
#include <optional>
#include <vector>

namespace SimpleFluid::VoF::general_vof_detail::limiter_replay
{
namespace bp = SimpleFluid::detail::bounded_fp;
struct Value
{
    long double value = 0, absolute_error = 0;
};
inline Value scalar(long double x)
{
    return {x, 0};
}
inline long double magnitude(Value x)
{
    return bp::bound_add(std::abs(x.value), x.absolute_error);
}
inline long double lower(Value x)
{
    return std::nextafter(x.value - x.absolute_error, -std::numeric_limits<long double>::infinity());
}
inline void evaluate(GeneralVoFDirectedStream& s, const std::array<double, 2>& density)
{
    const auto blend = [&](long double low, long double high)
    {
        if (s.theta == 0)
            return low;
        if (s.theta == 1)
            return high;
        return (1 - s.theta) * low + s.theta * high;
    };
    s.gas = blend(s.gas_low, s.gas_high);
    s.liquid = blend(s.liquid_low, s.liquid_high);
    if (s.gas <= s.liquid)
        s.liquid = s.volume - s.gas;
    else
        s.gas = s.volume - s.liquid;
    if (s.gas < 0 || s.liquid < 0 || s.gas > s.volume || s.liquid > s.volume)
        throw std::invalid_argument("Limited phase stream is inadmissible.");
    s.mass = density[0] * s.gas + density[1] * s.liquid;
}
inline long double correction(const GeneralVoFDirectedStream& s)
{
    return s.gas_low <= s.liquid_low ? s.gas_high - s.gas_low : s.liquid_low - s.liquid_high;
}
inline long double low(const GeneralVoFDirectedStream& s, unsigned p)
{
    return p ? s.liquid_low : s.gas_low;
}
inline long double high(const GeneralVoFDirectedStream& s, unsigned p)
{
    return p ? s.liquid_high : s.gas_high;
}
inline long double phase(const GeneralVoFDirectedStream& s, unsigned p)
{
    return p ? s.liquid : s.gas;
}
struct Summary
{
    std::array<long double, 2> exported{}, rounding{};
};
struct Evidence
{
    std::array<long double, 2> original_export{}, final_export{}, control_bound{}, original_rounding{},
        final_rounding{}, coefficient_resolution{}, movement{}, movement_limit{};
    unsigned evaluations = 0;
    bool accepted = false;
};
inline Summary summarize(const std::vector<GeneralVoFDirectedStream>& streams)
{
    Summary out;
    for (unsigned p = 0; p < 2; ++p)
    {
        SimpleFluid::detail::CompensatedSum<long double> sum;
        Value stored{};
        for (const auto& s : streams)
        {
            const auto ideal = bp::add(bp::multiply(bp::subtract(scalar(1), scalar(s.theta)), scalar(low(s, p))),
                bp::multiply(scalar(s.theta), scalar(high(s, p))));
            out.rounding[p] = bp::bound_add(out.rounding[p], magnitude(bp::subtract(scalar(phase(s, p)), ideal)));
            stored = bp::add(stored, scalar(phase(s, p)));
            sum += phase(s, p);
        }
        out.exported[p] = sum.value();
        out.rounding[p] = bp::bound_add(out.rounding[p], magnitude(bp::subtract(scalar(sum.value()), stored)));
    }
    return out;
}
// Input streams are sorted by canonical face ID, exactly as in the ledger.
inline std::optional<std::vector<GeneralVoFDirectedStream>> repair(
    const std::vector<GeneralVoFDirectedStream>& original, const std::array<long double, 2>& old,
    const std::array<long double, 4>& available, const std::array<long double, 4>& demand,
    const std::array<long double, 4>& ratio, const std::array<double, 2>& density, Evidence* evidence = nullptr)
{
    if (evidence)
        *evidence = {};
    if (std::fegetround() != FE_TONEAREST)
        return std::nullopt;
    // The pointwise residual is meaningful only for streams actually produced
    // by this helper, not arbitrary stored phase or mass values.
    for (const auto& s : original)
    {
        auto verified = s;
        evaluate(verified, density);
        if (verified.gas != s.gas || verified.liquid != s.liquid || verified.mass != s.mass)
            return std::nullopt;
    }
    const auto initial = summarize(original);
    if (evidence)
    {
        evidence->original_export = initial.exported;
        evidence->original_rounding = initial.rounding;
    }
    const unsigned target = initial.exported[0] > old[0] ? 0 : 1;
    if (initial.exported[target] <= old[target])
        return std::nullopt;
    std::array<long double, 2> control{}, resolution{};
    Value slope{};
    for (unsigned p = 0; p < 2; ++p)
    {
        const unsigned index = p + 2;
        const auto d = demand[index], a = available[index], r = ratio[index];
        auto expected = d > 0 ? std::min(1.L, a / d) : 1.L;
        if (expected > 0 && expected < 1)
            expected = std::nextafter(expected, 0.L);
        if (r != expected || r < 0 || r > 1)
            return std::nullopt;
        Value lows{}, positive{}, difference{};
        for (const auto& s : original)
        {
            lows = bp::add(lows, scalar(low(s, p)));
            const auto c = (p ? -1 : 1) * correction(s);
            if (c > 0 && s.theta > r)
                return std::nullopt;
            positive = bp::add(positive, scalar(std::max(c, 0.L)));
            const auto mismatch = bp::subtract(bp::subtract(scalar(high(s, p)), scalar(low(s, p))), scalar(c));
            difference = bp::add(difference, bp::multiply(scalar(s.theta), scalar(magnitude(mismatch))));
            if (high(s, target) > low(s, target))
            {
                const auto delta = bp::subtract(scalar(high(s, p)), scalar(low(s, p)));
                resolution[p] =
                    bp::bound_add(resolution[p], bp::bound_multiply(bp::rounding(s.theta), magnitude(delta)));
                if (p == target)
                    slope = bp::add(slope, bp::multiply(scalar(s.theta), delta));
            }
        }
        const auto availability_error = magnitude(bp::subtract(scalar(a), bp::subtract(scalar(old[p]), lows)));
        const auto product_error = bp::subtract(bp::multiply(scalar(r), scalar(d)), scalar(a));
        const auto positive_error = bp::bound_multiply(r, magnitude(bp::subtract(scalar(d), positive)));
        const auto product_upper = std::nextafter(
            product_error.value + product_error.absolute_error, std::numeric_limits<long double>::infinity());
        control[p] = bp::bound_add(availability_error,
            bp::bound_add(std::max(0.L, product_upper), bp::bound_add(positive_error, magnitude(difference))));
        if (initial.exported[p] > old[p] && magnitude(bp::subtract(scalar(initial.exported[p]), scalar(old[p]))) >
                                                bp::bound_add(control[p], initial.rounding[p]))
            return std::nullopt;
    }
    const auto slope_lower = lower(slope);
    if (evidence)
    {
        evidence->control_bound = control;
        evidence->coefficient_resolution = resolution;
    }
    if (!(slope_lower > 0))
        return std::nullopt;
    const auto budget = bp::bound_add(bp::bound_add(control[target], initial.rounding[target]), resolution[target]);
    long double delta = std::nextafter(budget / slope_lower, std::numeric_limits<long double>::infinity());
    if (!(delta > 0) || !std::isfinite(delta))
        return std::nullopt;
    std::optional<std::vector<GeneralVoFDirectedStream>> passing;
    long double safe = 0, failed = 1;
    unsigned count = 0;
    const unsigned cap = std::numeric_limits<long double>::digits + 2;
    const auto trial = [&](long double lambda)
    {
        auto result = original;
        for (auto& s : result)
            if (high(s, target) > low(s, target))
            {
                s.theta *= lambda;
                evaluate(s, density);
            }
        return result;
    };
    while (count < cap && delta < 1)
    {
        const auto lambda = std::nextafter(1 - delta, 0.L);
        if (!(lambda > 0 && lambda < 1))
            break;
        auto candidate = trial(lambda);
        ++count;
        if (evidence)
            evidence->evaluations = count;
        const auto values = summarize(candidate);
        if (values.exported[0] <= old[0] && values.exported[1] <= old[1])
        {
            safe = lambda;
            passing = std::move(candidate);
            break;
        }
        failed = lambda;
        delta = bp::bound_add(delta, delta);
    }
    if (!passing)
        return std::nullopt;
    while (count < cap)
    {
        const auto mid = safe + (failed - safe) / 2;
        if (!(mid > safe && mid < failed))
            break;
        auto candidate = trial(mid);
        ++count;
        if (evidence)
            evidence->evaluations = count;
        const auto values = summarize(candidate);
        if (values.exported[0] <= old[0] && values.exported[1] <= old[1])
        {
            safe = mid;
            passing = std::move(candidate);
        }
        else
            failed = mid;
    }
    const auto final = summarize(*passing);
    if (evidence)
    {
        evidence->evaluations = count;
        evidence->final_export = final.exported;
        evidence->final_rounding = final.rounding;
    }
    const auto next = safe + (failed - safe) / 2;
    if (count == cap && next > safe && next < failed)
        return std::nullopt;
    for (std::size_t i = 0; i < original.size(); ++i)
        if ((original[i].theta > 0 && (*passing)[i].theta == 0) || (original[i].gas > 0 && (*passing)[i].gas == 0) ||
            (original[i].liquid > 0 && (*passing)[i].liquid == 0))
            return std::nullopt;
    for (unsigned p = 0; p < 2; ++p)
    {
        long double change = 0;
        for (std::size_t i = 0; i < original.size(); ++i)
            change = bp::bound_add(
                change, magnitude(bp::subtract(scalar(phase((*passing)[i], p)), scalar(phase(original[i], p)))));
        const auto envelope = bp::bound_add(
            bp::bound_add(control[p], initial.rounding[p]), bp::bound_add(final.rounding[p], resolution[p]));
        if (evidence)
        {
            evidence->movement[p] = change;
            evidence->movement_limit[p] = envelope;
        }
        if (change > envelope)
            return std::nullopt;
    }
    if (evidence)
        evidence->accepted = true;
    return passing;
}
} // namespace SimpleFluid::VoF::general_vof_detail::limiter_replay
