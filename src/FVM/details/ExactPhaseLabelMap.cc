/** @file ExactPhaseLabelMap.cc
 * @brief Compiled bounded donor-wide complementary geometric label correction.
 */
#include "FVM/details/ExactPhaseLabelMap.hh"
#include <stdexcept>
namespace SimpleFluid::VoF::detail
{
using M = Meshes::ExactGeometryMeasure;
M exact_phase_product(const M& a, const M& b, std::size_t limit)
{
    if (!a.sign() || !b.sign())
        return {};
    return a.divided_by(M::from_value(1).divided_by(b, limit), limit);
}
ExactPhaseLabelMap::ExactPhaseLabelMap(bool gas, M a, M b, M discrepancy, M bound, std::size_t limit)
    : gas_(gas), a_(std::move(a)), b_(std::move(b)), discrepancy_(std::move(discrepancy)), bound_(std::move(bound)),
      limit_(limit)
{
}
ExactPhaseLabelMap ExactPhaseLabelMap::create(
    const Meshes::ExactPhaseMeasures& full, bool gas, const M& target, const M& bound, std::size_t limit)
{
    if (!limit || full.bulk.sign() <= 0 || full.gas.sign() < 0 || full.carrier.sign() < 0 ||
        (full.gas + full.carrier).compare(full.bulk) || target.sign() < 0 || target.compare(full.bulk) > 0 ||
        bound.sign() < 0)
        throw std::invalid_argument("Invalid complementary exact phase correction inputs.");
    const auto p = gas ? full.gas : full.carrier;
    const auto q = gas ? full.carrier : full.gas;
    const auto difference = target - p;
    const auto absolute = difference.sign() < 0 ? difference.scaled(-1) : difference;
    if (absolute.compare(bound) > 0)
        throw std::invalid_argument("Exact phase correction exceeds independent reconstruction bound.");
    auto a = M::from_value(1);
    M b;
    if (difference.sign() > 0)
    {
        if (!q.sign())
            throw std::invalid_argument("Exact phase correction has no dominant geometric support.");
        b = difference.divided_by(q, limit);
    }
    else if (difference.sign() < 0)
    {
        if (!p.sign())
            throw std::invalid_argument("Exact phase correction has no minority geometric support.");
        a = target.divided_by(p, limit);
    }
    const auto one = M::from_value(1);
    if (a.sign() < 0 || b.sign() < 0 || a.compare(one) > 0 || b.compare(one) > 0)
        throw std::invalid_argument("Exact phase correction has no nonnegative complementary coefficients.");
    return ExactPhaseLabelMap(gas, std::move(a), std::move(b), difference, bound, limit);
}
Meshes::ExactPhaseMeasures ExactPhaseLabelMap::apply(const Meshes::ExactPhaseMeasures& raw) const
{
    if ((raw.gas + raw.carrier).compare(raw.bulk))
        throw std::invalid_argument("Exact phase correction requires a complementary geometric partition.");
    const auto p = gas_ ? raw.gas : raw.carrier;
    const auto q = gas_ ? raw.carrier : raw.gas;
    const auto minority = exact_phase_product(a_, p, limit_) + exact_phase_product(b_, q, limit_);
    const auto dominant = raw.bulk - minority;
    if ((raw.bulk.sign() >= 0 && (minority.sign() < 0 || dominant.sign() < 0)) ||
        (raw.bulk.sign() < 0 && (minority.sign() > 0 || dominant.sign() > 0)))
        throw std::invalid_argument("Exact phase correction violated positive atom capacity.");
    return {raw.bulk, gas_ ? minority : dominant, gas_ ? dominant : minority};
}
Meshes::ExactPhaseMeasures ExactPhaseLabelMap::apply(
    const Meshes::ExactPhaseMeasures& raw, std::size_t operation_limit) const
{
    if (!operation_limit || operation_limit > limit_)
        throw std::invalid_argument("Exact phase operation cap must be positive and within its captured limit.");
    // This arena is scoped to ONE application, not accumulated over atoms.
    // Conservatively retain charges for its constant number of intermediates
    // until return, even when a particular reciprocal has already died.
    std::size_t used = 0;
    const auto keep = [&](const M& value)
    {
        const auto bytes = value.storage_bytes();
        if (bytes > operation_limit - used)
            throw std::invalid_argument("Exact phase operation live payload exceeds its arena.");
        used += bytes;
    };
    for (const auto* value : {&raw.bulk, &raw.gas, &raw.carrier, &a_, &b_})
        keep(*value);
    const auto operation = [&](const M& x, const M& y, auto work)
    {
        const auto available = operation_limit - used;
        const auto first = x.storage_bytes(), second = y.storage_bytes();
        // Exact rational +/- and division numerator/denominator payloads are
        // bounded from their operands before growth. The same conservative
        // eight-times envelope is used by existing application arithmetic;
        // backend allocation/GCD workspace retains its documented exclusion.
        if (first > available / 8 || second > available / 8 - first)
            throw std::invalid_argument("Exact phase operation operands exceed their pending growth arena.");
        const auto pending = 8 * (first + second);
        auto result = work(pending);
        if (result.storage_bytes() > pending)
            throw std::invalid_argument("Exact phase operation result exceeds its admitted growth envelope.");
        keep(result);
        return result;
    };
    const auto sum = operation(raw.gas, raw.carrier, [&](std::size_t) { return raw.gas + raw.carrier; });
    if (sum.compare(raw.bulk))
        throw std::invalid_argument("Exact phase correction requires a complementary geometric partition.");
    // A fresh unit scalar fits the inline capacity of any admitted coefficient.
    // Admit that known bound before construction, then retain its actual size.
    if (a_.storage_bytes() > operation_limit - used)
        throw std::invalid_argument("Exact phase unit scalar exceeds its operation arena.");
    const auto one = M::from_value(1);
    keep(one);
    const auto product = [&](const M& coefficient, const M& value)
    {
        if (!coefficient.sign())
            return coefficient;
        if (!value.sign())
            return value;
        const auto reciprocal =
            operation(one, value, [&](std::size_t pending) { return one.divided_by(value, pending); });
        return operation(
            coefficient, reciprocal, [&](std::size_t pending) { return coefficient.divided_by(reciprocal, pending); });
    };
    const auto& p = gas_ ? raw.gas : raw.carrier;
    const auto& q = gas_ ? raw.carrier : raw.gas;
    const auto first = product(a_, p), second = product(b_, q);
    const auto minority = operation(first, second, [&](std::size_t) { return first + second; });
    const auto dominant = operation(raw.bulk, minority, [&](std::size_t) { return raw.bulk - minority; });
    if ((raw.bulk.sign() >= 0 && (minority.sign() < 0 || dominant.sign() < 0)) ||
        (raw.bulk.sign() < 0 && (minority.sign() > 0 || dominant.sign() > 0)))
        throw std::invalid_argument("Exact phase correction violated positive atom capacity.");
    return {raw.bulk, gas_ ? minority : dominant, gas_ ? dominant : minority};
}
std::size_t ExactPhaseLabelMap::storage_bytes() const
{
    return sizeof(*this) + a_.storage_bytes() + b_.storage_bytes() + discrepancy_.storage_bytes() +
           bound_.storage_bytes();
}
} // namespace SimpleFluid::VoF::detail
