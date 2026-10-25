#pragma once

#include <array>
#include <boost/config.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::PLIC::exact_geometry_detail
{
using SupportPoint = std::array<boost::multiprecision::cpp_rational, 3>;
} // namespace SimpleFluid::PLIC::exact_geometry_detail

namespace SimpleFluid::PLIC::detail
{

template<unsigned Bits>
using SupportSignInteger = boost::multiprecision::number<
    boost::multiprecision::cpp_int_backend<0, Bits, boost::multiprecision::signed_magnitude,
        boost::multiprecision::checked, std::allocator<boost::multiprecision::limb_type>>,
    boost::multiprecision::et_off>;

template<unsigned Bits = 32768> std::size_t integer_support_workspace_bytes()
{
    static_assert(Bits >= 128 && Bits <= 32768);
    using Integer = SupportSignInteger<Bits>;
    using Limb = boost::multiprecision::limb_type;
    constexpr auto limb_bits = std::numeric_limits<Limb>::digits;
    constexpr std::size_t limbs = (Bits + limb_bits - 1) / limb_bits;
    // Five live operands, plus conservative room for bounded assignment/alias
    // temporaries and backend metadata. Bounded Boost multiplication can still
    // allocate Karatsuba scratch: include its declared storage bound and an
    // additional full product. This lease is not a whole-process allocator cap.
    return 12 * (sizeof(Integer) + limbs * sizeof(Limb)) +
           (2 * limbs + boost::multiprecision::backends::karatsuba_storage_size(limbs)) * sizeof(Limb);
}

inline void require_positive_support_denominator(const boost::multiprecision::cpp_rational& value)
{
    const auto& denominator = value.backend().denom();
    if (!denominator.size() || denominator.sign() || (denominator.size() == 1 && denominator.limbs()[0] == 0))
        throw std::logic_error("Exact support sign requires a positive rational denominator.");
}

template<unsigned Bits>
BOOST_NOINLINE std::optional<int> integer_support_sign_nontrivial(const exact_geometry_detail::SupportPoint& normal,
    const exact_geometry_detail::SupportPoint& anchor, const exact_geometry_detail::SupportPoint& point,
    const std::array<unsigned, 3>& active, unsigned count)
{
    using Integer = SupportSignInteger<Bits>;
    try
    {
        Integer sum_n = 0, sum_d = 1, term_n, term_d, work;
        for (unsigned term = 0; term < count; ++term)
        {
            const auto axis = active[term];
            // Read backend references directly. numerator(q)/denominator(q)
            // would first allocate unbounded integer copies of the inputs.
            term_n.backend() = point[axis].backend().num();
            term_d.backend() = anchor[axis].backend().denom();
            term_n *= term_d;
            term_d.backend() = anchor[axis].backend().num();
            work.backend() = point[axis].backend().denom();
            term_d *= work;
            term_n -= term_d;
            work.backend() = normal[axis].backend().num();
            term_n *= work;

            term_d.backend() = normal[axis].backend().denom();
            work.backend() = point[axis].backend().denom();
            term_d *= work;
            work.backend() = anchor[axis].backend().denom();
            term_d *= work;

            // term_n/term_d is n * (p-a), and term_d is strictly positive.
            // No normalization is needed when only the final sign is used.
            if (!term || sum_n == 0)
            {
                sum_n = term_n;
                sum_d = term_d;
            }
            else if (sum_d == term_d)
            {
                // Already-common denominators need neither multiplication nor
                // reduction. This is frequent for constructed shared faces.
                sum_n += term_n;
            }
            else
            {
                sum_n *= term_d;
                term_n *= sum_d;
                sum_n += term_n;
                if (term + 1 < count)
                    sum_d *= term_d;
            }
        }
        return sum_n < 0 ? -1 : sum_n > 0 ? 1 : 0;
    }
    catch (const std::overflow_error&)
    {
        // Checked conversion/arithmetic never supplies a truncated sign. A
        // capacity miss asks the caller to run the original rational predicate.
        return std::nullopt;
    }
}

/** Exact sign-only acceleration for canonical rationals from admitted geometry.
 * reserve(bytes) must cover the live integer workspace before it is constructed.
 * A denied lease or checked capacity overflow is inconclusive, not a sign.
 * The caller releases that lease before evaluating its original Q fallback.
 */
template<unsigned Bits = 32768, class Reserve>
std::optional<int> try_integer_support_sign(const exact_geometry_detail::SupportPoint& normal,
    const exact_geometry_detail::SupportPoint& anchor, const exact_geometry_detail::SupportPoint& point,
    Reserve&& reserve)
{
    static_assert(Bits >= 128 && Bits <= 32768);
    std::array<unsigned, 3> active{};
    unsigned count = 0;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        require_positive_support_denominator(normal[axis]);
        require_positive_support_denominator(anchor[axis]);
        require_positive_support_denominator(point[axis]);
        if (normal[axis] != 0 && point[axis] != anchor[axis])
            active[count++] = axis;
    }
    if (!count)
        return 0;
    if (count == 1)
    {
        const auto axis = active[0];
        const bool negative = (normal[axis] < 0) != (point[axis] < anchor[axis]);
        return negative ? -1 : 1;
    }
    // Keep the reservation outside the arithmetic overflow handler: a failure
    // in the caller's accounting must not be mistaken for integer overflow.
    if (!std::forward<Reserve>(reserve)(integer_support_workspace_bytes<Bits>()))
        return std::nullopt;
    return integer_support_sign_nontrivial<Bits>(normal, anchor, point, active, count);
}
} // namespace SimpleFluid::PLIC::detail
