/** Literal canonical scalar preflight must never allocate a comparison packet. */
#include "geometry/PLIC/details/ordinary/NativeGeometrySupport.hh"
#include "pipeline_allocation_operators.inc"
#include <array>
#include <cfenv>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace
{
namespace N = SimpleFluid::PLIC::details::ordinary_native;
template<std::size_t Size> constexpr auto literal(const std::array<std::uint64_t, Size>& words)
{
    std::array<char, Size * 8> bytes{};
    for (std::size_t word = 0; word < Size; ++word)
        for (unsigned byte = 0; byte < 8; ++byte)
            bytes[8 * word + byte] = static_cast<char>((words[word] >> (8 * byte)) & 255);
    return bytes;
}
struct Environment
{
    std::fenv_t previous{};
    Environment()
    {
        if (std::fegetenv(&previous) || std::fesetround(FE_TONEAREST))
            std::terminate();
    }
    ~Environment() { (void) std::fesetenv(&previous); }
};
template<unsigned Precision, std::uint64_t MinimumDescriptor> struct Packets
{
    static constexpr std::size_t limbs = (Precision + 31) / 32;
    static constexpr auto words(
        std::uint64_t descriptor, std::uint64_t bits, std::uint64_t first, std::uint64_t last = 0)
    {
        std::array<std::uint64_t, 2 + limbs> result{};
        result[0] = descriptor;
        result[1] = bits;
        result[2] = first;
        result.back() = last;
        return literal(result);
    }
    static constexpr auto one = words(2, Precision, 0x80000000);
    static constexpr auto negative = words(3, Precision, 0xc0000000);
    static constexpr auto minimum = words(MinimumDescriptor, Precision, 0x80000000);
    static constexpr auto bad_bits = words(2, Precision - 1, 0x80000000);
    static constexpr auto bad_limb = words(2, Precision, 0x180000000);
    static constexpr auto rounded = words(MinimumDescriptor, Precision, 0x80000000, 1);
};
// Independent format-contract literals. The descriptors encode the exact
// frexp exponents -1073, -16444 and -16493; no production writer is used.
using Binary64 = Packets<53, 0x1fffff79e>;
using Binary80 = Packets<64, 0x1ffff7f88>;
using Binary128 = Packets<113, 0x1ffff7f26>;
template<class F> bool with_supported_packets(F&& test)
{
    using L = std::numeric_limits<long double>;
    if constexpr (L::radix == 2 && L::digits == 53 && L::min_exponent == -1021 && L::max_exponent == 1024)
        test(Binary64{});
    else if constexpr (L::radix == 2 && L::digits == 64 && L::min_exponent == -16381 && L::max_exponent == 16384)
        test(Binary80{});
    else if constexpr (L::radix == 2 && L::digits == 113 && L::min_exponent == -16381 && L::max_exponent == 16384)
        test(Binary128{});
    else
        return false;
    return true;
}
TEST(NativeScalarPreflight, LiteralNonzeroAndSubnormalPacketsAllocateNoCppStorage)
{
    const auto run = []<class Profile>(Profile)
    {
        const Environment environment;
        constexpr auto one = Profile::one;
        constexpr auto negative = Profile::negative;
        constexpr auto minimum = Profile::minimum;
        constexpr auto zero = literal(std::array<std::uint64_t, 2>{0, 0});
        pipeline_trace::begin();
        N::Reader a{one, 0, true}, b{negative, 0, true}, c{minimum, 0, true}, d{zero, 0, true};
        const auto av = a.real(), bv = b.real(), cv = c.real(), dv = d.real();
        a.finish();
        b.finish();
        c.finish();
        d.finish();
        pipeline_trace::recording.store(false);
        const auto raw = pipeline_trace::summary();
        EXPECT_FALSE(pipeline_trace::overflow.load());
        std::size_t requests = 0;
        for (const auto& phase : raw.phase)
            requests += phase.allocations;
        EXPECT_EQ(requests, 0u);
        EXPECT_EQ(raw.all_live, 0u);
        EXPECT_EQ(av, 1.L);
        EXPECT_EQ(bv, -1.5L);
        EXPECT_EQ(cv, std::numeric_limits<long double>::denorm_min());
        EXPECT_EQ(dv, 0.L);
    };
    if (!with_supported_packets(run))
        GTEST_SKIP() << "Unsupported long double scalar profile; scalar preflight is unqualified.";
}
TEST(NativeScalarPreflight, LiteralNoncanonicalAndTruncatedPacketsRetainTheOriginalRejectionGate)
{
    const auto run = []<class Profile>(Profile)
    {
        const Environment environment;
        constexpr auto bad_zero = literal(std::array<std::uint64_t, 2>{1, 0});
        constexpr auto bad_bits = Profile::bad_bits;
        constexpr auto bad_limb = Profile::bad_limb;
        constexpr auto one = Profile::one;
        const auto reject = [](std::span<const char> bytes)
        {
            N::Reader reader{bytes, 0, true};
            (void) reader.real();
            reader.finish();
        };
        EXPECT_THROW(reject(bad_zero), std::invalid_argument);
        EXPECT_THROW(reject(bad_bits), std::invalid_argument);
        EXPECT_THROW(reject(bad_limb), std::invalid_argument);
        EXPECT_THROW(reject(std::span(one).first(one.size() - 1)), std::invalid_argument);
        // Rounding during decode loses the extra low limb of this subnormal.
        // Earlier syntax/range gates accept it; exact canonical comparison rejects.
        constexpr auto rounded = Profile::rounded;
        try
        {
            reject(rounded);
            FAIL() << "Noncanonical subnormal scalar was accepted.";
        }
        catch (const std::invalid_argument& error)
        {
            EXPECT_STREQ(error.what(), "Ordinary FP geometry: noncanonical finite scalar encoding.");
        }
    };
    if (!with_supported_packets(run))
        GTEST_SKIP() << "Unsupported long double scalar profile; scalar preflight is unqualified.";
}
} // namespace
