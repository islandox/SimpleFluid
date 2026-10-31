/** @file testNativeSupportOrdering.cc
 * @brief Native field/provenance ordering keeps support identities distinct.
 */
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <gtest/gtest.h>
#include <optional>
#include <set>
#if defined(__unix__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace
{
using namespace SimpleFluid;
namespace P = SimpleFluid::PLIC;
using Access = P::details::GeometryIdentityAccess;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr auto exact = ArithmeticMode::Exact;
using Support = P::PlaneSupportIdentity<ordinary>;
std::vector<P::details::OrdinarySupportRecipe> recipes()
{
    return {{true, {0, 0, 0}, {}}, {true, {1, 2, 0}, {}}, {true, {3, -1, 0}, {}}, {false, {}, {0, 1, 2}}};
}
Support retained_exact(double x)
{
    const std::array<P::GeometryPoint, 3> points{{{x, 0, 0}, {x, 1, 0}, {x, 0, 1}}};
    return Access::retained_exact_support(P::planar_support<exact>(*P::PlanarFace<exact>::make(points)));
}
TEST(NativeSupportOrderingTest, VariantRanksAreStableAndCoincidentExpressionsStayDistinct)
{
    const auto axis = Access::support_axis(0, 0);
    const std::vector<Support> values{axis, Access::support_source({{0, 0, 0}, {0, 1, 0}, {1, 0, 1}}),
        Access::support_input({1, 1, 0}, {}, 0), Access::support_plane(0, {1, 0, 0}, 0),
        Access::translated_support(axis, {0, 1, 0}), retained_exact(0), Access::support_dag(recipes(), 0, 3)};
    for (std::size_t i = 0; i < values.size(); ++i)
        for (std::size_t j = 0; j < values.size(); ++j)
        {
            EXPECT_EQ(values[i] < values[j], i < j);
            EXPECT_EQ(values[i] == values[j], i == j);
        }
    EXPECT_NE(values[0], values[3]); // Axis and normalized-plane tokens are different expressions.
    EXPECT_NE(values[0], values[4]); // Tangential translation retains its original ancestry.
    EXPECT_NE(values[0], values[5]); // Retained Exact authority remains a different variant.
}
TEST(NativeSupportOrderingTest, FixedAndVariableFieldsUseNumericLexicographicOrder)
{
    EXPECT_LT(Access::support_axis(0, -1), Access::support_axis(0, 1));
    EXPECT_LT(Access::support_axis(0, 1000), Access::support_axis(1, -1000));
    EXPECT_LT(Access::support_source({{0, 0, 0}, {0, 1, 0}, {1, 0, 1}}),
        Access::support_source({{0, 0, 0}, {0, 2, 0}, {1, 0, 1}}));
    EXPECT_LT(Access::support_input({1, 1, 0}, {100, 0, 0}, 100), Access::support_input({2, 1, 0}, {-100, 0, 0}, -100));
    EXPECT_LT(Access::support_input({1, 1, 0}, {0, 1, 0}, 100), Access::support_input({1, 1, 0}, {0, 2, 0}, -100));
    EXPECT_LT(Access::support_input({1, 1, 0}, {}, -1), Access::support_input({1, 1, 0}, {}, 1));
    EXPECT_LT(Access::support_plane(1, {.5L, 1, 0}, 100), Access::support_plane(1, {.75L, 1, 0}, -100));
    EXPECT_LT(retained_exact(-.5), retained_exact(.5));
}
TEST(NativeSupportOrderingTest, TranslationComparesParentsBeforeShiftWithoutFlattening)
{
    const auto first = Access::support_axis(0, 0), second = Access::support_axis(0, 1);
    EXPECT_LT(Access::translated_support(first, {100, 0, 0}), Access::translated_support(second, {-100, 0, 0}));
    EXPECT_LT(Access::translated_support(first, {-2, 0, 0}), Access::translated_support(first, {-1, 0, 0}));
    const auto nested = Access::translated_support(Access::translated_support(first, {1, 0, 0}), {1, 0, 0});
    const auto direct = Access::translated_support(first, {2, 0, 0});
    EXPECT_NE(nested, direct);
    EXPECT_GT(nested, direct); // Its parent is a Translation, whose rank follows Axis.
    const auto restored = P::decode_identity<Support>(P::encode_identity(nested));
    EXPECT_EQ(restored, nested);
    EXPECT_EQ(restored <=> nested, std::strong_ordering::equal);
    EXPECT_EQ(P::canonical_key_hash(restored), P::canonical_key_hash(nested));
}
TEST(NativeSupportOrderingTest, DagOrderingUsesRecordsThenEndpointsAndKeepsKeyIdentity)
{
    auto changed = recipes();
    changed[2].root[0] = 4;
    const auto base = Access::support_dag(recipes(), 0, 3);
    EXPECT_LT(base, Access::support_dag(std::move(changed), 0, 3));
    EXPECT_LT(base, Access::support_dag(recipes(), 1, 3));
    const auto restored = P::decode_identity<Support>(P::encode_identity(base));
    EXPECT_EQ(restored, base);
    EXPECT_EQ(restored <=> base, std::strong_ordering::equal);
    EXPECT_EQ(P::canonical_key(restored), P::canonical_key(base));
}

std::pair<std::size_t, std::size_t> unique_payload(const std::vector<Support>& identities)
{
    std::set<const void*> owners;
    std::size_t bytes = 0;
    for (const auto& identity : identities)
        P::visit_identity_allocations(identity,
            [&](const P::GeometryIdentityAllocation& allocation)
            {
                if (!owners.insert(allocation.owner.get()).second)
                    return false;
                bytes += allocation.payload_bytes;
                return true;
            });
    return {owners.size(), bytes};
}
TEST(NativeSupportOrderingTest, UniqueRetainedTranslationPayloadGrowsLinearly)
{
    const auto root = Access::support_axis(0, 0);
    const auto root_bytes = unique_payload({root}).second;
    const auto one = Access::translated_support(root, {0, 1, 0});
    const auto per_translation = one.payload_bytes();
    for (const std::size_t depth : {32, 64, 128})
    {
        auto head = root;
        for (std::size_t i = 0; i < depth; ++i)
            head = Access::translated_support(head, {0, 1, 0});
        const auto [owners, bytes] = unique_payload({head});
        EXPECT_EQ(owners, depth + 1);
        EXPECT_EQ(bytes, root_bytes + depth * per_translation);
        EXPECT_EQ(head.payload_bytes(), per_translation);
        const auto before_export = bytes;
        const auto wire = P::encode_identity(head);
        EXPECT_EQ(unique_payload({head}).second, before_export);
        const auto restored = P::decode_identity<Support>(wire);
        EXPECT_EQ(restored, head);
        EXPECT_EQ(unique_payload({restored}).first, depth + 1);
        EXPECT_EQ(unique_payload({restored}).second, bytes);
        EXPECT_EQ(unique_payload({head, restored}).first, 2 * (depth + 1));
        EXPECT_EQ(unique_payload({head, restored}).second, 2 * bytes);
    }
}
TEST(NativeSupportOrderingTest, SharedAncestorIsCountedOnceAndIndependentImportsRemainOwned)
{
    const auto root = Access::support_axis(0, 0);
    const auto ancestor = Access::translated_support(root, {0, 1, 0});
    const auto first = Access::translated_support(ancestor, {1, 0, 0});
    const auto second = Access::translated_support(ancestor, {-1, 0, 0});
    const auto shared = unique_payload({first, second});
    EXPECT_EQ(shared.first, 4u);
    EXPECT_EQ(shared.second,
        root.payload_bytes() + ancestor.payload_bytes() + first.payload_bytes() + second.payload_bytes());
    const auto copy = first;
    EXPECT_EQ(unique_payload({first, second, copy}), shared);
    std::optional<Support> independent;
    std::weak_ptr<const void> owner;
    {
        const auto wire = P::encode_identity(first);
        independent = P::decode_identity<Support>(wire);
        owner = independent->storage_owner();
    }
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(*independent, first);
    EXPECT_EQ(unique_payload({first, second, *independent}).first, 7u);
    independent.reset();
    EXPECT_TRUE(owner.expired());
}
std::vector<char> deep_translation_body(std::size_t depth)
{
    // Build the nested wire in linear time: all length prefixes, one leaf,
    // then one frozen binary scalar tuple per layer. Each shift is (0,1,0).
    // This avoids measuring cumulative eager construction while testing deep
    // import, emission, native comparison and final-owner destruction.
    const auto leaf = P::canonical_key(Access::support_axis(0, 0));
    std::vector<char> shift;
    const auto word = [](std::vector<char>& output, std::uint64_t value)
    {
        for (unsigned i = 0; i < 8; ++i)
            output.push_back(static_cast<char>(value >> (8 * i)));
    };
    word(shift, 0);
    word(shift, 0);
    word(shift, 2);
    word(shift, std::numeric_limits<long double>::digits);
    word(shift, 0x80000000ULL);
    for (int written = 32; written < std::numeric_limits<long double>::digits; written += 32)
        word(shift, 0);
    word(shift, 0);
    word(shift, 0);
    const auto layer = 9 + shift.size();
    std::vector<char> result;
    result.reserve(leaf.size() + depth * layer);
    for (std::size_t i = 0; i < depth; ++i)
    {
        result.push_back(static_cast<char>(P::GeometryTag::OrdinaryTranslatedSupport));
        word(result, leaf.size() + (depth - i - 1) * layer);
    }
    result.insert(result.end(), leaf.begin(), leaf.end());
    for (std::size_t i = 0; i < depth; ++i)
        result.insert(result.end(), shift.begin(), shift.end());
    return result;
}
bool deep_support_roundtrip()
{
    constexpr std::size_t depth = 8192;
    const auto body = deep_translation_body(depth);
    const auto wire = P::encode_geometry_packet(P::GeometryPacketKind::PlaneSupportIdentity, ordinary, body);
    std::optional<Support> first = P::decode_identity<Support>(wire);
    std::optional<Support> second = P::decode_identity<Support>(wire);
    if (unique_payload({*first}).first != depth + 1 || *first != *second || (*first <=> *second) != 0 ||
        P::canonical_key(*first) != body || P::encode_identity(*first) != wire)
        return false;
    auto weak = std::weak_ptr<const void>(first->storage_owner());
    first.reset();
    if (!weak.expired() || P::canonical_key(*second) != body)
        return false;
    second.reset();
    return true;
}
TEST(NativeSupportOrderingTest, DeepSupportImportEmissionComparisonAndFinalDestructionAreIterative)
{
#if defined(__unix__) && GTEST_HAS_DEATH_TEST
    // Bound the child stack, not the process running unrelated tests. Eight
    // thousand nested owners must survive every operation on a 256 KiB stack.
    EXPECT_EXIT(
        {
            rlimit limit{};
            if (getrlimit(RLIMIT_STACK, &limit) != 0)
                _exit(2);
            limit.rlim_cur = std::min<rlim_t>(limit.rlim_max, 256 * 1024);
            if (setrlimit(RLIMIT_STACK, &limit) != 0)
                _exit(3);
            _exit(deep_support_roundtrip() ? 0 : 4);
        },
        ::testing::ExitedWithCode(0), "");
#else
    EXPECT_TRUE(deep_support_roundtrip());
#endif
}

} // namespace
