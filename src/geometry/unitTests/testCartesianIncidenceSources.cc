#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/SupportSurfaceCodec.hh"
#include "geometry/mesh/NativeFragmentRecipe.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <stdexcept>
namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Member = std::pair<AdaptiveCellKey, std::size_t>;
using Packets = std::map<Member, AdaptiveBytes>;
TEST(CartesianIncidenceSources, LiteralIntegerLatticeAndOriginalPacketsHaveExactlyTheSameMembers)
{
    auto value = grid();
    const auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    Packets expected, actual;
    std::map<AdaptiveBytes, std::size_t> group_sizes;
    for (auto leaf : native)
    {
        leaf.surfaces = kernel->surfaces(leaf);
        ASSERT_EQ(leaf.surfaces.size(), 6);
        for (std::size_t face = 0; face != 6; ++face)
        {
            expected.emplace(Member{leaf.key, face}, encode_support_surface(leaf, face));
            ++group_sizes[leaf.surfaces[face].key];
        }
    }
    EXPECT_EQ(expected.size(), 48);
    EXPECT_EQ(group_sizes.size(), 36);
    EXPECT_EQ(std::ranges::count_if(group_sizes, [](const auto& item) { return item.second == 1; }), 24);
    EXPECT_EQ(std::ranges::count_if(group_sizes, [](const auto& item) { return item.second == 2; }), 12);
    auto budget = std::make_shared<Budget>(Budget{ample});
    {
        const auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, *budget, budget);
        EXPECT_EQ(source->cell_count(), 8);
        ASSERT_EQ(source->ranges().size(), 1);
        source->visit_surfaces(*budget,
            [&](const auto& member)
            {
                EXPECT_TRUE(actual
                        .emplace(Member{member.cell, member.ordinal},
                            AdaptiveBytes(member.packet.begin(), member.packet.end()))
                        .second);
            });
        EXPECT_EQ(actual, expected);
        for (const auto& [key, count] : group_sizes)
        {
            Packets selected;
            source->visit_surfaces(
                *budget,
                [&](const auto& member)
                {
                    selected.emplace(
                        Member{member.cell, member.ordinal}, AdaptiveBytes(member.packet.begin(), member.packet.end()));
                },
                key);
            EXPECT_EQ(selected.size(), count);
            for (const auto& [member, bytes] : selected)
                EXPECT_EQ(bytes, expected.at(member));
        }
    }
    EXPECT_EQ(budget->used, 0);
}
TEST(CartesianIncidenceSources, TranslatedNonbinaryAllPeriodicMasksUseOriginalPhysicalPacketBytes)
{
    RestoreEnvironment restore;
    for (int rounding : {FE_TONEAREST, FE_UPWARD})
        for (unsigned mask = 0; mask != 8; ++mask)
        {
            ASSERT_EQ(std::fesetround(rounding), 0);
            auto value = grid();
            value.origin = {0.1, -0.7, 2.3};
            value.extent = {0.9, 1.3, 2.7};
            for (unsigned a = 0; a != 3; ++a)
                value.periodic[a] = bool(mask & (1U << a));
            const auto native = leaves(value);
            const auto kernel = cartesian_adaptive_kernel(value);
            Packets expected, actual;
            for (auto leaf : native)
            {
                leaf.surfaces = kernel->surfaces(leaf);
                for (std::size_t f = 0; f != leaf.surfaces.size(); ++f)
                    expected.emplace(Member{leaf.key, f}, encode_support_surface(leaf, f));
            }
            Budget budget{ample};
            {
                auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, budget);
                source->visit_surfaces(budget,
                    [&](const auto& member)
                    {
                        actual.emplace(Member{member.cell, member.ordinal},
                            AdaptiveBytes(member.packet.begin(), member.packet.end()));
                    });
                EXPECT_EQ(actual, expected) << "mask=" << mask << " rounding=" << rounding;
            }
            EXPECT_EQ(budget.used, 0);
        }
}
TEST(CartesianIncidenceSources, MalformedNativeDescriptorsAgreeWithUnrestrictedOriginalValidation)
{
    using SimpleFluid::Meshes::NativeFragmentBytes;
    using SimpleFluid::Meshes::NativeFragmentDescriptor;
    const auto value = grid();
    const auto leaf = leaves(value).front();
    const auto kernel = cartesian_adaptive_kernel(value);
    for (unsigned variant = 0; variant != 7; ++variant)
    {
        auto bad = leaf;
        auto descriptor = std::make_shared<NativeFragmentDescriptor>(*leaf.native_recipe);
        if (variant == 0)
            descriptor->kind = NativeFragmentDescriptor::Kind::LegacyProcedural;
        if (variant == 1)
            descriptor->view.back() ^= 1;
        if (variant == 2 || variant == 3 || variant == 5)
        {
            const auto part = variant == 2 ? 0 : 1;
            auto bytes = std::make_shared<NativeFragmentBytes>(*descriptor->parts[part]);
            if (variant == 5)
                bytes->push_back('x');
            else
                bytes->back() ^= 1;
            descriptor->parts[part] = std::move(bytes);
        }
        if (variant == 4)
            descriptor->parts[2].reset();
        if (variant == 6)
            bad.native_cell = 99;
        bad.native_recipe = std::move(descriptor);
        EXPECT_THROW(kernel->validate(bad), std::exception) << variant;
        Budget budget{ample};
        EXPECT_THROW(
            (void) CartesianIncidenceSource::make(value, std::span(&bad, 1), {'d'}, 0, 1, {}, budget), std::exception)
            << variant;
        EXPECT_EQ(budget.used, 0);
    }
}
TEST(CartesianIncidenceSources, ThrowingCallbacksAndFailedReevaluationLeaveImmutableSourceAndCreditsIntact)
{
    const auto value = grid();
    const auto native = leaves(value);
    Budget budget{ample};
    {
        auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, budget);
        const auto identity = source->identity();
        const auto before = budget.used;
        EXPECT_THROW(source->visit_surfaces(budget, [](const auto&) { throw std::runtime_error("bounded callback"); }),
            std::runtime_error);
        EXPECT_EQ(budget.used, before);
        Budget rejected{0};
        EXPECT_THROW((void) source->reevaluated(rejected), std::exception);
        EXPECT_EQ(rejected.used, 0);
        EXPECT_EQ(source->identity(), identity);
        auto other_native = native;
        for (auto& leaf : other_native)
            leaf.domain = {'e'};
        auto other = CartesianIncidenceSource::make(value, other_native, {'e'}, 0, 1, {}, budget);
        EXPECT_NE(other->identity(), identity);
    }
    EXPECT_EQ(budget.used, 0);
}
TEST(CartesianIncidenceSources, DerivedSupportMembershipMatchesCompleteEnumeration)
{
    auto value = grid();
    auto native = leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    std::map<AdaptiveBytes, std::set<Member>> expected;
    for (auto leaf : native)
    {
        leaf.surfaces = kernel->surfaces(leaf);
        for (std::size_t face = 0; face != leaf.surfaces.size(); ++face)
            expected[leaf.surfaces[face].key].insert(Member{leaf.key, face});
    }
    ASSERT_EQ(expected.size(), 36);
    auto budget = std::make_shared<Budget>(Budget{ample});
    {
        const auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, *budget, budget);
        for (const auto& [key, members] : expected)
        {
            std::set<Member> derived;
            const bool complete =
                source->visit_support_members(key, *budget, [&](const CartesianIncidenceSurface& surface)
                    { derived.insert(Member{surface.cell, surface.ordinal}); });
            ASSERT_TRUE(complete) << "each uniform nonperiodic support must derive";
            EXPECT_EQ(derived, members);
        }
        auto foreign = AdaptiveBytes{'x'};
        EXPECT_FALSE(source->visit_support_members(
            foreign, *budget, [&](const CartesianIncidenceSurface&) { FAIL() << "foreign key must emit nothing"; }));
    }
    EXPECT_EQ(budget->used, 0);
}
} // namespace
