/** @file testConstructionRegistry.cc
 * @brief Finite source-edge incidence and authenticated construction packet replay.
 */
#include "geometry/PLIC/ConstructionRegistry.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include <algorithm>
#include <array>
#include <gtest/gtest.h>

namespace
{
using SimpleFluid::ArithmeticMode;
namespace P = SimpleFluid::PLIC;
namespace D = P::details;
using Faces = std::vector<std::vector<std::size_t>>;
template<ArithmeticMode Mode> using Geometry = P::ConvexCellGeometry<Mode>;
template<ArithmeticMode Mode> using Point = P::Point<Mode>;

template<ArithmeticMode Mode> Point<Mode> source_point(std::array<long double, 3> coordinates)
{
    const std::array input{Point<Mode>(coordinates)};
    return P::point_average<Mode>(input);
}
template<ArithmeticMode Mode>
std::size_t upper_face(const Geometry<Mode>& geometry, std::size_t axis, long double value)
{
    const auto points = geometry.points();
    for (std::size_t face = 0; face < geometry.faces().size(); ++face)
        if (std::all_of(geometry.faces()[face].begin(), geometry.faces()[face].end(),
                [&](std::size_t vertex) { return points[vertex].value[axis] == value; }))
            return face;
    throw std::runtime_error("Fixture face not found.");
}
std::uint64_t word(const std::vector<char>& packet, std::size_t offset)
{
    if (offset + 8 > packet.size())
        throw std::runtime_error("Fixture packet offset is invalid.");
    std::uint64_t result = 0;
    for (unsigned i = 0; i < 8; ++i)
        result |= std::uint64_t(static_cast<unsigned char>(packet[offset + i])) << (8 * i);
    return result;
}
void word(std::vector<char>& packet, std::size_t offset, std::uint64_t value)
{
    if (offset + 8 > packet.size())
        throw std::runtime_error("Fixture packet offset is invalid.");
    for (unsigned i = 0; i < 8; ++i)
        packet[offset + i] = static_cast<char>(value >> (8 * i));
}
void provenance_failure(const auto& action)
{
    try
    {
        action();
        FAIL() << "Expected a provenance rejection.";
    }
    catch (const P::GeometryError& error)
    {
        EXPECT_EQ(error.code(), P::GeometryErrorCode::ProvenanceMismatch);
        EXPECT_FALSE(error.recoverable_trial());
    }
}

TEST(ConstructionRegistryTest, CapturedWallSourceEdgeSurvivesComplementExportAndPacket)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    if (!P::certification_environment())
        GTEST_SKIP() << "Source incidence filter requires its supported FP environment.";
    const std::array<P::GeometryPoint, 8> corners{
        {{0, 0, 0}, {.25, 0, 0}, {.35, .5, 0}, {.1, .5, 0}, {0, 0, .5}, {.25, 0, .5}, {.35, .5, .5}, {.1, .5, .5}}};
    const Faces faces{{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    const auto native = Geometry<mode>::make(corners, faces);
    const auto donor = Geometry<mode>::from_geometry(*native);
    std::vector<Point<mode>> top;
    for (const auto& point : donor->points())
        if (point.value[2] == .5L)
            top.push_back(point);
    const auto a = P::point_average<mode>(top);
    const auto d = source_point<mode>({0.174991507443973570943L, 0.250042432475806645484L, 0.499994860273139090859L});
    const auto f = source_point<mode>({0.350007862053653909896L, 0.500055715528393436662L, 0.500000065164287879681L});
    const auto g = source_point<mode>({0.100022802874966634604L, 0.500114014374833145272L, 0.499999705241252620898L});
    const auto b = source_point<mode>({0, 0, 0.499997040787199163156L});
    const auto first = Geometry<mode>::tetrahedron({a, d, f, g});
    const auto query = Geometry<mode>::tetrahedron({a, d, b, g});
    auto parent = donor;
    for (const auto face : {3u, 1u})
    {
        const auto split = parent->clip_by_face(*first, face, false);
        ASSERT_TRUE(split.inside);
        parent = split.inside;
    }
    ASSERT_GT(parent->volume().value, 0);
    bool generated_edge = false;
    for (const auto& point : parent->points())
        if (point.value[1] == .5L)
        {
            const auto identity =
                P::DynamicGeometryAccess::original_source(P::DynamicGeometryAccess::identity(point.dynamic()));
            generated_edge |= identity && identity->constructed && !identity->source_edges.empty();
        }
    EXPECT_TRUE(generated_edge);
    const auto split = parent->clip_by_face(*query, 3, false);
    ASSERT_TRUE(split.inside);
    EXPECT_FALSE(split.outside);
    EXPECT_EQ(split.inside->storage_identity(), parent->storage_identity());
    const P::GeometryPacketContext pins{{17, 0, 0, 19}};
    P::ConstructionRegistry<mode> registry(pins);
    registry.register_native(0, 0, donor);
    registry.register_sweep(17, 0, first);
    registry.register_sweep(17, 1, query);
    const auto packet = parent->encode(registry);
    const auto decoded = Geometry<mode>::decode(packet, registry);
    EXPECT_EQ(decoded->volume().value, parent->volume().value);
    const auto repeated = decoded->clip_by_face(*query, 3, false);
    ASSERT_TRUE(repeated.inside);
    EXPECT_FALSE(repeated.outside);
    EXPECT_EQ(repeated.inside->storage_identity(), decoded->storage_identity());
    const auto exported = Geometry<mode>::make_points(decoded->points(), decoded->faces());
    const auto exported_cut = exported->clip_by_face(*query, 3, false);
    ASSERT_TRUE(exported_cut.inside);
    EXPECT_FALSE(exported_cut.outside);
}

template<ArithmeticMode Mode> void distinct_edge_control()
{
    const auto a = source_point<Mode>({.25L, .75L, .25L});
    const auto d = source_point<Mode>({.25L, .25L, .25L});
    const auto f = source_point<Mode>({.75L, .25L, .25L});
    const auto g = source_point<Mode>({.75L, .75L, .75L});
    const auto b = source_point<Mode>({.25L, .25L, .75L});
    const auto shifted = source_point<Mode>({.75L + 0x1p-30L, .75L, .75L});
    const auto tetra = Geometry<Mode>::tetrahedron({a, d, f, g});
    const auto wall = Geometry<Mode>::box({0, 0, 0}, {1, .5, 1});
    const auto parent = tetra->clip_by_face(*wall, upper_face(*wall, 1, .5L), false).inside;
    ASSERT_TRUE(parent);
    const auto query = Geometry<Mode>::tetrahedron({a, d, b, g});
    const auto nearby = Geometry<Mode>::tetrahedron({a, d, b, shifted});
    const auto exact_split = parent->clip_by_face(*query, 3, false);
    const auto changed = parent->clip_by_face(*nearby, 3, false);
    ASSERT_TRUE(changed.inside);
    ASSERT_TRUE(changed.outside);
    ASSERT_TRUE(exact_split.inside);
    EXPECT_GT(changed.inside->volume().value, 0);
    EXPECT_GT(changed.outside->volume().value, 0);
    EXPECT_NE(changed.inside->volume().value, exact_split.inside->volume().value);
    EXPECT_NEAR(changed.inside->volume().value + changed.outside->volume().value, parent->volume().value,
        parent->volume().value * 8e-15L);
}
TEST(ConstructionRegistryTest, NearbyDistinctEdgeKeepsPositiveRegions)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Source incidence filter requires its supported FP environment.";
    distinct_edge_control<ArithmeticMode::Ordinary>();
    distinct_edge_control<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode> void replay_and_reject()
{
    const auto donor = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    const auto a = source_point<Mode>({.5L, -1, -1});
    const auto b = source_point<Mode>({.5L, 2, -1});
    const auto c = source_point<Mode>({.5L, -1, 2});
    const auto d = source_point<Mode>({-1, -1, -1});
    const auto cutter = Geometry<Mode>::tetrahedron({a, b, c, d});
    const auto face = upper_face(*cutter, 0, .5L);
    const auto split = donor->clip_by_face(*cutter, face, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const P::GeometryPacketContext pins{{23, 0, 0, 29}};
    P::ConstructionRegistry<Mode> registry(pins);
    registry.register_native(0, 0, donor);
    registry.register_native(1, 0, donor);
    registry.register_sweep(0, 0, cutter);
    const P::ConstructionRootKey native_key{P::ConstructionRootKey::Kind::Native, 0, 0, 0, {}};
    for (const auto& child : {split.inside, split.outside})
    {
        const auto packet = child->encode(registry);
        EXPECT_EQ(static_cast<std::uint8_t>(packet[1]), 5u);
        const auto roots = P::ConstructionRegistry<Mode>::required_roots(packet, pins);
        EXPECT_EQ(roots.size(), 2u);
        const auto decoded = Geometry<Mode>::decode(packet, registry);
        EXPECT_EQ(decoded->volume().value, child->volume().value);
        EXPECT_TRUE(registry.derived_from(*decoded, native_key));
        EXPECT_EQ(decoded->encode(registry), packet);
        auto stale = packet;
        word(stale, 19 + 3 * 8, pins.publication[3] + 1);
        provenance_failure([&] { (void) Geometry<Mode>::decode(stale, registry); });
        auto mixed = packet;
        mixed[2] = static_cast<char>(Mode == ArithmeticMode::Ordinary ? 1 : 0);
        provenance_failure([&] { (void) Geometry<Mode>::decode(mixed, registry); });
        P::ConstructionRegistry<Mode> missing(pins);
        missing.register_native(0, 0, donor);
        provenance_failure([&] { (void) Geometry<Mode>::decode(packet, missing); });
        auto changed = packet;
        changed.back() ^= 1;
        provenance_failure([&] { (void) Geometry<Mode>::decode(changed, registry); });
        auto cycle = packet;
        constexpr std::size_t root_count_offset = 19 + 4 * 8;
        std::size_t cursor = root_count_offset + 8;
        for (std::size_t root = 0; root < word(packet, root_count_offset); ++root)
        {
            cursor += 1 + 3 * 8;
            const auto bytes = word(packet, cursor);
            cursor += 8 + static_cast<std::size_t>(bytes);
        }
        auto unused = packet;
        std::vector<char> extra_root(1 + 4 * 8, 0);
        extra_root[0] = static_cast<char>(P::ConstructionRootKey::Kind::Native);
        word(extra_root, 1, 1);
        unused.insert(unused.begin() + cursor, extra_root.begin(), extra_root.end());
        word(unused, root_count_offset, word(packet, root_count_offset) + 1);
        provenance_failure([&] { (void) Geometry<Mode>::decode(unused, registry); });
        ASSERT_EQ(word(packet, cursor), 2u);
        cursor += 8;
        ASSERT_EQ(static_cast<std::uint8_t>(packet[cursor]), 1u);
        cursor += 1 + 8;
        ASSERT_EQ(static_cast<std::uint8_t>(packet[cursor]), 2u);
        word(cycle, cursor + 1, 1);
        provenance_failure([&] { (void) Geometry<Mode>::decode(cycle, registry); });
        P::GeometryLimits capped;
        capped.byte_budget = packet.size() - 1;
        EXPECT_THROW((void) Geometry<Mode>::decode(packet, registry, capped), P::GeometryError);
    }
    // A valid geometry may use the native donor only as a cutting plane. Its
    // manifest contains that donor, but its body descends from the sweep root.
    const auto sweep_based = cutter->clip_by_face(*donor, 0, false).inside;
    ASSERT_TRUE(sweep_based);
    const auto replayed = Geometry<Mode>::decode(sweep_based->encode(registry), registry);
    EXPECT_FALSE(registry.derived_from(*replayed, native_key));
}
TEST(ConstructionRegistryTest, V4ReplayRejectsMissingStaleCyclicModeAndNumericClaims)
{
    replay_and_reject<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        replay_and_reject<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode> void rebase_clip_chain()
{
    using Node = D::GeometryConstruction;
    auto source = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    auto source_impl = P::DynamicGeometryAccess::implementation(*source->dynamic_geometry());
    std::weak_ptr<const P::DynamicConvexGeometry::Impl> source_lifetime = source_impl;
    const P::GeometryHalfspace cut{{2, 0, 0}, {.125, .25, .375}, .75}; // x <= 1/2
    const std::array<P::GeometryHalfspace, 2> noncutting{{
        {{0, 4, 0}, {.25, .5, .75}, 6}, // y <= 2, wholly inside
        {{0, 2, 0}, {.75, .25, .5}, -2.5} // y <= -1, wholly outside
    }};
    const auto split = source->clip(cut, true);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const auto inside_node = P::DynamicGeometryAccess::implementation(*split.inside->dynamic_geometry())->construction;
    const auto outside_node =
        P::DynamicGeometryAccess::implementation(*split.outside->dynamic_geometry())->construction;
    ASSERT_EQ(inside_node->kind(), Node::Kind::Clip);
    EXPECT_FALSE(inside_node->clip().outside);
    EXPECT_TRUE(outside_node->clip().outside);
    EXPECT_EQ(inside_node->clip().clip, outside_node->clip().clip);
    const auto rebased = inside_node->clip().clip->parent;
    ASSERT_EQ(rebased->kind(), Node::Kind::Rebase);
    EXPECT_EQ(rebased->rebase().input.normal, cut.normal);
    EXPECT_EQ(rebased->rebase().input.anchor, cut.anchor);
    EXPECT_EQ(rebased->rebase().input.offset, cut.offset);
    ASSERT_EQ(rebased->rebase().parent->kind(), Node::Kind::Source);
    EXPECT_EQ(rebased->rebase().parent->source().source, source_impl);
    EXPECT_FALSE(source_impl->construction); // The pinned source cannot point back to its recipe.

    const P::GeometryPacketContext pins{{47, 53, 59, 61}};
    const P::ConstructionRootKey native{P::ConstructionRootKey::Kind::Native, 7, 11, 0, {}};
    for (std::size_t side = 0; side < 2; ++side)
        for (std::size_t noncut = 0; noncut < noncutting.size(); ++noncut)
        {
            SCOPED_TRACE(::testing::Message() << "side=" << side << " noncut=" << noncut);
            const auto& child = side ? split.outside : split.inside;
            const auto repeated = child->clip(noncutting[noncut], false);
            const auto terminal = noncut ? repeated.outside : repeated.inside;
            ASSERT_TRUE(terminal);
            EXPECT_FALSE(noncut ? repeated.inside : repeated.outside);
            const auto node = P::DynamicGeometryAccess::implementation(*terminal->dynamic_geometry())->construction;
            ASSERT_EQ(node->kind(), Node::Kind::Rebase);
            EXPECT_EQ(node->rebase().parent, side ? outside_node : inside_node);
            EXPECT_EQ(node->rebase().input.normal, noncutting[noncut].normal);
            EXPECT_EQ(node->rebase().input.anchor, noncutting[noncut].anchor);
            EXPECT_EQ(node->rebase().input.offset, noncutting[noncut].offset);
            P::ConstructionRegistry<Mode> registry(pins);
            registry.register_native(7, 11, source);
            registry.register_plane(13, cut);
            registry.register_plane(17, noncutting[noncut]);
            const auto packet = terminal->encode(registry);
            const std::vector<P::ConstructionRootKey> expected_roots{native,
                {P::ConstructionRootKey::Kind::InputPlane, 13, 0, 0, {}},
                {P::ConstructionRootKey::Kind::InputPlane, 17, 0, 0, {}}};
            EXPECT_EQ(P::ConstructionRegistry<Mode>::required_roots(packet, pins), expected_roots);
            EXPECT_TRUE(registry.derived_from(*terminal, native));
            const auto decoded = Geometry<Mode>::decode(packet, registry);
            EXPECT_TRUE(registry.derived_from(*decoded, native));
            EXPECT_EQ(decoded->encode(registry), packet);
            EXPECT_EQ(decoded->faces(), terminal->faces());
            const auto original_points = terminal->points(), replayed_points = decoded->points();
            ASSERT_EQ(replayed_points.size(), original_points.size());
            for (std::size_t point = 0; point < original_points.size(); ++point)
            {
                EXPECT_EQ(replayed_points[point].value, original_points[point].value);
                if constexpr (Mode == ArithmeticMode::BoundedCertification)
                    EXPECT_EQ(replayed_points[point].absolute_error, original_points[point].absolute_error);
            }
            const auto expected_x = side ? .375L : .125L;
            for (const auto& geometry : {terminal, decoded})
            {
                const auto moments = geometry->moments({0, 0, 0});
                const auto check = [](const auto& value, long double expected)
                {
                    if constexpr (Mode == ArithmeticMode::Ordinary)
                        EXPECT_NEAR(value.value, expected, 2e-15L);
                    else
                    {
                        EXPECT_LE(value.lower(), expected);
                        EXPECT_GE(value.upper(), expected);
                    }
                };
                // Integral over [0,1/2]x[0,1]^2 or [1/2,1]x[0,1]^2.
                check(moments.volume, .5L);
                check(moments.first_moment[0], expected_x);
                check(moments.first_moment[1], .25L);
                check(moments.first_moment[2], .25L);
            }
        }
    source_impl.reset();
    source.reset();
    EXPECT_FALSE(source_lifetime.expired());
    EXPECT_NEAR(split.inside->moments({0, 0, 0}).first_moment[0].value, .125L, 2e-15L);
}
TEST(ConstructionRegistryTest, BinaryExactSourceRebaseClipRebaseChainsKeepRootsAndMoments)
{
    rebase_clip_chain<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        rebase_clip_chain<ArithmeticMode::BoundedCertification>();
}
} // namespace
