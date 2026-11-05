/** @file testConstructionRegistry.cc
 * @brief Finite source-edge incidence and authenticated construction packet replay.
 */
#include "geometry/PLIC/ConstructionRegistry.hh"
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/PLIC/details/CanonicalExportCoordinates.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/unitTests/point_export_test_helpers.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include <algorithm>
#include <array>
#include <cfenv>
#include <gtest/gtest.h>

namespace
{
using SimpleFluid::PLIC::test::exported_points;
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
    const auto points = exported_points(geometry);
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
    for (const auto& point : exported_points(*donor))
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
    for (const auto& point : exported_points(*parent))
        if (point.value[1] == .5L)
        {
            const auto identity =
                P::DynamicGeometryAccess::original_source(P::DynamicGeometryAccess::identity(point.dynamic()));
            generated_edge |= identity && identity.view().constructed() && !identity.view().source_edges().empty();
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
    const auto exported = Geometry<mode>::make_points(exported_points(*decoded), decoded->faces());
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
    EXPECT_EQ(rebased->rebase().parent->source().geometry(), source_impl);
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
            const auto original_points = exported_points(*terminal), replayed_points = exported_points(*decoded);
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

std::shared_ptr<const P::CartesianCellSource> native_box(double upper_x = 1,
    D::NativeCellGeometryMoments policy = D::NativeCellGeometryMoments::PrimedState)
{
    P::CartesianSourceDescription input;
    input.axes = {{{0, upper_x}, {0, 1}, {0, 1}}};
    P::CartesianSourceCellSlot cell;
    cell.upper = {1, 1, 1};
    input.cells.push_back(cell);
    return D::NativeCellGeometrySourceAccess::make(std::move(input), {}, policy);
}
P::ConvexGeometryPtr<ArithmeticMode::Ordinary> native_materialize(
    const std::shared_ptr<const P::CartesianCellSource>& source, P::CellGeometryOperationWorkspace& workspace)
{
    const auto catalog = P::CellGeometryCatalog<ArithmeticMode::Ordinary>::cartesian(source);
    return P::materialize_cell(P::CellGeometryView<ArithmeticMode::Ordinary>(catalog, 0), workspace);
}

TEST(ConstructionRegistryTest, NativeCutsReleaseInputShellsAndMatchExplicitReplayBytes)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    const P::GeometryHalfspace x{{1, 0, 0}, {0, 0, 0}, .25};
    const P::GeometryHalfspace y{{0, 1, 0}, {0, 0, 0}, .5};
    const P::GeometryPacketContext pins{{41, 0, 0, 43}};
    P::ConstructionRegistry<mode> registry(pins);
    registry.register_plane(0, x);
    registry.register_plane(1, y);
    P::ConvexGeometryPtr<mode> quarter, eighth, outside;
    std::weak_ptr<const P::DynamicConvexGeometry::Impl> original;
    {
        P::CellGeometryOperationWorkspace workspace;
        const auto source = native_materialize(native_box(), workspace);
        original = P::DynamicGeometryAccess::implementation(*source->dynamic_geometry());
        registry.register_native(9, 0, source);
        registry.register_native(0, 0, source);
        const auto first = source->clip(x, false);
        quarter = first.inside;
        outside = first.outside;
        eighth = quarter->clip(y, false).inside;
    }
    EXPECT_TRUE(original.expired());
    ASSERT_TRUE(quarter);
    ASSERT_TRUE(outside);
    ASSERT_TRUE(eighth);
    EXPECT_EQ(quarter->volume().value, .25L);
    EXPECT_EQ(outside->volume().value, .75L);
    EXPECT_EQ(eighth->volume().value, .125L);
    EXPECT_EQ(quarter->centroid(), (P::GeometryPoint{.125, .5, .5}));
    EXPECT_EQ(eighth->centroid(), (P::GeometryPoint{.125, .25, .5}));
    const auto packet = eighth->encode(registry);
    const auto decoded = Geometry<mode>::decode(packet, registry);
    EXPECT_EQ(decoded->encode(registry), packet);
    EXPECT_EQ(decoded->encode_certificate(), eighth->encode_certificate());
    const auto explicit_source = Geometry<mode>::from_geometry(*Geometry<mode>::box({0, 0, 0}, {1, 1, 1}));
    P::ConstructionRegistry<mode> explicit_registry(pins);
    explicit_registry.register_native(0, 0, explicit_source);
    explicit_registry.register_plane(0, x);
    explicit_registry.register_plane(1, y);
    const auto expected = explicit_source->clip(x, false).inside->clip(y, false).inside;
    EXPECT_EQ(packet, expected->encode(explicit_registry));
    const auto replay_work = [&](const auto& authority, P::GeometryLimits limits)
    {
        limits.arithmetic_mode = mode;
        D::Operation operation(limits);
        static_cast<void>(Geometry<mode>::decode(packet, authority, limits));
        return std::pair{operation.work.operations, operation.work.pairs};
    };
    const auto work = replay_work(explicit_registry, {});
    EXPECT_EQ(replay_work(registry, {}), work);
    P::GeometryLimits exact;
    exact.maximum_operations = work.first;
    exact.maximum_pair_tests = work.second;
    EXPECT_EQ(replay_work(registry, exact), work);
    EXPECT_EQ(replay_work(explicit_registry, exact), work);
    ASSERT_GT(work.first, 1u);
    --exact.maximum_operations;
    EXPECT_THROW(static_cast<void>(replay_work(registry, exact)), P::GeometryError);
    EXPECT_THROW(static_cast<void>(replay_work(explicit_registry, exact)), P::GeometryError);
}

TEST(ConstructionRegistryTest, NativeParentAndCutterAuthenticateAcrossMaterializationOnlyWithinTheirSource)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    const auto input = native_box();
    const auto cutter_input = native_box(.25);
    P::ConvexGeometryPtr<mode> cut;
    std::weak_ptr<const P::DynamicConvexGeometry::Impl> original, original_cutter;
    std::weak_ptr<const D::NativeGeometryBody> cutter_body;
    {
        P::CellGeometryOperationWorkspace workspace;
        const auto source = native_materialize(input, workspace);
        const auto cutter = native_materialize(cutter_input, workspace);
        original = P::DynamicGeometryAccess::implementation(*source->dynamic_geometry());
        original_cutter = P::DynamicGeometryAccess::implementation(*cutter->dynamic_geometry());
        cutter_body = P::DynamicGeometryAccess::implementation(*cutter->dynamic_geometry())
                          ->construction->source().native()->body;
        cut = source->clip_by_face(*cutter, 1, false).inside;
    }
    ASSERT_TRUE(cut);
    EXPECT_TRUE(original.expired());
    EXPECT_TRUE(original_cutter.expired());
    EXPECT_FALSE(cutter_body.expired());
    ASSERT_TRUE(P::DynamicGeometryAccess::implementation(*cut->dynamic_geometry())
                    ->construction->clip().clip->native_plane_source);
    const P::GeometryPacketContext pins{{47, 0, 0, 53}};
    {
        P::ConstructionRegistry<mode> registry(pins);
        {
            P::CellGeometryOperationWorkspace workspace;
            registry.register_native(9, 0, native_materialize(input, workspace));
            registry.register_native(0, 0, native_materialize(input, workspace));
            registry.register_native(1, 0, native_materialize(cutter_input, workspace));
        }
        const auto packet = cut->encode(registry);
        EXPECT_EQ(Geometry<mode>::decode(packet, registry)->encode(registry), packet);
        EXPECT_TRUE(registry.derived_from(*cut, {P::ConstructionRootKey::Kind::Native, 0, 0, 0, {}}));
        const auto explicit_parent = Geometry<mode>::from_geometry(*Geometry<mode>::box({0, 0, 0}, {1, 1, 1}));
        const auto explicit_cutter = Geometry<mode>::from_geometry(*Geometry<mode>::box({0, 0, 0}, {.25, 1, 1}));
        P::ConstructionRegistry<mode> explicit_registry(pins);
        explicit_registry.register_native(0, 0, explicit_parent);
        explicit_registry.register_native(1, 0, explicit_cutter);
        EXPECT_EQ(packet, explicit_parent->clip_by_face(*explicit_cutter, 1, false).inside->encode(explicit_registry));
        P::ConstructionRegistry<mode> foreign(pins);
        {
            P::CellGeometryOperationWorkspace workspace;
            foreign.register_native(0, 0, native_materialize(native_box(), workspace));
            foreign.register_native(1, 0, native_materialize(cutter_input, workspace));
        }
        provenance_failure([&] { static_cast<void>(cut->encode(foreign)); });
        EXPECT_FALSE(foreign.derived_from(*cut, {P::ConstructionRootKey::Kind::Native, 0, 0, 0, {}}));
    }
    EXPECT_FALSE(cutter_body.expired()); // Only the cut retains its authenticated native input body now.
    cut.reset();
    EXPECT_TRUE(cutter_body.expired());
}

TEST(ConstructionRegistryTest, DeferredNativeRepresentationsKeepIndependentFirstQueryCaches)
{
    std::fenv_t saved;
    ASSERT_EQ(std::fegetenv(&saved), 0);
    struct Restore
    {
        std::fenv_t state;
        ~Restore() { static_cast<void>(std::fesetenv(&state)); }
    } restore{saved};
    ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
    const P::GeometryPoint lower{-.1, .3, -.7}, upper{.8, 1.7, .4};
    P::CartesianSourceDescription description;
    for (std::size_t axis = 0; axis < 3; ++axis)
        description.axes[axis] = {lower[axis], upper[axis]};
    P::CartesianSourceCellSlot cell;
    cell.upper = {1, 1, 1};
    description.cells.push_back(cell);
    const auto input = D::NativeCellGeometrySourceAccess::make(
        std::move(description), {}, D::NativeCellGeometryMoments::DeferredRaw);
    P::CellGeometryOperationWorkspace first_workspace, second_workspace;
    const auto first = native_materialize(input, first_workspace);
    const auto second = native_materialize(input, second_workspace);
    const auto a = P::DynamicGeometryAccess::implementation(*first->dynamic_geometry())->construction;
    const auto b = P::DynamicGeometryAccess::implementation(*second->dynamic_geometry())->construction;
    ASSERT_TRUE(a->source().native());
    ASSERT_TRUE(b->source().native());
    EXPECT_FALSE(D::same_native_construction(*a, *b));
    EXPECT_NE(a->source().native()->moments, b->source().native()->moments);
    std::size_t index = 0;
    for (const auto direction : {FE_UPWARD, FE_DOWNWARD})
    {
        ASSERT_EQ(std::fesetround(FE_TONEAREST), 0);
        const auto explicit_source = Geometry<ArithmeticMode::Ordinary>::box(lower, upper);
        ASSERT_EQ(std::fesetround(direction), 0);
        const auto expected = explicit_source->moments();
        const auto& candidate = index++ ? second : first;
        const auto actual = candidate->moments();
        EXPECT_EQ(actual.volume.value, expected.volume.value);
        for (std::size_t axis = 0; axis < 3; ++axis)
            EXPECT_EQ(actual.first_moment[axis].value, expected.first_moment[axis].value);
        ASSERT_EQ(std::feclearexcept(FE_ALL_EXCEPT), 0);
        ASSERT_EQ(std::feraiseexcept(FE_DIVBYZERO), 0);
        static_cast<void>(D::OrdinaryExportCoordinates(*candidate).at(0));
        EXPECT_EQ(std::fegetround(), direction);
        EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), FE_DIVBYZERO);
    }
}

TEST(ConstructionRegistryTest, NativeRebaseExportsAndCachedMomentsSurviveOriginalShell)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    const P::GeometryHalfspace enclosing{{1, 0, 0}, {8, 0, 0}, 0};
    const P::GeometryPacketContext pins{{59, 0, 0, 61}};
    P::ConstructionRegistry<mode> registry(pins);
    registry.register_plane(0, enclosing);
    P::ConvexGeometryPtr<mode> rebased;
    std::weak_ptr<const P::DynamicConvexGeometry::Impl> original;
    std::vector<char> expected_certificate;
    {
        P::CellGeometryOperationWorkspace workspace;
        const auto source = native_materialize(native_box(), workspace);
        original = P::DynamicGeometryAccess::implementation(*source->dynamic_geometry());
        EXPECT_EQ(source->moments().first_moment[0].value, .5L);
        expected_certificate = source->encode_certificate();
        registry.register_native(0, 0, source);
        rebased = source->clip(enclosing, false).inside;
    }
    ASSERT_TRUE(rebased);
    EXPECT_TRUE(original.expired());
    EXPECT_EQ(rebased->encode_certificate(), expected_certificate);
    EXPECT_EQ(rebased->moments().first_moment[0].value, .5L);
    const auto points = exported_points(*rebased);
    ASSERT_EQ(points.size(), 8u);
    for (const auto& point : points)
        for (const auto value : point.value)
            EXPECT_TRUE(value == 0 || value == 1);
    const auto packet = rebased->encode(registry);
    const auto decoded = Geometry<mode>::decode(packet, registry);
    EXPECT_EQ(decoded->encode(registry), packet);
    EXPECT_EQ(decoded->encode_certificate(), expected_certificate);
}

TEST(ConstructionRegistryTest, InheritedNativeFacesKeepCanonicalMinimumRegisteredRoot)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    const auto input = native_box();
    const auto cutter_input = native_box(.25);
    const P::GeometryPacketContext pins{{67, 0, 0, 71}};
    for (const bool rebase_only : {false, true})
    {
        SCOPED_TRACE(rebase_only);
        P::CellGeometryOperationWorkspace workspace;
        const auto source = native_materialize(input, workspace);
        const auto cutter = native_materialize(cutter_input, workspace);
        const P::GeometryHalfspace change = rebase_only
            ? P::GeometryHalfspace{{1, 0, 0}, {8, 0, 0}, 0}
            : P::GeometryHalfspace{{0, 1, 0}, {0, 0, 0}, .75};
        const auto derived_cutter = cutter->clip(change, false).inside;
        ASSERT_TRUE(derived_cutter);
        const auto face = upper_face(*derived_cutter, 0, .25L);
        const auto cut = source->clip_by_face(*derived_cutter, face, false).inside;
        ASSERT_TRUE(cut);
        P::ConstructionRegistry<mode> registry(pins);
        registry.register_native(9, 0, source);
        registry.register_native(2, 0, cutter);
        const auto native_packet = cut->encode(registry);
        EXPECT_EQ(Geometry<mode>::decode(native_packet, registry)->encode(registry), native_packet);
        registry.register_native(1, 0, derived_cutter);
        const auto canonical = cut->encode(registry);
        const auto roots = P::ConstructionRegistry<mode>::required_roots(canonical, pins);
        const P::ConstructionRootKey derived_key{P::ConstructionRootKey::Kind::Native, 1, 0, 0, {}};
        const P::ConstructionRootKey original_key{P::ConstructionRootKey::Kind::Native, 2, 0, 0, {}};
        EXPECT_NE(std::find(roots.begin(), roots.end(), derived_key), roots.end());
        EXPECT_EQ(std::find(roots.begin(), roots.end(), original_key), roots.end());
        EXPECT_EQ(Geometry<mode>::decode(canonical, registry)->encode(registry), canonical);
    }
}

TEST(ConstructionRegistryTest, InheritedOtherNativeCutterBodySurvivesWithoutBecomingAGeometricParent)
{
    constexpr auto mode = ArithmeticMode::Ordinary;
    const P::GeometryPacketContext pins{{73, 0, 0, 79}};
    for (const bool rebase : {false, true})
    {
        SCOPED_TRACE(rebase);
        const auto donor_input = native_box();
        const auto irrelevant_parent_input = native_box();
        const auto cutter_input = native_box(.5);
        P::ConvexGeometryPtr<mode> cut;
        std::weak_ptr<const D::NativeGeometryBody> cutter_body, irrelevant_parent_body;
        std::weak_ptr<const P::DynamicConvexGeometry::Impl> temporary_cutter;
        {
            P::CellGeometryOperationWorkspace workspace;
            const auto donor = native_materialize(donor_input, workspace);
            const auto parent = native_materialize(irrelevant_parent_input, workspace);
            const auto cutter = native_materialize(cutter_input, workspace);
            irrelevant_parent_body = P::DynamicGeometryAccess::implementation(*parent->dynamic_geometry())
                                         ->construction->source().native()->body;
            cutter_body = P::DynamicGeometryAccess::implementation(*cutter->dynamic_geometry())
                              ->construction->source().native()->body;
            temporary_cutter = P::DynamicGeometryAccess::implementation(*cutter->dynamic_geometry());
            auto inherited = parent->clip_by_face(*cutter, 1, false).inside;
            ASSERT_TRUE(inherited);
            if (rebase)
                inherited = inherited->clip({{1, 0, 0}, {8, 0, 0}, 0}, false).inside;
            ASSERT_TRUE(inherited);
            cut = donor->clip_by_face(*inherited, upper_face(*inherited, 0, .5L), false).inside;
        }
        ASSERT_TRUE(cut);
        EXPECT_TRUE(temporary_cutter.expired());
        EXPECT_TRUE(irrelevant_parent_body.expired());
        EXPECT_FALSE(cutter_body.expired());
        {
            P::ConstructionRegistry<mode> registry(pins);
            P::CellGeometryOperationWorkspace workspace;
            registry.register_native(0, 0, native_materialize(donor_input, workspace));
            registry.register_native(1, 0, native_materialize(cutter_input, workspace));
            const auto packet = cut->encode(registry);
            const auto roots = P::ConstructionRegistry<mode>::required_roots(packet, pins);
            EXPECT_EQ(roots.size(), 2u);
            EXPECT_EQ(Geometry<mode>::decode(packet, registry)->encode(registry), packet);
            EXPECT_TRUE(registry.derived_from(*cut, {P::ConstructionRootKey::Kind::Native, 0, 0, 0, {}}));
            EXPECT_FALSE(registry.derived_from(*cut, {P::ConstructionRootKey::Kind::Native, 1, 0, 0, {}}));
            EXPECT_EQ(cut->volume().value, .5L);
        }
        cut.reset();
        EXPECT_TRUE(cutter_body.expired());
    }
}
} // namespace
