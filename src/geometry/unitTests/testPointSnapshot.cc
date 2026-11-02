/** @file testPointSnapshot.cc
 * @brief Independent snapshot authority, publication, lifetime and numerical contracts.
 */
#include "geometry/PLIC/ConstructionRegistry.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
namespace P = SimpleFluid::PLIC;
using SimpleFluid::ArithmeticMode;
using Access = P::DynamicGeometryAccess;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr auto bounded = ArithmeticMode::BoundedCertification;
constexpr std::size_t byte_limit = 64 * 1024 * 1024;
template<ArithmeticMode Mode> using Geometry = P::ConvexCellGeometry<Mode>;
template<ArithmeticMode Mode> using Builder = P::PointSnapshotBuilder<Mode>;

// Public callers cannot turn an arbitrary context/index pair or an authority
// key into an owning provenance token or admitted point incidence.
static_assert(!std::is_constructible_v<P::PointProvenanceRef,
    std::shared_ptr<const P::PointSnapshotContextBase>, P::DynamicStoredPointRef>);
static_assert(!std::is_constructible_v<P::PointProvenanceView,
    const P::PointSnapshotContextBase*, P::DynamicStoredPointRef>);
static_assert(!std::is_constructible_v<P::StoredPointRef<ordinary>,
    std::uint32_t, std::uint32_t, P::PointProvenanceRole>);
static_assert(!std::is_constructible_v<P::PrivateAuthorityRef<ordinary>, std::uint32_t>);
static_assert(!std::is_convertible_v<P::PrivateAuthorityRef<ordinary>, P::PrivateAuthorityRef<bounded>>);
static_assert(!std::is_constructible_v<P::AdmittedPointIncidence<ordinary>, P::PointAuthorityRange>);
static_assert(sizeof(P::StoredPointRef<ordinary>) <= 3 * sizeof(std::uint32_t));
static_assert(!std::is_copy_constructible_v<Builder<ordinary>>);
static_assert(!std::is_copy_assignable_v<Builder<ordinary>>);

struct Admission
{
    std::size_t used = 0, limit = byte_limit;
    std::optional<std::size_t> requests_left;
    static void charge(void* state, std::size_t count, std::size_t width)
    {
        auto& budget = *static_cast<Admission*>(state);
        if (budget.used > budget.limit || (width && count > (budget.limit - budget.used) / width))
            throw P::GeometryError(P::GeometryErrorCode::ResourceLimit, "Snapshot test admission exhausted.");
        if (count && width && budget.requests_left)
        {
            if (!*budget.requests_left)
                throw P::GeometryError(P::GeometryErrorCode::ResourceLimit, "Injected snapshot growth rejection.");
            --*budget.requests_left;
        }
        budget.used += count * width;
    }
    P::PointSnapshotAllocation allocation() { return {this, charge}; }
};

P::DynamicPoint point(P::PointProvenanceView view)
{
    P::DynamicPoint result(view.value(), view.absolute_error(), view.arithmetic_mode());
    Access::attach_identity(result, view.owning());
    return result;
}

template<ArithmeticMode Mode> void context_binding_contract()
{
    Admission admission;
    Builder<Mode> first(admission.allocation()), second(admission.allocation());
    const auto a = first.native_source({1, 2, 3}, {}, {});
    const auto b = second.native_source({7, 11, 13}, {}, {});
    EXPECT_EQ(a.local_ref().index, b.local_ref().index);
    EXPECT_NE(a.context_identity(), b.context_identity());
    EXPECT_THROW((void) a.owning(), P::GeometryError);
    const auto first_context = first.freeze(), second_context = second.freeze();
    ASSERT_TRUE(first_context->frozen);
    ASSERT_TRUE(second_context->frozen);
    const auto a_owner = a.owning(), b_owner = b.owning();
    EXPECT_NE(a_owner, b_owner);

    Builder<Mode> child(admission.allocation());
    const auto imported_a = child.import(a_owner), imported_b = child.import(b_owner);
    EXPECT_EQ(child.import(a_owner), imported_a);
    EXPECT_NE(imported_a.dependency_slot(), imported_b.dependency_slot());
    const auto child_context = child.freeze();
    ASSERT_EQ(child_context->dependencies.size(), 2u);
    EXPECT_EQ(P::resolve_point(child_context, imported_a).value(), (std::array<long double, 3>{1, 2, 3}));
    EXPECT_EQ(P::resolve_point(child_context, imported_b).value(), (std::array<long double, 3>{7, 11, 13}));
    EXPECT_EQ(P::bind_point(child_context, imported_a), a_owner);
    EXPECT_EQ(P::bind_point(child_context, imported_b), b_owner);

    auto moved = a_owner;
    const auto retained = std::move(moved);
    EXPECT_FALSE(moved);
    EXPECT_EQ(moved, P::PointProvenanceRef{});
    EXPECT_EQ(retained, a_owner);
    constexpr auto other = Mode == ordinary ? bounded : ordinary;
    EXPECT_THROW((void) P::TypedPointProvenanceRef<other>(a_owner), P::GeometryError);
    Builder<other> foreign(admission.allocation());
    EXPECT_THROW((void) foreign.import(a_owner), P::GeometryError);
    EXPECT_TRUE(foreign.context()->dependencies.empty());
    EXPECT_THROW((void) foreign.freeze(), P::GeometryError);
}

TEST(PointSnapshot, EqualLocalIndicesRequireTheirBoundContextAndArithmeticMode)
{
    context_binding_contract<ordinary>();
    context_binding_contract<bounded>();
}

template<ArithmeticMode Mode> void role_contract()
{
    Admission admission;
    Builder<Mode> builder(admission.allocation());
    const auto left = builder.native_source({0, 0, 0}, {}, {});
    const auto right = builder.native_source({2, 0, 0}, {}, {});
    const auto coincident_native = builder.native_source({1, 0, 0}, {}, {});
    const std::array<std::array<P::PointProvenanceView, 2>, 1> edges{std::array{left, right}};
    const auto generated = builder.generated({1, 0, 0}, {}, edges, {});
    const auto exported = builder.export_alias({1, 0, 0}, {}, generated, {});
    const auto numerical = builder.numerical({1, 0, 0}, {});
    const auto sealed = builder.sealed({1, 0, 0}, {}, {});
    const auto context = builder.freeze();
    EXPECT_EQ(generated.local_ref().index, exported.local_ref().index);
    EXPECT_EQ(exported.local_ref().index, numerical.local_ref().index);
    EXPECT_NE(generated.owning(), exported.owning());
    EXPECT_NE(generated.owning(), numerical.owning());
    EXPECT_EQ(Access::borrowed_original_source(exported), generated);
    EXPECT_TRUE(Access::same_represented_source(exported, generated));
    EXPECT_FALSE(Access::same_source(point(exported), point(generated)));
    EXPECT_FALSE(Access::same_represented_source(generated, coincident_native));
    EXPECT_FALSE(Access::same_represented_source(sealed, coincident_native));
    EXPECT_FALSE(sealed.represented_native_source());
    EXPECT_FALSE(sealed.constructed());
    EXPECT_TRUE(sealed.source_edges().empty());
    EXPECT_TRUE(generated.constructed());
    EXPECT_FALSE(Access::same_represented_source(numerical, coincident_native));
    EXPECT_FALSE(Access::borrowed_original_source(numerical));
    ASSERT_EQ(generated.source_edges().size(), 1u);
    EXPECT_EQ(generated.source_edges()[0][0], left);
    EXPECT_EQ(generated.source_edges()[0][1], right);
    EXPECT_EQ(context->native_refs.size(), 3u);
    EXPECT_EQ(context->generated_recipes.size(), 2u);
    EXPECT_EQ(context->export_aliases.size(), 1u);
    EXPECT_EQ(context->numerical_points.size(), 1u);
    EXPECT_TRUE(context->witnesses.empty());

    Builder<Mode> independent(admission.allocation());
    const auto rebuilt = independent.native_source({0, 0, 0}, {}, {});
    const auto independent_context = independent.freeze();
    EXPECT_TRUE(independent_context->dependencies.empty());
    EXPECT_FALSE(Access::same_source(point(left), point(rebuilt)));
    EXPECT_TRUE(Access::same_represented_source(left, rebuilt));
}

TEST(PointSnapshot, NativeGeneratedExportAndNumericalRolesRetainDifferentProofs)
{
    role_contract<ordinary>();
    role_contract<bounded>();
}

template<ArithmeticMode Mode> void local_alias_work_contract()
{
    Admission admission;
    Builder<Mode> builder(admission.allocation());
    const auto original = builder.native_source({1, 2, 4}, {}, {});
    auto head = original;
    constexpr std::size_t edges = 65;
    for (std::size_t i = 0; i < edges; ++i)
        head = builder.export_alias(head.value(), head.absolute_error(), head, {});
    const auto context = builder.freeze();
    const auto owner = head.owning();
    const auto owners = context->weak_from_this().use_count();
    P::GeometryLimits limits;
    // This is metadata traversal even when the snapshot's value policy is Bounded.
    limits.arithmetic_mode = ordinary;
    limits.maximum_pair_tests = edges;
    {
        P::details::Operation operation(limits);
        EXPECT_EQ(Access::borrowed_original_source(owner), original);
        EXPECT_EQ(operation.work.pairs, edges);
        EXPECT_EQ(operation.work.operations, 0u);
        EXPECT_EQ(operation.work.bytes, 0u);
        EXPECT_EQ(context->weak_from_this().use_count(), owners);
    }
    limits.maximum_pair_tests = edges - 1;
    {
        P::details::Operation operation(limits);
        EXPECT_THROW(Access::borrowed_original_source(owner), P::GeometryError);
        EXPECT_EQ(operation.work.pairs, edges);
        EXPECT_EQ(operation.work.operations, 0u);
        EXPECT_EQ(context->weak_from_this().use_count(), owners);
    }
}

TEST(PointSnapshot, LocallyBatchedAliasEdgesKeepTheirPairCapWithoutOwningReferenceChurn)
{
    local_alias_work_contract<ordinary>();
    local_alias_work_contract<bounded>();
}

TEST(PointSnapshot, RegisteringAnAuthenticPlaneDoesNotIssuePointIncidence)
{
    const std::array<P::GeometryPoint, 3> face_points{{{0, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const auto face = P::PlanarFace<ordinary>::make(face_points);
    const auto authority = P::details::original_support_authority(P::planar_support<ordinary>(*face));
    Admission admission;
    Builder<ordinary> builder(admission.allocation());
    (void) builder.register_authority(authority);
    const std::array views{builder.native_source({0, 0, 0}, {}, {}),
        builder.native_source({0, 1, 0}, {}, {}), builder.native_source({0, 0, 1}, {}, {}),
        builder.native_source({1, 0, 0}, {}, {})};
    const auto context = builder.freeze();
    ASSERT_EQ(context->authority_table.size(), 1u);
    EXPECT_TRUE(context->incidence.empty());
    std::array<P::Point<ordinary>, 4> tetrahedron;
    for (std::size_t i = 0; i < views.size(); ++i)
    {
        EXPECT_TRUE(views[i].supports().empty());
        EXPECT_FALSE(views[i].proves_incidence(authority));
        tetrahedron[i] = P::Point<ordinary>(point(views[i]));
    }
    // These four analytic points have determinant magnitude one. A registered
    // plane must not turn the off-plane fourth point into a coplanar proof.
    EXPECT_EQ(std::abs(P::orientation<ordinary>(tetrahedron).value), 1.L);
}

TEST(PointSnapshot, AdmittedIncidenceViewsSurviveDraftTableGrowthWithoutAcquiringRegisteredKeys)
{
    const std::array<P::GeometryPoint, 3> face_points{{{0, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const auto face = P::PlanarFace<ordinary>::make(face_points);
    const std::array support{P::details::original_support_authority(P::planar_support<ordinary>(*face))};
    Admission admission;
    Builder<ordinary> builder(admission.allocation());
    // The fixture explicitly admits the origin's membership in its x=0 face.
    const auto origin = builder.native_source({0, 0, 0}, {}, Access::admit_incidence<ordinary>(support));
    const auto original_range = origin.supports();
    auto original_iterator = original_range.begin();
    for (std::uint64_t i = 0; i < 65; ++i)
    {
        (void) builder.register_authority(P::PlaneAuthority::opaque({37, i}));
        const auto alias = builder.native_source({0, 0, 0}, {},
            Access::admit_incidence<ordinary>(original_range));
        ASSERT_EQ(alias.supports().size(), 1u);
        EXPECT_EQ(alias.supports().front(), support.front());
        EXPECT_EQ(*original_iterator, support.front());
    }
    const auto context = builder.freeze();
    EXPECT_GT(context->authority_table.size(), original_range.size());
    EXPECT_EQ(origin.supports(), (std::vector<P::PlaneAuthority>{support.front()}));
}

template<ArithmeticMode Mode> void immutable_facts_contract()
{
    Admission admission;
    Builder<Mode> source(admission.allocation());
    const auto original = source.native_source({2, 4, 8}, {.125L, 0, 0}, {});
    Builder<Mode> premature(admission.allocation());
    EXPECT_THROW((void) premature.native_source(original.value(), original.absolute_error(), {}, original),
        P::GeometryError);
    EXPECT_THROW((void) premature.freeze(), P::GeometryError);
    // Pass actual references into the draft's own arrays while forcing repeated
    // growth; numerical inputs must be staged before those arrays reallocate.
    for (std::size_t i = 0; i < 65; ++i)
    {
        const auto appended = source.native_source(original.value(), original.absolute_error(), {});
        EXPECT_EQ(appended.value(), (std::array<long double, 3>{2, 4, 8}));
        EXPECT_EQ(appended.absolute_error(), (std::array<long double, 3>{.125L, 0, 0}));
    }
    const auto frozen = source.freeze();
    const auto count = frozen->native_fact_blocks.front()->points.size();
    EXPECT_THROW((void) source.native_source({9, 9, 9}, {}, {}), P::GeometryError);
    EXPECT_EQ(frozen->native_fact_blocks.front()->points.size(), count);
    Builder<Mode> imported(admission.allocation());
    const auto imported_point = imported.native_source(original.value(), original.absolute_error(), {}, original);
    const auto imported_context = imported.freeze();
    ASSERT_EQ(imported_context->native_fact_blocks.size(), 1u);
    EXPECT_TRUE(imported_context->dependencies.empty());
    EXPECT_EQ(imported_context->native_fact_blocks.front(), frozen->native_fact_blocks.front());
    EXPECT_EQ(imported_point.value(), original.value());
    EXPECT_EQ(imported_point.absolute_error(), original.absolute_error());
    EXPECT_TRUE(imported_context->generated_recipes.empty());
    EXPECT_TRUE(imported_context->edges.empty());
}

TEST(PointSnapshot, OnlyFrozenNativeFactsMayBeSharedAndDraftGrowthPreservesAliasedInputs)
{
    immutable_facts_contract<ordinary>();
    immutable_facts_contract<bounded>();
}

TEST(PointSnapshot, PartialGrowthFailurePoisonsTheBuilderWithoutPublishingOrChangingItsParent)
{
    Admission admission;
    Builder<ordinary> parent(admission.allocation());
    const auto original = parent.native_source({3, 5, 7}, {}, {});
    const auto parent_context = parent.freeze();
    Builder<ordinary> candidate(admission.allocation());
    (void) candidate.import(original);
    const auto before = admission.used;
    // Admit one new buffer and reject the next without depending on the order
    // in which a recipe's independent arrays are populated.
    admission.requests_left = 1;
    EXPECT_THROW((void) candidate.generated({1, 0, 0}, {}, {}, {}), P::GeometryError);
    EXPECT_GT(admission.used, before);
    EXPECT_FALSE(candidate.context()->frozen);
    admission.requests_left.reset();
    EXPECT_THROW((void) candidate.freeze(), P::GeometryError);
    EXPECT_THROW((void) candidate.numerical({1, 0, 0}, {}), P::GeometryError);
    EXPECT_EQ(original.value(), (std::array<long double, 3>{3, 5, 7}));
    EXPECT_EQ(parent_context->native_refs.size(), 1u);
    Builder<ordinary> retry(admission.allocation());
    const auto replacement = retry.generated({1, 0, 0}, {}, {}, {});
    const auto accepted = retry.freeze();
    EXPECT_TRUE(replacement.owning());
    EXPECT_TRUE(accepted->frozen);
}

TEST(PointSnapshot, SharedParentContextsAndNativeFactsAreAdmittedOnceAtTheExactBoundary)
{
    Admission admission;
    Builder<ordinary> parent(admission.allocation());
    const auto a = parent.native_source({0, 0, 0}, {}, {});
    const auto b = parent.native_source({1, 0, 0}, {}, {});
    const auto parent_context = parent.freeze();
    Builder<ordinary> child(admission.allocation());
    (void) child.export_alias(a.value(), {}, a, {});
    (void) child.export_alias(b.value(), {}, b, {});
    const auto child_context = child.freeze();
    ASSERT_EQ(child_context->dependencies.size(), 1u);
    ASSERT_TRUE(child_context->native_fact_blocks.empty());
    std::size_t required = 0;
    P::GeometryAllocationLedger measured;
    measured.retain_snapshot<ordinary>(child_context, required, byte_limit);
    // The graph has exactly two context owners and one native fact block.
    EXPECT_EQ(measured.owner_count(), 3u);
    const auto original_bytes = required;
    measured.retain_snapshot<ordinary>(parent_context, required, byte_limit);
    EXPECT_EQ(required, original_bytes);
    P::GeometryAllocationLedger exact;
    std::size_t used = 0;
    EXPECT_NO_THROW(exact.retain_snapshot<ordinary>(child_context, used, required));
    EXPECT_EQ(used, required);
    P::GeometryAllocationLedger short_ledger;
    std::size_t short_used = 0;
    EXPECT_THROW(short_ledger.retain_snapshot<ordinary>(child_context, short_used, required - 1), P::GeometryError);
    EXPECT_THROW(short_ledger.retain_snapshot<ordinary>(child_context, short_used, byte_limit), P::GeometryError);
}

TEST(PointSnapshot, MutatingNumericalFieldsInvalidatesProofButDoesNotHideTheHeldSnapshot)
{
    Admission admission;
    Builder<ordinary> builder(admission.allocation());
    const auto view = builder.native_source({1, 2, 3}, {}, {});
    const auto context = builder.freeze();
    const auto valid = point(view);
    auto mutated = valid;
    mutated.value[0] = 9;
    EXPECT_FALSE(Access::identity(mutated));
    EXPECT_TRUE(mutated.supports().empty());
    P::GeometryAllocationLedger original_ledger, mutated_ledger;
    std::size_t original_bytes = 0, mutated_bytes = 0;
    original_ledger.retain_point_provenance(valid, original_bytes, byte_limit);
    mutated_ledger.retain_point_provenance(mutated, mutated_bytes, byte_limit);
    EXPECT_GT(original_bytes, 0u);
    EXPECT_EQ(mutated_bytes, original_bytes);
    EXPECT_EQ(mutated_ledger.owner_count(), original_ledger.owner_count());
    mutated_ledger.retain_snapshot<ordinary>(context, mutated_bytes, byte_limit);
    EXPECT_EQ(mutated_bytes, original_bytes);
}

TEST(PointSnapshot, DyadicClipAndConstructionReplayKeepAnalyticGlobalMoments)
{
    const auto native = Geometry<ordinary>::box({2, -4, .125}, {2.5, -3.875, 1.125});
    const auto imported = Geometry<ordinary>::from_geometry(*native);
    const P::GeometryHalfspace cut{{1, 0, 0}, {2, -4, .125}, .25};
    const auto piece = imported->clip(cut, false).inside;
    ASSERT_TRUE(piece);
    const auto moments = piece->moments({0, 0, 0});
    EXPECT_EQ(moments.volume.value, 1.L / 32);
    const std::array<long double, 3> expected{17.L / 256, -63.L / 512, 5.L / 256};
    for (std::size_t axis = 0; axis < 3; ++axis)
        EXPECT_EQ(moments.first_moment[axis].value, expected[axis]);
    P::ConstructionRegistry<ordinary> registry({{2, 3, 5, 7}});
    registry.register_native(11, 0, imported);
    registry.register_plane(13, cut);
    const auto packet = piece->encode(registry);
    const auto replay = Geometry<ordinary>::decode(packet, registry);
    EXPECT_EQ(replay->encode(registry), packet);
    EXPECT_EQ(replay->encode_certificate(), piece->encode_certificate());
    const auto replay_moments = replay->moments({0, 0, 0});
    EXPECT_EQ(replay_moments.volume.value, 1.L / 32);
    for (std::size_t axis = 0; axis < 3; ++axis)
        EXPECT_EQ(replay_moments.first_moment[axis].value, expected[axis]);
}

template<ArithmeticMode Mode> void export_reuse_contract()
{
    const auto geometry = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    std::optional<P::Point<Mode>> survivor;
    std::weak_ptr<const P::PointSnapshotContextBase> snapshot;
    std::weak_ptr<const P::DynamicPointExportPayload> dynamic;
    const void* context_identity = nullptr;
    P::DynamicStoredPointRef token;
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        const auto publication = geometry->export_points(ledger, used, byte_limit);
        ASSERT_EQ(publication.points().size(), 8u);
        survivor = publication.points()[3];
        const auto identity = Access::identity(survivor->dynamic());
        ASSERT_TRUE(identity);
        snapshot = identity.context_pin();
        context_identity = identity.context_identity();
        token = identity.local_ref();
        EXPECT_EQ(token.role, P::PointProvenanceRole::Export);
        for (const auto& vertex : publication.points())
            EXPECT_EQ(Access::identity(vertex.dynamic()).context_identity(), context_identity);
        dynamic = Access::implementation(*geometry->dynamic_geometry())->point_export_cache->dynamic_root;
    }
    EXPECT_TRUE(dynamic.expired());
    ASSERT_FALSE(snapshot.expired());
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        const auto regenerated = geometry->export_points(ledger, used, byte_limit);
        const auto identity = Access::identity(regenerated.points()[3].dynamic());
        EXPECT_EQ(identity.context_identity(), context_identity);
        EXPECT_EQ(identity.local_ref(), token);
        EXPECT_TRUE(Access::same_source(regenerated.points()[3].dynamic(), survivor->dynamic()));
        EXPECT_EQ(regenerated.points()[3].value, survivor->value);
        if constexpr (Mode == bounded)
            EXPECT_EQ(regenerated.points()[3].absolute_error, survivor->absolute_error);
    }
    survivor.reset();
    EXPECT_TRUE(snapshot.expired());
}

TEST(PointSnapshot, OrdinaryCopiedPointPinsItsBatchAndReexportReusesTheExactToken)
{
    export_reuse_contract<ordinary>();
}

TEST(PointSnapshot, BoundedCopiedPointPinsItsBatchAndReexportReusesTheExactToken)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded numerical geometry requires its supported FP environment.";
    export_reuse_contract<bounded>();
}

TEST(PointSnapshot, CopiedGeneratedPointKeepsSourceFactsWithoutRetainingAnyGeometryPayload)
{
    std::optional<P::Point<ordinary>> survivor;
    std::vector<std::weak_ptr<const P::DynamicConvexGeometry::Impl>> geometries;
    std::weak_ptr<const P::PointSnapshotContextBase> snapshot;
    {
        const auto native = Geometry<ordinary>::box({2, -4, .125}, {2.5, -3.875, 1.125});
        const auto imported = Geometry<ordinary>::from_geometry(*native);
        const auto piece = imported->clip({{1, 0, 0}, {2, -4, .125}, .25}, false).inside;
        ASSERT_TRUE(piece);
        for (const auto& geometry : {native, imported, piece})
            geometries.emplace_back(Access::implementation(*geometry->dynamic_geometry()));
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        const auto publication = piece->export_points(ledger, used, byte_limit);
        const auto found = std::find_if(publication.points().begin(), publication.points().end(),
            [](const auto& vertex) { return vertex.value == std::array<long double, 3>{2.25L, -4, .125L}; });
        ASSERT_NE(found, publication.points().end());
        survivor = *found;
        snapshot = Access::identity(survivor->dynamic()).context_pin();
    }
    for (const auto& geometry : geometries)
        EXPECT_TRUE(geometry.expired());
    ASSERT_FALSE(snapshot.expired());
    ASSERT_TRUE(survivor);
    EXPECT_EQ(survivor->value, (std::array<long double, 3>{2.25L, -4, .125L}));
    const auto token = Access::identity(survivor->dynamic());
    const auto source = Access::borrowed_original_source(token);
    ASSERT_TRUE(source);
    bool original_edge = false;
    for (const auto edge : source.source_edges())
    {
        const auto& first = edge[0].value();
        const auto& second = edge[1].value();
        const std::array<long double, 3> left{2, -4, .125L}, right{2.5L, -4, .125L};
        original_edge |= (first == left && second == right) || (first == right && second == left);
    }
    EXPECT_TRUE(original_edge);
}

TEST(PointSnapshot, BoundedNonbinaryCutPreservesIndependentVolumeAndMomentEnclosures)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded numerical geometry requires its supported FP environment.";
    const auto cube = Geometry<bounded>::box({0, 0, 0}, {1, 1, 1});
    const auto piece = cube->clip({{3, 0, 0}, {0, 0, 0}, 1}, false).inside;
    ASSERT_TRUE(piece);
    const auto moments = piece->moments({0, 0, 0});
    EXPECT_LE(moments.volume.lower() * 3, 1.L);
    EXPECT_GE(moments.volume.upper() * 3, 1.L);
    const std::array<long double, 3> denominators{18, 6, 6};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        EXPECT_LE(moments.first_moment[axis].lower() * denominators[axis], 1.L);
        EXPECT_GE(moments.first_moment[axis].upper() * denominators[axis], 1.L);
    }
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto publication = piece->export_points(ledger, used, byte_limit);
    EXPECT_TRUE(std::any_of(publication.points().begin(), publication.points().end(),
        [](const auto& vertex) { return vertex.value[0] > 0 && vertex.absolute_error[0] > 0; }));
}

P::PointIdentity<ordinary> native_identity(const Geometry<ordinary>& geometry, P::GeometryPoint coordinate)
{
    for (std::size_t face = 0; face < geometry.num_faces(); ++face)
        for (const auto& identity : geometry.face_geometry(face)->vertex_identities())
            if (P::point_display<ordinary>(identity) == coordinate)
                return identity;
    throw std::logic_error("Snapshot test fixture omitted its requested native vertex.");
}

template<ArithmeticMode Mode> void native_batch_contract()
{
    std::vector<P::Point<Mode>> published;
    std::vector<std::weak_ptr<const P::DynamicConvexGeometry::Impl>> geometries;
    std::weak_ptr<const P::PointSnapshotContextBase> snapshot;
    auto ledger = std::make_unique<P::GeometryAllocationLedger>();
    std::size_t retained_bytes = 0;
    {
        const auto left = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
        const auto right = Geometry<ordinary>::box({1, 0, 0}, {2, 1, 1});
        const std::array sources{left, right};
        for (const auto& source : sources)
            geometries.emplace_back(Access::implementation(*source->dynamic_geometry()));
        const auto shared = native_identity(*left, {1, 0, 0});
        const auto far = native_identity(*right, {2, 1, 1});
        // Reversing the source fan preserves membership. Request order and
        // separately issued equal-coordinate tokens must remain observable.
        const std::array<std::size_t, 5> indices{1, 0, 1, 0, 1};
        const std::array<P::NativeSourcePointRequest, 3> requests{
            P::NativeSourcePointRequest{shared, 0, 2}, P::NativeSourcePointRequest{far, 2, 1},
            P::NativeSourcePointRequest{shared, 3, 2}};
        const std::array certificates{left->encode_certificate(), right->encode_certificate()};
        published = P::source_points<Mode>(sources, indices, requests);
        ASSERT_EQ(published.size(), 3u);
        EXPECT_EQ(published[0].value, (std::array<long double, 3>{1, 0, 0}));
        EXPECT_EQ(published[1].value, (std::array<long double, 3>{2, 1, 1}));
        EXPECT_EQ(published[2].value, published[0].value);
        const auto first = Access::identity(published[0].dynamic());
        const auto repeated = Access::identity(published[2].dynamic());
        ASSERT_TRUE(first);
        ASSERT_TRUE(repeated);
        EXPECT_EQ(first.context_identity(), repeated.context_identity());
        EXPECT_NE(first.local_ref(), repeated.local_ref());
        EXPECT_FALSE(Access::same_source(published[0].dynamic(), published[2].dynamic()));
        EXPECT_TRUE(Access::same_represented_source(first, repeated));
        snapshot = first.context_pin();
        for (const auto& value : published)
        {
            EXPECT_EQ(Access::identity(value.dynamic()).context_identity(), first.context_identity());
            EXPECT_EQ(value.supports().size(), Mode == ordinary ? 3u : 0u);
            if constexpr (Mode == bounded)
                EXPECT_EQ(value.absolute_error, (std::array<long double, 3>{}));
            ledger->retain_point_provenance(value, retained_bytes, byte_limit);
        }
        EXPECT_EQ(published[0].supports(), published[2].supports());
        const auto before_duplicate = retained_bytes;
        ledger->retain_point_provenance(published[0], retained_bytes, byte_limit);
        EXPECT_EQ(retained_bytes, before_duplicate);

        std::vector<P::Point<Mode>> candidate{P::Point<Mode>({7, 11, 13})};
        auto invalid = requests;
        invalid.back().first_source = std::numeric_limits<std::size_t>::max();
        EXPECT_THROW(candidate = P::source_points<Mode>(sources, indices, invalid), std::invalid_argument);
        invalid = requests;
        invalid.back().source_count = 0;
        EXPECT_THROW(candidate = P::source_points<Mode>(sources, indices, invalid), std::invalid_argument);
        auto bad_indices = indices;
        bad_indices.back() = sources.size();
        EXPECT_THROW(candidate = P::source_points<Mode>(sources, bad_indices, requests), std::invalid_argument);
        invalid = requests;
        invalid[0] = {far, 1, 1}; // The far corner is absent from the left-only fan.
        EXPECT_THROW(candidate = P::source_points<Mode>(sources, indices, invalid), std::invalid_argument);
        const P::GeometryPacketContext numerical_context{{31, 37, 41, 43}};
        const std::array numerical_registry{left};
        const auto numerical =
            Geometry<ordinary>::decode(left->encode(numerical_context), numerical_context, numerical_registry);
        const std::array numerical_sources{numerical};
        const std::array<std::size_t, 1> numerical_indices{0};
        const std::array<P::NativeSourcePointRequest, 1> numerical_request{
            P::NativeSourcePointRequest{shared, 0, 1}};
        EXPECT_THROW(candidate = P::source_points<Mode>(numerical_sources, numerical_indices, numerical_request),
            P::GeometryError);
        P::GeometryLimits constrained;
        constrained.arithmetic_mode = Mode;
        constrained.byte_budget = 1;
        EXPECT_THROW(candidate = P::source_points<Mode>(sources, indices, requests, constrained), P::GeometryError);
        {
            P::GeometryLimits enclosing_limits;
            enclosing_limits.arithmetic_mode = ordinary;
            P::details::Operation enclosing(enclosing_limits);
            const auto before = enclosing.work.bytes;
            EXPECT_THROW(candidate = P::source_points<Mode>(sources, indices, requests), P::GeometryError);
            EXPECT_EQ(enclosing.work.bytes, before);
        }
        ASSERT_EQ(candidate.size(), 1u);
        EXPECT_EQ(candidate.front().value, (std::array<long double, 3>{7, 11, 13}));
        EXPECT_EQ(left->encode_certificate(), certificates[0]);
        EXPECT_EQ(right->encode_certificate(), certificates[1]);
        EXPECT_EQ(indices, (std::array<std::size_t, 5>{1, 0, 1, 0, 1}));
        EXPECT_EQ(requests[0].identity, shared);
        EXPECT_EQ(requests[1].identity, far);
        EXPECT_EQ(requests[2].identity, shared);
        EXPECT_TRUE(P::source_points<Mode>(sources, indices, {}).empty());
    }
    for (const auto& geometry : geometries)
        EXPECT_TRUE(geometry.expired());
    ASSERT_FALSE(snapshot.expired());
    ASSERT_EQ(published.size(), 3u);
    std::optional<P::Point<Mode>> survivor = published[2];
    published.clear();
    EXPECT_EQ(survivor->value, (std::array<long double, 3>{1, 0, 0}));
    survivor.reset();
    EXPECT_FALSE(snapshot.expired()); // The explicit accounting transaction still pins it.
    ledger.reset();
    EXPECT_TRUE(snapshot.expired());
}

TEST(PointSnapshot, OrdinaryNativeBatchPreservesFanOrderAuthorityTokensAndIndependentLifetime)
{
    native_batch_contract<ordinary>();
}

TEST(PointSnapshot, BoundedNativeBatchPreservesFanOrderWithoutInheritingOrdinaryIncidence)
{
    if (!P::certification_environment())
        GTEST_SKIP() << "Bounded numerical geometry requires its supported FP environment.";
    native_batch_contract<bounded>();
}

template<ArithmeticMode Mode> void compact_facade_contract()
{
    std::optional<P::Point<Mode>> survivor;
    std::weak_ptr<const P::PointSnapshotContextBase> context_observer;
    std::weak_ptr<const P::PointSnapshotPayload<Mode>> payload_observer;
    P::DynamicStoredPointRef surviving_token;
    {
        Admission admission;
        Builder<Mode> builder(admission.allocation());
        const auto a = builder.native_source({1, 2, 3}, {}, {});
        const auto b = builder.native_source({7, 11, 13}, {}, {});
        const auto numeric = builder.numerical({17, 19, 23}, {});
        const auto context = builder.freeze();
        context_observer = context;
        const std::array<P::Point<Mode>, 4> input{
            P::Point<Mode>(point(b)), P::Point<Mode>(point(a)), P::Point<Mode>(point(b)),
            P::Point<Mode>(point(numeric))};
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain_snapshot<Mode>(context, used, byte_limit);
        const auto before_payload = ledger.payload_bytes(), before_owners = ledger.owner_count();
        auto snapshot = P::snapshot_points<Mode>(input, ledger, used, byte_limit);
        const auto& payload = Access::snapshot_payload(snapshot);
        ASSERT_TRUE(payload);
        payload_observer = payload;
        const auto added = sizeof(P::PointSnapshotPayload<Mode>) + 2 * sizeof(void*) +
                           payload->refs.capacity() * sizeof(P::StoredPointRef<Mode>);
        EXPECT_EQ(ledger.payload_bytes() - before_payload, added);
        EXPECT_EQ(ledger.owner_count(), before_owners + 1);
        EXPECT_EQ(snapshot.size(), input.size());
        EXPECT_EQ(snapshot.context_identity(), context.get());
        EXPECT_EQ(payload->refs[0], payload->refs[2]);
        EXPECT_NE(payload->refs[0], payload->refs[1]);
        auto copy = snapshot;
        const auto admitted = used;
        ledger.retain(copy, used, byte_limit);
        EXPECT_EQ(used, admitted);
        {
            P::GeometryLimits limits;
            limits.arithmetic_mode = ordinary;
            limits.byte_budget = 1;
            P::details::Operation observation(limits);
            for (std::size_t i = 0; i < input.size(); ++i)
            {
                const auto materialized = copy.point(i);
                EXPECT_EQ(materialized.value, input[i].value);
                EXPECT_EQ(Access::identity(materialized.dynamic()), Access::identity(input[i].dynamic()));
                if constexpr (Mode == bounded)
                    EXPECT_EQ(materialized.absolute_error, input[i].absolute_error);
            }
            EXPECT_EQ(observation.work.bytes, 0u);
            EXPECT_EQ(observation.work.operations, 0u);
        }
        EXPECT_TRUE(Access::identity(snapshot.point(3).dynamic()).view().numerical_only());
        EXPECT_TRUE(snapshot.point(3).supports().empty());
        auto changed = snapshot.point(1);
        changed.value[0] = 99;
        EXPECT_FALSE(Access::identity(changed.dynamic()));
        EXPECT_EQ(snapshot.point(1).value, (std::array<long double, 3>{1, 2, 3}));
        EXPECT_THROW((void) snapshot.point(input.size()), std::out_of_range);
        survivor = copy.point(0);
        surviving_token = Access::identity(survivor->dynamic()).local_ref();
        snapshot = {};
        copy = {};
        EXPECT_FALSE(payload_observer.expired()); // The ledger pins the admitted facade.
    }
    EXPECT_TRUE(payload_observer.expired());
    ASSERT_FALSE(context_observer.expired());
    ASSERT_TRUE(survivor);
    EXPECT_EQ(survivor->value, (std::array<long double, 3>{7, 11, 13}));
    EXPECT_EQ(Access::identity(survivor->dynamic()).local_ref(), surviving_token);
    survivor.reset();
    EXPECT_TRUE(context_observer.expired());
}

TEST(PointSnapshot, CompactFacadeOwnsOnlyOrderedSlotsAndMaterializesIndependentOwningPoints)
{
    compact_facade_contract<ordinary>();
    compact_facade_contract<bounded>();
}

template<ArithmeticMode Mode> void compact_facade_rejection_contract()
{
    Admission admission;
    Builder<Mode> first(admission.allocation()), second(admission.allocation());
    const auto a = first.native_source({1, 2, 3}, {}, {});
    const auto b = second.native_source({1, 2, 3}, {}, {});
    const auto first_context = first.freeze(), second_context = second.freeze();
    const P::Point<Mode> valid(point(a)), independent(point(b));
    auto mutated = valid;
    mutated.value[1] = 99;
    const P::Point<Mode> unsealed({1, 2, 3});
    const std::array<P::Point<Mode>, 1> retry{valid};
    for (const auto& rejected : {independent, mutated, unsealed})
    {
        const std::array<P::Point<Mode>, 2> input{valid, rejected};
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        P::PointSnapshot<Mode> output;
        EXPECT_THROW(output = P::snapshot_points<Mode>(input, ledger, used, byte_limit), P::GeometryError);
        EXPECT_EQ(output.size(), 0u);
        EXPECT_EQ(output.context_identity(), nullptr);
        const auto rejected_bytes = used;
        EXPECT_THROW(output = P::snapshot_points<Mode>(retry, ledger, used, byte_limit), P::GeometryError);
        EXPECT_EQ(used, rejected_bytes);
        EXPECT_EQ(valid.value, (std::array<long double, 3>{1, 2, 3}));
        EXPECT_EQ(a.value(), valid.value);
        EXPECT_EQ(b.value(), valid.value);
    }
    EXPECT_NE(first_context.get(), second_context.get());
}

TEST(PointSnapshot, CompactFacadeRejectsMixedContextsAndInvalidSealsWithoutCloningOrPublishing)
{
    compact_facade_rejection_contract<ordinary>();
    compact_facade_rejection_contract<bounded>();
}

template<ArithmeticMode Mode> void compact_facade_admission_contract()
{
    Admission admission;
    Builder<Mode> builder(admission.allocation());
    const auto original = builder.native_source({2, 3, 5}, {}, {});
    const auto context = builder.freeze();
    const P::Point<Mode> owning(point(original));
    const std::array<P::Point<Mode>, 2> input{owning, owning};
    P::GeometryAllocationLedger measured;
    std::size_t required = 0;
    const auto baseline = P::snapshot_points<Mode>(input, measured, required, byte_limit);
    ASSERT_EQ(baseline.size(), 2u);
    EXPECT_EQ(baseline.context_identity(), context.get());
    P::GeometryAllocationLedger exact;
    std::size_t used = 0;
    EXPECT_NO_THROW((void) P::snapshot_points<Mode>(input, exact, used, required));
    EXPECT_EQ(used, required);
    P::GeometryAllocationLedger short_ledger;
    std::size_t short_used = 0;
    P::PointSnapshot<Mode> rejected;
    EXPECT_THROW(rejected = P::snapshot_points<Mode>(input, short_ledger, short_used, required - 1), P::GeometryError);
    EXPECT_EQ(rejected.size(), 0u);
    const auto charged = short_used;
    EXPECT_THROW(rejected = P::snapshot_points<Mode>(input, short_ledger, short_used, byte_limit), P::GeometryError);
    EXPECT_EQ(short_used, charged);
    P::GeometryAllocationLedger empty_ledger;
    std::size_t empty_used = 0;
    const auto empty = P::snapshot_points<Mode>({}, empty_ledger, empty_used, 0);
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ(empty.context_identity(), nullptr);
    EXPECT_EQ(empty_used, 0u);
    EXPECT_NO_THROW(empty_ledger.retain(empty, empty_used, 0));
    EXPECT_EQ(empty_used, 0u);
    EXPECT_THROW((void) empty.point(0), std::out_of_range);
}

TEST(PointSnapshot, CompactFacadeHasExactAdmissionPoisonedFailureAndZeroCostEmptyPublication)
{
    compact_facade_admission_contract<ordinary>();
    compact_facade_admission_contract<bounded>();
}
} // namespace
