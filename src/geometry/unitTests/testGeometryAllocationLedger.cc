/** @file testGeometryAllocationLedger.cc
 * @brief Unique immutable-payload admission, strong pins and cumulative cap checks.
 */
#include "geometry/PLIC/ConstructionRegistry.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include <cmath>
#include <gtest/gtest.h>
#include <optional>

namespace
{
using SimpleFluid::ArithmeticMode;
namespace P = SimpleFluid::PLIC;
template<ArithmeticMode Mode> using Geometry = P::ConvexCellGeometry<Mode>;
constexpr std::size_t byte_limit = 64 * 1024 * 1024;

template<ArithmeticMode Mode> void reuse_contract()
{
    const auto root = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(root, used, byte_limit);
    const auto original = used, owners = ledger.owner_count(), payload = ledger.payload_bytes();
    EXPECT_EQ(payload, root->storage_bytes());
    EXPECT_GT(ledger.bookkeeping_bytes(), 0u);
    EXPECT_EQ(used, payload + ledger.bookkeeping_bytes());
    ledger.retain(root, used, byte_limit);
    EXPECT_EQ(used, original);
    EXPECT_EQ(ledger.owner_count(), owners);
    const auto alias = Geometry<Mode>::from_dynamic(root->dynamic_geometry());
    ledger.retain(alias, used, byte_limit);
    EXPECT_EQ(ledger.payload_bytes(), payload + sizeof(Geometry<Mode>));
    EXPECT_EQ(ledger.owner_count(), owners + 1);
    const auto equal_values = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    const auto before_distinct = ledger.payload_bytes();
    ledger.retain(equal_values, used, byte_limit);
    EXPECT_GT(ledger.payload_bytes(), before_distinct);
}
TEST(GeometryAllocationLedgerTest, RepeatedAndAliasedInputsChargeOnlyNewOwners)
{
    reuse_contract<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        reuse_contract<ArithmeticMode::BoundedCertification>();
}

TEST(GeometryAllocationLedgerTest, ImportedOrdinaryRootsPreserveExactNativeBoxRepresentation)
{
    using Ordinary = Geometry<ArithmeticMode::Ordinary>;
    using Box = std::array<P::GeometryPoint, 2>;
    const std::array<Box, 3> boxes{Box{P::GeometryPoint{0, 0, 0}, P::GeometryPoint{1, 1, 1}},
        Box{P::GeometryPoint{-.125, 2.25, -8}, P::GeometryPoint{.375, 2.375, 8}},
        Box{P::GeometryPoint{.3, -.7, .01}, P::GeometryPoint{.315625, -.66875, .0725}}};
    for (const auto& bounds : boxes)
    {
        const auto native = Ordinary::box(bounds[0], bounds[1]);
        const auto imported = Ordinary::from_geometry(*native);
        EXPECT_NE(native->storage_identity(), imported->storage_identity());
        EXPECT_TRUE(native->same_native_representation(*imported));
        EXPECT_TRUE(imported->same_native_representation(*native));
        EXPECT_EQ(native->encode_certificate(), imported->encode_certificate());
        EXPECT_EQ(native->volume().value, imported->volume().value);
        EXPECT_EQ(native->centroid(), imported->centroid());
        const auto before = native->moments(), after = imported->moments();
        for (std::size_t axis = 0; axis < 3; ++axis)
            EXPECT_EQ(before.first_moment[axis].value, after.first_moment[axis].value);
        for (std::size_t face = 0; face < native->num_faces(); ++face)
            EXPECT_EQ(P::planar_support<ArithmeticMode::Ordinary>(*native->face_geometry(face)),
                P::planar_support<ArithmeticMode::Ordinary>(*imported->face_geometry(face)));
    }
}

TEST(GeometryAllocationLedgerTest, PromotedImportedRootReleasesNativeOwnerAndRetainsReplayAuthority)
{
    using Ordinary = Geometry<ArithmeticMode::Ordinary>;
    auto native = Ordinary::box({2, -4, .125}, {2.5, -3.875, 1.125});
    const std::weak_ptr<const Ordinary> discarded = native;
    const std::weak_ptr<const P::DynamicConvexGeometry> discarded_dynamic = native->dynamic_geometry();
    const std::weak_ptr<const P::DynamicConvexGeometry::Impl> discarded_payload =
        P::DynamicGeometryAccess::implementation(*native->dynamic_geometry());
    const auto imported = Ordinary::from_geometry(*native);
    ASSERT_TRUE(native->same_native_representation(*imported));
    P::GeometryAllocationLedger publications;
    std::size_t publication_bytes = 0;
    const auto native_points = native->export_points(publications, publication_bytes, byte_limit);
    const auto imported_points = imported->export_points(publications, publication_bytes, byte_limit);
    for (std::size_t vertex = 0; vertex < native_points.points().size(); ++vertex)
    {
        EXPECT_TRUE(native_points.points()[vertex].supports().empty());
        EXPECT_EQ(imported_points.points()[vertex].supports().size(), 3u);
    }
    native.reset();
    EXPECT_TRUE(discarded.expired());
    EXPECT_TRUE(discarded_dynamic.expired());
    EXPECT_TRUE(discarded_payload.expired());
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(imported, used, byte_limit);
    const auto before = used, payload = ledger.payload_bytes(), owners = ledger.owner_count();
    ledger.retain<ArithmeticMode::Ordinary>(imported->dynamic_geometry(), used, byte_limit);
    EXPECT_EQ(used, before);
    EXPECT_EQ(ledger.owner_count(), owners);
    const auto alias = Ordinary::from_dynamic(imported->dynamic_geometry());
    ledger.retain(alias, used, byte_limit);
    EXPECT_EQ(ledger.payload_bytes(), payload + sizeof(Ordinary));
    EXPECT_EQ(ledger.owner_count(), owners + 1);

    const P::GeometryPacketContext context{{2, 3, 5, 7}};
    P::ConstructionRegistry<ArithmeticMode::Ordinary> registry(context);
    registry.register_native(11, 0, imported);
    const P::GeometryHalfspace cut{{1, 0, 0}, {2, -4, .125}, .25};
    registry.register_plane(13, cut);
    const auto piece = imported->clip(cut, false).inside;
    ASSERT_TRUE(piece);
    const auto replay = Ordinary::decode(piece->encode(registry), registry);
    EXPECT_EQ(piece->volume().value, replay->volume().value);
    EXPECT_EQ(piece->encode_certificate(), replay->encode_certificate());
    const auto before_moments = piece->moments(), after_moments = replay->moments();
    for (std::size_t axis = 0; axis < 3; ++axis)
        EXPECT_EQ(before_moments.first_moment[axis].value, after_moments.first_moment[axis].value);
}

TEST(GeometryAllocationLedgerTest, NativeRepresentationProofRejectsChangedOrConstructedSources)
{
    using Ordinary = Geometry<ArithmeticMode::Ordinary>;
    const auto native = Ordinary::box({0, 0, 0}, {1, 1, 1});
    const auto imported = Ordinary::from_geometry(*native);
    const auto changed = Ordinary::box({0, 0, 0}, {1, 1, 1.125});
    EXPECT_FALSE(native->same_native_representation(*changed));
    const auto last_bit = Ordinary::box({0, 0, 0}, {1, 1, std::nextafter(1., 2.)});
    EXPECT_FALSE(native->same_native_representation(*last_bit));
    const auto signed_zero = Ordinary::box({-0., 0, 0}, {1, 1, 1});
    EXPECT_FALSE(native->same_native_representation(*signed_zero));
    std::vector<P::PlaneSupportIdentity<ArithmeticMode::Ordinary>> altered_supports;
    for (std::size_t face = 0; face < native->num_faces(); ++face)
        altered_supports.push_back(P::planar_support<ArithmeticMode::Ordinary>(*native->face_geometry(face)));
    altered_supports[0] = altered_supports[1];
    const auto wrong_support = P::details::NativeSupportAccess::cell_supports(*native, altered_supports);
    EXPECT_EQ(native->volume().value, wrong_support->volume().value);
    EXPECT_EQ(native->local_points()[0].value, wrong_support->local_points()[0].value);
    EXPECT_FALSE(native->same_native_representation(*wrong_support));
    const auto shifted = native->translated({1, 0, 0});
    EXPECT_FALSE(shifted->same_native_representation(*shifted));
    const auto piece = imported->clip({{1, 0, 0}, {}, .5}, false).inside;
    ASSERT_TRUE(piece);
    EXPECT_FALSE(piece->same_native_representation(*piece));
    const auto restored = Ordinary::decode_certificate(native->encode_certificate());
    EXPECT_EQ(native->encode_certificate(), restored->encode_certificate());
    EXPECT_FALSE(native->same_native_representation(*restored));
    // An explicit source import can enrich that decoded shell, but it cannot
    // authorize replacement of its weaker decoded point definitions.
    const auto restored_source = Ordinary::from_geometry(*restored);
    EXPECT_FALSE(restored->same_native_representation(*restored_source));
}

template<ArithmeticMode Mode> void sibling_contract()
{
    const auto root = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    auto split = root->clip({{1, 0, 0}, {}, .5}, false);
    ASSERT_TRUE(split.inside);
    ASSERT_TRUE(split.outside);
    const auto separate = split.inside->storage_bytes() + split.outside->storage_bytes();
    std::weak_ptr<const Geometry<Mode>> discarded = split.inside;
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain(split.inside, used, byte_limit);
        ledger.retain(split.outside, used, byte_limit);
        EXPECT_LT(ledger.payload_bytes(), separate);
        const auto charged = used;
        split.inside.reset();
        EXPECT_FALSE(discarded.expired());
        EXPECT_EQ(used, charged);
        ledger.retain(split.outside, used, byte_limit);
        EXPECT_EQ(used, charged);
        const auto fresh = root->clip({{1, 0, 0}, {}, .5}, false).inside;
        ASSERT_TRUE(fresh);
        const auto previous_payload = ledger.payload_bytes();
        ledger.retain(fresh, used, byte_limit);
        EXPECT_GT(used, charged);
        EXPECT_GT(ledger.payload_bytes(), previous_payload);
        EXPECT_FALSE(discarded.expired());
    }
    EXPECT_TRUE(discarded.expired());
}
TEST(GeometryAllocationLedgerTest, SharedSiblingsAreDeduplicatedAndDiscardedOwnersStayPinned)
{
    sibling_contract<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        sibling_contract<ArithmeticMode::BoundedCertification>();
}

TEST(GeometryAllocationLedgerTest, RejectedAndDecreasedBudgetsCannotResumeAdmission)
{
    const auto root = Geometry<ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger measured;
    std::size_t required = 0;
    measured.retain(root, required, byte_limit);
    ASSERT_GT(required, 1u);
    for (const auto cap : {std::size_t(1), required - 1})
    {
        P::GeometryAllocationLedger limited;
        std::size_t used = 0;
        EXPECT_THROW(limited.retain(root, used, cap), P::GeometryError);
        const auto charged = used;
        EXPECT_LE(charged, cap);
        EXPECT_THROW(limited.retain(root, used, byte_limit), P::GeometryError);
        EXPECT_EQ(used, charged);
    }
    auto decreased = required - 1;
    EXPECT_THROW(measured.retain(root, decreased, byte_limit), P::GeometryError);
    EXPECT_EQ(decreased, required - 1);
}

TEST(GeometryAllocationLedgerTest, SharedSupportOwnersAreCountedAcrossDifferentStoredPlanes)
{
    constexpr auto ordinary = ArithmeticMode::Ordinary;
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    std::vector<P::PlaneSupportIdentity<ordinary>> supports;
    std::size_t support_bytes = 0;
    for (std::size_t face = 0; face < root->num_faces(); ++face)
    {
        supports.push_back(P::planar_support<ordinary>(*root->face_geometry(face)));
        support_bytes += supports.back().payload_bytes();
    }
    const auto clone = P::details::NativeSupportAccess::cell_supports(*root, supports);
    for (std::size_t face = 0; face < root->num_faces(); ++face)
        EXPECT_EQ(P::planar_support<ordinary>(*clone->face_geometry(face)).storage_identity(),
            supports[face].storage_identity());
    P::GeometryAllocationLedger geometry_only;
    std::size_t geometry_used = 0;
    geometry_only.retain(root, geometry_used, byte_limit);
    P::GeometryAllocationLedger shared;
    std::size_t shared_used = 0;
    shared.retain(supports.front(), shared_used, byte_limit);
    shared.retain(root, shared_used, byte_limit);
    EXPECT_EQ(shared.payload_bytes(), geometry_only.payload_bytes());
    EXPECT_EQ(shared.owner_count(), geometry_only.owner_count());
    const auto independent =
        P::decode_identity<P::PlaneSupportIdentity<ordinary>>(P::encode_identity(supports.front()));
    ASSERT_EQ(independent, supports.front());
    ASSERT_NE(independent.storage_identity(), supports.front().storage_identity());
    P::GeometryAllocationLedger distinct;
    std::size_t distinct_used = 0;
    distinct.retain(independent, distinct_used, byte_limit);
    distinct.retain(root, distinct_used, byte_limit);
    EXPECT_EQ(distinct.payload_bytes(), geometry_only.payload_bytes() + independent.payload_bytes());
    EXPECT_EQ(distinct.owner_count(), geometry_only.owner_count() + 1);
    shared.retain(clone, shared_used, byte_limit);
    EXPECT_LE(shared.payload_bytes(), root->storage_bytes() + clone->storage_bytes() - support_bytes);
}
TEST(GeometryAllocationLedgerTest, IdentityPinsAndBudgetPoisoningHaveTheSameAdmissionContract)
{
    constexpr auto exact = ArithmeticMode::Exact;
    const std::array<P::GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    std::optional<P::PointIdentity<exact>> identity;
    std::weak_ptr<const void> weak;
    {
        const auto face = P::PlanarFace<exact>::make(points);
        identity = face->vertex_identities().front();
        weak = identity->storage_owner();
    }
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain(*identity, used, byte_limit);
        EXPECT_EQ(ledger.owner_count(), 1u);
        EXPECT_EQ(ledger.payload_bytes(), identity->payload_bytes());
        EXPECT_EQ(used, ledger.payload_bytes() + ledger.bookkeeping_bytes());
        const auto required = used;
        ledger.retain(*identity, used, byte_limit);
        EXPECT_EQ(used, required);
        for (const auto cap : {std::size_t(1), required - 1})
        {
            P::GeometryAllocationLedger limited;
            std::size_t admitted = 0;
            EXPECT_THROW(limited.retain(*identity, admitted, cap), P::GeometryError);
            EXPECT_LE(admitted, cap);
            const auto charged = admitted;
            EXPECT_THROW(limited.retain(*identity, admitted, byte_limit), P::GeometryError);
            EXPECT_EQ(admitted, charged);
        }
        auto decreased = required - 1;
        EXPECT_THROW(ledger.retain(*identity, decreased, byte_limit), P::GeometryError);
        EXPECT_EQ(decreased, required - 1);
        identity.reset();
        EXPECT_FALSE(weak.expired());
    }
    EXPECT_TRUE(weak.expired());
}
TEST(GeometryAllocationLedgerTest, IdentityModesDoNotSelectNumericalGeometryModeAndZeroNeedsNoOwner)
{
    constexpr auto ordinary = ArithmeticMode::Ordinary;
    constexpr auto exact = ArithmeticMode::Exact;
    P::GeometryAllocationLedger empty;
    std::size_t zero_used = 0;
    empty.retain(P::DisplacementIdentity<exact>{}, zero_used, 0);
    EXPECT_EQ(zero_used, 0u);
    EXPECT_EQ(empty.owner_count(), 0u);
    EXPECT_EQ(empty.bookkeeping_bytes(), 0u);
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto support = P::planar_support<ordinary>(*root->face_geometry(0));
    const std::array<P::GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto point = P::PlanarFace<exact>::make(points)->vertex_identities().front();
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(support, used, byte_limit);
    ledger.retain(point, used, byte_limit);
    EXPECT_NO_THROW(ledger.retain(root, used, byte_limit));
    EXPECT_EQ(ledger.payload_bytes(), root->storage_bytes() + point.payload_bytes());
}

TEST(GeometryAllocationLedgerTest, DynamicGeometryUsesTheSameOwnersAndNumericalModeContract)
{
    constexpr auto ordinary = ArithmeticMode::Ordinary;
    auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto total_payload = root->storage_bytes();
    std::weak_ptr<const P::DynamicConvexGeometry> weak = root->dynamic_geometry();
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain<ordinary>(root->dynamic_geometry(), used, byte_limit);
        EXPECT_EQ(ledger.payload_bytes(), total_payload - sizeof(Geometry<ordinary>));
        const auto dynamic_owners = ledger.owner_count();
        ledger.retain(root, used, byte_limit);
        EXPECT_EQ(ledger.payload_bytes(), total_payload);
        EXPECT_EQ(ledger.owner_count(), dynamic_owners + 1);
        root.reset();
        EXPECT_FALSE(weak.expired());
        if (P::certification_environment())
        {
            const auto bounded = Geometry<ArithmeticMode::BoundedCertification>::box({0, 0, 0}, {1, 1, 1});
            const auto charged = used;
            EXPECT_THROW(
                ledger.retain<ArithmeticMode::BoundedCertification>(bounded->dynamic_geometry(), used, byte_limit),
                std::invalid_argument);
            EXPECT_EQ(used, charged);
            EXPECT_THROW(ledger.retain<ordinary>(weak.lock(), used, byte_limit), P::GeometryError);
        }
    }
    EXPECT_TRUE(weak.expired());
}

template<ArithmeticMode Mode> void shared_support_payload_contract()
{
    using Access = P::DynamicGeometryAccess;
    const std::array<P::GeometryPoint, 3> face_points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto face = P::PlanarFace<ArithmeticMode::Ordinary>::make(face_points);
    const auto support = P::planar_support<ArithmeticMode::Ordinary>(*face);
    P::DynamicPoint source({1, 2, 4}, {.125L, 0, 0}, Mode);
    std::vector<std::uint64_t> first_words{2, 7, 13}, second_words{17, 19};
    first_words.reserve(11);
    second_words.reserve(17);
    const auto opaque_bytes = 2 * sizeof(std::vector<std::uint64_t>) +
                              (first_words.capacity() + second_words.capacity()) * sizeof(std::uint64_t);
    Access::seal(
        source, {P::PlaneAuthority::opaque(std::move(first_words)), P::PlaneAuthority::opaque(std::move(second_words)),
                    P::details::original_support_authority(support)});
    P::DynamicPoint exported(source.value, source.absolute_error, Mode);
    const auto root = Access::identity(source);
    const auto keys = source.supports();
    Access::seal_export(exported, root, {keys.begin(), keys.end()});
    const auto copy = Access::identity(exported);
    ASSERT_NE(root, copy);
    EXPECT_EQ(root.view().supports(), copy.view().supports());
    EXPECT_EQ(copy.view().source_origin(), root.view());
    EXPECT_EQ(
        Access::identity_bytes(copy) - Access::identity_bytes(copy, false), opaque_bytes + support.payload_bytes());
    // Three backing key owners remain shared even though source and export
    // snapshots have independently issued CSR rows.
    for (const bool vertex_incidence : {false, true})
    {
        P::PointSnapshotBuilder<Mode> builder;
        const auto point_ref = builder.import(copy);
        const auto range =
            vertex_incidence ? builder.import_authority_range(source.supports()) : P::LocalAuthorityRange<Mode>{};
        const auto context = builder.freeze();
        auto storage = std::make_shared<P::details::GeometryStorage<Mode>>();
        storage->ancestry.reset();
        storage->moment_cache.reset();
        storage->point_context = context;
        storage->vertices.resize(1);
        storage->vertices[0].source_identity = point_ref;
        storage->vertices[0].authorities = range;
        const auto geometry = Geometry<Mode>::from_dynamic(Access::geometry(storage));
        const auto added_context =
            sizeof(*context) + 2 * sizeof(void*) +
            context->dependencies.capacity() * sizeof(std::shared_ptr<const P::PointSnapshotContextBase>) +
            context->authority_table.capacity() * sizeof(P::PlaneAuthority) +
            context->incidence.capacity() * sizeof(P::PrivateAuthorityRef<Mode>);
        const auto expected = sizeof(Geometry<Mode>) + sizeof(P::DynamicConvexGeometry) + sizeof(*storage) +
                              storage->vertices.capacity() * sizeof(P::details::StoredVertex<Mode>) + added_context +
                              Access::identity_bytes(copy);
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain(geometry, used, byte_limit);
        // Three wrappers/storage, three snapshots, and three key payloads.
        EXPECT_EQ(ledger.owner_count(), 9u);
        EXPECT_EQ(ledger.payload_bytes(), expected);
        EXPECT_EQ(used, expected + ledger.bookkeeping_bytes());
        const auto admitted = used;
        ledger.retain(geometry, used, byte_limit);
        EXPECT_EQ(used, admitted);
    }
}
TEST(GeometryAllocationLedgerTest, SharedPointSupportsAreChargedAsOneSeparateImmutableOwner)
{
    shared_support_payload_contract<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        shared_support_payload_contract<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode> void provenance_dag_contract()
{
    using Access = P::DynamicGeometryAccess;
    using Context = P::PointSnapshotContext<Mode>;
    const std::array<P::GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto support =
        P::planar_support<ArithmeticMode::Ordinary>(*P::PlanarFace<ArithmeticMode::Ordinary>::make(points));
    std::vector<std::uint64_t> first_words{2, 7}, second_words{13, 17, 19};
    first_words.reserve(7);
    second_words.reserve(11);
    const auto opaque_bytes =
        2 * sizeof(first_words) + (first_words.capacity() + second_words.capacity()) * sizeof(std::uint64_t);
    const auto first = P::PlaneAuthority::opaque(std::move(first_words));
    const auto second = P::PlaneAuthority::opaque(std::move(second_words));
    const auto typed = P::details::original_support_authority(support);
    const std::array first_keys{first, typed}, second_keys{second, first};
    struct Fixture
    {
        P::PointProvenanceRef root;
        P::StoredPointRef<Mode> root_slot;
        std::array<std::weak_ptr<const P::PointSnapshotContextBase>, 4> contexts;
        std::weak_ptr<const P::NativePointFactBlock<Mode>> facts;
    };
    auto fixture = [&]
    {
        P::PointSnapshotBuilder<Mode> native;
        const auto leaf = native.native_source({0, 0, 0}, {}, Access::admit_incidence<Mode>(first_keys));
        const auto other = native.native_source({2, 0, 0}, {}, Access::admit_incidence<Mode>(second_keys));
        const auto native_context = native.freeze();
        P::PointSnapshotBuilder<Mode> generated;
        const std::array branch_edges{std::array{other, leaf}, std::array{other, P::PointProvenanceView{}}};
        const auto branch =
            generated.generated({1, 0, 0}, {}, branch_edges, Access::admit_incidence<Mode>(second_keys));
        const std::array root_edges{std::array{branch, leaf}, std::array{P::PointProvenanceView{}, branch}};
        const auto root_value =
            generated.generated({.5L, 0, 0}, {}, root_edges, Access::admit_incidence<Mode>(first_keys));
        const auto generated_context = generated.freeze();
        std::vector<P::PointProvenanceView> children;
        P::details::visit_provenance_children(root_value, [&](auto child) { children.push_back(child); });
        EXPECT_EQ(children, (std::vector<P::PointProvenanceView>{branch, leaf, {}, branch}));
        children.clear();
        P::details::visit_provenance_children(branch, [&](auto child) { children.push_back(child); });
        EXPECT_EQ(children, (std::vector<P::PointProvenanceView>{other, leaf, other, {}}));
        P::PointSnapshotBuilder<Mode> chain;
        const auto alias =
            chain.export_alias(root_value.value(), {}, root_value, Access::admit_incidence<Mode>(first_keys));
        const auto chain_context = chain.freeze();
        P::PointSnapshotBuilder<Mode> head;
        const auto root = head.export_alias(alias.value(), {}, alias, Access::admit_incidence<Mode>(first_keys));
        const auto root_slot = head.import(root);
        const auto head_context = head.freeze();
        children.clear();
        P::details::visit_provenance_children(root, [&](auto child) { children.push_back(child); });
        EXPECT_EQ(children, (std::vector<P::PointProvenanceView>{alias}));
        children.clear();
        P::details::visit_provenance_children(alias, [&](auto child) { children.push_back(child); });
        EXPECT_EQ(children, (std::vector<P::PointProvenanceView>{root_value}));
        children.clear();
        P::details::visit_provenance_children(leaf, [&](auto child) { children.push_back(child); });
        EXPECT_TRUE(children.empty());
        EXPECT_EQ(Access::borrowed_original_source(root), root_value);
        return Fixture{root.owning(), root_slot, {native_context, generated_context, chain_context, head_context},
            native_context->native_fact_blocks.front()};
    }();
    const auto attachment_bytes = opaque_bytes + support.payload_bytes();
    EXPECT_EQ(Access::identity_bytes(fixture.root) - Access::identity_bytes(fixture.root, false), attachment_bytes);
    P::GeometryAllocationLedger attachments, manual;
    std::size_t attachment_used = 0, manual_used = 0;
    Access::retain_supports(fixture.root, attachments, attachment_used, byte_limit);
    manual.retain(first, manual_used, byte_limit);
    manual.retain(typed, manual_used, byte_limit);
    manual.retain(second, manual_used, byte_limit);
    EXPECT_EQ(attachments.owner_count(), 3u);
    EXPECT_EQ(attachments.payload_bytes(), attachment_bytes);
    EXPECT_EQ(attachments.payload_bytes(), manual.payload_bytes());
    EXPECT_EQ(attachment_used, manual_used);
    Access::retain_supports(fixture.root, attachments, attachment_used, byte_limit);
    EXPECT_EQ(attachment_used, manual_used);
    auto storage = std::make_shared<P::details::GeometryStorage<Mode>>();
    storage->ancestry.reset();
    storage->moment_cache.reset();
    storage->vertices.resize(1);
    storage->point_context = std::static_pointer_cast<const Context>(fixture.root.context_pin());
    storage->vertices[0].source_identity = fixture.root_slot;
    auto geometry = Geometry<Mode>::from_dynamic(Access::geometry(storage));
    const auto expected = sizeof(Geometry<Mode>) + sizeof(P::DynamicConvexGeometry) + sizeof(*storage) +
                          storage->vertices.capacity() * sizeof(P::details::StoredVertex<Mode>) +
                          Access::identity_bytes(fixture.root);
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain(geometry, used, byte_limit);
        // Three geometry owners, four contexts, one native block, three keys.
        EXPECT_EQ(ledger.owner_count(), 11u);
        EXPECT_EQ(ledger.payload_bytes(), expected);
        EXPECT_EQ(used, expected + ledger.bookkeeping_bytes());
        const auto required = used;
        ledger.retain(geometry, used, byte_limit);
        EXPECT_EQ(used, required);
        P::GeometryAllocationLedger exact;
        std::size_t exact_used = 0;
        EXPECT_NO_THROW(exact.retain(geometry, exact_used, required));
        EXPECT_EQ(exact_used, required);
        P::GeometryAllocationLedger limited;
        std::size_t limited_used = 0;
        EXPECT_THROW(limited.retain(geometry, limited_used, required - 1), P::GeometryError);
        const auto rejected = limited_used;
        EXPECT_THROW(limited.retain(geometry, limited_used, byte_limit), P::GeometryError);
        EXPECT_EQ(limited_used, rejected);
        geometry.reset();
        storage.reset();
        fixture.root.reset();
        for (const auto& weak : fixture.contexts)
            EXPECT_FALSE(weak.expired());
        EXPECT_FALSE(fixture.facts.expired());
    }
    for (const auto& weak : fixture.contexts)
        EXPECT_TRUE(weak.expired());
    EXPECT_TRUE(fixture.facts.expired());
}
TEST(GeometryAllocationLedgerTest, ProvenanceDagPreservesUniqueOwnersCapsAndDescendantLifetimes)
{
    provenance_dag_contract<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        provenance_dag_contract<ArithmeticMode::BoundedCertification>();
}

template<ArithmeticMode Mode> void construction_payload_contract()
{
    namespace D = P::details;
    using Node = D::GeometryConstruction;
    using Storage = D::GeometryStorage<Mode>;
    P::GeometryLimits limits;
    limits.arithmetic_mode = Mode;
    D::Operation operation(limits);
    const auto empty_storage = []
    {
        auto storage = std::make_shared<Storage>();
        storage->ancestry.reset();
        storage->moment_cache.reset();
        return storage;
    };
    const P::GeometryHalfspace first{{2, 0, 0}, {.125, .25, .375}, .75};
    const P::GeometryHalfspace second{{0, 4, 0}, {.25, .5, .75}, 6};
    const auto make_pair = [&]
    {
        const auto source = D::construction_source<Mode>(empty_storage());
        const auto parent = D::construction_rebase(source, first);
        auto clip = std::make_shared<D::GeometryClipConstruction>();
        clip->mode = Mode;
        clip->parent = parent;
        std::array<P::ConvexGeometryPtr<Mode>, 2> pair;
        for (std::size_t side = 0; side < pair.size(); ++side)
        {
            auto storage = empty_storage();
            storage->construction = D::construction_rebase(D::construction_node(Node::Clip{clip, side != 0}), second);
            pair[side] = Geometry<Mode>::from_dynamic(P::DynamicGeometryAccess::geometry(storage));
        }
        return pair;
    };
    const auto pair = make_pair();
    // No shell buffers, ancestry or moment caches: independently enumerate
    // two wrappers, two storages, four recipe owners, two exclusive rebase
    // payloads, and one shared clip owner. Exclusive payloads add no owners.
    const auto expected = sizeof(Geometry<Mode>) + sizeof(P::DynamicConvexGeometry) + 2 * sizeof(Storage) +
                          4 * sizeof(Node) + 2 * sizeof(Node::Rebase) + sizeof(D::GeometryClipConstruction);
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain(pair[0], used, byte_limit);
    EXPECT_EQ(ledger.payload_bytes(), expected);
    EXPECT_EQ(ledger.owner_count(), 9u);
    EXPECT_EQ(pair[0]->storage_bytes(), expected);
    const auto first_used = used;
    ledger.retain(pair[0], used, byte_limit);
    EXPECT_EQ(used, first_used);
    // The sibling adds its wrappers, shell, clip recipe and final rebase
    // recipe/payload, while the shared clip and predecessor chain stay pinned.
    const auto sibling_bytes = sizeof(Geometry<Mode>) + sizeof(P::DynamicConvexGeometry) + sizeof(Storage) +
                               2 * sizeof(Node) + sizeof(Node::Rebase);
    ledger.retain(pair[1], used, byte_limit);
    EXPECT_EQ(ledger.payload_bytes(), expected + sibling_bytes);
    EXPECT_EQ(ledger.owner_count(), 14u);
    const auto distinct = make_pair();
    ledger.retain(distinct[0], used, byte_limit);
    EXPECT_EQ(ledger.payload_bytes(), 2 * expected + sibling_bytes);
    EXPECT_EQ(ledger.owner_count(), 23u);

    // Independent ledgers reproduce the same admission cost including scratch
    // bookkeeping. The exact cap succeeds; one byte less poisons admission.
    P::GeometryAllocationLedger exact;
    std::size_t exact_used = 0;
    exact.retain(pair[0], exact_used, first_used);
    EXPECT_EQ(exact_used, first_used);
    P::GeometryAllocationLedger short_cap;
    std::size_t short_used = 0;
    EXPECT_THROW(short_cap.retain(pair[0], short_used, first_used - 1), P::GeometryError);
    EXPECT_LE(short_used, first_used - 1);
    const auto rejected_used = short_used;
    EXPECT_THROW(short_cap.retain(pair[0], short_used, byte_limit), P::GeometryError);
    EXPECT_EQ(short_used, rejected_used);

    const auto& recipe = P::DynamicGeometryAccess::implementation(*pair[0]->dynamic_geometry())->construction;
    const auto source = recipe->rebase().parent->clip().clip->parent->rebase().parent->source().source;
    const auto source_bytes = P::DynamicGeometryAccess::geometry(source)->template storage_bytes<Mode>();
    EXPECT_EQ(D::construction_storage_bytes(recipe),
        4 * sizeof(Node) + 2 * sizeof(Node::Rebase) + sizeof(D::GeometryClipConstruction) + source_bytes);
}
TEST(GeometryAllocationLedgerTest, ExclusiveRebasePayloadsAndSharedRecipesHaveExactAdmissionCosts)
{
    construction_payload_contract<ArithmeticMode::Ordinary>();
    if (P::certification_environment())
        construction_payload_contract<ArithmeticMode::BoundedCertification>();
}
} // namespace
