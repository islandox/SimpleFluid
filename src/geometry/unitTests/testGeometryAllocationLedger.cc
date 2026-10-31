/** @file testGeometryAllocationLedger.cc
 * @brief Unique immutable-payload admission, strong pins and cumulative cap checks.
 */
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
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
    Access::seal_export(exported, root, source.supports());
    const auto copy = Access::identity(exported);
    ASSERT_NE(root, copy);
    ASSERT_EQ(root->support_payload, copy->support_payload);

    auto provenance_bytes = 2 * sizeof(P::DynamicPointProvenance) + sizeof(P::DynamicPointSupportPayload) +
                            root->supports().capacity() * sizeof(P::PlaneAuthority);
    EXPECT_EQ(Access::identity_bytes(copy, false), provenance_bytes);
    EXPECT_EQ(Access::identity_bytes(copy), provenance_bytes + opaque_bytes + support.payload_bytes());

    enum class VertexSupports
    {
        Empty,
        Owned,
        Shared
    };
    for (const auto vertex_supports : {VertexSupports::Empty, VertexSupports::Owned, VertexSupports::Shared})
    {
        SCOPED_TRACE(static_cast<unsigned>(vertex_supports));
        // Independently enumerate two wrappers, storage, one vertex slot,
        // two provenance records, one shared support payload and its typed
        // support attachment. Empty vertex authorities make the attachment
        // reachable only through the vertex source-provenance chain. The owned vertex case
        // additionally charges its private outer buffer; the two immutable
        // opaque payload owners remain shared across every authority copy.
        auto storage = std::make_shared<P::details::GeometryStorage<Mode>>();
        storage->ancestry.reset();
        storage->moment_cache.reset();
        storage->vertices.resize(1);
        auto& vertex = storage->vertices[0];
        vertex.source_identity = copy;
        if (vertex_supports != VertexSupports::Empty)
            vertex.authorities = source.supports();
        if (vertex_supports == VertexSupports::Shared)
            vertex.authorities.share_if_equal(root->support_payload);
        const auto geometry = Geometry<Mode>::from_dynamic(Access::geometry(storage));
        auto expected = sizeof(Geometry<Mode>) + sizeof(P::DynamicConvexGeometry) + sizeof(*storage) +
                        storage->vertices.capacity() * sizeof(P::details::StoredVertex<Mode>) + provenance_bytes +
                        opaque_bytes + support.payload_bytes();
        if (vertex_supports == VertexSupports::Owned)
        {
            const auto& keys = vertex.authorities.values();
            expected += keys.capacity() * sizeof(P::PlaneAuthority);
        }
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain(geometry, used, byte_limit);
        EXPECT_EQ(ledger.owner_count(), 9u);
        EXPECT_EQ(ledger.payload_bytes(), expected);
        EXPECT_EQ(used, expected + ledger.bookkeeping_bytes());
        const auto first_used = used;
        ledger.retain(geometry, used, byte_limit);
        EXPECT_EQ(used, first_used);
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
    using Record = P::DynamicPointProvenance;
    using Owner = std::shared_ptr<const Record>;
    using Payload = P::DynamicPointSupportPayload;
    const std::array<P::GeometryPoint, 3> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto support =
        P::planar_support<ArithmeticMode::Ordinary>(*P::PlanarFace<ArithmeticMode::Ordinary>::make(points));
    std::vector<std::uint64_t> first_words{2, 7}, second_words{13, 17, 19};
    first_words.reserve(7);
    second_words.reserve(11);
    const auto opaque_bytes = 2 * sizeof(first_words) +
                              (first_words.capacity() + second_words.capacity()) * sizeof(std::uint64_t);
    const auto first = P::PlaneAuthority::opaque(std::move(first_words));
    const auto second = P::PlaneAuthority::opaque(std::move(second_words));
    const auto typed = P::details::original_support_authority(support);
    std::vector<P::PlaneAuthority> first_supports{first, typed}, second_supports{second, first};
    first_supports.reserve(5);
    second_supports.reserve(9);
    std::shared_ptr<const Payload> first_payload = std::make_shared<Payload>(std::move(first_supports));
    std::shared_ptr<const Payload> second_payload = std::make_shared<Payload>(std::move(second_supports));
    const auto make_record = [&](const std::shared_ptr<const Payload>& payload)
    {
        auto record = std::make_shared<Record>();
        record->value = {1, 2, 4};
        record->absolute_error = {};
        record->arithmetic_mode = Mode;
        record->support_payload = payload;
        return record;
    };
    auto leaf = make_record(first_payload), chain = make_record(first_payload), root = make_record(first_payload);
    auto other = make_record(second_payload), branch = make_record(second_payload);
    chain->source_origin = leaf;
    root->source_origin = chain;
    root->source_edges.reserve(7);
    root->source_edges = {{branch, leaf}, {nullptr, branch}};
    branch->source_edges.reserve(11);
    branch->source_edges = {{other, leaf}, {other, nullptr}};

    // Verify actual borrowed slots, including duplicate and null endpoints.
    // The temporary visitor can only be invoked as a named lvalue.
    struct Visitor
    {
        std::vector<const Owner*>& slots;
        void operator()(const Owner& owner) & { slots.push_back(&owner); }
        void operator()(const Owner&) && = delete;
    };
    std::vector<const Owner*> slots;
    P::details::visit_provenance_children(*root, Visitor{slots});
    EXPECT_EQ(slots, (std::vector<const Owner*>{&root->source_origin, &root->source_edges[0][0],
                         &root->source_edges[0][1], &root->source_edges[1][0], &root->source_edges[1][1]}));
    slots.clear();
    P::details::visit_provenance_children(*branch, Visitor{slots});
    EXPECT_EQ(slots, (std::vector<const Owner*>{&branch->source_edges[0][0], &branch->source_edges[0][1],
                         &branch->source_edges[1][0], &branch->source_edges[1][1]}));
    slots.clear();
    P::details::visit_provenance_children(*leaf, Visitor{slots});
    EXPECT_TRUE(slots.empty());

    // Fixed graph: root -> chain -> leaf; root edges reach branch and leaf;
    // branch edges reach other and leaf. Five records, two shared support
    // payloads, two opaque word owners and one typed support owner.
    const auto provenance_bytes = 5 * sizeof(Record) +
                                  (root->source_edges.capacity() + branch->source_edges.capacity()) *
                                      sizeof(root->source_edges.front()) +
                                  2 * sizeof(Payload) +
                                  (first_payload->authorities.capacity() + second_payload->authorities.capacity()) *
                                      sizeof(P::PlaneAuthority);
    const auto attachment_bytes = opaque_bytes + support.payload_bytes();
    EXPECT_EQ(Access::identity_bytes(root, false), provenance_bytes);
    EXPECT_EQ(Access::identity_bytes(root), provenance_bytes + attachment_bytes);
    P::GeometryAllocationLedger manual, attachments;
    std::size_t manual_used = 0, attachment_used = 0;
    manual.retain(first, manual_used, byte_limit);
    manual.retain(typed, manual_used, byte_limit);
    manual.retain(second, manual_used, byte_limit);
    Access::retain_supports(root, attachments, attachment_used, byte_limit);
    EXPECT_EQ(attachments.owner_count(), 3u);
    EXPECT_EQ(attachments.owner_count(), manual.owner_count());
    EXPECT_EQ(attachments.payload_bytes(), attachment_bytes);
    EXPECT_EQ(attachments.payload_bytes(), manual.payload_bytes());
    EXPECT_EQ(attachment_used, manual_used);
    Access::retain_supports(root, attachments, attachment_used, byte_limit);
    EXPECT_EQ(attachment_used, manual_used);

    auto storage = std::make_shared<P::details::GeometryStorage<Mode>>();
    storage->ancestry.reset();
    storage->moment_cache.reset();
    storage->vertices.resize(1);
    storage->vertices[0].source_identity = root;
    auto geometry = Geometry<Mode>::from_dynamic(Access::geometry(storage));
    const auto expected = sizeof(Geometry<Mode>) + sizeof(P::DynamicConvexGeometry) + sizeof(*storage) +
                          storage->vertices.capacity() * sizeof(P::details::StoredVertex<Mode>) + provenance_bytes +
                          attachment_bytes;
    const std::array<std::weak_ptr<const Record>, 5> weak_records{root, chain, leaf, branch, other};
    const std::array<std::weak_ptr<const Payload>, 2> weak_payloads{first_payload, second_payload};
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        ledger.retain(geometry, used, byte_limit);
        EXPECT_EQ(ledger.owner_count(), 13u);
        EXPECT_EQ(ledger.payload_bytes(), expected);
        EXPECT_EQ(used, expected + ledger.bookkeeping_bytes());
        const auto required = used;
        ledger.retain(geometry, used, byte_limit);
        EXPECT_EQ(used, required);
        {
            P::GeometryAllocationLedger exact;
            std::size_t exact_used = 0;
            EXPECT_NO_THROW(exact.retain(geometry, exact_used, required));
            EXPECT_EQ(exact_used, required);
            P::GeometryAllocationLedger limited;
            std::size_t limited_used = 0;
            EXPECT_THROW(limited.retain(geometry, limited_used, required - 1), P::GeometryError);
            EXPECT_LE(limited_used, required - 1);
            const auto rejected = limited_used;
            EXPECT_THROW(limited.retain(geometry, limited_used, byte_limit), P::GeometryError);
            EXPECT_EQ(limited_used, rejected);
        }
        geometry.reset();
        storage.reset();
        root.reset();
        chain.reset();
        leaf.reset();
        branch.reset();
        other.reset();
        first_payload.reset();
        second_payload.reset();
        for (const auto& weak : weak_records)
            EXPECT_FALSE(weak.expired());
        for (const auto& weak : weak_payloads)
            EXPECT_FALSE(weak.expired());
    }
    for (const auto& weak : weak_records)
        EXPECT_TRUE(weak.expired());
    for (const auto& weak : weak_payloads)
        EXPECT_TRUE(weak.expired());
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
            storage->construction = D::construction_rebase(
                D::construction_node(Node::Clip{clip, side != 0}), second);
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
