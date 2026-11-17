/** Independent graph-set, provenance, lifetime and observational checks. */
#include "geometry/PLIC/details/ConvexGeometryInternal.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/PLIC/details/GeometryIdentityRetention.hh"
#include "geometry/PLIC/details/GeometryOwnerGraph.hh"
#include "geometry/PLIC/details/GeometryRetentionCensus.hh"
#include "geometry/PLIC/details/GeometryRetentionCensusAccess.hh"
#include "geometry/PLIC/details/exact/GeometryIdentity.hh"
#include "pipeline_allocation_operators.inc"
#include <algorithm>
#include <array>
#include <cfenv>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace
{
namespace P = SimpleFluid::PLIC;
namespace R = P::details::retention;
using SimpleFluid::ArithmeticMode;
constexpr auto ordinary = ArithmeticMode::Ordinary;
constexpr std::size_t cap = 64 * 1024 * 1024;
using Access = P::DynamicGeometryAccess;
void uncharged(void*, std::size_t, std::size_t) {}
using Builder = P::PointSnapshotBuilder<ordinary>;
P::DynamicPoint public_point(P::PointProvenanceView view)
{
    P::DynamicPoint p(view.value(), view.absolute_error(), ordinary);
    Access::attach_identity(p, view.owning());
    return p;
}
P::DynamicPoint native(std::array<long double, 3> coordinates = {1, 2, 3})
{
    Builder builder({nullptr, uncharged});
    const auto view = builder.native_source(coordinates, {}, {});
    const auto owner = builder.freeze();
    return public_point(view);
}
struct Scratch
{
    std::vector<std::byte> bytes = std::vector<std::byte>(8 * 1024 * 1024);
    R::ProbeScratch resource{bytes};
};
std::size_t kind_count(const R::Report& report, const char* kind)
{
    return std::count_if(report.owners.begin(), report.owners.end(),
        [&](const auto& owner) { return std::string_view(owner.kind) == kind; });
}
} // namespace

TEST(GeometryRetentionCensus, ExactOwnerSetsDuplicateRootsAndGroupOverlap)
{
    auto a = native(), equal_b = native(), ledger_only = native({7, 8, 9});
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain_point_provenance(a, used, cap);
    ledger.retain_point_provenance(ledger_only, used, cap);
    const std::array roots{R::root(a, 0), R::root(a, 0), R::root(a, 1), R::root(equal_b, 1)};
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, ledgers, scratch.resource, {1000, 1000, true});
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_TRUE(report.complete_root_manifest);
    EXPECT_EQ(report.root_count, 4u);
    EXPECT_EQ(report.owner_union.owners, 6u); // Three independent contexts and three fact blocks.
    EXPECT_EQ(report.application_only.owners, 2u);
    EXPECT_EQ(report.intersection.owners, 2u);
    EXPECT_EQ(report.ledger_only.owners, 2u);
    EXPECT_EQ(report.required_ledger_only.owners, 0u);
    EXPECT_EQ(report.unneeded_ledger_only.owners, 2u);
    EXPECT_EQ(
        report.owner_union.bytes, report.application_only.bytes + report.intersection.bytes + report.ledger_only.bytes);
    EXPECT_EQ(report.application_bytes, report.application_only.bytes + report.intersection.bytes);
    EXPECT_EQ(report.ledger_bytes, report.ledger_only.bytes + report.intersection.bytes);
    EXPECT_EQ(report.group_overlap_bytes, report.intersection.bytes);
    ASSERT_EQ(report.groups.size(), 2u);
    EXPECT_EQ(report.groups[0].roots, 2u);
    EXPECT_EQ(report.groups[0].exclusive_bytes, 0u);
    EXPECT_EQ(report.groups[0].inclusive_bytes, report.intersection.bytes);
    EXPECT_EQ(report.required_records, 2u); // Equal values in separate contexts stay separate.
    EXPECT_EQ(kind_count(report, "point_context"), 3u);
    EXPECT_EQ(kind_count(report, "native_facts"), 3u);
}

TEST(GeometryRetentionCensus, EveryGeneratedEdgeEndpointAndExportOriginRemainRequiredAfterMutation)
{
    P::DynamicPoint point, copied_generated;
    {
        Builder left({nullptr, uncharged}), right({nullptr, uncharged});
        const auto a = left.native_source({0, 0, 0}, {}, {});
        const auto b = left.native_source({1, 0, 0}, {}, {});
        const auto c = right.native_source({0, 1, 0}, {}, {});
        const auto d = right.native_source({1, 1, 0}, {}, {});
        auto left_owner = left.freeze(), right_owner = right.freeze();
        Builder generated({nullptr, uncharged});
        const std::array edges{std::array{a, b}, std::array{c, d}};
        const auto intersection = generated.generated({0.5L, 0.5L, 0}, {}, edges, {});
        auto generated_owner = generated.freeze();
        Builder exported({nullptr, uncharged});
        const auto alias = exported.export_alias({0.5L, 0.5L, 0}, {}, intersection, {});
        auto exported_owner = exported.freeze();
        point = public_point(alias);
        copied_generated = public_point(intersection);
        left_owner.reset();
        right_owner.reset();
        generated_owner.reset();
        exported_owner.reset();
    }
    // Public coordinate and mode mutation invalidate proofs but do not release provenance.
    point.value = {10, 20, 30};
    point.arithmetic_mode = ArithmeticMode::BoundedCertification;
    const std::array roots{R::root(point)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(report.required_records, 6u);
    EXPECT_EQ(report.maximum_required_point_dependency_depth, 2u);
    EXPECT_EQ(report.maximum_context_dependency_depth, 2u);
    EXPECT_EQ(report.contexts.size(), 4u);
    std::array<std::size_t, 4> required{};
    std::size_t required_edges = 0;
    for (const auto& context : report.contexts)
    {
        for (std::size_t i = 0; i < required.size(); ++i)
            required[i] += context.required_records[i];
        required_edges += context.required_edges;
    }
    EXPECT_EQ(required, (std::array<std::size_t, 4>{4, 1, 1, 0}));
    EXPECT_EQ(required_edges, 2u);
    EXPECT_EQ(report.uncovered_required_owners, 0u);
    const auto export_weak = P::details::RetentionAccess::provenance(point).context()->weak_from_this();
    point = {};
    EXPECT_TRUE(export_weak.expired());
    const std::array generated_roots{R::root(copied_generated)};
    Scratch generated_scratch;
    const auto generated_report = R::inspect_retention(generated_roots, {}, generated_scratch.resource);
    ASSERT_TRUE(generated_report.complete()) << generated_report.detail;
    EXPECT_EQ(generated_report.required_records, 5u);
    EXPECT_EQ(generated_report.maximum_required_point_dependency_depth, 1u);
    EXPECT_EQ(generated_report.contexts.size(), 3u);
}

TEST(GeometryRetentionCensus, SharedFactsDoNotMergeIndependentContextRows)
{
    Builder original({nullptr, uncharged});
    const auto source = original.native_source({1, 2, 3}, {}, {});
    const auto original_owner = original.freeze();
    Builder first({nullptr, uncharged}), second({nullptr, uncharged});
    const auto a = first.native_source({1, 2, 3}, {}, {}, source);
    const auto b = second.native_source({1, 2, 3}, {}, {}, source);
    const auto first_owner = first.freeze(), second_owner = second.freeze();
    const auto pa = public_point(a), pb = public_point(b);
    const std::array roots{R::root(pa), R::root(pb)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(report.required_records, 2u);
    EXPECT_EQ(kind_count(report, "native_facts"), 1u);
    EXPECT_EQ(report.contexts.size(), 2u);
}

TEST(GeometryRetentionCensus, ScratchFailureDoesNotPoisonLedgerOrChangeNumericalState)
{
    auto p = native();
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain_point_provenance(p, used, cap);
    const auto& pin = P::details::RetentionAccess::provenance(p).context()->weak_from_this();
    const auto count_before = pin.use_count();
    const auto owner_count = ledger.owner_count(), payload = ledger.payload_bytes(),
               bookkeeping = ledger.bookkeeping_bytes();
    const auto stats = P::geometry_statistics();
    const int rounding = std::fegetround(), exceptions = std::fetestexcept(FE_ALL_EXCEPT);
    const std::array roots{R::root(p)};
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    std::array<std::byte, 1> tiny{};
    R::ProbeScratch scratch(tiny);
    const auto report = R::inspect_retention(roots, ledgers, scratch);
    EXPECT_FALSE(report.complete());
    EXPECT_EQ(report.status, R::Status::ScratchExhausted);
    EXPECT_EQ(pin.use_count(), count_before);
    EXPECT_EQ(ledger.owner_count(), owner_count);
    EXPECT_EQ(ledger.payload_bytes(), payload);
    EXPECT_EQ(ledger.bookkeeping_bytes(), bookkeeping);
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), exceptions);
    EXPECT_EQ(P::geometry_statistics().floating_operations, stats.floating_operations);
    EXPECT_EQ(P::geometry_statistics().rejections, stats.rejections);
    EXPECT_NO_THROW(ledger.retain_point_provenance(p, used, cap));
}

TEST(GeometryRetentionCensus, ReportContainsNoStrongPinsAndNumericalRowsHaveNoProofAncestry)
{
    P::DynamicPoint p;
    std::weak_ptr<const P::PointSnapshotContextBase> weak;
    {
        Builder builder({nullptr, uncharged});
        const auto view = builder.numerical({1, 2, 3}, {});
        const auto context = builder.freeze();
        p = public_point(view);
        weak = context;
    }
    const std::array roots{R::root(p)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(report.required_records, 1u);
    EXPECT_EQ(kind_count(report, "native_facts"), 0u);
    EXPECT_EQ(report.contexts.front().required_authorities, 0u);
    EXPECT_EQ(report.contexts.front().required_edges, 0u);
    p = {};
    EXPECT_TRUE(weak.expired()); // Report and raw input manifest do not pin roots.
}

TEST(GeometryRetentionCensus, FailedLedgerRetainsActualPendingPinsSeparatelyFromAdmissionCounters)
{
    // Search narrowly over admission caps to exercise a failed walk with a
    // surviving pending pin; capacity layout is a standard-library detail.
    const auto geometry = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    bool found_pending = false;
    for (std::size_t limit = 256; limit <= 16384 && !found_pending; limit += 128)
    {
        P::GeometryAllocationLedger ledger;
        std::size_t used = 0;
        try
        {
            ledger.retain(geometry, used, limit);
        }
        catch (const P::GeometryError&)
        {
        }
        Scratch scratch;
        const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
        const auto report = R::inspect_retention({}, ledgers, scratch.resource);
        ASSERT_TRUE(report.complete()) << report.detail;
        if (!report.pending_pins)
            continue;
        found_pending = true;
        EXPECT_EQ(report.failed_ledgers, 1u);
        EXPECT_GE(report.ledger_only.owners, ledger.owner_count());
        EXPECT_EQ(report.application_bytes, 0u);
        EXPECT_EQ(report.ledger_bytes, report.owner_union.bytes);
        EXPECT_TRUE(P::details::RetentionAccess::failed(ledger));
        EXPECT_THROW(ledger.retain(geometry, used, cap), P::GeometryError);
    }
    EXPECT_TRUE(found_pending);
}

TEST(GeometryRetentionCensus, BorrowedPlaneContextNeedsDeclaredStrongPathAndCanBeRequiredLedgerOnly)
{
    const auto authority = P::PlaneAuthority::opaque({11, 13, 17});
    Builder builder({nullptr, uncharged});
    const auto source = builder.native_source(
        {0, 0, 0}, {}, Access::admit_incidence<ordinary>(std::span<const P::PlaneAuthority>(&authority, 1)));
    const auto ref = builder.import(source);
    const auto context = builder.freeze();
    auto plane = std::make_shared<P::details::StoredPlane<ordinary>>();
    plane->point_context = context.get();
    plane->authorities = context->native_refs[0].incidence;
    plane->source_vertices.push_back(ref);
    auto topology = std::make_shared<P::details::FrozenTopology<ordinary>>();
    topology->planes.push_back(plane);
    auto storage = std::make_shared<P::details::GeometryStorage<ordinary>>();
    storage->topology = topology;
    // The shell intentionally does not strongly own the plane's context.
    const auto geometry = Access::geometry(storage);
    const std::array roots{R::root(*geometry)};
    {
        Scratch scratch;
        const auto uncovered = R::inspect_retention(roots, {}, scratch.resource);
        EXPECT_EQ(uncovered.status, R::Status::InvalidGraph);
        EXPECT_EQ(uncovered.uncovered_required_owners, 1u);
        const auto contexts = std::count_if(uncovered.owners.begin(), uncovered.owners.end(),
            [](const auto& owner) { return owner.application && std::string_view(owner.kind) == "point_context"; });
        EXPECT_EQ(contexts, 0); // A raw plane pointer cannot create a strong edge.
    }
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain_snapshot<ordinary>(context, used, cap);
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, ledgers, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(report.required_records, 1u);
    EXPECT_EQ(report.required_ledger_only.owners, 3u); // Context, native facts, opaque authority.
    EXPECT_EQ(report.unneeded_ledger_only.owners, 0u);
    EXPECT_EQ(report.contexts[0].required_authorities, 1u);
    EXPECT_EQ(report.contexts[0].required_incidence, 1u);
}

TEST(GeometryRetentionCensus, FullConstructionReplayChainAndRebaseAreRequiredWithoutNumericalReplay)
{
    auto original = std::make_shared<P::details::GeometryStorage<ordinary>>();
    auto source =
        std::make_shared<P::details::GeometryConstruction>(P::details::GeometryConstruction::Source{original});
    auto rebase_payload = std::make_unique<P::details::GeometryConstruction::Rebase>();
    rebase_payload->parent = source;
    auto rebase = std::make_shared<P::details::GeometryConstruction>(std::move(rebase_payload));
    auto clipping = std::make_shared<P::details::GeometryClipConstruction>();
    clipping->parent = rebase;
    auto recipe =
        std::make_shared<P::details::GeometryConstruction>(P::details::GeometryConstruction::Clip{clipping, false});
    auto result = std::make_shared<P::details::GeometryStorage<ordinary>>();
    result->construction = recipe;
    const auto geometry = Access::geometry(result);
    const std::array roots{R::root(*geometry)};
    Scratch scratch;
    const auto before = P::geometry_statistics();
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(kind_count(report, "recipe"), 3u);
    EXPECT_EQ(kind_count(report, "clip"), 1u);
    EXPECT_EQ(kind_count(report, "geometry"), 2u);
    for (const auto& owner : report.owners)
        if (std::string_view(owner.kind) == "recipe" || std::string_view(owner.kind) == "clip")
            EXPECT_TRUE(owner.required);
    // Result geometry -> clip recipe -> clip record -> rebase recipe ->
    // source recipe -> original geometry: five retained replay edges.
    EXPECT_EQ(report.maximum_recipe_depth, 5u);
    EXPECT_EQ(P::geometry_statistics().constructions, before.constructions);
    EXPECT_EQ(P::geometry_statistics().floating_operations, before.floating_operations);
}

TEST(GeometryRetentionCensus, IdentityAncestryIsBorrowedAndCanonicalBytesAndActiveBudgetsRemainUnchanged)
{
    auto support = std::optional(P::details::GeometryIdentityAccess::support_axis(0, 1));
    for (unsigned i = 0; i < 8; ++i)
        *support = P::details::GeometryIdentityAccess::translated_support(*support, {1, 0, 0});
    const auto key = P::canonical_key(*support);
    const auto& backing = P::details::GeometryIdentityAccess::backing(*support);
    const auto count = backing.use_count();
    const auto weak = std::weak_ptr(backing);
    P::GeometryLimits limits;
    limits.arithmetic_mode = ordinary;
    P::details::Operation operation(limits);
    const auto used = operation.work.bytes, operations = operation.work.operations, pairs = operation.work.pairs;
    const auto ledger_owners = operation.work.allocations.owner_count();
    const auto previous_flags = std::fetestexcept(FE_ALL_EXCEPT), previous_round = std::fegetround();
    const std::array roots{R::root(*support)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(kind_count(report, "identity"), 9u);
    EXPECT_EQ(report.maximum_identity_depth, 9u);
    EXPECT_EQ(report.maximum_identity_dependency_depth, 8u);
    EXPECT_EQ(backing.use_count(), count);
    EXPECT_EQ(operation.work.bytes, used);
    EXPECT_EQ(operation.work.operations, operations);
    EXPECT_EQ(operation.work.pairs, pairs);
    EXPECT_EQ(operation.work.allocations.owner_count(), ledger_owners);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), previous_flags);
    EXPECT_EQ(std::fegetround(), previous_round);
    EXPECT_EQ(P::canonical_key(*support), key);
    support.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(GeometryRetentionCensus, RebaseSharesPointStorageWhilePhysicalTranslationDetachesIt)
{
    const auto original = P::ConvexCellGeometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto shifted = original->translated({2, 0, 0});
    const auto& implementation = Access::implementation(*original->dynamic_geometry());
    const auto* original_storage = static_cast<const P::details::GeometryStorage<ordinary>*>(implementation.get());
    // Hand-build the retained ownership shape of a numerical frame rebase.
    // The probe observes storage and recipes, so no arithmetic replay is needed.
    auto rebased_storage = std::make_shared<P::details::GeometryStorage<ordinary>>(*original_storage);
    rebased_storage->export_source = implementation;
    auto source =
        std::make_shared<P::details::GeometryConstruction>(P::details::GeometryConstruction::Source{implementation});
    auto payload = std::make_unique<P::details::GeometryConstruction::Rebase>();
    payload->parent = source;
    rebased_storage->construction = std::make_shared<P::details::GeometryConstruction>(std::move(payload));
    const auto rebased = Access::geometry(rebased_storage);
    P::GeometryAllocationLedger original_ledger, shifted_ledger;
    std::size_t original_used = 0, shifted_used = 0;
    const auto original_export = original->export_points(original_ledger, original_used, cap);
    const auto shifted_export = shifted->export_points(shifted_ledger, shifted_used, cap);
    const std::array roots{R::root(*original, 0), R::root(original_export, 0), R::root(*rebased, 1),
        R::root(*shifted, 2), R::root(shifted_export, 2)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    bool shared_rebase_context = false;
    for (const auto& owner : report.owners)
    {
        if (std::string_view(owner.kind) != "point_context")
            continue;
        if ((owner.root_groups & 3u) == 3u)
            shared_rebase_context = true;
        if (owner.root_groups & 4u)
            EXPECT_EQ(owner.root_groups & 3u, 0u);
    }
    EXPECT_TRUE(shared_rebase_context);
    EXPECT_GT(report.group_overlap_bytes, 0u);
}

TEST(GeometryRetentionCensus, NativeSupportBreakdownCountsSourceAndNestedRecipeCapacity)
{
    using IA = P::details::GeometryIdentityAccess;
    using Coordinate = P::details::OrdinaryIdentityPoint;
    std::vector<Coordinate> points;
    points.reserve(9);
    points.push_back({0, 0, 0});
    points.push_back({0, 1, 0});
    points.push_back({1, 0, 0});
    const auto point_capacity = points.capacity();
    const auto source = IA::support_source(std::move(points));
    std::vector<P::details::OrdinarySupportRecipe> recipes;
    recipes.reserve(7);
    recipes.resize(3);
    recipes[2].original = false;
    recipes[2].parents.reserve(8);
    recipes[2].parents = {0, 1};
    const auto recipe_capacity = recipes.capacity(), parent_capacity = recipes[2].parents.capacity();
    const auto dag = IA::support_dag(std::move(recipes), 1, 2);
    const auto& source_backing = IA::backing(source);
    const auto& dag_backing = IA::backing(dag);
    const auto source_native = P::details::ordinary_identity_retention_bytes(*source_backing->native);
    const auto dag_native = P::details::ordinary_identity_retention_bytes(*dag_backing->native);
    ASSERT_TRUE(source_native.complete);
    ASSERT_TRUE(dag_native.complete);
    EXPECT_FALSE(source_native.overflow);
    EXPECT_FALSE(dag_native.overflow);
    EXPECT_EQ(source_native.populated, 3 * sizeof(Coordinate));
    EXPECT_EQ(source_native.unused, (point_capacity - 3) * sizeof(Coordinate));
    EXPECT_EQ(dag_native.populated, 3 * sizeof(P::details::OrdinarySupportRecipe) + 2 * sizeof(std::size_t));
    EXPECT_EQ(dag_native.unused, (recipe_capacity - 3) * sizeof(P::details::OrdinarySupportRecipe) +
                                     (parent_capacity - 2) * sizeof(std::size_t));
    EXPECT_EQ(source_native.metadata + source_native.populated + source_native.unused,
        source_backing->native->storage_bytes());
    EXPECT_EQ(dag_native.metadata + dag_native.populated + dag_native.unused, dag_backing->native->storage_bytes());
    const std::array roots{R::root(source), R::root(dag)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    std::size_t populated = 0, unused = 0;
    for (const auto& row : report.owners)
    {
        EXPECT_EQ(row.bytes, row.metadata_bytes + row.populated_bytes + row.unused_capacity_bytes);
        populated += row.populated_bytes;
        unused += row.unused_capacity_bytes;
    }
    EXPECT_EQ(populated, source_native.populated + dag_native.populated);
    EXPECT_EQ(unused, source_native.unused + dag_native.unused);
}

TEST(GeometryRetentionCensus, ExactTupleSeparatesUsedHeapLimbsFromAllocatedLimbCapacity)
{
    namespace E = P::exact_geometry_detail;
    E::I x = 1, y = 1;
    x <<= 512; // Nine 64-bit limbs; other coordinates/denominators remain inline.
    y <<= 640; // Eleven 64-bit limbs.
    const auto identity = E::make_point_identity(E::P{E::Q(x), E::Q(-y), E::Q(1)});
    const auto& coefficients = E::authority_point(identity);
    const auto& first = coefficients[0].backend().num();
    const auto& second = coefficients[1].backend().num();
    const auto first_used = 512 / (8 * sizeof(boost::multiprecision::limb_type)) + 1;
    const auto second_used = 640 / (8 * sizeof(boost::multiprecision::limb_type)) + 1;
    ASSERT_EQ(first.size(), first_used);
    ASSERT_EQ(second.size(), second_used);
    const auto expected_populated = (first_used + second_used) * sizeof(boost::multiprecision::limb_type);
    const auto expected_unused =
        (first.capacity() - first_used + second.capacity() - second_used) * sizeof(boost::multiprecision::limb_type);
    const auto& backing = P::details::GeometryIdentityAccess::backing(identity);
    const auto count_before = backing.use_count();
    const auto rounding = std::fegetround(), flags = std::fetestexcept(FE_ALL_EXCEPT);
    const std::array roots{R::root(identity)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    ASSERT_EQ(report.owners.size(), 1u);
    EXPECT_EQ(report.owners[0].populated_bytes, expected_populated);
    EXPECT_EQ(report.owners[0].unused_capacity_bytes, expected_unused);
    EXPECT_EQ(report.owners[0].bytes, sizeof(P::details::GeometryIdentityBacking) + backing->native->storage_bytes());
    EXPECT_EQ(report.owners[0].mode, ArithmeticMode::Exact);
    EXPECT_EQ(backing.use_count(), count_before);
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
}

TEST(GeometryRetentionCensus, PlanarPayloadSeparatesPolygonMetadataAndCapacity)
{
    const std::array<P::GeometryPoint, 3> triangle{P::GeometryPoint{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    const auto face = P::PlanarFace<ordinary>::make(triangle);
    const std::array roots{R::root(*face)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    const auto row = std::find_if(report.owners.begin(), report.owners.end(),
        [](const auto& owner) { return std::string_view(owner.kind) == "planar_payload"; });
    ASSERT_NE(row, report.owners.end());
    EXPECT_EQ(row->populated_bytes, face->points().size() * sizeof(P::PlanarFace<ordinary>::WidePoint));
    EXPECT_EQ(row->unused_capacity_bytes,
        (face->points().capacity() - face->points().size()) * sizeof(P::PlanarFace<ordinary>::WidePoint));
    EXPECT_EQ(row->bytes, face->exclusive_storage_bytes());
    EXPECT_GT(row->metadata_bytes, 0u);
}

TEST(GeometryRetentionCensus, SharedSourceSnapshotAndLedgerUseTheSamePlanarOwnerKeys)
{
    using Face = P::PlanarFace<ordinary>;
    const std::array<P::GeometryPoint, 3> triangle{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}};
    const auto source = Face::make(triangle);
    const auto snapshot = source;
    const auto tile_shell = std::make_shared<const Face>(*source);
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    ledger.retain_planar_face(source, used, cap);
    ledger.retain_planar_face(snapshot, used, cap);
    ledger.retain_planar_face(tile_shell, used, cap);
    const std::array roots{R::root(*source, 0), R::root(*snapshot, 1), R::root(*tile_shell, 2)};
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, ledgers, scratch.resource, {1000, 1000, true});
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(kind_count(report, "planar_wrapper"), 2u);
    EXPECT_EQ(kind_count(report, "planar_payload"), 1u);
    EXPECT_EQ(kind_count(report, "external_owner"), 0u);
    EXPECT_EQ(report.owner_union.bytes, ledger.payload_bytes());
    EXPECT_EQ(report.intersection.bytes, ledger.payload_bytes());
    EXPECT_EQ(report.ledger_payload_admission, ledger.payload_bytes());
    EXPECT_EQ(report.uncovered_required_owners, 0u);
    const auto polygon = std::find_if(report.owners.begin(), report.owners.end(),
        [](const auto& owner) { return std::string_view(owner.kind) == "planar_payload"; });
    ASSERT_NE(polygon, report.owners.end());
    EXPECT_EQ(polygon->bytes, source->exclusive_storage_bytes());
    EXPECT_EQ(polygon->root_groups, 7u);
    EXPECT_TRUE(polygon->application);
    EXPECT_TRUE(polygon->ledger);
}

TEST(GeometryRetentionCensus, StrongOwnerCyclesAreIncompleteWithoutFollowingBorrowedPlaneContexts)
{
    using Context = P::PointSnapshotContext<ordinary>;
    auto self = std::make_shared<Context>();
    self->frozen = true;
    self->dependencies.push_back(self);
    {
        const std::array roots{R::root(*self)};
        Scratch scratch;
        const auto report = R::inspect_retention(roots, {}, scratch.resource);
        EXPECT_EQ(report.status, R::Status::InvalidGraph);
    }
    self->dependencies.clear();
    auto first = std::make_shared<Context>(), second = std::make_shared<Context>();
    first->frozen = second->frozen = true;
    first->dependencies.push_back(second);
    second->dependencies.push_back(first);
    {
        const std::array roots{R::root(*first)};
        Scratch scratch;
        const auto report = R::inspect_retention(roots, {}, scratch.resource);
        EXPECT_EQ(report.status, R::Status::InvalidGraph);
    }
    first->dependencies.clear();
    second->dependencies.clear();
    auto support = P::details::GeometryIdentityAccess::support_axis(0, 1);
    const auto& backing = P::details::GeometryIdentityAccess::backing(support);
    auto* malformed_identity = const_cast<P::details::GeometryIdentityBacking*>(backing.get());
    malformed_identity->parent = backing;
    {
        const std::array roots{R::root(support)};
        Scratch scratch;
        const auto report = R::inspect_retention(roots, {}, scratch.resource);
        EXPECT_EQ(report.status, R::Status::InvalidGraph);
    }
    malformed_identity->parent.reset();
    auto storage = std::make_shared<P::details::GeometryStorage<ordinary>>();
    auto ancestry = std::make_shared<P::details::Ancestry>();
    ancestry->parent = ancestry;
    storage->ancestry = ancestry;
    const auto geometry = Access::geometry(storage);
    {
        const std::array roots{R::root(*geometry)};
        Scratch scratch;
        const auto report = R::inspect_retention(roots, {}, scratch.resource);
        EXPECT_EQ(report.status, R::Status::InvalidGraph);
    }
    ancestry->parent.reset();
}

TEST(GeometryRetentionCensus, RequiredPointCyclesAreIncompleteAndUnusedRowsAreOutsideRequiredClosureValidation)
{
    auto source = native();
    Builder builder({nullptr, uncharged});
    const auto alias = builder.export_alias({1, 2, 3}, {}, P::details::RetentionAccess::provenance(source), {});
    const auto alias_ref = builder.import(alias);
    const auto numerical = builder.numerical({7, 8, 9}, {});
    const auto numerical_ref = builder.import(numerical);
    const auto original = builder.freeze();
    auto malformed = std::make_shared<P::PointSnapshotContext<ordinary>>(*original);
    malformed->export_aliases[0].source = alias_ref; // Valid slot, cyclic semantic source.
    {
        const std::array roots{R::root(*malformed)};
        Scratch scratch;
        const auto report = R::inspect_retention(roots, {}, scratch.resource);
        EXPECT_EQ(report.status, R::Status::InvalidGraph);
    }
    // The numerical point requires no export evidence. Retained unused record
    // edges are outside the explicitly named required-record dependency metric.
    const auto point = public_point(P::resolve_point(malformed.get(), numerical_ref));
    const std::array roots{R::root(point)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(report.required_records, 1u);
    EXPECT_EQ(report.maximum_required_point_dependency_depth, 0u);
}

TEST(GeometryRetentionCensus, ContextDepthCountsEveryDependencyAndIsIndependentOfRootOrderAndSharedSuffixes)
{
    using Context = P::PointSnapshotContext<ordinary>;
    auto leaf = std::make_shared<Context>(), shared = std::make_shared<Context>();
    auto left = std::make_shared<Context>(), right = std::make_shared<Context>(), top = std::make_shared<Context>();
    shared->dependencies.push_back(leaf);
    left->dependencies.push_back(shared);
    right->dependencies.push_back(shared);
    top->dependencies = {left, right};
    for (const auto* context : {leaf.get(), shared.get(), left.get(), right.get(), top.get()})
        const_cast<Context*>(context)->frozen = true;
    const std::array first_roots{R::root(*top, 0), R::root(*shared, 1)};
    const std::array reverse_roots{R::root(*shared, 1), R::root(*top, 0)};
    Scratch first_scratch, second_scratch;
    const auto first_report = R::inspect_retention(first_roots, {}, first_scratch.resource);
    const auto second_report = R::inspect_retention(reverse_roots, {}, second_scratch.resource);
    ASSERT_TRUE(first_report.complete()) << first_report.detail;
    ASSERT_TRUE(second_report.complete()) << second_report.detail;
    EXPECT_EQ(first_report.maximum_context_dependency_depth, 3u);
    EXPECT_EQ(second_report.maximum_context_dependency_depth, 3u);
    EXPECT_EQ(first_report.owner_union.owners, 5u);
    EXPECT_EQ(first_report.owner_union.bytes, second_report.owner_union.bytes);
    EXPECT_EQ(first_report.group_overlap_bytes, second_report.group_overlap_bytes);
    std::vector<std::shared_ptr<Context>> chain;
    for (std::size_t i = 0; i <= 64; ++i)
    {
        auto context = std::make_shared<Context>();
        context->frozen = true;
        if (i)
            context->dependencies.push_back(chain.back());
        chain.push_back(std::move(context));
    }
    const std::array chain_roots{R::root(*chain.back())};
    Scratch chain_scratch;
    const auto chain_report = R::inspect_retention(chain_roots, {}, chain_scratch.resource);
    ASSERT_TRUE(chain_report.complete()) << chain_report.detail;
    EXPECT_EQ(chain_report.maximum_context_dependency_depth, 64u);
}

TEST(GeometryRetentionCensus, GenuineBoundedGeometryAndPublicationKeepModeRoleSlotAndObservationContracts)
{
    constexpr auto bounded = ArithmeticMode::BoundedCertification;
    auto geometry = P::ConvexCellGeometry<bounded>::box({0, 0, 0}, {1, 1, 1});
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    auto publication = geometry->export_points(ledger, used, cap);
    ASSERT_EQ(publication.points().size(), 8u);
    std::optional<P::Point<bounded>> copied(publication.points()[0]);
    const auto point_view = P::details::RetentionAccess::provenance(*copied);
    const auto context_weak = point_view.context()->weak_from_this();
    const auto equal_ordinary = native(copied->value);
    const auto& implementation = Access::implementation(*geometry->dynamic_geometry());
    const auto* storage = static_cast<const P::details::GeometryStorage<bounded>*>(implementation.get());
    const auto* cache = storage->point_export_cache.get();
    ASSERT_NE(cache, nullptr);
    const auto context_count = context_weak.use_count(), geometry_count = geometry.use_count();
    const auto publication_count = cache->bounded_root.use_count();
    const auto dynamic_expired = cache->dynamic_root.expired();
    const auto admission = used, ledger_owners = ledger.owner_count();
    const auto payload = ledger.payload_bytes(), bookkeeping = ledger.bookkeeping_bytes();
    const auto statistics = P::geometry_statistics();
    const auto rounding = std::fegetround(), flags = std::fetestexcept(FE_ALL_EXCEPT);
    const std::array roots{
        R::root(*geometry, 0), R::root(publication, 0), R::root(*copied, 1), R::root(equal_ordinary, 2)};
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, ledgers, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    std::array<std::size_t, 4> bounded_records{}, ordinary_records{};
    for (const auto& context : report.contexts)
        for (std::size_t role = 0; role < bounded_records.size(); ++role)
            (context.mode == bounded ? bounded_records : ordinary_records)[role] += context.required_records[role];
    EXPECT_GE(bounded_records[static_cast<unsigned>(P::PointProvenanceRole::Native)], 8u);
    EXPECT_GE(bounded_records[static_cast<unsigned>(P::PointProvenanceRole::Export)], 8u);
    EXPECT_EQ(ordinary_records, (std::array<std::size_t, 4>{1, 0, 0, 0}));
    EXPECT_EQ(report.required_records, report.retained_records);
    EXPECT_EQ(report.uncovered_required_owners, 0u);
    EXPECT_EQ(context_weak.use_count(), context_count);
    EXPECT_EQ(geometry.use_count(), geometry_count);
    EXPECT_EQ(cache->bounded_root.use_count(), publication_count);
    EXPECT_EQ(cache->dynamic_root.expired(), dynamic_expired);
    EXPECT_EQ(used, admission);
    EXPECT_EQ(ledger.owner_count(), ledger_owners);
    EXPECT_EQ(ledger.payload_bytes(), payload);
    EXPECT_EQ(ledger.bookkeeping_bytes(), bookkeeping);
    EXPECT_EQ(P::geometry_statistics().floating_operations, statistics.floating_operations);
    EXPECT_EQ(P::geometry_statistics().interval_operations, statistics.interval_operations);
    EXPECT_EQ(P::geometry_statistics().constructions, statistics.constructions);
    EXPECT_EQ(P::geometry_statistics().runtime_dispatches, statistics.runtime_dispatches);
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    publication = {};
    ledger = P::GeometryAllocationLedger{};
    geometry.reset();
    EXPECT_FALSE(context_weak.expired()); // The copied Bounded point owns its publication provenance.
    copied.reset();
    EXPECT_TRUE(context_weak.expired()); // Report and raw manifest retain no owners.
}

TEST(GeometryRetentionCensus, PopulatedClipRetainedPointsSeedAllRecipeDependenciesBeyondPlaneVertices)
{
    Builder native_builder({nullptr, uncharged});
    const auto a = native_builder.native_source({0, 0, 0}, {}, {});
    const auto b = native_builder.native_source({1, 0, 0}, {}, {});
    const auto c = native_builder.native_source({0, 1, 0}, {}, {});
    const auto d = native_builder.native_source({1, 1, 0}, {}, {});
    const auto plane_ref = native_builder.import(a);
    const auto native_context = native_builder.freeze();
    Builder generated_builder({nullptr, uncharged});
    const std::array edges{std::array{a, b}, std::array{c, d}};
    const auto generated = generated_builder.generated({0.5L, 0.5L, 0}, {}, edges, {});
    const auto retained = generated_builder.import(generated);
    const auto generated_context = generated_builder.freeze();
    auto plane = std::make_shared<P::details::StoredPlane<ordinary>>();
    plane->point_context = native_context.get();
    plane->source_vertices.push_back(plane_ref); // Only a; b/c/d must come through the retained recipe.
    auto source = std::make_shared<P::details::GeometryStorage<ordinary>>();
    auto source_recipe =
        std::make_shared<P::details::GeometryConstruction>(P::details::GeometryConstruction::Source{source});
    auto clip = std::make_shared<P::details::GeometryClipConstruction>();
    clip->parent = source_recipe;
    clip->plane = plane;
    clip->point_context = generated_context;
    clip->retained_points.push_back(P::erase_point_ref(retained));
    auto result = std::make_shared<P::details::GeometryStorage<ordinary>>();
    result->construction =
        std::make_shared<P::details::GeometryConstruction>(P::details::GeometryConstruction::Clip{clip, false});
    const auto geometry = Access::geometry(result);
    const auto context_count = generated_context.use_count(), plane_count = plane.use_count();
    const auto statistics = P::geometry_statistics();
    const auto rounding = std::fegetround(), flags = std::fetestexcept(FE_ALL_EXCEPT);
    const std::array roots{R::root(*geometry)};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, {}, scratch.resource);
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(report.required_records, 5u);
    EXPECT_EQ(report.maximum_required_point_dependency_depth, 1u);
    EXPECT_EQ(report.maximum_context_dependency_depth, 1u);
    EXPECT_EQ(kind_count(report, "clip"), 1u);
    EXPECT_EQ(kind_count(report, "recipe"), 2u);
    EXPECT_EQ(kind_count(report, "geometry"), 2u);
    std::array<std::size_t, 4> required{};
    std::size_t required_edges = 0;
    for (const auto& context : report.contexts)
    {
        for (std::size_t role = 0; role < required.size(); ++role)
            required[role] += context.required_records[role];
        required_edges += context.required_edges;
    }
    EXPECT_EQ(required, (std::array<std::size_t, 4>{4, 1, 0, 0}));
    EXPECT_EQ(required_edges, 2u);
    EXPECT_EQ(report.uncovered_required_owners, 0u);
    EXPECT_EQ(generated_context.use_count(), context_count);
    EXPECT_EQ(plane.use_count(), plane_count);
    EXPECT_EQ(P::geometry_statistics().constructions, statistics.constructions);
    EXPECT_EQ(P::geometry_statistics().floating_operations, statistics.floating_operations);
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
}

TEST(GeometryRetentionCensus, LedgerTrackingIncludesLazyCertificateStorageMeasuredByRawAllocations)
{
    namespace D = P::details;
    namespace G = D::geometry_demand;
    const auto owner = std::make_shared<const std::array<unsigned char, 32>>();
    const auto other = std::make_shared<const std::array<unsigned char, 32>>();
    G::GeometryOwnerManifest first, second;
    P::GeometryAllocationLedger ledger;
    std::size_t used = 37;
    pipeline_trace::Summary actual;
    bool overflow = false;
    {
        pipeline_trace::begin();
        struct Stop
        {
            ~Stop() { pipeline_trace::recording.store(false); }
        } stop;
        // Both terminal graphs are closed. Keeping two rows makes every
        // created index allocation reachable from its returned manifest.
        D::GeometryOwnerGraphEditor one({}, 0, used, cap, &ledger);
        one.external(owner, sizeof(*owner), true);
        one.external(other, sizeof(*other), true);
        first = one.finish();
        D::GeometryOwnerGraphEditor two({}, 0, used, cap, &ledger);
        two.external(owner, sizeof(*owner), true);
        two.external(other, sizeof(*other), true);
        second = two.finish();
        actual = pipeline_trace::summary();
        overflow = pipeline_trace::overflow.load();
    }
    ASSERT_FALSE(overflow);
    ASSERT_GT(actual.live_allocations, 1u);
    // Remove the actual manifest allocations from the raw total. The
    // remaining live C++ extents are only the ledger's own tracking storage.
    pipeline_trace::prepare_census();
    std::array<std::uint64_t, 16> seen{};
    std::size_t count = 0, metadata_bytes = 0;
    const auto mark = [&](const void* address, std::size_t)
    {
        const auto* extent = pipeline_trace::mark(address, 1);
        ASSERT_TRUE(extent);
        if (std::find(seen.begin(), seen.begin() + count, extent->serial) == seen.begin() + count)
        {
            ASSERT_LT(count, seen.size());
            seen[count++] = extent->serial;
            metadata_bytes += extent->bytes;
        }
    };
    first.visit_payload(mark);
    second.visit_payload(mark);
    const std::array roots{
        R::external_root(owner.get(), sizeof(*owner)), R::external_root(other.get(), sizeof(*other))};
    const std::array<const P::GeometryAllocationLedger*, 1> ledgers{&ledger};
    Scratch scratch;
    const auto report = R::inspect_retention(roots, ledgers, scratch.resource, {1000, 1000, true});
    ASSERT_TRUE(report.complete()) << report.detail;
    EXPECT_EQ(report.ledger_tracking_bytes, actual.all_live - metadata_bytes);
}
