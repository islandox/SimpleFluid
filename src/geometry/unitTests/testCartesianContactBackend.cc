#include "cartesian_incidence_test_helpers.hh"
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/mesh/AMR/CartesianContactBackend.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include "utils/testing_environment.hh"
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <vector>

namespace
{
using namespace cartesian_incidence_test;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

std::vector<AdaptiveLeaf> uniform_leaves(const AMRGrid& value, unsigned level = 0)
{
    std::vector<AdaptiveLeaf> result;
    const auto scale = std::uint64_t{1} << level;
    for (std::uint64_t z = 0; z != value.root_blocks[2] * value.block_cells[2] * scale; ++z)
        for (std::uint64_t y = 0; y != value.root_blocks[1] * value.block_cells[1] * scale; ++y)
            for (std::uint64_t x = 0; x != value.root_blocks[0] * value.block_cells[0] * scale; ++x)
                result.push_back(
                    cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{level, {x, y, z}}}, {'d'}));
    return result;
}

std::shared_ptr<CartesianContactState> build_state(
    const AdaptiveProviderKernel& kernel, std::span<const AdaptiveLeaf> native)
{
    auto budget = std::make_shared<Budget>(Budget{ample});
    return CartesianContactState::build(kernel, native, 0, *budget, budget, 1);
}

void compare_evaluated_grading(const AMRGrid& value, bool require_capability)
{
    const auto native = uniform_leaves(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    const auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, *budget, budget);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto capability = issue_cartesian_contact_capability(*kernel, *source);
    if (!capability)
    {
        EXPECT_FALSE(require_capability) << "Required native arithmetic fixture was declined";
        EXPECT_FALSE(build_state(*kernel, native));
        return;
    }
    CartesianContactBackend backend(std::move(*capability), 0);
    std::vector<ContactRecord> records;
    std::size_t trace_records = 0, transient_peak = 0;
    for (const auto& leaf : native)
    {
        IdentityBudget identities;
        records.push_back(prepare_contact(leaf, *kernel, *budget, identities, trace_records, transient_peak));
    }
    for (std::size_t i = 0; i != native.size(); ++i)
        for (std::size_t j = i + 1; j != native.size(); ++j)
        {
            const auto& a = std::get<CartesianCellAncestry>(native[i].key.ancestry);
            const auto& b = std::get<CartesianCellAncestry>(native[j].key.ancestry);
            bool touches = true;
            for (std::size_t axis = 0; axis != 3; ++axis)
                touches = touches && (a.index[axis] > b.index[axis] ? a.index[axis] - b.index[axis]
                                                                    : b.index[axis] - a.index[axis]) <= 1;
            if (!touches)
                continue;
            ContactClosureView first, second;
            ASSERT_TRUE(backend.closure_views(native[i].key, native[j].key, first, second));
            const auto oracle_first = closure_view(records[i], 0), oracle_second = closure_view(records[j], 0);
            for (std::size_t axis = 0; axis != 3; ++axis)
            {
                EXPECT_EQ(first.spans[axis], oracle_first.spans[axis]);
                EXPECT_EQ(second.spans[axis], oracle_second.spans[axis]);
            }
            for (const auto ratio : {1.0, 1.25, 2.0, 4.0})
            {
                // The public halo currently accepts maximum_trace_jump == 1 only.
                ClosureRequirements actual, expected;
                grade_contact_spans(first, second, ratio, 0, actual);
                grade_contact_spans(oracle_first, oracle_second, ratio, 0, expected);
                grade_contact_trace_point(first, second, 1, 0, point(0), point(0), actual);
                grade_contact_trace_point(oracle_first, oracle_second, 1, 0, point(0), point(0), expected);
                EXPECT_EQ(actual.first, expected.first) << "ratio=" << ratio;
                EXPECT_EQ(actual.second, expected.second) << "ratio=" << ratio;
            }
        }
}

TEST(CartesianContactBackendTest, SixCellAxisUsesRepresentedAdjacentWidths)
{
    auto value = grid();
    value.root_blocks = {1, 1, 1};
    value.block_cells = {6, 2, 2};
    value.origin = {0, 0, 0};
    value.extent = {1, 1, 1};
    compare_evaluated_grading(value, true);

    const auto native = uniform_leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto state = build_state(*kernel, native);
    ASSERT_TRUE(state);
    AdaptiveClosureOptions options;
    options.maximum_span_ratio = 1;
    state->evaluate(options);
    std::vector<AdaptiveMark> marks;
    Budget marks_budget{ample};
    cartesian_detail::Allocation marks_admission(marks_budget);
    state->collect_marks(marks, marks_admission);
    const AdaptiveCellKey wider{value.region_id, CartesianCellAncestry{0, {2, 0, 0}}};
    ASSERT_NE(std::ranges::find(marks, wider, &AdaptiveMark::key), marks.end());
}

TEST(CartesianContactBackendTest, AsymmetricEvaluatedSpansMatchExplicitGradingOrDecline)
{
    for (const auto dimensions : {std::array<std::uint64_t, 3>{6, 2, 2}, {2, 6, 2}, {2, 2, 10}})
    {
        auto value = grid();
        value.block_cells = dimensions;
        value.origin = {0.125, -0.25, 0.5};
        value.extent = {1, 0.75, 2};
        compare_evaluated_grading(value, false);
    }
    auto dyadic = grid();
    dyadic.block_cells = {4, 2, 2};
    dyadic.origin = {0.125, -0.25, 0.5};
    dyadic.extent = {2, 0.75, 1};
    compare_evaluated_grading(dyadic, true);
}

TEST(CartesianContactBackendTest, SourceKernelIdentityIncludesAllGridFacts)
{
    auto original = grid();
    original.block_cells[0] = 4;
    const auto native = uniform_leaves(original);
    auto budget = std::make_shared<Budget>(Budget{ample});
    const auto source = CartesianIncidenceSource::make(original, native, {'d'}, 0, 1, {}, *budget, budget);
    std::vector<AMRGrid> foreign(6, original);
    foreign[0].origin[0] = 0.125;
    foreign[1].extent[1] = 2;
    foreign[2].root_blocks[0] = 2;
    foreign[2].block_cells[0] = 2; // Same total cells, different native block recipe.
    foreign[3].max_level = 2;
    foreign[4].material_id += 1;
    foreign[5].region_id += 1;
    for (const auto& value : foreign)
    {
        const auto kernel = cartesian_adaptive_kernel(value);
        EXPECT_FALSE(issue_cartesian_contact_capability(*kernel, *source));
    }
}

TEST(CartesianContactBackendTest, ProductionBuildAuthenticatesRecipesAndLevels)
{
    const auto value = grid();
    const auto kernel = cartesian_adaptive_kernel(value);
    const auto native = uniform_leaves(value);
    ASSERT_TRUE(build_state(*kernel, native));
    auto tampered = native;
    tampered.front().native_cell += 1;
    EXPECT_FALSE(build_state(*kernel, tampered));
    tampered = native;
    tampered.front().native_recipe.reset();
    EXPECT_FALSE(build_state(*kernel, tampered));
    tampered = native;
    for (auto& leaf : tampered)
        std::get<CartesianCellAncestry>(leaf.key.ancestry).level = 64;
    EXPECT_FALSE(build_state(*kernel, tampered));
}

TEST(CartesianContactBackendTest, CapabilityDeclinesChangedArithmeticEnvironment)
{
    RestoreEnvironment restore;
    const auto value = grid();
    const auto native = uniform_leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    const auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, *budget, budget);
    ASSERT_TRUE(issue_cartesian_contact_capability(*kernel, *source));
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    EXPECT_FALSE(issue_cartesian_contact_capability(*kernel, *source));
    EXPECT_FALSE(build_state(*kernel, native));
}

TEST(CartesianContactBackendTest, EligibilityBoundsProofWorkBeforeCoordinateScanning)
{
    auto value = grid();
    value.max_level = 40;
    EXPECT_FALSE(cartesian_contact_arithmetic_eligible(value, 40));
}

struct ThrowingCopy
{
    static inline bool fail = false;
    ThrowingCopy() = default;
    ThrowingCopy(const ThrowingCopy&)
    {
        if (fail)
            throw std::runtime_error("Replacement buffer copy failed.");
    }
};
TEST(CartesianContactBackendTest, VectorAdmissionRefundsFailedReplacementAndKeepsOldBuffer)
{
    Budget budget{ample};
    std::vector<ThrowingCopy> values;
    budget.append_capacity(values);
    values.emplace_back();
    const auto* old = values.data();
    const auto retained = budget.used;
    ThrowingCopy::fail = true;
    EXPECT_THROW(budget.append_capacity(values), std::runtime_error);
    ThrowingCopy::fail = false;
    EXPECT_EQ(values.data(), old);
    EXPECT_EQ(values.size(), 1u);
    EXPECT_EQ(budget.used, retained);
    EXPECT_GT(budget.peak, retained); // Old and replacement were admitted together.
}

TEST(CartesianContactBackendTest, BorrowedVisitorAndAdmissionIncludeEveryRetainedField)
{
    auto value = grid();
    value.block_cells = {6, 2, 2};
    auto native = uniform_leaves(value);
    std::erase_if(native,
        [](const AdaptiveLeaf& leaf) { return std::get<CartesianCellAncestry>(leaf.key.ancestry).index[0] >= 3; });
    const auto kernel = cartesian_adaptive_kernel(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    auto state = CartesianContactState::build(*kernel, native, 0, *budget, budget, 1);
    ASSERT_TRUE(state);
    {
        cartesian_detail::Allocation scratch(*budget);
        std::vector<AdaptiveCellKey> boundary;
        state->boundary_keys(boundary, scratch);
        ASSERT_FALSE(boundary.empty());
        for (const auto& key : boundary)
            state->install_owner(key, 1);
    }
    AdaptiveClosureOptions options;
    options.maximum_span_ratio = 1;
    state->evaluate(options);
    std::map<ContactPayloadRole, std::size_t> payloads, counts;
    std::size_t payload = 0;
    state->visit_borrowed_payload(
        [&](const ContactPayloadAllocation& allocation)
        {
            payload += allocation.payload_bytes;
            payloads[allocation.identity.role] += allocation.payload_bytes;
            counts[allocation.identity.role] += allocation.live_count;
        });
    for (const auto role : {GeometryContactPayloadRole::CartesianContactStateObject,
             GeometryContactPayloadRole::CartesianContactKeys, GeometryContactPayloadRole::CartesianContactIndex,
             GeometryContactPayloadRole::CartesianContactRevisions, GeometryContactPayloadRole::CartesianContactFrame,
             GeometryContactPayloadRole::CartesianContactDomain, GeometryContactPayloadRole::CartesianContactTraceStem,
             GeometryContactPayloadRole::CartesianContactRemoteOwners,
             GeometryContactPayloadRole::CartesianContactFailures, GeometryContactPayloadRole::CartesianContactMarks})
        EXPECT_GT(payloads[role], 0u);
    EXPECT_EQ(counts[GeometryContactPayloadRole::CartesianContactKeys], native.size());
    EXPECT_EQ(counts[GeometryContactPayloadRole::CartesianContactIndex], native.size());
    EXPECT_EQ(counts[GeometryContactPayloadRole::CartesianContactRevisions], native.size());
    EXPECT_EQ(counts[GeometryContactPayloadRole::CartesianContactRemoteOwners], state->remote_cells());
    EXPECT_EQ(payload + state->admission_allowance_bytes(), state->metadata_bytes());
    EXPECT_EQ(budget->used, state->metadata_bytes());
    state.reset();
    EXPECT_EQ(budget->used, 0u);
}

void check_retained_cap(std::uint64_t side)
{
    auto value = grid();
    value.block_cells = {side, side, side};
    const auto native = uniform_leaves(value);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    auto state = CartesianContactState::build(*kernel, native, 0, *budget, budget, 1);
    ASSERT_TRUE(state);
    state->evaluate(AdaptiveClosureOptions{});
    const auto retained = state->metadata_bytes();
    ASSERT_LT(retained, ample);
    std::size_t payload = 0;
    state->visit_borrowed_payload(
        [&](const ContactPayloadAllocation& allocation) { payload += allocation.payload_bytes; });
    EXPECT_EQ(payload + state->admission_allowance_bytes(), retained);
    EXPECT_EQ(budget->used, retained);
    ::testing::Test::RecordProperty("retained_bytes", std::to_string(retained));
    ::testing::Test::RecordProperty("payload_bytes", std::to_string(payload));
    ::testing::Test::RecordProperty("construction_peak_bytes", std::to_string(budget->peak));
    state.reset();
    EXPECT_EQ(budget->used, 0u);
    auto below = std::make_shared<Budget>(Budget{retained - 1});
    EXPECT_THROW((void) CartesianContactState::build(*kernel, native, 0, *below, below, 1), std::invalid_argument);
    EXPECT_EQ(below->used, 0u);
}

TEST(CartesianContactBackendTest, N8RetainedCensusHonors64MiBCapAndRollback)
{
    check_retained_cap(8);
}
TEST(CartesianContactBackendTest, N32RetainedCensusHonors64MiBCapAndRollback)
{
    check_retained_cap(32);
}

TEST(CartesianContactBackendTest, StandaloneGraderJoinsPositiveSpanAndDeepTraceViolations)
{
    // These independent expectations exercise joins across axes and corners.
    // They do not expand the public halo's maximum_trace_jump == 1 contract.
    const std::array<long double, 3> wide{8, 1, 1}, narrow{1, 8, 1};
    const std::array<std::uint64_t, 3> factors{1, 2, 0};
    std::array<AdaptiveTraceFace, 2> coarse{
        {{{'t'}, 0, AdaptiveSplit::Base, {}, true}, {{'u'}, 0, AdaptiveSplit::Axial, {}, true}}};
    std::array<AdaptiveTraceFace, 2> fine{
        {{{'t'}, 4, AdaptiveSplit::Base, {}, true}, {{'u'}, 4, AdaptiveSplit::Axial, {}, true}}};
    const ContactClosureView first{0, wide, factors, coarse, true};
    const ContactClosureView second{0, narrow, factors, fine, true};
    for (const auto ratio : {1.0, 1.25, 2.0, 4.0})
    {
        ClosureRequirements result;
        grade_contact_spans(first, second, ratio, 0, result);
        EXPECT_EQ(result.first, AdaptiveSplit::Base);
        EXPECT_EQ(result.second, AdaptiveSplit::Axial);
        grade_contact_trace_point(first, second, 2, 0, point(0), point(0), result);
        EXPECT_EQ(result.first, AdaptiveSplit::Isotropic);
        EXPECT_EQ(result.second, AdaptiveSplit::Axial);
    }
    for (const unsigned jump : {1u, 2u, 3u, 4u})
    {
        ClosureRequirements result;
        grade_contact_trace_point(first, second, jump, 0, point(0), point(0), result);
        if (jump < 4)
            EXPECT_EQ(result.first, AdaptiveSplit::Isotropic);
        else
            EXPECT_FALSE(result.first);
        EXPECT_FALSE(result.second);
    }
}

TEST(CartesianContactBackendTest, CornerSpecificTraceRequirementsJoinAfterDifferentPoints)
{
    const auto shape = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto lower = point(0, 0, 0), upper = point(1, 1, 1);
    decltype(shape->face_geometry(0)) lower_face, upper_face;
    for (std::size_t face = 0; face < shape->num_faces(); ++face)
    {
        const auto candidate = shape->face_geometry(face);
        if (candidate->contains_point(lower) && !candidate->contains_point(upper))
            lower_face = candidate;
        if (candidate->contains_point(upper) && !candidate->contains_point(lower))
            upper_face = candidate;
    }
    ASSERT_TRUE(lower_face);
    ASSERT_TRUE(upper_face);
    const std::array<long double, 3> spans{1, 1, 1};
    const std::array<std::uint64_t, 3> factors{1, 2, 0};
    const std::array<AdaptiveTraceFace, 2> coarse{
        {{{'t'}, 0, AdaptiveSplit::Base, lower_face, true}, {{'u'}, 0, AdaptiveSplit::Axial, upper_face, true}}};
    const std::array<AdaptiveTraceFace, 2> fine{
        {{{'t'}, 4, AdaptiveSplit::Base, {}, true}, {{'u'}, 4, AdaptiveSplit::Axial, {}, true}}};
    const ContactClosureView first{0, spans, factors, coarse, false}, second{0, spans, factors, fine, true};
    ClosureRequirements required;
    grade_contact_trace_point(first, second, 2, 0, lower, lower, required);
    EXPECT_EQ(required.first, AdaptiveSplit::Base);
    EXPECT_FALSE(required.second);
    grade_contact_trace_point(first, second, 2, 0, upper, upper, required);
    EXPECT_EQ(required.first, AdaptiveSplit::Isotropic);
    EXPECT_FALSE(required.second);
}

TEST(CartesianContactBackendTest, CapabilityGrantedForUniformNonperiodicCompleteSource)
{
    auto value = grid();
    auto native = leaves(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    const auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, *budget, budget);
    const auto kernel = cartesian_adaptive_kernel(value);
    const auto capability = issue_cartesian_contact_capability(*kernel, *source);
    ASSERT_TRUE(capability.has_value());
    EXPECT_EQ(capability->counts, (std::array<std::uint64_t, 3>{2, 2, 2}));
    EXPECT_EQ(capability->level, 0u);
    EXPECT_EQ(capability->region, value.region_id);
    EXPECT_EQ(capability->factors, (std::array<std::uint64_t, 3>{0, 0, 0}));
    EXPECT_EQ(capability->frame, kernel->descriptor());
    for (std::size_t axis = 0; axis != 3; ++axis)
        EXPECT_DOUBLE_EQ(static_cast<double>(capability->cell_spans[axis]), 0.5);
}

TEST(CartesianContactBackendTest, ViewsMatchExplicitRecordsAndGradeIdentically)
{
    auto value = grid();
    auto native = leaves(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    const auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, *budget, budget);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto capability = *issue_cartesian_contact_capability(*kernel, *source);
    CartesianContactBackend backend(std::move(capability), 0);
    std::vector<ContactRecord> records;
    std::size_t trace_records = 0, transient_peak = 0;
    for (const auto& leaf : native)
    {
        IdentityBudget identities;
        records.push_back(prepare_contact(leaf, *kernel, *budget, identities, trace_records, transient_peak));
    }
    ASSERT_EQ(records.size(), native.size());
    for (std::size_t i = 0; i != native.size(); ++i)
        for (std::size_t j = i + 1; j != native.size(); ++j)
        {
            ContactClosureView view_a, view_b;
            ASSERT_TRUE(backend.closure_views(native[i].key, native[j].key, view_a, view_b));
            const auto record_a = closure_view(records[i], 0);
            const auto record_b = closure_view(records[j], 0);
            for (std::size_t axis = 0; axis != 3; ++axis)
            {
                EXPECT_EQ(view_a.factors[axis], record_a.factors[axis]);
                EXPECT_EQ(view_b.factors[axis], record_b.factors[axis]);
                EXPECT_LE(std::fabs(view_a.spans[axis] - record_a.spans[axis]),
                    std::fabs(record_a.spans[axis]) * 4 * std::numeric_limits<long double>::epsilon());
            }
            ASSERT_EQ(view_a.trace_faces.size(), 1u);
            ASSERT_EQ(record_a.trace_faces.size(), 1u);
            EXPECT_EQ(view_a.trace_faces[0].depth, record_a.trace_faces[0].depth);
            EXPECT_EQ(view_a.trace_faces[0].split, record_a.trace_faces[0].split);
            EXPECT_EQ(view_a.trace_faces[0].terminal, record_a.trace_faces[0].terminal);
            EXPECT_EQ(view_a.trace_faces[0].stem, record_a.trace_faces[0].stem);
            EXPECT_TRUE(view_a.traces_point_independent);
            for (const double ratio : {1.0, 1.25, 2.0})
            {
                ClosureRequirements via_backend, via_record;
                grade_contact_spans(view_a, view_b, ratio, 0, via_backend);
                grade_contact_spans(record_a, record_b, ratio, 0, via_record);
                EXPECT_EQ(via_backend.first, via_record.first);
                EXPECT_EQ(via_backend.second, via_record.second);
            }
        }
}

TEST(CartesianContactBackendTest, CapabilityRejectsPeriodicMixedLevelAndForeignCells)
{
    auto value = grid();
    auto native = leaves(value);
    auto budget = std::make_shared<Budget>(Budget{ample});
    const auto source = CartesianIncidenceSource::make(value, native, {'d'}, 0, 1, {}, *budget, budget);
    const auto kernel = cartesian_adaptive_kernel(value);
    auto capability = *issue_cartesian_contact_capability(*kernel, *source);
    CartesianContactBackend backend(std::move(capability), 0);
    ContactClosureView view_a, view_b;
    EXPECT_FALSE(
        backend.closure_views(native[0].key, {value.region_id, CartesianCellAncestry{1, {0, 0, 0}}}, view_a, view_b));
    EXPECT_FALSE(
        backend.closure_views(native[0].key, {value.region_id, CartesianCellAncestry{0, {5, 0, 0}}}, view_a, view_b));
    EXPECT_FALSE(backend.closure_views(
        native[0].key, {value.region_id + 1, CartesianCellAncestry{0, {0, 0, 0}}}, view_a, view_b));

    auto periodic = grid();
    periodic.periodic[0] = true;
    auto periodic_budget = std::make_shared<Budget>(Budget{ample});
    const auto periodic_source =
        CartesianIncidenceSource::make(periodic, leaves(periodic), {'d'}, 0, 1, {}, *periodic_budget, periodic_budget);
    const auto periodic_kernel = cartesian_adaptive_kernel(periodic);
    EXPECT_FALSE(issue_cartesian_contact_capability(*periodic_kernel, *periodic_source).has_value());

    auto mixed = grid();
    auto mixed_budget = std::make_shared<Budget>(Budget{ample});
    const auto mixed_source =
        CartesianIncidenceSource::make(mixed, leaves(mixed, true), {'d'}, 0, 1, {}, *mixed_budget, mixed_budget);
    const auto mixed_kernel = cartesian_adaptive_kernel(mixed);
    EXPECT_FALSE(issue_cartesian_contact_capability(*mixed_kernel, *mixed_source).has_value());
}
} // namespace
