/** Independent grading decisions and owning-record compatibility checks. */
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include "utils/testing_environment.hh"
#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace AMR = SimpleFluid::Meshes::AMR;
namespace D = AMR::adaptive_detail;
namespace P = SimpleFluid::PLIC;
constexpr auto ordinary = SimpleFluid::ArithmeticMode::Ordinary;
using Cell = P::ConvexCellGeometry<ordinary>;
using Face = P::PlanarFace<ordinary>;
constexpr std::size_t ample = 16 * 1024 * 1024;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

P::PointIdentity<ordinary> point(long double x, long double y, long double z)
{
    return P::details::GeometryIdentityAccess::point({x, y, z});
}
P::PlanarFacePtr<ordinary> rectangle(double x)
{
    const std::array<P::GeometryPoint, 4> vertices{{{x, 0, 0}, {x, 2, 0}, {x, 2, 2}, {x, 0, 2}}};
    return Face::make(vertices);
}
class RecordKernel final : public AMR::AdaptiveProviderKernel
{
public:
    std::vector<AMR::AdaptiveTraceFace> trace_faces;
    bool foreign_trace = false;
    AMR::AdaptiveBytes descriptor() const override { return {'r'}; }
    P::ConvexGeometryPtr<ordinary> geometry(const AMR::AdaptiveLeaf& leaf) const override { return leaf.geometry; }
    std::vector<AMR::AdaptiveSurface> surfaces(const AMR::AdaptiveLeaf& leaf) const override { return leaf.surfaces; }
    AMR::AdaptiveTrace traces(const AMR::AdaptiveLeaf& leaf) const override
    {
        auto key = leaf.key;
        if (foreign_trace)
            ++key.region;
        return {std::move(key), trace_faces};
    }
    AMR::AdaptiveSplit closure_split(const AMR::AdaptiveLeaf&, std::size_t axis) const override
    {
        return axis == 2 ? AMR::AdaptiveSplit::Axial : AMR::AdaptiveSplit::Base;
    }
    AMR::AdaptiveSplitResult split(const AMR::AdaptiveLeaf&, AMR::AdaptiveSplit) const override
    {
        throw std::invalid_argument("Fixture has no split implementation.");
    }
};
AMR::AdaptiveLeaf source_leaf()
{
    AMR::AdaptiveLeaf leaf;
    leaf.key = {7, AMR::ConvexCellAncestry{19, {1, 3}}};
    leaf.domain = {'d', 'o', 'm'};
    leaf.geometry = Cell::box({7, 11, 17}, {11, 13, 18});
    return leaf;
}

TEST(AdaptiveClosureRecords, SpanThresholdIsStrictAndPreservesLongDoublePrecision)
{
    EXPECT_FALSE(D::span_exceeds_ratio(2.L, 1.L, 2.));
    EXPECT_FALSE(D::span_exceeds_ratio(1.L, 2.L, 2.));
    EXPECT_TRUE(D::span_exceeds_ratio(std::nextafter(2.L, 3.L), 1.L, 2.));
    EXPECT_FALSE(D::span_exceeds_ratio(std::nextafter(2.L, 1.L), 1.L, 2.));
    if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits)
    {
        const long double wider = 2.L + 0x1p-54L;
        ASSERT_GT(wider, 2.L);
        EXPECT_EQ(double(wider), 2.);
        EXPECT_TRUE(D::span_exceeds_ratio(wider, 1.L, 2.));
    }
}

TEST(AdaptiveClosureRecords, TraceProvenanceTerminalAndDepthHaveIndependentExpectedDecisions)
{
    AMR::AdaptiveTraceFace first{{'s'}, 0, AMR::AdaptiveSplit::Base, {}, true};
    AMR::AdaptiveTraceFace second{{'s'}, 1, AMR::AdaptiveSplit::Axial, {}, true};
    const auto location = point(0, 0, 0);
    auto result = D::trace_pair_requirements(first, second, location, location, 1);
    EXPECT_FALSE(result.first);
    EXPECT_FALSE(result.second);
    second.depth = 2;
    result = D::trace_pair_requirements(first, second, location, location, 1);
    EXPECT_EQ(result.first, AMR::AdaptiveSplit::Base);
    EXPECT_FALSE(result.second);
    result = D::trace_pair_requirements(second, first, location, location, 1);
    EXPECT_FALSE(result.first);
    EXPECT_EQ(result.second, AMR::AdaptiveSplit::Base);
    second.stem = {'t'};
    EXPECT_FALSE(D::trace_pair_requirements(first, second, location, location, 1).first);
    second.stem = {'s'};
    first.terminal = second.terminal = false;
    EXPECT_FALSE(D::trace_pair_requirements(first, second, location, location, 1).first);
    second.terminal = true;
    EXPECT_EQ(D::trace_pair_requirements(first, second, location, location, 1).first, AMR::AdaptiveSplit::Base);
}

TEST(AdaptiveClosureRecords, RestrictedTracesUseEachPhysicalPointIncludingEdgeInteriors)
{
    AMR::AdaptiveTraceFace first{{'s'}, 0, AMR::AdaptiveSplit::Base, rectangle(0), true};
    AMR::AdaptiveTraceFace second{{'s'}, 3, AMR::AdaptiveSplit::Axial, rectangle(4), true};
    // Distinct periodic physical images share an authority, but the support
    // membership query belongs to each physical face separately.
    const auto a = point(0, 0, 1), b = point(4, 0, 1);
    const auto valid = D::trace_pair_requirements(first, second, a, b, 1);
    EXPECT_EQ(valid.first, AMR::AdaptiveSplit::Base);
    EXPECT_FALSE(valid.second);
    EXPECT_FALSE(D::trace_pair_requirements(first, second, a, a, 1).first);
    EXPECT_FALSE(D::trace_pair_requirements(first, second, point(0, 0, 3), b, 1).first);
    EXPECT_FALSE(D::trace_pair_requirements(first, second, a, point(4, 3, 1), 1).first);
    EXPECT_EQ(
        D::trace_pair_requirements(first, second, point(0, 2, 2), point(4, 2, 2), 1).first, AMR::AdaptiveSplit::Base);
}

TEST(AdaptiveClosureRecords, ConflictingDirectionsJoinIsotropicallyAndStaticDemandRejects)
{
    D::ContactRecord contact;
    contact.factors = {1, 1, 2};
    std::optional<AMR::AdaptiveSplit> requirement;
    D::join_closure_requirement(contact, requirement, std::uint64_t(AMR::AdaptiveSplit::Base));
    EXPECT_EQ(requirement, AMR::AdaptiveSplit::Base);
    D::join_closure_requirement(contact, requirement, std::uint64_t(AMR::AdaptiveSplit::Base));
    EXPECT_EQ(requirement, AMR::AdaptiveSplit::Base);
    D::join_closure_requirement(contact, requirement, std::uint64_t(AMR::AdaptiveSplit::Axial));
    EXPECT_EQ(requirement, AMR::AdaptiveSplit::Isotropic);
    D::join_closure_requirement(contact, requirement, std::uint64_t(AMR::AdaptiveSplit::Base));
    EXPECT_EQ(requirement, AMR::AdaptiveSplit::Isotropic);
    EXPECT_THROW(D::join_closure_requirement(contact, requirement, 3), std::invalid_argument);
    contact.factors = {3, 3, 3};
    requirement.reset();
    EXPECT_THROW(D::join_closure_requirement(contact, requirement, 0), std::invalid_argument);
    EXPECT_FALSE(requirement);
    const AMR::AdaptiveTraceFace shallow{{'s'}, 0, AMR::AdaptiveSplit::Axial, {}, true};
    const AMR::AdaptiveTraceFace deep{{'s'}, 2, AMR::AdaptiveSplit::Base, {}, true};
    const auto trace = D::trace_pair_requirements(shallow, deep, point(0, 0, 0), point(0, 0, 0), 1);
    ASSERT_TRUE(trace.first);
    EXPECT_THROW(D::join_closure_requirement(contact, requirement, std::uint64_t(*trace.first)), std::invalid_argument);
}

TEST(AdaptiveClosureRecords, OwningPreparationMatchesLegacyRecordAndOutlivesSource)
{
    auto leaf = source_leaf();
    RecordKernel kernel;
    kernel.trace_faces = {
        {{'s'}, 3, AMR::AdaptiveSplit::Base, rectangle(7), true}, {{'a'}, 1, AMR::AdaptiveSplit::Axial, {}, false}};
    D::Budget legacy_budget{ample}, direct_budget{ample}, output_budget{ample};
    D::IdentityBudget identities;
    std::size_t legacy_traces = 0, direct_traces = 0, legacy_peak = 0, direct_peak = 0;
    const auto legacy = D::encode_contact(leaf, kernel, legacy_budget, legacy_traces, legacy_peak);
    const auto direct = D::prepare_contact(leaf, kernel, direct_budget, identities, direct_traces, direct_peak);
    EXPECT_EQ(direct.key, leaf.key);
    EXPECT_EQ(direct.domain, (AMR::AdaptiveBytes{'d', 'o', 'm'}));
    EXPECT_EQ(direct.spans, (std::array<long double, 3>{4, 2, 1}));
    EXPECT_EQ(direct.factors, (std::array<std::uint64_t, 3>{1, 1, 2}));
    EXPECT_EQ(direct_traces, 2u);
    EXPECT_EQ(legacy_traces, 2u);
    EXPECT_FALSE(direct.traces_point_independent);
    ASSERT_EQ(direct.traces.size(), 2u);
    ASSERT_TRUE(direct.traces[0].face);
    EXPECT_EQ(D::encode_contact(direct, output_budget), legacy);
    const auto decoded = D::decode_contact(legacy, ample);
    EXPECT_EQ(decoded.spans, direct.spans);
    EXPECT_EQ(decoded.factors, direct.factors);
    EXPECT_EQ(decoded.traces[0].face->points(), direct.traces[0].face->points());
    EXPECT_GT(direct_budget.used, sizeof(D::ContactRecord));
    EXPECT_GT(direct_budget.peak, direct_budget.used);
    leaf = {};
    kernel.trace_faces.clear();
    EXPECT_EQ(direct.key, (AMR::AdaptiveCellKey{7, AMR::ConvexCellAncestry{19, {1, 3}}}));
    EXPECT_EQ(direct.spans, (std::array<long double, 3>{4, 2, 1}));
    EXPECT_TRUE(direct.traces[0].face->contains_point(point(7, 1, 1)));
}

TEST(AdaptiveClosureRecords, ExactAdmissionAndOneByteShortRejectionLeaveSourceUnchanged)
{
    const auto leaf = source_leaf();
    RecordKernel kernel;
    kernel.trace_faces = {{{'s'}, 3, AMR::AdaptiveSplit::Base, rectangle(7), true}};
    const auto key_before = AMR::encode_adaptive_key(leaf.key);
    const auto geometry_before = leaf.geometry->encode_certificate();
    const auto trace_before = kernel.trace_faces[0].face->encode_incidence();
    // Nested geometry construction has its own simultaneous-storage admission.
    // The outer ledger's observed peak is not the minimum accepted total cap.
    // Every probe has independent accounting and pins; only resource rejection
    // is allowed to move the lower endpoint of this admission search.
    const auto accepted = [&](std::size_t cap) -> std::optional<bool>
    {
        D::Budget budget{cap};
        D::IdentityBudget identities;
        std::size_t traces = 0, peak = 0;
        try
        {
            (void) D::prepare_contact(leaf, kernel, budget, identities, traces, peak);
            return true;
        }
        catch (const P::GeometryError& error)
        {
            if (error.code() == P::GeometryErrorCode::ResourceLimit)
                return false;
            ADD_FAILURE() << "Non-resource geometry rejection at cap " << cap << ": " << error.what();
        }
        catch (const std::invalid_argument& error)
        {
            const std::string_view message = error.what();
            if (message.find("byte budget") != std::string_view::npos ||
                message.find("payload budget exhausted") != std::string_view::npos ||
                message.find("native decoding budget") != std::string_view::npos ||
                message.find("native record budget") != std::string_view::npos ||
                message.find("construction budget is already exhausted") != std::string_view::npos)
                return false;
            ADD_FAILURE() << "Non-resource rejection at cap " << cap << ": " << error.what();
        }
        catch (const std::exception& error)
        {
            ADD_FAILURE() << "Unexpected rejection at cap " << cap << ": " << error.what();
        }
        return {};
    };
    auto result = accepted(1);
    ASSERT_TRUE(result.has_value());
    ASSERT_FALSE(*result);
    result = accepted(ample);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(*result);
    std::size_t rejected_cap = 1, admitted_cap = ample;
    while (admitted_cap - rejected_cap > 1)
    {
        const auto middle = rejected_cap + (admitted_cap - rejected_cap) / 2;
        result = accepted(middle);
        ASSERT_TRUE(result.has_value());
        if (*result)
            admitted_cap = middle;
        else
            rejected_cap = middle;
    }
    result = accepted(admitted_cap);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(*result);
    result = accepted(admitted_cap - 1);
    ASSERT_TRUE(result.has_value());
    EXPECT_FALSE(*result);
    EXPECT_EQ(AMR::encode_adaptive_key(leaf.key), key_before);
    EXPECT_EQ(leaf.domain, (AMR::AdaptiveBytes{'d', 'o', 'm'}));
    EXPECT_EQ(leaf.geometry->encode_certificate(), geometry_before);
    EXPECT_EQ(kernel.trace_faces[0].face->encode_incidence(), trace_before);
    D::Budget rejected{ample};
    D::IdentityBudget rejected_identities;
    std::size_t traces = 0, peak = 0;
    kernel.foreign_trace = true;
    EXPECT_THROW(
        (void) D::prepare_contact(leaf, kernel, rejected, rejected_identities, traces, peak), std::invalid_argument);
    kernel.foreign_trace = false;
    kernel.trace_faces[0].stem.clear();
    EXPECT_THROW(
        (void) D::prepare_contact(leaf, kernel, rejected, rejected_identities, traces, peak), std::invalid_argument);
}
} // namespace
