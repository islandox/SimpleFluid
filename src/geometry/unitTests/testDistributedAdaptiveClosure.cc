/** @file testDistributedAdaptiveClosure.cc
 * @brief Independent exact closed-contact and sparse closure rejection oracles.
 */
#include "geometry/PLIC/ScalarPacket.hh"
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <array>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
/** Fixture geometry is deliberately independent of dyadic provider coordinates.
 * It isolates contact discovery and trace provenance from provider construction. */
class ContactKernel final : public ::SimpleFluid::Meshes::AMR::AdaptiveProviderKernel
{
    bool trace_, terminal_, restricted_;

public:
    explicit ContactKernel(bool trace = false, bool terminal = true, bool restricted = false)
        : trace_(trace), terminal_(terminal), restricted_(restricted)
    {
    }
    ::SimpleFluid::Meshes::AMR::AdaptiveBytes descriptor() const override
    {
        return {'f', 'i', 'x', 't', 'u', 'r', 'e'};
    }
    SimpleFluid::PLIC::ConvexGeometryPtr<SimpleFluid::ArithmeticMode::Ordinary> geometry(
        const ::SimpleFluid::Meshes::AMR::AdaptiveLeaf& leaf) const override
    {
        return leaf.geometry;
    }
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveSurface> surfaces(
        const ::SimpleFluid::Meshes::AMR::AdaptiveLeaf& leaf) const override
    {
        return leaf.surfaces;
    }
    ::SimpleFluid::Meshes::AMR::AdaptiveTrace traces(
        const ::SimpleFluid::Meshes::AMR::AdaptiveLeaf& leaf) const override
    {
        if (!trace_)
            return {leaf.key, {}};
        ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer stem;
        stem.word(leaf.key.region);
        const auto& ancestry = std::get<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(leaf.key.ancestry);
        PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> support;
        if (restricted_)
            for (const auto& surface : leaf.surfaces)
                if (surface.geometry->area_vector().z > 0)
                    support = surface.geometry;
        return {leaf.key, {{std::move(stem.data), ancestry.level, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic,
                              std::move(support), terminal_}}};
    }
    ::SimpleFluid::Meshes::AMR::AdaptiveSplitResult split(
        const ::SimpleFluid::Meshes::AMR::AdaptiveLeaf&, ::SimpleFluid::Meshes::AMR::AdaptiveSplit) const override
    {
        throw std::invalid_argument("Fixture has no refinement kernel.");
    }
};
::SimpleFluid::Meshes::AMR::AdaptiveLeaf leaf(unsigned id, SimpleFluid::PLIC::GeometryPoint lower,
    SimpleFluid::PLIC::GeometryPoint upper, unsigned depth = 0, std::uint64_t region = 0)
{
    ::SimpleFluid::Meshes::AMR::AdaptiveLeaf value;
    value.key = {region, ::SimpleFluid::Meshes::AMR::CartesianCellAncestry{depth, {id, 0, 0}}};
    value.domain = {'d'};
    value.provider = ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Polyhedral;
    value.geometry = SimpleFluid::PLIC::ConvexCellGeometry<SimpleFluid::ArithmeticMode::Ordinary>::box(lower, upper);
    for (std::size_t f = 0; f < value.geometry->num_faces(); ++f)
    {
        ::SimpleFluid::Meshes::AMR::adaptive_detail::Writer key;
        key.bytes(::SimpleFluid::Meshes::AMR::encode_adaptive_key(value.key));
        key.word(f);
        value.surfaces.push_back({std::move(key.data), value.geometry->face_geometry(f), 1, true, "wall"});
    }
    return value;
}
std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> owned(
    ::SimpleFluid::Meshes::AMR::AdaptiveLeaf coarse, ::SimpleFluid::Meshes::AMR::AdaptiveLeaf fine)
{
    auto comm = Tpetra::getDefaultComm();
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf> result;
    if (comm->getRank() == 0)
        result.push_back(std::move(coarse));
    if (comm->getRank() == 1 % comm->getSize())
        result.push_back(std::move(fine));
    return result;
}
void require_coarse(const std::vector<::SimpleFluid::Meshes::AMR::AdaptiveLeaf>& source, const ContactKernel& kernel,
    ::SimpleFluid::Meshes::AMR::AdaptiveClosureOptions options = {})
{
    auto comm = Tpetra::getDefaultComm();
    auto marks = ::SimpleFluid::Meshes::AMR::required_adaptive_refinements(source, kernel, comm, options);
    EXPECT_EQ(marks.size(), comm->getRank() == 0 ? 1 : 0);
    for (const auto& mark : marks)
    {
        EXPECT_EQ(std::get<::SimpleFluid::Meshes::AMR::CartesianCellAncestry>(mark.key.ancestry).index[0], 0);
        EXPECT_EQ(mark.split, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic);
    }
    int local = int(marks.size()), global = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local, &global);
    EXPECT_EQ(global, 1);
}
class ClosureDiagnosticCapture
{
    std::optional<std::string> previous_;
    bool active_ = true;
    void restore()
    {
        if (previous_)
            setenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS", previous_->c_str(), 1);
        else
            unsetenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS");
    }

public:
    ClosureDiagnosticCapture()
    {
        if (const auto* value = std::getenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS"))
            previous_ = value;
        setenv("SIMPLEFLUID_ADAPTIVE_DIAGNOSTICS", "1", 1);
        testing::internal::CaptureStderr();
    }
    ~ClosureDiagnosticCapture()
    {
        if (active_)
            (void) testing::internal::GetCapturedStderr();
        restore();
    }
    std::string finish()
    {
        auto output = testing::internal::GetCapturedStderr();
        active_ = false;
        restore();
        return output;
    }
};
std::uint64_t closure_diagnostic(const std::string& log, const std::string& name)
{
    const auto start = log.rfind("ADAPTIVE_CLOSURE rank=");
    if (start == std::string::npos)
    {
        ADD_FAILURE() << "Missing closure diagnostic.";
        return 0;
    }
    const auto token = " " + name + "=";
    const auto position = log.find(token, start);
    if (position == std::string::npos || position > log.find('\n', start))
    {
        ADD_FAILURE() << "Missing closure diagnostic field " << name;
        return 0;
    }
    return std::stoull(log.substr(position + token.size()));
}
struct ObservedClosure
{
    std::vector<::SimpleFluid::Meshes::AMR::AdaptiveMark> marks;
    ::SimpleFluid::Meshes::AMR::AdaptiveClosureStats statistics;
    std::uint64_t point_observations = 0, global_physical_nodes = 0, planned_reply_records = 0, retired_groups = 0;
};
ObservedClosure observe_closure(std::span<const ::SimpleFluid::Meshes::AMR::AdaptiveLeaf> source,
    const ::SimpleFluid::Meshes::AMR::AdaptiveProviderKernel& kernel)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    ObservedClosure result;
    ClosureDiagnosticCapture capture;
    AMR::AdaptiveClosureOptions options;
    options.statistics = &result.statistics;
    result.marks = AMR::required_adaptive_refinements(source, kernel, Tpetra::getDefaultComm(), options);
    const auto log = capture.finish();
    result.point_observations = closure_diagnostic(log, "point_observations");
    result.global_physical_nodes = closure_diagnostic(log, "global_physical_nodes");
    result.planned_reply_records = closure_diagnostic(log, "planned_reply_records");
    result.retired_groups = closure_diagnostic(log, "retired_groups");
    const auto request_peak = closure_diagnostic(log, "request_state_peak_bytes");
    EXPECT_LE(request_peak, options.byte_budget);
    EXPECT_GE(closure_diagnostic(log, "tile_and_request_peak_bytes"), request_peak);
    return result;
}
void require_contact_counts(const ObservedClosure& observed, std::uint64_t contacts, std::uint64_t observations,
    std::uint64_t physical_nodes, std::uint64_t point_groups)
{
    const std::array<std::uint64_t, 4> local{observed.statistics.contacts, observed.point_observations,
        observed.planned_reply_records, observed.retired_groups};
    std::array<std::uint64_t, 4> global{};
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 4, local.data(), global.data());
    EXPECT_EQ(global[0], contacts);
    EXPECT_EQ(global[1], observations);
    EXPECT_EQ(global[2], contacts);
    EXPECT_EQ(global[3], point_groups);
    EXPECT_EQ(observed.global_physical_nodes, physical_nodes);
}
TEST(DistributedAdaptiveClosure, ScalarPacketRoundTripsNativePrecisionAndRejectsMalformedWords)
{
    // The exact packet for one is sign zero, binary exponent one, and a
    // leading half-unit significand limb followed by zero limbs.
    const auto one = PLIC::scalar_packet(1.L);
    ASSERT_GE(one.size(), 3);
    EXPECT_EQ(one[0], 0);
    EXPECT_EQ(one[1], 1);
    EXPECT_EQ(one[2], std::uint64_t(1) << 31);
    for (std::size_t i = 3; i < one.size(); ++i)
        EXPECT_EQ(one[i], 0);
    for (const auto value : {0.L, -1.L, 2.L + std::ldexp(1.L, -54), std::numeric_limits<long double>::min(),
             std::numeric_limits<long double>::max(), std::numeric_limits<long double>::denorm_min()})
        EXPECT_EQ(PLIC::scalar_from_packet(PLIC::scalar_packet(value)), value);
    EXPECT_THROW(PLIC::scalar_packet(std::numeric_limits<long double>::infinity()), std::invalid_argument);
    auto bad = one;
    bad[0] = 2;
    EXPECT_THROW(PLIC::scalar_from_packet(bad), std::invalid_argument);
    bad = one;
    bad[1] = std::numeric_limits<long double>::max_exponent + 1;
    EXPECT_THROW(PLIC::scalar_from_packet(bad), std::invalid_argument);
    bad = one;
    bad[2] = std::uint64_t(1) << 32;
    EXPECT_THROW(PLIC::scalar_from_packet(bad), std::invalid_argument);
    EXPECT_THROW(PLIC::scalar_from_packet(std::span<const std::uint64_t>(one).first(1)), std::invalid_argument);
}
TEST(DistributedAdaptiveClosure, FaceContactRequiresActualCoarseProvider)
{
    auto a = leaf(0, {0, 0, 0}, {4, 1, 1}), b = leaf(1, {4, 0, 0}, {5, 1, 1});
    for (auto* value : {&a, &b})
        for (auto& surface : value->surfaces)
        {
            auto normal = surface.geometry->area_vector();
            if ((value == &a && normal.x > 0) || (value == &b && normal.x < 0))
            {
                surface.key = {'j'};
                surface.exterior = false;
                surface.boundary.clear();
                surface.orientation = value == &a ? 1 : -1;
            }
        }
    auto source = owned(a, b);
    require_coarse(source, ContactKernel{});
    for (const auto& value : source)
        EXPECT_TRUE(value.geometry->equivalent(*(value.key == a.key ? a.geometry : b.geometry)));
}
TEST(DistributedAdaptiveClosure, IsolatedEdgeContactRequiresCoarseRefinement)
{
    require_coarse(owned(leaf(0, {0, 0, 0}, {4, 4, 1}), leaf(1, {4, 4, 0}, {5, 5, 1})), ContactKernel{});
}
TEST(DistributedAdaptiveClosure, IsolatedVertexContactRequiresCoarseRefinement)
{
    require_coarse(owned(leaf(0, {0, 0, 0}, {4, 4, 4}), leaf(1, {4, 4, 4}, {5, 5, 5})), ContactKernel{});
}
TEST(DistributedAdaptiveClosure, GeometryAndTracesAreInternedAcrossClosedPointIncidences)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    const auto source = owned(leaf(0, {0, 0, 0}, {4, 4, 4}), leaf(1, {4, 4, 4}, {5, 5, 5}));
    AMR::AdaptiveClosureStats statistics;
    AMR::AdaptiveClosureOptions options;
    options.statistics = &statistics;
    require_coarse(source, ContactKernel{true}, options);
    EXPECT_EQ(statistics.geometry_records, source.size());
    EXPECT_EQ(statistics.trace_records, source.size());
    EXPECT_EQ(statistics.contacts, 8 * source.size());
    EXPECT_LE(statistics.transmitted_geometry_records, source.size() * std::min(8, comm->getSize()));
    EXPECT_LE(statistics.decoded_geometry_records, 2);
    EXPECT_EQ(statistics.decoded_geometry_bytes, 3 * sizeof(long double) * statistics.decoded_geometry_records);
    EXPECT_LE(statistics.live_peak_bytes, options.byte_budget);
    EXPECT_GE(statistics.cumulative_bytes, statistics.live_peak_bytes);
    if (!source.empty())
        EXPECT_GT(statistics.cumulative_bytes, statistics.live_peak_bytes);
    if (comm->getSize() < 8 && !source.empty())
        EXPECT_LT(statistics.transmitted_geometry_records, statistics.contacts);
    if (comm->getSize() == 1)
        EXPECT_EQ(statistics.replicated_record_bytes, statistics.unique_record_bytes);
    // The same contact/trace work cap remains cumulative across all points.
    options.contact_budget = 1;
    EXPECT_THROW(AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options), std::invalid_argument);
    EXPECT_TRUE(AMR::required_adaptive_refinements({}, ContactKernel{}, comm).empty());
}
TEST(DistributedAdaptiveClosure, CompactSpansPreserveTranslatedUnequalAxesAndExactRatio)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    // Independent widths (4,2,1) and (2,1,2): every directional ratio is
    // at most two, despite unequal axes and a nonzero physical origin.
    auto a = leaf(0, {7, 11, 17}, {11, 13, 18});
    auto b = leaf(1, {11, 13, 18}, {13, 14, 20});
    EXPECT_TRUE(AMR::required_adaptive_refinements(owned(a, b), ContactKernel{}, comm).empty());
    // Increase only the coarse x extent. Its translated physical shape, rather
    // than ancestry or a common axis length, must select the coarse leaf.
    a = leaf(0, {6, 11, 17}, {11, 13, 18});
    require_coarse(owned(a, b), ContactKernel{});
}
TEST(DistributedAdaptiveClosure, CompactSpansKeepLongDoubleSensitiveComparison)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    if (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "The platform has no wider long-double significand.";
    // Both source endpoints are representable doubles, but their span
    // 2+2^-54 rounds to two in binary64. The existing Ordinary comparison
    // must still recognize that it strictly exceeds twice the unit span.
    const auto a = leaf(0, {-std::ldexp(1.0, -54), 0, 0}, {2, 1, 1});
    const auto b = leaf(1, {2, 1, 1}, {3, 2, 2});
    const auto span = a.geometry->axis_spans()[0].value;
    ASSERT_GT(span, 2.L);
    ASSERT_EQ(double(span), 2.);
    ASSERT_TRUE(a.geometry->axis_span_exceeds(*b.geometry, 0, 2));
    const auto decoded_a =
        PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::decode_certificate(a.geometry->encode_certificate());
    const auto decoded_b =
        PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::decode_certificate(b.geometry->encode_certificate());
    ASSERT_TRUE(decoded_a->axis_span_exceeds(*decoded_b, 0, 2));
    require_coarse(owned(a, b), ContactKernel{});
}
TEST(DistributedAdaptiveClosure, PointIndependentTraceWorkIsCachedAndCumulative)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    const auto source = owned(leaf(0, {0, 0, 0}, {1, 1, 1}, 0), leaf(1, {1, 1, 0}, {2, 2, 1}, 3));
    AMR::AdaptiveClosureStats statistics;
    AMR::AdaptiveClosureOptions options;
    options.maximum_span_ratio = 100;
    options.statistics = &statistics;
    require_coarse(source, ContactKernel{true}, options);
    if (comm->getSize() == 1)
    {
        // Exactly two shared edge endpoints produce two original cell-pair
        // admissions, and the one immutable trace pair is evaluated once.
        EXPECT_EQ(statistics.work, 3);
        options.contact_budget = 2;
        EXPECT_THROW(
            AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options), std::invalid_argument);
        options.contact_budget = 3;
        require_coarse(source, ContactKernel{true}, options);
        EXPECT_EQ(statistics.work, 3);
    }
    options.contact_budget = 1;
    EXPECT_THROW(AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options), std::invalid_argument);
    require_coarse(source, ContactKernel{true});
    EXPECT_TRUE(AMR::required_adaptive_refinements({}, ContactKernel{}, comm).empty());
}
TEST(DistributedAdaptiveClosure, FaceRestrictedTracesRemainSpecificToEachContactPoint)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    const auto source = owned(leaf(0, {0, 0, 0}, {1, 1, 1}, 0), leaf(1, {1, 1, 0}, {2, 2, 1}, 3));
    AMR::AdaptiveClosureStats statistics;
    AMR::AdaptiveClosureOptions options;
    options.maximum_span_ratio = 100;
    options.statistics = &statistics;
    // The top face contains only the z=1 endpoint of the shared edge.
    // The z=0 contact cannot reuse a successful membership comparison.
    require_coarse(source, ContactKernel{true, true, true}, options);
    if (comm->getSize() == 1)
    {
        EXPECT_EQ(statistics.work, 4);
        options.contact_budget = 3;
        EXPECT_THROW(AMR::required_adaptive_refinements(source, ContactKernel{true, true, true}, comm, options),
            std::invalid_argument);
        options.contact_budget = 4;
        require_coarse(source, ContactKernel{true, true, true}, options);
    }
}
TEST(DistributedAdaptiveClosure, EdgeTJunctionPropagatesInteriorSegmentEndpoints)
{
    const auto coarse = leaf(0, {0, 0, 0}, {4, 4, 4});
    const auto source = owned(coarse, leaf(1, {4, 4, 1}, {5, 5, 2}));
    const auto observed = observe_closure(source, ContactKernel{});
    // The fine edge contributes both interior z=1 and z=2 endpoints to the
    // coarse shell: ten coarse plus eight fine contacts, with 28+24 visits.
    require_contact_counts(observed, 18, 52, 16, 16);
    EXPECT_EQ(observed.marks.size(), Tpetra::getDefaultComm()->getRank() == 0 ? 1 : 0);
    for (const auto& mark : observed.marks)
        EXPECT_EQ(mark.key, coarse.key);
}
TEST(DistributedAdaptiveClosure, DistantRefinementDoesNotCreateNeighborhood)
{
    auto source = owned(leaf(0, {0, 0, 0}, {4, 4, 4}), leaf(1, {10, 10, 10}, {10.25, 10.25, 10.25}));
    EXPECT_TRUE(
        ::SimpleFluid::Meshes::AMR::required_adaptive_refinements(source, ContactKernel{}, Tpetra::getDefaultComm())
            .empty());
}
TEST(DistributedAdaptiveClosure, DisconnectedDomainsDoNotCreateContact)
{
    auto a = leaf(0, {0, 0, 0}, {4, 4, 4}), b = leaf(1, {4, 4, 4}, {5, 5, 5});
    b.domain = {'c'};
    const auto observed = observe_closure(owned(a, b), ContactKernel{});
    EXPECT_TRUE(observed.marks.empty());
    // One coincident coordinate belongs to two domains and must remain two
    // catalogue identities. Each cube retains all eight points and 24 visits.
    require_contact_counts(observed, 16, 48, 16, 16);
    EXPECT_EQ(observed.statistics.work, 0);
}
TEST(DistributedAdaptiveClosure, DyadicPeriodicNodeCatalogueRetainsAllPhysicalImages)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    AMR::AMRGrid grid;
    grid.block_cells = {2, 2, 2};
    grid.origin = {0, 0, 0};
    grid.extent = {1, 1, 1};
    grid.periodic = {true, true, true};
    AMR::CartesianAMR provider(grid);
    std::vector<std::uint64_t> selection;
    for (std::uint64_t cell = 0; cell < 8; ++cell)
        if (cell % comm->getSize() == std::uint64_t(comm->getRank()))
            selection.push_back(cell);
    const auto bootstrap = provider.detach_owned(selection, {'p'});
    const auto observed = observe_closure(bootstrap.owned, *bootstrap.kernel);
    EXPECT_TRUE(observed.marks.empty());
    // Physical coordinates retain 3^3 nodes even though normalization maps
    // each periodic zero/one boundary pair to the same contact authority.
    require_contact_counts(observed, 64, 192, 27, 8);
    if (comm->getSize() == 1)
        EXPECT_EQ(observed.statistics.work, 224 + 28);
}
TEST(DistributedAdaptiveClosure, ComparableTraceJumpIsIndependentOfPhysicalRatio)
{
    ::SimpleFluid::Meshes::AMR::AdaptiveClosureOptions options;
    options.maximum_span_ratio = 100;
    require_coarse(
        owned(leaf(0, {0, 0, 0}, {1, 1, 1}, 0), leaf(1, {1, 1, 1}, {2, 2, 2}, 3)), ContactKernel{true}, options);
}
TEST(DistributedAdaptiveClosure, UnrelatedRootTracesAndNonterminalCapsAreNotCompared)
{
    ::SimpleFluid::Meshes::AMR::AdaptiveClosureOptions options;
    options.maximum_span_ratio = 100;
    auto comm = Tpetra::getDefaultComm();
    auto unrelated = owned(leaf(0, {0, 0, 0}, {1, 1, 1}, 0, 0), leaf(1, {1, 1, 1}, {2, 2, 2}, 3, 1));
    EXPECT_TRUE(::SimpleFluid::Meshes::AMR::required_adaptive_refinements(unrelated, ContactKernel{true}, comm, options)
            .empty());
    auto caps = owned(leaf(0, {0, 0, 0}, {1, 1, 1}, 0), leaf(1, {1, 1, 1}, {2, 2, 2}, 3));
    EXPECT_TRUE(
        ::SimpleFluid::Meshes::AMR::required_adaptive_refinements(caps, ContactKernel{true, false}, comm, options)
            .empty());
}
TEST(DistributedAdaptiveClosure, StaticDemandAndWorkOrByteExhaustionRejectCollectively)
{
    auto comm = Tpetra::getDefaultComm();
    auto a = leaf(0, {0, 0, 0}, {4, 4, 4}), b = leaf(1, {4, 4, 4}, {5, 5, 5});
    a.provider = ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Static;
    auto source = owned(a, b);
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::required_adaptive_refinements(source, ContactKernel{}, comm),
        std::invalid_argument);
    a.provider = ::SimpleFluid::Meshes::AMR::AdaptiveProviderKind::Polyhedral;
    source = owned(a, b);
    ::SimpleFluid::Meshes::AMR::AdaptiveClosureOptions options;
    options.byte_budget = 1;
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::required_adaptive_refinements(source, ContactKernel{}, comm, options),
        std::invalid_argument);
    options = {};
    options.contact_budget = 1;
    EXPECT_THROW(::SimpleFluid::Meshes::AMR::required_adaptive_refinements(source, ContactKernel{true}, comm, options),
        std::invalid_argument);
    for (const auto& value : source)
        EXPECT_EQ(value.domain, ::SimpleFluid::Meshes::AMR::AdaptiveBytes{'d'});
    require_coarse(source, ContactKernel{});
    EXPECT_TRUE(::SimpleFluid::Meshes::AMR::required_adaptive_refinements({}, ContactKernel{}, comm).empty());
}
// Explicit construction-only reproduction. Dimensions, coordinates and owners
// match the motion/capillary qualification fixtures; no phase initialization or
// physical advancement is performed. Disabled to keep ordinary unit runs small.
TEST(DistributedAdaptiveClosure, DISABLED_QualificationSetupOnly)
{
    namespace AMR = ::SimpleFluid::Meshes::AMR;
    const auto comm = Tpetra::getDefaultComm();
    const unsigned n = std::getenv("SIMPLEFLUID_SETUP_N") ? std::stoul(std::getenv("SIMPLEFLUID_SETUP_N")) : 16;
    const std::string kind = std::getenv("SIMPLEFLUID_SETUP_KIND") ? std::getenv("SIMPLEFLUID_SETUP_KIND") : "q2";
    ASSERT_TRUE(n > 0 && n % 2 == 0);
    ASSERT_TRUE(kind == "q2" || kind == "three_dimensional" || kind == "sphere");
    const unsigned nz = kind == "q2" ? 2 : n;
    const bool sphere = kind == "sphere";
    const std::string representation =
        std::getenv("SIMPLEFLUID_SETUP_REPRESENTATION") ? std::getenv("SIMPLEFLUID_SETUP_REPRESENTATION") : "expanded";
    const int compact = representation == "compact", valid = compact || representation == "expanded";
    int minimum_compact = 0, maximum_compact = 0, all_valid = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &valid, &all_valid);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &compact, &minimum_compact);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &compact, &maximum_compact);
    ASSERT_EQ(all_valid, 1);
    ASSERT_EQ(minimum_compact, maximum_compact);
    RecordProperty("native_draft_representation", representation);
    AMR::AMRGrid grid;
    grid.root_blocks = {n / 2, n / 2, nz / 2};
    grid.block_cells = {2, 2, 2};
    if (sphere)
        grid.extent = {double(n), double(n), double(nz)};
    std::vector<std::uint64_t> selected;
    const AMR::AdaptiveBytes domain =
        sphere ? AMR::AdaptiveBytes{'c', 'a', 'p', 'i', 'l', 'l', 'a', 'r', 'y', 'b', 'o', 'x'}
               : AMR::AdaptiveBytes{'q', '2', 'm', 'o', 't', 'i', 'o', 'n'};
    auto bootstrap = [&]
    {
        AMR::CartesianAMR provider(grid);
        for (std::uint64_t gid = 0; gid < std::uint64_t(n) * n * nz; ++gid)
        {
            const auto box = provider.cell_box(gid);
            const auto x = (box.lower[0] + box.upper[0]) / 2;
            const auto owner = sphere ? int(gid % std::min(2, comm->getSize())) : int(x * comm->getSize());
            if (owner == comm->getRank())
                selected.push_back(gid);
        }
        return provider.detach_owned(selected, domain);
    }();
    std::cout << "SETUP_ONLY rank=" << comm->getRank() << " kind=" << kind << " N=" << n
              << " stage=hierarchy_begin owned=" << selected.size() << " byte_budget=67108864 maximum_tiles=262144\n"
              << std::flush;
    auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    const auto storage = hierarchy->storage();
    std::cout << "SETUP_ONLY rank=" << comm->getRank()
              << " stage=hierarchy_complete owned_bytes=" << storage.owned_bytes
              << " ancestor_bytes=" << storage.ancestor_bytes << " directory_bytes=" << storage.directory_bytes
              << " procedural_bytes=" << storage.procedural_bytes << " native_bytes=" << storage.native_bytes << '\n'
              << std::flush;
    const auto draft =
        compact ? AMR::adaptive_detail::compact_adaptive_native_draft(*hierarchy) : hierarchy->native_draft();
    EXPECT_EQ(hierarchy->global_cells(), std::uint64_t(n) * n * nz);
    std::cout << "SETUP_ONLY rank=" << comm->getRank() << " stage=native_draft_complete tiles=" << draft.tiles.size()
              << " nodes=" << draft.nodes.size() << " representation=" << representation << " physical_steps=0\n"
              << std::flush;
}
} // namespace
