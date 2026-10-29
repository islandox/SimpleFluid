/** @file testDistributedAdaptiveClosure.cc
 * @brief Independent exact closed-contact and sparse closure rejection oracles.
 */
#include "geometry/mesh/AMR/CartesianAMR.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::Meshes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
/** Fixture geometry is deliberately independent of dyadic provider coordinates.
 * It isolates contact discovery and trace provenance from provider construction. */
class ContactKernel final : public ::SimpleFluid::Meshes::AMR::AdaptiveProviderKernel
{
    bool trace_, terminal_;

public:
    explicit ContactKernel(bool trace = false, bool terminal = true) : trace_(trace), terminal_(terminal) {}
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
        return {leaf.key, {{std::move(stem.data), ancestry.level, ::SimpleFluid::Meshes::AMR::AdaptiveSplit::Isotropic,
                              {}, terminal_}}};
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
TEST(DistributedAdaptiveClosure, EdgeTJunctionPropagatesInteriorSegmentEndpoints)
{
    require_coarse(owned(leaf(0, {0, 0, 0}, {4, 4, 4}), leaf(1, {4, 4, 1}, {5, 5, 2})), ContactKernel{});
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
    EXPECT_TRUE(::SimpleFluid::Meshes::AMR::required_adaptive_refinements(
        owned(a, b), ContactKernel{}, Tpetra::getDefaultComm())
            .empty());
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
    const auto draft = hierarchy->native_draft();
    EXPECT_EQ(hierarchy->global_cells(), std::uint64_t(n) * n * nz);
    std::cout << "SETUP_ONLY rank=" << comm->getRank() << " stage=native_draft_complete tiles=" << draft.tiles.size()
              << " nodes=" << draft.nodes.size() << " physical_steps=0\n"
              << std::flush;
}
} // namespace
