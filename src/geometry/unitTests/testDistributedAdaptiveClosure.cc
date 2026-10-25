/** @file testDistributedAdaptiveClosure.cc
 * @brief Independent exact closed-contact and sparse closure rejection oracles.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveClosure.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <gtest/gtest.h>
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
} // namespace
