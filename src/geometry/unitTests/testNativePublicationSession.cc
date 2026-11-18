/** Independent source/occurrence capability checks for the private producer. */
#include "geometry/details/NativeObjectOwnerAccess.hh"
#include "geometry/PLIC/details/NativePlanarOwnerAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptivePackets.hh"
#include "geometry/mesh/AMR/DistributedCartesianKernel.hh"
#include "geometry/mesh/AMR/NativePublicationExchange.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/AMR/NativeSourceOwnerStore.hh"
#include "geometry/mesh/details/NativeCartesianAccess.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <gtest/gtest.h>
#include <numeric>
namespace
{
using namespace SimpleFluid;
namespace M = Meshes;
namespace A = Meshes::AMR;
namespace D = A::adaptive_detail;
using Admission = geometry_detail::NativeConstructionAdmission;
constexpr auto ordinary = ArithmeticMode::Ordinary;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const A::DistributedAdaptiveHierarchy> source()
{
    A::AMRGrid config;
    config.block_cells = {2, 2, 2};
    const A::CartesianAMR original(config);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> owned;
    for (std::uint64_t cell = 0; cell != 8; ++cell)
        if (static_cast<int>(cell % comm->getSize()) == comm->getRank())
            owned.push_back(cell);
    auto detached = original.detach_owned(owned, {'d'});
    return std::make_shared<const A::DistributedAdaptiveHierarchy>(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
}
TEST(NativePublicationSession, ExactOriginDoesNotTransferToCopiedDescriptor)
{
    Admission admission(geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(8 * 1024 * 1024));
    const std::array<double, 2> edges{0, 1};
    const std::array<std::span<const double>, 3> axes{edges, edges, edges};
    const std::array<std::string_view, 6> labels{"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    auto native = M::details::NativeCartesianAccess::make("cube", axes, labels, admission);
    M::NativeFragmentCache cache;
    const auto descriptor = M::NativeFragmentAccess::extract(native.region, cache, admission);
    const auto record = M::NativeFragmentAccess::find(descriptor, A::geometry_construction_controls());
    ASSERT_TRUE(record);
    const auto copy = std::make_shared<const M::NativeFragmentDescriptor>(*descriptor);
    EXPECT_FALSE(M::NativeFragmentAccess::find(copy, A::geometry_construction_controls()));
    const auto& geometry = std::get<M::CartesianRegion>(record->region()).geometry();
    EXPECT_DOUBLE_EQ(geometry.cell_volume(0), 1);
    auto changed = A::geometry_construction_controls();
    ++changed.rounding;
    EXPECT_FALSE(M::NativeFragmentAccess::find(descriptor, changed));
    // The public decoder still fully validates an ordinary byte-identical copy.
    EXPECT_NO_THROW((void) M::materialize_native_fragment(*copy, cache));
}
TEST(NativePublicationSession, WarmExactSourceAndIncidenceReusePhysicalControls)
{
    const auto hierarchy = source();
    D::NativePublicationSession session(D::NativePublicationSession::capture(*hierarchy));
    ASSERT_TRUE(session.qualified());
    for (const auto& leaf : hierarchy->owned_leaves())
    {
        const auto proof = session.source(leaf);
        ASSERT_TRUE(proof);
        ASSERT_TRUE(proof->fragment());
        EXPECT_DOUBLE_EQ(proof->metrics().volume, .125);
        const auto& ancestry = std::get<A::CartesianCellAncestry>(leaf.key.ancestry);
        for (std::size_t axis = 0; axis != 3; ++axis)
            EXPECT_DOUBLE_EQ(proof->metrics().centroid[axis], .25 + .5 * ancestry.index[axis]);
        std::vector<D::CanonicalSubfaceView> pieces;
        for (std::size_t surface = 0; surface != 6; ++surface)
            session.inputs().subfaces->visit_face_pieces(
                leaf.key, surface, [&](const auto& view) { pieces.push_back(view); });
        const auto incidence = session.incidence(proof, pieces);
        ASSERT_TRUE(incidence);
        const auto before = PLIC::geometry_statistics();
        EXPECT_EQ(session.source(leaf), proof);
        EXPECT_EQ(session.incidence(proof, pieces), incidence);
        const auto after = PLIC::geometry_statistics();
        EXPECT_EQ(after.constructions, before.constructions);
        EXPECT_EQ(after.source_imports, before.source_imports);
        const auto seal = session.seal(proof, incidence);
        EXPECT_EQ(&seal.source(), proof.get());
        EXPECT_EQ(&seal.incidence(), incidence.get());
        // A copied public row is not the private current owner-local selector.
        auto copied = leaf;
        EXPECT_THROW((void) session.source(copied), std::invalid_argument);
        auto omitted = pieces;
        omitted.pop_back();
        EXPECT_THROW((void) session.incidence(proof, omitted), std::invalid_argument);
        EXPECT_EQ(session.incidence(proof, pieces), incidence);
    }
}
TEST(NativePublicationSession, StagedFailureLeavesAcceptedCacheUnchanged)
{
    const auto hierarchy = source();
    auto inputs = D::NativePublicationSession::capture(*hierarchy);
    const auto accepted = inputs.cache->indexes;
    const auto revision = inputs.cache->revision;
    {
        D::NativePublicationSession session(inputs);
        if (!hierarchy->owned_leaves().empty())
        {
            const auto proof = session.source(*hierarchy->owned_leaves().begin());
            EXPECT_TRUE(proof);
        }
        EXPECT_EQ(inputs.cache->indexes, accepted);
        EXPECT_EQ(inputs.cache->revision, revision);
    }
    EXPECT_EQ(inputs.cache->indexes, accepted);
    EXPECT_EQ(inputs.cache->revision, revision);
}
TEST(NativePublicationSession, EmptyRemoteSelectionStillParticipatesCollectively)
{
    const auto hierarchy = source();
    D::NativePublicationSession session(D::NativePublicationSession::capture(*hierarchy));
    D::NativePublicationExchangeStatistics statistics;
    const auto result = D::NativePublicationExchange::fetch(session, *hierarchy, {}, &statistics);
    EXPECT_TRUE(result.empty());
    EXPECT_EQ(statistics.full_replies, 0);
    EXPECT_EQ(statistics.confirmed_reuse, 0);
}
} // namespace
