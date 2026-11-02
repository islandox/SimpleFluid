/** @file testFrozenTopology.cc
 * @brief Frozen topology order, identity, width, borrowing and admission contracts.
 */
#include "geometry/PLIC/ConstructionRegistry.hh"
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include <atomic>
#include <cstdlib>
#include <gtest/gtest.h>
#include <new>
#include <ranges>

namespace
{
thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;
} // namespace
void* operator new(std::size_t bytes)
{
    if (count_allocations)
        ++allocation_count;
    if (void* result = std::malloc(bytes ? bytes : 1))
        return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes)
{
    return ::operator new(bytes);
}
void operator delete(void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept
{
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}

namespace
{
using SimpleFluid::ArithmeticMode;
namespace P = SimpleFluid::PLIC;
namespace D = P::details;
constexpr auto ordinary = ArithmeticMode::Ordinary;
template<ArithmeticMode Mode> using Geometry = P::ConvexCellGeometry<Mode>;
template<ArithmeticMode Mode> auto storage(const P::ConvexGeometryPtr<Mode>& geometry)
{
    return D::checked_geometry_storage<Mode>(P::DynamicGeometryAccess::implementation(*geometry->dynamic_geometry()));
}
template<ArithmeticMode Mode> auto publish(std::shared_ptr<D::GeometryStorage<Mode>> geometry)
{
    return Geometry<Mode>::from_dynamic(P::DynamicGeometryAccess::geometry(std::move(geometry)));
}
static_assert(std::ranges::random_access_range<P::FaceRange>);
static_assert(std::ranges::random_access_range<P::FaceVertexRange>);
static_assert(std::same_as<std::ranges::range_reference_t<P::FaceVertexRange>, std::size_t>);
static_assert(sizeof(D::IncidentPlaneRange<ordinary>) == 2 * sizeof(void*));
static_assert(sizeof(D::SurfaceRange<ordinary>) == sizeof(void*));

template<ArithmeticMode Mode> void box_contract()
{
    const auto root = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    const std::array<std::array<std::size_t, 4>, 6> expected{
        {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}}};
    ASSERT_EQ(root->faces().size(), expected.size());
    for (std::size_t face = 0; face < expected.size(); ++face)
        EXPECT_TRUE(std::ranges::equal(root->faces()[face], expected[face]));
    if constexpr (Mode == ordinary)
        EXPECT_EQ(root->volume().value, 1);
    else
    {
        EXPECT_LE(root->volume().lower(), 1);
        EXPECT_GE(root->volume().upper(), 1);
    }
    const auto frozen = storage(root);
    ASSERT_TRUE(frozen->topology);
    EXPECT_EQ(frozen->topology->arrays.index(), 0u);
    EXPECT_EQ(frozen->topology->planes.size(), 6u);
    // First vertex meets faces 0, 2, 4 in that order; table order must retain it.
    const auto incidence = frozen->vertices[0].incident;
    ASSERT_EQ(incidence.size(), 3u);
    EXPECT_EQ(incidence[0].get(), frozen->surfaces[0].plane.get());
    EXPECT_EQ(incidence[1].get(), frozen->surfaces[2].plane.get());
    EXPECT_EQ(incidence[2].get(), frozen->surfaces[4].plane.get());
    const auto cut = root->clip({{1, 0, 0}, {0, 0, 0}, .25}, true);
    ASSERT_TRUE(cut.inside);
    ASSERT_TRUE(cut.outside);
    if constexpr (Mode == ordinary)
    {
        EXPECT_EQ(cut.inside->volume().value, .25L);
        EXPECT_EQ(cut.outside->volume().value, .75L);
        EXPECT_EQ(cut.section.area.value, 1);
    }
    else
    {
        EXPECT_LE(cut.inside->volume().lower(), .25L);
        EXPECT_GE(cut.inside->volume().upper(), .25L);
        EXPECT_LE(cut.outside->volume().lower(), .75L);
        EXPECT_GE(cut.outside->volume().upper(), .75L);
        EXPECT_LE(cut.section.area.lower(), 1);
        EXPECT_GE(cut.section.area.upper(), 1);
    }
}
TEST(FrozenTopologyTest, OrdinaryPreservesAnalyticCubeWindingAndComplementaryMeasures)
{
    box_contract<ordinary>();
}
TEST(FrozenTopologyTest, CertificationPreservesAnalyticCubeWindingAndComplementaryMeasures)
{
    if (!P::certification_environment())
        GTEST_SKIP();
    box_contract<ArithmeticMode::BoundedCertification>();
}
TEST(FrozenTopologyTest, RowSurvivesTemporaryOuterRangeAndQueriesAllocateNothing)
{
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto row = root->faces()[0];
    const auto frozen = storage(root);
    allocation_count = 0;
    count_allocations = true;
    std::size_t sum = 0;
    for (std::size_t repeat = 0; repeat < 50; ++repeat)
    {
        for (auto face : root->faces())
            for (auto vertex : face)
                sum += vertex;
        for (const auto& vertex : frozen->vertices)
            for (const auto& plane : vertex.incident)
                sum += bool(plane);
        sum += row[2];
    }
    count_allocations = false;
    EXPECT_GT(sum, 0u);
    EXPECT_EQ(allocation_count, 0u);
    EXPECT_EQ(row[2], 6u);
    EXPECT_EQ(*(row.begin() + 3), 2u);
    EXPECT_EQ(root->faces().end() - root->faces().begin(), 6);
}
TEST(FrozenTopologyTest, OnlyActualPlaneIdentityDeduplicatesAndEveryIncidentEntrySurvives)
{
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    D::Operation operation(D::mode_limits(ordinary));
    auto draft = D::draft_geometry<ordinary>(*storage(root));
    const auto existing = draft->vertices[0].incident[0];
    const auto equal_coefficients = std::make_shared<D::StoredPlane<ordinary>>(*existing);
    draft->vertices[0].incident.push_back(existing);
    draft->vertices[0].incident.push_back(equal_coefficients);
    const auto frozen = D::freeze_geometry<ordinary>(std::move(*draft));
    EXPECT_EQ(frozen->topology->planes.size(), 7u);
    const auto row = frozen->vertices[0].incident;
    ASSERT_EQ(row.size(), 5u);
    EXPECT_EQ(row[0].get(), row[3].get());
    EXPECT_NE(row[0].get(), row[4].get());
    EXPECT_EQ(row[4].get(), equal_coefficients.get());
    EXPECT_EQ(frozen->topology->planes[3].get(), equal_coefficients.get());
}
template<ArithmeticMode Mode> void wide_contract()
{
    const auto root = Geometry<Mode>::box({0, 0, 0}, {1, 1, 1});
    D::Operation operation(D::mode_limits(Mode));
    auto draft = D::draft_geometry<Mode>(*storage(root));
    const auto wide_storage = D::freeze_geometry<Mode>(std::move(*draft), true);
    ASSERT_EQ(wide_storage->topology->arrays.index(), 1u);
    const auto wide = publish(wide_storage);
    EXPECT_EQ(root->faces(), wide->faces());
    EXPECT_EQ(root->volume().value, wide->volume().value);
    const P::GeometryPacketContext context{{2, 3, 5, 7}};
    EXPECT_EQ(root->encode(context), wide->encode(context));
    if constexpr (Mode == ordinary)
        EXPECT_EQ(root->encode_certificate(), wide->encode_certificate());
}
TEST(FrozenTopologyTest, ForcedWideFixtureHasIdenticalValuesAndPacketBytes)
{
    wide_contract<ordinary>();
    if (P::certification_environment())
        wide_contract<ArithmeticMode::BoundedCertification>();
}
TEST(FrozenTopologyTest, SyntheticCountsSelectWidthAndRejectOverflowWithoutAllocation)
{
    const auto maximum = std::numeric_limits<std::uint32_t>::max();
    EXPECT_TRUE(D::compact_topology({maximum, maximum, maximum, maximum, maximum}));
    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t))
    {
        EXPECT_FALSE(D::compact_topology({std::size_t(maximum) + 1, 1, 1, 1, 1}));
        EXPECT_FALSE(D::compact_topology({1, 1, 1, std::size_t(maximum) + 1, 1}));
        EXPECT_FALSE(D::compact_topology({1, 1, 1, 1, std::size_t(maximum) + 1}));
    }
    EXPECT_THROW(D::topology_sum(std::numeric_limits<std::size_t>::max(), 1), P::GeometryError);
    EXPECT_THROW(D::topology_product(std::numeric_limits<std::size_t>::max(), 2), P::GeometryError);
    EXPECT_THROW(D::compact_topology({std::numeric_limits<std::size_t>::max(), 0, 0, 0, 0}), P::GeometryError);
}
TEST(FrozenTopologyTest, FreezeExactCapIncludesDraftOverlapAndOneByteShortLeavesSourceIntact)
{
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto source = storage(root);
    const auto packet = root->encode_certificate();
    // Independently enumerate the cube's new allocations while its mutable
    // draft stays charged: six identity-map nodes, two owners, the unique plane
    // table, five flat index arrays, orientations and eight frozen payloads.
    const std::size_t freeze_bytes =
        6 * (sizeof(std::pair<const D::StoredPlane<ordinary>* const, std::size_t>) + 4 * sizeof(void*)) +
        sizeof(D::FrozenTopology<ordinary>) + sizeof(D::GeometryStorage<ordinary>) +
        6 * sizeof(D::StoredPlanePtr<ordinary>) + (7 + 24 + 9 + 24 + 6) * sizeof(std::uint32_t) + 6 * sizeof(int) +
        8 * sizeof(D::FrozenVertex<ordinary>);
    {
        D::Operation operation(D::mode_limits(ordinary));
        auto draft = D::draft_geometry<ordinary>(*source);
        const auto before = D::active_work->bytes;
        EXPECT_GT(before, 0u);
        (void) D::freeze_geometry<ordinary>(std::move(*draft));
        EXPECT_EQ(D::active_work->bytes - before, freeze_bytes);
    }
    for (const auto shortage : {0u, 1u})
    {
        D::Operation operation(D::mode_limits(ordinary));
        auto draft = D::draft_geometry<ordinary>(*source);
        D::active_work->limits.byte_budget = D::active_work->bytes + freeze_bytes - shortage;
        if (!shortage)
            EXPECT_NO_THROW((void) D::freeze_geometry<ordinary>(std::move(*draft)));
        else
        {
            EXPECT_THROW((void) D::freeze_geometry<ordinary>(std::move(*draft)), P::GeometryError);
            EXPECT_EQ(draft->vertices.size(), source->vertices.size());
            EXPECT_EQ(draft->point_context, source->point_context);
        }
    }
    EXPECT_EQ(root->encode_certificate(), packet);
    EXPECT_EQ(storage(root), source);
}
TEST(FrozenTopologyTest, RebaseSharesTopologyAndLedgerPinsItOnce)
{
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    // A wholly retained cut with a new anchor exercises the rebase publication path.
    const auto retained = root->clip({{1, 0, 0}, {.125, 0, 0}, 2}, false).inside;
    ASSERT_TRUE(retained);
    EXPECT_EQ(storage(root)->topology.get(), storage(retained)->topology.get());
    P::GeometryAllocationLedger ledger;
    std::size_t used = 0;
    constexpr std::size_t cap = 64 * 1024 * 1024;
    ledger.retain(root, used, cap);
    const auto before = ledger.payload_bytes();
    ledger.retain(retained, used, cap);
    EXPECT_LT(ledger.payload_bytes() - before, retained->storage_bytes());
    const auto once = used;
    ledger.retain(retained, used, cap);
    EXPECT_EQ(used, once);
}
TEST(FrozenTopologyTest, AnalyticTetrahedronRetainsItsExplicitFaceSequences)
{
    const std::array<P::GeometryPoint, 4> points{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const std::vector<std::vector<std::size_t>> faces{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    const auto tetra = Geometry<ordinary>::make(points, faces);
    EXPECT_EQ(tetra->faces(), faces);
    EXPECT_EQ(tetra->volume().value, 1.L / 6);
    const auto frozen = storage(tetra);
    EXPECT_EQ(frozen->topology->planes.size(), 4u);
    EXPECT_EQ(std::get<0>(frozen->topology->arrays).face_vertices.size(), 12u);
    EXPECT_EQ(std::get<0>(frozen->topology->arrays).incident_planes.size(), 12u);
}
TEST(FrozenTopologyTest, PartitionCertificatePublicationSharesFrozenTopology)
{
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const std::array donors{root};
    const auto pieces = P::ConvexPartition<ordinary>(donors).partition(*root);
    ASSERT_EQ(pieces.size(), 1u);
    EXPECT_TRUE(storage(pieces[0].geometry)->partition_certificate);
    EXPECT_NE(storage(pieces[0].geometry).get(), storage(root).get());
    EXPECT_EQ(storage(pieces[0].geometry)->topology.get(), storage(root)->topology.get());
}

TEST(FrozenTopologyTest, NativeBindingTransfersPrepaidOwnersWithoutChargingThemTwice)
{
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    const auto original = storage(root);
    std::vector<P::PlaneSupportIdentity<ordinary>> supports;
    supports.reserve(original->surfaces.size());
    for (std::size_t face = 0; face < original->surfaces.size(); ++face)
    {
        ASSERT_TRUE(original->surfaces[face].plane->native_support);
        supports.push_back(*original->surfaces[face].plane->native_support);
    }
    D::Operation operation(D::mode_limits(ordinary));
    auto& work = *D::active_work;
    work.allocations.retain(root, work.bytes, work.limits.byte_budget);
    const auto before = work.bytes, payload = work.allocations.payload_bytes(),
               bookkeeping = work.allocations.bookkeeping_bytes();
    const auto rebound = D::NativeSupportAccess::cell_supports(*root, supports);
    EXPECT_TRUE(root->same_native_representation(*rebound));
    // This path allocates only the freshly retained binding owners and ledger
    // metadata. Their measured payload + bookkeeping must equal total admission.
    EXPECT_EQ(work.bytes - before,
        work.allocations.payload_bytes() - payload + work.allocations.bookkeeping_bytes() - bookkeeping);
    const auto once = work.bytes;
    work.allocations.retain(rebound, work.bytes, work.limits.byte_budget);
    EXPECT_EQ(work.bytes, once);
    EXPECT_EQ(storage(root), original);
}
TEST(FrozenTopologyTest, PreparedTranslatedDraftDoesNotReadmitItsAlreadyPaidBuffers)
{
    const auto root = Geometry<ordinary>::box({0, 0, 0}, {1, 1, 1});
    D::Operation operation(D::mode_limits(ordinary));
    auto& work = *D::active_work;
    work.allocations.retain(root, work.bytes, work.limits.byte_budget);
    auto draft = D::draft_geometry<ordinary>(*storage(root));
    ASSERT_TRUE(draft->buffers_prepaid);
    draft->planes_prepaid = true; // The unchanged plane owners are already retained above.
    draft->moment_cache.reset();
    const auto before = work.bytes;
    D::ordinary_kernel::prepare_finish<ordinary>(*draft, false);
    const auto expected = sizeof(D::StoredMomentCache<ordinary>) + sizeof(P::PointExportCache) +
                          8 * sizeof(P::DynamicStoredPointRef) +
                          6 * (sizeof(D::StoredPlanePtr<ordinary>) + 4 * sizeof(void*));
    EXPECT_EQ(work.bytes - before, expected);
    EXPECT_EQ(draft->vertices.size(), 8u);
    EXPECT_EQ(draft->faces.size(), 6u);
}

} // namespace
