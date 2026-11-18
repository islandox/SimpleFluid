/** Focused producer oracles. Whole-pipeline N/2N and MPI qualification remains
 * in its separate acceptance target; these are component correctness checks. */
#include "geometry/mesh/AMR/AdaptiveGeometrySnapshotAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include "geometry/mesh/AMR/NativePublicationOperation.hh"
#include "geometry/mesh/AMR/NativePublicationNumbering.hh"
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/AMR/NativePublicationViewData.hh"
#include "geometry/mesh/NativeSparseAccess.hh"
#include "native_constructor_allocation_operators.inc"
#include "parallel/details/NativePartitionAccess.hh"
#include "parallel/details/NativeTpetraMapAdmission.hh"
#include "utils/testing_environment.hh"
#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Core.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <exception>
#include <gtest/gtest.h>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
namespace
{
using namespace SimpleFluid;
namespace A = Meshes::AMR;
namespace D = A::adaptive_detail;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const A::DistributedAdaptiveHierarchy> cartesian(A::AMRGrid grid = {})
{
    grid.block_cells = {2, 2, 2};
    grid.max_level = 2;
    A::CartesianAMR source(grid);
    const auto comm = Tpetra::getDefaultComm();
    std::vector<std::uint64_t> cells;
    const auto count = grid.root_blocks[0] * grid.root_blocks[1] * grid.root_blocks[2] * 8;
    for (std::uint64_t c = 0; c < count; ++c)
        if (c % comm->getSize() == std::uint64_t(comm->getRank()))
            cells.push_back(c);
    auto owned = source.detach_owned(cells, {'p', 'r', 'o', 'd', 'u', 'c', 'e', 'r'});
    return std::make_shared<const A::DistributedAdaptiveHierarchy>(
        std::move(owned.owned), std::move(owned.families), std::move(owned.kernel), comm);
}
TEST(NativePublicationProducer, SinglePolyhedralSourceKeepsOriginalAuthorityAndAnalyticMetrics)
{
    ASSERT_TRUE(geometry_detail::native_constructor_capacity_profile_supported());
    auto counter = geometry_detail::NativeConstructionCounterAccess::create<D::Budget>(64 * 1024 * 1024);
    geometry_detail::NativeConstructionAdmission admission(counter);
    A::AdaptiveLeaf leaf;
    leaf.provider = A::AdaptiveProviderKind::Polyhedral;
    leaf.geometry = PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::box({0, 0, 0}, {1, 2, 3});
    const auto original = leaf.geometry;
    const auto prefix = counter.owner()->used;
    native_constructor_test::Capture trace(counter.owner()->used);
    auto fragment = D::NativeSparseAccess::source_cell_fragment(leaf, {6, {.5, 1, 1.5}}, admission);
    const auto raw = trace.finish();
    EXPECT_FALSE(pipeline_trace::overflow.load());
    EXPECT_EQ(native_constructor_test::uncovered, 0u);
    EXPECT_EQ(raw.all_live, counter.owner()->used - prefix);
    const auto& r = std::get<Meshes::DistributedExplicitRegion>(fragment.region);
    ASSERT_EQ(r.layout().cells, 1u);
    EXPECT_EQ(r.layout().faces, 6u);
    EXPECT_EQ(r.layout().nodes, 8u);
    EXPECT_EQ(r.geometry().certified_cell_geometry(0).get(), original.get());
    EXPECT_DOUBLE_EQ(r.geometry().cell_volume(0), 6);
    const auto center = r.geometry().cell_centroid(0);
    EXPECT_DOUBLE_EQ(center.x, .5);
    EXPECT_DOUBLE_EQ(center.y, 1);
    EXPECT_DOUBLE_EQ(center.z, 1.5);
    for (auto f : r.topology().cell_faces(0))
    {
        EXPECT_EQ(r.topology().owner_cell(f), 0u);
        EXPECT_EQ(r.topology().neighbor_cell(f), Meshes::MultiRegionMesh::invalid_cell_id());
        const auto source_face = original->face_geometry(f);
        const auto fragment_face = r.geometry().certified_face_geometry(f);
        EXPECT_EQ(fragment_face->encode_incidence(), source_face->encode_incidence());
        const auto nodes = r.topology().face_nodes(f);
        ASSERT_EQ(nodes.size(), 4u);
        ASSERT_EQ(nodes.size(), source_face->points().size());
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            const auto actual = r.geometry().node_coordinates(nodes[i]);
            EXPECT_EQ((std::array<long double, 3>{actual.x, actual.y, actual.z}), source_face->points()[i]);
        }
    }
    EXPECT_THROW(r.topology().cell_faces(1), std::out_of_range);
}
TEST(NativePublicationProducer, QualifiedViewMatchesFullOracleAndWarmPublicationKeepsFragments)
{
    ASSERT_TRUE(geometry_detail::native_constructor_capacity_profile_supported());
    auto source = cartesian();
    const auto oracle = A::assemble_adaptive_native(*source, source->configuration_descriptor());
    const auto view = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
    ASSERT_EQ(view.owned_gids().size(), oracle.owned_gids.size());
    ASSERT_EQ(view.local_keys().size(), oracle.local_keys.size());
    for (const auto& [key, gid] : oracle.owned_gids)
        EXPECT_EQ(view.owned_gids().at(key), gid);
    for (const auto& [gid, key] : oracle.local_keys)
        EXPECT_EQ(view.local_keys().at(gid), key);
    const auto& expected = *oracle.partition->geometry();
    const auto& actual = *view.partition()->geometry();
    EXPECT_EQ(view.partition()->plan().face_ids(), oracle.partition->plan().face_ids());
    EXPECT_EQ(view.partition()->plan().node_ids(), oracle.partition->plan().node_ids());
    for (const auto gid : view.partition()->plan().cell_ids())
    {
        EXPECT_DOUBLE_EQ(actual.cell_volume(gid), expected.cell_volume(gid));
        EXPECT_EQ(actual.cell_centroid(gid), expected.cell_centroid(gid));
    }
    for (const auto gid : view.partition()->plan().face_ids())
    {
        EXPECT_EQ(actual.owner_cell(gid), expected.owner_cell(gid));
        EXPECT_EQ(actual.neighbor_cell(gid), expected.neighbor_cell(gid));
        EXPECT_EQ(actual.boundary_id(gid), expected.boundary_id(gid));
        EXPECT_DOUBLE_EQ(actual.face_area(gid), expected.face_area(gid));
        EXPECT_EQ(actual.face_centroid(gid), expected.face_centroid(gid));
        EXPECT_EQ(actual.face_normal(gid), expected.face_normal(gid));
    }
    auto bound = source->with_partition(view.partition(), view.mesh()->owned_cell_map());
    D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
    const auto warm = A::publish_adaptive_native(*bound, A::native_publication_schema(*bound));
    const auto work = D::GeometrySnapshotAccess::pipeline_geometry_statistics();
    EXPECT_EQ(work.source_imports, 0u);
    EXPECT_EQ(work.constructions, 0u);
    const auto& repeated = *warm.partition()->geometry();
    ASSERT_EQ(actual.region_count(), repeated.region_count());
    for (std::size_t r = 0; r < actual.region_count(); ++r)
        EXPECT_EQ(&actual.region(r), &repeated.region(r));
}
TEST(NativePublicationProducer, WarmOutputSubsetRetainsExactSharedReceiptsAndReleasesReplacedPayload)
{
    using Charge = PLIC::details::NativePaidExtentCharge;
    using Editor = PLIC::details::GeometryOwnerGraphEditor;
    auto source = cartesian();
    struct Trace
    {
        Trace() { pipeline_trace::begin(); }
        ~Trace() { pipeline_trace::recording.store(false); }
    } trace;
    const auto output_root = [&]
    {
        D::NativePublicationOperation operation(*source);
        return operation.previous_native_root();
    };
    auto first = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
    auto prior = output_root();
    ASSERT_TRUE(prior);
    const auto first_points = prior->payload()->numbering->local_points();
    ASSERT_FALSE(first_points.empty());
    const auto* shared_point = first_points.front().record;
    ASSERT_TRUE(shared_point && shared_point->allocation);
    std::array<const void*, 5> replaced_addresses{};
    std::size_t replaced_blocks = 0;
    {
        const auto& replaced = prior->payload()->allocation;
        replaced->visit_allocations([&](const auto& block) { replaced_addresses.at(replaced_blocks++) = block.address; });
    }
    const auto next = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
    const auto current = output_root();
    ASSERT_TRUE(current);
    auto current_points = current->payload()->numbering->local_points();
    ASSERT_TRUE(std::any_of(current_points.begin(), current_points.end(),
        [&](const auto& point) { return point.record == shared_point; }));
    {
        const auto kept = Charge::fixed(shared_point->allocation);
        const auto retired = Charge::fixed(prior->payload()->allocation);
        ASSERT_TRUE(prior->paid_receipts().contains(kept));
        ASSERT_TRUE(prior->paid_receipts().contains(retired));
        EXPECT_TRUE(current->owners().owners[0].admits(
            0, Editor::external_kind(), shared_point->allocation, kept.bytes()));
        EXPECT_TRUE(current->paid_receipts().contains(kept));
        EXPECT_FALSE(current->paid_receipts().contains(retired));
    }
    first = {};
    prior.reset();
    source.reset();
    pipeline_trace::prepare_census();
    const pipeline_trace::Pause pause;
    EXPECT_FALSE(pipeline_trace::overflow.load());
    for (std::size_t i = 0; i != replaced_blocks; ++i)
        EXPECT_EQ(pipeline_trace::mark(replaced_addresses[i], 1), nullptr);
    std::size_t matched_bytes = 0;
    shared_point->allocation->visit_allocations([&](const auto& block)
        {
            const auto* raw = pipeline_trace::mark(block.address, 2);
            ASSERT_TRUE(raw);
            EXPECT_EQ(raw->address, block.address);
            EXPECT_EQ(raw->bytes, block.bytes);
            matched_bytes += raw->bytes;
        });
    EXPECT_EQ(matched_bytes, Charge::fixed(shared_point->allocation).bytes());
    EXPECT_FALSE(next.owned_gids().empty());
}
TEST(NativePublicationProducer, EqualSchemaBytesFromForeignFactoryDoNotGrantAuthority)
{
    auto first = cartesian(), second = cartesian();
    const auto schema = A::native_publication_schema(*second);
    EXPECT_THROW((void) A::publish_adaptive_native(*first, schema), std::invalid_argument);
}

void same_bits(double first, double second)
{
    EXPECT_EQ(std::bit_cast<std::uint64_t>(first), std::bit_cast<std::uint64_t>(second));
}
void same_vector(MeshUtils::Vec3 first, MeshUtils::Vec3 second)
{
    same_bits(first.x, second.x);
    same_bits(first.y, second.y);
    same_bits(first.z, second.z);
}
template<class Range> std::vector<std::uint64_t> ids(const Range& range)
{
    return {range.begin(), range.end()};
}
void complete_parity(const A::AdaptiveNativePublicationView& view, const A::AdaptiveNativePublication& oracle)
{
    const auto& actual = *view.partition()->geometry();
    const auto& expected = *oracle.partition->geometry();
    EXPECT_EQ(view.partition()->plan().cell_ids(), oracle.partition->plan().cell_ids());
    EXPECT_EQ(view.partition()->plan().face_ids(), oracle.partition->plan().face_ids());
    EXPECT_EQ(view.partition()->plan().node_ids(), oracle.partition->plan().node_ids());
    for (const auto cell : view.partition()->plan().cell_ids())
    {
        EXPECT_EQ(view.local_keys().at(cell), oracle.local_keys.at(cell));
        EXPECT_EQ(ids(actual.cell_faces(cell)), ids(expected.cell_faces(cell)));
        EXPECT_EQ(ids(actual.cell_nodes(cell)), ids(expected.cell_nodes(cell)));
        same_bits(actual.cell_volume(cell), expected.cell_volume(cell));
        same_vector(actual.cell_centroid(cell), expected.cell_centroid(cell));
        const auto [ar, ac] = actual.native_cell(cell);
        const auto [er, ec] = expected.native_cell(cell);
        EXPECT_EQ(ac, ec);
        EXPECT_EQ(std::visit([](const auto& r) { return r.name(); }, actual.region(ar)),
            std::visit([](const auto& r) { return r.name(); }, expected.region(er)));
    }
    EXPECT_EQ(actual.boundary_batch_ids(), expected.boundary_batch_ids());
    for (const auto boundary : actual.boundary_batch_ids())
        EXPECT_EQ(actual.boundary_batch_name(boundary), expected.boundary_batch_name(boundary));
    for (const auto face : view.partition()->plan().face_ids())
    {
        EXPECT_EQ(ids(actual.face_nodes(face)), ids(expected.face_nodes(face)));
        EXPECT_EQ(actual.owner_cell(face), expected.owner_cell(face));
        EXPECT_EQ(actual.neighbor_cell(face), expected.neighbor_cell(face));
        EXPECT_EQ(actual.boundary_id(face), expected.boundary_id(face));
        same_bits(actual.face_area(face), expected.face_area(face));
        same_vector(actual.face_centroid(face), expected.face_centroid(face));
        same_vector(actual.face_area_vector(face), expected.face_area_vector(face));
        const auto af = actual.native_face(face), ef = expected.native_face(face);
        EXPECT_EQ(af.face, ef.face);
        EXPECT_EQ(std::visit([](const auto& r) { return r.name(); }, actual.region(af.region)),
            std::visit([](const auto& r) { return r.name(); }, expected.region(ef.region)));
        same_bits(actual.native_flux_orientation(af, face), expected.native_flux_orientation(ef, face));
        if (!actual.is_exterior_face(face))
        {
            same_vector(actual.cell_center_vector(face, actual.owner_cell(face)),
                expected.cell_center_vector(face, expected.owner_cell(face)));
            same_vector(actual.cell_center_vector(face, actual.neighbor_cell(face)),
                expected.cell_center_vector(face, expected.neighbor_cell(face)));
        }
    }
    for (const auto node : view.partition()->plan().node_ids())
        same_vector(actual.node_coordinates(node), expected.node_coordinates(node));
}
struct Proofs
{
    std::shared_ptr<const D::NativeCellSourceProof> source;
    std::shared_ptr<const D::NativeCellIncidenceProof> incidence;
};
Proofs observe(const A::DistributedAdaptiveHierarchy& hierarchy, const A::AdaptiveCellKey& key)
{
    auto input = D::NativePublicationSession::capture(hierarchy);
    std::lock_guard lock(input.cache->mutex);
    if (!input.cache->indexes)
        throw std::invalid_argument("Built-in publication did not install its actual private source index.");
    const auto row = input.cache->indexes->find(key);
    if (!row)
        throw std::invalid_argument("Built-in publication has no proof for its actual owned key.");
    return {row->source, row->incidence};
}
TEST(NativePublicationProducer, CompleteOriginalOracleIncludesTranslatedMetricBitsAliasesAndPeriodicImages)
{
    ASSERT_TRUE(geometry_detail::native_constructor_capacity_profile_supported());
    for (const bool periodic : {false, true})
    {
        A::AMRGrid grid;
        grid.origin = {-.7, .125, -2.3};
        grid.extent = {1.8, 2.625, 6.2};
        grid.periodic = {periodic, false, false};
        auto hierarchy = cartesian(grid);
        const auto oracle = A::assemble_adaptive_native(*hierarchy, hierarchy->configuration_descriptor());
        const auto view = A::publish_adaptive_native(*hierarchy, A::native_publication_schema(*hierarchy));
        complete_parity(view, oracle);
        ASSERT_TRUE(D::NativePublicationSession::receipt_matches(
            D::NativePublicationSession::capture(*hierarchy), view.partition(), view.mesh()->owned_cell_map()));
        D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
        auto bound = hierarchy->with_partition(view.partition(), view.mesh()->owned_cell_map());
        EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, 0u);
        EXPECT_EQ(bound->native_partition().get(), view.partition().get());
    }
}
struct FixedEditResult
{
    std::uint64_t constructions = 0;
};
FixedEditResult fixed_edit(unsigned blocks)
{
    A::AMRGrid grid;
    grid.root_blocks = {blocks, 1, 1};
    grid.extent = {double(blocks), 1, 1};
    auto source = cartesian(grid);
    auto first = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
    source = source->with_partition(first.partition(), first.mesh()->owned_cell_map());
    const A::AdaptiveCellKey neighbor{grid.region_id, A::CartesianCellAncestry{0, {2, 0, 0}}};
    const A::AdaptiveCellKey far{grid.region_id, A::CartesianCellAncestry{0, {2 * blocks - 1, 1, 1}}};
    const auto old_neighbor = observe(*source, neighbor), old_far = observe(*source, far);
    D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
    const auto warm = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
    auto warmed = source->with_partition(warm.partition(), warm.mesh()->owned_cell_map());
    EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, 0u);
    EXPECT_EQ(observe(*warmed, far).source, old_far.source);
    const A::AdaptiveCellKey edited{grid.region_id, A::CartesianCellAncestry{0, {0, 0, 0}}};
    const std::array marks{A::AdaptiveMark{edited, A::AdaptiveSplit::Isotropic}};
    auto changed = warmed->apply(warmed->plan_adaptation(marks));
    // Capture only actual publication and receipt consumption. Planning,
    // reference oracles and observation/census work stay outside this window.
    D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
    const auto second = A::publish_adaptive_native(*changed, A::native_publication_schema(*changed));
    auto rebound = changed->with_partition(second.partition(), second.mesh()->owned_cell_map());
    const auto work = D::GeometrySnapshotAccess::pipeline_geometry_statistics();
    const auto new_neighbor = observe(*rebound, neighbor), new_far = observe(*rebound, far);
    EXPECT_EQ(new_neighbor.source, old_neighbor.source);
    EXPECT_EQ(new_neighbor.source->fragment(), old_neighbor.source->fragment());
    EXPECT_NE(new_neighbor.incidence, old_neighbor.incidence);
    EXPECT_EQ(new_far.source, old_far.source);
    // A partially repacked immutable contact page can replace an occurrence
    // control while its distant source remains identical. Determine expected
    // incidence reuse from current authoritative membership, independently of
    // the newly issued incidence proof.
    auto current = D::NativePublicationSession::capture(*rebound);
    std::vector<D::NativeSubfaceOccurrence> occurrences;
    for (const auto& face : old_far.source->faces())
        current.subfaces->visit_face_pieces(far, face.surface,
            [&](const auto& piece) { occurrences.push_back(D::NativePublicationSession::occurrence(piece)); });
    const auto same = [](const auto& a, const auto& b)
    {
        return a.slot == b.slot && a.side == b.side && a.control.get() == b.control.get() &&
               !a.control.owner_before(b.control) && !b.control.owner_before(a.control);
    };
    const bool unchanged_incidence = occurrences.size() == old_far.incidence->pieces().size() &&
                                     std::all_of(old_far.incidence->pieces().begin(), old_far.incidence->pieces().end(),
                                         [&](const auto& piece)
                                         {
                                             const auto old = D::NativePublicationSession::occurrence(piece);
                                             return std::any_of(occurrences.begin(), occurrences.end(),
                                                 [&](const auto& now) { return same(old, now); });
                                         });
    if (unchanged_incidence)
        EXPECT_EQ(new_far.incidence, old_far.incidence);
    else
        EXPECT_NE(new_far.incidence, old_far.incidence);
    const auto oracle = A::assemble_adaptive_native(*changed, changed->configuration_descriptor());
    complete_parity(second, oracle);
    return {work.constructions};
}
TEST(NativePublicationProducer, FixedEditReusesUnchangedSourceAndRevalidatesChangedIncidenceAtMatchedSizes)
{
    ASSERT_EQ(Tpetra::getDefaultComm()->getSize(), 1) << "Serial first-proof/reuse gate has a dedicated rank contract.";
    ASSERT_TRUE(geometry_detail::native_constructor_capacity_profile_supported());
    const auto n = fixed_edit(4), twice_n = fixed_edit(8);
    EXPECT_GT(n.constructions, 0u); // New children receive original full first verification.
    EXPECT_EQ(twice_n.constructions, n.constructions);
    RecordProperty("fixed_edit_n32_constructions", std::to_string(n.constructions));
    RecordProperty("fixed_edit_n64_constructions", std::to_string(twice_n.constructions));
}
TEST(NativePublicationProducer, ThreeNativeGenerationsOutliveSourceHierarchiesLayoutsAndContactPages)
{
    ASSERT_EQ(Tpetra::getDefaultComm()->getSize(), 1);
    ASSERT_TRUE(geometry_detail::native_constructor_capacity_profile_supported());
    A::AMRGrid grid;
    grid.root_blocks = {4, 1, 1};
    grid.extent = {4, 1, 1};
    auto source = cartesian(grid);
    std::array<A::AdaptiveNativePublicationView, 3> output;
    std::array<std::weak_ptr<const A::DistributedAdaptiveHierarchy>, 3> weak_source;
    std::array<std::weak_ptr<const A::AdaptiveLeafLayout>, 3> weak_layout;
    std::array<std::weak_ptr<const A::AdaptiveProviderKernel>, 3> weak_kernel;
    std::array<std::weak_ptr<const D::AdaptiveSubfaceSnapshot>, 3> weak_subfaces;
    std::array<std::weak_ptr<const D::NativeSourceCoverage>, 3> weak_coverage;
    std::vector<std::weak_ptr<const void>> weak_contact_pages, weak_contact_controls;
    for (std::size_t generation = 0; generation < output.size(); ++generation)
    {
        output[generation] = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
        auto bound =
            source->with_partition(output[generation].partition(), output[generation].mesh()->owned_cell_map());
        weak_source[generation] = source;
        weak_layout[generation] = source->leaf_layout();
        {
            const auto input = D::NativePublicationSession::capture(*source);
            ASSERT_TRUE(input.source_coverage);
            ASSERT_TRUE(D::NativePublicationSession::receipt_matches(input, output[generation].partition(),
                output[generation].mesh()->owned_cell_map()));
            ASSERT_GT(input.subfaces->size(), 0u);
            weak_kernel[generation] = input.kernel;
            weak_subfaces[generation] = input.subfaces;
            weak_coverage[generation] = input.source_coverage;
            input.subfaces->visit_pieces(
                [&](const auto& view)
                {
                    const auto occurrence = D::NativePublicationSession::occurrence(view);
                    const auto fixed = view.weak_page_observation();
                    ASSERT_TRUE(fixed.matches_live(occurrence.control));
                    ASSERT_TRUE(fixed.contact_control_group());
                    weak_contact_pages.emplace_back(occurrence.control);
                    weak_contact_controls.emplace_back(fixed.contact_control_group()->owner);
                });
        }
        if (generation + 1 < output.size())
        {
            const A::AdaptiveCellKey key{grid.region_id, A::CartesianCellAncestry{0, {generation * 4, 0, 0}}};
            const std::array marks{A::AdaptiveMark{key, A::AdaptiveSplit::Isotropic}};
            source = bound->apply(bound->plan_adaptation(marks));
        }
        else
            source.reset();
    }
    ASSERT_FALSE(weak_contact_pages.empty());
    for (const auto& page : weak_contact_pages)
        EXPECT_TRUE(page.expired());
    weak_contact_pages.clear();
    for (const auto& control : weak_contact_controls)
        EXPECT_FALSE(control.expired());
    for (std::size_t generation = 0; generation < output.size(); ++generation)
    {
        EXPECT_TRUE(weak_source[generation].expired());
        EXPECT_TRUE(weak_layout[generation].expired());
        EXPECT_TRUE(weak_kernel[generation].expired());
        EXPECT_TRUE(weak_subfaces[generation].expired());
        EXPECT_TRUE(weak_coverage[generation].expired());
        const auto& mesh = *output[generation].partition()->geometry();
        for (const auto cell : output[generation].partition()->plan().owned_cells())
            EXPECT_GT(mesh.cell_volume(cell), 0);
        for (const auto face : output[generation].partition()->plan().face_ids())
        {
            EXPECT_GT(mesh.face_area(face), 0);
            EXPECT_GE(mesh.face_nodes(face).size(), 3u);
        }
        EXPECT_GT(output[generation].mesh()->owned_cell_map()->getLocalNumElements(), 0u);
    }
    output = {};
    for (const auto& control : weak_contact_controls)
        EXPECT_TRUE(control.expired());
}
TEST(NativePublicationProducer, EqualGidForeignMapNeverConsumesTheExactResultReceipt)
{
    ASSERT_TRUE(geometry_detail::native_constructor_capacity_profile_supported());
    auto source = cartesian();
    const auto view = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
    const auto original = view.mesh()->owned_cell_map();
    using Map = DefaultTpetraTypes::map_type;
    auto foreign = Teuchos::rcp(new Map(original->getGlobalNumElements(), original->getLocalNumElements(),
        original->getIndexBase(), source->communicator()));
    ASSERT_FALSE(original.shares_resource(foreign));
    EXPECT_FALSE(D::NativePublicationSession::receipt_matches(
        D::NativePublicationSession::capture(*source), view.partition(), foreign));
    D::GeometrySnapshotAccess::reset_pipeline_geometry_statistics();
    bool accepted = false;
    try
    {
        const auto bound = source->with_partition(view.partition(), foreign);
        accepted = bool(bound);
        EXPECT_FALSE(D::NativePublicationSession::capture(*bound).source_coverage);
    }
    catch (const std::invalid_argument&)
    {
    }
    if (accepted)
        EXPECT_GT(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, 0u);
    EXPECT_TRUE(D::NativePublicationSession::receipt_matches(
        D::NativePublicationSession::capture(*source), view.partition(), original));
    auto same_address_observation = D::NativePublicationSession::capture(*source);
    same_address_observation.subfaces =
        std::shared_ptr<const D::AdaptiveSubfaceSnapshot>(same_address_observation.subfaces.get(), [](const auto*) {});
    EXPECT_FALSE(D::NativePublicationSession::receipt_matches(
        same_address_observation, view.partition(), original));
}
TEST(NativePublicationProducer, ActualBoundInputRemainsAdmittedWhenPublicationOutputChanges)
{
    auto source = cartesian();
    auto first = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
    auto bound = source->with_partition(first.partition(), first.mesh()->owned_cell_map());
    const auto a_allocation = geometry_detail::NativePartitionAccess::allocation(first.partition());
    const std::weak_ptr<const CompositePartition<>> weak_input = first.partition();
    const std::weak_ptr<const A::AdaptiveLeafLayout> weak_layout = bound->leaf_layout();
    const std::weak_ptr<const void> weak_owned_map_token = bound->observation().partition_identity;
    first = {};
    source.reset();

    auto second = A::publish_adaptive_native(*bound, A::native_publication_schema(*bound));
    const auto b_allocation = geometry_detail::NativePartitionAccess::allocation(second.partition());
    ASSERT_TRUE(a_allocation);
    ASSERT_TRUE(b_allocation);
    ASSERT_NE(bound->native_partition(), second.partition());
    {
        D::NativePublicationOperation operation(*bound);
        auto& accounting = operation.geometry_accounting();
        EXPECT_NO_THROW(accounting.authorize_fixed(a_allocation));
        EXPECT_NO_THROW(accounting.authorize_fixed(b_allocation));
    }

    const auto third = A::publish_adaptive_native(*bound, A::native_publication_schema(*bound));
    std::size_t imported_with_previous_output = 0;
    {
        D::NativePublicationOperation operation(*bound);
        auto& accounting = operation.geometry_accounting();
        const auto* map_token_allocation = operation.inputs().fixed_observations.ownership.native_envelope();
        ASSERT_TRUE(map_token_allocation);
        EXPECT_NO_THROW(accounting.authorize_fixed(a_allocation));
        EXPECT_NO_THROW(accounting.authorize_fixed(*map_token_allocation));
        EXPECT_THROW(accounting.authorize_fixed(b_allocation), PLIC::GeometryError);
        imported_with_previous_output = operation.admission().statistics().imported_live;
    }
    EXPECT_EQ(second.owned_gids().size(), third.owned_gids().size());
    second = {};
    {
        D::NativePublicationOperation operation(*bound);
        EXPECT_LT(operation.admission().statistics().imported_live, imported_with_previous_output);
    }
    EXPECT_EQ(weak_input.lock(), bound->native_partition());
    EXPECT_EQ(weak_owned_map_token.lock(), bound->observation().partition_identity);
    bound.reset();
    EXPECT_TRUE(weak_input.expired());
    EXPECT_TRUE(weak_layout.expired());
    EXPECT_TRUE(weak_owned_map_token.expired());
    EXPECT_GT(third.owned_gids().size(), 0u);
}

struct WarmPublicationSource
{
    A::AdaptiveNativePublicationView bound_input, accepted;
    std::shared_ptr<const A::DistributedAdaptiveHierarchy> hierarchy;
};

WarmPublicationSource warm_publication_source()
{
    const auto source = cartesian();
    auto bound_input = A::publish_adaptive_native(*source, A::native_publication_schema(*source));
    auto hierarchy = source->with_partition(bound_input.partition(), bound_input.mesh()->owned_cell_map());
    auto accepted = A::publish_adaptive_native(*hierarchy, A::native_publication_schema(*hierarchy));
    return {std::move(bound_input), std::move(accepted), std::move(hierarchy)};
}

class ScopedArenaLimit
{
public:
    ScopedArenaLimit(D::Budget& arena, std::size_t limit) : arena_(arena), limit_(arena.limit), peak_(arena.peak)
    {
        arena_.limit = limit;
        arena_.peak = arena_.used;
    }
    ~ScopedArenaLimit()
    {
        arena_.limit = limit_;
        arena_.peak = peak_;
    }
    ScopedArenaLimit(const ScopedArenaLimit&) = delete;
    ScopedArenaLimit& operator=(const ScopedArenaLimit&) = delete;

private:
    D::Budget& arena_;
    std::size_t limit_, peak_;
};

enum class PublicationTrialMode
{
    ShortCounter,
    FailResultAllocation,
    ExactCounter
};

struct PublicationRequests
{
    std::size_t counter = 0;
    std::size_t mirror_before_result = 0, result_request = 0, mirror_at_result_request = 0;
    bool result_phase_entered = false, counter_limit_restored = false;
};

class PublicationRequestProbe
{
public:
    PublicationRequestProbe(D::Budget& arena, PublicationTrialMode mode, std::size_t original_limit)
        : arena_(arena), mode_(mode), original_limit_(original_limit), used_(arena.used),
          mirrored_(arena.publication_mirror_live),
          previous_fault_(std::exchange(native_constructor_test::fail_after, std::nullopt)),
          previous_request_(std::exchange(
              native_constructor_test::allocation_request_observer, {this, observe_request})),
          previous_phase_(D::set_pipeline_observer({this, observe_phase, nullptr}))
    {
    }
    ~PublicationRequestProbe()
    {
        (void) D::set_pipeline_observer(previous_phase_);
        native_constructor_test::allocation_request_observer = previous_request_;
        native_constructor_test::fail_after = previous_fault_;
    }
    PublicationRequestProbe(const PublicationRequestProbe&) = delete;
    PublicationRequestProbe& operator=(const PublicationRequestProbe&) = delete;
    PublicationRequests requests;

private:
    static void observe_request(void* context, std::size_t bytes) noexcept
    {
        auto& self = *static_cast<PublicationRequestProbe*>(context);
        auto& arena = self.arena_;
        if (native_constructor_test::admitted_used != &arena.used)
            return;
        if (bytes == sizeof(D::Budget) && arena.used == self.used_ + bytes &&
            arena.publication_mirror_live == self.mirrored_ + bytes)
        {
            ++self.requests.counter;
            if (self.mode_ == PublicationTrialMode::ExactCounter && arena.limit == arena.used)
            {
                // The named counter request has already passed its exact
                // mirror cap. Later requests use the original production limit.
                self.requests.counter_limit_restored = true;
                arena.limit = self.original_limit_;
            }
        }
        if (self.mode_ == PublicationTrialMode::FailResultAllocation && self.requests.result_phase_entered &&
            native_constructor_test::fail_after == 0)
        {
            self.requests.result_request = bytes;
            self.requests.mirror_at_result_request = arena.publication_mirror_live;
        }
    }
    static void observe_phase(void* context, D::PipelinePhase phase) noexcept
    {
        auto& self = *static_cast<PublicationRequestProbe*>(context);
        pipeline_trace::phase = static_cast<pipeline_trace::Phase>(phase);
        if (phase != D::PipelinePhase::NativePublication || self.requests.result_phase_entered)
            return;
        self.requests.result_phase_entered = true;
        self.requests.mirror_before_result = self.arena_.publication_mirror_live;
        if (self.mode_ == PublicationTrialMode::FailResultAllocation)
            native_constructor_test::fail_after = 0;
    }

    D::Budget& arena_;
    PublicationTrialMode mode_;
    std::size_t original_limit_, used_, mirrored_;
    std::optional<std::size_t> previous_fault_;
    native_constructor_test::AllocationRequestObserver previous_request_;
    D::PipelineObserver previous_phase_;
};

struct PublicationTrial
{
    A::AdaptiveNativePublicationView view;
    std::exception_ptr rejection;
    pipeline_trace::Summary trace;
    PublicationRequests requests;
    std::size_t peak = 0;
    bool trace_overflow = false;
};

PublicationTrial run_publication_trial(const A::DistributedAdaptiveHierarchy& hierarchy, PublicationTrialMode mode)
{
    const auto schema = A::native_publication_schema(hierarchy);
    const auto arena = D::GeometrySnapshotAccess::contact_cache(hierarchy).allocation_arena();
    const auto original_limit = arena->limit;
    const auto limit = mode == PublicationTrialMode::FailResultAllocation
        ? original_limit
        : arena->used + sizeof(D::Budget) - (mode == PublicationTrialMode::ShortCounter ? 1 : 0);
    ScopedArenaLimit scoped(*arena, limit);
    PublicationRequestProbe probe(*arena, mode, original_limit);
    PublicationTrial result;
    native_constructor_test::Capture trace(arena->used);
    try
    {
        result.view = A::publish_adaptive_native(hierarchy, schema);
    }
    catch (...)
    {
        result.rejection = std::current_exception();
    }
    result.trace = trace.finish();
    result.requests = probe.requests;
    result.trace_overflow = pipeline_trace::overflow.load();
    result.peak = arena->peak;
    return result;
}

std::array<PLIC::details::GeometryOwnerRootBinding, 2> publication_roots(
    const A::DistributedAdaptiveHierarchy& hierarchy)
{
    std::array<PLIC::details::GeometryOwnerRootBinding, 2> roots;
    std::size_t i = 0;
    D::GeometrySnapshotAccess::visit_native_cache_bindings(hierarchy,
        [&](const auto& binding) { roots.at(i++) = binding; }, [](const auto&) {});
    return roots;
}

std::size_t requested_allocations(const PublicationTrial& trial)
{
    return std::accumulate(trial.trace.phase.begin(), trial.trace.phase.end(), std::size_t{0},
        [](std::size_t count, const auto& phase) { return count + phase.allocations; });
}

TEST(NativePublicationProducer, WarmCartesianPublicationExactCounterAdmissionAndPostMapRollback)
{
    ASSERT_EQ(Tpetra::getDefaultComm()->getSize(), 1);
    using Maps = geometry_detail::NativeTpetraMapAdmission;
    ASSERT_NO_THROW(Maps::require_profile());
    const auto target = warm_publication_source();
    const auto arena = D::GeometrySnapshotAccess::contact_cache(*target.hierarchy).allocation_arena();
    const auto admitted = arena->used, mirror_live = arena->publication_mirror_live;
    const auto original_limit = arena->limit;
    const auto exact_counter_limit = admitted + sizeof(D::Budget);
    ASSERT_LE(exact_counter_limit, original_limit);

    const auto accepted_roots = publication_roots(*target.hierarchy);
    for (const auto& root : accepted_roots)
        ASSERT_TRUE(root);
    const auto input = D::NativePublicationSession::capture(*target.hierarchy);
    ASSERT_TRUE(input.source_coverage);
    ASSERT_TRUE(D::NativePublicationSession::receipt_matches(
        input, target.accepted.partition(), target.accepted.mesh()->owned_cell_map()));
    const auto indexes = input.cache->indexes;
    const auto receipt = input.cache->receipt;
    const auto revision = input.cache->revision;
    const auto storage = D::GeometrySnapshotAccess::cache_admission_statistics(*target.hierarchy);

    const auto expect_rollback = [&]
    {
        const auto after = D::NativePublicationSession::capture(*target.hierarchy);
        const auto after_roots = publication_roots(*target.hierarchy);
        const auto after_storage = D::GeometrySnapshotAccess::cache_admission_statistics(*target.hierarchy);
        EXPECT_EQ(arena->used, admitted);
        EXPECT_EQ(arena->publication_mirror_live, mirror_live);
        EXPECT_EQ(arena->limit, original_limit);
        EXPECT_EQ(after_storage.resident_bytes, storage.resident_bytes);
        EXPECT_EQ(after_storage.lease_bytes, storage.lease_bytes);
        EXPECT_EQ(after.observation, input.observation);
        EXPECT_EQ(after.layout, input.layout);
        EXPECT_EQ(after.subfaces, input.subfaces);
        EXPECT_EQ(after.ownership_owner, input.ownership_owner);
        EXPECT_EQ(after.source_coverage, input.source_coverage);
        EXPECT_EQ(after.cache, input.cache);
        EXPECT_EQ(after.cache->indexes, indexes);
        EXPECT_EQ(after.cache->receipt, receipt);
        EXPECT_EQ(after.cache->revision, revision);
        EXPECT_EQ(after.cache->commit_token, nullptr);
        for (std::size_t i = 0; i < accepted_roots.size(); ++i)
        {
            EXPECT_EQ(after_roots[i].header(), accepted_roots[i].header());
            EXPECT_FALSE(after_roots[i].header().owner_before(accepted_roots[i].header()));
            EXPECT_FALSE(accepted_roots[i].header().owner_before(after_roots[i].header()));
            for (std::size_t channel = 0; channel < 2; ++channel)
                EXPECT_TRUE(
                    after_roots[i].payload().owners[channel].same_root(accepted_roots[i].payload().owners[channel]));
        }
        EXPECT_EQ(target.hierarchy->native_partition(), target.bound_input.partition());
        EXPECT_TRUE(D::NativePublicationSession::receipt_matches(
            after, target.accepted.partition(), target.accepted.mesh()->owned_cell_map()));
        EXPECT_FALSE(D::NativePublicationSession::receipt_matches(
            after, target.accepted.partition(), target.bound_input.mesh()->owned_cell_map()));
    };

    const auto short_trial = run_publication_trial(*target.hierarchy, PublicationTrialMode::ShortCounter);
    ASSERT_TRUE(short_trial.rejection);
    EXPECT_FALSE(short_trial.trace_overflow);
    EXPECT_EQ(short_trial.peak, admitted);
    EXPECT_EQ(short_trial.requests.counter, 0u);
    EXPECT_FALSE(short_trial.requests.result_phase_entered);
    try
    {
        std::rethrow_exception(short_trial.rejection);
    }
    catch (const std::invalid_argument& rejected)
    {
        EXPECT_STREQ(rejected.what(), "Native counter object exceeds source arena headroom.");
        RecordProperty("short_counter_rejection", rejected.what());
    }
    expect_rollback();

    const auto late = run_publication_trial(*target.hierarchy, PublicationTrialMode::FailResultAllocation);
    ASSERT_TRUE(late.rejection);
    EXPECT_THROW(std::rethrow_exception(late.rejection), std::bad_alloc);
    EXPECT_FALSE(late.trace_overflow);
    EXPECT_EQ(late.requests.counter, 1u);
    EXPECT_TRUE(late.requests.result_phase_entered);
    EXPECT_GT(late.requests.mirror_before_result, mirror_live + sizeof(D::Budget));
    EXPECT_EQ(late.requests.result_request, sizeof(geometry_detail::native_envelope_detail::Record));
    EXPECT_EQ(late.requests.mirror_at_result_request, late.requests.mirror_before_result + late.requests.result_request);
    expect_rollback();

    const auto retry_trial = run_publication_trial(*target.hierarchy, PublicationTrialMode::ExactCounter);
    ASSERT_NO_THROW(if (retry_trial.rejection) std::rethrow_exception(retry_trial.rejection));
    EXPECT_FALSE(retry_trial.trace_overflow);
    EXPECT_EQ(retry_trial.requests.counter, 1u);
    EXPECT_TRUE(retry_trial.requests.counter_limit_restored);
    EXPECT_TRUE(retry_trial.requests.result_phase_entered);
    const auto retry = D::NativePublicationSession::capture(*target.hierarchy);
    ASSERT_TRUE(D::NativePublicationSession::receipt_matches(
        retry, retry_trial.view.partition(), retry_trial.view.mesh()->owned_cell_map()));
    EXPECT_EQ(retry.cache->revision, revision + 1);
    const auto before_bind = D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions;
    auto bound = target.hierarchy->with_partition(retry_trial.view.partition(), retry_trial.view.mesh()->owned_cell_map());
    EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, before_bind);
    EXPECT_EQ(bound->native_partition(), retry_trial.view.partition());
    const auto expected = A::assemble_adaptive_native(*target.hierarchy, target.hierarchy->configuration_descriptor());
    complete_parity(retry_trial.view, expected);
    complete_parity(target.accepted, expected);

    const auto roots = publication_roots(*target.hierarchy);
    ASSERT_TRUE(roots[1]);
    const auto root = std::static_pointer_cast<const D::NativePublicationRoot>(roots[1].header());
    const auto measurements = root->allocation()->state()->measurements();
    RecordProperty("mirror_baseline_live", std::to_string(admitted));
    RecordProperty("short_counter_limit", std::to_string(exact_counter_limit - 1));
    RecordProperty("exact_counter_limit", std::to_string(exact_counter_limit));
    RecordProperty("counter_requested_bytes", std::to_string(sizeof(D::Budget)));
    RecordProperty("short_counter_object_requests", std::to_string(short_trial.requests.counter));
    RecordProperty("short_trial_cpp_requests", std::to_string(requested_allocations(short_trial)));
    RecordProperty("late_failure_cpp_requests", std::to_string(requested_allocations(late)));
    RecordProperty("late_failed_request_bytes", std::to_string(late.requests.result_request));
    RecordProperty("late_mirror_live_before_result", std::to_string(late.requests.mirror_before_result));
    RecordProperty("retry_counter_object_requests", std::to_string(retry_trial.requests.counter));
    RecordProperty("retry_cpp_requests", std::to_string(requested_allocations(retry_trial)));
    RecordProperty("retry_cpp_peak", std::to_string(retry_trial.trace.all_peak));
    RecordProperty("retry_observed_mirror_peak", std::to_string(retry_trial.peak));
    RecordProperty("retry_admitted_cpp_cumulative", std::to_string(measurements.actual_cumulative));
    RecordProperty("retry_library_context_upper", std::to_string(Maps::library_context_bytes()));
    RecordProperty("retry_operation_profiled_upper", std::to_string(measurements.profiled_cumulative_upper));
    RecordProperty("retry_logical_ledger_work", std::to_string(measurements.ledger_work_bytes));
}

template<class T> T rank_sum(const Teuchos::Comm<int>& comm, T local)
{
    T total{};
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, 1, &local, &total);
    return total;
}

struct DistributedPublication
{
    A::AdaptiveNativePublicationView view;
    A::AdaptiveNativePublication oracle;
    std::shared_ptr<const A::DistributedAdaptiveHierarchy> bound;
};

DistributedPublication publish_and_bind(const A::DistributedAdaptiveHierarchy& source, std::uint64_t global_cells,
    std::size_t local_cells, int empty_owners)
{
    const auto& comm = *source.communicator();
    auto view = A::publish_adaptive_native(source, A::native_publication_schema(source));
    D::collective(comm,
        [&]
        {
            if (!D::NativePublicationSession::receipt_matches(
                    D::NativePublicationSession::capture(source), view.partition(), view.mesh()->owned_cell_map()))
                throw std::invalid_argument("Distributed fixture requires a qualified publication receipt.");
        });
    const int empty = view.mesh()->owned_cell_map()->getLocalNumElements() == 0;
    EXPECT_EQ(rank_sum(comm, empty), empty_owners);
    const auto before_bind = D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions;
    auto bound = source.with_partition(view.partition(), view.mesh()->owned_cell_map());
    EXPECT_EQ(D::GeometrySnapshotAccess::pipeline_geometry_statistics().constructions, before_bind);
    A::AdaptiveBytes schema;
    D::collective(comm, [&] { schema = source.configuration_descriptor(); });
    auto oracle = A::assemble_adaptive_native(source, schema);
    D::collective(comm,
        [&]
        {
            EXPECT_EQ(source.global_cells(), global_cells);
            EXPECT_EQ(source.owned_leaves().size(), local_cells);
            EXPECT_EQ(bound->native_partition(), view.partition());
            EXPECT_EQ(view.mesh()->owned_cell_map()->getGlobalNumElements(), global_cells);
            EXPECT_EQ(view.mesh()->owned_cell_map()->getLocalNumElements(), local_cells);
            EXPECT_EQ(view.owned_gids().size(), local_cells);
            if (!local_cells)
                EXPECT_TRUE(view.local_keys().empty());
            complete_parity(view, oracle);
        });
    return {std::move(view), std::move(oracle), std::move(bound)};
}

Proofs distributed_proof(const A::DistributedAdaptiveHierarchy& source, const A::AdaptiveCellKey& key, int owner)
{
    const auto& comm = *source.communicator();
    Proofs result;
    D::collective(comm,
        [&]
        {
            if (comm.getRank() >= 2)
                return;
            result = observe(source, key);
            if (!result.source || !result.incidence)
                throw std::invalid_argument("Distributed fixture requires both accepted source and incidence proofs.");
            EXPECT_EQ(result.source->source().key, key);
            EXPECT_EQ(result.source->ticket().issuer, owner);
            EXPECT_EQ(result.incidence->ticket().issuer, owner);
            EXPECT_EQ(&result.incidence->source(), result.source.get());
            EXPECT_EQ(bool(result.incidence->remote_record()), comm.getRank() != owner);
            EXPECT_EQ(source.owned_leaves().contains(key), comm.getRank() == owner);
        });
    return result;
}

void expect_left_neighbors(const DistributedPublication& publication, const A::AdaptiveCellKey& right,
    std::span<const A::AdaptiveCellKey> expected, std::size_t boundary_loop_size)
{
    const auto& comm = *publication.bound->communicator();
    const auto lookup = publication.bound->lookup_cells(std::span(&right, 1));
    D::collective(comm,
        [&]
        {
            const auto& cell = lookup.at(0);
            EXPECT_EQ(cell.owner, 1);
            if (comm.getRank() >= 2)
                return;
            const auto& mesh = *publication.view.partition()->geometry();
            const auto gid = cell.address;
            EXPECT_EQ(publication.view.local_keys().at(gid), right);
            std::vector<A::AdaptiveCellKey> neighbors;
            double area = 0;
            std::size_t lower_boundary_faces = 0, edge_midpoints = 0;
            for (const auto face : mesh.cell_faces(gid))
            {
                if (mesh.is_exterior_face(face) && mesh.face_centroid(face).y == 0)
                {
                    ++lower_boundary_faces;
                    const auto nodes = mesh.face_nodes(face);
                    EXPECT_EQ(nodes.size(), boundary_loop_size);
                    for (const auto node : nodes)
                        edge_midpoints += mesh.node_coordinates(node) == MeshUtils::Vec3{1, 0, .25};
                }
                const auto first = mesh.owner_cell(face);
                const auto other = first == gid ? mesh.neighbor_cell(face) : first;
                if (other != Meshes::MultiRegionMesh::invalid_cell_id() &&
                    mesh.cell_centroid(other).x < mesh.cell_centroid(gid).x)
                {
                    neighbors.push_back(publication.view.local_keys().at(other));
                    area += mesh.face_area(face);
                }
            }
            std::sort(neighbors.begin(), neighbors.end());
            EXPECT_TRUE(std::ranges::equal(neighbors, expected));
            EXPECT_DOUBLE_EQ(area, .25);
            EXPECT_EQ(lower_boundary_faces, 1u);
            EXPECT_EQ(edge_midpoints, boundary_loop_size - 4);
        });
}

TEST(NativePublicationProducerMPI, CartesianRemoteReuseRefinementMigrationAndEmptyOwners)
{
    const auto comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank(), ranks = comm->getSize();
    ASSERT_TRUE(ranks == 2 || ranks == 4);
    ASSERT_TRUE(geometry_detail::native_constructor_capacity_profile_supported());
    A::AMRGrid grid;
    grid.root_blocks = {2, 1, 1};
    grid.block_cells = {2, 2, 2};
    grid.extent = {2, 1, 1};
    grid.max_level = 2;
    const auto cell = [&](unsigned level, std::uint64_t x, std::uint64_t y, std::uint64_t z)
    { return A::AdaptiveCellKey{grid.region_id, A::CartesianCellAncestry{level, {x, y, z}}}; };
    const auto right = cell(0, 2, 0, 0);
    const std::array coarse_left{cell(0, 1, 0, 0)};
    const std::array fine_left{cell(1, 3, 0, 0), cell(1, 3, 0, 1), cell(1, 3, 1, 0), cell(1, 3, 1, 1)};
    A::AdaptiveBootstrap detached;
    D::collective(*comm,
        [&]
        {
            const A::CartesianAMR factory(grid);
            std::vector<std::uint64_t> owned;
            for (std::uint64_t gid = 0; gid < 16; ++gid)
            {
                const auto key = factory.cell_key(gid);
                const auto& ancestry = std::get<A::CartesianCellAncestry>(key.ancestry);
                if (rank == (ancestry.index[0] < 2 ? 0 : 1))
                    owned.push_back(gid);
            }
            detached = factory.detach_owned(owned, {'p', 'u', 'b', 'l', 'i', 's', 'h'});
        });
    const A::DistributedAdaptiveHierarchy source(
        std::move(detached.owned), std::move(detached.families), std::move(detached.kernel), comm);
    const auto first = publish_and_bind(source, 16, rank < 2 ? 8 : 0, ranks - 2);
    const auto original_right = distributed_proof(*first.bound, right, 1);
    const auto warm = publish_and_bind(*first.bound, 16, rank < 2 ? 8 : 0, ranks - 2);
    const auto warm_right = distributed_proof(*warm.bound, right, 1);
    if (rank < 2)
    {
        EXPECT_EQ(warm_right.source, original_right.source);
        EXPECT_EQ(warm_right.incidence, original_right.incidence);
        EXPECT_FALSE(warm_right.source.owner_before(original_right.source));
        EXPECT_FALSE(original_right.source.owner_before(warm_right.source));
        EXPECT_FALSE(warm_right.incidence.owner_before(original_right.incidence));
        EXPECT_FALSE(original_right.incidence.owner_before(warm_right.incidence));
    }
    expect_left_neighbors(warm, right, coarse_left, 4);

    std::vector<A::AdaptiveMark> marks;
    D::collective(*comm,
        [&]
        {
            if (rank == 0)
                for (const auto& leaf : warm.bound->owned_leaves())
                    marks.push_back({leaf.key, A::AdaptiveSplit::Isotropic});
        });
    EXPECT_EQ(rank_sum(*comm, int(marks.size())), 8);
    const auto refined = warm.bound->apply(warm.bound->plan_adaptation(marks));
    const auto changed = publish_and_bind(*refined, 72, rank == 0 ? 64 : (rank == 1 ? 8 : 0), ranks - 2);
    const auto changed_right = distributed_proof(*changed.bound, right, 1);
    if (rank < 2)
    {
        EXPECT_EQ(changed_right.source, warm_right.source);
        EXPECT_EQ(changed_right.source->ticket(), warm_right.source->ticket());
        EXPECT_NE(changed_right.incidence, warm_right.incidence);
        EXPECT_NE(changed_right.incidence->ticket(), warm_right.incidence->ticket());
        if (rank == 0)
            EXPECT_NE(changed_right.incidence->remote_record(), warm_right.incidence->remote_record());
    }
    // Refinement adds one collinear point on the right ghost's y-min face.
    // On rank 0 this face exists only through the full remote incidence reply.
    expect_left_neighbors(changed, right, fine_left, 5);

    const auto& migrated_key = fine_left.front();
    const auto before = changed.bound->lookup_cells(std::span(&migrated_key, 1));
    const auto before_move = distributed_proof(*changed.bound, migrated_key, 0);
    std::vector<int> destinations;
    D::collective(*comm,
        [&]
        {
            EXPECT_EQ(before.at(0).owner, 0);
            for (const auto& leaf : changed.bound->owned_leaves())
                destinations.push_back(leaf.key == migrated_key ? 1 : rank);
        });
    const auto plan = changed.bound->plan_migration(destinations);
    EXPECT_GT(rank_sum(*comm, plan.native_migration().remote_sent_packets), 0u);
    const auto migrated = changed.bound->apply(plan);
    const auto moved = publish_and_bind(*migrated, 72, rank == 0 ? 63 : (rank == 1 ? 9 : 0), ranks - 2);
    const auto after = moved.bound->lookup_cells(std::span(&migrated_key, 1));
    const auto after_move = distributed_proof(*moved.bound, migrated_key, 1);
    D::collective(*comm,
        [&]
        {
            const auto& owner = after.at(0);
            EXPECT_EQ(owner.owner, 1);
            EXPECT_EQ(owner.revision, before.at(0).revision);
            EXPECT_EQ(moved.view.owned_gids().contains(migrated_key), rank == 1);
            const auto gid = static_cast<DefaultTpetraTypes::global_ordinal_type>(owner.address);
            EXPECT_EQ(moved.view.mesh()->owned_cell_map()->isNodeGlobalElement(gid), rank == 1);
            if (rank == 1)
                EXPECT_EQ(moved.view.owned_gids().at(migrated_key), owner.address);
            if (rank < 2)
            {
                EXPECT_EQ(after_move.source->source().revision, owner.revision);
                EXPECT_NE(after_move.source->ticket(), before_move.source->ticket());
                EXPECT_EQ(moved.view.local_keys().at(owner.address), migrated_key);
                const auto& routes = moved.view.partition()->plan();
                const auto at = std::ranges::find(routes.cell_ids(), owner.address);
                if (at == routes.cell_ids().end())
                    throw std::invalid_argument("Published migration route omits the owner-confirmed cell.");
                EXPECT_EQ(routes.cell_owner_ranks().at(at - routes.cell_ids().begin()), 1);
            }
        });
    expect_left_neighbors(moved, right, fine_left, 5);

    // Exercise an empty owner even with a two-rank communicator.
    std::vector<int> all_to_zero;
    D::collective(*comm, [&] { all_to_zero.assign(moved.bound->owned_leaves().size(), 0); });
    const auto gather = moved.bound->plan_migration(all_to_zero);
    EXPECT_GT(rank_sum(*comm, gather.native_migration().remote_sent_packets), 0u);
    const auto gathered = moved.bound->apply(gather);
    const auto final = publish_and_bind(*gathered, 72, rank == 0 ? 72 : 0, ranks - 1);
    D::collective(*comm,
        [&]
        {
            EXPECT_EQ(final.view.local_keys().size(), rank == 0 ? 72u : 0u);
            complete_parity(first.view, first.oracle);
            complete_parity(changed.view, changed.oracle);
            complete_parity(moved.view, moved.oracle);
        });
}
} // namespace
