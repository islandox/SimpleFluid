#include "cartesian_incidence_test_helpers.hh"
#include "geometry/mesh/AMR/AdaptiveContactSnapshot.hh"
#include "geometry/mesh/AMR/AdaptiveSubfaceSnapshot.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveContactHalo.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveHierarchy.hh"
#include "utils/testing_environment.hh"
#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <map>
#include <optional>
#include <tuple>

namespace
{
using namespace cartesian_incidence_test;
using Hierarchy = DistributedAdaptiveHierarchy;
using Owner = std::shared_ptr<const Hierarchy>;
using Coordinates = std::array<long double, 3>;
using Witness = std::tuple<AdaptiveCellKey, int, std::uint64_t, Coordinates, Coordinates, Coordinates>;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
struct Gate
{
    std::optional<std::string> saved;
    explicit Gate(const char* value)
    {
        if (const auto* old = std::getenv("SIMPLEFLUID_CARTESIAN_CONTACT"))
            saved = old;
        if (value)
            ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", value, 1);
        else
            ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
    }
    ~Gate()
    {
        if (saved)
            ::setenv("SIMPLEFLUID_CARTESIAN_CONTACT", saved->c_str(), 1);
        else
            ::unsetenv("SIMPLEFLUID_CARTESIAN_CONTACT");
    }
};
AMRGrid config()
{
    return grid();
}
int owner_of(const AdaptiveCellKey& key, int ranks)
{
    return ranks == 1 ? 0 : int(std::get<CartesianCellAncestry>(key.ancestry).index[0] % 2);
}
std::vector<AdaptiveLeaf> all_leaves(const AMRGrid& value)
{
    std::vector<AdaptiveLeaf> all;
    const auto nx = value.root_blocks[0] * value.block_cells[0];
    const auto ny = value.root_blocks[1] * value.block_cells[1];
    const auto nz = value.root_blocks[2] * value.block_cells[2];
    for (std::uint64_t z = 0; z < nz; ++z)
        for (std::uint64_t y = 0; y < ny; ++y)
            for (std::uint64_t x = 0; x < nx; ++x)
            {
                auto leaf =
                    cartesian_adaptive_leaf(value, {value.region_id, CartesianCellAncestry{0, {x, y, z}}}, {'d'});
                leaf.revision = 101 + x + nx * (y + ny * z);
                all.push_back(std::move(leaf));
            }
    return all;
}
Owner source(const AMRGrid& value)
{
    const auto comm = Tpetra::getDefaultComm();
    std::vector<AdaptiveLeaf> owned;
    for (auto& leaf : all_leaves(value))
        if (owner_of(leaf.key, comm->getSize()) == comm->getRank())
            owned.push_back(std::move(leaf));
    return std::make_shared<const Hierarchy>(
        std::move(owned), std::vector<AdaptiveFamily>{}, cartesian_adaptive_kernel(value), comm);
}
bool procedural(const AdaptiveContactSnapshot& snapshot)
{
    bool result = false;
    snapshot.visit_borrowed_payload([&](const ContactPayloadAllocation& allocation)
        { result |= allocation.identity.role == GeometryContactPayloadRole::CartesianContactStateObject; });
    return result;
}
std::vector<Witness> witnesses(const AdaptiveContactSnapshot& snapshot, const AdaptiveCellKey& key)
{
    std::vector<Witness> result;
    snapshot.visit_closed_contacts(key,
        [&](const ContactWitness& witness)
        {
            EXPECT_TRUE(std::ranges::equal(witness.domain, std::array{'d'}));
            result.emplace_back(witness.neighbor, witness.owner, witness.revision,
                PLIC::details::GeometryIdentityAccess::point(witness.normalized),
                PLIC::details::GeometryIdentityAccess::point(witness.local_physical),
                PLIC::details::GeometryIdentityAccess::point(witness.neighbor_physical));
        });
    std::sort(result.begin(), result.end());
    return result;
}
std::vector<Witness> lattice(const AdaptiveCellKey& key, const AMRGrid& value, int ranks)
{
    const auto& local = std::get<CartesianCellAncestry>(key.ancestry).index;
    std::vector<Witness> result;
    for (const auto& other : all_leaves(value))
    {
        if (other.key == key)
            continue;
        const auto& remote = std::get<CartesianCellAncestry>(other.key.ancestry).index;
        for (unsigned a = 0; a < 8; ++a)
            for (unsigned b = 0; b < 8; ++b)
            {
                Coordinates first{}, second{};
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    first[axis] = (local[axis] + ((a >> axis) & 1)) / 2.0L;
                    second[axis] = (remote[axis] + ((b >> axis) & 1)) / 2.0L;
                }
                if (first == second)
                    result.emplace_back(other.key, owner_of(other.key, ranks), other.revision, first, first, second);
            }
    }
    std::sort(result.begin(), result.end());
    return result;
}
std::vector<AdaptiveBytes> identities(const AdaptiveNativeDraft& draft)
{
    std::vector<AdaptiveBytes> result;
    for (const auto& tile : draft.tiles)
        result.push_back(tile.identity);
    std::sort(result.begin(), result.end());
    return result;
}
} // namespace

TEST(CartesianContactSnapshots, HierarchyWitnessesAndNativeDraftMatchIndependentOracles)
{
    const auto value = config();
    Owner explicit_state;
    {
        const Gate gate(nullptr);
        explicit_state = source(value);
    }
    Owner generated;
    {
        const Gate gate("0");
        generated = source(value);
    }
    const auto snapshot = contact_snapshot(*generated);
    ASSERT_TRUE(procedural(*snapshot));
    EXPECT_FALSE(procedural(*contact_snapshot(*explicit_state)));
    bool retained_regular_graph = false;
    snapshot->visit_borrowed_payload(
        [&](const ContactPayloadAllocation& allocation)
        {
            retained_regular_graph |=
                allocation.identity.role == GeometryContactPayloadRole::ContactIncidenceRowObject ||
                allocation.identity.role == ParallelSupportPayloadRole::FanBlockObject;
        });
    EXPECT_FALSE(retained_regular_graph);
    for (const auto& leaf : generated->owned_leaves())
    {
        const auto actual = witnesses(*snapshot, leaf.key);
        EXPECT_EQ(actual, lattice(leaf.key, value, generated->communicator()->getSize()));
        EXPECT_EQ(actual, witnesses(*contact_snapshot(*explicit_state), leaf.key));
        EXPECT_EQ(actual.size(), 19u); // Three faces, three edges and one corner: 12 + 6 + 1.
    }
    const auto expected = explicit_state->native_draft();
    const auto actual = generated->native_draft();
    EXPECT_EQ(identities(actual), identities(expected));
    EXPECT_EQ(actual.global_faces, expected.global_faces);
    EXPECT_EQ(actual.global_nodes, expected.global_nodes);
    EXPECT_EQ(actual.global_faces, 36u);
    EXPECT_EQ(actual.global_nodes, 27u);
}

TEST(CartesianContactSnapshots, DetachedReadersSurviveRefinementMigrationAndControllerDestruction)
{
    const Gate gate("");
    auto state = source(config());
    const auto old = contact_snapshot(*state);
    const auto old_pieces = subface_snapshot(*state);
    std::optional<AdaptiveCellKey> local;
    std::vector<Witness> saved;
    if (!state->owned_leaves().empty())
    {
        local = state->owned_leaves().begin()->key;
        saved = witnesses(*old, *local);
    }
    CanonicalSubfaceView escaped;
    old_pieces->visit_pieces(
        [&](const CanonicalSubfaceView& view)
        {
            if (!escaped)
                escaped = view;
        });
    std::optional<ContactPointIdentity> point;
    if (escaped)
        point = escaped.point(0);
    std::vector<AdaptiveMark> marks;
    for (const auto& leaf : state->owned_leaves())
        marks.push_back({leaf.key, AdaptiveSplit::Isotropic});
    auto refined = state->apply(state->plan_adaptation(marks));
    EXPECT_TRUE(procedural(*contact_snapshot(*refined)));
    EXPECT_EQ(refined->global_cells(), 64u);
    if (refined->communicator()->getSize() > 1)
    {
        std::vector<int> targets(refined->owned_leaves().size(), refined->communicator()->getRank() ^ 1);
        refined = refined->apply(refined->plan_migration(targets));
        EXPECT_TRUE(procedural(*contact_snapshot(*refined)));
    }
    EXPECT_NO_THROW((void) refined->native_draft());
    refined.reset();
    state.reset();
    if (local)
        EXPECT_EQ(witnesses(*old, *local), saved);
    if (escaped)
    {
        EXPECT_EQ(escaped.point(0), *point);
        Budget budget{ample};
        IdentityBudget identities;
        EXPECT_NO_THROW((void) escaped.materialize(budget, identities));
        const auto exact_peak = budget.peak;
        Budget exact{exact_peak};
        IdentityBudget exact_identities;
        EXPECT_NO_THROW((void) escaped.materialize(exact, exact_identities));
        ASSERT_GT(exact_peak, 0u);
        Budget short_budget{exact_peak - 1};
        IdentityBudget short_identities;
        EXPECT_THROW((void) escaped.materialize(short_budget, short_identities), std::invalid_argument);
        EXPECT_EQ(escaped.point(0), *point);
    }
}

TEST(CartesianContactSnapshots, UnsupportedMixedLevelEditPreservesAcceptedRootsAndNativePublication)
{
    const Gate gate("1");
    auto value = config();
    value.root_blocks[0] = 2;
    const auto state = source(value);
    const auto contacts = contact_snapshot(*state);
    const auto layout = state->leaf_layout();
    const auto before = identities(state->native_draft());
    std::vector<AdaptiveMark> marks;
    const AdaptiveCellKey selected{config().region_id, CartesianCellAncestry{0, {0, 0, 0}}};
    if (state->owned_leaves().contains(selected))
        marks.push_back({selected, AdaptiveSplit::Isotropic});
    EXPECT_THROW((void) state->plan_adaptation(marks), std::invalid_argument);
    EXPECT_EQ(state->leaf_layout(), layout);
    EXPECT_EQ(contact_snapshot(*state), contacts);
    EXPECT_EQ(identities(state->native_draft()), before);
}

TEST(CartesianContactSnapshots, SealAdmissionFailureAllowsRetryAndRejectsForeignLayoutOwnership)
{
    const Gate gate("1");
    const auto accepted = source(config());
    const auto layout = accepted->leaf_layout();
    std::vector<AdaptiveLeaf> owned;
    for (const auto& leaf : *layout)
        owned.push_back(leaf);
    const auto kernel = cartesian_adaptive_kernel(config());
    DistributedAdaptiveContactHalo halo(*kernel, accepted->communicator(), {}, {});
    halo.bootstrap_canonical(owned, [&](const auto& key) -> const AdaptiveLeaf& { return layout->at(key); }, layout);
    const auto used = halo.allocation().used;
    const auto limit = halo.allocation().limit;
    halo.allocation().limit = used;
    EXPECT_THROW((void) halo.seal_contacts(layout), std::invalid_argument);
    EXPECT_EQ(halo.allocation().used, used);
    halo.allocation().limit = limit;
    const auto fake = std::shared_ptr<const AdaptiveLeafLayout>(layout.get(), [](const auto*) {});
    EXPECT_THROW((void) halo.seal_contacts(fake), std::invalid_argument);
    const auto sealed = halo.seal_contacts(layout);
    ASSERT_TRUE(procedural(*sealed));
    EXPECT_NO_THROW((void) halo.pin_incidence_rows());
    EXPECT_TRUE(halo.subface_snapshot());
}
