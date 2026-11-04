#include "general_vof_finite_test_helpers.hh"
#include "general_vof_retention_probe.hh"
#include <cfenv>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
namespace detail = SimpleFluid::VoF::general_vof_detail;
namespace census = PLIC::details::retention;
using Group = detail::RetentionGroup;
using Access = detail::RetentionAccess;
using SimpleFluid::VoF::test::retention_probe::checkpoint;
template<class Collect> auto inspect(std::vector<std::byte>& storage, Collect&& collect)
{
    // Report owns only scalar metadata; its scratch resource must survive it.
    // The caller instead uses this helper only to count inside one lexical scope.
    census::ProbeScratch scratch(storage);
    detail::RetentionRoots roots(scratch);
    const auto rounding = std::fegetround();
    const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
    collect(roots);
    auto report = census::inspect_retention(
        roots.roots, roots.ledgers, scratch, {.complete_root_manifest = roots.complete_manifest});
    EXPECT_EQ(std::fegetround(), rounding);
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), flags);
    EXPECT_TRUE(report.complete()) << report.detail;
    EXPECT_TRUE(report.complete_root_manifest);
    return std::array{report.owner_union.bytes, report.root_count, report.retained_records, report.required_records};
}
TEST(GeneralVoFRetentionCensus, BorrowedRootsCoverPreparedAndCopiedTrajectoriesWithoutPinningSource)
{
    const auto input = fixture();
    auto source = std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier);
    EXPECT_EQ(source->nodes().size(), 27u);
    const auto* accepted = Access::data(*source);
    const auto source_owners = source.use_count();
    const auto identity = source->accepted_identity();
    const auto identity_owners = identity.use_count();
    const auto gas = source->gas_volumes(), carrier = source->carrier_volumes();
    std::vector<std::byte> storage(64 * 1024 * 1024);
    const auto baseline = inspect(storage, [&](auto& roots) { roots.state(*source); });
    EXPECT_GT(baseline[0], 0u);
    EXPECT_GT(baseline[1], 0u);
    EXPECT_EQ(source.use_count(), source_owners);
    EXPECT_EQ(identity.use_count(), identity_owners);
    checkpoint("prepared27", "accepted_before_preparation", [&](auto& roots) { roots.state(*source); });
    std::shared_ptr<const GeneralVoFPreparedTransport> prepared;
    GeneralVoFFiniteTrajectories detached;
    std::weak_ptr<const void> candidate_lifetime;
    std::shared_ptr<const GeneralVoF> next;
    {
        const auto trial = source->plan_transport(stationary(*source), circulation(*source), .25);
        EXPECT_EQ(Access::source(trial), accepted);
        EXPECT_NE(Access::candidate(trial), accepted);
        checkpoint("prepared27", "trial_prepared", [&](auto& roots) { roots.trial(trial); });
        prepared = trial.prepare_transport({1, 1000}, 31);
        candidate_lifetime = prepared->candidate_identity;
        detached = trial.finite_node_trajectories();
        const auto copied = *prepared;
        // Deliberate application publication, made outside the census callback.
        const auto copied_point = detached.at(0).point;
        EXPECT_EQ(detached.size(), 27u);
        const auto before = identity.use_count();
        const auto before_copies = inspect(storage,
            [&](auto& roots)
            {
                roots.state(*source);
                roots.trial(trial);
                roots.prepared(*prepared);
            });
        const auto result = inspect(storage,
            [&](auto& roots)
            {
                roots.state(*source);
                roots.trial(trial);
                roots.prepared(*prepared);
                roots.prepared(copied);
                roots.trajectories(detached, Group::CopiedTrajectories);
                roots.add(copied_point, Group::CopiedPoints);
            });
        EXPECT_GT(result[1], baseline[1]);
        EXPECT_EQ(result[0], before_copies[0]);
        EXPECT_GT(result[1], before_copies[1]);
        EXPECT_EQ(identity.use_count(), before);
        checkpoint("prepared27", "prepared_and_copied_trajectories",
            [&](auto& roots)
            {
                roots.state(*source);
                roots.trial(trial);
                roots.prepared(*prepared);
                roots.prepared(copied);
                roots.trajectories(detached, Group::CopiedTrajectories);
                roots.add(copied_point, Group::CopiedPoints);
            });
        next = source->apply(trial);
        checkpoint("prepared27", "accepted",
            [&](auto& roots)
            {
                roots.state(*source);
                roots.state(*next, Group::Candidate);
                roots.trial(trial);
                roots.prepared(*prepared);
            });
    }
    checkpoint("prepared27", "trial_scope_exit",
        [&](auto& roots)
        {
            roots.state(*source);
            roots.state(*next, Group::Candidate);
            roots.prepared(*prepared);
            roots.trajectories(detached, Group::CopiedTrajectories);
        });
    EXPECT_EQ(Access::data(*source), accepted);
    EXPECT_EQ(source->gas_volumes(), gas);
    EXPECT_EQ(source->carrier_volumes(), carrier);
    EXPECT_EQ(prepared->node_trajectories.size(), 27u);
    EXPECT_EQ(detached.size(), 27u);
    EXPECT_FALSE(candidate_lifetime.expired());
    prepared.reset();
    next.reset();
    // Copied trajectories retain geometry provenance, never the phase-bearing Data owner.
    EXPECT_TRUE(candidate_lifetime.expired());
    detached = {};
    const auto after = inspect(storage, [&](auto& roots) { roots.state(*source); });
    // Normal transport may initialize the shared immutable spatial cache.
    EXPECT_GE(after[0], baseline[0]);
    EXPECT_EQ(identity.use_count(), identity_owners);
    checkpoint("prepared27", "all_trial_roots_released", [&](auto& roots) { roots.state(*source); });
}
TEST(GeneralVoFRetentionCensus, ReboundPreparationPinsMakeManifestIncompleteWithoutInspectingOpaqueOwners)
{
    const auto input = fixture();
    GeneralVoFOptions options;
    options.record_balance_work = true;
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto* accepted = Access::data(source);
    const auto gas = source.gas_volumes(), carrier = source.carrier_volumes();
    const auto trial = source.plan_transport(stationary(source), circulation(source), .25);
    const auto prepared = trial.prepare_transport({1, 1000}, 31);
    ASSERT_TRUE(prepared->balance_work);
    const auto opaque = std::make_shared<const unsigned char>(0x5a);
    const auto identity = source.accepted_identity();
    std::vector<std::byte> storage(64 * 1024 * 1024);
    const auto check = [&](const GeneralVoFPreparedTransport& rebound)
    {
        const auto pins_before = identity.use_count();
        census::ProbeScratch scratch(storage);
        detail::RetentionRoots roots(scratch);
        roots.prepared(rebound);
        EXPECT_FALSE(roots.complete_manifest);
        const auto report = census::inspect_retention(
            roots.roots, roots.ledgers, scratch, {.complete_root_manifest = roots.complete_manifest});
        EXPECT_TRUE(report.complete()) << report.detail;
        EXPECT_FALSE(report.complete_root_manifest);
        EXPECT_EQ(identity.use_count(), pins_before);
        EXPECT_EQ(Access::data(source), accepted);
        EXPECT_EQ(source.gas_volumes(), gas);
        EXPECT_EQ(source.carrier_volumes(), carrier);
    };
    for (unsigned field = 0; field < 5; ++field)
    {
        SCOPED_TRACE(field);
        auto rebound = *prepared;
        switch (field)
        {
            case 0:
                rebound.source_identity = opaque;
                break;
            case 1:
                rebound.candidate_identity = opaque;
                break;
            case 2:
                rebound.mesh_identity = opaque;
                break;
            case 3:
                rebound.topology_identity = opaque;
                break;
            case 4:
                // An alias with sufficient ownership, deliberately no WorkSample object.
                // A diagnostic must reject its manifest by pointer identity before reading it.
                rebound.balance_work = std::shared_ptr<const GeneralVoFWorkSample>(
                    opaque, reinterpret_cast<const GeneralVoFWorkSample*>(opaque.get()));
                break;
        }
        check(rebound);
    }
    for (unsigned field = 0; field < 3; ++field)
    {
        SCOPED_TRACE(field);
        auto rebound = *prepared;
        auto work = std::make_shared<GeneralVoFWorkSample>(*prepared->balance_work);
        if (field == 0)
            work->source_identity = opaque;
        if (field == 1)
            work->mesh_identity = opaque;
        if (field == 2)
            work->topology_identity = opaque;
        rebound.balance_work = std::move(work);
        check(rebound);
    }
    EXPECT_TRUE(Access::authentic_public_pins(*prepared));
}
TEST(GeneralVoFRetentionCensus, RejectionAndExhaustedDiagnosticScratchPreserveAcceptedState)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto* accepted = Access::data(source);
    const auto gas = source.gas_volumes(), carrier = source.carrier_volumes();
    auto nodes = stationary(source);
    if (!nodes.empty())
        nodes.front().velocity[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) source.plan_transport(nodes, circulation(source), 1), std::invalid_argument);
    checkpoint("prepared27", "rejected", [&](auto& roots) { roots.state(source); });
    std::array<std::byte, 1> storage{};
    census::ProbeScratch scratch(storage);
    ASSERT_FALSE(accepted->cells.empty());
    const auto root = census::root(*accepted->topology->cell_geometry);
    const auto report = census::inspect_retention(std::span(&root, 1), {}, scratch, {.complete_root_manifest = true});
    EXPECT_FALSE(report.complete());
    EXPECT_EQ(Access::data(source), accepted);
    EXPECT_EQ(source.gas_volumes(), gas);
    EXPECT_EQ(source.carrier_volumes(), carrier);
}
} // namespace
