#include "general_vof_finite_test_helpers.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "plic_cleanup_capture.hh"
#include <limits>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Declaration = GeneralVoFPhaseGeometry;
using Backend = GeneralVoFTransportBackend;
constexpr std::array cleanup_face_backends{Backend::GeometricFace};
const char* backend_name(Backend backend)
{
    return backend == Backend::DonorCell ? "DonorCell" : "GeometricFace";
}
GeneralVoFOptions cleanup_options(double cutoff = 1e-14, double budget = 1e-12)
{
    GeneralVoFOptions result;
    result.phase_cleanup = {cutoff, {budget, budget}};
    return result;
}
auto cycle(const GeneralVoF& source, double q)
{
    auto result = circulation(source);
    for (auto& face : result)
        face.integrated_volume *= q * 64;
    return result;
}
TEST(GeneralVoFPhaseCleanup, CapturedTailChargesActualChangeAndBypassesMixedFit)
{
    if (std::numeric_limits<long double>::digits != 64)
        GTEST_SKIP() << "Captured Ordinary certificate uses a 64-bit long-double significand.";
    constexpr double tail = 0x1.45693118ee60cp-184;
    auto geometry = cleanup_captured_geometry();
    const PLIC::GeometryPoint normal{0x1.f7ee8d4364952p-1, 0x1.a119432b4a1dcp-4, -0x1.2810933c9e17ap-3};
    // The untouched raw fixed-normal problem must retain its strict rejection.
    EXPECT_THROW(
        (void) PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::solve_offset({geometry, tail, 0x1p-7}, normal),
        PLIC::GeometryError);
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), tail);
    std::fill(input.carrier.begin(), input.carrier.end(), 1);
    auto options = cleanup_options(1e-14, 16 * tail);
    options.phase_cleanup.maximum_absolute_volume[1] = 0;
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
    EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 8u);
    EXPECT_EQ(source.phase_cleanup_state().signed_volume[0], -8.L * tail);
    EXPECT_EQ(source.phase_cleanup_state().absolute_volume[0], 8.L * tail);
    EXPECT_EQ(source.phase_cleanup_state().absolute_volume[1], 0);
    for (const auto& delta : source.latest_phase_cleanup())
    {
        EXPECT_EQ(delta.before[0], tail);
        EXPECT_EQ(delta.after[0], 0);
        EXPECT_EQ(delta.volume[0], -static_cast<long double>(tail));
        EXPECT_EQ(delta.before[1], delta.after[1]);
        EXPECT_EQ(delta.volume[1], 0);
    }
    for (auto declaration : source.phase_geometry_declarations())
        EXPECT_EQ(declaration, Declaration::PureCarrier);
    auto trial = source.plan_transport(stationary(source), circulation(source, true), 1);
    const auto prepared = trial.prepare_transport({.1, 1});
    EXPECT_TRUE(prepared->phase_cleanup.empty());
    EXPECT_EQ(prepared->diagnostics.plane_requests, 0u);
    EXPECT_EQ(trial.candidate()->phase_cleanup_state().cleaned_cells, 8u);
    EXPECT_TRUE(trial.candidate()->latest_phase_cleanup().empty());
    const auto pure =
        PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::solve_offset({geometry, 0, 0x1p-7}, normal);
    EXPECT_TRUE(pure.pure_carrier);
}
TEST(GeneralVoFPhaseCleanup, SymmetricChargesUsePublishedEndpoints)
{
    const auto input = fixture();
    for (bool minority_gas : {false, true})
        for (const double small : {0x1p-48, 0x1p-60})
        {
            // At 2^-60 the represented dominant is already one; no credit is representable.
            const double dominant = 1 - small;
            std::vector<double> gas(input.order.size(), minority_gas ? small : dominant);
            std::vector<double> carrier(input.order.size(), minority_gas ? dominant : small);
            GeneralVoF source(input.hierarchy, input.order, gas, carrier, cleanup_options());
            const unsigned m = minority_gas ? 0 : 1, d = 1 - m;
            EXPECT_EQ(source.phase_cleanup_state().signed_volume[m], -8.L * small);
            EXPECT_EQ(source.phase_cleanup_state().absolute_volume[m], 8.L * small);
            const auto credit = small == 0x1p-48 ? 8.L * small : 0.L;
            EXPECT_EQ(source.phase_cleanup_state().signed_volume[d], credit);
            EXPECT_EQ(source.phase_cleanup_state().absolute_volume[d], credit);
            EXPECT_EQ(source.phase_cleanup_state().largest_absolute_volume[m], small);
            EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 8u);
        }
}
TEST(GeneralVoFPhaseCleanup, DisabledAndAboveCutoffPreserveInventory)
{
    auto input = fixture();
    constexpr double small = 0x1p-40;
    std::fill(input.gas.begin(), input.gas.end(), small);
    std::fill(input.carrier.begin(), input.carrier.end(), 1 - small);
    for (const auto options : {cleanup_options(0), GeneralVoFOptions{}, cleanup_options()})
    {
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
        EXPECT_EQ(source.gas_volumes(), input.gas);
        EXPECT_EQ(source.carrier_volumes(), input.carrier);
        EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 0u);
        const auto geometry = source.owned_finite_geometry<ArithmeticMode::Ordinary>();
        for (std::size_t i = 0; i < geometry.size(); ++i)
        {
            const auto plane = PLIC::PolyhedralReconstruction<ArithmeticMode::Ordinary>::solve_offset(
                {geometry[i], small, 1 - small}, {1, 0, 0});
            EXPECT_FALSE(plane.pure_carrier);
            EXPECT_FALSE(plane.pure_gas);
            EXPECT_LE(std::abs(plane.target_volume_residual), 8e-15 * small);
        }
    }
}
TEST(GeneralVoFPhaseCleanup, DisabledTinyPhaseRemainsIndependentAndUncertaintyIsNotUsedForCleanup)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 0x1p-60);
    std::fill(input.carrier.begin(), input.carrier.end(), 1);
    GeneralVoF raw(input.hierarchy, input.order, input.gas, input.carrier, cleanup_options(0));
    GeneralVoF clean(input.hierarchy, input.order, input.gas, input.carrier, cleanup_options());
    EXPECT_EQ(raw.gas_volumes(), input.gas);
    EXPECT_EQ(raw.carrier_volumes(), input.carrier);
    EXPECT_EQ(raw.phase_cleanup_state().cleaned_cells, 0u);
    for (std::size_t c = 0; c < input.order.size(); ++c)
    {
        EXPECT_EQ(raw.phase_geometry_declarations()[c], Declaration::Infer);
        EXPECT_EQ(raw.phase_roundoff_bounds()[c].gas, clean.phase_roundoff_bounds()[c].gas);
        EXPECT_EQ(raw.phase_roundoff_bounds()[c].carrier, clean.phase_roundoff_bounds()[c].carrier);
        EXPECT_EQ(raw.accumulated_phase_errors()[c].gas, clean.accumulated_phase_errors()[c].gas);
        EXPECT_EQ(raw.accumulated_phase_errors()[c].carrier, clean.accumulated_phase_errors()[c].carrier);
        EXPECT_EQ(raw.initial_occupancy_errors()[c].admitted_bound, clean.initial_occupancy_errors()[c].admitted_bound);
    }
}

TEST(GeneralVoFPhaseCleanup, GeometryBackendsDefaultToPinnedGlobalBudgets)
{
    constexpr double small = 0x1p-48;
    auto input = fixture();
    for (std::size_t c = 0; c < input.order.size(); ++c)
    {
        const auto x = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[c]))->centroid();
        if (x[0] > 1 && x[1] < 1 && x[2] < 1)
        {
            input.gas[c] = small;
            input.carrier[c] = 1 - small;
        }
    }
    for (const auto backend : {GeneralVoFTransportBackend::SweptPartition, GeneralVoFTransportBackend::GeometricFace})
    {
        GeneralVoFOptions options;
        options.transport_backend = backend;
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
        const auto& policy = source.options().phase_cleanup;
        EXPECT_EQ(policy.relative_cutoff, 1e-14);
        EXPECT_DOUBLE_EQ(policy.maximum_absolute_volume[0], 1e-13 * (1 + small));
        EXPECT_DOUBLE_EQ(policy.maximum_absolute_volume[1], 1e-13 * (7 - small));
        EXPECT_GT(policy.maximum_absolute_volume[0], 1e-13); // Raw initial inventory, before cleanup.
        EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 1u);
        EXPECT_EQ(source.phase_cleanup_state().signed_volume, (std::array<long double, 2>{-small, small}));
        EXPECT_EQ(source.phase_cleanup_state().absolute_volume, (std::array<long double, 2>{small, small}));
        for (std::size_t c = 0; c < input.order.size(); ++c)
            EXPECT_TRUE(source.gas_volumes()[c] == 0 || source.carrier_volumes()[c] == 0);
        const auto packet = source.phase_cleanup_checkpoint();
        GeneralVoF restored(source.hierarchy(), source.owned_order(), source.gas_volumes(), source.carrier_volumes(),
            options, source.phase_geometry_declarations(), GeneralVoF::decode_phase_cleanup_checkpoint(packet));
        EXPECT_EQ(restored.phase_cleanup_checkpoint(), packet);
        EXPECT_EQ(restored.options().phase_cleanup.maximum_absolute_volume, policy.maximum_absolute_volume);
        EXPECT_TRUE(restored.latest_phase_cleanup().empty());

        // Ownership changes may not derive another allowance from the already-cleaned state.
        const auto comm = Tpetra::getDefaultComm();
        const auto plan = source.hierarchy()->plan_migration(
            std::vector<int>(source.owned_order().size(), (comm->getRank() + 1) % comm->getSize()));
        const auto raw = source.hierarchy()->apply(plan);
        const auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
        const auto target =
            source.hierarchy()->apply(plan, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
        const auto moved = source.plan_remap(plan, target).candidate();
        EXPECT_EQ(moved->phase_cleanup_checkpoint(), packet);
        EXPECT_EQ(moved->options().phase_cleanup.maximum_absolute_volume, policy.maximum_absolute_volume);
        EXPECT_TRUE(moved->latest_phase_cleanup().empty());

        options.phase_cleanup.relative_cutoff = 0;
        GeneralVoF disabled(input.hierarchy, input.order, input.gas, input.carrier, options);
        EXPECT_EQ(disabled.gas_volumes(), input.gas);
        EXPECT_EQ(disabled.carrier_volumes(), input.carrier);
        EXPECT_EQ(disabled.phase_cleanup_state().cleaned_cells, 0u);
        options.phase_cleanup.relative_cutoff = GeneralVoFPhaseCleanupOptions::automatic;
        options.phase_cleanup.maximum_absolute_volume[0] = 0;
        EXPECT_THROW(
            (void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options), std::invalid_argument);
    }
}

TEST(GeneralVoFPhaseCleanup, AutomaticDefaultsLeaveOtherBackendsAndModesUnchanged)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 0x1p-60);
    std::fill(input.carrier.begin(), input.carrier.end(), 1);
    for (const auto mode : {ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification, ArithmeticMode::Exact})
    {
        GeneralVoFOptions options;
        options.arithmetic_mode = mode;
        if (mode == ArithmeticMode::Ordinary)
            options.transport_backend = GeneralVoFTransportBackend::DonorCell;
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
        EXPECT_EQ(source.options().phase_cleanup.relative_cutoff, 0);
        EXPECT_EQ(source.options().phase_cleanup.maximum_absolute_volume, (std::array<double, 2>{0, 0}));
        EXPECT_EQ(source.gas_volumes(), input.gas);
        EXPECT_EQ(source.carrier_volumes(), input.carrier);
        options.phase_cleanup.relative_cutoff = 1e-14;
        EXPECT_THROW(
            (void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options), std::invalid_argument);
    }
}

TEST(GeneralVoFPhaseCleanup, AutomaticBudgetRejectsRemovalOfEntireDiluteGlobalPhase)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 0x1p-60);
    std::fill(input.carrier.begin(), input.carrier.end(), 1);
    EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier), std::invalid_argument);
    GeneralVoFOptions disabled;
    disabled.phase_cleanup.relative_cutoff = 0;
    GeneralVoF previous(input.hierarchy, input.order, input.gas, input.carrier, disabled);
    GeneralVoF restored(previous.hierarchy(), previous.owned_order(), previous.gas_volumes(),
        previous.carrier_volumes(), {}, previous.phase_geometry_declarations(),
        GeneralVoF::decode_phase_cleanup_checkpoint(previous.phase_cleanup_checkpoint()));
    EXPECT_EQ(restored.options().phase_cleanup.relative_cutoff, 0);
    EXPECT_EQ(restored.gas_volumes(), input.gas);
    EXPECT_EQ(restored.phase_cleanup_checkpoint(), previous.phase_cleanup_checkpoint());
}

TEST(GeneralVoFPhaseCleanup, AutomaticRequestsAndRestoredPoliciesAgreeBeforeResolution)
{
    const auto input = fixture();
    const auto comm = Tpetra::getDefaultComm();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    auto invalid = source.options();
    invalid.phase_cleanup.relative_cutoff = -2;
    EXPECT_THROW(
        (void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, invalid), std::invalid_argument);
    if (comm->getSize() > 1)
    {
        GeneralVoFOptions requests;
        if (comm->getRank() == comm->getSize() - 1)
            requests.phase_cleanup.maximum_absolute_volume[0] = 0;
        EXPECT_THROW(
            (void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, requests), std::invalid_argument);
        auto state = source.phase_cleanup_state();
        if (comm->getRank() == comm->getSize() - 1)
            state.policy.maximum_absolute_volume[0] *= 2;
        EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, {}, {}, state),
            std::invalid_argument);
    }
}
TEST(GeneralVoFPhaseCleanup, AbsoluteBudgetDoesNotCancelOppositeRepairs)
{
    auto input = fixture();
    constexpr double small = 0x1p-48;
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto c = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = c[0] < 1 ? small : 1 - small;
        input.carrier[i] = c[0] < 1 ? 1 - small : small;
    }
    EXPECT_THROW(
        (void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, cleanup_options(1e-14, 6 * small)),
        std::invalid_argument);
    GeneralVoF accepted(input.hierarchy, input.order, input.gas, input.carrier, cleanup_options(1e-14, 8 * small));
    for (unsigned p = 0; p < 2; ++p)
    {
        EXPECT_EQ(accepted.phase_cleanup_state().signed_volume[p], 0);
        EXPECT_EQ(accepted.phase_cleanup_state().absolute_volume[p], 8.L * small);
    }
}
TEST(GeneralVoFPhaseCleanup, CumulativeBudgetAndCheckpointSurviveTrialsAndRestart)
{
    const auto input = fixture();
    constexpr double q = 0x1p-48;
    const auto options = cleanup_options(1e-14, 3 * q);
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto identity = source.accepted_identity();
    const auto first = source.plan_transport(stationary(source), cycle(source, q), 1);
    const auto sibling = source.plan_transport(stationary(source), cycle(source, q), 1);
    EXPECT_EQ(source.accepted_identity(), identity);
    EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 0u);
    const auto next = source.apply(first);
    EXPECT_EQ(next->gas_volumes(), source.gas_volumes());
    EXPECT_EQ(next->carrier_volumes(), source.carrier_volumes());
    EXPECT_EQ(next->phase_cleanup_state().cleaned_cells, 2u);
    EXPECT_EQ(next->phase_cleanup_state().absolute_volume, (std::array<long double, 2>{2 * q, 2 * q}));
    EXPECT_EQ(sibling.candidate()->phase_cleanup_state().absolute_volume, next->phase_cleanup_state().absolute_volume);
    const auto prepared = first.prepare_transport({.1, 1000}, 42);
    EXPECT_EQ(prepared->new_phase_cleanup.absolute_volume, next->phase_cleanup_state().absolute_volume);
    for (const auto& delta : prepared->phase_cleanup)
    {
        const auto row = next->mesh()->owned_cell_map()->getLocalElement(delta.gid);
        ASSERT_GE(row, 0);
        // Use the represented density, not the decimal real .1.
        EXPECT_EQ(
            prepared->raw_new_mass[row], static_cast<long double>(.1) * delta.before[0] + 1000.L * delta.before[1]);
        EXPECT_EQ(prepared->new_mass[row], static_cast<long double>(.1) * delta.after[0] + 1000.L * delta.after[1]);
        EXPECT_EQ(
            prepared->cleanup_mass[row], static_cast<long double>(.1) * delta.volume[0] + 1000.L * delta.volume[1]);
    }
    EXPECT_THROW((void) next->plan_transport(stationary(*next), cycle(*next, q), 1), std::invalid_argument);
    EXPECT_EQ(next->phase_cleanup_state().cleaned_cells, 2u);
    const auto packet = next->phase_cleanup_checkpoint();
    const auto checkpoint = GeneralVoF::decode_phase_cleanup_checkpoint(packet);
    GeneralVoF restored(next->hierarchy(), next->owned_order(), next->gas_volumes(), next->carrier_volumes(), options,
        next->phase_geometry_declarations(), checkpoint);
    EXPECT_EQ(restored.phase_cleanup_state().absolute_volume, next->phase_cleanup_state().absolute_volume);
    EXPECT_THROW((void) restored.plan_transport(stationary(restored), cycle(restored, q), 1), std::invalid_argument);
    auto changed = options;
    changed.phase_cleanup.maximum_absolute_volume[0] *= 2;
    EXPECT_THROW((void) GeneralVoF(next->hierarchy(), next->owned_order(), next->gas_volumes(), next->carrier_volumes(),
                     changed, next->phase_geometry_declarations(), checkpoint),
        std::invalid_argument);
    auto malformed = packet;
    malformed.push_back(0);
    EXPECT_THROW((void) GeneralVoF::decode_phase_cleanup_checkpoint(malformed), std::invalid_argument);
}
TEST(GeneralVoFPhaseCleanup, PolicyIsCollectivelyValidatedIncludingEmptyOwners)
{
    const auto input = fixture();
    auto options = cleanup_options();
    options.phase_cleanup.relative_cutoff = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        (void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options), std::invalid_argument);
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() > 1)
    {
        options = cleanup_options();
        if (comm->getRank() == comm->getSize() - 1)
            options.phase_cleanup.maximum_absolute_volume[0] *= 2;
        EXPECT_THROW(
            (void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options), std::invalid_argument);
    }
}
void ownership_migration_retains_ledger(Backend backend)
{
    auto input = fixture();
    constexpr double small = 0x1p-60;
    std::fill(input.gas.begin(), input.gas.end(), small);
    std::fill(input.carrier.begin(), input.carrier.end(), 1);
    auto options = cleanup_options();
    options.transport_backend = backend;
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto before = source.phase_cleanup_state();
    const auto identity = source.accepted_identity();
    EXPECT_EQ(before.cleaned_cells, 8u);
    EXPECT_EQ(before.signed_volume, (std::array<long double, 2>{-8.L * small, 0}));
    EXPECT_EQ(before.absolute_volume, (std::array<long double, 2>{8.L * small, 0}));
    const auto comm = Tpetra::getDefaultComm();
    std::vector<int> destinations(source.owned_order().size(), (comm->getRank() + 1) % comm->getSize());
    const auto migration = source.hierarchy()->plan_migration(destinations);
    // Bind the actual migrated hierarchy through the same native publication
    // seam used by the independent finite-remap tests.
    const auto raw = source.hierarchy()->apply(migration);
    const auto native = assemble_adaptive_native(*raw, raw->configuration_descriptor());
    const auto target =
        source.hierarchy()->apply(migration, raw->with_partition(native.partition, native.mesh->owned_cell_map()));
    const auto trial = source.plan_remap(migration, target);
    EXPECT_TRUE(trial.ownership_only());
    const auto migrated = trial.candidate();
    const auto& after = migrated->phase_cleanup_state();
    EXPECT_EQ(after.cleaned_cells, before.cleaned_cells);
    EXPECT_EQ(after.signed_volume, before.signed_volume);
    EXPECT_EQ(after.absolute_volume, before.absolute_volume);
    EXPECT_EQ(after.largest_absolute_volume, before.largest_absolute_volume);
    EXPECT_EQ(after.policy_bound, before.policy_bound);
    EXPECT_EQ(after.policy.relative_cutoff, before.policy.relative_cutoff);
    EXPECT_EQ(after.policy.maximum_absolute_volume, before.policy.maximum_absolute_volume);
    EXPECT_TRUE(migrated->latest_phase_cleanup().empty());
    std::uint64_t local_cells = migrated->owned_order().size(), global_cells = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_cells, &global_cells);
    EXPECT_EQ(global_cells, 8u);
    const auto declarations = migrated->phase_geometry_declarations();
    for (std::size_t i = 0; i < migrated->owned_order().size(); ++i)
    {
        EXPECT_EQ(migrated->gas_volumes()[i], 0);
        EXPECT_EQ(migrated->carrier_volumes()[i], 1);
        EXPECT_EQ(declarations[i], Declaration::PureCarrier);
    }
    GeneralVoFFiniteRemapOptions coupled;
    coupled.channels.push_back({"carrier_tracer", GeneralVoFFiniteRemapCarrier::CarrierVolume, true});
    EXPECT_THROW((void) source.plan_remap(migration, target, {}, coupled), std::invalid_argument);
    EXPECT_EQ(source.accepted_identity(), identity);
    EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, before.cleaned_cells);
    EXPECT_EQ(source.phase_cleanup_state().signed_volume, before.signed_volume);
    EXPECT_EQ(source.phase_cleanup_state().absolute_volume, before.absolute_volume);
    auto malformed = source.phase_cleanup_checkpoint();
    ASSERT_FALSE(malformed.empty());
    malformed.front() ^= 1;
    EXPECT_THROW((void) GeneralVoF::decode_phase_cleanup_checkpoint(malformed), std::invalid_argument);
    if (comm->getSize() > 1)
    {
        auto disagreement = before;
        // This remains a locally valid ledger; only the last rank disagrees.
        // With >=3 ranks the fixture leaves that rank empty.
        if (comm->getRank() == comm->getSize() - 1)
            ++disagreement.cleaned_cells;
        EXPECT_THROW((void) GeneralVoF(source.hierarchy(), source.owned_order(), source.gas_volumes(),
                         source.carrier_volumes(), options, source.phase_geometry_declarations(), disagreement),
            std::invalid_argument);
    }
}

TEST(GeneralVoFPhaseCleanup, OwnershipMigrationRetainsLedgerAndRejectsCoupledChannels)
{
    ownership_migration_retains_ledger(Backend::SweptPartition);
}

TEST(GeneralVoFPhaseCleanup, GeometricInitializationChargesActualEndpoints)
{
    const auto input = fixture();
    for (const auto backend : cleanup_face_backends)
    {
        SCOPED_TRACE(backend_name(backend));
        for (const bool minority_gas : {false, true})
            for (const double small : {0x1p-48, 0x1p-60})
            {
                const double dominant = 1 - small;
                auto options = cleanup_options();
                options.transport_backend = backend;
                const std::vector<double> gas(input.order.size(), minority_gas ? small : dominant);
                const std::vector<double> carrier(input.order.size(), minority_gas ? dominant : small);
                GeneralVoF source(input.hierarchy, input.order, gas, carrier, options);
                const unsigned m = minority_gas ? 0 : 1, d = 1 - m;
                // Binary64 can represent the credit at 2^-48, but adding 2^-60
                // to one leaves the dominant endpoint unchanged.
                const long double credit = small == 0x1p-48 ? small : 0;
                EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 8u);
                EXPECT_EQ(source.phase_cleanup_state().signed_volume[m], -8.L * small);
                EXPECT_EQ(source.phase_cleanup_state().signed_volume[d], 8 * credit);
                EXPECT_EQ(source.phase_cleanup_state().absolute_volume[m], 8.L * small);
                EXPECT_EQ(source.phase_cleanup_state().absolute_volume[d], 8 * credit);
                ASSERT_EQ(source.latest_phase_cleanup().size(), input.order.size());
                for (const auto& delta : source.latest_phase_cleanup())
                {
                    EXPECT_EQ(delta.before[m], small);
                    EXPECT_EQ(delta.before[d], dominant);
                    EXPECT_EQ(delta.after[m], 0);
                    EXPECT_EQ(delta.after[d], 1);
                    EXPECT_EQ(delta.volume[m], -static_cast<long double>(small));
                    EXPECT_EQ(delta.volume[d], credit);
                }
                for (const auto declaration : source.phase_geometry_declarations())
                    EXPECT_EQ(declaration, minority_gas ? Declaration::PureCarrier : Declaration::PureGas);
            }
    }
}

TEST(GeneralVoFPhaseCleanup, GeometricCanonicalTransportRetainsRawMassAndCumulativeBudget)
{
    const auto input = fixture();
    constexpr double q = 0x1p-48;
    constexpr std::array<double, 2> density{.1, 1000};
    for (const auto backend : cleanup_face_backends)
    {
        SCOPED_TRACE(backend_name(backend));
        auto options = cleanup_options(1e-14, 3 * q);
        options.transport_backend = backend;
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
        const auto source_identity = source.accepted_identity();
        const auto first = source.plan_face_transport(cycle(source, q), 1, density, 42);
        const auto sibling = source.plan_face_transport(cycle(source, q), 1, density, 42);
        const auto next = source.apply(first);
        EXPECT_EQ(source.accepted_identity(), source_identity);
        EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 0u);
        EXPECT_EQ(source.gas_volumes(), input.gas);
        EXPECT_EQ(source.carrier_volumes(), input.carrier);
        EXPECT_EQ(next->gas_volumes(), input.gas);
        EXPECT_EQ(next->carrier_volumes(), input.carrier);
        EXPECT_EQ(next->phase_cleanup_state().cleaned_cells, 2u);
        EXPECT_EQ(next->phase_cleanup_state().signed_volume, (std::array<long double, 2>{0, 0}));
        EXPECT_EQ(next->phase_cleanup_state().absolute_volume, (std::array<long double, 2>{2 * q, 2 * q}));
        EXPECT_EQ(sibling.candidate()->phase_cleanup_state().absolute_volume,
            next->phase_cleanup_state().absolute_volume);
        const auto prepared = first.prepare_transport(density, 42);
        ASSERT_TRUE(prepared);
        first.validate_prepared_transport(*prepared, density, 42);
        EXPECT_EQ(prepared->family, GeneralVoFTransportFamily::AlgebraicFaceVolume);
        EXPECT_EQ(prepared->backend, backend);
        EXPECT_EQ(prepared->old_phase_cleanup.cleaned_cells, 0u);
        EXPECT_EQ(prepared->new_phase_cleanup.absolute_volume, next->phase_cleanup_state().absolute_volume);
        EXPECT_EQ(prepared->phase_cleanup.size(), next->latest_phase_cleanup().size());
        std::uint64_t local_receipts = prepared->phase_cleanup.size(), global_receipts = 0;
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_receipts, &global_receipts);
        EXPECT_EQ(global_receipts, 2u);
        for (const auto& delta : prepared->phase_cleanup)
        {
            const auto row = next->mesh()->owned_cell_map()->getLocalElement(delta.gid);
            ASSERT_GE(row, 0);
            const auto center = source.hierarchy()->geometry(source.hierarchy()->owned().at(delta.key))->centroid();
            // The closed square cycle exports q gas from the sole gas cell
            // into its +x neighbour; all other streams carry pure carrier.
            const bool gas_origin = center[0] < 1;
            EXPECT_LT(center[1], 1);
            EXPECT_LT(center[2], 1);
            EXPECT_EQ(delta.before[0], gas_origin ? 1 - q : q);
            EXPECT_EQ(delta.before[1], gas_origin ? q : 1 - q);
            EXPECT_EQ(delta.after[0], gas_origin ? 1 : 0);
            EXPECT_EQ(delta.after[1], gas_origin ? 0 : 1);
            EXPECT_EQ(delta.volume[0], gas_origin ? q : -q);
            EXPECT_EQ(delta.volume[1], gas_origin ? -q : q);
            EXPECT_EQ(prepared->raw_new_mass[row], static_cast<long double>(density[0]) * delta.before[0] +
                                                      static_cast<long double>(density[1]) * delta.before[1]);
            EXPECT_EQ(prepared->new_mass[row], gas_origin ? static_cast<long double>(density[0]) : density[1]);
            EXPECT_EQ(prepared->cleanup_mass[row], static_cast<long double>(density[0]) * delta.volume[0] +
                                                      static_cast<long double>(density[1]) * delta.volume[1]);
        }
        const auto accepted_identity = next->accepted_identity();
        const auto checkpoint_bytes = next->phase_cleanup_checkpoint();
        EXPECT_THROW((void) next->plan_face_transport(cycle(*next, q), 1, density), std::invalid_argument);
        EXPECT_EQ(next->accepted_identity(), accepted_identity);
        EXPECT_EQ(next->gas_volumes(), input.gas);
        EXPECT_EQ(next->carrier_volumes(), input.carrier);
        EXPECT_EQ(next->phase_cleanup_checkpoint(), checkpoint_bytes);
        const auto checkpoint = GeneralVoF::decode_phase_cleanup_checkpoint(checkpoint_bytes);
        GeneralVoF restored(next->hierarchy(), next->owned_order(), next->gas_volumes(), next->carrier_volumes(),
            options, next->phase_geometry_declarations(), checkpoint);
        EXPECT_EQ(restored.phase_cleanup_checkpoint(), checkpoint_bytes);
        EXPECT_THROW((void) restored.plan_face_transport(cycle(restored, q), 1, density), std::invalid_argument);
        const auto still = next->plan_face_transport(circulation(*next, true), 1, density);
        const auto stationary_state = next->apply(still);
        const auto stationary_prepared = still.prepare_transport(density);
        EXPECT_TRUE(stationary_state->latest_phase_cleanup().empty());
        EXPECT_TRUE(stationary_prepared->phase_cleanup.empty());
        EXPECT_EQ(stationary_state->phase_cleanup_checkpoint(), checkpoint_bytes);
        EXPECT_EQ(stationary_state->gas_volumes(), input.gas);
        EXPECT_EQ(stationary_state->carrier_volumes(), input.carrier);
        EXPECT_EQ(stationary_prepared->raw_new_mass, stationary_prepared->new_mass);
        for (const auto mass : stationary_prepared->cleanup_mass)
            EXPECT_EQ(mass, 0);
        // The preceding accepted publication still owns its original receipt.
        EXPECT_EQ(next->latest_phase_cleanup().size(), prepared->phase_cleanup.size());
    }
}

TEST(GeneralVoFPhaseCleanup, GeometricDisabledAndAboveCutoffPreserveInventory)
{
    auto input = fixture();
    for (const auto backend : cleanup_face_backends)
    {
        SCOPED_TRACE(backend_name(backend));
        for (const double small : {0x1p-48, 0x1p-40})
        {
            std::fill(input.gas.begin(), input.gas.end(), small);
            std::fill(input.carrier.begin(), input.carrier.end(), 1 - small);
            for (const bool enabled : {false, true})
            {
                if (enabled && small == 0x1p-48)
                    continue;
                auto options = cleanup_options(enabled ? 1e-14 : 0);
                options.transport_backend = backend;
                GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
                EXPECT_EQ(source.gas_volumes(), input.gas);
                EXPECT_EQ(source.carrier_volumes(), input.carrier);
                EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 0u);
                EXPECT_TRUE(source.latest_phase_cleanup().empty());
                for (const auto declaration : source.phase_geometry_declarations())
                    EXPECT_EQ(declaration, Declaration::Infer);
            }
        }
    }
}

TEST(GeneralVoFPhaseCleanup, GeometricOwnershipMigrationRetainsLedgerAndRejectsCoupledChannels)
{
    for (const auto backend : cleanup_face_backends)
    {
        SCOPED_TRACE(backend_name(backend));
        ownership_migration_retains_ledger(backend);
    }
}

TEST(GeneralVoFPhaseCleanup, EnabledCleanupRejectsBoundedAndExactArithmetic)
{
    const auto input = fixture();
    for (const auto mode : {ArithmeticMode::BoundedCertification, ArithmeticMode::Exact})
        for (const auto backend : {Backend::SweptPartition, Backend::DonorCell, Backend::GeometricFace})
        {
            auto options = cleanup_options();
            options.arithmetic_mode = mode;
            options.transport_backend = backend;
            EXPECT_THROW((void) GeneralVoF(input.hierarchy, input.order, input.gas, input.carrier, options),
                std::invalid_argument);
        }
}

} // namespace
