#include "FVM/details/VoFFailure.hh"
#include "general_vof_finite_test_helpers.hh"
#include <type_traits>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Prepared = GeneralVoFPreparedTransport;
static_assert(
    std::is_same_v<decltype(std::declval<GeneralVoF::Trial>().prepare_transport()), std::shared_ptr<const Prepared>>);
void check_capacities(const GeneralVoF& source, const GeneralVoF& next, const Prepared& prepared)
{
    ASSERT_EQ(prepared.old_mass.size(), source.phase_volumes().size());
    ASSERT_EQ(prepared.new_mass.size(), next.phase_volumes().size());
    ASSERT_EQ(prepared.old_phase_history.size(), source.accumulated_phase_errors().size());
    ASSERT_EQ(prepared.new_phase_history.size(), next.accumulated_phase_errors().size());
    for (std::size_t c = 0; c < prepared.old_mass.size(); ++c)
    {
        const auto& old = source.phase_volumes()[c];
        const auto& current = next.phase_volumes()[c];
        EXPECT_EQ(prepared.old_mass[c], static_cast<long double>(old.gas) + 1000.L * old.carrier);
        EXPECT_EQ(prepared.new_mass[c], static_cast<long double>(current.gas) + 1000.L * current.carrier);
        EXPECT_EQ(prepared.old_phase_history[c].gas, source.accumulated_phase_errors()[c].gas);
        EXPECT_EQ(prepared.new_phase_history[c].carrier, next.accumulated_phase_errors()[c].carrier);
    }
    EXPECT_FALSE(prepared.momentum_preparation);
    EXPECT_EQ(prepared.momentum_builds + prepared.momentum_reuses, 0u);
    EXPECT_NO_THROW(prepared.validate_byte_budget(source.options().byte_budget));
    EXPECT_THROW(prepared.validate_byte_budget(sizeof(Prepared) - 1), std::invalid_argument);
}
TEST(GeneralVoFPreparedTransport, AlgebraicBackendsRetainTheirDirectedStreamsAndIndependentPhases)
{
    const auto input = fixture();
    for (const auto backend : {GeneralVoFTransportBackend::DonorCell, GeneralVoFTransportBackend::GeometricFace})
    {
        GeneralVoFOptions options;
        options.transport_backend = backend;
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
        const auto trial = source.plan_face_transport(circulation(source), 1, {1, 1000}, 13);
        const auto prepared = trial.prepare_transport({1, 1000}, 13);
        EXPECT_EQ(prepared, trial.prepared_transport());
        EXPECT_EQ(prepared->family, GeneralVoFTransportFamily::AlgebraicFaceVolume);
        EXPECT_EQ(prepared->backend, backend);
        EXPECT_EQ(prepared->source_identity, source.accepted_identity());
        EXPECT_EQ(prepared->source_control, source.accepted_control());
        EXPECT_EQ(prepared->time_stage, 13u);
        EXPECT_EQ(prepared->timestep, 1);
        EXPECT_EQ(prepared->donor_budgets.size(), input.order.size());
        EXPECT_EQ(prepared->signed_face_contributions.size(), trial.finite_face_contributions().size());
        for (const auto& stream : prepared->streams)
        {
            EXPECT_GT(stream.volume, 0);
            EXPECT_EQ(stream.gas + stream.liquid, stream.volume);
            EXPECT_EQ(stream.mass, stream.gas + 1000.L * stream.liquid);
        }
        check_capacities(source, *trial.candidate(), *prepared);
        EXPECT_THROW((void) trial.prepare_transport({1, 10}, 13), std::invalid_argument);
        EXPECT_THROW((void) trial.prepare_transport({1, 1000}, 14), std::invalid_argument);
    }
}
TEST(GeneralVoFPreparedTransport, SweptSignedLedgerRemainsSeparateFromPositiveRouteSupport)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto controls = circulation(source);
    const auto trial = source.plan_transport(stationary(source), controls, 1);
    const auto prepared = trial.prepare_transport({1, 1000}, 17);
    EXPECT_EQ(prepared->family, GeneralVoFTransportFamily::GeometricSweptTetrahedra);
    EXPECT_TRUE(prepared->streams.empty());
    ASSERT_EQ(prepared->signed_face_contributions.size(), trial.finite_face_contributions().size());
    ASSERT_EQ(prepared->positive_transfers.size(), trial.finite_transfers().size());
    std::map<std::uint64_t, long double> signed_bulk;
    for (std::size_t i = 0; i < prepared->signed_face_contributions.size(); ++i)
    {
        const auto& face = prepared->signed_face_contributions[i];
        const auto& original = trial.finite_face_contributions()[i];
        EXPECT_EQ(face.face, original.face);
        EXPECT_EQ(face.donor, original.donor);
        EXPECT_EQ(face.direction, original.direction);
        EXPECT_EQ(face.amount.gas.value, original.amount.gas.value);
        signed_bulk[face.face] += face.direction * face.amount.bulk.value;
    }
    for (const auto& control : controls)
    {
        const auto row = source.mesh()->owned_face_map()->getLocalElement(control.face);
        if (row >= 0)
            EXPECT_NEAR(signed_bulk[control.face], control.integrated_volume, 1e-14);
    }
    for (const auto& route : prepared->positive_transfers)
    {
        EXPECT_GE(route.amount.gas.value, 0);
        EXPECT_GE(route.amount.carrier.value, 0);
    }
    check_capacities(source, *trial.candidate(), *prepared);
    const auto candidate = trial.candidate();
    EXPECT_THROW(candidate->validate_prepared_transport_source(*prepared), std::invalid_argument);
    EXPECT_NO_THROW(source.validate_prepared_transport_source(*prepared));
    auto stale = *prepared;
    ++stale.source_control;
    EXPECT_THROW(source.validate_prepared_transport_source(stale), std::invalid_argument);
    stale = *prepared;
    ++stale.mesh_generation;
    EXPECT_THROW(source.validate_prepared_transport_source(stale), std::invalid_argument);
    stale = *prepared;
    ++stale.mesh_topology_generation;
    EXPECT_THROW(source.validate_prepared_transport_source(stale), std::invalid_argument);
    const auto another = source.plan_transport(stationary(source), controls, 1);
    EXPECT_THROW(another.validate_prepared_transport(*prepared, {1, 1000}, 17), std::invalid_argument);
    EXPECT_THROW(trial.validate_prepared_transport(*prepared, {1, 1000}, 18), std::invalid_argument);
}
TEST(GeneralVoFPreparedTransport, StationarySweptPreparationPreservesBoundedModeAndEmptyOwners)
{
    const auto input = fixture();
    for (const auto mode : {ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification})
    {
        GeneralVoFOptions options;
        options.arithmetic_mode = mode;
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
        const auto trial = source.plan_transport(stationary(source), circulation(source, true), .25);
        const auto prepared = trial.prepare_transport({1, 1000}, 3);
        EXPECT_EQ(prepared->arithmetic_mode, mode);
        EXPECT_EQ(prepared->family, GeneralVoFTransportFamily::GeometricSweptTetrahedra);
        EXPECT_EQ(prepared->old_mass, prepared->new_mass);
        EXPECT_EQ(prepared->donor_budgets.size(), input.order.size());
        EXPECT_EQ(prepared->positive_transfers.size(), input.order.size());
        EXPECT_TRUE(prepared->signed_face_contributions.empty());
        EXPECT_TRUE(prepared->streams.empty());
        EXPECT_NO_THROW(trial.validate_prepared_transport(*prepared, {1, 1000}, 3));
        check_capacities(source, *trial.candidate(), *prepared);
    }
    int empty = input.order.empty(), total_empty = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(total_empty, std::max(0, Tpetra::getDefaultComm()->getSize() - 2));
}
TEST(GeneralVoFPreparedTransport, RankLocalInvalidContextRejectsCollectivelyWithoutChangingSource)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto trial = source.plan_transport(stationary(source), circulation(source, true), 1);
    const auto accepted = source.accepted_identity();
    const auto rank = Tpetra::getDefaultComm()->getRank();
    try
    {
        (void) trial.prepare_transport({rank == 0 ? -1. : 1., 1000}, 0);
        FAIL() << "Invalid density was accepted.";
    }
    catch (const std::invalid_argument& error)
    {
        const auto* failure = dynamic_cast<const SimpleFluid::detail::vof_failure::Failure*>(&error);
        EXPECT_NE(failure, nullptr);
        if (failure)
        {
            EXPECT_EQ(failure->record().origin_rank, 0);
            EXPECT_EQ(failure->record().category, SimpleFluid::detail::vof_failure::Category::Metadata);
            EXPECT_STREQ(failure->record().stage.data(), "phase_preparation");
            EXPECT_EQ(bool(failure->local_cause()), rank == 0);
        }
        EXPECT_NE(std::string(error.what()).find("positive densities"), std::string::npos);
    }
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        EXPECT_THROW((void) trial.prepare_transport({1, rank == 0 ? 100. : 1000.}, 0), std::invalid_argument);
        EXPECT_THROW((void) trial.prepare_transport({1, 1000}, rank), std::invalid_argument);
    }
    EXPECT_EQ(source.accepted_identity(), accepted);
    EXPECT_EQ(source.gas_volumes(), input.gas);
}
} // namespace
