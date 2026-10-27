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
TEST(GeneralVoFPreparedTransport, InterfaceSignedLedgerKeepsSeparateNumericalSupport)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto controls = circulation(source);
    const auto trial = source.plan_transport(stationary(source), controls, 1);
    const auto prepared = trial.prepare_transport({1, 1000}, 17);
    EXPECT_EQ(prepared->family, GeneralVoFTransportFamily::InterfaceLocalSwept);
    EXPECT_TRUE(prepared->streams.empty());
    ASSERT_EQ(prepared->face_chain_diagnostics.size(), source.tiles().size());
    ASSERT_EQ(prepared->cell_closure_diagnostics.size(), input.order.size());
    int nonzero_controls = 0, total_nonzero_controls = 0;
    for (std::size_t i = 0; i < prepared->face_chain_diagnostics.size(); ++i)
    {
        const auto& diagnostic = prepared->face_chain_diagnostics[i];
        const auto& original = trial.finite_face_chain_diagnostics()[i];
        EXPECT_EQ(diagnostic.face, original.face);
        EXPECT_EQ(diagnostic.control.value, original.control.value);
        EXPECT_EQ(diagnostic.constructed.value, original.constructed.value);
        EXPECT_EQ(diagnostic.signed_difference.value, original.signed_difference.value);
        EXPECT_EQ(diagnostic.allowed_difference, original.allowed_difference);
        nonzero_controls += diagnostic.control.value != 0;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &nonzero_controls, &total_nonzero_controls);
    EXPECT_GT(total_nonzero_controls, 0);
    ASSERT_EQ(prepared->signed_face_contributions.size(), trial.finite_face_contributions().size());
    EXPECT_TRUE(prepared->positive_transfers.empty());
    EXPECT_TRUE(trial.finite_transfers().empty());
    EXPECT_EQ(prepared->signed_donor_budgets.size(), input.order.size());
    EXPECT_EQ(prepared->numerical_support_budgets.size(), input.order.size());
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
    for (const auto& route : prepared->numerical_support_transfers)
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
template<ArithmeticMode Mode> void check_typed_stationary(const Fixture& input)
{
    GeneralVoFOptions options;
    options.arithmetic_mode = Mode;
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
    // Both input buffers die before the control view is consumed.
    const auto controls = source.transport_controls<Mode>(stationary(source), circulation(source, true), .25);
    const auto view = [&]
    {
        const auto trial = source.plan_transport(controls);
        auto result = trial.template prepare_transport_view<Mode>({1, 1000}, 21);
        EXPECT_NO_THROW(trial.validate_prepared_transport(result, {1, 1000}, 21));
        const auto& prepared = result.ledger();
        check_capacities(source, *trial.candidate(), prepared);
        EXPECT_EQ(prepared.face_chain_diagnostics.size(), trial.finite_face_chain_diagnostics().size());
        EXPECT_EQ(prepared.cell_closure_diagnostics.size(), trial.finite_cell_closure_diagnostics().size());
        for (std::size_t i = 0; i < prepared.face_chain_diagnostics.size(); ++i)
        {
            EXPECT_EQ(prepared.face_chain_diagnostics[i].face, trial.finite_face_chain_diagnostics()[i].face);
            EXPECT_EQ(prepared.face_chain_diagnostics[i].signed_difference.value, 0);
        }
        for (const auto& closure : prepared.cell_closure_diagnostics)
            EXPECT_EQ(closure.combined_residual.value, 0);
        return result;
    }();
    static_assert(std::is_same_v<decltype(view.ledger()), const Prepared&>);
    EXPECT_EQ(view.arithmetic_mode, Mode);
    EXPECT_EQ(view.ledger().source_identity, source.accepted_identity());
    EXPECT_TRUE(view.ledger().candidate_identity);
    EXPECT_EQ(view.ledger().old_mass, view.ledger().new_mass);
    EXPECT_EQ(view.ledger().face_chain_diagnostics.size(), source.tiles().size());
    EXPECT_EQ(view.ledger().cell_closure_diagnostics.size(), input.order.size());
    EXPECT_EQ(view.ledger().diagnostics.swept_pieces, 0u);
    // Unit cells in this fixture give independent analytic capacities.
    for (std::size_t c = 0; c < input.order.size(); ++c)
        EXPECT_EQ(view.ledger().old_mass[c], input.gas[c] == 1 ? 1.L : 1000.L);
    if (input.order.empty())
    {
        EXPECT_TRUE(controls.nodes().empty());
        EXPECT_TRUE(controls.faces().empty());
        EXPECT_TRUE(view.ledger().old_mass.empty());
    }
}
TEST(GeneralVoFPreparedTransport, TypedViewsRetainModeDiagnosticsAndImmutableLifetime)
{
    const auto input = fixture();
    check_typed_stationary<ArithmeticMode::Ordinary>(input);
    check_typed_stationary<ArithmeticMode::BoundedCertification>(input);
    int empty = input.order.empty(), total_empty = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &empty, &total_empty);
    EXPECT_EQ(total_empty, std::max(0, Tpetra::getDefaultComm()->getSize() - 2));
}
TEST(GeneralVoFPreparedTransport, TypedFactoriesRejectWrongModeOnAnEmptyOwnerCollectively)
{
    const auto input = fixture();
    const auto comm = Tpetra::getDefaultComm();
    for (const auto mode : {ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification})
    {
        GeneralVoFOptions options;
        options.arithmetic_mode = mode;
        GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
        const auto nodes = stationary(source);
        const auto faces = circulation(source, true);
        const auto trial = source.plan_transport(nodes, faces, 1);
        const bool wrong_rank = comm->getRank() == comm->getSize() - 1;
        const bool request_ordinary = (mode == ArithmeticMode::Ordinary) != wrong_rank;
        if (request_ordinary)
        {
            EXPECT_THROW(
                (void) source.transport_controls<ArithmeticMode::Ordinary>(nodes, faces, 1), std::invalid_argument);
            EXPECT_THROW((void) trial.prepare_transport_view<ArithmeticMode::Ordinary>(), std::invalid_argument);
        }
        else
        {
            EXPECT_THROW((void) source.transport_controls<ArithmeticMode::BoundedCertification>(nodes, faces, 1),
                std::invalid_argument);
            EXPECT_THROW(
                (void) trial.prepare_transport_view<ArithmeticMode::BoundedCertification>(), std::invalid_argument);
        }
        if (comm->getSize() >= 4 && wrong_rank)
            EXPECT_TRUE(input.order.empty());
    }
}
TEST(GeneralVoFPreparedTransport, TypedControlsRejectForeignSourcesAndPayloads)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    GeneralVoF other(input.hierarchy, input.order, input.gas, input.carrier);
    const auto controls =
        source.transport_controls<ArithmeticMode::Ordinary>(stationary(source), circulation(source, true), 1);
    EXPECT_THROW((void) other.plan_transport(controls), std::invalid_argument);
    auto faces = circulation(source, true);
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getRank() == 0)
    {
        ASSERT_FALSE(faces.empty());
        faces.front().bounded_integrated_volume = fp::make_value(ArithmeticMode::BoundedCertification, 0);
    }
    EXPECT_THROW((void) source.transport_controls<ArithmeticMode::Ordinary>(stationary(source), faces, 1),
        std::invalid_argument);
    EXPECT_THROW((void) source.plan_transport(stationary(source), faces, 1), std::invalid_argument);
    GeneralVoFOptions exact_options;
    exact_options.arithmetic_mode = ArithmeticMode::Exact;
    GeneralVoF exact_source(input.hierarchy, input.order, input.gas, input.carrier, exact_options);
    if (comm->getRank() == 0)
        faces.front().exact_integrated_volume = SimpleFluid::PLIC::GeometryMeasure<ArithmeticMode::Exact>{};
    EXPECT_THROW((void) exact_source.plan_transport(stationary(exact_source), faces, 1), std::invalid_argument);
    const auto exact_controls = exact_source.transport_controls<ArithmeticMode::Exact>(
        stationary(exact_source), circulation(exact_source, true), 1);
    const auto exact_trial = exact_source.plan_transport(exact_controls);
    EXPECT_THROW((void) exact_trial.prepare_transport(), std::invalid_argument);
    EXPECT_THROW((void) exact_trial.prepare_transport_view<ArithmeticMode::Ordinary>(), std::invalid_argument);
}
TEST(GeneralVoFPreparedTransport, TypedPreparedContextRejectsForgedCompatibilityCopies)
{
    const auto input = fixture();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto controls =
        source.transport_controls<ArithmeticMode::Ordinary>(stationary(source), circulation(source), 1);
    const auto trial = source.plan_transport(controls);
    const auto view = trial.prepare_transport_view<ArithmeticMode::Ordinary>({1, 1000}, 9);
    const auto another = source.plan_transport(controls);
    EXPECT_THROW(another.validate_prepared_transport(view, {1, 1000}, 9), std::invalid_argument);
    EXPECT_THROW(trial.validate_prepared_transport(view, {1, 1000}, 10), std::invalid_argument);
    EXPECT_THROW(trial.validate_prepared_transport(view, {1, 10}, 9), std::invalid_argument);
    for (unsigned mutation = 0; mutation != 6; ++mutation)
    {
        auto forged = view.ledger();
        switch (mutation)
        {
            case 0:
                forged.density = {1, 10};
                break;
            case 1:
                ++forged.time_stage;
                break;
            case 2:
                forged.timestep *= 2;
                break;
            case 3:
                ++forged.source_control;
                break;
            case 4:
                forged.family = GeneralVoFTransportFamily::AlgebraicFaceVolume;
                break;
            case 5:
                forged.arithmetic_mode = ArithmeticMode::BoundedCertification;
                break;
        }
        EXPECT_THROW(source.validate_prepared_transport_source(forged), std::invalid_argument);
        EXPECT_THROW(
            trial.validate_prepared_transport(forged, forged.density, forged.time_stage), std::invalid_argument);
    }
    EXPECT_TRUE(view.ledger().positive_transfers.empty());
    EXPECT_EQ(view.ledger().family, GeneralVoFTransportFamily::InterfaceLocalSwept);
    EXPECT_NO_THROW(trial.validate_prepared_transport(view, {1, 1000}, 9));
}
TEST(GeneralVoFPreparedTransport, TypedFaceOnlyControlsPreserveAlgebraicBypass)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 0);
    std::fill(input.carrier.begin(), input.carrier.end(), 1);
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    const auto controls = source.transport_controls<ArithmeticMode::Ordinary>({}, circulation(source), 1);
    const auto trial = source.plan_homogeneous_transport(controls);
    const auto view = trial.prepare_transport_view<ArithmeticMode::Ordinary>({1, 1000}, 1);
    EXPECT_TRUE(view.ledger().node_trajectories.empty());
    EXPECT_EQ(view.ledger().family, GeneralVoFTransportFamily::AlgebraicFaceVolume);
    for (const auto mass : view.ledger().old_mass)
        EXPECT_EQ(mass, 1000);
    for (const auto mass : view.ledger().new_mass)
        EXPECT_EQ(mass, 1000);
    GeneralVoFOptions options;
    options.transport_backend = GeneralVoFTransportBackend::DonorCell;
    GeneralVoF face_source(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto face_controls =
        face_source.transport_controls<ArithmeticMode::Ordinary>({}, circulation(face_source), 1);
    const auto face_trial = face_source.plan_face_transport(face_controls, {1, 1000}, 1);
    EXPECT_EQ(face_trial.prepare_transport_view<ArithmeticMode::Ordinary>({1, 1000}, 1).ledger().family,
        GeneralVoFTransportFamily::AlgebraicFaceVolume);
}
TEST(GeneralVoFPreparedTransport, FaceOnlyViewsRejectRankLocalNodesAndMovedFromViewsCollectively)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 0);
    std::fill(input.carrier.begin(), input.carrier.end(), 1);
    const auto rank = Tpetra::getDefaultComm()->getRank();
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier);
    auto nodes = stationary(source);
    if (rank != 0)
        nodes.clear();
    const auto node_bearing = source.transport_controls<ArithmeticMode::Ordinary>(nodes, circulation(source), 1);
    EXPECT_THROW((void) source.plan_homogeneous_transport(node_bearing), std::invalid_argument);
    GeneralVoFOptions face_options;
    face_options.transport_backend = GeneralVoFTransportBackend::DonorCell;
    GeneralVoF face_source(input.hierarchy, input.order, input.gas, input.carrier, face_options);
    const auto face_node_bearing =
        face_source.transport_controls<ArithmeticMode::Ordinary>(nodes, circulation(face_source), 1);
    EXPECT_THROW((void) face_source.plan_face_transport(face_node_bearing), std::invalid_argument);
    auto controls =
        source.transport_controls<ArithmeticMode::Ordinary>(stationary(source), circulation(source, true), 1);
    std::optional<GeneralVoFControlView<ArithmeticMode::Ordinary>> retained_controls;
    if (rank == 0)
        retained_controls.emplace(std::move(controls));
    EXPECT_THROW((void) source.plan_transport(controls), std::invalid_argument);
    const auto trial = source.plan_transport(stationary(source), circulation(source, true), 1);
    auto prepared = trial.prepare_transport_view<ArithmeticMode::Ordinary>();
    std::optional<GeneralVoFPreparedTrialView<ArithmeticMode::Ordinary>> retained_prepared;
    if (rank == 0)
        retained_prepared.emplace(std::move(prepared));
    EXPECT_THROW(trial.validate_prepared_transport(prepared, {1, 1}, 0), std::invalid_argument);
}
TEST(GeneralVoFPreparedTransport, TypedControlsRejectMixedCollectiveSourcesIncludingEmptyOwners)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() == 1)
        GTEST_SKIP() << "Mixed collective publications require multiple ranks.";
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 0);
    std::fill(input.carrier.begin(), input.carrier.end(), 1);
    const bool use_first = comm->getRank() == comm->getSize() - 1;
    for (const auto backend : {GeneralVoFTransportBackend::SweptPartition, GeneralVoFTransportBackend::DonorCell})
    {
        GeneralVoFOptions options;
        options.transport_backend = backend;
        GeneralVoF first(input.hierarchy, input.order, input.gas, input.carrier, options);
        GeneralVoF second(input.hierarchy, input.order, input.gas, input.carrier, options);
        const auto& selected = use_first ? first : second;
        const auto nodes = backend == GeneralVoFTransportBackend::SweptPartition
                               ? stationary(selected)
                               : std::vector<GeneralVoFNodeVelocity>{};
        const auto faces = circulation(selected, true);
        EXPECT_THROW(
            (void) selected.transport_controls<ArithmeticMode::Ordinary>(nodes, faces, 1), std::invalid_argument);
        const auto first_controls = first.transport_controls<ArithmeticMode::Ordinary>(nodes, faces, 1);
        const auto second_controls = second.transport_controls<ArithmeticMode::Ordinary>(nodes, faces, 1);
        const auto& selected_controls = use_first ? first_controls : second_controls;
        if (backend == GeneralVoFTransportBackend::SweptPartition)
        {
            EXPECT_THROW((void) selected.plan_transport(selected_controls), std::invalid_argument);
            EXPECT_THROW((void) selected.plan_transport(nodes, faces, 1), std::invalid_argument);
            EXPECT_THROW((void) selected.plan_homogeneous_transport(faces, 1), std::invalid_argument);
        }
        else
        {
            EXPECT_THROW((void) selected.plan_face_transport(selected_controls), std::invalid_argument);
            EXPECT_THROW((void) selected.plan_face_transport(faces, 1), std::invalid_argument);
        }
        const auto first_trial = backend == GeneralVoFTransportBackend::SweptPartition
                                     ? first.plan_transport(first_controls)
                                     : first.plan_face_transport(first_controls);
        const auto second_trial = backend == GeneralVoFTransportBackend::SweptPartition
                                      ? second.plan_transport(second_controls)
                                      : second.plan_face_transport(second_controls);
        const auto first_prepared = first_trial.prepare_transport_view<ArithmeticMode::Ordinary>();
        const auto second_prepared = second_trial.prepare_transport_view<ArithmeticMode::Ordinary>();
        const auto& selected_prepared = use_first ? first_prepared : second_prepared;
        EXPECT_THROW(selected.validate_prepared_transport_source(selected_prepared.ledger()), std::invalid_argument);
    }
    if (comm->getSize() >= 4 && use_first)
        EXPECT_TRUE(input.order.empty());
}
} // namespace
