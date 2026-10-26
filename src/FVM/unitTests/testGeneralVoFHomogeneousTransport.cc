#include "general_vof_finite_test_helpers.hh"
#include <limits>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
using Family = GeneralVoFTransportFamily;
using Declaration = GeneralVoFPhaseGeometry;
GeneralVoF pure_state(const Fixture& input, bool gas = true)
{
    return GeneralVoF(input.hierarchy, input.order, std::vector<double>(input.order.size(), gas ? 1 : 0),
        std::vector<double>(input.order.size(), gas ? 0 : 1), {},
        std::vector<Declaration>(input.order.size(), gas ? Declaration::PureGas : Declaration::PureCarrier));
}
std::vector<GeneralVoFFaceVolume> cycle(const GeneralVoF& state, double q)
{
    auto result = circulation(state);
    for (auto& f : result)
        f.integrated_volume *= q * 64;
    return result;
}
void expect_no_partition_work()
{
    const auto work = GeneralVoF::geometry_statistics();
    EXPECT_EQ(work.bounded.constructions, 0u);
    EXPECT_EQ(work.bounded.intersections, 0u);
    EXPECT_EQ(work.bounded.partition_checks, 0u);
}
TEST(GeneralVoFHomogeneousTransport, PureCirculationUsesCanonicalStreamsAtEveryPositiveMagnitude)
{
    const auto input = fixture();
    for (const bool gas : {false, true})
        for (const double q : {1. / 64, 1e-100, std::numeric_limits<double>::denorm_min() * 64})
        {
            auto state = pure_state(input, gas);
            GeneralVoF::reset_geometry_statistics();
            const auto trial = state.plan_transport(stationary(state), cycle(state, q), 1);
            EXPECT_FALSE(trial.prepared_transport());
            const auto prepared = trial.prepare_transport({2, 1000}, 19);
            const auto next = state.apply(trial);
            EXPECT_EQ(next->gas_volumes(), state.gas_volumes());
            EXPECT_EQ(next->carrier_volumes(), state.carrier_volumes());
            EXPECT_EQ(next->options().transport_backend, GeneralVoFTransportBackend::SweptPartition);
            EXPECT_EQ(prepared->backend, GeneralVoFTransportBackend::SweptPartition);
            EXPECT_EQ(prepared->family, Family::AlgebraicFaceVolume);
            EXPECT_EQ(prepared->arithmetic_mode, ArithmeticMode::Ordinary);
            EXPECT_EQ(prepared->source_identity, state.accepted_identity());
            EXPECT_EQ(prepared->candidate_identity, next->accepted_identity());
            EXPECT_EQ(prepared->time_stage, 19u);
            EXPECT_EQ(prepared->node_trajectories.size(), state.nodes().size());
            EXPECT_TRUE(trial.finite_face_chain_diagnostics().empty());
            EXPECT_TRUE(trial.finite_cell_closure_diagnostics().empty());
            for (const auto& s : prepared->streams)
            {
                EXPECT_EQ(s.volume, static_cast<long double>(q));
                EXPECT_GT(s.volume, 0);
                EXPECT_EQ(gas ? s.gas : s.liquid, s.volume);
                EXPECT_EQ(s.mass, 2 * s.gas + 1000 * s.liquid);
            }
            EXPECT_NO_THROW(state.validate_prepared_transport_source(*prepared));
            EXPECT_NO_THROW(trial.validate_prepared_transport(*prepared, {2, 1000}, 19));
            EXPECT_THROW((void) state.plan_face_transport(cycle(state, q), 1), std::invalid_argument);
            expect_no_partition_work();
            int local = int(prepared->streams.size()), global = 0;
            Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
            EXPECT_GT(global, 0);
        }
}
TEST(GeneralVoFHomogeneousTransport, VaryingIndependentTraceMatchesAnalyticCycle)
{
    const auto input = fixture();
    for (const bool gas : {false, true})
        for (const double trace_scale : {1e-25, 2e-16})
        {
            std::vector<double> minority;
            for (const auto& key : input.order)
            {
                const auto c = input.hierarchy->geometry(input.hierarchy->owned().at(key))->centroid();
                minority.push_back(trace_scale * (1 + int(c[0]) + 2 * int(c[1])));
            }
            GeneralVoF state(input.hierarchy, input.order, gas ? std::vector<double>(input.order.size(), 1) : minority,
                gas ? minority : std::vector<double>(input.order.size(), 1), {},
                std::vector<Declaration>(input.order.size(), gas ? Declaration::PureGas : Declaration::PureCarrier));
            for (const double q : {1. / 64, 1.})
            {
                GeneralVoF::reset_geometry_statistics();
                const auto trial = state.plan_transport(stationary(state), cycle(state, q), 1);
                const auto next = trial.candidate();
                for (std::size_t i = 0; i < input.order.size(); ++i)
                {
                    const auto c = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
                    const int x = int(c[0]), y = int(c[1]);
                    const int previous_x = y, previous_y = 1 - x;
                    // BL <- TL <- TR <- BR <- BL in both z layers.
                    const long double t = minority[i], previous = trace_scale * (1 + previous_x + 2 * previous_y);
                    const auto expected = t - q * t / (1 + t) + q * previous / (1 + previous);
                    const auto actual = gas ? next->carrier_volumes()[i] : next->gas_volumes()[i];
                    EXPECT_NEAR(actual, expected, 8 * std::numeric_limits<double>::epsilon() * expected);
                    EXPECT_GT(actual, 0);
                    EXPECT_EQ(
                        next->phase_geometry_declarations()[i], gas ? Declaration::PureGas : Declaration::PureCarrier);
                }
                const auto prepared = trial.prepare_transport({3, 997}, 22);
                for (const auto& s : prepared->streams)
                {
                    EXPECT_GT(gas ? s.liquid : s.gas, 0);
                    EXPECT_EQ(s.mass, 3 * s.gas + 997 * s.liquid);
                }
                expect_no_partition_work();
            }
        }
}
TEST(GeneralVoFHomogeneousTransport, AbsentNominalPhaseKeepsItsTransportedUncertainty)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 1);
    std::fill(input.carrier.begin(), input.carrier.end(), 0);
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto c = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        if (c[0] < 1 && c[1] < 1)
            input.carrier[i] = 1e-25;
    }
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, {},
        std::vector<Declaration>(input.order.size(), Declaration::PureGas));
    const auto first = state.plan_transport(stationary(state), cycle(state, 1), 1).candidate();
    const auto second = first->plan_transport(stationary(*first), cycle(*first, 1. / 64), 1);
    std::map<std::uint64_t, std::size_t> indices;
    for (std::size_t i = 0; i < first->owned_order().size(); ++i)
        indices.emplace(first->mesh()->owned_cell_map()->getGlobalElement(i), i);
    int local_absent_history = 0, absent_history = 0;
    for (const auto& route : second.finite_transfers())
        if (indices.contains(route.donor))
        {
            const auto i = indices.at(route.donor);
            if (first->carrier_volumes()[i] == 0 && first->accumulated_phase_errors()[i].carrier > 0)
            {
                ++local_absent_history;
                const long double fraction = route.donor == route.recipient ? 63.L / 64 : 1.L / 64;
                const auto expected = fraction * first->accumulated_phase_errors()[i].carrier;
                EXPECT_GE(route.inherited_error.carrier, expected);
                EXPECT_NEAR(
                    route.inherited_error.carrier, expected, 4 * std::numeric_limits<double>::epsilon() * expected);
            }
        }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_absent_history, &absent_history);
    EXPECT_GT(absent_history, 0);
    const auto next = second.candidate();
    double local_history[2]{}, global_history[2]{};
    for (const auto& h : first->accumulated_phase_errors())
        local_history[0] += h.carrier;
    for (const auto& h : next->accumulated_phase_errors())
        local_history[1] += h.carrier;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 2, local_history, global_history);
    EXPECT_GE(global_history[1], global_history[0]);
}
TEST(GeneralVoFHomogeneousTransport, CompleteNodeAndFaceAuthenticationStillRejectsAtomically)
{
    const auto input = fixture();
    auto state = pure_state(input);
    const auto identity = state.accepted_identity();
    auto nodes = stationary(state);
    if (!nodes.empty())
        nodes.front().velocity[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void) state.plan_transport(nodes, circulation(state), 1), std::invalid_argument);
    nodes = stationary(state);
    if (!nodes.empty())
        nodes.front().node = std::numeric_limits<std::uint64_t>::max();
    EXPECT_THROW((void) state.plan_transport(nodes, circulation(state), 1), std::invalid_argument);
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        auto flux = cycle(state, scenario == 0 ? 2 : 1. / 64);
        for (std::size_t f = 0; f < flux.size(); ++f)
        {
            if (scenario == 1 && state.tiles()[f].exterior)
                flux[f].integrated_volume = 1e-4;
            if (scenario == 2 && flux[f].integrated_volume > 0)
                flux[f].integrated_volume *= 2;
            if (scenario == 3)
                flux[f].bounded_integrated_volume = fp::Arithmetic<ArithmeticMode::BoundedCertification>::make_value(0);
        }
        EXPECT_THROW((void) state.plan_transport(stationary(state), flux, 1), std::invalid_argument);
    }
    // One rank mutates a shared node, preserving a valid local shape.
    nodes = stationary(state);
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        if (Tpetra::getDefaultComm()->getRank() == 0)
            for (auto& node : nodes)
                node.velocity = {1e-4, 2e-4, 3e-4};
        EXPECT_THROW((void) state.plan_transport(nodes, circulation(state), 1), std::invalid_argument);
        auto flux = circulation(state);
        if (Tpetra::getDefaultComm()->getRank() == 0)
            for (auto& f : flux)
                f.integrated_volume *= 1.5;
        EXPECT_THROW((void) state.plan_transport(stationary(state), flux, 1), std::invalid_argument);
    }
    EXPECT_EQ(state.accepted_identity(), identity);
}
TEST(GeneralVoFHomogeneousTransport, ActualFamilyIsAuthenticatedAndNoMotionRetainsSweptIdentity)
{
    const auto input = fixture();
    auto state = pure_state(input);
    const auto trial = state.plan_transport(stationary(state), circulation(state), 1);
    const auto prepared = trial.prepare_transport({1, 1000}, 5);
    auto copy = *prepared;
    copy.family = Family::GeometricSweptTetrahedra;
    EXPECT_THROW(state.validate_prepared_transport_source(copy), std::invalid_argument);
    copy = *prepared;
    copy.backend = GeneralVoFTransportBackend::DonorCell;
    EXPECT_THROW(state.validate_prepared_transport_source(copy), std::invalid_argument);
    const auto stationary_trial = state.plan_transport(stationary(state), circulation(state, true), 1);
    EXPECT_EQ(stationary_trial.candidate()->accepted_identity(), state.accepted_identity());
    const auto stationary_prepared = stationary_trial.prepare_transport({1, 1000}, 5);
    EXPECT_EQ(stationary_prepared->family, Family::GeometricSweptTetrahedra);
    EXPECT_TRUE(stationary_prepared->streams.empty());
    copy = *stationary_prepared;
    copy.family = Family::AlgebraicFaceVolume;
    EXPECT_THROW(state.validate_prepared_transport_source(copy), std::invalid_argument);
    EXPECT_THROW(stationary_trial.validate_prepared_transport(*prepared, {1, 1000}, 5), std::invalid_argument);
}
TEST(GeneralVoFHomogeneousTransport, OppositeDeclarationsRemainOnGeometricSweptPath)
{
    const auto input = fixture();
    std::vector<Declaration> declarations;
    for (const auto g : input.gas)
        declarations.push_back(g == 0 ? Declaration::PureCarrier : Declaration::PureGas);
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, {}, declarations);
    const auto trial = state.plan_transport(stationary(state), circulation(state), 1);
    EXPECT_EQ(trial.prepare_transport()->family, Family::InterfaceLocalSwept);
    EXPECT_TRUE(trial.prepare_transport()->streams.empty());
}
TEST(GeneralVoFHomogeneousTransport, InferAndBoundedStatesRemainOnGeometricSweptPath)
{
    auto input = fixture();
    for (const auto mode : {ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification})
    {
        GeneralVoFOptions options;
        options.arithmetic_mode = mode;
        GeneralVoF state(input.hierarchy, input.order, std::vector<double>(input.order.size(), 1),
            std::vector<double>(input.order.size(), 0), options,
            std::vector<Declaration>(input.order.size(), Declaration::Infer));
        for (const auto declaration : state.phase_geometry_declarations())
            EXPECT_EQ(declaration, Declaration::Infer);
        const auto trial = state.plan_transport(stationary(state), circulation(state), 1);
        EXPECT_EQ(trial.prepare_transport()->family, mode == ArithmeticMode::Ordinary ?
            Family::InterfaceLocalSwept : Family::GeometricSweptTetrahedra);
        EXPECT_TRUE(trial.prepare_transport()->streams.empty());
    }
}
TEST(GeneralVoFHomogeneousTransport, ExplicitFixedMeshEntryRetainsBackendWithoutSyntheticTrajectories)
{
    const auto input = fixture();
    auto state = pure_state(input);
    EXPECT_TRUE(state.homogeneous_transport_eligible());
    const auto trial = state.plan_homogeneous_transport(circulation(state), 1);
    const auto prepared = trial.prepare_transport({2, 1000}, 31);
    EXPECT_FALSE(trial.prepared_transport());
    EXPECT_EQ(prepared->family, Family::AlgebraicFaceVolume);
    EXPECT_EQ(prepared->backend, GeneralVoFTransportBackend::SweptPartition);
    EXPECT_TRUE(prepared->node_trajectories.empty());
    EXPECT_EQ(prepared->source_identity, state.accepted_identity());
    EXPECT_EQ(trial.candidate()->gas_volumes(), state.gas_volumes());
    for (const auto& stream : prepared->streams)
    {
        EXPECT_EQ(stream.volume, 1.L / 64);
        EXPECT_EQ(stream.mass, 2 * stream.gas + 1000 * stream.liquid);
    }
    EXPECT_NO_THROW(trial.validate_prepared_transport(*prepared, {2, 1000}, 31));
}
TEST(GeneralVoFHomogeneousTransport, ExplicitZeroFluxEntryPreservesPublicationAndHistories)
{
    const auto input = fixture();
    auto state = pure_state(input);
    const auto moved = state.plan_homogeneous_transport(circulation(state), 1).candidate();
    const auto trial = moved->plan_homogeneous_transport(circulation(*moved, true), 1);
    const auto prepared = trial.prepare_transport({1, 1000}, 32);
    EXPECT_EQ(trial.candidate()->accepted_identity(), moved->accepted_identity());
    EXPECT_EQ(prepared->candidate_identity, moved->accepted_identity());
    EXPECT_EQ(prepared->family, Family::AlgebraicFaceVolume);
    EXPECT_TRUE(prepared->node_trajectories.empty());
    EXPECT_TRUE(prepared->streams.empty());
    EXPECT_TRUE(prepared->signed_face_contributions.empty());
    EXPECT_TRUE(trial.finite_face_chain_diagnostics().empty());
    EXPECT_TRUE(trial.finite_cell_closure_diagnostics().empty());
    EXPECT_EQ(prepared->positive_transfers.size(), input.order.size());
    EXPECT_EQ(prepared->old_mass, prepared->new_mass);
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        EXPECT_EQ(prepared->old_phase_history[i].gas, prepared->new_phase_history[i].gas);
        EXPECT_EQ(prepared->old_phase_history[i].carrier, prepared->new_phase_history[i].carrier);
        EXPECT_EQ(prepared->phase_roundoff[i].gas, 0);
        EXPECT_EQ(prepared->phase_roundoff[i].carrier, 0);
        EXPECT_EQ(prepared->positive_transfers[i].inherited_error.gas, prepared->old_phase_history[i].gas);
    }
    auto invalid = circulation(*moved, true);
    if (!invalid.empty())
        invalid.front().face = std::numeric_limits<std::uint64_t>::max();
    EXPECT_THROW((void) moved->plan_homogeneous_transport(invalid, 1), std::invalid_argument);
}
TEST(GeneralVoFHomogeneousTransport, ExplicitEntryRejectsIneligibleSourcesAndAuthenticatesFaces)
{
    auto input = fixture();
    for (const auto mode : {ArithmeticMode::Ordinary, ArithmeticMode::BoundedCertification})
    {
        GeneralVoFOptions options;
        options.arithmetic_mode = mode;
        for (const auto declaration : {Declaration::PureGas, Declaration::Infer})
        {
            if (mode == ArithmeticMode::Ordinary && declaration == Declaration::PureGas)
                continue;
            GeneralVoF state(input.hierarchy, input.order, std::vector<double>(input.order.size(), 1),
                std::vector<double>(input.order.size(), 0), options,
                std::vector<Declaration>(input.order.size(), declaration));
            EXPECT_FALSE(state.homogeneous_transport_eligible());
            EXPECT_THROW((void) state.plan_homogeneous_transport(circulation(state), 1), std::invalid_argument);
        }
    }
    GeneralVoF opposite(input.hierarchy, input.order, input.gas, input.carrier);
    EXPECT_FALSE(opposite.homogeneous_transport_eligible());
    EXPECT_THROW((void) opposite.plan_homogeneous_transport(circulation(opposite), 1), std::invalid_argument);
    GeneralVoF mixed(input.hierarchy, input.order, std::vector<double>(input.order.size(), .5),
        std::vector<double>(input.order.size(), .5));
    EXPECT_FALSE(mixed.homogeneous_transport_eligible());
    EXPECT_THROW((void) mixed.plan_homogeneous_transport(circulation(mixed), 1), std::invalid_argument);
    auto state = pure_state(input);
    auto invalid = circulation(state);
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        if (Tpetra::getDefaultComm()->getRank() == 0)
            for (auto& control : invalid)
                control.integrated_volume *= 1.5;
        EXPECT_THROW((void) state.plan_homogeneous_transport(invalid, 1), std::invalid_argument);
    }
    invalid = circulation(state, true);
    for (std::size_t f = 0; f < invalid.size(); ++f)
        if (state.tiles()[f].exterior)
            invalid[f].integrated_volume = 1e-4;
    EXPECT_THROW((void) state.plan_homogeneous_transport(invalid, 1), std::invalid_argument);
    EXPECT_THROW((void) state.plan_homogeneous_transport(circulation(state, true), -1), std::invalid_argument);
    for (const long double error : {-1.L, std::numeric_limits<long double>::quiet_NaN()})
    {
        invalid = circulation(state, true);
        for (auto& control : invalid)
        {
            auto value = fp::Arithmetic<ArithmeticMode::Ordinary>::make_value(0);
            value.absolute_error = error;
            control.bounded_integrated_volume = value;
        }
        EXPECT_THROW((void) state.plan_homogeneous_transport(invalid, 1), std::invalid_argument);
    }
    if (Tpetra::getDefaultComm()->getSize() > 1)
    {
        invalid = circulation(state, true);
        for (auto& control : invalid)
        {
            auto value = fp::Arithmetic<ArithmeticMode::Ordinary>::make_value(0);
            value.absolute_error = Tpetra::getDefaultComm()->getRank() == 0 ? 1e-30L : 0;
            control.bounded_integrated_volume = value;
        }
        EXPECT_THROW((void) state.plan_homogeneous_transport(invalid, 1), std::invalid_argument);
    }
}
} // namespace
