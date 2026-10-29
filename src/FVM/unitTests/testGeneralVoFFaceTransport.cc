#include "FVM/details/VoFFailure.hh"
#include "general_vof_finite_test_helpers.hh"
#include "geometry/PLIC/GeometryError.hh"
#include <limits>
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
GeneralVoFOptions face_options()
{
    GeneralVoFOptions o;
    o.transport_backend = GeneralVoFTransportBackend::DonorCell;
    return o;
}
TEST(GeneralVoFFaceTransport, MovingPurePhasesNeedNoInterfaceGeometry)
{
    for (const bool gas : {false, true})
        for (const double scale : {1., 1e-100, std::numeric_limits<double>::denorm_min() * 64})
        {
            auto input = fixture();
            std::fill(input.gas.begin(), input.gas.end(), gas ? 1 : 0);
            std::fill(input.carrier.begin(), input.carrier.end(), gas ? 0 : 1);
            GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, face_options());
            auto flux = circulation(state);
            for (auto& f : flux)
                f.integrated_volume *= scale;
            GeneralVoF::reset_geometry_statistics();
            const auto trial = state.plan_face_transport(flux, 1, {1, 1000}, 9);
            const auto next = state.apply(trial);
            EXPECT_EQ(next->gas_volumes(), state.gas_volumes());
            EXPECT_EQ(next->carrier_volumes(), state.carrier_volumes());
            ASSERT_TRUE(trial.prepared_transport());
            EXPECT_EQ(trial.prepared_transport()->time_stage, 9u);
            EXPECT_EQ(trial.prepared_transport()->diagnostics.plane_measurements, 0u);
            EXPECT_TRUE(trial.finite_node_trajectories().empty());
            const auto work = GeneralVoF::geometry_statistics();
            EXPECT_EQ(work.bounded.constructions, 0u);
            EXPECT_EQ(work.bounded.partition_checks, 0u);
            for (const auto& s : trial.prepared_transport()->streams)
            {
                EXPECT_GT(s.volume, 0);
                EXPECT_EQ(s.gas + s.liquid, s.volume);
                EXPECT_EQ(s.mass, s.gas + 1000 * s.liquid);
            }
        }
}
TEST(GeneralVoFFaceTransport, OppositePureNeighbourReceivesIndependentPhase)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, face_options());
    const auto trial = state.plan_face_transport(circulation(state), 1, {1, 1000});
    const auto next = state.apply(trial);
    double local[3]{}, global[3]{};
    for (std::size_t i = 0; i < next->phase_volumes().size(); ++i)
    {
        const auto p = next->phase_volumes()[i];
        EXPECT_GE(p.gas, 0);
        EXPECT_GE(p.carrier, 0);
        EXPECT_DOUBLE_EQ(p.gas + p.carrier, 1);
        local[0] += p.gas;
        local[1] += p.carrier;
        if (state.gas_volumes()[i] == 0 && p.gas > 0)
        {
            ++local[2];
            EXPECT_DOUBLE_EQ(p.gas, 1. / 64);
        }
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 3, local, global);
    EXPECT_DOUBLE_EQ(global[0], 1);
    EXPECT_DOUBLE_EQ(global[1], 7);
    EXPECT_EQ(global[2], 1);
    EXPECT_EQ(state.gas_volumes(), input.gas);
}
TEST(GeneralVoFFaceTransport, GeometricallyPureTraceIsTransported)
{
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), 1.);
    std::fill(input.carrier.begin(), input.carrier.end(), 1e-25);
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, face_options(),
        std::vector<GeneralVoFPhaseGeometry>(input.order.size(), GeneralVoFPhaseGeometry::PureGas));
    const auto trial = state.plan_face_transport(circulation(state), 1);
    const auto next = state.apply(trial);
    for (std::size_t i = 0; i < next->phase_volumes().size(); ++i)
    {
        EXPECT_EQ(next->carrier_volumes()[i], 1e-25);
        EXPECT_EQ(next->phase_geometry_declarations()[i], GeneralVoFPhaseGeometry::PureGas);
    }
    for (const auto& s : trial.prepared_transport()->streams)
        EXPECT_GT(s.liquid, 0);
}
TEST(GeneralVoFFaceTransport, RejectsCflDivergenceBoundaryAndForeignSourceAtomically)
{
    const auto input = fixture();
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, face_options());
    const auto original = state.accepted_identity();
    auto flux = circulation(state);
    for (auto& f : flux)
        f.integrated_volume *= 128;
    EXPECT_THROW((void) state.plan_face_transport(flux, 1), std::invalid_argument);
    flux = circulation(state);
    for (auto& f : flux)
        if (f.integrated_volume > 0)
            f.integrated_volume *= 2;
    EXPECT_THROW((void) state.plan_face_transport(flux, 1), std::invalid_argument);
    flux = circulation(state);
    for (std::size_t i = 0; i < flux.size(); ++i)
        if (state.tiles()[i].exterior)
            flux[i].integrated_volume = 1e-4;
    EXPECT_THROW((void) state.plan_face_transport(flux, 1), std::invalid_argument);
    const auto trial = state.plan_face_transport(circulation(state), 1);
    const auto next = state.apply(trial);
    EXPECT_THROW((void) next->apply(trial), std::invalid_argument);
    EXPECT_EQ(state.accepted_identity(), original);
}
TEST(GeneralVoFFaceTransport, GeometricPlanarFaceIntegralUsesPrescribedBulk)
{
    auto input = fixture();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto c = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = c[0] < 1 ? .75 : 0;
        input.carrier[i] = 1 - input.gas[i];
    }
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    for (const double q : {1. / 64, .25, .5})
    {
        auto flux = circulation(state);
        for (auto& f : flux)
            f.integrated_volume *= q * 64;
        const auto trial = state.plan_face_transport(flux, 1, {1, 1000});
        const auto next = state.apply(trial);
        EXPECT_EQ(trial.prepared_transport()->diagnostics.fallback_streams, 0u);
        std::map<std::uint64_t, const AdaptiveTile*> tiles;
        for (const auto& tile : state.tiles())
            tiles.emplace(tile.face_gid, &tile);
        for (const auto& stream : trial.prepared_transport()->streams)
        {
            const auto& tile = *tiles.at(stream.face);
            const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
            const auto expected = n.y != 0 && c.x < 1   ? .75L * stream.volume
                                  : n.x != 0 && c.y < 1 ? std::max(0.L, stream.volume - .25L)
                                                        : 0.L;
            EXPECT_NEAR(stream.gas_high, expected, 2e-12L * stream.volume);
            EXPECT_EQ(stream.gas + stream.liquid, stream.volume);
            EXPECT_EQ(stream.mass, stream.gas + 1000 * stream.liquid);
        }
        for (const auto& p : next->phase_volumes())
            EXPECT_NEAR(p.gas + p.carrier, p.bulk, 1e-15);
    }
}
TEST(GeneralVoFFaceTransport, DefaultGeometricCleanupPreparesPublishedEndpointMass)
{
    const auto input = fixture();
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    GeneralVoF source(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto identity = source.accepted_identity();
    constexpr double q = 0x1p-48;
    constexpr std::array<double, 2> density{.1, 1000};
    constexpr std::uint64_t stage = 42;
    auto flux = circulation(source);
    for (auto& face : flux)
        face.integrated_volume *= q * 64;
    const auto trial = source.plan_face_transport(flux, 1, density, stage);
    const auto next = source.apply(trial);
    EXPECT_EQ(source.accepted_identity(), identity);
    EXPECT_EQ(source.gas_volumes(), input.gas);
    EXPECT_EQ(source.carrier_volumes(), input.carrier);
    EXPECT_EQ(source.phase_cleanup_state().cleaned_cells, 0u);
    EXPECT_TRUE(source.latest_phase_cleanup().empty());
    EXPECT_EQ(next->gas_volumes(), input.gas);
    EXPECT_EQ(next->carrier_volumes(), input.carrier);
    const auto cleanup = next->phase_cleanup_state();
    EXPECT_EQ(cleanup.cleaned_cells, 2u);
    EXPECT_EQ(cleanup.signed_volume, (std::array<long double, 2>{0, 0}));
    EXPECT_EQ(cleanup.absolute_volume, (std::array<long double, 2>{2 * q, 2 * q}));
    EXPECT_EQ(cleanup.policy.relative_cutoff, 1e-14);
    EXPECT_DOUBLE_EQ(cleanup.policy.maximum_absolute_volume[0], 1e-13);
    EXPECT_DOUBLE_EQ(cleanup.policy.maximum_absolute_volume[1], 7 * 1e-13);
    for (std::size_t phase = 0; phase < 2; ++phase)
        EXPECT_LE(cleanup.absolute_volume[phase], cleanup.policy.maximum_absolute_volume[phase]);
    double local_count = next->latest_phase_cleanup().size(), global_count = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_count, &global_count);
    EXPECT_EQ(global_count, 2);

    const auto prepared = trial.prepared_transport();
    ASSERT_TRUE(prepared);
    EXPECT_NO_THROW(trial.validate_prepared_transport(*prepared, density, stage));
    EXPECT_EQ(prepared->source_identity, identity);
    EXPECT_EQ(prepared->candidate_identity, next->accepted_identity());
    EXPECT_EQ(prepared->old_phase_cleanup.cleaned_cells, 0u);
    EXPECT_EQ(prepared->new_phase_cleanup.cleaned_cells, 2u);
    EXPECT_EQ(prepared->new_phase_cleanup.absolute_volume, cleanup.absolute_volume);
    ASSERT_EQ(prepared->phase_cleanup.size(), next->latest_phase_cleanup().size());
    ASSERT_EQ(prepared->new_mass.size(), input.order.size());
    ASSERT_EQ(prepared->raw_new_mass.size(), input.order.size());
    ASSERT_EQ(prepared->cleanup_mass.size(), input.order.size());
    for (std::size_t row = 0; row < input.order.size(); ++row)
        EXPECT_EQ(prepared->new_mass[row], static_cast<long double>(density[0]) * next->gas_volumes()[row] +
                                               static_cast<long double>(density[1]) * next->carrier_volumes()[row]);
    for (const auto& delta : prepared->phase_cleanup)
    {
        const auto row = next->mesh()->owned_cell_map()->getLocalElement(delta.gid);
        ASSERT_GE(row, 0);
        const bool gas = input.gas[row] == 1;
        EXPECT_EQ(delta.before, (gas ? std::array<double, 2>{1 - q, q} : std::array<double, 2>{q, 1 - q}));
        EXPECT_EQ(delta.after, (std::array<double, 2>{input.gas[row], input.carrier[row]}));
        for (std::size_t phase = 0; phase < 2; ++phase)
            EXPECT_EQ(delta.volume[phase], static_cast<long double>(delta.after[phase]) - delta.before[phase]);
        // Use the represented density, not the decimal real .1.
        EXPECT_EQ(prepared->raw_new_mass[row], static_cast<long double>(density[0]) * delta.before[0] +
                                                   static_cast<long double>(density[1]) * delta.before[1]);
        EXPECT_EQ(prepared->cleanup_mass[row], static_cast<long double>(density[0]) * delta.volume[0] +
                                                   static_cast<long double>(density[1]) * delta.volume[1]);
    }
    const auto repeated = trial.prepare_transport(density, stage);
    EXPECT_NO_THROW(trial.validate_prepared_transport(*repeated, density, stage));
    EXPECT_EQ(repeated->candidate_identity, prepared->candidate_identity);
    EXPECT_EQ(repeated->raw_new_mass, prepared->raw_new_mass);
    EXPECT_EQ(repeated->new_mass, prepared->new_mass);
    EXPECT_EQ(repeated->cleanup_mass, prepared->cleanup_mass);
    EXPECT_EQ(repeated->new_phase_cleanup.cleaned_cells, 2u);
    EXPECT_EQ(repeated->new_phase_cleanup.signed_volume, cleanup.signed_volume);
    EXPECT_EQ(repeated->new_phase_cleanup.absolute_volume, cleanup.absolute_volume);
    EXPECT_EQ(next->phase_cleanup_state().cleaned_cells, 2u);
    EXPECT_EQ(next->phase_cleanup_state().absolute_volume, cleanup.absolute_volume);
}
TEST(GeneralVoFFaceTransport, LimiterSharesEndpointAndOldExportCapacities)
{
    auto input = fixture();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto c = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = c[0] < 1 && c[1] < 1 ? .125 : c[0] > 1 && c[1] > 1 ? 1 : .875;
        input.carrier[i] = 1 - input.gas[i];
    }
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    auto flux = circulation(state);
    for (std::size_t f = 0; f < flux.size(); ++f)
    {
        flux[f].integrated_volume *= 12.8;
        const auto& face = state.tiles()[f];
        const auto c = face.geometry->centroid(), n = face.geometry->area_vector();
        if (!face.exterior && c.y == 1 && n.y != 0)
            flux[f].integrated_volume += (c.z < 1 ? .4 : -.4) * (n.y > 0 ? 1 : -1);
        if (!face.exterior && c.z == 1 && n.z != 0)
            flux[f].integrated_volume += (c.y > 1 ? .4 : -.4) * (n.z > 0 ? 1 : -1);
    }
    const auto trial = state.plan_face_transport(flux, 1);
    const auto next = state.apply(trial);
    double local[3]{}, global[3]{};
    local[2] = trial.prepared_transport()->diagnostics.limited_streams;
    for (const auto& p : next->phase_volumes())
    {
        EXPECT_GE(p.gas, 0);
        EXPECT_GE(p.carrier, 0);
        EXPECT_NEAR(p.gas + p.carrier, 1, 1e-14);
        local[0] += p.gas;
        local[1] += p.carrier;
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 3, local, global);
    EXPECT_NEAR(global[0], 5.75, 1e-14);
    EXPECT_NEAR(global[1], 2.25, 1e-14);
    EXPECT_GT(global[2], 0);
    for (const auto& budget : trial.finite_donor_budgets())
    {
        EXPECT_LE(budget.exported.gas.value, budget.old.gas.value);
        EXPECT_LE(budget.exported.carrier.value, budget.old.carrier.value);
    }
}

TEST(GeneralVoFFaceTransport, MixedDonorPureFaceDoesNotManufactureSurfaceFromRecipientTrace)
{
    auto input = fixture();
    std::vector<GeneralVoFPhaseGeometry> declarations;
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        const bool mixed = center[0] < 1;
        input.gas[i] = mixed ? .75 : 1e-25;
        input.carrier[i] = mixed ? .25 : 1;
        declarations.push_back(mixed ? GeneralVoFPhaseGeometry::Infer : GeneralVoFPhaseGeometry::PureCarrier);
    }
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    options.phase_cleanup.relative_cutoff = 0;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options, declarations);
    const auto next = state.apply(state.plan_face_transport(circulation(state), 1));
    const auto hints = next->phase_geometry_declarations();
    for (std::size_t i = 0; i < hints.size(); ++i)
        if (state.carrier_volumes()[i] == 1)
        {
            EXPECT_GT(next->gas_volumes()[i], 0);
            EXPECT_EQ(hints[i], GeneralVoFPhaseGeometry::PureCarrier);
        }
}

TEST(GeneralVoFFaceTransport, NumericalFloorFallsBackButResourceExhaustionRejects)
{
    auto input = fixture();
    for (std::size_t i = 0; i < input.order.size(); ++i)
    {
        const auto center = input.hierarchy->geometry(input.hierarchy->owned().at(input.order[i]))->centroid();
        input.gas[i] = center[0] < 1 ? .75 : 0;
        input.carrier[i] = 1 - input.gas[i];
    }
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    options.reconstruction.volume_tolerance = 1e-30;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto trial = state.plan_face_transport(circulation(state), 1);
    double local = trial.prepared_transport()->diagnostics.fallback_streams, global = 0;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
    EXPECT_GT(global, 0);
    for (const auto& stream : trial.prepared_transport()->streams)
    {
        EXPECT_EQ(stream.gas, stream.gas_low);
        EXPECT_EQ(stream.liquid, stream.liquid_low);
    }
    options.reconstruction.volume_tolerance = 8e-15;
    options.reconstruction.maximum_clip_evaluations = 1;
    GeneralVoF exhausted(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto before = exhausted.accepted_identity();
    bool rejected = false;
    try
    {
        (void) exhausted.plan_face_transport(circulation(exhausted), 1);
    }
    catch (const SimpleFluid::detail::vof_failure::Failure& error)
    {
        const auto& record = error.record();
        EXPECT_EQ(record.category, SimpleFluid::detail::vof_failure::Category::ResourceLimit);
        EXPECT_TRUE(record.has(SimpleFluid::detail::vof_failure::Field::GeometryCode));
        EXPECT_EQ(record.geometry_code, int(PLIC::GeometryErrorCode::ResourceLimit));
        EXPECT_GE(record.origin_rank, 0);
        EXPECT_LT(record.origin_rank, Tpetra::getDefaultComm()->getSize());
        rejected = true;
    }
    EXPECT_TRUE(rejected);
    EXPECT_EQ(exhausted.accepted_identity(), before);
}

} // namespace
