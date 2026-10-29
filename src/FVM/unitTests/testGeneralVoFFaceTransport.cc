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
// Independent unit-square integration; this fixture never uses a PLIC cut to
// initialize the plane. A streamfunction closes the flow at the box walls,
// while the central [2,4]^2 patch has constant displacement (dx,dy,0).
Fixture planar_patch(bool diagonal, double cell_height = 1)
{
    const auto comm = Tpetra::getDefaultComm();
    AMRGrid grid;
    grid.block_cells = {6, 6, 2};
    grid.extent = {6, 6, 2 * cell_height};
    CartesianAMR mesh(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t id = 0; id < 72; ++id)
        if (int(id % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(id);
    auto source = mesh.detach_owned(owned, {'p', 'l', 'a', 'n', 'a', 'r', '_', 'p', 'a', 't', 'c', 'h'});
    Fixture input;
    input.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(source.owned), std::move(source.families), source.kernel, comm);
    for (const auto& [key, leaf] : input.hierarchy->owned())
    {
        const auto c = input.hierarchy->geometry(leaf)->centroid();
        const auto x = std::floor(c[0]), y = std::floor(c[1]);
        long double gas = std::clamp(2.75L - x, 0.L, 1.L);
        if (diagonal)
        {
            const auto t = 5.2L - x - y;
            gas = t <= 0 ? 0 : t >= 2 ? 1 : t <= 1 ? t * t / 2 : 1 - (2 - t) * (2 - t) / 2;
        }
        input.order.push_back(key);
        input.gas.push_back(double(gas * cell_height));
        input.carrier.push_back(double((1 - gas) * cell_height));
    }
    return input;
}
std::vector<GeneralVoFFaceVolume> planar_patch_controls(const GeneralVoF& state, long double dx, long double dy)
{
    const auto taper = [](long double x) { return std::min({1.L, x / 2, (6 - x) / 2}); };
    const auto psi = [&](long double x, long double y)
    { return (dx * (y - 3) - dy * (x - 3)) * taper(x) * taper(y); };
    std::vector<GeneralVoFFaceVolume> result;
    for (const auto& face : state.tiles())
    {
        const auto c = face.geometry->centroid(), n = face.geometry->area_vector();
        long double q = 0;
        if (!face.exterior && n.x != 0)
            q = (psi(c.x, c.y + .5L) - psi(c.x, c.y - .5L)) * n.x;
        if (!face.exterior && n.y != 0)
            q = (psi(c.x - .5L, c.y) - psi(c.x + .5L, c.y)) * n.y;
        result.push_back({face.face_gid, double(q)});
    }
    return result;
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
    const auto input = planar_patch(false);
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    for (const double q : {1. / 64, .25, .5})
    {
        const auto flux = planar_patch_controls(state, q, 0);
        const auto trial = state.plan_face_transport(flux, 1, {1, 1000});
        const auto next = state.apply(trial);
        EXPECT_EQ(trial.prepared_transport()->diagnostics.fallback_streams, 0u);
        std::map<std::uint64_t, const AdaptiveTile*> tiles;
        for (const auto& tile : state.tiles())
            tiles.emplace(tile.face_gid, &tile);
        int local_faces = 0, global_faces = 0;
        for (const auto& stream : trial.prepared_transport()->streams)
        {
            const auto& tile = *tiles.at(stream.face);
            const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
            if (n.x != 0 && c.x == 3 && c.y == 2.5)
            {
                // The plane x=2.75 reaches x=3 at t=.25/q. Its independent
                // space-time face integral is max(0,q-.25).
                EXPECT_NEAR(stream.volume, q, 1e-15);
                EXPECT_NEAR(stream.gas_high, std::max(0.L, q - .25L), 2e-12L * stream.volume);
                local_faces += stream.donor_owner == Tpetra::getDefaultComm()->getRank();
            }
            EXPECT_EQ(stream.gas + stream.liquid, stream.volume);
            EXPECT_EQ(stream.mass, stream.gas + 1000 * stream.liquid);
        }
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
        EXPECT_EQ(global_faces, 2);
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
TEST(GeneralVoFFaceTransport, ObliquePlanarFaceIntegralIncludesTangentialDisplacement)
{
    const auto input = planar_patch(true);
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto trial = state.plan_face_transport(planar_patch_controls(state, .2L, .1L), 1);
    std::map<std::uint64_t, const AdaptiveTile*> tiles;
    for (const auto& tile : state.tiles())
        tiles.emplace(tile.face_gid, &tile);
    int local_faces = 0, global_faces = 0;
    for (const auto& stream : trial.prepared_transport()->streams)
    {
        const auto& tile = *tiles.at(stream.face);
        const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
        const bool x_face = n.x != 0 && c.x == 3 && c.y == 2.5;
        const bool y_face = n.y != 0 && c.y == 3 && c.x == 2.5;
        if (!x_face && !y_face)
            continue;
        // In donor [2,3]^2 the exact moving gas plane is
        // x+y <= 5.2+.3t. Each selected unit face has gas area .2+.3t.
        // Assert the candidate before limiting, independently of reconstruction
        // and of the final conservative blend.
        const long double q = x_face ? .2L : .1L;
        EXPECT_NEAR(stream.volume, q, 1e-15);
        EXPECT_FALSE(stream.lower_order);
        EXPECT_NEAR(stream.gas_high, q * (.2L + .3L / 2), 2e-12L * q);
        EXPECT_EQ(stream.gas_high + stream.liquid_high, stream.volume);
        local_faces += stream.donor_owner == Tpetra::getDefaultComm()->getRank();
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
    EXPECT_EQ(global_faces, 4);
}
TEST(GeneralVoFFaceTransport, ReversedObliquePlanarFaceIntegralUsesIncomingDonor)
{
    const auto input = planar_patch(true);
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    const auto trial = state.plan_face_transport(planar_patch_controls(state, -.2L, -.1L), 1);
    std::map<std::uint64_t, const AdaptiveTile*> tiles;
    for (const auto& tile : state.tiles())
        tiles.emplace(tile.face_gid, &tile);
    int local_faces = 0, global_faces = 0;
    for (const auto& stream : trial.prepared_transport()->streams)
    {
        const auto& tile = *tiles.at(stream.face);
        const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
        const bool x_face = n.x != 0 && c.x == 3 && c.y == 2.5;
        const bool y_face = n.y != 0 && c.y == 3 && c.x == 2.5;
        if (!x_face && !y_face)
            continue;
        // Reversal selects the neighboring old donors. On either selected
        // unit face x+y<=5.2-.3t gives gas area max(0,.2-.3t),
        // whose exact integral is the triangle .2^2/(2*.3).
        const long double q = x_face ? .2L : .1L;
        EXPECT_NEAR(stream.volume, q, 1e-15);
        EXPECT_FALSE(stream.lower_order);
        EXPECT_NEAR(stream.gas_high, q * .2L * .2L / (2 * .3L), 2e-12L * q);
        EXPECT_EQ(stream.gas_high + stream.liquid_high, stream.volume);
        EXPECT_EQ(stream.gas + stream.liquid, stream.volume);
        local_faces += stream.donor_owner == Tpetra::getDefaultComm()->getRank();
    }
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local_faces, &global_faces);
    EXPECT_EQ(global_faces, 4);
}
TEST(GeneralVoFFaceTransport, PredictorConditioningRejectsAfterAdmittedPlaneReconstruction)
{
    // Unit x/y widths and dyadic height 1/16 give the area-weighted normal
    // matrix diag(1/8,1/8,2), hence condition 16. Reconstruction normalizes
    // neighbor directions; preflight below must admit its stencil at limit 8.
    // Use the declared gradient orientation and measure-only offset path so
    // angular fitting and construction arithmetic cannot mask this condition
    // rejection. The original target-volume tolerance remains unchanged.
    const auto input = planar_patch(false, 1. / 16);
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    options.reconstruction.maximum_condition = 8;
    options.reconstruction.minimum_optimized_phase_length_ratio = 1;
    options.reconstruction.measure_only = true;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    ASSERT_NO_THROW((void) state.reconstruct());
    const auto before = state.accepted_identity();
    bool rejected = false;
    try
    {
        (void) state.plan_face_transport(planar_patch_controls(state, .2L, 0), 1);
    }
    catch (const std::exception& e)
    {
        rejected = true;
        const auto* failure = dynamic_cast<const SimpleFluid::detail::vof_failure::Failure*>(&e);
        ASSERT_NE(failure, nullptr);
        EXPECT_STREQ(failure->record().message.data(), "Face predictor incident normals are ill conditioned.");
    }
    EXPECT_TRUE(rejected);
    EXPECT_EQ(state.accepted_identity(), before);
    EXPECT_EQ(state.gas_volumes(), input.gas);
    EXPECT_EQ(state.carrier_volumes(), input.carrier);
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
