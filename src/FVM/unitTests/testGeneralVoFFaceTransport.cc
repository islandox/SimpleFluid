#include "FVM/details/GeneralVoF/FaceSweptPlaneBounds.hh"
#include "FVM/details/GeneralVoF/LimiterReplay.hh"
#include "FVM/details/GeneralVoF/OccupancyAdmission.hh"
#include "FVM/details/VoFFailure.hh"
#include "general_vof_finite_test_helpers.hh"
#include "geometry/PLIC/GeometryError.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
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
    const auto psi = [&](long double x, long double y) { return (dx * (y - 3) - dy * (x - 3)) * taper(x) * taper(y); };
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
TEST(GeneralVoFFaceTransport, RepeatedCanonicalDiscrepancyUsesSignedHistoryAndRejectsBeforePublication)
{
    // An exactly represented dyadic transfer across one internal face gives
    // delta(k)=+/-k*q. Twenty-eight increments fit the unchanged 1e-13 physical gate;
    // a twenty-ninth does not. Each increment is below the independent 8e-15
    // canonical continuity limit. Neither the arithmetic history nor a fresh per-step
    // allowance can authorize that twenty-ninth update.
    auto input = fixture();
    std::fill(input.gas.begin(), input.gas.end(), .25);
    std::fill(input.carrier.begin(), input.carrier.end(), .75);
    auto options = face_options();
    options.phase_cleanup.relative_cutoff = 0;
    auto state = std::make_shared<const GeneralVoF>(input.hierarchy, input.order, input.gas, input.carrier, options);
    constexpr double q = 0x1p-48;
    const auto controls = [&](double transfer)
    {
        std::vector<GeneralVoFFaceVolume> result;
        for (const auto& face : state->tiles())
        {
            const auto center = face.geometry->centroid();
            const auto area = face.geometry->area_vector();
            const bool selected = !face.exterior && center.x == 1 && center.y == .5 && center.z == .5;
            result.push_back({face.face_gid, selected ? std::copysign(transfer, area.x) : 0});
        }
        return result;
    };
    const auto check = [&](unsigned increments)
    {
        double local[3]{}, global[3]{};
        for (std::size_t c = 0; c < state->phase_volumes().size(); ++c)
        {
            const auto center =
                state->hierarchy()->geometry(state->hierarchy()->owned().at(state->owned_order()[c]))->centroid();
            const long double expected =
                center[1] == .5 && center[2] == .5 ? (center[0] < 1 ? -1.L : 1.L) * increments * q : 0;
            const auto& phase = state->phase_volumes()[c];
            const long double delta = static_cast<long double>(phase.gas) + phase.carrier - phase.bulk;
            EXPECT_EQ(delta, expected);
            EXPECT_LE(std::abs(delta), 1e-13L * phase.bulk);
            local[0] += phase.gas;
            local[1] += phase.carrier;
            local[2] += delta;
        }
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 3, local, global);
        EXPECT_EQ(global[0], 2);
        EXPECT_EQ(global[1], 6);
        EXPECT_EQ(global[2], 0);
    };
    for (unsigned step = 1; step <= 28; ++step)
    {
        const auto trial = state->plan_face_transport(controls(q), 1, {1, 1000}, step);
        state = state->apply(trial);
        check(step);
    }
    const auto identity = state->accepted_identity();
    const auto gas = state->gas_volumes(), carrier = state->carrier_volumes();
    EXPECT_THROW((void) state->plan_face_transport(controls(q), 1, {1, 1000}, 29), std::invalid_argument);
    EXPECT_EQ(state->accepted_identity(), identity);
    EXPECT_EQ(state->gas_volumes(), gas);
    EXPECT_EQ(state->carrier_volumes(), carrier);
    // Reversing the same canonical transfer restores capacity; a magnitude-only
    // history test would reject this valid recovery.
    auto reverse = controls(q);
    for (auto& face : reverse)
        face.integrated_volume = -face.integrated_volume;
    const auto recovery = state->plan_face_transport(reverse, 1, {1, 1000}, 29);
    state = state->apply(recovery);
    check(27);
}

TEST(GeneralVoFFaceTransport, PhysicalOccupancyPreservesNativeVolumeGateWithWideGeometry)
{
    if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "The wide/native box-volume witness needs an extended significand.";
    // Independent exact dyadic box-volume witness. The geometric product
    // differs from the native binary64 volume by exactly 2^-53.
    const double native = 1 + 3 * 0x1p-27;
    const long double geometric = (1 + 0x1p-26L) * (1 + 0x1p-27L);
    const double gas = native / 4 + 1803 * 0x1p-54, carrier = 3 * native / 4;
    const long double defect = static_cast<long double>(gas) + carrier - geometric;
    const long double uncertainty = 104 * std::numeric_limits<long double>::epsilon() * native;
    const auto shifted = algebraic_occupancy_admission(gas, carrier, native, geometric, defect, defect, uncertainty);
    EXPECT_EQ(geometric - native, 0x1p-53L);
    EXPECT_TRUE(shifted.geometry_admitted);
    EXPECT_FALSE(shifted.native_admitted);
    EXPECT_FALSE(shifted.publication_admitted);
    EXPECT_FALSE(shifted.admitted());
}

TEST(GeneralVoFFaceTransport, PhysicalOccupancyPreservesBinary64PublicationGate)
{
    // Here both wide-volume tests pass, but rounding the public double sum
    // alone crosses its original limit. This needs its own explicit guard.
    constexpr double volume = 1.25;
    const double g = volume / 4 + 2251 * 0x1p-54, l = 3 * volume / 4;
    const long double d = 2251 * 0x1p-54L;
    const auto rounded = algebraic_occupancy_admission(g, l, volume, volume, d, d, 0);
    EXPECT_TRUE(rounded.geometry_admitted);
    EXPECT_TRUE(rounded.native_admitted);
    EXPECT_FALSE(rounded.publication_admitted);
    EXPECT_FALSE(rounded.admitted());
    const double safe_g = volume / 4 + 2248 * 0x1p-54;
    const long double safe_d = 2248 * 0x1p-54L;
    EXPECT_TRUE(algebraic_occupancy_admission(safe_g, l, volume, volume, safe_d, safe_d, 0).admitted());
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

// Recorded donor-46 operands, rounded back to their original x87 values.
TEST(GeneralVoFFaceTransport, LimiterReplayRepairsOnlyArithmeticExcess)
{
    if (std::numeric_limits<long double>::digits != 64)
        GTEST_SKIP() << "Captured operands require the original x87 long-double format.";
    namespace replay = SimpleFluid::VoF::general_vof_detail::limiter_replay;
    using Stream = SimpleFluid::VoF::GeneralVoFDirectedStream;
    Stream a, b, untouched;
    a.face = 65;
    a.volume = 0x8959de30fdeed376p-83L;
    a.gas_low = 0x86ccac7872d913dep-109L;
    a.gas_high = 0xd82bbd41671cd438p-105L;
    a.liquid_low = 0x8959de0f4ac3b559p-83L;
    a.liquid_high = 0x8959dad04ef9cddap-83L;
    b.face = 159;
    b.volume = 0x8ed2cf1791916875p-75L;
    b.gas_low = 0x8c2b96fa8d926af1p-101L;
    b.gas_high = 0x94d31023fcc3ca40p-96L;
    b.liquid_low = 0x8ed2cef486aba9d2p-75L;
    b.liquid_high = 0x8ed2ca70f910488fp-75L;
    a.theta = b.theta = 0xd63f4ff2a1329e21p-64L;
    untouched.face = 199;
    untouched.volume = untouched.liquid_low = untouched.liquid_high = 0x1p-20L;
    const std::array<double, 2> density{1, 1000};
    for (auto* s : {&a, &b, &untouched}) replay::evaluate(*s, density);
    const std::vector<Stream> original{a, b, untouched};
    const std::array<long double, 2> old{0x1.f67d1ac23a12ep-34L, 0x1.ffffff8260b95p-8L};
    SimpleFluid::detail::CompensatedSum<long double> gl, ll;
    std::array<long double, 4> demand{};
    for (const auto& s : original)
    {
        gl += s.gas_low;
        ll += s.liquid_low;
        const auto c = replay::correction(s);
        if (c >= 0) demand[2] += c;
        else demand[3] -= c;
    }
    std::array<long double, 4> available{0, 0, old[0] - gl.value(), old[1] - ll.value()};
    std::array<long double, 4> ratio{1, 1, 1, 1};
    for (unsigned p = 2; p != 4; ++p)
        if (demand[p] > 0)
        {
            ratio[p] = std::min(1.L, available[p] / demand[p]);
            if (ratio[p] > 0 && ratio[p] < 1) ratio[p] = std::nextafter(ratio[p], 0.L);
        }
    ASSERT_EQ(ratio[2], a.theta);
    ASSERT_GT(replay::summarize(original).exported[0], old[0]);
    const auto repaired = replay::repair(original, old, available, demand, ratio, density);
    ASSERT_TRUE(repaired);
    const auto result = replay::summarize(*repaired);
    EXPECT_LE(result.exported[0], old[0]);
    EXPECT_LE(result.exported[1], old[1]);
    EXPECT_LT((*repaired)[0].theta, a.theta);
    EXPECT_GT((*repaired)[0].gas, 0);
    EXPECT_EQ((*repaired)[2].theta, untouched.theta);
    EXPECT_EQ((*repaired)[2].gas, 0);
    EXPECT_EQ((*repaired)[2].liquid, untouched.liquid);
    for (const auto& s : *repaired) EXPECT_EQ(s.mass, s.gas + 1000 * s.liquid);

    auto scaled = original;
    for (auto& s : scaled)
    {
        s.volume *= 2;
        s.gas_low *= 2;
        s.gas_high *= 2;
        s.liquid_low *= 2;
        s.liquid_high *= 2;
        replay::evaluate(s, density);
    }
    auto scaled_old = old;
    auto scaled_available = available;
    auto scaled_demand = demand;
    for (auto& v : scaled_old) v *= 2;
    for (auto& v : scaled_available) v *= 2;
    for (auto& v : scaled_demand) v *= 2;
    const auto scaled_repair = replay::repair(scaled, scaled_old, scaled_available, scaled_demand, ratio, density);
    ASSERT_TRUE(scaled_repair);
    EXPECT_EQ((*scaled_repair)[0].theta, (*repaired)[0].theta);
    EXPECT_EQ((*scaled_repair)[1].theta, (*repaired)[1].theta);

    auto bad_old = old;
    bad_old[0] *= .99L;
    auto bad_available = available;
    bad_available[2] = bad_old[0] - gl.value();
    auto bad_ratio = ratio;
    bad_ratio[2] = std::nextafter(bad_available[2] / demand[2], 0.L);
    EXPECT_FALSE(replay::repair(original, bad_old, bad_available, demand, bad_ratio, density));

    auto corrupted = original;
    corrupted[0].gas += 0x1p-40L;
    EXPECT_FALSE(replay::repair(corrupted, old, available, demand, ratio, density));

}

TEST(GeneralVoFFaceTransport, LimiterBlendRetainsEndpointsAndTinyMinority)
{
    namespace replay = SimpleFluid::VoF::general_vof_detail::limiter_replay;
    SimpleFluid::VoF::GeneralVoFDirectedStream s;
    s.volume = 1;
    s.gas_low = .25L;
    s.gas_high = 0x1p-100L;
    s.liquid_low = .75L;
    s.liquid_high = 1 - s.gas_high;
    for (const auto theta : {0.L, 1.L})
    {
        s.theta = theta;
        replay::evaluate(s, {1, 1000});
        EXPECT_EQ(s.gas, theta == 0 ? s.gas_low : s.gas_high);
        EXPECT_GT(s.gas, 0);
        EXPECT_EQ(s.mass, s.gas + 1000 * s.liquid);
    }
}

// The radial face is the one captured in the cylinder continuation failure:
// its independently integrated full area exceeds |published area_vector| by
// 2.54e-19 on the original GCC/x86 build. Closed streamfunction controls below
// are independent of the solver/projection and keep the sweep far from R=1.
struct RadialFaceFixture
{
    Fixture phases;
    std::map<std::pair<long double, long double>, long double> potential;
};
RadialFaceFixture radial_face_fixture()
{
    ExtrudedAMRGrid grid;
    const std::array<double, 4> radii{.8, .95, 1.05, 1.2};
    RadialFaceFixture result;
    constexpr long double amplitude = 0x1p-24L;
    for (unsigned a = 0; a < 5; ++a)
        for (unsigned r = 0; r < radii.size(); ++r)
        {
            const double angle = -.2 + .1 * a;
            const double x = radii[r] * std::cos(angle), y = radii[r] * std::sin(angle);
            grid.xy_nodes.push_back({x, y, 0});
            const auto weight = a == 0 || a == 4 || r == 0 || r == 3 ? 0 : a == 2 ? 2 : 1;
            result.potential.emplace(std::pair<long double, long double>{x, y}, amplitude * weight);
        }
    for (unsigned a = 0; a < 4; ++a)
        for (unsigned r = 0; r < 3; ++r)
        {
            const unsigned p = 4 * a + r;
            grid.base_cells.push_back({p, p + 1, p + 5, p + 4});
        }
    for (unsigned a = 0; a < 4; ++a)
    {
        grid.boundaries.push_back({4 * a, 4 * (a + 1), "wall"});
        grid.boundaries.push_back({4 * a + 3, 4 * (a + 1) + 3, "wall"});
    }
    for (unsigned r = 0; r < 3; ++r)
    {
        grid.boundaries.push_back({r, r + 1, "wall"});
        grid.boundaries.push_back({16 + r, 17 + r, "wall"});
    }
    grid.z_edges = {0, .2 / 3, 2 * .2 / 3, .2};
    const auto comm = Tpetra::getDefaultComm();
    ExtrudedAMR provider(grid);
    std::vector<std::uint64_t> owned;
    for (std::uint64_t c = 0; c < 36; ++c)
        if (int(c % std::min(2, comm->getSize())) == comm->getRank())
            owned.push_back(c);
    auto bootstrap = provider.detach_owned(owned, {'r', 'a', 'd', 'i', 'a', 'l', 'f', 'a', 'c', 'e'});
    auto& input = result.phases;
    input.hierarchy = std::make_shared<const DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    for (const auto& [key, leaf] : input.hierarchy->owned())
    {
        const auto geometry = input.hierarchy->geometry(leaf);
        const auto center = geometry->centroid();
        const auto radius = std::hypot(center[0], center[1]);
        double gas = 0, carrier = 0;
        if (radius < .95)
            gas = geometry->moments().volume.as_double();
        else if (radius > 1.05)
            carrier = geometry->moments().volume.as_double();
        else
        {
            // Independent polar integrals for the circular gas boundary R=1.
            const double chord = std::cos(.05) * std::cos(.05) * std::tan(.05);
            const double height = geometry->axis_spans()[2].as_double();
            gas = (.05 - radii[1] * radii[1] * chord) * height;
            carrier = (radii[2] * radii[2] * chord - .05) * height;
        }
        input.order.push_back(key);
        input.gas.push_back(gas);
        input.carrier.push_back(carrier);
    }
    return result;
}
std::vector<GeneralVoFFaceVolume> radial_face_controls(const GeneralVoF& state, const RadialFaceFixture& fixture)
{
    std::map<AdaptiveCellKey, PLIC::GeometryPoint> centers;
    for (const auto& cell : state.support_cells())
        centers.emplace(cell.key, cell.centroid);
    std::vector<GeneralVoFFaceVolume> controls;
    for (const auto& tile : state.tiles())
    {
        long double q = 0;
        if (!tile.exterior && tile.second)
        {
            // Stokes' theorem: curl(psi e_z) has canonical face integral
            // sum_edge psi * dz. Shared endpoint potentials telescope exactly
            // around each native cell; no fitted velocity supplies this oracle.
            const auto& points = tile.geometry->points();
            for (std::size_t i = 0; i < points.size(); ++i)
            {
                const auto& a = points[i];
                const auto& b = points[(i + 1) % points.size()];
                if (a[2] != b[2])
                    q += fixture.potential.at({a[0], a[1]}) * (b[2] - a[2]);
            }
            const auto av = tile.geometry->area_vector();
            const auto& first = centers.at(tile.first);
            const auto& second = centers.at(*tile.second);
            const auto orientation =
                av.x * (second[0] - first[0]) + av.y * (second[1] - first[1]) + av.z * (second[2] - first[2]);
            if (orientation < 0)
                q = -q;
        }
        controls.push_back({tile.face_gid, double(q)});
    }
    return controls;
}
TEST(GeneralVoFFaceTransport, CapturedRadialPureFacesRetainGeometricZeroWithoutFallback)
{
    const auto input = radial_face_fixture();
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    options.phase_cleanup.relative_cutoff = 0;
    GeneralVoF state(input.phases.hierarchy, input.phases.order, input.phases.gas, input.phases.carrier, options);
    const auto identity = state.accepted_identity();
    const auto trial = state.plan_face_transport(radial_face_controls(state, input), 1);
    std::map<std::uint64_t, PLIC::GeometryPoint> centers;
    for (const auto& cell : state.support_cells())
        centers.emplace(cell.cell, cell.centroid);
    unsigned local[3]{}, global[3]{};
    for (const auto& stream : trial.prepared_transport()->streams)
    {
        const auto& donor = centers.at(stream.donor);
        const auto& recipient = centers.at(stream.recipient);
        const auto donor_radius = std::hypot(donor[0], donor[1]);
        const auto recipient_radius = std::hypot(recipient[0], recipient[1]);
        if (donor_radius <= .95 || donor_radius >= 1.05)
            continue;
        if (recipient_radius > 1.05)
        {
            EXPECT_FALSE(stream.lower_order);
            EXPECT_GT(stream.gas_low, 0);
            EXPECT_EQ(stream.gas_high, 0);
            EXPECT_EQ(stream.liquid_high, stream.volume);
            if (stream.donor_owner == Tpetra::getDefaultComm()->getRank())
                ++local[0];
        }
        if (recipient_radius < .95)
        {
            EXPECT_FALSE(stream.lower_order);
            EXPECT_GT(stream.liquid_low, 0);
            EXPECT_EQ(stream.liquid_high, 0);
            EXPECT_EQ(stream.gas_high, stream.volume);
            if (stream.donor_owner == Tpetra::getDefaultComm()->getRank())
                ++local[1];
        }
    }
    local[2] = trial.prepared_transport()->diagnostics.fallback_streams;
    Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 3, local, global);
    EXPECT_GT(global[0], 0u);
    EXPECT_GT(global[1], 0u);
    EXPECT_EQ(global[2], 0u);
    const auto next = state.apply(trial);
    for (std::size_t i = 0; i < input.phases.order.size(); ++i)
    {
        if (input.phases.gas[i] == 0)
            EXPECT_EQ(next->gas_volumes()[i], 0);
        if (input.phases.carrier[i] == 0)
            EXPECT_EQ(next->carrier_volumes()[i], 0);
    }
    EXPECT_EQ(state.accepted_identity(), identity);
}
TEST(GeneralVoFFaceTransport, SweptExtremumOffsetBoundsRetainSubUlpPhysicalCrossing)
{
    constexpr double initial = 1;
    constexpr long double motion = 0x1p-54L, face_increment = 0x1p-55L;
    // A face with exact normal height 1+2^-55 is crossed at t=1/2 by
    // offset(t)=1+2^-54*t, although every binary64 endpoint rounds to 1.
    // The opposite displacement crosses the corresponding lower face.
    // These exact binary increments remain an independent oracle even when
    // long double and double have the same precision.
    for (const auto direction : {-1.L, 1.L})
    {
        ASSERT_EQ(static_cast<double>(static_cast<long double>(initial) + direction * motion), initial);
        const auto bounds = general_vof_detail::swept_face_offset_bounds(initial, direction * motion);
        ASSERT_TRUE(bounds);
        EXPECT_LE((*bounds)[0], initial);
        EXPECT_GE((*bounds)[1], initial);
        const auto outward = direction > 0 ? (*bounds)[1] - initial : initial - (*bounds)[0];
        EXPECT_GE(outward, motion);
        EXPECT_GT(outward, face_increment);
    }
    EXPECT_FALSE(general_vof_detail::swept_face_offset_bounds(initial, std::numeric_limits<long double>::infinity()));
}
TEST(GeneralVoFFaceTransport, SweptExtremumRetainsEndpointContactAndPositiveGrazingCrossing)
{
    const auto input = planar_patch(false);
    auto options = face_options();
    options.transport_backend = GeneralVoFTransportBackend::GeometricFace;
    options.phase_cleanup.relative_cutoff = 0;
    GeneralVoF state(input.hierarchy, input.order, input.gas, input.carrier, options);
    for (const double q : {.25, .25 + 0x1p-24})
    {
        const auto trial = state.plan_face_transport(planar_patch_controls(state, q, 0), 1);
        std::map<std::uint64_t, const AdaptiveTile*> tiles;
        for (const auto& tile : state.tiles())
            tiles.emplace(tile.face_gid, &tile);
        unsigned local = 0, global = 0;
        for (const auto& stream : trial.prepared_transport()->streams)
        {
            const auto& tile = *tiles.at(stream.face);
            const auto c = tile.geometry->centroid(), n = tile.geometry->area_vector();
            if (n.x == 0 || c.x != 3 || c.y != 2.5)
                continue;
            // Plane x=2.75 touches x=3 at the endpoint when q=.25.
            // The larger displacement crosses during a positive interval;
            // its independent time-integrated gas volume is q-.25.
            EXPECT_FALSE(stream.lower_order);
            EXPECT_NEAR(stream.gas_high, q - .25L, 2e-12L * q);
            if (q > .25)
                EXPECT_GT(stream.gas_high, 0);
            EXPECT_EQ(stream.gas_high + stream.liquid_high, stream.volume);
            local += stream.donor_owner == Tpetra::getDefaultComm()->getRank();
        }
        Teuchos::reduceAll(*Tpetra::getDefaultComm(), Teuchos::REDUCE_SUM, 1, &local, &global);
        EXPECT_EQ(global, 2u);
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
