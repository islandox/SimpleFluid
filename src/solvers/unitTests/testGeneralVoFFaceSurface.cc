/** @file testGeneralVoFFaceSurface.cc
 * @brief Explicit face-aligned surfaces and bounded planar capillary admission.
 */
#include "general_vof_solver_test_helpers.hh"
namespace
{
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);
std::shared_ptr<const VoF::GeneralVoF> face_surface(bool stepped = false, bool traces = false,
    bool single_layer = false, VoF::GeneralVoFTransportBackend backend = VoF::GeneralVoFTransportBackend::GeometricFace)
{
    const auto source = phases(1, false, single_layer);
    auto gas = source->gas_volumes(), liquid = source->carrier_volumes();
    auto declarations = source->phase_geometry_declarations();
    if (stepped)
        for (std::size_t c = 0; c < source->owned_order().size(); ++c)
        {
            const auto geometry =
                source->hierarchy()->geometry(source->hierarchy()->owned().at(source->owned_order()[c]));
            const auto center = geometry->centroid();
            const bool is_gas = center[0] + center[1] < 2;
            const auto volume = geometry->volume().as_double();
            gas[c] = is_gas ? volume : 0;
            liquid[c] = is_gas ? 0 : volume;
            declarations[c] =
                is_gas ? VoF::GeneralVoFPhaseGeometry::PureGas : VoF::GeneralVoFPhaseGeometry::PureCarrier;
        }
    if (traces)
        for (std::size_t c = 0; c < gas.size(); ++c)
            (declarations[c] == VoF::GeneralVoFPhaseGeometry::PureGas ? liquid[c] : gas[c]) = 1e-25;
    VoF::GeneralVoFOptions options;
    options.transport_backend = backend;
    return std::make_shared<const VoF::GeneralVoF>(
        source->hierarchy(), source->owned_order(), gas, liquid, options, declarations);
}
TEST(GeneralVoFFaceSurface, NativeOppositePureFacesExposeSurfaceWithoutChangingTraceInventories)
{
    const auto state = face_surface(false, true);
    const auto gas = state->gas_volumes(), liquid = state->carrier_volumes();
    const auto faces = state->face_interfaces();
    const auto comm = state->hierarchy()->communicator();
    double local[2]{}, global[2]{};
    for (const auto& face : faces)
    {
        EXPECT_EQ(face.normal, (PLIC::GeometryPoint{1, 0, 0}));
        EXPECT_EQ(face.centroid[0], 1);
        EXPECT_EQ(face.area, 1);
        EXPECT_NE(face.first_cell, face.second_cell);
        if (face.first_owner == comm->getRank())
        {
            ++local[0];
            local[1] += face.area;
        }
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local, global);
    EXPECT_EQ(global[0], 9);
    EXPECT_EQ(global[1], 9);
    EXPECT_EQ(state->gas_volumes(), gas);
    EXPECT_EQ(state->carrier_volumes(), liquid);
}
TEST(GeneralVoFFaceSurface, SupportedPlanarAlignedInterfaceRetainsStaticSurfaceTensionEquilibrium)
{
    auto material = inviscid();
    material.surface_tension = .07;
    const auto state = face_surface();
    GeneralVoFSolver solver(state, material);
    for (const auto k : solver.curvature())
        EXPECT_EQ(k, 0);
    const auto before = solver.diagnostics();
    for (unsigned step = 0; step < 3; ++step)
    {
        const auto after = solver.step(.01);
        EXPECT_EQ(after.maximum_velocity, 0);
        EXPECT_EQ(after.gas_volume, before.gas_volume);
        EXPECT_EQ(after.gas_mass, before.gas_mass);
        EXPECT_EQ(after.liquid_mass, before.liquid_mass);
    }
}
TEST(GeneralVoFFaceSurface, BinaryStaircaseAndDeficientPlanarSupportRejectCapillaryAdmission)
{
    auto material = inviscid();
    material.surface_tension = .07;
    EXPECT_THROW((void) GeneralVoFSolver(face_surface(true), material), std::runtime_error);
    EXPECT_THROW((void) GeneralVoFSolver(face_surface(false, false, true), material), std::runtime_error);
    // The legacy backend retains its existing pure-interface rejection.
    EXPECT_THROW((void) GeneralVoFSolver(phases(1), material), std::runtime_error);
}
std::shared_ptr<const VoF::GeneralVoF> striped_surface(unsigned owners)
{
    const auto comm = Tpetra::getDefaultComm();
    AMR::ExtrudedAMRGrid grid;
    for (unsigned y = 0; y <= owners; ++y)
        for (unsigned x = 0; x <= 3; ++x)
            grid.xy_nodes.push_back({double(x), double(y), 0});
    for (unsigned y = 0; y < owners; ++y)
        for (unsigned x = 0; x < 3; ++x)
        {
            const auto first = 4 * y + x;
            grid.base_cells.push_back({first, first + 1, first + 5, first + 4});
        }
    for (unsigned x = 0; x < 3; ++x)
    {
        grid.boundaries.push_back({x, x + 1, "wall"});
        grid.boundaries.push_back({4 * owners + x, 4 * owners + x + 1, "wall"});
    }
    for (unsigned y = 0; y < owners; ++y)
    {
        grid.boundaries.push_back({4 * y, 4 * (y + 1), "wall"});
        grid.boundaries.push_back({4 * y + 3, 4 * (y + 1) + 3, "wall"});
    }
    grid.z_edges = {0, 1, 2, 3};
    const auto provider = std::make_shared<AMR::ExtrudedAMR>(grid);
    std::vector<std::uint64_t> owned;
    // Complete x/z columns belong to one tangent y stripe. Every interface
    // face incident on an owner lies on its one-dimensional z line; all
    // independent y-tangent samples belong exclusively to other owners.
    for (std::uint64_t gid = 0; gid < 9 * owners; ++gid)
        if (int((gid / 3) % owners) == comm->getRank())
            owned.push_back(gid);
    auto bootstrap = provider->detach_owned(owned, {'s', 'u', 'r', 'f', 'a', 'c', 'e', 's', 't', 'r', 'i', 'p', 'e'});
    const auto hierarchy = std::make_shared<const AMR::DistributedAdaptiveHierarchy>(
        std::move(bootstrap.owned), std::move(bootstrap.families), bootstrap.kernel, comm);
    std::vector<VoF::GeneralVoF::Key> order;
    std::vector<double> gas, liquid;
    for (const auto& [key, leaf] : hierarchy->owned())
    {
        const auto geometry = hierarchy->geometry(leaf);
        const auto volume = geometry->volume().as_double();
        const bool is_gas = geometry->centroid()[0] < 1;
        order.push_back(key);
        gas.push_back(is_gas ? volume : 0);
        liquid.push_back(is_gas ? 0 : volume);
    }
    VoF::GeneralVoFOptions options;
    options.transport_backend = VoF::GeneralVoFTransportBackend::GeometricFace;
    return std::make_shared<const VoF::GeneralVoF>(hierarchy, order, gas, liquid, options);
}
void check_striped_surface(unsigned owners)
{
    const auto state = striped_surface(owners);
    const auto comm = state->hierarchy()->communicator();
    const bool owns_cells = !state->owned_order().empty();
    int local[2]{int(owns_cells), int(!owns_cells)}, global[2]{};
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local, global);
    EXPECT_EQ(global[0], int(owners));
    EXPECT_EQ(global[1], comm->getSize() - int(owners));
    const auto faces = state->face_interfaces();
    unsigned local_authoritative = 0, remote_only = 0;
    for (const auto& face : faces)
    {
        EXPECT_EQ(face.normal, (PLIC::GeometryPoint{1, 0, 0}));
        EXPECT_EQ(face.centroid[0], 1);
        EXPECT_EQ(face.area, 1);
        if (face.first_owner == comm->getRank())
            ++local_authoritative;
        if (face.first_owner != comm->getRank() && face.second_owner != comm->getRank())
            ++remote_only;
    }
    EXPECT_EQ(local_authoritative, owns_cells ? 3u : 0u);
    if (owns_cells)
        EXPECT_GT(remote_only, 0u);
    else
        EXPECT_TRUE(faces.empty());
    unsigned authoritative = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_authoritative, &authoritative);
    EXPECT_EQ(authoritative, 3 * owners);
    auto material = inviscid();
    material.surface_tension = .07;
    GeneralVoFSolver solver(state, material);
    for (const auto k : solver.curvature())
        EXPECT_EQ(k, 0);
    const auto before = solver.diagnostics();
    const auto after = solver.step(.01);
    EXPECT_EQ(after.maximum_velocity, 0);
    EXPECT_EQ(after.gas_mass, before.gas_mass);
    EXPECT_EQ(after.liquid_mass, before.liquid_mass);
}
TEST(GeneralVoFFaceSurface, ThreeOrMoreTangentStripesUseAuthenticatedHaloFaceSupport)
{
    const auto ranks = Tpetra::getDefaultComm()->getSize();
    if (ranks < 3)
        GTEST_SKIP() << "Requires at least three owning ranks for independent tangent stripes.";
    check_striped_surface(unsigned(ranks));
}
TEST(GeneralVoFFaceSurface, ThreeOrMoreTangentStripesRetainCollectivesWithAnEmptyOwner)
{
    const auto ranks = Tpetra::getDefaultComm()->getSize();
    if (ranks < 4)
        GTEST_SKIP() << "Requires at least three owning ranks and one empty rank.";
    check_striped_surface(unsigned(ranks - 1));
}
} // namespace
