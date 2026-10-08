/**
 * @file testMultiRegionALE.cc
 * @author islandox
 * @brief Composite affine ALE through the existing motion and transport contracts.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#include <gtest/gtest.h>
#include "FVM/ALEControlVolumeState.tcc"
#include "FVM/Operators.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "geometry/unitTests/region_mesh_helpers.hh"
#include "solvers/BelosLinearSolver.hh"
#include "utils/testing_environment.hh"
#include <Tpetra_Core.hpp>
#include <array>
#include <limits>
#include <utility>
namespace
{
using namespace SimpleFluid;
using Pack=DefaultTpetraTypes;
using Handle=MeshHandle<>;
testing::Environment* const environment=testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

class ThrowingHandle : public Handle
{
public:
    using Handle::Handle;
    bool fail_epoch = false, fail_faces = false, fail_owner = false;
    mutable size_t face_traversals = 0;

    std::uint64_t geometry_epoch() const
    {
        if (fail_epoch) throw std::runtime_error("Injected rank-local static geometry failure.");
        return Handle::geometry_epoch();
    }

    CellFaceRange faces(local_ordinal_type cell) const
    {
        ++face_traversals;
        if (fail_faces) throw std::runtime_error("Injected rank-local face traversal failure.");
        return Handle::faces(cell);
    }
    local_ordinal_type owner_cell(local_ordinal_type face) const
    {
        if (fail_owner) throw std::runtime_error("Injected rank-local owner lookup failure.");
        return Handle::owner_cell(face);
    }
};

class MutableTestMotion final : public MeshMotionModel
{
public:
    explicit MutableTestMotion(std::shared_ptr<ThrowingHandle> mesh)
        : d_mesh(std::move(mesh)), d_old(d_mesh->num_local_cells()), d_new(d_mesh->num_local_cells()),
          d_flux(d_mesh->num_faces(), 0.0)
    {
        for (size_t local = 0; local < d_old.size(); ++local)
        {
            const auto lid = static_cast<Handle::local_ordinal_type>(local);
            d_old[local] = d_new[local] = d_mesh->cell_volume(lid);
        }
        d_diagnostics.time_step = 1.0;
        d_diagnostics.old_geometry_epoch = d_mesh->geometry_epoch();
        d_diagnostics.new_geometry_epoch = d_mesh->geometry_epoch();
        d_diagnostics.trial_active = true;
    }
    void begin_trial(real_t, real_t) override {}
    void accept_trial() override { d_active = false; d_diagnostics.trial_active = false; }
    void rollback_trial() override { d_active = false; d_diagnostics.trial_active = false; }
    bool has_active_trial() const noexcept override { return d_active; }
    std::string_view mesh_family() const noexcept override { return "test"; }
    std::span<const real_t> old_cell_volumes() const noexcept override { return d_old; }
    std::span<const real_t> new_cell_volumes() const noexcept override { return d_new; }
    std::span<const real_t> face_mesh_fluxes() const noexcept override { return d_flux; }
    const MeshMotionDiagnostics& diagnostics() const noexcept override { return d_diagnostics; }
    const std::shared_ptr<ThrowingHandle>& mesh_ptr() const noexcept { return d_mesh; }
    void set_active(bool active) { d_active = active; d_diagnostics.trial_active = active; }
    void set_new_epoch(std::uint64_t epoch) { d_diagnostics.new_geometry_epoch = epoch; }
    void set_flux(size_t face, real_t flux) { d_flux[face] = flux; }
private:
    std::shared_ptr<ThrowingHandle> d_mesh;
    std::vector<real_t> d_old, d_new, d_flux;
    MeshMotionDiagnostics d_diagnostics;
    bool d_active = true;
};

enum class RejectionCategory { None, InvalidArgument, LogicError, Other };

template<class Action>
void expect_collective_error(const Teuchos::Comm<int>& comm, Action&& action,
    RejectionCategory expected_category, const char* expected_message)
{
    std::string message;
    auto category = RejectionCategory::None;
    try { action(); }
    catch (const std::invalid_argument& error)
    {
        category = RejectionCategory::InvalidArgument;
        message = error.what();
    }
    catch (const std::logic_error& error)
    {
        category = RejectionCategory::LogicError;
        message = error.what();
    }
    catch (const std::exception& error)
    {
        category = RejectionCategory::Other;
        message = error.what();
    }
    catch (...) { category = RejectionCategory::Other; }
    EXPECT_EQ(category, expected_category);
    EXPECT_EQ(message, expected_message);
    const std::array<int, 2> local{1, category == expected_category && message == expected_message};
    std::array<int, 2> global{};
    Teuchos::reduceAll(comm, Teuchos::REDUCE_SUM, static_cast<int>(local.size()), local.data(), global.data());
    EXPECT_EQ(global[0], comm.getSize());
    EXPECT_EQ(global[1], comm.getSize());
}

void check_gcl_face_order(const Handle& mesh, const FVM::ALEControlVolumeState& ale)
{
    const auto execution = mesh.acquire_execution_view();
    for (size_t owned = 0; owned < mesh.num_owned_cells(); ++owned)
    {
        const auto cell = static_cast<Handle::local_ordinal_type>(owned);
        std::vector<Handle::local_ordinal_type> from_range, from_visit;
        real_t range_balance = 0, visit_balance = 0;
        for (const auto face : mesh.faces(cell))
        {
            from_range.push_back(face);
            const auto flux = ale.face_mesh_fluxes()[static_cast<size_t>(face)];
            range_balance += mesh.owner_cell(face) == cell ? flux : -flux;
        }
        mesh.visit_cell_faces(cell, [&](const auto face)
        {
            from_visit.push_back(face);
            const auto flux = ale.face_mesh_fluxes()[static_cast<size_t>(face)];
            visit_balance += mesh.owner_cell(face) == cell ? flux : -flux;
        });
        EXPECT_EQ(from_visit, from_range);
        EXPECT_DOUBLE_EQ(visit_balance, range_balance);
    }
}
}
TEST(MultiRegionALETest, AffineMotionPreservesConstantsAndCanonicalGcl)
{
    auto geometry=test::two_regions(); auto mesh=std::make_shared<Handle>(geometry);
    auto alias=std::make_shared<Handle>(std::static_pointer_cast<const Meshes::MultiRegionMesh>(geometry));
    const auto topology=geometry->regions();
    ScalarCellFieldStored<Pack> old(mesh,2.75,"old"), unit(mesh,1.0,"unit"), zero(mesh,0.0,"zero");
    FVM::TransportGeometryCache<Handle> cache(*mesh);
    PlanarALEMeshMotion<> motion(mesh);
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh)),std::logic_error);
    motion.begin_trial(1.3,0.2);
    EXPECT_EQ(alias->geometry_epoch(),mesh->geometry_epoch());
    EXPECT_LT(motion.diagnostics().maximum_absolute_gcl_residual,2e-12);
    auto ale=FVM::make_ale_control_volume_state(*mesh,motion);
    ScalarFaceFieldStored<Pack> absolute(mesh,0.0,"absolute"), relative(mesh,"relative");
    FVM::mesh_relative_face_fluxes(absolute,ale,relative);
    const auto system=FVM::weighted_scalar_transport_system<Pack>(FVM::MeshWeightedScalarTransportRequest<Pack,Handle>{
        .old_values=old,.face_fluxes=relative,.time_step=0.2,.storage_weight=unit,.advection_weight=unit,.diffusivity=zero,
        .boundary_condition=[](int,size_t){return BoundaryCondition{BoundaryConditionType::Dirichlet,2.75};},
        .boundary_value=[](int,size_t){return 2.75;},.source=[](int){return 0.0;},
        .treatment=FVM::NonOrthogonalTreatment::Explicit,.ale=&ale});
    Pack::vector_type result(mesh->owned_cell_map(),true); BelosLinearSolver<> solver;
    LinearSolverOptions options; options.tolerance=1e-12;
    EXPECT_TRUE(solver.solve(system.matrix,*system.rhs,result,options));
    const auto values=result.getData(); for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(values[c],2.75,2e-11);
    motion.rollback_trial(); EXPECT_EQ(mesh->geometry_epoch(),2U);
    for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(mesh->cell_volume(c),0.125,2e-14);
    motion.begin_trial(0.8,0.2); EXPECT_LT(motion.diagnostics().maximum_absolute_gcl_residual,2e-12); motion.accept_trial();
    EXPECT_EQ(mesh->connectivity_storage_bytes(),0U); EXPECT_FALSE(mesh->has_materialized_indexer());
    // Constituent coordinates stay immutable; only the composite geometry moves.
    std::visit([](const auto& r){EXPECT_EQ(r.geometry().geometry_epoch(),0U);},topology[0]);
}
TEST(MultiRegionALETest, InvalidMotionAndQualityFailureLeaveAcceptedGeometry)
{
    auto mesh=std::make_shared<Handle>(test::two_regions());
    PlanarALEMeshMotionOptions invalid; invalid.axis=Dimension::X;
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh,invalid)),std::invalid_argument);
    invalid.axis=Dimension::Z; invalid.deformation_start_elevation=0.4;
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh,invalid)),std::invalid_argument);
    PlanarALEMeshMotionOptions options; options.quality_limits.maximum_aspect_ratio=2.0;
    PlanarALEMeshMotion<> motion(mesh,options);
    EXPECT_THROW(motion.begin_trial(5.0,0.2),std::runtime_error);
    EXPECT_FALSE(motion.has_active_trial());
    for(size_t c=0;c<mesh->num_owned_cells();++c) EXPECT_NEAR(mesh->cell_volume(c),0.125,2e-14);
    motion.begin_trial(1.1,0.2); motion.accept_trial();
}

TEST(MultiRegionALETest, ExtendedFamiliesPreserveAffineGcl)
{
    for(const auto& geometry:{test::coarse_fine_regions(),test::periodic_regions(),test::cylindrical_regions(),test::independent_extruded_regions()})
    {
        auto mesh=std::make_shared<Handle>(geometry);
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.25,0.2);
        EXPECT_LT(motion.diagnostics().maximum_absolute_gcl_residual,2e-12);
        const auto expansion = FVM::make_ale_control_volume_state(*mesh, motion);
        EXPECT_NO_THROW(expansion.validate(*mesh));
        check_gcl_face_order(*mesh, expansion);
        motion.rollback_trial();
        motion.begin_trial(0.75,0.2);
        EXPECT_LT(motion.diagnostics().maximum_absolute_gcl_residual,2e-12);
        const auto contraction = FVM::make_ale_control_volume_state(*mesh, motion);
        EXPECT_NO_THROW(contraction.validate(*mesh));
        check_gcl_face_order(*mesh, contraction);
        motion.accept_trial();
        EXPECT_EQ(mesh->connectivity_storage_bytes(),0U);
    }
}

TEST(MultiRegionALETest, RetainedStateRejectsAcceptAndIdenticalTrialReplay)
{
    auto mesh = std::make_shared<Handle>(test::two_regions());
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.25, 0.2);
    const auto accepted = FVM::make_ale_control_volume_state(*mesh, motion);
    EXPECT_NO_THROW(accepted.validate(*mesh));
    motion.accept_trial();
    EXPECT_THROW(accepted.validate(*mesh), std::logic_error);

    motion.begin_trial(1.25, 0.2);
    const auto replayed = FVM::make_ale_control_volume_state(*mesh, motion);
    EXPECT_NO_THROW(replayed.validate(*mesh));
    motion.rollback_trial();
    EXPECT_THROW(replayed.validate(*mesh), std::logic_error);
    motion.begin_trial(1.25, 0.2);
    EXPECT_THROW(replayed.validate(*mesh), std::invalid_argument);
    const auto current = FVM::make_ale_control_volume_state(*mesh, motion);
    EXPECT_NO_THROW(current.validate(*mesh));
    motion.rollback_trial();
}

TEST(MultiRegionALETest, EmptyRanksParticipateInCollectiveGclValidation)
{
    auto geometry = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{
            Meshes::cartesian_region("single", {{{0,1}, {0,1}, {0,1}}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    auto mesh = std::make_shared<Handle>(geometry, options);
    const int local_empty = mesh->num_owned_cells() == 0;
    int global_empty = 0;
    const auto comm = mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &global_empty);
    EXPECT_EQ(global_empty, comm->getSize() - 1);
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.1, 0.2);
    const auto ale = FVM::make_ale_control_volume_state(*mesh, motion);
    EXPECT_NO_THROW(ale.validate(*mesh));
    motion.rollback_trial();
}

TEST(MultiRegionALETest, EmptyRanksReceiveCollectiveGclFailure)
{
    auto geometry = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{
            Meshes::cartesian_region("single", {{{0,1}, {0,1}, {0,1}}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    Handle::DistributionOptions options;
    options.allow_empty_partitions = true;
    auto mesh = std::make_shared<ThrowingHandle>(geometry, options);
    const auto comm = mesh->owned_cell_map()->getComm();
    const int local_empty = mesh->num_owned_cells() == 0;
    int global_empty = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &global_empty);
    EXPECT_EQ(global_empty, comm->getSize() - 1);
    MutableTestMotion motion(mesh);
    const auto ale = FVM::make_ale_control_volume_state(*mesh, motion);
    if (!local_empty)
    {
        const auto face = static_cast<size_t>(mesh->faces(Handle::local_ordinal_type{0}).front());
        motion.set_flux(face, 1.0);
    }
    const auto gcl_error = std::string("ALE control-volume state violates the cellwise geometric conservation law; maximum residual is ")
        + std::to_string(1.0) + " m^3/s.";
    expect_collective_error(*comm, [&] { ale.validate(*mesh); },
        RejectionCategory::InvalidArgument, gcl_error.c_str());

    auto fresh_mesh = std::make_shared<ThrowingHandle>(geometry, options);
    MutableTestMotion fresh_motion(fresh_mesh);
    const auto fresh_ale = FVM::make_ale_control_volume_state(*fresh_mesh, fresh_motion);
    EXPECT_NO_THROW(fresh_ale.validate(*fresh_mesh));
}

TEST(MultiRegionALETest, RankLocalStaleChildFailsValidationCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    {
        const Vec3D<ArrReal> left_edges{{{0,0.5,1}, {0,0.5,1}, {0,0.5,1}}};
        const Vec3D<ArrReal> right_edges{{{1,1.5,2}, {0,0.5,1}, {0,0.5,1}}};
        auto left = std::make_shared<Meshes::OrthogonalCartesian3D>(left_edges);
        auto right = std::make_shared<Meshes::OrthogonalCartesian3D>(right_edges);
        auto geometry = std::make_shared<Meshes::MultiRegionMesh>(
            std::vector<Meshes::MultiRegionMesh::Region>{
                Meshes::native_region("left", left), Meshes::native_region("right", right)},
            std::vector<Meshes::MultiRegionMesh::Interface>{Meshes::StructuredPatchInterface{{0,1}, {1,0}}});
        auto mesh = std::make_shared<Handle>(geometry);
        PlanarALEMeshMotion<> motion(mesh);
        motion.begin_trial(1.1, 0.2);
        const auto ale = FVM::make_ale_control_volume_state(*mesh, motion);
        if (comm->getRank() == 0) *left = Meshes::OrthogonalCartesian3D(left_edges);
        expect_collective_error(*comm, [&] { ale.validate(*mesh); },
            RejectionCategory::LogicError,
            "Static region constituent changed; rebuild MultiRegionMesh and its fields/operators.");
    }
    auto fresh_mesh = std::make_shared<Handle>(test::two_regions());
    PlanarALEMeshMotion<> fresh_motion(fresh_mesh);
    fresh_motion.begin_trial(1.1, 0.2);
    const auto fresh_ale = FVM::make_ale_control_volume_state(*fresh_mesh, fresh_motion);
    EXPECT_NO_THROW(fresh_ale.validate(*fresh_mesh));
    fresh_motion.rollback_trial();
}

TEST(MultiRegionALETest, RankLocalTraversalFailureReleasesLeaseAndFailsCollectively)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    auto mesh = std::make_shared<ThrowingHandle>(test::two_regions());
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.1, 0.2);
    const auto ale = FVM::make_ale_control_volume_state(*mesh, motion);

    mesh->fail_faces = comm->getRank() == 0;
    expect_collective_error(*comm, [&] { ale.validate(*mesh); },
        RejectionCategory::LogicError,
        "ALE control-volume state mesh traversal failed on at least one rank.");
    mesh->fail_faces = false;
    mesh->fail_owner = comm->getRank() == 0;
    expect_collective_error(*comm, [&] { ale.validate(*mesh); },
        RejectionCategory::LogicError,
        "ALE control-volume state mesh traversal failed on at least one rank.");
    mesh->fail_owner = false;
    EXPECT_NO_THROW(ale.validate(*mesh));
    EXPECT_NO_THROW(motion.rollback_trial());
}

TEST(MultiRegionALETest, RankDivergentFaultsRetainCollectiveErrorPriority)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    auto mesh = std::make_shared<ThrowingHandle>(test::two_regions());
    MutableTestMotion motion(mesh);
    const auto ale = FVM::make_ale_control_volume_state(*mesh, motion);
    const auto face = static_cast<size_t>(mesh->faces(Handle::local_ordinal_type{0}).front());
    const auto expect_error = [&](RejectionCategory category, const char* expected)
    { expect_collective_error(*comm, [&] { ale.validate(*mesh); }, category, expected); };

    motion.set_active(comm->getRank() != 0);
    if (comm->getRank() == 1) motion.set_new_epoch(ale.new_geometry_epoch() + 1);
    expect_error(RejectionCategory::LogicError,
        "ALE control-volume state requires its originating motion trial to remain active.");
    motion.set_active(true);
    motion.set_new_epoch(ale.new_geometry_epoch());

    mesh->fail_epoch = comm->getRank() == 0;
    if (comm->getRank() == 1) motion.set_new_epoch(ale.new_geometry_epoch() + 1);
    expect_error(RejectionCategory::LogicError,
        "Static region constituent changed; rebuild MultiRegionMesh and its fields/operators.");
    mesh->fail_epoch = false;
    motion.set_new_epoch(ale.new_geometry_epoch());

    motion.set_flux(face, comm->getRank() == 0 ? 1.0 : 0.0);
    const auto gcl_error = std::string("ALE control-volume state violates the cellwise geometric conservation law; maximum residual is ")
        + std::to_string(1.0) + " m^3/s.";
    expect_error(RejectionCategory::InvalidArgument, gcl_error.c_str());
    motion.set_flux(face, 0.0);

    motion.set_flux(face, comm->getRank() == 0 ? std::numeric_limits<real_t>::quiet_NaN() : 1.0);
    expect_error(RejectionCategory::InvalidArgument,
        "ALE control-volume state contains invalid volume, mesh-flux, or GCL data.");
    motion.set_flux(face, comm->getRank() == 1 ? 1.0 : 0.0);

    mesh->fail_owner = comm->getRank() == 0;
    expect_error(RejectionCategory::LogicError,
        "ALE control-volume state mesh traversal failed on at least one rank.");
    mesh->fail_owner = false;
    motion.set_flux(face, 0.0);
    auto fresh_mesh = std::make_shared<ThrowingHandle>(test::two_regions());
    MutableTestMotion fresh_motion(fresh_mesh);
    const auto fresh_ale = FVM::make_ale_control_volume_state(*fresh_mesh, fresh_motion);
    EXPECT_NO_THROW(fresh_ale.validate(*fresh_mesh));
}
TEST(MultiRegionALETest, ValidatedPlanarProofWalksGclOnceAndPreservesMovedFromOwner)
{
    auto mesh = std::make_shared<ThrowingHandle>(test::two_regions());
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.1, .2);
    mesh->face_traversals = 0;
    auto owned = FVM::make_validated_planar_ale_control_volume_state(*mesh, motion);
    EXPECT_EQ(mesh->face_traversals, mesh->num_owned_cells());
    mesh->face_traversals = 0;
    for (int consumer = 0; consumer < 4; ++consumer) EXPECT_NO_THROW(owned.validate(*mesh, .2));
    EXPECT_EQ(mesh->face_traversals, 0U);
    {
        auto moved = std::move(owned);
        EXPECT_NO_THROW(moved.validate(*mesh));
    }
    // The destination has been destroyed: source spans must still own their
    // immutable storage, and the reusable certificate must remain attached.
    EXPECT_NO_THROW(owned.validate(*mesh));
    EXPECT_EQ(mesh->face_traversals, 0U);
    const auto borrowed = FVM::make_ale_control_volume_state(*mesh, motion);
    EXPECT_EQ(mesh->face_traversals, mesh->num_owned_cells());
    EXPECT_NO_THROW(borrowed.validate(*mesh));
    EXPECT_EQ(mesh->face_traversals, 2 * mesh->num_owned_cells());
    motion.rollback_trial();
}

TEST(MultiRegionALETest, ValidatedPlanarMixedEligibilityUsesCollectiveFullFallback)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    auto mesh = std::make_shared<ThrowingHandle>(test::two_regions());
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.1, .2);
    const auto borrowed = FVM::make_ale_control_volume_state(*mesh, motion);
    const auto owned = FVM::make_validated_planar_ale_control_volume_state(*mesh, motion);
    const auto& mixed = comm->getRank() == 0 ? owned : borrowed;
    mesh->face_traversals = 0;
    EXPECT_NO_THROW(mixed.validate(*mesh));
    EXPECT_EQ(mesh->face_traversals, mesh->num_owned_cells());
    const auto face = static_cast<size_t>(mesh->faces(Handle::local_ordinal_type{0}).front());
    const double original = motion.face_mesh_fluxes()[face];
    if (comm->getRank() == 1) const_cast<double*>(motion.face_mesh_fluxes().data())[face] += 1.;
    const std::string message = "ALE control-volume state violates the cellwise geometric conservation law; maximum residual is "
        + std::to_string(1.) + " m^3/s.";
    expect_collective_error(*comm, [&] { mixed.validate(*mesh); }, RejectionCategory::InvalidArgument, message.c_str());
    if (comm->getRank() == 1) const_cast<double*>(motion.face_mesh_fluxes().data())[face] = original;
    EXPECT_NO_THROW(mixed.validate(*mesh));
    motion.rollback_trial();
}

TEST(MultiRegionALETest, ValidatedPlanarProofRetainsRankLocalMetadataChecks)
{
    const auto comm = Tpetra::getDefaultComm();
    if (comm->getSize() != 2) GTEST_SKIP() << "Requires two MPI ranks.";
    auto mesh = std::make_shared<ThrowingHandle>(test::two_regions());
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.1, .2);
    const auto owned = FVM::make_validated_planar_ale_control_volume_state(*mesh, motion);
    mesh->fail_epoch = comm->getRank() == 0;
    expect_collective_error(*comm, [&] { owned.validate(*mesh); }, RejectionCategory::LogicError,
        "Static region constituent changed; rebuild MultiRegionMesh and its fields/operators.");
    mesh->fail_epoch = false;
    expect_collective_error(*comm, [&] { owned.validate(*mesh, comm->getRank() == 0 ? .1 : .2); },
        RejectionCategory::InvalidArgument, "ALE transport timestep must exactly match the active mesh-motion trial.");
    EXPECT_NO_THROW(owned.validate(*mesh, .2));
    motion.rollback_trial();
    expect_collective_error(*comm, [&] { owned.validate(*mesh); }, RejectionCategory::LogicError,
        "ALE control-volume state requires its originating motion trial to remain active.");
}

TEST(MultiRegionALETest, ValidatedPlanarSnapshotsIncludeEmptyRanksAndRejectBadConstruction)
{
    auto geometry = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{Meshes::cartesian_region("single", {{{0,1}, {0,1}, {0,1}}})},
        std::vector<Meshes::MultiRegionMesh::Interface>{});
    Handle::DistributionOptions distribution;
    distribution.allow_empty_partitions = true;
    auto mesh = std::make_shared<Handle>(geometry, distribution);
    const auto comm = mesh->owned_cell_map()->getComm();
    const int local_empty = mesh->num_owned_cells() == 0;
    int empty_ranks = 0;
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_empty, &empty_ranks);
    EXPECT_EQ(empty_ranks, comm->getSize() - 1);
    PlanarALEMeshMotion<> motion(mesh);
    motion.begin_trial(1.1, .2);
    const auto owned = FVM::make_validated_planar_ale_control_volume_state(*mesh, motion);
    EXPECT_NO_THROW(owned.validate(*mesh));
    size_t face = 0;
    double original = 0;
    if (!local_empty)
    {
        face = static_cast<size_t>(mesh->faces(Handle::local_ordinal_type{0}).front());
        original = motion.face_mesh_fluxes()[face];
        const_cast<double*>(motion.face_mesh_fluxes().data())[face] += 1.;
    }
    const std::string message = "ALE control-volume state violates the cellwise geometric conservation law; maximum residual is "
        + std::to_string(1.) + " m^3/s.";
    expect_collective_error(*comm, [&] { (void)FVM::make_validated_planar_ale_control_volume_state(*mesh, motion); },
        RejectionCategory::InvalidArgument, message.c_str());
    EXPECT_NO_THROW(owned.validate(*mesh));
    if (!local_empty) const_cast<double*>(motion.face_mesh_fluxes().data())[face] = original;
    motion.rollback_trial();
    expect_collective_error(*comm, [&] { owned.validate(*mesh); }, RejectionCategory::LogicError,
        "ALE control-volume state requires its originating motion trial to remain active.");
}

TEST(MultiRegionALETest, ChangingThePeriodicLengthIsRejected)
{
    using namespace Meshes;
    StructuredPatchInterface periodic{{0,4},{0,5}}; periodic.periodic_translation=MeshUtils::Vec3{0,0,1};
    auto geometry=std::make_shared<MultiRegionMesh>(std::vector<MultiRegionMesh::Region>{
        cartesian_region("periodic",{{{0,1},{0,1},{0,0.5,1}}})},std::vector<MultiRegionMesh::Interface>{periodic});
    auto mesh=std::make_shared<Handle>(geometry);
    EXPECT_THROW((PlanarALEMeshMotion<>(mesh)),std::invalid_argument);
}
