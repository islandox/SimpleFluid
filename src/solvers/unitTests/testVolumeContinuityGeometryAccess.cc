/** @file testVolumeContinuityGeometryAccess.cc
 * @brief Preview traversal parity, scoped geometry access and collective failure.
 */
#include "solvers/VolumeContinuityModel.hh"
#include "solvers/VolumeContinuityModel.tcc"
#include "FVM/ALEControlVolumeState.tcc"
#include "geometry/mesh/MultiRegionMesh.hh"
#include "geometry/PlanarALEMeshMotion.hh"
#include "utils/testing_environment.hh"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

namespace
{
using namespace SimpleFluid;
using Pack = DefaultTpetraTypes;
using Handle = MeshHandle<Pack>;
using Composite = Meshes::MultiRegionMesh;
using Cartesian = Meshes::OrthogonalCartesian3D;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

class ObservedMesh : public Handle
{
public:
    using Handle::Handle;
    using NativeExecution = decltype(std::declval<const Handle&>().acquire_execution_view());
    mutable int acquisitions = 0, active = 0, visits = 0;
    bool observe = false, use_lease = true, fail_acquisition = false, fail_traversal = false;
    struct Execution
    {
        const ObservedMesh& mesh;
        std::unique_ptr<NativeExecution> native;
        explicit Execution(const ObservedMesh& source) : mesh(source)
        {
            if (mesh.use_lease)
                native.reset(new NativeExecution(mesh.Handle::acquire_execution_view()));
            if (mesh.observe)
            {
                ++mesh.acquisitions;
                if (mesh.fail_acquisition) throw std::logic_error("Injected continuity acquisition failure.");
                ++mesh.active;
            }
        }
        ~Execution() { if (mesh.observe) --mesh.active; }
        Execution(const Execution&) = delete;
        Execution& operator=(const Execution&) = delete;
    };
    Execution acquire_execution_view() const { return Execution(*this); }
    auto faces(local_ordinal_type cell) const
    {
        if (observe)
        {
            ++visits;
            if (active != 1) throw std::logic_error("Continuity traversal escaped its execution scope.");
            if (fail_traversal) throw std::out_of_range("Injected continuity traversal failure.");
        }
        return Handle::faces(cell);
    }
};
using Model = VolumeContinuityModel<Pack, ObservedMesh>;
using Flux = ScalarFaceFieldStored<Pack, ObservedMesh>;

struct Fixture
{
    std::shared_ptr<Cartesian> child;
    std::shared_ptr<Composite> geometry;
    std::shared_ptr<ObservedMesh> mesh;
    std::unique_ptr<PlanarALEMeshMotion<Pack>> motion;
    explicit Fixture(bool one_cell = false)
    {
        child = std::make_shared<Cartesian>(Vec3D<ArrReal>{{{0, 1}, {0, 1}, one_cell ? ArrReal{0, 1} : ArrReal{0, .5, 1}}});
        geometry = std::make_shared<Composite>(std::vector<Composite::Region>{Meshes::native_region("column", child)},
            std::vector<Composite::Interface>{});
        Handle::DistributionOptions options;
        options.allow_empty_partitions = true;
        mesh = std::make_shared<ObservedMesh>(geometry, options);
        motion = std::make_unique<PlanarALEMeshMotion<Pack>>(mesh);
        motion->begin_trial(1.25, .5);
    }
    void released() const
    {
        EXPECT_EQ(mesh->active, 0);
        EXPECT_NO_THROW(geometry->require_geometry_writable());
        EXPECT_NO_THROW(child->require_geometry_writable());
    }
};

void compare(const Model::Trial& a, const Model::Trial& b)
{
    ASSERT_EQ(a.material_source().size(), b.material_source().size());
    for (size_t i = 0; i < a.material_source().size(); ++i)
    {
        EXPECT_EQ(a.material_source()[i], b.material_source()[i]);
        EXPECT_EQ(a.bubble_slip_contribution()[i], b.bubble_slip_contribution()[i]);
        EXPECT_EQ(a.target().integrated_rate(i), b.target().integrated_rate(i));
    }
    const auto& x = a.diagnostics(); const auto& y = b.diagnostics();
    EXPECT_EQ(x.old_material_volume, y.old_material_volume);
    EXPECT_EQ(x.new_material_volume, y.new_material_volume);
    EXPECT_EQ(x.global_material_source, y.global_material_source);
    EXPECT_EQ(x.global_carrier_transport, y.global_carrier_transport);
    EXPECT_EQ(x.global_bubble_slip_divergence, y.global_bubble_slip_divergence);
    EXPECT_EQ(x.source_pool_closure_residual, y.source_pool_closure_residual);
    EXPECT_EQ(x.normalized_source_pool_closure_residual, y.normalized_source_pool_closure_residual);
    EXPECT_EQ(x.maximum_target_change, y.maximum_target_change);
}
} // namespace

TEST(VolumeContinuityAccessTest, ScopedTraversalMatchesUnleasedFieldsAndReleasesBeforeReplay)
{
    Fixture f;
    auto ale = FVM::make_validated_planar_ale_control_volume_state(*f.mesh, *f.motion);
    Flux carrier(f.mesh, 0.0, "carrier"), exact(f.mesh, 0.0, "exact"), slip(f.mesh, 0.0, "slip");
    for (const auto face : carrier.owned_face_ids())
        if (f.mesh->is_interior_face(face))
        {
            carrier.set_owned_value(face, .125);
            exact.set_owned_value(face, .125);
            slip.set_owned_value(face, .0625);
        }
    carrier.sync_ghosts(); exact.sync_ghosts(); slip.sync_ghosts();
    std::vector<double> old(ale.old_cell_volumes().begin(), ale.old_cell_volumes().end());
    std::vector<double> next(ale.new_cell_volumes().begin(), ale.new_cell_volumes().end());
    std::vector<double> fraction(next.size(), 1.0), previous(f.mesh->num_owned_cells(), 0.0);
    Model model(f.mesh);
    for (bool exact_flux : {false, true})
    {
        const Model::Inputs inputs{.ale=ale, .old_material_volume=old, .new_material_volume=next,
            .carrier_material_fraction=exact_flux ? std::span<const double>(fraction) : std::span<const double>{},
            .carrier_relative_flux=carrier, .carrier_material_volume_flux=exact_flux ? &exact : nullptr,
            .bubble_slip_volume_flux=&slip, .old_pool_volume=1.0, .new_pool_volume=1.25,
            .previous_target=previous};
        f.mesh->use_lease=false;
        const auto reference=model.preview(inputs, 1);
        f.mesh->use_lease=true; f.mesh->observe=true;
        f.mesh->acquisitions=0; f.mesh->visits=0;
        const auto candidate=model.preview(inputs, 1);
        EXPECT_EQ(f.mesh->acquisitions, 1);
        EXPECT_EQ(f.mesh->visits, f.mesh->num_owned_cells());
        f.released();
        compare(reference, candidate);
        f.mesh->observe=false;
    }
    f.motion->rollback_trial();
    f.motion->begin_trial(1.25, .5);
    EXPECT_THROW(ale.validate(*f.mesh), std::logic_error);
    ale=FVM::make_validated_planar_ale_control_volume_state(*f.mesh, *f.motion);
    EXPECT_NO_THROW((void)model.preview({.ale=ale, .old_material_volume=old, .new_material_volume=next,
        .carrier_relative_flux=carrier, .old_pool_volume=1.0, .new_pool_volume=1.25}, 2));
    f.motion->rollback_trial();
}

TEST(VolumeContinuityAccessTest, RankLocalFaultsReleaseLeaseBeforeCollectiveFailure)
{
    Fixture f;
    const auto comm=f.mesh->owned_cell_map()->getComm();
    const auto ale=FVM::make_validated_planar_ale_control_volume_state(*f.mesh,*f.motion);
    Flux carrier(f.mesh,0.0,"carrier");
    std::vector<double> old(ale.old_cell_volumes().begin(),ale.old_cell_volumes().end());
    std::vector<double> next(ale.new_cell_volumes().begin(),ale.new_cell_volumes().end());
    Model model(f.mesh);
    for (bool acquisition : {true,false})
    {
        f.mesh->observe=true;
        f.mesh->fail_acquisition=acquisition && comm->getRank()==0;
        f.mesh->fail_traversal=!acquisition && comm->getRank()==0;
        int rejected=0;
        try { (void)model.preview({.ale=ale,.old_material_volume=old,.new_material_volume=next,
            .carrier_relative_flux=carrier,.old_pool_volume=1.0,.new_pool_volume=1.25},1); }
        catch (const std::exception& error)
        {
            rejected=1;
            EXPECT_EQ(std::string(error.what()),comm->getRank()==0
                ? (acquisition ? "Injected continuity acquisition failure." : "Injected continuity traversal failure.")
                : "VolumeContinuityModel geometry traversal failed on another rank.");
        }
        int all=0;
        Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&rejected,&all);
        EXPECT_EQ(all,comm->getSize());
        f.released();
        f.mesh->observe=false; f.mesh->fail_acquisition=false; f.mesh->fail_traversal=false;
    }
    EXPECT_NO_THROW((void)model.preview({.ale=ale,.old_material_volume=old,.new_material_volume=next,
        .carrier_relative_flux=carrier,.old_pool_volume=1.0,.new_pool_volume=1.25},2));
    f.motion->rollback_trial();
}

TEST(VolumeContinuityAccessTest, EmptyRankParticipatesAndReleasesItsLease)
{
    Fixture f(true);
    const auto ale=FVM::make_validated_planar_ale_control_volume_state(*f.mesh,*f.motion);
    Flux carrier(f.mesh,0.0,"carrier");
    std::vector<double> old(ale.old_cell_volumes().begin(),ale.old_cell_volumes().end());
    std::vector<double> next(ale.new_cell_volumes().begin(),ale.new_cell_volumes().end());
    Model model(f.mesh);
    f.mesh->observe=true; f.mesh->acquisitions=0;
    const auto trial=model.preview({.ale=ale,.old_material_volume=old,.new_material_volume=next,
        .carrier_relative_flux=carrier,.old_pool_volume=1.0,.new_pool_volume=1.25},1);
    EXPECT_EQ(f.mesh->acquisitions,1);
    EXPECT_EQ(trial.diagnostics().global_material_source,.5);
    const auto comm=f.mesh->owned_cell_map()->getComm();
    int empty=f.mesh->num_owned_cells()==0, empties=0;
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM,1,&empty,&empties);
    EXPECT_EQ(empties,comm->getSize()-1);
    f.released(); f.mesh->observe=false;
    f.motion->rollback_trial();
}
