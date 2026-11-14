/** @file FVM/unitTests/testFvmLibrary.cc
 * @brief Declaration-only link guard for every compiled transport factory overload.
 */
#include <gtest/gtest.h>

#include "FVM/ALEControlVolumeState.hh"
#include "FVM/NonOrthogonalCorrection.hh"
#include "FVM/TransportSystemFactory.hh"
#include "geometry/SolidSubdomain.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include "geometry/mesh/PartitionedMeshBase.hh"
#include "utils/testing_environment.hh"

namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::FVM;
using Pack = DefaultTpetraTypes;
testing::Environment* const environment = testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

// Volatile function pointers retain real provider relocations under LTO.
// Exact signatures distinguish every supported family from other assemble overloads.
template<class Factory> void scalar_links()
{
    using Function = TransportSystem<Pack> (*)(typename Factory::ScalarTerms, typename Factory::ScalarBoundary,
        typename Factory::ScalarOptions, typename Factory::ScalarContext);
    Function volatile assemble = &Factory::assemble;
    EXPECT_TRUE(assemble != nullptr);
}
template<class Factory> void temperature_links()
{
    using Function =
        TransportSystem<Pack> (*)(typename Factory::TemperatureTerms, typename Factory::TemperatureBoundary,
            typename Factory::TemperatureOptions, typename Factory::TemperatureContext);
    Function volatile assemble = &Factory::assemble;
    Function volatile refresh = &Factory::refresh_temperature_rhs;
    EXPECT_TRUE(assemble != nullptr);
    EXPECT_TRUE(refresh != nullptr);
}
template<class Factory> void flow_links()
{
    using Weighted =
        TransportSystem<Pack> (*)(typename Factory::WeightedScalarTerms, typename Factory::WeightedScalarBoundary,
            typename Factory::WeightedScalarOptions, typename Factory::WeightedScalarContext);
    using Vector = VectorTransportSystem<Pack> (*)(typename Factory::VectorTerms, typename Factory::VelocityBoundary,
        typename Factory::VectorOptions, typename Factory::VectorContext);
    using Momentum = VectorTransportSystem<Pack> (*)(typename Factory::MomentumTerms,
        typename Factory::MomentumBoundary, typename Factory::MomentumOptions, typename Factory::MomentumContext);
    Weighted volatile weighted = &Factory::assemble;
    Vector volatile vector = &Factory::assemble;
    Momentum volatile momentum = &Factory::assemble;
    EXPECT_TRUE(weighted != nullptr);
    EXPECT_TRUE(vector != nullptr);
    EXPECT_TRUE(momentum != nullptr);
}

TEST(FvmLibraryTest, ExportedFvmApiLinksWithoutSolverDso)
{
    using ValidateMember = void (ALEControlVolumeState::*)(const Mesh<Pack>&) const;
    ValidateMember volatile validate = &ALEControlVolumeState::validate<Mesh<Pack>>;
    auto* volatile residual = &full_diffusion_residual<Pack>;
    EXPECT_TRUE(validate != nullptr);
    EXPECT_TRUE(residual != nullptr);

    using Legacy = TransportSystemFactory<Pack, Mesh<Pack>>;
    using Native = TransportSystemFactory<Pack, MeshHandle<Pack>>;
    using Cartesian = TransportSystemFactory<Pack, Meshes::PartitionedMesh<Meshes::OrthogonalCartesian3D, Pack>>;
    using Solid = TransportSystemFactory<Pack, SolidSubdomain<Pack>>;
    scalar_links<Legacy>();
    scalar_links<Native>();
    scalar_links<Cartesian>();
    scalar_links<Solid>();
    temperature_links<Legacy>();
    temperature_links<Native>();
    temperature_links<Cartesian>();
    temperature_links<Solid>();
    flow_links<Legacy>();
    flow_links<Native>();
    flow_links<Cartesian>();

    using BasicVector = VectorTransportSystem<Pack> (*)(
        Legacy::VectorTerms, Legacy::BasicVectorBoundary, Legacy::BasicVectorOptions, Legacy::BasicVectorContext);
    BasicVector volatile basic_vector = &Legacy::assemble;
    EXPECT_TRUE(basic_vector != nullptr);
}
} // namespace
