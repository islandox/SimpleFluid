/**
 * @file FVM/unitTests/testFvmLibrary.cc
 * @brief Link guard for the FVM shared library's exported API.
 *
 * This target links `SimpleFluid::FVM` alone. It takes the addresses of
 * FVM-exported symbols so the executable link fails if `libSimpleFluidFVM`
 * ever again needs a symbol that only `libSimpleFluidSolvers` defines. GNU ld
 * reports undefined shared-library symbols when it creates an executable, so
 * the regression fails on Linux too; macOS `ld64` rejects the dylib outright.
 * FVM-side export coverage is enforced separately by
 * `simplefluid_fvm_elf_export_boundary`.
 */

#include <gtest/gtest.h>

#include "FVM/ALEControlVolumeState.hh"
#include "FVM/NonOrthogonalCorrection.hh"
#include "FVM/TransportSystem.hh"
#include "geometry/Mesh.hh"
#include "utils/testing_environment.hh"

namespace
{
using Pack = SimpleFluid::DefaultTpetraTypes;
testing::Environment* const environment =
    testing::AddGlobalTestEnvironment(new utils_test::KokkosEnvironment);

TEST(FvmLibraryTest, ExportedFvmApiLinksWithoutSolverDso)
{
    using MeshType = SimpleFluid::Mesh<Pack>;
    using ValidateMember = void (SimpleFluid::FVM::ALEControlVolumeState::*)(const MeshType&) const;
    ValidateMember validate = &SimpleFluid::FVM::ALEControlVolumeState::validate<MeshType>;
    auto* residual = &SimpleFluid::FVM::full_diffusion_residual<Pack>;
    EXPECT_TRUE(validate != nullptr);
    EXPECT_TRUE(residual != nullptr);

    // Keep a relocation to the named-request provider even under Release LTO.
    // Equation DSOs call this overload directly; positional wrappers alone can
    // inline it and therefore do not prove that it is exported.
    using WeightedAssembler = SimpleFluid::FVM::TransportSystem<Pack> (*)(
        SimpleFluid::FVM::WeightedScalarTransportRequest<Pack>);
    WeightedAssembler volatile weighted = &SimpleFluid::FVM::weighted_scalar_transport_system<Pack>;
    EXPECT_TRUE(weighted != nullptr);

    using namespace SimpleFluid;
    using namespace SimpleFluid::FVM;
    using Scalar = Pack::scalar_type;
    using LocalOrdinal = Pack::local_ordinal_type;
    using ScalarField = CellField<Pack>;
    using FluxField = FaceField<Pack>;
    using GeometryCache = TransportGeometryCache<Mesh<Pack>>;
    using Sink = std::function<Scalar(LocalOrdinal)>;
    using FixedValue = std::function<std::optional<Scalar>(LocalOrdinal)>;
    using HistoricalAssembler = TransportSystem<Pack> (*)(const ScalarField&, const FluxField&, Scalar,
        const ScalarField&, const ScalarField&, const ScalarField&, ScalarBoundaryConditionProvider<Pack>,
        ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>, NonOrthogonalTreatment, const ScalarField*,
        Teuchos::RCP<Pack::matrix_type>, Sink, FixedValue, const SimpleFluid::BoundaryCache<Pack>*, const GeometryCache*,
        FaceCoefficientInterpolation);
    using AdvancedAssembler = TransportSystem<Pack> (*)(const ScalarField&, const FluxField&, Scalar,
        const ScalarField&, const ScalarField&, const ScalarField&, ScalarBoundaryConditionProvider<Pack>,
        ScalarBoundaryValueProvider<Pack>, ScalarCellValueProvider<Pack>, ScalarTransportDiscretization,
        NonOrthogonalTreatment, const ScalarField*, const ScalarField*, Teuchos::RCP<Pack::matrix_type>, Sink,
        FixedValue, const SimpleFluid::BoundaryCache<Pack>*, const GeometryCache*, FaceCoefficientInterpolation);
    HistoricalAssembler volatile historical = &weighted_scalar_transport_system<Pack>;
    AdvancedAssembler volatile advanced = &weighted_scalar_transport_system<Pack>;
    EXPECT_TRUE(historical != nullptr);
    EXPECT_TRUE(advanced != nullptr);
}
} // namespace
