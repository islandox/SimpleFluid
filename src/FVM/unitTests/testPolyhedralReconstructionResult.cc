/** @file testPolyhedralReconstructionResult.cc */
#include "FVM/details/GeneralVoF/Wire.hh"
#include "geometry/PLIC/PolyhedralInterface.hh"
#include "geometry/PLIC/PolyhedralReconstruction.hh"
#include "plic_result_wire_fixture.hh"
#include <gtest/gtest.h>
#include <iomanip>
#include <sstream>
#include <type_traits>
namespace
{
using namespace SimpleFluid;
using namespace SimpleFluid::PLIC;
namespace wire = SimpleFluid::VoF::general_vof_detail;
std::string hex(const std::vector<char>& bytes)
{
    std::ostringstream out;
    for (unsigned char value : bytes)
        out << std::hex << std::setw(2) << std::setfill('0') << unsigned(value);
    return out.str();
}
std::vector<DynamicPolyhedralReconstructionResult> records()
{
    PolyhedralReconstructionResult<ArithmeticMode::Ordinary> ordinary;
    ordinary.geometry.gas = {{1, 0, 0}, {2, 3, 4}, .375};
    ordinary.target_volume_difference = {-0x1.000000000000001p-8L};
    ordinary.target_volume_numerical_error_estimate = 0x1p-60L;
    ordinary.stencil_volume_residual = .125;
    ordinary.stencil_volume_residual_measured = true;
    ordinary.model_quality = PlicModelQuality::WithinThreshold;
    ordinary.orientation_method = PolyhedralOrientationMethod::Gradient;
    ordinary.stencil_condition = 3.25;
    PolyhedralReconstructionResult<ArithmeticMode::BoundedCertification> bounded;
    copy_reconstruction_metadata(bounded, ordinary);
    bounded.target_volume_difference = {.25L, 0x1p-55L};
    bounded.target_volume_numerical_error_estimate = 0;
    bounded.orientation_method = PolyhedralOrientationMethod::AngularVolumeFit;
    bounded.fit_converged = true;
    bounded.fit_iterations = 7;
    bounded.orientation_step = .002;
    PolyhedralReconstructionResult<ArithmeticMode::Exact> exact;
    copy_reconstruction_metadata(exact, bounded);
    exact.target_volume_difference = {-.125L, 0x1p-70L, true};
    exact.target_volume_residual = -.125;
    PolyhedralReconstructionResult<ArithmeticMode::Ordinary> pure_gas;
    pure_gas.geometry.pure_gas = true;
    pure_gas.target_volume_residual = .125;
    pure_gas.target_volume_difference = {-.125L};
    PolyhedralReconstructionResult<ArithmeticMode::Exact> pure_carrier;
    pure_carrier.geometry.pure_carrier = true;
    pure_carrier.target_volume_residual = .125;
    pure_carrier.target_volume_difference = {-.125L, 0, true};
    return {
        to_dynamic(ordinary), to_dynamic(bounded), to_dynamic(exact), to_dynamic(pure_gas), to_dynamic(pure_carrier)};
}
template<class T>
concept HasFitMetadata = requires(T value) { value.fit_iterations; };
template<class T>
concept HasModelMetadata = requires(T value) { value.model_policy; };
static_assert(!HasFitMetadata<PolyhedralInterface> && !HasModelMetadata<PolyhedralInterface>);
static_assert(std::is_trivially_copyable_v<PolyhedralInterface>);
static_assert(
    decltype(PolyhedralReconstructionResult<ArithmeticMode::Ordinary>{}.target_volume_difference)::arithmetic_mode ==
    ArithmeticMode::Ordinary);
static_assert(!std::is_convertible_v<PolyhedralReconstructionResult<ArithmeticMode::Ordinary>,
    PolyhedralReconstructionResult<ArithmeticMode::Exact>>);
static_assert(std::variant_size_v<DynamicPolyhedralReconstructionResult> == 3);
TEST(PolyhedralReconstructionResultTest, GeometryIsIndependentOfReconstructionMetadata)
{
    PolyhedralInterface shape{{{1, 0, 0}, {2, 3, 4}, .375}, false, false};
    PolyhedralReconstructionResult<ArithmeticMode::Ordinary> result;
    result.geometry = shape;
    result.fit_iterations = 123;
    EXPECT_EQ(result.geometry.gas.normal, shape.gas.normal);
    EXPECT_EQ(result.geometry.gas.offset, .375);
    EXPECT_FALSE(result.geometry.pure_gas);
}
TEST(PolyhedralReconstructionResultTest, ProductionWireMatchesIndependentOriginalPackets)
{
    const auto cases = records();
    for (std::size_t i = 0; i < cases.size(); ++i)
    {
        SCOPED_TRACE(i);
        wire::wire::Writer writer;
        wire::plane(writer, cases[i]);
        EXPECT_EQ(hex(writer.data), reconstruction_result_test::general_packets[i]);
        wire::wire::Reader reader{writer.data};
        const auto result = wire::plane(reader);
        EXPECT_NO_THROW(reader.done());
        const auto& geometry = reconstruction_geometry(result);
        const auto& original = reconstruction_geometry(cases[i]);
        EXPECT_EQ(geometry.gas.normal, original.gas.normal);
        EXPECT_EQ(geometry.gas.anchor, original.gas.anchor);
        EXPECT_EQ(geometry.gas.offset, original.gas.offset);
        EXPECT_EQ(geometry.pure_gas, original.pure_gas);
        EXPECT_EQ(geometry.pure_carrier, original.pure_carrier);
        EXPECT_EQ(reconstruction_mode(result), reconstruction_mode(cases[i]));
        wire::wire::Writer repeated;
        wire::plane(repeated, result);
        EXPECT_EQ(repeated.data, writer.data);
    }
}
} // namespace
