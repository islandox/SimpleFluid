/** @file FVM/TransportSystemFactory.hh
 * @brief Typed transport assembly for legacy and native mesh fields.
 */
#pragma once

#include "FVM/TransportAssemblyPacks.hh"
#include "FVM/TransportSystem.hh"

namespace SimpleFluid::FVM
{
/** Stateless assembly. Packs are consumed synchronously and never retained.
 * Include TransportSystemFactory.tcc to provide custom mesh/type instantiations.
 */
template<TpetraTypePack Pack, class MeshType = Mesh<Pack>> class TransportSystemFactory
{
public:
    using ScalarTerms = FVM::ScalarTerms<Pack, MeshType>;
    using VectorTerms = FVM::VectorTerms<Pack, MeshType>;
    using WeightedScalarTerms = FVM::WeightedScalarTerms<Pack, MeshType>;
    using TemperatureTerms = FVM::TemperatureTerms<Pack, MeshType>;
    using MomentumTerms = FVM::MomentumTerms<Pack, MeshType>;
    using ScalarBoundary = ScalarTransportBoundary<Pack>;
    using BasicVectorBoundary = BasicVectorTransportBoundary<Pack>;
    using VelocityBoundary = VelocityTransportBoundary<Pack>;
    using WeightedScalarBoundary = CoefficientTransportBoundary<Pack, MeshType>;
    using TemperatureBoundary = CoefficientTransportBoundary<Pack, MeshType>;
    using MomentumBoundary = MomentumTransportBoundary<Pack, MeshType>;
    using ScalarOptions = ScalarAssemblyOptions<Pack, MeshType>;
    using BasicVectorOptions = BasicVectorAssemblyOptions;
    using VectorOptions = VectorAssemblyOptions<Pack, MeshType>;
    using WeightedScalarOptions = WeightedScalarAssemblyOptions<Pack, MeshType>;
    using TemperatureOptions = TemperatureAssemblyOptions<Pack, MeshType>;
    using MomentumOptions = MomentumAssemblyOptions<Pack, MeshType>;
    using ScalarContext = ScalarAssemblyContext<Pack, MeshType>;
    using BasicVectorContext = BasicVectorAssemblyContext<Pack>;
    using VectorContext = VectorAssemblyContext<Pack, MeshType>;
    using WeightedScalarContext = WeightedScalarAssemblyContext<Pack, MeshType>;
    using TemperatureContext = TemperatureAssemblyContext<Pack, MeshType>;
    using MomentumContext = MomentumAssemblyContext<Pack, MeshType>;

    SIMPLEFLUID_FVM_EXPORT static TransportSystem<Pack> assemble(
        ScalarTerms terms, ScalarBoundary boundary, ScalarOptions options, ScalarContext context = {});
    SIMPLEFLUID_FVM_EXPORT static VectorTransportSystem<Pack> assemble(
        VectorTerms terms, BasicVectorBoundary boundary, BasicVectorOptions options, BasicVectorContext context = {})
        requires std::is_same_v<MeshType, Mesh<Pack>>;
    SIMPLEFLUID_FVM_EXPORT static VectorTransportSystem<Pack> assemble(
        VectorTerms terms, VelocityBoundary boundary, VectorOptions options, VectorContext context = {});
    SIMPLEFLUID_FVM_EXPORT static TransportSystem<Pack> assemble(WeightedScalarTerms terms,
        WeightedScalarBoundary boundary, WeightedScalarOptions options, WeightedScalarContext context = {});
    SIMPLEFLUID_FVM_EXPORT static TransportSystem<Pack> assemble(TemperatureTerms terms, TemperatureBoundary boundary,
        TemperatureOptions options, TemperatureContext context = {});
    SIMPLEFLUID_FVM_EXPORT static TransportSystem<Pack> refresh_temperature_rhs(
        TemperatureTerms terms, TemperatureBoundary boundary, TemperatureOptions options, TemperatureContext context);
    SIMPLEFLUID_FVM_EXPORT static VectorTransportSystem<Pack> assemble(
        MomentumTerms terms, MomentumBoundary boundary, MomentumOptions options, MomentumContext context = {});
};
} // namespace SimpleFluid::FVM
