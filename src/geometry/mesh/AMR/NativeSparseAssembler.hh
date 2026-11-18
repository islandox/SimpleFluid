#pragma once
#include "geometry/mesh/AMR/NativePublicationExchange.hh"
#include "geometry/mesh/AMR/NativePublicationNumbering.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class SIMPLEFLUID_LOCAL NativeSparseAssembler final
{
public:
    [[nodiscard]] static AdaptiveNativePublicationView publish(
        const DistributedAdaptiveHierarchy&, const AdaptiveNativeSchemaView&, std::span<const AdaptiveCellKey>);
    [[nodiscard]] static AdaptiveNativePublicationView compatibility(
        const DistributedAdaptiveHierarchy&, std::span<const char>, std::span<const AdaptiveCellKey>);
    [[nodiscard]] static std::shared_ptr<NativePublicationNumbering> number(
        NativePublicationSession&, std::span<const NativeRemoteSource>);
    [[nodiscard]] static std::shared_ptr<NativePublicationViewData> assemble(NativePublicationSession&,
        const std::shared_ptr<NativePublicationNumbering>&, std::span<const NativeRemoteSource>,
        std::span<const NativeRemoteIncidence>, std::span<const NativeRemoteSource>);
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
