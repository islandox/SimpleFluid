/** @file GeneralVoFSurface.hh
 * @brief Source-pinned geometry-only capillary patches and filter diagnostics.
 */
#pragma once
#include "SimpleFluidExport.hh"
#include "geometry/PLIC/CertifiedGeometryProperties.hh"
#include <compare>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace SimpleFluid::VoF
{
enum class GeneralVoFSurfaceKind
{
    MixedCell,
    AlignedFace,
    FilteredCell
};
struct GeneralVoFSurfaceKey
{
    GeneralVoFSurfaceKind kind = GeneralVoFSurfaceKind::MixedCell;
    std::uint64_t identity = 0, ordinal = 0;
    auto operator<=>(const GeneralVoFSurfaceKey&) const = default;
};
enum class GeneralVoFCapillaryModel
{
    Reconstructed,
    FilteredBinary
};
struct GeneralVoFSurfaceOptions
{
    GeneralVoFCapillaryModel model = GeneralVoFCapillaryModel::Reconstructed;
    double filter_width = 2; ///< Width in local cell lengths; physical width shrinks with refinement.
};
enum class GeneralVoFSurfaceResolutionReason
{
    IncompleteSupport,
    AmbiguousTopology,
    ComponentMerging,
    UnsupportedCavity,
    VolumeConstraint,
    DeficientFit
};
class SIMPLEFLUID_PUBLIC_TYPE GeneralVoFSurfaceResolutionError : public std::runtime_error
{
public:
    GeneralVoFSurfaceResolutionError(GeneralVoFSurfaceResolutionReason reason, const std::string& message)
        : std::runtime_error(message), reason_(reason)
    {
    }
    [[nodiscard]] GeneralVoFSurfaceResolutionReason reason() const noexcept { return reason_; }

private:
    GeneralVoFSurfaceResolutionReason reason_;
};
struct GeneralVoFSurfacePatch
{
    GeneralVoFSurfaceKey key;
    std::uint64_t first_cell = 0, second_cell = std::numeric_limits<std::uint64_t>::max(), component = 0;
    int owner = -1;
    PLIC::GeometryPoint centroid{}, normal{}, physical_spans{};
    double area = 0, cell_scale = 0;
    std::vector<PLIC::GeometryPoint> vertices;
};
struct GeneralVoFSurfaceComponent
{
    std::uint64_t identity = 0;
    int owner = -1;
    long double original_volume = 0, smoothed_volume = 0, extracted_volume = 0;
    double filter_width_min = 0, filter_width_max = 0, isovalue = 0;
    bool closed = false;
};
struct GeneralVoFSurfaceCatalog
{
    std::uint64_t source_control = 0;
    std::size_t allocation_bytes = 0; ///< Cumulative admitted catalog construction storage.
    GeneralVoFCapillaryModel model = GeneralVoFCapillaryModel::Reconstructed;
    std::vector<GeneralVoFSurfacePatch> patches;
    std::map<GeneralVoFSurfaceKey, std::vector<GeneralVoFSurfaceKey>> adjacency;
    std::vector<GeneralVoFSurfaceComponent> components;
    std::vector<std::uint64_t> filtered_source_cells; ///< Original geometric gas cells replaced by filtered patches.
};
struct GeneralVoFSurfaceFitDiagnostic
{
    GeneralVoFSurfaceKey key;
    double curvature = 0, condition = 0, normalized_residual = 0;
    std::size_t samples = 0;
};
struct GeneralVoFCapillaryDiagnostics
{
    GeneralVoFCapillaryModel model = GeneralVoFCapillaryModel::Reconstructed;
    std::vector<GeneralVoFSurfaceFitDiagnostic> fits;   ///< Canonical owned patches only.
    std::vector<GeneralVoFSurfaceComponent> components; ///< Canonical owned components only.
};
} // namespace SimpleFluid::VoF
