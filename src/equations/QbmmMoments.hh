/** @file QbmmMoments.hh
 * @brief Local extensive mass-coordinate moments and derived common-pressure bubble geometry.
 */
#pragma once

#include "utils/export_helper.hh"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

namespace SimpleFluid::Reacting
{
class QbmmPopulationSchema;

struct QbmmShapeState
{
    double H2 = 0, H3 = 0;
    bool operator==(const QbmmShapeState&) const = default;
};

/** Temporary mathematical block in order H0..H3. Only QbmmShapeState adds canonical storage. */
using QbmmMomentBlock = std::array<double, 4>;

struct QbmmCellView
{
    std::string_view population;
    std::uint64_t catalog_fingerprint;
    std::span<const double> species_moles; ///< Exact named dispersed compartment, catalog order.
    const double& number;                  ///< Engaged extensive population number.
    const QbmmShapeState& shape;
    std::uint64_t schema_fingerprint; ///< Stamp of the family owning the shape, including its coordinate scale.
};

struct QbmmAtom
{
    double count = 0, mass_over_scale = 0;
};

enum class InversionStatus
{
    Absent,
    RankOne,
    TwoNodes,
    InvalidInput,
    InconsistentRank,
    NonpositiveSupport,
    IllConditioned,
    ReconstructionFailure
};

/** Stable names used by diagnostics for accepted and rejected moment blocks. */
[[nodiscard]] constexpr const char* qbmm_inversion_status_name(InversionStatus status) noexcept
{
    switch (status)
    {
        case InversionStatus::Absent:
            return "Absent";
        case InversionStatus::RankOne:
            return "RankOne";
        case InversionStatus::TwoNodes:
            return "TwoNodes";
        case InversionStatus::InvalidInput:
            return "InvalidInput";
        case InversionStatus::InconsistentRank:
            return "InconsistentRank";
        case InversionStatus::NonpositiveSupport:
            return "NonpositiveSupport";
        case InversionStatus::IllConditioned:
            return "IllConditioned";
        case InversionStatus::ReconstructionFailure:
            return "ReconstructionFailure";
    }
    return "Unknown";
}

/** All thresholds act on relative quantities. No inventory floor is applied. */
inline constexpr double qbmm_rank_tolerance = 32 * std::numeric_limits<double>::epsilon();
inline constexpr double qbmm_reconstruction_tolerance = 128 * std::numeric_limits<double>::epsilon();

struct MomentResiduals
{
    /// Scaled moment units and fraction of the input. NaN means the residual is outside the double range.
    std::array<double, 4> absolute{}, relative{};
};

struct ConditioningReport
{
    int coordinate_exponent = 0; ///< Internal binary rescaling of the mass_over_scale coordinate.
    /// (ad-bc)/(ad+bc) in each Hankel matrix. NaN marks an unrepresentable margin.
    double hankel_margin = 0, shifted_hankel_margin = 0;
    /// Jacobi recurrence in the caller's mass_over_scale coordinate. NaN marks an unrepresentable coefficient.
    double a0 = 0, b1 = 0, a1 = 0;
};

struct QbmmQuadrature
{
    std::array<QbmmAtom, 2> atoms{}; ///< Active prefix is sorted by increasing mass.
    unsigned active_atoms = 0;
    InversionStatus status = InversionStatus::InvalidInput;
    MomentResiduals reconstruction;
    ConditioningReport conditioning;
    [[nodiscard]] bool valid() const noexcept
    {
        return status == InversionStatus::Absent || status == InversionStatus::RankOne ||
               status == InversionStatus::TwoNodes;
    }
};

/** Sum positive extensive atoms. Empty input gives exact vacuum. Invalid or unrepresentable stocks throw. */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmMomentBlock qbmm_atom_moments(std::span<const QbmmAtom> atoms);

/** Check family/catalog, composition and stock presence, and derive H1 from the material owner.
 * Use invert_qbmm_moments to classify rank and realizability.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmMomentBlock qbmm_cell_moments(
    const QbmmPopulationSchema& schema, QbmmCellView cell);

/** Symmetric two-node Wheeler inversion. Failure returns a reason and zero usable atoms. */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmQuadrature invert_qbmm_moments(QbmmMomentBlock moments) noexcept;

struct QbmmGeometryView
{
    unsigned active_atoms = 0;
    double number_density = 0, mass_kg = 0, area_density = 0; ///< count/m3 (bulk), kg/cell, m2/m3 (bulk).
    std::array<double, 2> mass_kg_per_bubble{}, radius_m{}, diameter_m{};
    std::array<double, 4> diameter_moment_density{};       ///< Sum(N*d^k)/Vcell, k=0..3.
    std::optional<double> d32, min_diameter, max_diameter; ///< Metres. Undefined for absence.
};

/** Evaluated view. The supplied specific volume is 1/rho at the host's common p,T,Y.
 * Absence needs no EOS evaluation. There is no independent volume state or capillary pressure here.
 */
[[nodiscard]] SIMPLEFLUID_EQUATIONS_EXPORT QbmmGeometryView qbmm_geometry(const QbmmPopulationSchema& schema,
    QbmmCellView cell, double cell_volume_m3, std::optional<double> phase_specific_volume_m3_per_kg);
} // namespace SimpleFluid::Reacting
