#pragma once

#include "SimpleFluidExport.hh"
#include "dataclass/typedefs.hh"

#include <span>
#include <vector>

namespace SimpleFluid
{
struct GrayDiffuseEnclosureResult
{
    std::vector<real_t> radiosity_delta, irradiation_delta;
    std::vector<real_t> radiosity, irradiation;
    std::vector<real_t> outward_heat_rate; ///< W; each pair contributes opposite heat rates.
    real_t equation_residual = 0.0; ///< Maximum absolute shifted radiosity residual, W/m2.
};

/** Closed diffuse-gray enclosure with transparent, nonparticipating gas.
 * Exchange areas C_ij=A_i F_ij are row-major, symmetric, nonnegative and
 * sum to A_i in every row. Emissivity is uniform within each patch.
 * Geometry and the LAPACK factorization are immutable after construction.
 */
class SIMPLEFLUID_EQUATIONS_EXPORT GrayDiffuseEnclosure
{
public:
    GrayDiffuseEnclosure(std::vector<real_t> areas, std::vector<real_t> exchange_areas,
        std::vector<real_t> emissivities);

    /** Input is sigma*(area-weighted mean T^4 - T_reference^4), W/m2.
     * The common reference is sigma*T_reference^4. Shifted outputs preserve
     * near-equilibrium differences; heat rates use only shifted radiosities.
     */
    GrayDiffuseEnclosureResult evaluate(std::span<const real_t> blackbody_emissive_power_delta_Wm2,
        real_t reference_blackbody = 0.0) const;

    static real_t blackbody_delta(real_t temperature, real_t reference_temperature);
    real_t reciprocal_condition_number() const noexcept { return reciprocal_condition_; }
    std::size_t size() const noexcept { return areas_.size(); }

private:
    std::vector<real_t> areas_, exchange_, emissivities_, matrix_, lu_;
    std::vector<int> pivots_;
    real_t reciprocal_condition_ = 0.0;
};
} // namespace SimpleFluid
