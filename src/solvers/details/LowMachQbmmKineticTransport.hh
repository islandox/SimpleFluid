/** @file LowMachQbmmKineticTransport.hh
 * @brief Synchronized fixed-grid kinetic advection of the native QBMM families.
 */
#pragma once

#include "equations/details/LowMachScaledFaceSum.hh"
#include "solvers/LowMachMixtureSolver.hh"

namespace SimpleFluid::Reacting::qbmm_kinetic_detail
{
using Flow = LowMachMixtureSolver;

struct SIMPLEFLUID_SOLVERS_LOCAL FaceIntegrals
{
    qbmm_face_detail::ScaledFaceSum forward_mass, backward_mass;  ///< Nonnegative kg per direction.
    std::array<qbmm_face_detail::ScaledFaceSum, 3> drift;         ///< Signed kg m/s in owner orientation.
    qbmm_face_detail::ScaledFaceSum theta, slip, material_volume; ///< Integrated physical pressure operands.
};

struct SIMPLEFLUID_SOLVERS_LOCAL TransportResult
{
    std::vector<std::vector<Flow::QbmmFaceFlux>> owned_flux;    ///< Physical phase, owned face.
    std::vector<std::vector<double>> advected_material_volumes; ///< Owned cell, physical phase.
    std::vector<FaceIntegrals> integrated_faces;                ///< Owned-face order, all kinetic families.
    std::vector<Flow::QbmmKineticLawReceipt> law_receipts;
    std::vector<double> substep_durations;
    int substeps = 0;
    double maximum_cfl = 0;
    double maximum_species_residual = 0, maximum_population_residual = 0;
    double maximum_shape_residual = 0, maximum_energy_residual = 0;
    double local_caloric_transport_magnitude = 0, local_energy_transport_magnitude = 0;
};

/** Updates only the bound families in the private corrector candidate. All
 * ranks take the same substeps. Carrier flux and the thermodynamic guess are
 * borrowed for this corrector and stay fixed throughout the interval.
 */
[[nodiscard]] SIMPLEFLUID_SOLVERS_LOCAL TransportResult transport(const SP<const Flow::Mesh>& mesh,
    const Flow::Options& options, const Flow::State& accepted, const Flow::State& thermodynamic_guess,
    Flow::State& candidate, const Flow::Traits::scalar_face_type& carrier_flux,
    const std::vector<std::vector<double>>& accepted_material_volumes, double dt, size_t corrector);
} // namespace SimpleFluid::Reacting::qbmm_kinetic_detail
