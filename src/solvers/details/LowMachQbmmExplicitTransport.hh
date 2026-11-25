/** @file LowMachQbmmExplicitTransport.hh
 * @brief Private arithmetic and receipts shared by the explicit QBMM transports.
 */
#pragma once

#include "equations/details/LowMachScaledFaceSum.hh"
#include "solvers/LowMachMixtureSolver.hh"

#include <initializer_list>
#include <limits>

namespace SimpleFluid::Reacting::qbmm_explicit_detail
{
using Flow = LowMachMixtureSolver;
using Mesh = Flow::Mesh;
using LO = Flow::Pack::local_ordinal_type;
using Vector = Flow::Vector;
using FaceSum = qbmm_face_detail::ScaledFaceSum;
using MultiVector = Flow::Pack::multi_vector_type;
using Map = Flow::Pack::map_type;
using Import = Flow::Pack::import_type;

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
    double maximum_stage_residual = 0, maximum_step_residual = 0;
    double local_caloric_transport_magnitude = 0, local_energy_transport_magnitude = 0;
};

inline constexpr double eps = std::numeric_limits<double>::epsilon();
inline constexpr size_t sides = 2, dimensions = 3;

enum PacketColumn : size_t
{
    H0,
    H1,
    H2,
    H3,
    U,
    W,
    FirstSpecies
};

enum MomentumColumn : size_t
{
    Mass,
    CarrierMomentum,
    DriftMomentum = CarrierMomentum + dimensions,
    MomentumColumns = DriftMomentum + dimensions
};

class SIMPLEFLUID_SOLVERS_LOCAL Overlap
{
    const Import& importer;
    MultiVector owned;
    MultiVector ghosted;

public:
    Overlap(const Import& importer_, const Teuchos::RCP<const Map>& owned_map,
        const Teuchos::RCP<const Map>& overlap_map, size_t columns)
        : importer(importer_), owned(owned_map, columns), ghosted(overlap_map, columns)
    {
    }

    auto write() { return owned.getLocalViewHost(Tpetra::Access::OverwriteAll); }
    auto read() const { return ghosted.getLocalViewHost(Tpetra::Access::ReadOnly); }
    void synchronize() { ghosted.doImport(owned, importer, Tpetra::REPLACE); }
};

struct SIMPLEFLUID_SOLVERS_LOCAL Product
{
    double fraction = 1;
    int exponent = 0;

    void multiply(double value);
    void divide(double positive);
    [[nodiscard]] FaceSum sum() const;
    void add_weighted(FaceSum& result, const FaceSum& value) const;
};

struct SIMPLEFLUID_SOLVERS_LOCAL FaceInterval
{
    std::array<std::vector<FaceSum>, sides> direction;
    std::array<std::array<FaceSum, MomentumColumns>, sides> momentum;
    FaceSum theta, slip;
};

struct SIMPLEFLUID_SOLVERS_LOCAL CellInterval
{
    FaceSum work_volume, gross_volume;
};

struct SIMPLEFLUID_SOLVERS_LOCAL StageThermal
{
    double material_mass = 0, density = 0, volume = 0, caloric_specific = 0;
};

/** The one-ring span must reproduce offsets on transporting sides of a face. */
class SIMPLEFLUID_SOLVERS_LOCAL FaceOffsetFit
{
    struct Sample
    {
        Mesh::Vec3 displacement, basis;
        double distance;
    };
    std::vector<Sample> samples;

public:
    explicit FaceOffsetFit(std::span<const Mesh::Vec3> displacements);
    bool weights(const Mesh::Vec3& offset, std::vector<double>& result) const;
};

/** Validate the immutable XY topology and walls of a polygonal disk or annulus. */
SIMPLEFLUID_SOLVERS_LOCAL void validate_cylindrical_base(const Mesh::SemiStructured& mesh);
/** Check the live metrics and incidence used by axial kinetic transport and pressure projection. */
SIMPLEFLUID_SOLVERS_LOCAL void validate_axial_kinetic_geometry(const Mesh& mesh);

SIMPLEFLUID_SOLVERS_LOCAL void require(bool valid, const char* message);
SIMPLEFLUID_SOLVERS_LOCAL Product product(
    std::initializer_list<double> numerator, std::initializer_list<double> denominator = {});
SIMPLEFLUID_SOLVERS_LOCAL double finite_rate(double value, const char* message);
SIMPLEFLUID_SOLVERS_LOCAL double relative_difference(const FaceSum& difference, const FaceSum& magnitude);
SIMPLEFLUID_SOLVERS_LOCAL void add_magnitude(FaceSum& sum, const FaceSum& operand);
SIMPLEFLUID_SOLVERS_LOCAL double represented(const FaceSum& amount, double divisor = 1.);
SIMPLEFLUID_SOLVERS_LOCAL double represented(const Product& amount);
SIMPLEFLUID_SOLVERS_LOCAL double rounded_rate(const FaceSum& amount, double duration = 1.);
SIMPLEFLUID_SOLVERS_LOCAL StageThermal thermal_trace(PhaseThermalState& phase, bool recover_arrival);
SIMPLEFLUID_SOLVERS_LOCAL double exact_interval_substep(double remaining, double limit);

SIMPLEFLUID_SOLVERS_LOCAL Flow::QbmmFaceFlux face_receipt(const FaceInterval& interval, double dt, double carrier_flux);
SIMPLEFLUID_SOLVERS_LOCAL void validate_closed_balances(const Mesh& mesh, const Flow::Options& options,
    const Flow::State& accepted, const Flow::State& candidate, const std::vector<size_t>& phases, size_t packet_columns,
    const std::vector<std::vector<double>>& accepted_volumes, const TransportResult& result);
} // namespace SimpleFluid::Reacting::qbmm_explicit_detail

namespace SimpleFluid::Reacting::qbmm_detail
{
SIMPLEFLUID_SOLVERS_LOCAL inline bool uses_axial_kinetic_transport(
    const LowMachMixtureSolver::Options& options, const LowMachMixtureSolver::Mesh& mesh)
{
    using Mesh = LowMachMixtureSolver::Mesh;
    return uses_qbmm_kinetic_transport(options.qbmm_model) &&
           (std::holds_alternative<Mesh::CylindricalPtr>(mesh.variant()) ||
               std::holds_alternative<Mesh::SemiStructuredPtr>(mesh.variant()));
}

SIMPLEFLUID_SOLVERS_LOCAL inline void require_axial(const LowMachMixtureSolver::Vector& vector, const char* message)
{
    if (vector.x != 0. || vector.y != 0. || !std::isfinite(vector.z))
        throw std::invalid_argument(message);
}
} // namespace SimpleFluid::Reacting::qbmm_detail
