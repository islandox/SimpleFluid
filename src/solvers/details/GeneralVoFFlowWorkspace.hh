/** @file GeneralVoFFlowWorkspace.hh
 * @brief Physical native-mesh operators for the general polyhedral VoF backend.
 */
#pragma once
#include "solvers/VoFSolver.hh"
#include "solvers/details/GeneralVoFPhiClosure.hh"
#include "solvers/details/GeneralVoFProjectionArithmetic.hh"
#include "solvers/details/VoFProjectionWorkspace.hh"
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace SimpleFluid::detail
{
/** Immutable native geometry with physical-coordinate conditioned stencils.
 * Every operation is collective. It owns numerical workspaces, not accepted state.
 */
struct GeneralVoFProjectionProvenance
{
    std::size_t byte_budget = 64 * 1024 * 1024;
    std::vector<GeneralVoFRoundedValue> face_velocity;   ///< Owned canonical order.
    std::vector<Meshes::ExactGeometryMeasure> face_rate; ///< Certified zero-divergence rate, when topology supplied.
};
class SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFFlowWorkspace
{
public:
    using Vector = VoF::Vector;
    explicit GeneralVoFFlowWorkspace(SP<const MeshHandle<>> mesh);
    [[nodiscard]] std::vector<double> cells(const std::vector<double>& owned) const;
    [[nodiscard]] std::vector<Vector> vectors(const std::vector<Vector>& owned) const;
    [[nodiscard]] std::vector<double> faces(const std::vector<double>& overlap) const;
    /** Physical least squares; globally unsampled coordinate directions have zero
     * derivative. Constant fields are exact even on isolated boundary cells;
     * unresolved nonconstant sampled directions are rejected collectively. */
    [[nodiscard]] std::vector<Vector> gradient(const std::vector<double>& owned) const;
    /** Divergence of unit interface normals using the gradient support contract. */
    [[nodiscard]] std::vector<double> curvature(const std::vector<Vector>& owned_normals) const;
    [[nodiscard]] std::vector<double> interpolate(const std::vector<Vector>& owned_velocity) const;
    [[nodiscard]] std::vector<Vector> viscous_force(
        const std::vector<Vector>& velocity, const std::vector<double>& viscosity) const;
    /** Mutates caller-private candidates only; physical divergence accepts the solve. */
    void project(std::vector<double>& face_velocity, std::vector<Vector>& velocity, std::vector<double>& pressure,
        const std::vector<double>& density, const std::vector<double>& fraction, const std::vector<double>& curvature,
        double dt, double surface_tension, const VoFSolverOptions& options, VoFStepDiagnostics& diagnostics,
        GeneralVoFProjectionProvenance* arithmetic = nullptr,
        std::optional<std::span<const Meshes::ExactGeometryMeasure>> physical_mass = std::nullopt,
        const GeneralVoFProjectionClosureTopology* closure_topology = nullptr);
    [[nodiscard]] double stable_timestep(const std::vector<double>& face_velocity, double minimum_density,
        double maximum_viscosity, double surface_tension, const VoFSolverOptions& options) const;
    /** Rank-local cached geometry view; unlike collective numerical operators,
     * this accessor may be called within a rank-local owned-cell loop. */
    [[nodiscard]] const std::vector<Vector>& physical_spans() const
    {
        if (!current_geometry())
            throw std::runtime_error("General VoF cached physical spans have stale geometry or partition.");
        return spans_;
    }

private:
    using Pack = DefaultTpetraTypes;
    using LO = Pack::local_ordinal_type;
    struct Face
    {
        LO owner = -1, neighbor = -1;
        bool boundary = true;
        double area = 0, left = 0, right = 0, distance = 0;
        Vector normal{}, tangent_left{}, tangent_right{};
    };
    SP<const MeshHandle<>> mesh_;
    std::vector<Face> geometry_;
    std::vector<double> volumes_;
    std::vector<Meshes::ExactGeometryMeasure> exact_volumes_;
    std::vector<Vector> spans_;
    struct GradientSample
    {
        LO neighbor = -1;
        Vector displacement{};
        double weight = 0;
    };
    struct GradientStencil
    {
        std::vector<GradientSample> samples;
        std::array<unsigned, 3> pivot{};
        Vector divisor{};
        std::array<Vector, 3> factor{};
        bool valid = true;
    };
    std::vector<GradientStencil> gradient_stencils_;
    std::array<bool, 3> absent_gradient_axis_{};
    std::uint64_t geometry_epoch_ = 0;
    MeshHandle<>::MultiRegionPtr native_owner_;
    void prepare_gradient_geometry();
    [[nodiscard]] bool current_geometry() const noexcept;
    Pack::import_type cell_import_, face_import_;
    mutable Pack::vector_type cell_owned_, cell_overlap_, face_owned_, face_overlap_;
    VoFProjectionWorkspace projection_;
    Pack::global_ordinal_type gauge_;
    void require(bool valid, const char* message) const;
    [[nodiscard]] double maximum(double value) const;
    [[nodiscard]] std::vector<double> divergence(const std::vector<double>& velocity) const;
};
} // namespace SimpleFluid::detail
