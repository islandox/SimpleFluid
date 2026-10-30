/** @file GeneralVoFCompatibleProjection.hh
 * @brief Private physical-mass compatible face/cell saddle-point projection.
 */
#pragma once
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"
#include <array>
#include <memory>
#include <span>
#include <vector>

namespace SimpleFluid::detail
{
/** Uses canonical integrated interior-face rates and the actual native
 * centroid reconstruction. Boundary rates and the smallest-GID pressure
 * impulse are fixed to zero. No regularization of the physical mass form is
 * introduced. All operations are collective, including on empty owners.
 *
 * The byte budget bounds explicit payload and conservative matrix/Krylov
 * workspace estimates; opaque Tpetra/Belos allocator overhead is excluded.
 */
class SIMPLEFLUID_SOLVERS_LOCAL GeneralVoFCompatibleProjection
{
public:
    using Vector = std::array<double, 3>;
    using WideVector = std::array<long double, 3>;
    struct Result
    {
        std::vector<long double> face_rate, pressure_impulse;
        std::vector<WideVector> velocity;
        LinearSolveSummary linear;
        std::size_t storage_bytes = 0;
    };
    struct Residual
    {
        /** Corrections in the original unscaled face and positive Bq=b conventions. */
        std::vector<long double> face_rhs, cell_rhs;
        double max_momentum = 0, max_constraint = 0;
    };
    GeneralVoFCompatibleProjection(SP<const MeshHandle<>> mesh, std::span<const long double> physical_mass,
        std::size_t byte_budget = 64 * 1024 * 1024);
    ~GeneralVoFCompatibleProjection();
    GeneralVoFCompatibleProjection(const GeneralVoFCompatibleProjection&) = delete;
    GeneralVoFCompatibleProjection& operator=(const GeneralVoFCompatibleProjection&) = delete;
    [[nodiscard]] Result solve(std::span<const Vector> predictor, std::span<const long double> cell_target,
        std::span<const long double> owned_face_force, double dt, const LinearSolverOptions& options);
    /** Fresh zero-start correction solve. The caller owns wide accumulation and
     * must deduct returned iterations from its original total iteration budget. */
    [[nodiscard]] Result solve_algebraic(std::span<const long double> face_rhs, std::span<const long double> cell_rhs,
        const LinearSolverOptions& options);
    /** Re-evaluates the physical mass operator with wide accumulation and the
     * original native geometry, independently of narrowed matrix coefficients. */
    [[nodiscard]] Residual residual(std::span<const long double> face_rate,
        std::span<const long double> pressure_impulse, std::span<const Vector> predictor,
        std::span<const long double> cell_target, std::span<const long double> owned_face_force, double dt) const;
    [[nodiscard]] std::vector<WideVector> reconstruct(std::span<const long double> face_rate) const;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace SimpleFluid::detail
