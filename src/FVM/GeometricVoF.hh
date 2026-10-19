/**
 * @file GeometricVoF.hh
 * @brief Geometric interface transport on fixed uniform Cartesian grids.
 */

#pragma once

#include "SimpleFluidExport.hh"

#include <array>
#include <cstddef>
#include <vector>

namespace SimpleFluid::VoF
{

using Vector = std::array<double, 3>;
using FaceField = std::array<std::vector<double>, 3>;

/**
 * @brief Serial Cartesian grid with x-fastest cells and face arrays.
 *
 * Face coordinates in direction a have cells[a]+1 entries; other coordinates
 * have cells[b] entries. Periodic end faces are duplicated and must carry
 * identical values. Nonperiodic boundaries are impermeable slip walls.
 * A direction with one cell is allowed for lower-dimensional verification.
 * Coordinates and spacing are measured in metres.
 */
struct SIMPLEFLUID_FVM_EXPORT UniformGrid
{
    std::array<std::size_t, 3> cells{1, 1, 1};
    Vector origin{0.0, 0.0, 0.0};
    Vector spacing{1.0, 1.0, 1.0};
    std::array<bool, 3> periodic{false, false, false};

    /** @brief Reject zero sizes, invalid coordinates, and indexing overflow. */
    void validate() const;
    [[nodiscard]] std::size_t cell_count() const;
    [[nodiscard]] std::size_t face_count(std::size_t axis) const;
    [[nodiscard]] std::size_t cell_index(std::size_t i, std::size_t j, std::size_t k) const;
    [[nodiscard]] std::array<std::size_t, 3> cell_coordinates(std::size_t id) const;
    [[nodiscard]] std::size_t face_index(std::size_t axis, std::size_t i, std::size_t j, std::size_t k) const;
    [[nodiscard]] double cell_volume() const;
    [[nodiscard]] double face_area(std::size_t axis) const;
    [[nodiscard]] Vector cell_center(std::size_t id) const;
};

/**
 * @brief Cell-local PLIC half-space: gas occupies normal dot (x-center) <= offset.
 *
 * Normal is a physical unit vector directed from gas into liquid. Offset is in
 * metres. Pure cells use zero normal and offset -1 (empty) or +1 (full).
 */
struct InterfacePlane
{
    Vector normal{0.0, 0.0, 0.0};
    double offset = -1.0;
};

/** @brief Area and area centroid of a planar interface cutting a Cartesian box. */
struct PlaneSection
{
    Vector centroid{};
    Vector normal{};   ///< Unit normal of the supplied gas-side half-space.
    double area = 0.0; ///< Square metres; zero when no two-dimensional cut exists.
};

/** @brief Geometric transport result and the phase fluxes used in its update. */
struct AdvectionResult
{
    std::vector<double> fraction;
    FaceField gas_volume;
    FaceField liquid_volume;
    /** Maximum absolute dt*div(u), before transport. */
    double maximum_volume_divergence = 0.0;
};

/**
 * @brief PLIC reconstruction, height-function curvature, and split transport.
 *
 * Owns its grid description and has no MPI, ALE, or adaptive topology contract.
 * The advection sweep uses a frozen Weymouth--Yue compression indicator. For
 * divergence-free face velocities the returned signed integrated phase face
 * volumes explain the entire fraction change, without inventory clipping.
 */
class SIMPLEFLUID_FVM_EXPORT GeometricVoF
{
public:
    explicit GeometricVoF(UniformGrid grid);
    [[nodiscard]] const UniformGrid& grid() const noexcept;

    /** @brief Reconstruct one conservative plane per cell from gas fractions. */
    [[nodiscard]] std::vector<InterfacePlane> reconstruct(const std::vector<double>& fraction) const;

    /**
     * @brief Curvature in 1/m, positive for convex gas; pure cells return zero.
     *
     * Uses complete local height columns, with transverse filtering of second
     * derivatives in 3D, then a local quadratic graph fit to PLIC interface area
     * centroids. The fit uses physical tangent
     * coordinates, periodic images, area/distance weights, and rank-revealing QR.
     * Insufficient or ill-conditioned patches retain smoothed-normal divergence
     * as a last fallback. Curvature is computed from fractions;
     * no bubble shape or radius is supplied. Curved interfaces must be supplied
     * as cell-integrated fractions containing mixed cells. Binary center-point
     * voxelizations provide no mixed-cell curvature and are not a substitute.
     */
    [[nodiscard]] std::vector<double> curvature(const std::vector<double>& fraction) const;

    /**
     * @brief Advance gas fractions with signed positive-axis velocity in m/s.
     * @param fraction Gas volume fraction per cell, in [0,1].
     * @param velocity Positive-axis face velocity; duplicated periodic ends agree.
     * @param dt Nonnegative elapsed time in seconds.
     * @param first_axis First directional sweep, in [0,2]; cyclic order follows.
     * @return Fractions and signed integrated gas/liquid face volumes in m3.
     * @throws std::invalid_argument For inconsistent input or nonzero wall flux.
     * @throws std::runtime_error For excessive CFL, divergence, or lost bounds.
     *
     * Requires sum_a max(|u_low|,|u_high|)*dt/h_a <= 0.5 per cell and
     * max|dt*div(u)| <= 1e-10. Roundoff-sized bound excursions (<= 2e-12)
     * remain explicit rather than silently changing gas inventory.
     */
    [[nodiscard]] AdvectionResult advect(
        const std::vector<double>& fraction, const FaceField& velocity, double dt, std::size_t first_axis = 0) const;

    /**
     * @brief Exact box/half-space intersection volume for normal dot x <= offset.
     * @param lower Lower box coordinates in metres.
     * @param upper Upper box coordinates in metres, componentwise >= lower.
     * @param normal Plane normal, any finite magnitude including zero.
     * @param offset Plane offset in the same scale as normal dot x.
     * @return Gas-side volume in m3; zero normal selects empty/full by offset.
     */
    [[nodiscard]] static double box_plane_volume(
        const Vector& lower, const Vector& upper, const Vector& normal, double offset);

    /**
     * @brief Geometric cut for normal dot x = offset in a positive-volume box.
     * @return Physical area, centroid, and unit normal; a zero normal has no cut.
     */
    [[nodiscard]] static PlaneSection plane_section(
        const Vector& lower, const Vector& upper, const Vector& normal, double offset);

private:
    UniformGrid grid_;
};

} // namespace SimpleFluid::VoF
