/**
 * @file ConservativeInterface.hh
 * @author islandox
 * @brief Explicit conservative planar coarse/fine interface declarations.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#pragma once
#include "geometry/MeshUtils.hh"
#include <cstdint>
#include <span>
#include <vector>
namespace SimpleFluid::Meshes
{
/** @brief One exterior coarse face tiled exactly by the declared exterior fine faces. */
struct CoarseFineFaceMapping
{
    uint64_t coarse_face = 0;
    std::vector<uint64_t> fine_faces;
};
/**
 * @brief Complete planar coarse/fine patches with canonical fine-face fluxes.
 *
 * Each fine face becomes an ordinary interior FVM face. The coarse cell visits
 * all covering subfaces. Convex, planar, gap-free and nonoverlapping coverage is
 * required; arbitrary mortar interpolation and curved subdivision are excluded.
 */
struct NonconformingInterface
{
    size_t coarse_region = 0, fine_region = 0;
    int coarse_boundary = 0, fine_boundary = 0;
    std::vector<CoarseFineFaceMapping> faces;
};
/** @brief A native integrated flux contributes coefficient*Q to this canonical logical face. */
struct FaceFluxWeight { uint64_t face; real_t coefficient; };
/** @brief Temporary validation geometry, released after interface construction. */
struct InterfaceFacePolygon
{
    MeshUtils::Vec3 centroid;
    MeshUtils::Vec3 area_vector;
    real_t area;
    std::vector<MeshUtils::Vec3> vertices;
};
/** @brief Reject nonplanar, nonconvex, overlapping, uncovered or inconsistently measured tiles. */
void validate_planar_subdivision(const InterfaceFacePolygon& coarse,
    std::span<const InterfaceFacePolygon> fine, real_t absolute_length, real_t relative);
} // namespace SimpleFluid::Meshes
