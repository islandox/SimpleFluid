/**
 * @file ConservativeInterface.hh
 * @author islandox
 * @brief Explicit conservative planar coarse/fine interface declarations.
 * @version 0.1
 * @date 2026-09-10
 * @copyright Copyright (c) 2026
 */
#pragma once
#include "geometry/mesh/MeshRegion.hh"
#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
namespace SimpleFluid::Meshes
{
struct FineNativePatch
{
    int boundary = 0;
    uint64_t face_count = 0;
    /** Full-source native membership retained by sparse descriptors. */
    std::vector<uint64_t> faces;
};
struct FineInterfacePatch
{
    size_t region = 0;
    std::vector<FineNativePatch> patches;
    FineInterfacePatch() = default;
    FineInterfacePatch(size_t r, int boundary, uint64_t count = 0) : region(r), patches{{boundary, count, {}}} {}
};
struct MultiProviderFaceMapping
{
    uint64_t coarse_face = 0;
    std::vector<RegionFace> fine_faces;
};
/** Same-fluid nested whole-face tiles spanning several fine providers.
 * Every declared fine patch is used completely and exactly once. One provider
 * appears once per descriptor and may declare several distinct complete native patches. Crossed nonnested traces must
 * close in native topology before this interface is built; no mortar faces are synthesized. Constructed participants
 * use authoritative exact planar-face certificates.
 */
struct MultiProviderNonconformingInterface
{
    size_t coarse_region = 0;
    int coarse_boundary = 0;
    std::vector<FineInterfacePatch> fine_patches;
    std::vector<MultiProviderFaceMapping> faces;
    bool partial_coarse_patch = false;
    uint64_t coarse_face_count = 0;
};
/** @brief One exterior coarse face tiled exactly by the declared exterior fine faces. */
struct CoarseFineFaceMapping
{
    uint64_t coarse_face = 0;
    std::vector<uint64_t> fine_faces;
};
/**
 * @brief Planar coarse/fine patches with canonical fine-face fluxes.
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
    /**
     * Permit a subset of the coarse boundary patch, tiled by the complete fine
     * patch. Every selected coarse face still requires exact, unique coverage.
     * This supports several child blocks covering disjoint coarse-face sets.
     */
    bool partial_coarse_patch = false;
};
/** @brief A native integrated flux contributes coefficient*Q to this canonical logical face. */
struct FaceFluxWeight
{
    uint64_t face;
    real_t coefficient;
};
/** @brief Temporary validation geometry, released after interface construction. */
struct InterfaceFacePolygon
{
    MeshUtils::Vec3 centroid;
    MeshUtils::Vec3 area_vector;
    real_t area;
    std::vector<MeshUtils::Vec3> vertices;
};
/** @brief Reject nonplanar, nonconvex, overlapping, uncovered or inconsistently measured tiles. */
void validate_planar_subdivision(const InterfaceFacePolygon& coarse, std::span<const InterfaceFacePolygon> fine,
    real_t absolute_length, real_t relative);
} // namespace SimpleFluid::Meshes
