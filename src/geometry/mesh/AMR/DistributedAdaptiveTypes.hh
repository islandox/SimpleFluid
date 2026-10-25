/** @file DistributedAdaptiveTypes.hh
 * @brief Lossless, rank-local contracts for distributed immutable adaptation.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include "geometry/mesh/AMR/AdaptiveHierarchy.hh"
#include "geometry/mesh/NativeFragmentRecipe.hh"
#include <limits>
#include <optional>
#include <span>
#include <string>

namespace SimpleFluid::Meshes::AMR
{
using AdaptiveBytes = std::vector<char>;
/** Canonical typed encoding. Hashes route messages; complete bytes prove identity. */
[[nodiscard]] AdaptiveBytes encode_adaptive_key(const AdaptiveCellKey&);
[[nodiscard]] AdaptiveCellKey decode_adaptive_key(std::span<const char>);
[[nodiscard]] std::uint64_t adaptive_route_hash(std::span<const char>) noexcept;

struct AdaptiveWork
{
    double reconstruction = 1, interface = 0, population = 0, migration = 0;
    [[nodiscard]] double total() const; ///< Recurring compute only; migration is separate.
};
/** An exact oriented surface occurrence. The key identifies a physical support
 * construction, never an ordinal in a repartitioned mesh. */
struct AdaptiveSurface
{
    AdaptiveBytes key;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> geometry; ///< Already cell-outward; never multiply by orientation.
    int orientation = 1;         ///< Sign relative to persistent support orientation.
    bool exterior = false;
    std::string boundary;
    std::uint64_t native_face = std::numeric_limits<std::uint64_t>::max();
    AdaptiveBytes authority_image;
    bool require_native_whole_tile = false;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary>
        native_whole_geometry; ///< Finite physical-to-canonical periodic displacement.
};
enum class AdaptiveProviderKind : std::uint64_t
{
    Cartesian,
    Product,
    Polyhedral,
    Static
};
enum class AdaptiveSplit : std::uint64_t
{
    Isotropic,
    Base,
    Axial
};
struct AdaptiveMark
{
    AdaptiveCellKey key;
    AdaptiveSplit split = AdaptiveSplit::Isotropic;
};
/** A native procedural descriptor is a lossless local recipe, not an expanded
 * mesh. It must contain only this leaf's needed construction closure. */
struct AdaptiveFactorRecipe
{
    std::shared_ptr<const AdaptiveBytes> root, base, axis;
};
struct AdaptiveLeaf
{
    AdaptiveCellKey key;
    AdaptiveBytes domain; ///< Connected-domain/join provenance, independent of material/region.
    std::uint64_t material = 0, revision = 0;
    AdaptiveProviderKind provider = AdaptiveProviderKind::Polyhedral;
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry;
    std::vector<AdaptiveSurface> surfaces;
    std::shared_ptr<const NativeFragmentDescriptor> native_recipe;
    std::uint64_t native_cell = 0;
    std::shared_ptr<const AdaptiveBytes> join_recipe;          ///< Needed immutable original-root join capsule.
    std::shared_ptr<const AdaptiveFactorRecipe> factor_recipe; ///< Pruned provider-local factor ancestry/DAG.
    AdaptiveBytes parent_family;
    AdaptiveWork work;
};
struct AdaptiveFamily
{
    AdaptiveBytes key;
    AdaptiveLeaf parent;
    std::vector<AdaptiveCellKey> children;
    AdaptiveBytes anchor; ///< Actual active descendant; capsule follows its owner.
    AdaptiveSplit split = AdaptiveSplit::Isotropic;
    std::uint64_t revision = 0;
};
struct AdaptiveTileIncident
{
    AdaptiveCellKey cell;
    int owner = -1, orientation = 1;
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> face;
};
struct AdaptiveTile
{
    AdaptiveBytes domain, surface, identity;
    std::uint64_t face_gid = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> node_gids;
    std::vector<std::uint64_t> second_node_gids; ///< Same loop order, in the second cell physical image.
    AdaptiveBytes second_image_identity;         ///< Finite neighbor-physical to first-physical displacement.
    PLIC::GeometryPoint second_image{};          ///< Checked display of second_image_identity for FVM geometry.
    PLIC::PlanarFacePtr<ArithmeticMode::Ordinary> geometry;
    AdaptiveCellKey first;
    std::optional<AdaptiveCellKey> second;
    int first_owner = -1, second_owner = -1;
    std::uint64_t first_native_face = std::numeric_limits<std::uint64_t>::max(),
                  second_native_face = std::numeric_limits<std::uint64_t>::max();
    bool exterior = false;
    std::string boundary;
};
struct AdaptiveSourceCell
{
    AdaptiveCellKey key;
    std::uint64_t gid = 0;
};
struct AdaptiveNativeCellBinding
{
    AdaptiveCellKey key;
    std::uint64_t gid = 0, material = 0, native_cell = 0;
    int owner = -1;
    AdaptiveProviderKind provider = AdaptiveProviderKind::Polyhedral;
    bool full = false;
    double volume = 0;
    PLIC::GeometryPoint centroid{};
    std::shared_ptr<const NativeFragmentDescriptor> native_recipe;
};
struct AdaptiveNode
{
    std::uint64_t gid = 0;
    AdaptiveBytes identity;
    PLIC::GeometryPoint coordinate{};
    AdaptiveBytes coordinate_identity; ///< Finite physical coordinate identity, independent of image aliases.
};
/** Complete authority-certified boundary tiles; full owned cells, reconstruction
 * ghosts and metric-only neighbors are separate. Remote cuts remain interior. */
struct AdaptiveNativeDraft
{
    std::vector<AdaptiveLeaf> owned, ghosts;
    std::vector<AdaptiveTile> tiles;
    std::vector<AdaptiveNode> nodes;
    std::vector<AdaptiveNativeCellBinding> cell_bindings;
    std::uint64_t global_cells = 0;
    std::uint64_t global_faces = 0, global_nodes = 0;
    std::vector<AdaptiveBytes> family_references;
};
[[nodiscard]] AdaptiveBytes encode_adaptive_leaf(const AdaptiveLeaf&);
[[nodiscard]] AdaptiveLeaf decode_adaptive_leaf(std::span<const char>);
[[nodiscard]] AdaptiveBytes encode_adaptive_family(const AdaptiveFamily&);
[[nodiscard]] AdaptiveFamily decode_adaptive_family(std::span<const char>);
} // namespace SimpleFluid::Meshes::AMR
