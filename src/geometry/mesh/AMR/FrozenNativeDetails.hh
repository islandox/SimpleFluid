/** @file FrozenNativeDetails.hh
 * @brief Internal immutable accepted-copy records; never an exact geometry certificate.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAncestryDirectory.hh"
#include "geometry/mesh/AMR/FrozenNative.hh"
#include "geometry/mesh/ExplicitRegionProvider.hh"
#include "geometry/mesh/FrozenNativeMesh.hh"
#include <set>
namespace SimpleFluid::Meshes::AMR
{
struct FrozenPart
{
    std::uint64_t source = 0, node_prefix = 0;
    std::shared_ptr<const NativeFragmentDescriptor> procedural;
    RegionLayout layout;
    std::vector<ExplicitRegionProvider::Cell> cells;
    std::vector<ExplicitRegionProvider::Face> faces;
    std::vector<ExplicitRegionProvider::Node> nodes;
    std::map<int, std::string> boundaries;
};
struct FrozenFace
{
    std::uint64_t gid = 0, native = 0;
    std::shared_ptr<const FrozenPart> part;
    AdaptiveCellKey first;
    std::optional<AdaptiveCellKey> second;
    PLIC::GeometryPoint second_image{};
    int boundary = -1;
};
struct FrozenRow
{
    std::uint64_t native = 0;
    std::vector<std::pair<std::shared_ptr<const FrozenFace>, int>> faces;
};
struct FrozenNativeRoot::Data
{
    FrozenNativeRoot::SourceObservation observation;
    std::uint64_t uid = 0, global_cells = 0, global_faces = 0, global_nodes = 0;
    std::array<double, 2> frame{};
    AdaptiveBytes schema;
    std::map<int, std::string> boundaries;
};
struct FrozenMetric
{
    double volume = 0;
    PLIC::GeometryPoint center{};
};
struct FrozenNativeBundle::Data
{
    AdaptiveCellKey key;
    std::uint64_t source_uid = 0, revision = 0, material = 0, source_gid = 0, native = 0;
    std::shared_ptr<const FrozenPart> part;
    std::optional<FrozenMetric> metric; ///< Engaged only for outer metric-only owner replies.
    std::vector<FrozenRow> rows;
};
} // namespace SimpleFluid::Meshes::AMR
namespace SimpleFluid::Meshes
{
struct FrozenNativeMeshInput::Data
{
    std::shared_ptr<const ::SimpleFluid::Meshes::AMR::FrozenNativeRoot::Data> root;
    std::vector<std::shared_ptr<const ::SimpleFluid::Meshes::AMR::FrozenNativeBundle::Data>> full;
    std::map<::SimpleFluid::Meshes::AMR::AdaptiveCellKey, ::SimpleFluid::Meshes::AMR::AdaptiveOwner> bindings;
    std::set<::SimpleFluid::Meshes::AMR::AdaptiveCellKey> owned;
    std::map<::SimpleFluid::Meshes::AMR::AdaptiveCellKey,
        std::shared_ptr<const ::SimpleFluid::Meshes::AMR::FrozenNativeBundle::Data>>
        metrics;
    std::size_t byte_budget = 0;
};
} // namespace SimpleFluid::Meshes
namespace SimpleFluid::Meshes::AMR
{
/** Full bytes establish equality; these codecs are internal owner-authenticated
 * transport and expose no public witness constructor or decoder. */
AdaptiveBytes encode_frozen_part(const FrozenPart&);
std::shared_ptr<const FrozenPart> decode_frozen_part(std::span<const char>);
AdaptiveBytes encode_frozen_batch(std::span<const std::shared_ptr<const FrozenNativeBundle::Data>>, std::size_t);
} // namespace SimpleFluid::Meshes::AMR

namespace SimpleFluid::Meshes::AMR
{
struct FrozenEvaluationCache
{
    NativeFragmentCache fragments;
    std::map<const FrozenPart*, MultiRegionMesh::Region> providers;
};
const MultiRegionMesh::Region& frozen_provider(const std::shared_ptr<const FrozenPart>&, FrozenEvaluationCache&);
PLIC::GeometryPoint frozen_transform(const FrozenNativeRoot::Data&, MeshUtils::Vec3);
FrozenMetric frozen_metric(const FrozenNativeBundle::Data&, const FrozenNativeRoot::Data&, FrozenEvaluationCache&);
} // namespace SimpleFluid::Meshes::AMR
