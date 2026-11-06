/** @file adaptive_storage_test_helpers.hh
 * @brief Independent translated native components for metadata scaling tests.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedMixedKernel.hh"
#include "geometry/mesh/AMR/ExtrudedAMR.hh"
#include "geometry/mesh/AMR/MixedAMR.hh"
#include "geometry/mesh/OrthogonalCartesian3D.hh"
#include <Teuchos_DefaultSerialComm.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace SimpleFluid::test::adaptive_storage
{
using namespace Meshes::AMR;
enum class Provider
{
    Cartesian,
    Product,
    Polyhedral,
    Static,
    Mixed
};
inline constexpr std::array providers{
    Provider::Cartesian, Provider::Product, Provider::Polyhedral, Provider::Static, Provider::Mixed};
inline std::string_view name(Provider provider)
{
    switch (provider)
    {
        case Provider::Cartesian:
            return "Cartesian";
        case Provider::Product:
            return "Product";
        case Provider::Polyhedral:
            return "Polyhedral";
        case Provider::Static:
            return "Static";
        case Provider::Mixed:
            return "Mixed";
    }
    throw std::invalid_argument("Unknown storage fixture provider.");
}
struct DisconnectedFixture
{
    std::shared_ptr<const AdaptiveProviderKernel> kernel;
    std::vector<AdaptiveLeaf> owned;
    AdaptiveCellKey changed;
};

namespace details
{
inline void append_word(AdaptiveBytes& out, std::uint64_t value)
{
    for (unsigned byte = 0; byte < 8; ++byte)
        out.push_back(static_cast<char>((value >> (8 * byte)) & 255));
}
inline AdaptiveBytes domain(std::size_t component)
{
    AdaptiveBytes result{'s', 't', 'o', 'r', 'a', 'g', 'e'};
    append_word(result, component);
    return result;
}
inline AdaptiveBytes component_identity(const AdaptiveBytes& domain, const AdaptiveBytes& local)
{
    AdaptiveBytes result{'s', 't', 'o', 'r', 'a', 'g', 'e', 2};
    append_word(result, domain.size());
    result.insert(result.end(), domain.begin(), domain.end());
    append_word(result, local.size());
    result.insert(result.end(), local.begin(), local.end());
    return result;
}
inline std::vector<std::uint64_t> selected(std::size_t count)
{
    std::vector<std::uint64_t> result(count);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

/** Dispatches actual detached kernels by physical region. This fixture wrapper
 * deliberately has no Cartesian source capability: these tests isolate shared
 * query metadata, while separate provider oracles exercise trusted sources. */
class DispatchKernel final : public AdaptiveProviderKernel
{
public:
    std::map<std::uint64_t, std::shared_ptr<const AdaptiveProviderKernel>> routes;
    void seal_routes()
    {
        local_repack_ = std::all_of(
            routes.begin(), routes.end(), [](const auto& route) { return route.second->supports_leaf_local_repack(); });
    }
    bool supports_leaf_local_repack() const noexcept override { return local_repack_; }
    std::vector<AdaptiveLeaf> repack(std::span<const AdaptiveLeaf> leaves) const override
    {
        std::map<const AdaptiveProviderKernel*, std::vector<AdaptiveLeaf>> selections;
        for (const auto& leaf : leaves)
            selections[&route(leaf)].push_back(leaf);
        std::vector<AdaptiveLeaf> result;
        result.reserve(leaves.size());
        for (const auto& [kernel, selection] : selections)
        {
            auto packed = kernel->repack(selection);
            for (auto& leaf : packed)
                result.push_back(std::move(leaf));
        }
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        return result;
    }
    AdaptiveBytes descriptor() const override
    {
        AdaptiveBytes result{'s', 't', 'o', 'r', 'a', 'g', 'e', 1};
        append_word(result, routes.size());
        for (const auto& [region, kernel] : routes)
        {
            append_word(result, region);
            const auto value = kernel->descriptor();
            append_word(result, value.size());
            result.insert(result.end(), value.begin(), value.end());
        }
        return result;
    }
    void validate(const AdaptiveLeaf& leaf) const override { route(leaf).validate(leaf); }
    AdaptiveCellMetrics metrics(const AdaptiveLeaf& leaf) const override { return route(leaf).metrics(leaf); }
    PLIC::ConvexGeometryPtr<ArithmeticMode::Ordinary> geometry(const AdaptiveLeaf& leaf) const override
    {
        return route(leaf).geometry(leaf);
    }
    std::vector<AdaptiveSurface> surfaces(const AdaptiveLeaf& leaf) const override
    {
        auto result = route(leaf).surfaces(leaf);
        // Pure kernels can use support IDs local to one detached assembly.
        // Keep joined sides equal within this component, and separate the
        // independent assemblies even if their raw support IDs coincide.
        for (auto& surface : result)
            surface.key = component_identity(leaf.domain, surface.key);
        return result;
    }
    PLIC::PointIdentity<ArithmeticMode::Ordinary> contact_point(
        const AdaptiveLeaf& leaf, const PLIC::PointIdentity<ArithmeticMode::Ordinary>& point) const override
    {
        return route(leaf).contact_point(leaf, point);
    }
    AdaptiveContactCapabilities contact_capabilities(const AdaptiveLeaf& leaf) const override
    {
        return route(leaf).contact_capabilities(leaf);
    }
    std::vector<AdaptiveContactImage> closed_contact_images(const AdaptiveLeaf& leaf) const override
    {
        return route(leaf).closed_contact_images(leaf);
    }
    std::vector<AdaptiveCellKey> matching_partners(const AdaptiveLeaf& leaf) const override
    {
        return route(leaf).matching_partners(leaf);
    }
    AdaptiveTrace traces(const AdaptiveLeaf& leaf) const override
    {
        auto result = route(leaf).traces(leaf);
        for (auto& face : result.faces)
            face.stem = component_identity(leaf.domain, face.stem);
        return result;
    }
    AdaptiveSplit closure_split(const AdaptiveLeaf& leaf, std::size_t axis) const override
    {
        return route(leaf).closure_split(leaf, axis);
    }
    AdaptiveSplitResult split(const AdaptiveLeaf& leaf, AdaptiveSplit mode) const override
    {
        return route(leaf).split(leaf, mode);
    }
    std::vector<AdaptiveFamily> ancestry_families(const AdaptiveLeaf& leaf) const override
    {
        return route(leaf).ancestry_families(leaf);
    }
    std::vector<AdaptiveCellKey> refinement_group(const AdaptiveCellKey& key, AdaptiveSplit mode) const override
    {
        return routes.at(key.region)->refinement_group(key, mode);
    }
    std::vector<AdaptiveBytes> coarsening_group(const AdaptiveFamily& family) const override
    {
        return route(family.parent).coarsening_group(family);
    }

private:
    const AdaptiveProviderKernel& route(const AdaptiveLeaf& leaf) const { return *routes.at(leaf.key.region); }
    bool local_repack_ = false;
};

inline std::shared_ptr<const Meshes::UnstructuredMesh> box(double lower, double upper)
{
    using U = Meshes::UnstructuredMesh;
    const Arr<MeshUtils::Vec3> nodes{{lower, 0, 0}, {upper, 0, 0}, {upper, 1, 0}, {lower, 1, 0}, {lower, 0, 1},
        {upper, 0, 1}, {upper, 1, 1}, {lower, 1, 1}};
    U::CellDefinition cell;
    cell.type = MeshUtils::CellType::HEXAHEDRON;
    cell.node_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    const Arr<U::CellDefinition> cells{cell};
    U unlabelled(nodes, cells);
    Arr<U::BoundaryFaceDefinition> boundaries;
    for (std::size_t face = 0; face < unlabelled.num_faces(); ++face)
    {
        const bool join = unlabelled.face_centroid(face).x == lower;
        boundaries.push_back({unlabelled.face_nodes(face), join ? 0 : 1, join ? "join" : "wall"});
    }
    return std::make_shared<const U>(nodes, cells, boundaries);
}
inline std::shared_ptr<const PolyhedralAMR> polyhedron(double lower, std::uint64_t region)
{
    PolyhedralAMRGrid config;
    config.mesh = std::make_shared<Meshes::MultiRegionMesh>(
        std::vector<Meshes::MultiRegionMesh::Region>{Meshes::native_region("polyhedral", box(lower, lower + 1))},
        std::vector<Meshes::MultiRegionMesh::Interface>{}, Teuchos::rcp(new Teuchos::SerialComm<int>()));
    config.physical_regions = {region};
    config.materials = {5};
    return std::make_shared<const PolyhedralAMR>(std::move(config));
}
inline ExtrudedAMRGrid product_grid(double lower, std::uint64_t region)
{
    ExtrudedAMRGrid config;
    config.region_id = region;
    config.material_id = 5;
    config.xy_nodes = {{lower, 0, 0}, {lower + 1, 0, 0}, {lower + 1, 1, 0}, {lower, 1, 0}};
    config.base_cells = {{0, 1, 2, 3}};
    config.boundaries = {{0, 1, "south"}, {1, 2, "east"}, {2, 3, "north"}, {3, 0, "west"}};
    config.z_edges = {0, 1};
    return config;
}
inline AdaptiveBootstrap component(Provider provider, std::size_t index)
{
    const auto region = std::uint64_t(1000 + 4 * index);
    const auto lower = double(3 * index);
    const auto provenance = domain(index);
    if (provider == Provider::Cartesian)
    {
        AMRGrid config;
        config.block_cells = {2, 2, 2};
        config.origin = {lower, 0, 0};
        config.extent = {1, 1, 1};
        config.region_id = region;
        const CartesianAMR mesh(config);
        return mesh.detach_owned(selected(mesh.geometry()->num_cells()), provenance);
    }
    if (provider == Provider::Product)
    {
        const ExtrudedAMR mesh(product_grid(lower, region));
        return mesh.detach_owned(selected(mesh.geometry()->num_cells()), provenance);
    }
    if (provider == Provider::Polyhedral)
    {
        const auto mesh = polyhedron(lower, region);
        return mesh->detach_owned(selected(mesh->geometry()->num_cells()), provenance);
    }
    if (provider == Provider::Static)
    {
        const Vec3D<Arr<real_t>> edges{Arr<real_t>{lower, lower + 1}, Arr<real_t>{0, 1}, Arr<real_t>{0, 1}};
        const auto mesh = std::make_shared<const Meshes::OrthogonalCartesian3D>(edges);
        return detach_static_participant(
            Meshes::native_region("stationary", mesh), region + 2, region, 5, selected(1), provenance);
    }
    if (provider == Provider::Mixed)
    {
        MixedAMRGrid config;
        config.product = std::make_shared<const LocalProductAMR>(product_grid(lower, region));
        config.physical_region = region;
        config.adaptive = {polyhedron(lower + 1, region + 1)};
        config.joins = {{0, 1, "east", "polyhedral/join"}};
        const MixedAMR mesh(std::move(config));
        return mesh.detach_owned(selected(mesh.geometry()->num_cells()), provenance);
    }
    throw std::invalid_argument("Unknown disconnected storage fixture provider.");
}
} // namespace details

/** Build in a serial test. The first component is identical at every count;
 * later components have disjoint physical boxes, keys, domains and supports.
 * The real Mixed component keeps its internal product/polyhedron join. */
inline DisconnectedFixture make_disconnected(Provider provider, std::size_t component_count)
{
    if (!component_count || component_count > 1024)
        throw std::invalid_argument("Disconnected storage fixture count is outside its focused test bound.");
    auto dispatch = std::make_shared<details::DispatchKernel>();
    DisconnectedFixture result;
    for (std::size_t index = 0; index < component_count; ++index)
    {
        auto input = details::component(provider, index);
        for (auto& leaf : input.owned)
        {
            const auto [route, added] = dispatch->routes.emplace(leaf.key.region, input.kernel);
            if (!added && route->second != input.kernel)
                throw std::logic_error("Disconnected fixture reuses another component's physical region.");
            result.owned.push_back(std::move(leaf));
        }
    }
    std::sort(result.owned.begin(), result.owned.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
    if (result.owned.empty())
        throw std::logic_error("Disconnected native fixture contains no leaves.");
    result.changed = result.owned.front().key;
    dispatch->seal_routes();
    result.kernel = std::move(dispatch);
    return result;
}
} // namespace SimpleFluid::test::adaptive_storage
