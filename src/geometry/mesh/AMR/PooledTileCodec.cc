#include "PooledTileCodec.hh"
#include "geometry/PLIC/PlanarGeometry.hh"
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
PooledTile freeze_tile(const ReceivedTileRecipe& recipe, int resident_rank, IncidencePools& pools)
{
    if (resident_rank < 0)
        throw std::invalid_argument("Invalid physical tile resident.");
    PooledTile result;
    result.support = pools.supports.intern(recipe.support);
    result.domain = pools.domains.intern(recipe.domain);
    if (recipe.first.owner == resident_rank)
        result.first_cell = pools.cells.intern(recipe.first.cell);
    if (recipe.second && recipe.second->owner == resident_rank)
        result.second_cell = pools.cells.intern(recipe.second->cell);
    if (result.first_cell == PooledTile::absent_cell && result.second_cell == PooledTile::absent_cell)
        throw std::invalid_argument("Physical tile has no authenticated resident incidence.");
    result.emission = recipe.emission;
    result.image_slot = pools.images.intern(recipe.second_image);
    const auto frame = PLIC::capture_incidence_frame(*recipe.face);
    result.plane = pools.intern_plane(frame.support);
    result.normal_slot = pools.normals.intern(IncidencePools::Normal(frame.normal));
    const auto points = recipe.face->vertex_identities();
    result.intrinsic_loop.reserve(points.size());
    for (const auto& point : points)
        result.intrinsic_loop.push_back(pools.points.intern(point));
    return result;
}
SupportFace restore_tile(const PooledTile& tile, std::span<const std::size_t> loop, const IncidencePools& pools)
{
    std::vector<IncidencePools::Point> points;
    points.reserve(loop.size());
    for (const auto slot : loop)
        points.push_back(pools.points.get(slot));
    const PLIC::OrdinaryIncidenceFrame frame{pools.planes.get(tile.plane), pools.normals.get(tile.normal_slot).values};
    return PLIC::restore_incidence_loop(frame, points);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
