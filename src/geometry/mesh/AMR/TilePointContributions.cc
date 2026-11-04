#include "TilePointContributions.hh"
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
void visit_tile_physical_points(const TileBodyScalars& tile, TileBodyLoopView loop, int rank, IncidencePools& pools,
    const std::function<void(std::size_t, std::size_t)>& visit)
{
    if (rank < 0)
        throw std::invalid_argument("Invalid physical tile incidence.");
    for (const auto point : loop)
    {
        if (tile.first_cell != PooledTile::absent_cell)
            visit(tile.first_cell, point);
        if (tile.second_cell != PooledTile::absent_cell)
        {
            const auto translated = PLIC::point_translated<ArithmeticMode::Ordinary>(pools.points.get(point),
                PLIC::displacement_difference<ArithmeticMode::Ordinary>({}, pools.images.get(tile.image_slot)));
            visit(tile.second_cell, pools.points.intern(translated));
        }
    }
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
