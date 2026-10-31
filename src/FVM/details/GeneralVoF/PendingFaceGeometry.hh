/** @file PendingFaceGeometry.hh
 * @brief Ordered scalar face metadata and consumable geometric preparation owners.
 */
#pragma once
#include "geometry/swept_volume/FiniteVoFTransport.hh"
#include <optional>
#include <utility>

namespace SimpleFluid::AMR
{
struct AdaptiveTile;
}

namespace SimpleFluid::VoF::general_vof_detail
{
struct PendingScalarFace
{
    const AMR::AdaptiveTile* tile;
    geometry::swept_volume::Value control;
    long double ordinary_volume_budget;
    std::optional<std::size_t> geometry;
};

struct PendingGeometricFace
{
    static constexpr auto mode = ArithmeticMode::Ordinary;
    using Point = PLIC::Point<mode>;
    geometry::swept_volume::Sweep<mode> recipe;
    std::vector<std::size_t> candidates;

    /** Move complete points in source-index order and destroy the old recipe
     * buffers. Their cumulative admission charges are deliberately retained. */
    [[nodiscard]] std::vector<Point> take_points(geometry::swept_volume::Work& work)
    {
        const auto n = recipe.current.size();
        std::vector<Point> points;
        if (recipe.departure.size() != n)
            throw std::invalid_argument("Pending sweep point arrays have different sizes.");
        if (n > (points.max_size() - 2) / 2 || n > (std::numeric_limits<std::size_t>::max() / sizeof(Point) - 2) / 2)
            throw PLIC::GeometryError(
                PLIC::GeometryErrorCode::ResourceLimit, "Pending sweep point table capacity overflows.");
        const auto count = 2 * n + 2;
        work.charge(count * sizeof(Point));
        points.reserve(count);
        work.charge((points.capacity() - count) * sizeof(Point));
        static_assert(std::is_nothrow_move_constructible_v<Point>);
        points.push_back(std::move(recipe.center));
        points.push_back(std::move(recipe.backcenter));
        for (auto& point : recipe.current)
            points.push_back(std::move(point));
        for (auto& point : recipe.departure)
            points.push_back(std::move(point));
        recipe = {};
        return points;
    }
};

/** Scalar entries preserve topology/work-capture order. Only geometric slots
 * own recipes; taking one destroys its pending owners even if processing throws.
 * Dropping owners never refunds Work's conservative cumulative charges. */
class PendingFaceGeometry
{
public:
    void push(const AMR::AdaptiveTile* tile, geometry::swept_volume::Sweep<ArithmeticMode::Ordinary> recipe,
        std::vector<std::size_t> candidates, bool geometric, geometry::swept_volume::Work& work)
    {
        PendingScalarFace scalar{tile, recipe.control, recipe.ordinary_volume_budget, std::nullopt};
        if (geometric)
        {
            scalar.geometry = geometry_.size();
            work.push(geometry_,
                std::optional<PendingGeometricFace>(std::in_place, std::move(recipe), std::move(candidates)));
        }
        work.push(scalars_, scalar);
    }

    [[nodiscard]] std::span<const PendingScalarFace> faces() const& noexcept { return scalars_; }
    std::span<const PendingScalarFace> faces() const&& = delete;

    [[nodiscard]] PendingGeometricFace take_geometry(std::size_t index)
    {
        auto& slot = geometry_.at(index);
        if (!slot)
            throw std::logic_error("Pending face geometry has already been consumed.");
        static_assert(std::is_nothrow_move_constructible_v<PendingGeometricFace>);
        auto result = std::move(*slot);
        slot.reset();
        return result;
    }

private:
    std::vector<PendingScalarFace> scalars_;
    std::vector<std::optional<PendingGeometricFace>> geometry_;
};
} // namespace SimpleFluid::VoF::general_vof_detail
