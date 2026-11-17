/** @file CanonicalExportCoordinates.hh
 * @brief Private allocation-free numerical access to canonical global coordinates.
 */
#pragma once
#include "geometry/PLIC/details/ordinary/FixedBoxGeometry.hh"

namespace SimpleFluid::PLIC::details
{
/** Coordinate publication uses the environment under which finish validated it.
 * The caller's complete floating-point environment is restored on every exit. */
class SIMPLEFLUID_LOCAL FrozenExportEnvironment
{
public:
    explicit FrozenExportEnvironment(const PointExportCache& cache)
    {
        if (std::fegetenv(&previous_) != 0)
            throw GeometryError(GeometryErrorCode::UnresolvedArithmetic,
                "Cannot capture caller floating-point environment for canonical coordinates.");
        if (std::fesetenv(&cache.environment) != 0)
        {
            (void) std::fesetenv(&previous_);
            throw GeometryError(GeometryErrorCode::UnresolvedArithmetic,
                "Cannot install canonical coordinate floating-point environment.");
        }
    }
    ~FrozenExportEnvironment() { (void) std::fesetenv(&previous_); }
    FrozenExportEnvironment(const FrozenExportEnvironment&) = delete;
    FrozenExportEnvironment& operator=(const FrozenExportEnvironment&) = delete;

private:
    std::fenv_t previous_{};
};

/** Numerical-only view of the original global coordinate expression. A pure
 * numerical rebase can lose world-coordinate extent in its working frame;
 * this view retains the pre-rebase source and never publishes point owners. */
class SIMPLEFLUID_LOCAL OrdinaryExportCoordinates
{
public:
    explicit OrdinaryExportCoordinates(const DynamicConvexGeometry& geometry)
        : operation_(DynamicGeometryAccess::implementation(geometry)->limits), geometry_(canonical(geometry))
    {
    }
    explicit OrdinaryExportCoordinates(const ConvexCellGeometry<ArithmeticMode::Ordinary>& geometry)
        : OrdinaryExportCoordinates(*geometry.dynamic_geometry())
    {
    }
    [[nodiscard]] std::size_t size() const noexcept { return geometry_->vertices.size(); }
    [[nodiscard]] std::array<long double, 3> at(std::size_t index) const
    {
        FrozenExportEnvironment environment(*geometry_->point_export_cache);
        namespace kernel = ordinary_kernel;
        return kernel::represented_world_corner(geometry_->frame, geometry_->vertices.at(index).coordinates);
    }

private:
    static std::shared_ptr<const GeometryStorage<ArithmeticMode::Ordinary>> canonical(
        const DynamicConvexGeometry& geometry)
    {
        const auto& storage = DynamicGeometryAccess::implementation(geometry);
        const auto source = storage->native_export_source
                                ? DynamicGeometryAccess::implementation(
                                      *materialize_native_construction(storage->native_export_source)->dynamic_geometry())
                                : (storage->export_source ? storage->export_source : storage);
        const auto typed = checked_geometry_storage<ArithmeticMode::Ordinary>(source);
        if (!typed->point_export_cache)
            throw GeometryError(
                GeometryErrorCode::InvalidTopology, "Canonical coordinates require a finished finite geometry.");
        return typed;
    }
    Operation operation_;
    const std::shared_ptr<const GeometryStorage<ArithmeticMode::Ordinary>> geometry_;
};
/** Private admitted producer coordinates. Reuses the original validated native
 * body directly rather than allocating a temporary GeometryStorage wrapper.
 * Its frame, vertices and frozen FP environment are the exact ones copied by
 * materialize_native_construction in the public compatibility accessor. */
class SIMPLEFLUID_LOCAL NativeOrdinaryExportCoordinates
{
public:
    explicit NativeOrdinaryExportCoordinates(const ConvexCellGeometry<ArithmeticMode::Ordinary>& cell)
        : operation_(DynamicGeometryAccess::implementation(*cell.dynamic_geometry())->limits)
    {
        const auto& stored = DynamicGeometryAccess::implementation(*cell.dynamic_geometry());
        if (stored->native_export_source)
        {
            const auto& root = stored->native_export_source;
            if (root->kind() != GeometryConstruction::Kind::Source || !root->source().native() ||
                !root->source().native()->body)
                throw GeometryError(
                    GeometryErrorCode::ProvenanceMismatch, "Missing validated original native coordinate body.");
            native_ = root->source().native()->body;
            if (!native_->export_cache)
                throw GeometryError(
                    GeometryErrorCode::InvalidTopology, "Native coordinates lack their frozen environment.");
        }
        else
        {
            geometry_ = checked_geometry_storage<ArithmeticMode::Ordinary>(
                stored->export_source ? stored->export_source : stored);
            if (!geometry_->point_export_cache)
                throw GeometryError(
                    GeometryErrorCode::InvalidTopology, "Canonical coordinates require finished finite geometry.");
        }
    }
    [[nodiscard]] std::size_t size() const noexcept
    {
        return native_ ? native_->vertices.size() : geometry_->vertices.size();
    }
    [[nodiscard]] std::array<long double, 3> at(std::size_t index) const
    {
        if (native_)
        {
            FrozenExportEnvironment environment(*native_->export_cache);
            return ordinary_kernel::represented_world_corner(native_->frame, native_->vertices.at(index).coordinates);
        }
        FrozenExportEnvironment environment(*geometry_->point_export_cache);
        return ordinary_kernel::represented_world_corner(geometry_->frame, geometry_->vertices.at(index).coordinates);
    }

private:
    Operation operation_;
    std::shared_ptr<const NativeGeometryBody> native_;
    std::shared_ptr<const GeometryStorage<ArithmeticMode::Ordinary>> geometry_;
};
} // namespace SimpleFluid::PLIC::details
