/** @file CanonicalExportCoordinates.hh
 * @brief Private allocation-free numerical access to canonical global coordinates.
 */
#pragma once
#include "geometry/PLIC/details/GeometryKernels.hh"

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
        : geometry_(canonical(geometry)), operation_(geometry_->limits)
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
        constexpr auto mode = ArithmeticMode::Ordinary;
        return kernel::bounded<mode>(kernel::operator+(kernel::scaled<mode>(geometry_->vertices.at(index).coordinates,
                                                           geometry_->frame.exponent),
                                         kernel::point<mode>(geometry_->frame.origin)))
            .value;
    }

private:
    static std::shared_ptr<const GeometryStorage<ArithmeticMode::Ordinary>> canonical(
        const DynamicConvexGeometry& geometry)
    {
        const auto& storage = DynamicGeometryAccess::implementation(geometry);
        const auto source = storage->export_source ? storage->export_source : storage;
        const auto typed = checked_geometry_storage<ArithmeticMode::Ordinary>(source);
        if (!typed->point_export_cache)
            throw GeometryError(
                GeometryErrorCode::InvalidTopology, "Canonical coordinates require a finished finite geometry.");
        return typed;
    }
    const std::shared_ptr<const GeometryStorage<ArithmeticMode::Ordinary>> geometry_;
    Operation operation_;
};
} // namespace SimpleFluid::PLIC::details
