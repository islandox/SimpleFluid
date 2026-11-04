/** @file CellGeometryView.cc
 * @brief Procedural Cartesian source facts and explicitly admitted geometry operations.
 */
#include "geometry/PLIC/CellGeometryView.hh"
#include "geometry/PLIC/details/CanonicalExportCoordinates.hh"
#include "geometry/PLIC/details/CellGeometryCatalog.tcc"
#include "geometry/PLIC/details/CellGeometryCoordinates.hh"
#include "geometry/PLIC/details/GeometryAllocationLedgerInternal.hh"
#include "geometry/PLIC/details/LocalCoordinates.hh"
#include "geometry/PLIC/details/NativeCellGeometrySourceAccess.hh"
#include "geometry/PLIC/details/ordinary/FixedBoxGeometry.hh"
#include "geometry/PLIC/details/ordinary/NativePlanarQueries.hh"
#include "geometry/PLIC/details/ordinary/NativeSupportAccess.hh"
#include <atomic>
#include <cfenv>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace SimpleFluid::PLIC
{
namespace
{
constexpr auto ordinary = ArithmeticMode::Ordinary;
namespace kernel = details::ordinary_kernel;
namespace planar = details::ordinary_planar_queries;
using kernel::operator+;
using kernel::operator-;
using kernel::operator*;
using kernel::operator/;
/** Restore every floating-point flag/control field, including exceptional exits. */
class FrozenEnvironment
{
public:
    explicit FrozenEnvironment(const std::fenv_t& environment)
    {
        if (std::fegetenv(&previous_) != 0)
            throw GeometryError(GeometryErrorCode::UnresolvedArithmetic, "Cannot capture geometry caller environment.");
        if (std::fesetenv(&environment) != 0)
        {
            (void) std::fesetenv(&previous_);
            throw GeometryError(
                GeometryErrorCode::UnresolvedArithmetic, "Cannot install Cartesian source environment.");
        }
    }
    ~FrozenEnvironment() { (void) std::fesetenv(&previous_); }

private:
    std::fenv_t previous_{};
};
GeometryLimits ordinary_limits(GeometryLimits limits)
{
    limits.arithmetic_mode = ordinary;
    return limits;
}
} // namespace
struct FixedFaceInput
{
    std::array<details::ordinary_native::WidePoint, 4> polygon;
    details::ordinary_native::WidePoint normal;
};
struct CartesianCellSource::Impl
{
    struct Facts
    {
        int exponent = 0;
        bool custom_supports = false;
        kernel::StorageScalar constructor_volume;
        mutable std::once_flag moments_once;
        mutable std::optional<kernel::Integrals> integrals;
        mutable std::atomic<bool> moments_ready{false};
    };
    /** One allocator block, with explicit element count and no array cookie or
     * per-cell control blocks. Allocation is admitted before allocate(). */
    struct FactsArray
    {
        Facts* data = nullptr;
        std::size_t count = 0;
        FactsArray() = default;
        FactsArray(const FactsArray&) = delete;
        FactsArray& operator=(const FactsArray&) = delete;
        ~FactsArray()
        {
            if (data)
            {
                std::destroy_n(data, count);
                std::allocator<Facts>{}.deallocate(data, count);
            }
        }
        void allocate(std::size_t n)
        {
            if (!n)
                return;
            auto* next = std::allocator<Facts>{}.allocate(n);
            try
            {
                std::uninitialized_value_construct_n(next, n);
            }
            catch (...)
            {
                std::allocator<Facts>{}.deallocate(next, n);
                throw;
            }
            data = next;
            count = n;
        }
        std::size_t size() const noexcept { return count; }
        std::size_t capacity() const noexcept { return count; }
        Facts& at(std::size_t i)
        {
            if (i >= count)
                throw std::out_of_range("Cartesian source fact slot.");
            return data[i];
        }
        const Facts& at(std::size_t i) const
        {
            if (i >= count)
                throw std::out_of_range("Cartesian source fact slot.");
            return data[i];
        }
        Facts& operator[](std::size_t i) { return at(i); }
        const Facts& operator[](std::size_t i) const { return at(i); }
    };
    CartesianSourceDescription description;
    FactsArray facts;
    details::NativeCellGeometryMoments moment_policy = details::NativeCellGeometryMoments::PrimedState;
    std::fenv_t environment{};
    GeometryLimits limits;
    std::pair<GeometryPoint, GeometryPoint> corners(std::size_t slot) const
    {
        const auto& cell = description.cells.at(slot);
        GeometryPoint lower, upper;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            lower[axis] = description.axes[axis].at(cell.lower[axis]);
            upper[axis] = description.axes[axis].at(cell.upper[axis]);
        }
        return {lower, upper};
    }
    kernel::FixedBoxEvaluation evaluate(std::size_t slot) const
    {
        FrozenEnvironment environment_guard(environment);
        const auto [lower, upper] = corners(slot);
        kernel::FixedBoxEvaluation result(lower, upper);
        if (result.frame.exponent != facts.at(slot).exponent)
            throw GeometryError(GeometryErrorCode::UnresolvedArithmetic, "Cartesian source frame changed.");
        result.volume = facts[slot].constructor_volume;
        return result;
    }
    kernel::Integrals moment_integrals(std::size_t slot) const
    {
        auto& fact = facts.at(slot);
        std::call_once(fact.moments_once,
            [&]
            {
                const auto box = evaluate(slot); // Rebuild immutable normalized values under ctor controls.
                ++details::statistics.moment_evaluations;
                auto complete = kernel::integrate_ordinary<ordinary>(box, true); // Caller controls, as raw moments().
                fact.integrals.emplace(std::move(complete));
                fact.moments_ready.store(true, std::memory_order_release);
            });
        return *fact.integrals;
    }
    std::array<long double, 3> canonical_corner(const kernel::FixedBoxEvaluation& box, std::size_t vertex) const
    {
        FrozenEnvironment environment_guard(environment);
        return box.world_corner(vertex);
    }
    kernel::Point original_normal(const kernel::FixedBoxEvaluation& box, std::size_t face) const
    {
        FrozenEnvironment environment_guard(environment);
        return kernel::represented_source_face_normal(box, kernel::fixed_box_faces.at(face)).normal;
    }
    FixedFaceInput face_input(std::size_t slot, std::size_t face) const
    {
        const auto box = evaluate(slot);
        FixedFaceInput result;
        const auto& loop = kernel::fixed_box_faces.at(face);
        for (std::size_t i = 0; i < 4; ++i)
            result.polygon[i] = canonical_corner(box, loop[i]);
        const auto normal = original_normal(box, face);
        for (std::size_t axis = 0; axis < 3; ++axis)
            result.normal[axis] = normal[axis].value;
        return result;
    }
};
CartesianCellSource::CartesianCellSource(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
std::shared_ptr<const CartesianCellSource> CartesianCellSource::make(
    CartesianSourceDescription description, GeometryLimits requested)
{
    if (!description.original_supports.empty() ||
        std::any_of(description.cells.begin(), description.cells.end(),
            [](const auto& cell)
            {
                return std::any_of(cell.support_slots.begin(), cell.support_slots.end(),
                    [](auto slot) { return slot != CartesianSourceCellSlot::canonical_support; });
            }))
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch,
            "Adopting original support ancestry requires the private native-provider boundary.");
    return make_authenticated(std::move(description), requested, details::NativeCellGeometryMoments::PrimedState);
}
std::shared_ptr<const CartesianCellSource> CartesianCellSource::make_authenticated(
    CartesianSourceDescription description, GeometryLimits requested, details::NativeCellGeometryMoments moment_policy)
{
    const auto limits = ordinary_limits(requested);
    details::Operation operation(limits);
    if (std::min(limits.maximum_vertices, details::active_work->limits.maximum_vertices) < 8 ||
        std::min(limits.maximum_faces, details::active_work->limits.maximum_faces) < 6)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Cartesian source exceeds fixed geometry shape caps.");
    auto& work = *details::active_work;
    const auto per_cell_guard = work.operation_limit;
    const bool owned_default_guard = !operation.previous && !limits.maximum_operations;
    if (owned_default_guard && !description.cells.empty())
    {
        if (per_cell_guard > std::numeric_limits<std::size_t>::max() / description.cells.size())
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Derived source-cell work guard overflows.");
        work.operation_limit = per_cell_guard * description.cells.size();
    }
    const auto aggregate_guard = work.operation_limit;
    work.charge(1, sizeof(CartesianCellSource) + sizeof(Impl));
    for (const auto& axis : description.axes)
    {
        work.charge(axis.capacity(), sizeof(double));
        if (axis.size() < 2 || axis.size() > std::numeric_limits<std::uint32_t>::max())
            throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid Cartesian source axis length.");
        for (std::size_t i = 0; i < axis.size(); ++i)
            if (!std::isfinite(axis[i]) || (i && !(axis[i] > axis[i - 1])))
                details::reject(ordinary, "Cartesian source axis is not finite/increasing.");
    }
    work.charge(description.cells.capacity(), sizeof(CartesianSourceCellSlot));
    work.charge(description.original_supports.capacity(), sizeof(PlaneSupportIdentity<ordinary>));
    for (const auto& support : description.original_supports)
    {
        if (canonical_key_size(support) == 0)
            throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Empty Cartesian original support.");
        work.retain_identity(support);
    }
    work.charge(description.cells.size(), sizeof(Impl::Facts));
    auto impl = std::make_shared<Impl>();
    impl->description = std::move(description);
    impl->limits = limits;
    impl->moment_policy = moment_policy;
    impl->facts.allocate(impl->description.cells.size());
    using SupportLookup = std::map<PlaneSupportIdentity<ordinary>, std::uint32_t>;
    SupportLookup support_lookup;
    for (std::size_t index = 0; index < impl->description.original_supports.size(); ++index)
    {
        if (index >= CartesianSourceCellSlot::canonical_support)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Cartesian support catalog exceeds compact slots.");
        if (!support_lookup.contains(impl->description.original_supports[index]))
        {
            work.charge(1, sizeof(SupportLookup::value_type) + 4 * sizeof(void*));
            support_lookup.emplace(impl->description.original_supports[index], static_cast<std::uint32_t>(index));
        }
    }
    std::array<std::vector<std::uint32_t>, 3> canonical_support_slots;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        work.charge(impl->description.axes[axis].size(), sizeof(std::uint32_t));
        canonical_support_slots[axis].assign(
            impl->description.axes[axis].size(), CartesianSourceCellSlot::canonical_support);
        if (canonical_support_slots[axis].capacity() > impl->description.axes[axis].size())
            work.charge(
                canonical_support_slots[axis].capacity() - impl->description.axes[axis].size(), sizeof(std::uint32_t));
    }
    for (std::size_t slot = 0; slot < impl->description.cells.size(); ++slot)
    {
        if (owned_default_guard)
            work.operation_limit =
                per_cell_guard > aggregate_guard - work.operations ? aggregate_guard : work.operations + per_cell_guard;
        auto& recipe = impl->description.cells[slot];
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (recipe.lower[axis] >= recipe.upper[axis] || recipe.upper[axis] >= impl->description.axes[axis].size())
                throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid Cartesian source interval.");
        for (const auto support : recipe.support_slots)
            if (support != CartesianSourceCellSlot::canonical_support &&
                support >= impl->description.original_supports.size())
                throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid Cartesian original-support slot.");
        const bool custom_supports = std::any_of(recipe.support_slots.begin(), recipe.support_slots.end(),
            [](const auto slot) { return slot != CartesianSourceCellSlot::canonical_support; });
        const auto [lower, upper] = impl->corners(slot);
        kernel::FixedBoxEvaluation box(lower, upper);
        std::array<kernel::Point, 6> normals;
        for (std::size_t face = 0; face < 6; ++face)
            normals[face] = kernel::represented_source_face_normal(box, kernel::fixed_box_faces[face]).normal;
        // Fixed original topology proves connectedness and outward convexity.
        // Reuse all scalar face-winding and finish publication checks in order.
        for (std::size_t face = 0; face < 6; ++face)
            kernel::represented_source_face_winding(box, kernel::fixed_box_faces[face], normals[face]);
        for (std::size_t vertex = 0; vertex < 8; ++vertex)
            (void) box.world_corner(vertex);
        ++details::statistics.volume_evaluations;
        box.volume = kernel::integrate_ordinary<ordinary>(box, false).volume;
        kernel::represented_positive_volume(box.volume);
        auto& fact = impl->facts.at(slot);
        fact.exponent = box.frame.exponent;
        fact.custom_supports = custom_supports;
        fact.constructor_volume = box.volume;
        if (moment_policy == details::NativeCellGeometryMoments::PrimedState)
        {
            std::call_once(fact.moments_once,
                [&]
                {
                    ++details::statistics.moment_evaluations;
                    auto complete = kernel::integrate_ordinary<ordinary>(box, true);
                    fact.integrals.emplace(std::move(complete));
                    fact.moments_ready.store(true, std::memory_order_release);
                });
        }
        for (std::size_t face = 0; face < 6; ++face)
        {
            if (recipe.support_slots[face] != CartesianSourceCellSlot::canonical_support)
                continue;
            const auto axis = face / 2;
            const auto axis_slot = face % 2 ? recipe.upper[axis] : recipe.lower[axis];
            auto& cached_slot = canonical_support_slots[axis][axis_slot];
            if (cached_slot != CartesianSourceCellSlot::canonical_support)
            {
                recipe.support_slots[face] = cached_slot;
                continue;
            }
            // Original identity uses the raw source evidence, matching make/import.
            // Membership and face metrics separately reconstruct captured-frame coordinates.
            const auto native_corners = kernel::fixed_box_corners(lower, upper);
            std::array<std::array<long double, 3>, 4> source_polygon;
            for (std::size_t i = 0; i < 4; ++i)
                source_polygon[i] = details::ordinary_native::wide(native_corners[kernel::fixed_box_faces[face][i]]);
            // Admit construction work before any support/backing owner exists.
            // Retained ownership is admitted independently below; failed prefixes
            // keep this conservative construction charge in the active scope.
            work.charge(details::ordinary_native::source_support_allocation_bound(source_polygon.size()), 1);
            auto support = details::ordinary_native::make_source_support(source_polygon);
            const auto found = support_lookup.find(support);
            if (found != support_lookup.end())
            {
                recipe.support_slots[face] = found->second;
                cached_slot = found->second;
                continue;
            }
            auto& supports = impl->description.original_supports;
            if (supports.size() >= CartesianSourceCellSlot::canonical_support)
                throw GeometryError(
                    GeometryErrorCode::ResourceLimit, "Cartesian support catalog exceeds compact slots.");
            if (supports.size() == supports.capacity())
            {
                const auto old_capacity = supports.capacity();
                if (old_capacity > supports.max_size() / 2)
                    throw GeometryError(
                        GeometryErrorCode::ResourceLimit, "Cartesian support catalog capacity overflows.");
                const auto capacity = old_capacity ? 2 * old_capacity : 1;
                work.charge(capacity, sizeof(PlaneSupportIdentity<ordinary>));
                supports.reserve(capacity);
                if (supports.capacity() > capacity)
                    work.charge(supports.capacity() - capacity, sizeof(PlaneSupportIdentity<ordinary>));
            }
            work.retain_identity(support);
            recipe.support_slots[face] = static_cast<std::uint32_t>(supports.size());
            work.charge(1, sizeof(SupportLookup::value_type) + 4 * sizeof(void*));
            support_lookup.emplace(support, recipe.support_slots[face]);
            cached_slot = recipe.support_slots[face];
            supports.push_back(std::move(support));
        }
        if (owned_default_guard)
            work.operation_limit = aggregate_guard;
    }
    if (std::fegetenv(&impl->environment) != 0)
        throw GeometryError(GeometryErrorCode::UnresolvedArithmetic, "Cannot capture Cartesian source environment.");
    return std::shared_ptr<const CartesianCellSource>(new CartesianCellSource(std::move(impl)));
}
std::size_t CartesianCellSource::size() const noexcept
{
    return impl_->description.cells.size();
}
std::size_t CartesianCellSource::storage_bytes() const
{
    std::size_t bytes = sizeof(*this) + sizeof(Impl) + impl_->facts.capacity() * sizeof(Impl::Facts) +
                        impl_->description.cells.capacity() * sizeof(CartesianSourceCellSlot) +
                        impl_->description.original_supports.capacity() * sizeof(PlaneSupportIdentity<ordinary>);
    for (const auto& axis : impl_->description.axes)
        bytes += axis.capacity() * sizeof(double);
    return bytes;
}
std::array<std::size_t, 3> CartesianCellSource::retention_bytes() const
{
    std::size_t populated = impl_->facts.size() * sizeof(Impl::Facts) +
                            impl_->description.cells.size() * sizeof(CartesianSourceCellSlot) +
                            impl_->description.original_supports.size() * sizeof(PlaneSupportIdentity<ordinary>);
    for (const auto& axis : impl_->description.axes)
        populated += axis.size() * sizeof(double);
    const auto metadata = sizeof(*this) + sizeof(Impl);
    return {metadata, populated, storage_bytes() - metadata - populated};
}
std::span<const PlaneSupportIdentity<ordinary>> CartesianCellSource::retained_supports() const
{
    return impl_->description.original_supports;
}
void GeometryAllocationLedger::retain(
    const std::shared_ptr<const CartesianCellSource>& source, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!source)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null Cartesian source admission.");
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, ordinary);
        else
            impl_->mode = ordinary;
        impl_->admit({allocation_detail::Kind::CartesianSource, source}, source->storage_bytes(), used, limit);
        for (const auto& support : source->retained_supports())
            impl_->retain_identity(support, used, limit);
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
void GeometryAllocationLedger::retain(
    const std::shared_ptr<const CellGeometryCatalog<M>>& catalog, std::size_t& used, std::size_t limit)
{
    try
    {
        if (!catalog)
            throw GeometryError(GeometryErrorCode::InvalidInput, "Null geometry catalog admission.");
        begin_admission(used, limit);
        if (impl_->mode)
            details::require_mode(*impl_->mode, M);
        else
            impl_->mode = M;
        if (impl_->admit({allocation_detail::Kind::CellCatalog, catalog}, catalog->storage_bytes(), used, limit))
        {
            if constexpr (M == ordinary)
                for (const auto& source : catalog->sources_)
                    retain(source, used, limit);
            for (const auto& entry : catalog->entries_)
                if (const auto* geometry = std::get_if<ConvexGeometryPtr<M>>(&entry))
                    retain(*geometry, used, limit);
        }
        impl_->last_counter = used;
    }
    catch (...)
    {
        failed_ = true;
        throw;
    }
}
template void GeometryAllocationLedger::retain<ordinary>(
    const std::shared_ptr<const CellGeometryCatalog<ordinary>>&, std::size_t&, std::size_t);
template void GeometryAllocationLedger::retain<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const CellGeometryCatalog<ArithmeticMode::BoundedCertification>>&, std::size_t&, std::size_t);
GeometryValue<ordinary> CartesianCellSource::volume(std::size_t slot) const
{
    return {impl_->facts.at(slot).constructor_volume.value};
}
GeometryPoint CartesianCellSource::origin(std::size_t slot) const
{
    return impl_->corners(slot).first;
}
GeometryMoments<ordinary> CartesianCellSource::moments(std::size_t slot, GeometryPoint origin) const
{
    details::Operation operation(impl_->limits);
    const auto frame = details::Frame{impl_->corners(slot).first, impl_->facts.at(slot).exponent};
    const auto values = kernel::represented_moments(impl_->moment_integrals(slot), frame, origin);
    GeometryMoments<ordinary> result;
    result.origin = origin;
    result.volume = {values.volume.value};
    for (std::size_t axis = 0; axis < 3; ++axis)
        result.first_moment[axis] = {values.first_moment[axis].value};
    return result;
}
GeometryPoint CartesianCellSource::centroid(std::size_t slot) const
{
    details::Operation operation(impl_->limits);
    const auto frame = details::Frame{origin(slot), impl_->facts.at(slot).exponent};
    const auto values = kernel::represented_moments(impl_->moment_integrals(slot), frame, origin(slot));
    return kernel::represented_centroid(values, impl_->facts.at(slot).constructor_volume, origin(slot));
}
std::array<long double, 3> CartesianCellSource::corner(std::size_t slot, std::size_t vertex) const
{
    FrozenEnvironment environment(impl_->environment);
    details::Operation operation(impl_->limits);
    return impl_->evaluate(slot).world_corner(vertex);
}
std::array<GeometryValue<ordinary>, 3> CartesianCellSource::axis_spans(std::size_t slot) const
{
    details::Operation operation(impl_->limits);
    const auto spans = kernel::represented_axis_spans(impl_->evaluate(slot));
    std::array<GeometryValue<ordinary>, 3> result;
    for (std::size_t axis = 0; axis < 3; ++axis)
        result[axis] = {spans[axis].value};
    return result;
}
std::array<std::array<long double, 2>, 3> CartesianCellSource::closed_contact_bounds(std::size_t slot) const
{
    FrozenEnvironment environment(impl_->environment);
    details::Operation operation(impl_->limits);
    struct Coordinates
    {
        const kernel::FixedBoxEvaluation box;
        std::size_t size() const { return 8; }
        std::array<long double, 3> at(std::size_t vertex) const { return box.world_corner(vertex); }
    } coordinates{impl_->evaluate(slot)};
    return kernel::represented_closed_bounds(coordinates);
}
Support<ordinary> CartesianCellSource::support(std::size_t slot, GeometryPoint normal, GeometryPoint anchor) const
{
    details::Operation operation(impl_->limits);
    const auto result = kernel::represented_support(impl_->evaluate(slot), normal, anchor);
    return {{result.lower.value}, {result.upper.value}};
}
PlaneSupportIdentity<ordinary> CartesianCellSource::source_support(std::size_t slot, std::size_t face) const
{
    return impl_->description.original_supports.at(impl_->description.cells.at(slot).support_slots.at(face));
}
namespace
{
template<class Query>
auto with_fixed_face(const CartesianCellSource::Impl& source, std::size_t slot, std::size_t face, Query query)
{
    details::Operation operation(source.limits);
    const auto input = source.face_input(slot, face);
    planar::validate_retained_loop(input.polygon, input.normal);
    return query(std::span<const details::ordinary_native::WidePoint>(input.polygon));
}
} // namespace
GeometryValue<ordinary> CartesianCellSource::face_area(std::size_t slot, std::size_t face) const
{
    return with_fixed_face(*impl_, slot, face, [](auto polygon) { return planar::area(polygon); });
}
GeometryPoint CartesianCellSource::face_centroid(std::size_t slot, std::size_t face) const
{
    return with_fixed_face(*impl_, slot, face, [](auto polygon) { return planar::centroid(polygon); });
}
GeometryPoint CartesianCellSource::face_area_vector(std::size_t slot, std::size_t face) const
{
    return with_fixed_face(*impl_, slot, face, [](auto polygon) { return planar::area_vector(polygon); });
}
int CartesianCellSource::face_orientation(std::size_t slot, std::size_t face) const
{
    return with_fixed_face(*impl_, slot, face, [](auto polygon) { return planar::orientation(polygon); });
}
bool CartesianCellSource::contains_source_point(
    std::size_t slot, std::size_t face, const PointIdentity<ordinary>& identity) const
{
    return with_fixed_face(*impl_, slot, face,
        [&](auto polygon) { return planar::point_inside(polygon, details::GeometryIdentityAccess::point(identity)); });
}
void details::CellCatalogAdmission::charge(std::size_t count, std::size_t width)
{
    if (used_ > limits_.byte_budget || (width && count > (limits_.byte_budget - used_) / width))
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Cell geometry catalog storage exhausted.");
    if (details::active_work)
        details::active_work->charge(count, width);
    used_ += count * width;
}
struct CellGeometryOperationWorkspace::Impl
{
    GeometryLimits limits;
    std::size_t bytes = 0, operations = 0, pairs = 0;
    GeometryAllocationLedger temporaries;
    bool failed = false;
    explicit Impl(GeometryLimits value) : limits(value)
    {
        details::Work guard(ordinary_limits(value));
        limits.maximum_operations = guard.operation_limit;
    }
    void check() const
    {
        if (failed || bytes >= limits.byte_budget || operations >= limits.maximum_operations ||
            pairs >= limits.maximum_pair_tests)
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry operation workspace is exhausted.");
    }
};
CellGeometryOperationWorkspace::CellGeometryOperationWorkspace(GeometryLimits limits)
    : impl_(std::make_unique<Impl>(limits))
{
}
CellGeometryOperationWorkspace::~CellGeometryOperationWorkspace() = default;
GeometryLimits CellGeometryOperationWorkspace::remaining_limits(ArithmeticMode mode) const
{
    impl_->check();
    auto limits = impl_->limits;
    limits.arithmetic_mode = mode;
    limits.byte_budget -= impl_->bytes;
    limits.maximum_operations -= impl_->operations;
    limits.maximum_pair_tests -= impl_->pairs;
    return limits;
}
std::size_t CellGeometryOperationWorkspace::bytes_used() const noexcept
{
    return impl_->bytes;
}
std::size_t CellGeometryOperationWorkspace::operations_used() const noexcept
{
    return impl_->operations;
}
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
void CellGeometryOperationWorkspace::retain_temporary(const ConvexGeometryPtr<M>& geometry)
{
    impl_->check();
    try
    {
        impl_->temporaries.retain(geometry, impl_->bytes, impl_->limits.byte_budget);
    }
    catch (...)
    {
        impl_->failed = true;
        throw;
    }
}
template void CellGeometryOperationWorkspace::retain_temporary<ArithmeticMode::Ordinary>(
    const ConvexGeometryPtr<ArithmeticMode::Ordinary>&);
template void CellGeometryOperationWorkspace::retain_temporary<ArithmeticMode::BoundedCertification>(
    const ConvexGeometryPtr<ArithmeticMode::BoundedCertification>&);
ConvexGeometryPtr<ordinary> CartesianCellSource::materialize(
    std::size_t slot, CellGeometryOperationWorkspace& workspace) const
{
    if (details::active_work)
        throw GeometryError(
            GeometryErrorCode::InvalidInput, "Explicit cell materialization requires an outer workspace.");
    auto& admission = *workspace.impl_;
    const auto limits = workspace.remaining_limits(ordinary);
    FrozenEnvironment environment(impl_->environment);
    try
    {
        auto [source, output] = [&]
        {
            details::Operation operation(limits);
            struct RecordWork
            {
                CellGeometryOperationWorkspace::Impl& admission;
                details::Work& work;
                ~RecordWork()
                {
                    admission.bytes += work.bytes;
                    admission.operations += work.operations;
                    admission.pairs += work.pairs;
                }
            } record{admission, operation.work};
            const auto [lower, upper] = impl_->corners(slot);
            auto source = ConvexCellGeometry<ordinary>::box(lower, upper, limits);
            if (impl_->facts.at(slot).custom_supports)
            {
                std::vector<PlaneSupportIdentity<ordinary>> supports;
                details::active_work->charge(6, sizeof(PlaneSupportIdentity<ordinary>));
                supports.reserve(6);
                if (supports.capacity() > 6)
                    details::active_work->charge(supports.capacity() - 6, sizeof(PlaneSupportIdentity<ordinary>));
                for (std::size_t face = 0; face < 6; ++face)
                    supports.push_back(source_support(slot, face));
                source = details::NativeSupportAccess::cell_supports(*source, supports);
            }
            auto output = impl_->moment_policy == details::NativeCellGeometryMoments::DeferredRaw
                              ? source
                              : ConvexCellGeometry<ordinary>::from_geometry(*source, limits);
            const auto& fact = impl_->facts.at(slot);
            // Snapshot established cache state without evaluating an unqueried
            // raw source. The copied values own no context/geometry references.
            if (fact.moments_ready.load(std::memory_order_acquire))
            {
                const auto copied = *fact.integrals;
                const auto stored = details::checked_geometry_storage<ordinary>(
                    DynamicGeometryAccess::implementation(*output->dynamic_geometry()));
                std::call_once(stored->moment_cache->prepared, [&] { stored->moment_cache->integrals = copied; });
            }
            return std::pair{std::move(source), std::move(output)};
        }();
        admission.temporaries.retain(source, admission.bytes, admission.limits.byte_budget);
        admission.temporaries.retain(output, admission.bytes, admission.limits.byte_budget);
        return output;
    }
    catch (...)
    {
        admission.failed = true;
        throw;
    }
}

template<ArithmeticMode M>
std::vector<Point<M>> source_points(const CellGeometryRange<ordinary>& sources,
    std::span<const std::size_t> source_indices, std::span<const NativeSourcePointRequest> requests,
    CellGeometryOperationWorkspace& workspace)
{
    static_assert(M == ordinary || M == ArithmeticMode::BoundedCertification);
    if (requests.empty())
        return {};
    if (details::active_work)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Source view publication requires an outer workspace.");
    auto& admission = *workspace.impl_;
    admission.check();
    struct Allocation
    {
        CellGeometryOperationWorkspace::Impl& admission;
        details::Work* metadata = nullptr;
        static void charge(void* state, std::size_t count, std::size_t width)
        {
            auto& self = *static_cast<Allocation*>(state);
            if (self.metadata)
            {
                self.metadata->charge(count, width);
                return;
            }
            const auto left = self.admission.limits.byte_budget - self.admission.bytes;
            if (width && count > left / width)
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Source view snapshot storage exhausted.");
            self.admission.bytes += count * width;
        }
        void add(std::size_t count, std::size_t width) { charge(this, count, width); }
    } allocation{admission};
    struct MetadataScope
    {
        Allocation& allocation;
        details::Work work;
        details::Work* previous;
        MetadataScope(Allocation& value, GeometryLimits limits)
            : allocation(value), work(limits), previous(details::active_work)
        {
            work.bytes = allocation.admission.bytes;
            work.operations = allocation.admission.operations;
            work.pairs = allocation.admission.pairs;
            allocation.metadata = &work;
            details::active_work = &work;
        }
        ~MetadataScope()
        {
            allocation.admission.bytes = work.bytes;
            allocation.admission.operations = work.operations;
            allocation.admission.pairs = work.pairs;
            allocation.metadata = nullptr;
            details::active_work = previous;
        }
    };
    try
    {
        for (const auto& request : requests)
        {
            if (!request.source_count || request.first_source > source_indices.size() ||
                request.source_count > source_indices.size() - request.first_source)
                throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid source view fan range.");
            if (request.source_count > admission.limits.maximum_faces)
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Source view fan exceeds its source-count cap.");
        }
        const auto reserve = [&](auto& values, std::size_t count)
        {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            if (count > values.max_size())
                throw GeometryError(GeometryErrorCode::ResourceLimit, "Source view batch vector limit exhausted.");
            allocation.add(count, sizeof(Value));
            values.reserve(count);
            if (values.capacity() > count)
                allocation.add(values.capacity() - count, sizeof(Value));
        };
        std::vector<PointProvenanceView> records;
        std::vector<Point<M>> result;
        reserve(records, requests.size());
        reserve(result, requests.size());
        PointSnapshotBuilder<M> builder({&allocation, Allocation::charge});
        GeometryAllocationLedger authorities;
        for (const auto& request : requests)
        {
            auto definition = [&]
            {
                // Ordinary face containment is the native membership authority
                // for both finite output modes, including hanging face/edge points.
                details::Operation query(workspace.remaining_limits(ordinary));
                struct RecordWork
                {
                    CellGeometryOperationWorkspace::Impl& admission;
                    details::Work& work;
                    ~RecordWork()
                    {
                        admission.bytes += work.bytes;
                        admission.operations += work.operations;
                        admission.pairs += work.pairs;
                    }
                } record{admission, query.work};
                if constexpr (M == ordinary)
                    ++details::statistics.source_imports;
                const auto native_limits = M == ordinary ? query.work.limits : GeometryLimits{};
                if (canonical_key_size(request.identity) > native_limits.byte_budget)
                {
                    if constexpr (M == ordinary)
                        details::reject(ordinary, "source point export requires a capped authenticated source fan.");
                    else
                        throw GeometryError(
                            GeometryErrorCode::ProvenanceMismatch, "Native point requires a capped source fan.");
                }
                // Validate the complete metadata fan before a potentially fallible
                // face predicate, preserving the native-input boundary's order.
                std::size_t faces = 0;
                const auto fan = source_indices.subspan(request.first_source, request.source_count);
                for (const auto index : fan)
                {
                    if (index >= sources.size())
                        throw GeometryError(GeometryErrorCode::InvalidInput, "Foreign source view slot.");
                    const auto source = sources[index];
                    auto source_limits = native_limits;
                    source_limits.arithmetic_mode = M;
                    source.validate_source_authority(source_limits);
                    if (source.num_faces() > native_limits.maximum_faces - faces)
                    {
                        if constexpr (M == ordinary)
                            details::reject(ordinary, "source point authority exceeds bootstrap storage/fan cap.");
                        else
                            throw GeometryError(
                                GeometryErrorCode::ResourceLimit, "Native source fan exceeds its existing caps.");
                    }
                    faces += source.num_faces();
                }
                details::NativePointDefinition<M> definition;
                if constexpr (M == ordinary)
                {
                    const auto coordinates = numerical_point<ordinary>(request.identity);
                    for (std::size_t axis = 0; axis < 3; ++axis)
                        definition.value[axis] = coordinates[axis].value;
                }
                bool membership = false;
                for (const auto index : fan)
                {
                    const auto source = sources[index];
                    for (std::size_t face = 0; face < source.num_faces(); ++face)
                    {
                        if (!source.contains_source_point(face, request.identity))
                            continue;
                        membership = true;
                        if constexpr (M == ordinary)
                        {
                            auto authority = details::original_support_authority(source.source_support(face));
                            query.work.allocations.retain(authority, query.work.bytes, query.work.limits.byte_budget);
                            if (definition.supports.size() == definition.supports.capacity())
                            {
                                const auto old = definition.supports.capacity();
                                if (old > definition.supports.max_size() / 2)
                                    throw GeometryError(
                                        GeometryErrorCode::ResourceLimit, "Source support capacity overflows.");
                                const auto next = old ? 2 * old : 1;
                                query.work.charge(next, sizeof(PlaneAuthority));
                                definition.supports.reserve(next);
                                if (definition.supports.capacity() > next)
                                    query.work.charge(definition.supports.capacity() - next, sizeof(PlaneAuthority));
                            }
                            definition.supports.push_back(std::move(authority));
                        }
                        else
                            break;
                    }
                    if constexpr (M != ordinary)
                        if (membership)
                            break;
                }
                if (!membership)
                {
                    if constexpr (M == ordinary)
                        details::reject(ordinary, "source point has no authenticated source-face membership.");
                    else
                        throw GeometryError(GeometryErrorCode::ProvenanceMismatch,
                            "Point identity is absent from its native source fan.");
                }
                std::sort(definition.supports.begin(), definition.supports.end());
                definition.supports.erase(
                    std::unique(definition.supports.begin(), definition.supports.end()), definition.supports.end());
                return definition;
            }();
            if constexpr (M == ArithmeticMode::BoundedCertification)
            {
                // Preserve the public native-input boundary: check Ordinary
                // membership first, then enter the bounded environment gate.
                details::Operation bounded_boundary(workspace.remaining_limits(M));
                ++details::statistics.source_imports;
                if (canonical_key_size(request.identity) > details::active_work->limits.byte_budget)
                    details::reject(M, "source point export requires a capped authenticated source fan.");
                std::size_t faces = 0;
                for (const auto index : source_indices.subspan(request.first_source, request.source_count))
                {
                    const auto source = sources[index];
                    try
                    {
                        source.validate_source_authority(details::active_work->limits);
                    }
                    catch (const GeometryError& error)
                    {
                        if (error.code() != GeometryErrorCode::ResourceLimit)
                            throw;
                        details::reject(M, "source point authority exceeds bootstrap storage/fan cap.");
                    }
                    if (source.num_faces() > details::active_work->limits.maximum_faces - faces)
                        details::reject(M, "source point authority exceeds bootstrap storage/fan cap.");
                    faces += source.num_faces();
                }
                const auto coordinates = numerical_point<ordinary>(request.identity);
                for (std::size_t axis = 0; axis < 3; ++axis)
                    definition.value[axis] = coordinates[axis].value;
            }
            auto metadata_limits = admission.limits;
            metadata_limits.arithmetic_mode = M;
            MetadataScope metadata(allocation, metadata_limits);
            allocation.add(definition.supports.capacity(), sizeof(PlaneAuthority));
            for (const auto& authority : definition.supports)
                authorities.retain(authority, metadata.work.bytes, admission.limits.byte_budget);
            records.push_back(builder.native_source(definition.value, definition.absolute_error,
                DynamicGeometryAccess::admit_incidence<M>(definition.supports)));
        }
        // Final freeze may require no additional bytes. Keep cumulative counters
        // in the metadata Work rather than treating an exactly spent budget as
        // a fresh operation whose zero remainder selects a default guard.
        auto metadata_limits = admission.limits;
        metadata_limits.arithmetic_mode = M;
        MetadataScope metadata(allocation, metadata_limits);
        const auto snapshot = builder.freeze();
        for (const auto record : records)
        {
            DynamicPoint point{record.value(), record.absolute_error(), M};
            DynamicGeometryAccess::attach_identity(point, record.owning());
            result.emplace_back(point);
        }
        return result;
    }
    catch (...)
    {
        admission.failed = true;
        throw;
    }
}
template std::vector<Point<ordinary>> source_points<ordinary>(const CellGeometryRange<ordinary>&,
    std::span<const std::size_t>, std::span<const NativeSourcePointRequest>, CellGeometryOperationWorkspace&);
template std::vector<Point<ArithmeticMode::BoundedCertification>> source_points<ArithmeticMode::BoundedCertification>(
    const CellGeometryRange<ordinary>&, std::span<const std::size_t>, std::span<const NativeSourcePointRequest>,
    CellGeometryOperationWorkspace&);

namespace details
{
namespace
{
struct RetainedFaceInput
{
    std::vector<ordinary_native::WidePoint> polygon;
    ordinary_native::WidePoint normal;
};
RetainedFaceInput retained_face_input(const ConvexCellGeometry<ordinary>& source, std::size_t face)
{
    const auto stored =
        checked_geometry_storage<ordinary>(DynamicGeometryAccess::implementation(*source.dynamic_geometry()));
    const auto loop = source.faces().at(face);
    active_work->charge(loop.size(), sizeof(ordinary_native::WidePoint));
    RetainedFaceInput result;
    result.polygon.reserve(loop.size());
    if (result.polygon.capacity() > loop.size())
        active_work->charge(result.polygon.capacity() - loop.size(), sizeof(ordinary_native::WidePoint));
    const OrdinaryExportCoordinates coordinates(source);
    for (const auto vertex : loop)
        result.polygon.push_back(coordinates.at(vertex));
    const auto& surface = stored->surfaces.at(face);
    for (std::size_t axis = 0; axis < 3; ++axis)
        result.normal[axis] = surface.orientation * surface.plane->normal[axis].value;
    return result;
}
template<class Query> auto with_retained_face(const ConvexCellGeometry<ordinary>& source, std::size_t face, Query query)
{
    const auto& stored = DynamicGeometryAccess::implementation(*source.dynamic_geometry());
    Operation operation(stored->limits);
    const auto input = retained_face_input(source, face);
    planar::validate_retained_loop(input.polygon, input.normal);
    return query(std::span<const ordinary_native::WidePoint>(input.polygon));
}
} // namespace
GeometryValue<ordinary> cell_view_face_area(const ConvexCellGeometry<ordinary>& source, std::size_t face)
{
    return with_retained_face(source, face, [](auto polygon) { return planar::area(polygon); });
}
GeometryPoint cell_view_face_centroid(const ConvexCellGeometry<ordinary>& source, std::size_t face)
{
    return with_retained_face(source, face, [](auto polygon) { return planar::centroid(polygon); });
}
GeometryPoint cell_view_face_area_vector(const ConvexCellGeometry<ordinary>& source, std::size_t face)
{
    return with_retained_face(source, face, [](auto polygon) { return planar::area_vector(polygon); });
}
int cell_view_face_orientation(const ConvexCellGeometry<ordinary>& source, std::size_t face)
{
    return with_retained_face(source, face, [](auto polygon) { return planar::orientation(polygon); });
}
void cell_view_validate_source_authority(const ConvexCellGeometry<ordinary>& source, GeometryLimits limits)
{
    const auto& stored = DynamicGeometryAccess::implementation(*source.dynamic_geometry());
    if (stored->numerical_only)
        throw GeometryError(
            GeometryErrorCode::ProvenanceMismatch, "Numerical-only geometry cannot issue native point authority.");
    if (source.storage_bytes() > limits.byte_budget)
    {
        if (limits.arithmetic_mode == ordinary)
            details::reject(ordinary, "source point authority exceeds bootstrap storage/fan cap.");
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Native point source fan exceeds its existing caps.");
    }
}
PlaneSupportIdentity<ordinary> cell_view_source_support(const ConvexCellGeometry<ordinary>& source, std::size_t face)
{
    const auto stored =
        checked_geometry_storage<ordinary>(DynamicGeometryAccess::implementation(*source.dynamic_geometry()));
    Operation operation(stored->limits);
    if (stored->surfaces.at(face).plane->native_support)
        return *stored->surfaces[face].plane->native_support;
    const auto input = retained_face_input(source, face);
    planar::validate_retained_loop(input.polygon, input.normal);
    const auto bound = ordinary_native::source_support_allocation_bound(input.polygon.size());
    if (bound > active_work->limits.byte_budget - active_work->bytes)
        throw GeometryError(
            GeometryErrorCode::ResourceLimit, "Fallback source support construction exceeds remaining storage.");
    auto support = ordinary_native::make_source_support(input.polygon);
    active_work->retain_identity(support);
    return support;
}
bool cell_view_contains_source_point(
    const ConvexCellGeometry<ordinary>& source, std::size_t face, const PointIdentity<ordinary>& identity)
{
    const auto& stored = DynamicGeometryAccess::implementation(*source.dynamic_geometry());
    // The batch boundary validates the complete fan before predicates. Do not
    // repeat its selected-mode storage test inside this Ordinary membership
    // query: bounded publication attributes that later failure to its own mode.
    if (stored->numerical_only)
        throw GeometryError(
            GeometryErrorCode::ProvenanceMismatch, "Numerical-only geometry cannot issue native point authority.");
    return with_retained_face(source, face,
        [&](auto polygon) { return planar::point_inside(polygon, GeometryIdentityAccess::point(identity)); });
}
std::size_t cell_view_box_face_vertex(std::size_t face, std::size_t vertex)
{
    return kernel::fixed_box_faces.at(face).at(vertex);
}
std::array<long double, 3> cell_view_corner(const ConvexCellGeometry<ordinary>& cell, std::size_t vertex)
{
    return OrdinaryExportCoordinates(cell).at(vertex);
}
} // namespace details
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
template class CellGeometryCatalog<ArithmeticMode::Ordinary>;
template class CellGeometryCatalog<ArithmeticMode::BoundedCertification>;
template class CellGeometryCatalog<ArithmeticMode::Exact>;
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
GeometryLimits CartesianCellSource::query_limits() const
{
    return impl_->limits;
}
details::NativeOriginalFace CartesianCellSource::original_face(std::size_t slot, std::size_t face) const
{
    if (face >= kernel::fixed_box_faces.size())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid original native face.");
    FrozenEnvironment environment(impl_->environment);
    details::Operation operation(impl_->limits);
    const auto box = impl_->evaluate(slot);
    const auto normal = kernel::represented_source_face_normal(box, kernel::fixed_box_faces[face]).normal;
    const auto& anchor = box.vertices[kernel::fixed_box_faces[face][0]].coordinates;
    details::NativeOriginalFace result;
    result.frame_origin = box.frame.origin;
    result.frame_exponent = box.frame.exponent;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        result.normal[axis] = normal[axis].value;
        result.anchor[axis] = anchor[axis].value;
    }
    return result;
}
void CartesianCellSource::visit_local_coordinates(
    std::size_t slot, void* context, void (*visitor)(void*, std::size_t, const PointCoordinates<ordinary>&)) const
{
    details::Operation operation(impl_->limits);
    const auto box = impl_->evaluate(slot);
    for (std::size_t index = 0; index < box.vertices.size(); ++index)
    {
        const auto coordinates = kernel::scaled<ordinary>(box.vertices[index].coordinates, box.frame.exponent) +
                                 (kernel::point<ordinary>(box.frame.origin) - kernel::point<ordinary>(origin(slot)));
        PointCoordinates<ordinary> point;
        for (std::size_t axis = 0; axis < 3; ++axis)
            point.value[axis] = kernel::bounded<ordinary>(coordinates[axis]).value;
        visitor(context, index, point);
    }
}
bool CartesianCellSource::face_contains(std::size_t slot, std::size_t face, const PlanarFace<ordinary>& other) const
{
    return with_fixed_face(*impl_, slot, face,
        [&](auto polygon)
        {
            return planar::contains(
                polygon, other.points(), source_support(slot, face) == planar_support<ordinary>(other));
        });
}
std::size_t CellGeometryOperationWorkspace::pair_tests_used() const noexcept
{
    return impl_->pairs;
}
void CellGeometryOperationWorkspace::restrict_remaining(GeometryLimits remaining)
{
    auto& state = *impl_;
    if (state.failed)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry operation workspace is exhausted.");
    const auto restrict = [](std::size_t absolute, std::size_t used, std::size_t left)
    {
        if (used > absolute)
            throw GeometryError(
                GeometryErrorCode::ResourceLimit, "Geometry operation workspace quota is inconsistent.");
        // Avoid addition overflow; a larger remainder cannot expand a prior cap.
        return left >= absolute - used ? absolute : used + left;
    };
    state.limits.byte_budget = restrict(state.limits.byte_budget, state.bytes, remaining.byte_budget);
    state.limits.maximum_operations =
        restrict(state.limits.maximum_operations, state.operations, remaining.maximum_operations);
    state.limits.maximum_pair_tests =
        restrict(state.limits.maximum_pair_tests, state.pairs, remaining.maximum_pair_tests);
    state.limits.maximum_vertices = std::min(state.limits.maximum_vertices, remaining.maximum_vertices);
    state.limits.maximum_faces = std::min(state.limits.maximum_faces, remaining.maximum_faces);
}
} // namespace SimpleFluid::PLIC
namespace SimpleFluid::PLIC::details
{
bool cell_view_face_contains(
    const ConvexCellGeometry<ordinary>& source, std::size_t face, const PlanarFace<ordinary>& other)
{
    const auto& stored = DynamicGeometryAccess::implementation(*source.dynamic_geometry());
    Operation operation(stored->limits);
    const auto support = cell_view_source_support(source, face);
    return with_retained_face(source, face, [&](auto polygon)
        { return planar::contains(polygon, other.points(), support == planar_support<ordinary>(other)); });
}
namespace
{
template<ArithmeticMode M> GeometryLimits view_query_limits(const CellGeometryView<M>& view)
{
    if constexpr (M == ordinary)
        if (view.is_procedural())
            return NativeCellGeometrySourceAccess::source_limits(NativeCellGeometrySourceAccess::source_cell(view));
    const auto geometry = NativeCellGeometrySourceAccess::existing_geometry(view);
    return DynamicGeometryAccess::implementation(*geometry->dynamic_geometry())->limits;
}
} // namespace
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
void visit_local_coordinates(
    const CellGeometryView<M>& view, void* context, void (*visitor)(void*, std::size_t, const PointCoordinates<M>&))
{
    if (!visitor)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Null local-coordinate visitor.");
    if constexpr (M == ordinary)
        if (view.is_procedural())
        {
            NativeCellGeometrySourceAccess::visit_local_coordinates(
                NativeCellGeometrySourceAccess::source_cell(view), context, visitor);
            return;
        }
    visit_local_coordinates<M>(*NativeCellGeometrySourceAccess::existing_geometry(view), context, visitor);
}
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
std::vector<PointCoordinates<M>> materialize_local_coordinates(const CellGeometryView<M>& view)
{
    Operation operation(view_query_limits(view));
    std::vector<PointCoordinates<M>> result;
    active_work->charge(view.vertex_count(), sizeof(PointCoordinates<M>));
    result.reserve(view.vertex_count());
    if (result.capacity() > view.vertex_count())
        active_work->charge(result.capacity() - view.vertex_count(), sizeof(PointCoordinates<M>));
    visit_local_coordinates<M>(view, &result, [](void* output, std::size_t, const PointCoordinates<M>& point)
        { static_cast<std::vector<PointCoordinates<M>>*>(output)->push_back(point); });
    return result;
}
template void visit_local_coordinates<ordinary>(
    const CellGeometryView<ordinary>&, void*, void (*)(void*, std::size_t, const PointCoordinates<ordinary>&));
template void visit_local_coordinates<ArithmeticMode::BoundedCertification>(
    const CellGeometryView<ArithmeticMode::BoundedCertification>&, void*,
    void (*)(void*, std::size_t, const PointCoordinates<ArithmeticMode::BoundedCertification>&));
template std::vector<PointCoordinates<ordinary>> materialize_local_coordinates(const CellGeometryView<ordinary>&);
template std::vector<PointCoordinates<ArithmeticMode::BoundedCertification>> materialize_local_coordinates(
    const CellGeometryView<ArithmeticMode::BoundedCertification>&);
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
class CellGeometryWorkspaceAccess
{
public:
    static CellGeometryOperationWorkspace::Impl& state(CellGeometryOperationWorkspace& workspace)
    {
        return *workspace.impl_;
    }
};
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
std::vector<PointCoordinates<M>> materialize_local_coordinates(
    const CellGeometryView<M>& view, CellGeometryOperationWorkspace& workspace)
{
    if (active_work)
        throw GeometryError(
            GeometryErrorCode::InvalidInput, "Coordinate workspace requires an outer operation boundary.");
    auto& state = CellGeometryWorkspaceAccess::state(workspace);
    const auto limits = workspace.remaining_limits(M);
    try
    {
        Operation operation(limits);
        struct Record
        {
            CellGeometryOperationWorkspace::Impl& state;
            Work& work;
            ~Record()
            {
                state.bytes += work.bytes;
                state.operations += work.operations;
                state.pairs += work.pairs;
            }
        } record{state, operation.work};
        return materialize_local_coordinates<M>(view);
    }
    catch (...)
    {
        state.failed = true;
        throw;
    }
}
template std::vector<PointCoordinates<ArithmeticMode::Ordinary>> materialize_local_coordinates(
    const CellGeometryView<ArithmeticMode::Ordinary>&, CellGeometryOperationWorkspace&);
template std::vector<PointCoordinates<ArithmeticMode::BoundedCertification>> materialize_local_coordinates(
    const CellGeometryView<ArithmeticMode::BoundedCertification>&, CellGeometryOperationWorkspace&);
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC
{
template<ArithmeticMode M>
    requires(M != ArithmeticMode::Exact)
std::vector<ConvexGeometryPtr<M>> materialize_cells(
    const CellGeometryRange<M>& range, CellGeometryOperationWorkspace& workspace)
{
    std::vector<ConvexGeometryPtr<M>> result;
    if (range.empty())
        return result;
    if (details::active_work)
        throw GeometryError(GeometryErrorCode::InvalidInput, "Cell batch materialization requires an outer workspace.");
    auto& state = details::CellGeometryWorkspaceAccess::state(workspace);
    try
    {
        state.check();
        workspace.charge_temporary_storage(range.size(), sizeof(ConvexGeometryPtr<M>));
        result.reserve(range.size());
        if (result.capacity() > range.size())
            workspace.charge_temporary_storage(result.capacity() - range.size(), sizeof(ConvexGeometryPtr<M>));
        for (const auto& view : range)
            result.push_back(materialize_cell(view, workspace));
        return result;
    }
    catch (...)
    {
        state.failed = true;
        throw;
    }
}
template std::vector<ConvexGeometryPtr<ArithmeticMode::Ordinary>> materialize_cells(
    const CellGeometryRange<ArithmeticMode::Ordinary>&, CellGeometryOperationWorkspace&);
template std::vector<ConvexGeometryPtr<ArithmeticMode::BoundedCertification>> materialize_cells(
    const CellGeometryRange<ArithmeticMode::BoundedCertification>&, CellGeometryOperationWorkspace&);
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
void CellGeometryOperationWorkspace::charge_temporary_storage(std::size_t count, std::size_t width)
{
    auto& state = *impl_;
    if (state.failed)
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry operation workspace is exhausted.");
    if (state.bytes > state.limits.byte_budget || (width && count > (state.limits.byte_budget - state.bytes) / width))
    {
        state.failed = true;
        throw GeometryError(GeometryErrorCode::ResourceLimit, "Geometry temporary storage quota exhausted.");
    }
    state.bytes += count * width;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
details::NativeFramedBounds CartesianCellSource::framed_bounds(std::size_t slot) const
{
    FrozenEnvironment environment(impl_->environment);
    details::Operation operation(impl_->limits);
    const auto box = impl_->evaluate(slot);
    details::NativeFramedBounds result;
    result.frame_origin = box.frame.origin;
    result.frame_exponent = box.frame.exponent;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        auto low = box.vertices.front().coordinates[axis].value;
        auto high = low;
        for (const auto& vertex : box.vertices)
        {
            low = std::min(low, vertex.coordinates[axis].value);
            high = std::max(high, vertex.coordinates[axis].value);
        }
        result.bounds[axis] = {low, high};
    }
    return result;
}
bool details::NativeCellGeometrySourceAccess::state_compatible(
    GeometryPoint lower, GeometryPoint upper, GeometryLimits requested)
{
    if (!certification_environment())
        return false;
    details::Operation operation(ordinary_limits(requested));
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || !(upper[axis] > lower[axis]))
            return false;
    const auto corners = kernel::fixed_box_corners(lower, upper);
    const kernel::FixedBoxEvaluation raw(lower, upper);
    const auto equal = [](long double a, long double b)
    { return a == b && (a != 0 || std::signbit(a) == std::signbit(b)); };
    std::array<kernel::Point, 8> imported;
    for (std::size_t vertex = 0; vertex < corners.size(); ++vertex)
    {
        const auto relative = kernel::represented_source_relative(corners[vertex], corners.front());
        const auto local =
            kernel::bounded<ordinary>(kernel::scaled<ordinary>(raw.vertices[vertex].coordinates, raw.frame.exponent) +
                                      (kernel::point<ordinary>(raw.frame.origin) - kernel::point<ordinary>(lower)));
        imported[vertex] = kernel::point<ordinary>(local);
        const auto world = raw.world_corner(vertex);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            if (kernel::source_witness::sum_gap(
                    corners[vertex][axis], -static_cast<long double>(lower[axis]), relative.value[axis]) != 0 ||
                kernel::source_witness::scale_gap(
                    relative.value[axis], -raw.frame.exponent, raw.vertices[vertex].coordinates[axis].value) != 0 ||
                !equal(world[axis], corners[vertex][axis]) || !equal(local.value[axis], relative.value[axis]))
                return false;
        }
    }
    const auto frame = kernel::represented_source_frame(lower, imported);
    if (frame.exponent != raw.frame.exponent)
        return false;
    for (std::size_t vertex = 0; vertex < corners.size(); ++vertex)
    {
        const auto normalized = kernel::scaled<ordinary>(imported[vertex], -frame.exponent);
        for (std::size_t axis = 0; axis < 3; ++axis)
            if (!equal(normalized[axis].value, raw.vertices[vertex].coordinates[axis].value) ||
                kernel::source_witness::scale_gap(
                    imported[vertex][axis].value, -frame.exponent, normalized[axis].value) != 0)
                return false;
    }
    return true;
}
} // namespace SimpleFluid::PLIC
