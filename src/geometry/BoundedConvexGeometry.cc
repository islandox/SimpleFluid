#include "geometry/details/BoundedGeometryKernels.hh"

namespace SimpleFluid::Meshes
{
const std::vector<BoundedPlaneAuthority>& BoundedPoint::supports() const
{
    static const std::vector<BoundedPlaneAuthority> empty;
    if (!provenance_ || provenance_->value != value || provenance_->absolute_error != absolute_error ||
        provenance_->arithmetic_mode != arithmetic_mode)
        return empty;
    return provenance_->supports;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
thread_local BoundedGeometryStatistics statistics;

[[noreturn]] void reject(const char* reason)
{
    ++statistics.rejections;
    throw std::invalid_argument(std::string("Bounded FP geometry: ") + reason);
}
} // namespace SimpleFluid::Meshes::bounded_geometry_detail

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
thread_local Work* active_work = nullptr;
}

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
BoundedGeometryLimits mode_limits(ArithmeticMode mode)
{
    BoundedGeometryLimits result;
    result.arithmetic_mode = mode;
    return result;
}
} // namespace SimpleFluid::Meshes::bounded_geometry_detail

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
void require_mode(ArithmeticMode a, ArithmeticMode b)
{
    if (a == ArithmeticMode::Exact || b == ArithmeticMode::Exact || a != b)
        reject("incompatible arithmetic modes in finite geometry.");
}
} // namespace SimpleFluid::Meshes::bounded_geometry_detail

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
void operation()
{
    ++statistics.floating_operations;
}
} // namespace SimpleFluid::Meshes::bounded_geometry_detail

namespace SimpleFluid::Meshes
{
std::shared_ptr<const BoundedPointProvenance> BoundedGeometryAccess::identity(const BoundedPoint& p)
{
    return p.provenance_ && p.provenance_->value == p.value && p.provenance_->absolute_error == p.absolute_error &&
                   p.provenance_->arithmetic_mode == p.arithmetic_mode
               ? p.provenance_
               : nullptr;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
std::size_t BoundedGeometryAccess::identity_bytes(const std::shared_ptr<const BoundedPointProvenance>& p)
{
    if (!p)
        return 0;
    std::size_t bytes = sizeof(BoundedPointProvenance) + p->supports.capacity() * sizeof(BoundedPlaneAuthority);
    for (const auto& key : p->supports)
        bytes += key.key.capacity() * sizeof(std::uint64_t);
    return bytes;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
std::size_t BoundedGeometryAccess::storage_bytes(const BoundedPoint& p)
{
    std::size_t bytes = sizeof(BoundedPoint);
    if (p.provenance_)
    {
        bytes += sizeof(BoundedPointProvenance) + p.provenance_->supports.capacity() * sizeof(BoundedPlaneAuthority);
        for (const auto& key : p.provenance_->supports)
            bytes += key.key.capacity() * sizeof(std::uint64_t);
    }
    return bytes;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
bool BoundedGeometryAccess::same_source(const BoundedPoint& a, const BoundedPoint& b)
{
    return a.provenance_ && a.provenance_ == b.provenance_ && a.provenance_->value == a.value &&
           a.provenance_->absolute_error == a.absolute_error && b.provenance_->value == b.value &&
           b.provenance_->absolute_error == b.absolute_error && a.arithmetic_mode == a.provenance_->arithmetic_mode &&
           b.arithmetic_mode == a.arithmetic_mode;
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes
{
void BoundedGeometryAccess::seal(BoundedPoint& point, std::vector<BoundedPlaneAuthority> supports)
{
    std::sort(supports.begin(), supports.end());
    supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
    point.provenance_ = std::make_shared<BoundedPointProvenance>(
        BoundedPointProvenance{point.value, point.absolute_error, std::move(supports), point.arithmetic_mode});
}
} // namespace SimpleFluid::Meshes

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
Work::Work(BoundedGeometryLimits requested) : limits(requested)
{
    operation_limit = limits.maximum_operations;
    if (!operation_limit)
    {
        // Admission visits O(F*V^2) face turns; an intersection performs
        // at most F complementary clips, each O(F*V). This independent
        // checked guard bounds their scalar interval work. It is not a
        // reinterpretation or increase of any public sweep/trajectory cap.
        const auto checked_add = [](std::size_t a, std::size_t b)
        {
            if (b > std::numeric_limits<std::size_t>::max() - a)
                reject("derived scalar-work guard overflow.");
            return a + b;
        };
        const auto checked_multiply = [](std::size_t a, std::size_t b)
        {
            if (b && a > std::numeric_limits<std::size_t>::max() / b)
                reject("derived scalar-work guard overflow.");
            return a * b;
        };
        const auto v = limits.maximum_vertices, f = limits.maximum_faces;
        operation_limit = checked_add(
            1024, checked_multiply(256, checked_multiply(v, checked_multiply(f, checked_add(checked_add(v, f), 1)))));
    }
}
} // namespace SimpleFluid::Meshes::bounded_geometry_detail

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
void Work::charge(std::size_t count, std::size_t size)
{
    if (size && count > (limits.byte_budget - bytes) / size)
        reject("construction storage cap exhausted.");
    bytes += count * size;
}
} // namespace SimpleFluid::Meshes::bounded_geometry_detail

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
void Work::pair()
{
    if (++pairs > limits.maximum_pair_tests)
        reject("pair-test cap exhausted.");
}
} // namespace SimpleFluid::Meshes::bounded_geometry_detail

namespace SimpleFluid::Meshes::bounded_geometry_detail
{
Operation::Operation(BoundedGeometryLimits limits) : work(limits), previous(active_work)
{
    if (limits.arithmetic_mode != ArithmeticMode::Ordinary &&
        limits.arithmetic_mode != ArithmeticMode::BoundedCertification)
        reject("Exact arithmetic is not a finite geometry mode.");
    if (previous)
        require_mode(previous->limits.arithmetic_mode, limits.arithmetic_mode);
    if (limits.arithmetic_mode == ArithmeticMode::BoundedCertification && !bounded_geometry_environment())
        reject("unsupported floating-point environment; no exact fallback.");
    if (!limits.maximum_vertices || !limits.maximum_faces || !limits.maximum_pair_tests || !limits.byte_budget)
        reject("zero resource cap.");
    if (!previous)
    {
        active_work = &work;
        ++statistics.wider_routes;
    }
}
} // namespace SimpleFluid::Meshes::bounded_geometry_detail

#include "geometry/details/ordinary/BoundedConvexGeometry.tcc"
#include "geometry/details/BoundedConvexGeometry.tcc"
