#include "geometry/PLIC/details/GeometryKernels.hh"

#include "geometry/PLIC/details/bounded/ConvexGeometry.tcc"
#include "geometry/PLIC/details/ordinary/ConvexGeometry.tcc"

namespace SimpleFluid::PLIC
{
const std::vector<PlaneAuthority>& DynamicPoint::supports() const
{
    static const std::vector<PlaneAuthority> empty;
    if (!provenance_ || provenance_->value != value || provenance_->absolute_error != absolute_error ||
        provenance_->arithmetic_mode != arithmetic_mode)
        return empty;
    return provenance_->supports;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC::details
{
thread_local GeometryStatistics statistics;

[[noreturn]] void reject(ArithmeticMode mode, std::string_view reason)
{
    ++statistics.rejections;
    const char* prefix = mode == ArithmeticMode::Ordinary               ? "Ordinary FP geometry: "
                         : mode == ArithmeticMode::BoundedCertification ? "Bounded FP geometry: "
                                                                        : "Finite FP geometry: ";
    throw std::invalid_argument(std::string(prefix) + std::string(reason));
}

[[noreturn]] void reject(std::string_view reason)
{
    reject(active_work ? active_work->limits.arithmetic_mode : ArithmeticMode::Exact, reason);
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
thread_local Work* active_work = nullptr;
}

namespace SimpleFluid::PLIC::details
{
GeometryLimits mode_limits(ArithmeticMode mode)
{
    GeometryLimits result;
    result.arithmetic_mode = mode;
    return result;
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
void require_mode(ArithmeticMode a, ArithmeticMode b)
{
    if (a == ArithmeticMode::Exact || b == ArithmeticMode::Exact || a != b)
        reject(a, "incompatible arithmetic modes in finite geometry.");
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
void operation()
{
    ++statistics.floating_operations;
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC
{
std::shared_ptr<const DynamicPointProvenance> DynamicGeometryAccess::identity(const DynamicPoint& p)
{
    return p.provenance_ && p.provenance_->value == p.value && p.provenance_->absolute_error == p.absolute_error &&
                   p.provenance_->arithmetic_mode == p.arithmetic_mode
               ? p.provenance_
               : nullptr;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
std::size_t DynamicGeometryAccess::identity_bytes(const std::shared_ptr<const DynamicPointProvenance>& p)
{
    if (!p)
        return 0;
    std::size_t bytes = sizeof(DynamicPointProvenance) + p->supports.capacity() * sizeof(PlaneAuthority);
    for (const auto& key : p->supports)
        bytes += key.key.capacity() * sizeof(std::uint64_t);
    return bytes;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
std::size_t DynamicGeometryAccess::storage_bytes(const DynamicPoint& p)
{
    std::size_t bytes = sizeof(DynamicPoint);
    if (p.provenance_)
    {
        bytes += sizeof(DynamicPointProvenance) + p.provenance_->supports.capacity() * sizeof(PlaneAuthority);
        for (const auto& key : p.provenance_->supports)
            bytes += key.key.capacity() * sizeof(std::uint64_t);
    }
    return bytes;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
bool DynamicGeometryAccess::same_source(const DynamicPoint& a, const DynamicPoint& b)
{
    return a.provenance_ && a.provenance_ == b.provenance_ && a.provenance_->value == a.value &&
           a.provenance_->absolute_error == a.absolute_error && b.provenance_->value == b.value &&
           b.provenance_->absolute_error == b.absolute_error && a.arithmetic_mode == a.provenance_->arithmetic_mode &&
           b.arithmetic_mode == a.arithmetic_mode;
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
void DynamicGeometryAccess::seal(DynamicPoint& point, std::vector<PlaneAuthority> supports)
{
    std::sort(supports.begin(), supports.end());
    supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
    point.provenance_ = std::make_shared<DynamicPointProvenance>(
        DynamicPointProvenance{point.value, point.absolute_error, std::move(supports), point.arithmetic_mode});
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC::details
{
Work::Work(GeometryLimits requested) : limits(requested)
{
    operation_limit = limits.maximum_operations;
    if (!operation_limit)
    {
        // Admission visits O(F*V^2) face turns; an intersection performs
        // at most F complementary clips, each O(F*V). This independent
        // checked guard bounds their scalar interval work. It is not a
        // reinterpretation or increase of any public sweep/trajectory cap.
        const auto checked_add = [this](std::size_t a, std::size_t b)
        {
            if (b > std::numeric_limits<std::size_t>::max() - a)
                reject(limits.arithmetic_mode, "derived scalar-work guard overflow.");
            return a + b;
        };
        const auto checked_multiply = [this](std::size_t a, std::size_t b)
        {
            if (b && a > std::numeric_limits<std::size_t>::max() / b)
                reject(limits.arithmetic_mode, "derived scalar-work guard overflow.");
            return a * b;
        };
        const auto v = limits.maximum_vertices, f = limits.maximum_faces;
        operation_limit = checked_add(
            1024, checked_multiply(256, checked_multiply(v, checked_multiply(f, checked_add(checked_add(v, f), 1)))));
    }
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
void Work::charge(std::size_t count, std::size_t size)
{
    if (size && count > (limits.byte_budget - bytes) / size)
        reject(limits.arithmetic_mode, "construction storage cap exhausted.");
    bytes += count * size;
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
void Work::pair()
{
    if (++pairs > limits.maximum_pair_tests)
        reject(limits.arithmetic_mode, "pair-test cap exhausted.");
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
Operation::Operation(GeometryLimits limits) : work(limits), previous(active_work)
{
    if (limits.arithmetic_mode != ArithmeticMode::Ordinary &&
        limits.arithmetic_mode != ArithmeticMode::BoundedCertification)
        reject(limits.arithmetic_mode, "Exact arithmetic is not a finite geometry mode.");
    if (previous)
        require_mode(previous->limits.arithmetic_mode, limits.arithmetic_mode);
    if (limits.arithmetic_mode == ArithmeticMode::BoundedCertification && !certification_environment())
        reject(limits.arithmetic_mode, "unsupported floating-point environment; no exact fallback.");
    if (!limits.maximum_vertices || !limits.maximum_faces || !limits.maximum_pair_tests || !limits.byte_budget)
        reject(limits.arithmetic_mode, "zero resource cap.");
    if (!previous)
    {
        active_work = &work;
        ++statistics.wider_routes;
    }
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC
{
void DynamicGeometryAccess::seal_represented_source(DynamicPoint& point)
{
    auto supports = point.supports();
    const bool represented =
        point.arithmetic_mode == ArithmeticMode::Ordinary && point.absolute_error == std::array<long double, 3>{};
    point.provenance_ = std::make_shared<DynamicPointProvenance>(DynamicPointProvenance{
        point.value, point.absolute_error, std::move(supports), point.arithmetic_mode, represented});
}
bool DynamicGeometryAccess::same_represented_source(
    const std::shared_ptr<const DynamicPointProvenance>& a, const std::shared_ptr<const DynamicPointProvenance>& b)
{
    // Native source coordinates are themselves the represented authority.
    // This comparison never promotes an arbitrary rounded construction token.
    return a && b &&
           (a == b || (a->represented_native_source && b->represented_native_source &&
                          a->arithmetic_mode == ArithmeticMode::Ordinary &&
                          b->arithmetic_mode == ArithmeticMode::Ordinary && a->value == b->value));
}
} // namespace SimpleFluid::PLIC
