#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/SupportClassification.hh"
#include "geometry/PLIC/details/CanonicalExportCoordinates.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"
#include <cassert>

#include "geometry/PLIC/details/bounded/ConvexGeometry.tcc"
#include "geometry/PLIC/details/ordinary/ConvexGeometry.tcc"

namespace SimpleFluid::PLIC
{
void details::charge_geometry_identity_scratch(std::size_t count, std::size_t bytes)
{
    if (active_work)
        active_work->charge(count, bytes);
}

bool has_support(const DynamicPoint& point, const PlaneAuthority& authority)
{
    if (authority.empty())
        return false;
    const auto& supports = point.supports();
    return std::find(supports.begin(), supports.end(), authority) != supports.end();
}

bool has_any_support(const DynamicPoint& point, std::span<const PlaneAuthority> authorities)
{
    return std::any_of(
        authorities.begin(), authorities.end(), [&](const auto& authority) { return has_support(point, authority); });
}

PlaneAuthority PlaneAuthority::opaque(std::vector<std::uint64_t> words)
{
    if (words.empty())
        return {};
    if (static_cast<std::uint8_t>(words.front() >> 56) == static_cast<std::uint8_t>(AuthorityKeyTag::OriginalSupport))
        throw GeometryError(
            GeometryErrorCode::ProvenanceMismatch, "Original support words require authenticated source authority.");
    PlaneAuthority result;
    result.storage_ = std::make_shared<const std::vector<std::uint64_t>>(std::move(words));
    return result;
}
bool PlaneAuthority::empty() const noexcept
{
    return std::holds_alternative<std::monostate>(storage_);
}
std::size_t PlaneAuthority::word_count() const noexcept
{
    if (const auto* native = std::get_if<Native>(&storage_))
        return 1 + (canonical_key_size(*native) + 7) / 8;
    if (const auto* opaque = std::get_if<Opaque>(&storage_))
        return (*opaque)->size();
    return 0;
}
std::uint64_t PlaneAuthority::word(std::size_t index) const noexcept
{
    assert(index < word_count());
    if (const auto* native = std::get_if<Native>(&storage_))
        return index == 0 ? original_support_key_header(canonical_key_size(*native))
                          : details::GeometryIdentityAccess::canonical_word_at(*native, 8 * (index - 1));
    return (*std::get<Opaque>(storage_))[index];
}
namespace
{
class AuthorityWords
{
public:
    explicit AuthorityWords(const PlaneAuthority& authority) : authority_(authority)
    {
        if (const auto* support = details::OriginalSupportAuthorityAccess::retained(authority))
            stream_.emplace(details::GeometryIdentityAccess::backing(*support));
    }
    std::uint64_t next()
    {
        const auto index = next_++;
        return stream_ && index ? stream_->next_word() : authority_.word(index);
    }

private:
    const PlaneAuthority& authority_;
    std::size_t next_ = 0;
    std::optional<details::GeometryIdentityCanonicalCursor> stream_;
};
} // namespace
void PlaneAuthority::emit_words(void* context, void (*emit)(void*, std::uint64_t)) const
{
    AuthorityWords stream(*this);
    for (std::size_t i = 0; i < word_count(); ++i)
        emit(context, stream.next());
}
std::vector<std::uint64_t> PlaneAuthority::words() const
{
    std::vector<std::uint64_t> result;
    details::charge_geometry_identity_scratch(word_count(), sizeof(std::uint64_t));
    result.reserve(word_count());
    details::charge_geometry_identity_scratch(result.capacity() - word_count(), sizeof(std::uint64_t));
    emit_words(&result, [](void* context, std::uint64_t value)
        { static_cast<std::vector<std::uint64_t>*>(context)->push_back(value); });
    return result;
}
std::strong_ordering PlaneAuthority::compare_words(std::span<const std::uint64_t> other) const
{
    const auto size = word_count();
    if (!size || other.empty())
        return size <=> other.size();
    if (const auto header = word(0) <=> other.front(); header != 0)
        return header;
    AuthorityWords stream(*this);
    (void) stream.next();
    for (std::size_t i = 1; i < std::min(size, other.size()); ++i)
        if (const auto order = stream.next() <=> other[i]; order != 0)
            return order;
    return size <=> other.size();
}
bool PlaneAuthority::operator==(const PlaneAuthority& other) const noexcept
{
    if (const auto* first = std::get_if<Native>(&storage_))
        if (const auto* second = std::get_if<Native>(&other.storage_))
            return *first == *second;
    if (const auto* first = std::get_if<Opaque>(&storage_))
        if (const auto* second = std::get_if<Opaque>(&other.storage_))
            return *first == *second || **first == **second;
    return empty() && other.empty();
}
std::strong_ordering PlaneAuthority::operator<=>(const PlaneAuthority& other) const
{
    if (*this == other)
        return std::strong_ordering::equal;
    const auto first = word_count(), second = other.word_count();
    if (!first || !second)
        return first <=> second;
    if (const auto header = word(0) <=> other.word(0); header != 0)
        return header;
    AuthorityWords a(*this), b(other);
    (void) a.next();
    (void) b.next();
    for (std::size_t i = 1; i < std::min(first, second); ++i)
        if (const auto order = a.next() <=> b.next(); order != 0)
            return order;
    return first <=> second;
}
std::shared_ptr<const void> PlaneAuthority::opaque_storage_owner() const noexcept
{
    const auto* value = std::get_if<Opaque>(&storage_);
    return value ? *value : std::shared_ptr<const void>{};
}
std::size_t PlaneAuthority::opaque_payload_bytes() const noexcept
{
    const auto* value = std::get_if<Opaque>(&storage_);
    return value ? sizeof(std::vector<std::uint64_t>) + (*value)->capacity() * sizeof(std::uint64_t) : 0;
}

namespace
{
void accept_authorities(const std::vector<PlaneAuthority>& authorities)
{
    for (const auto& authority : authorities)
    {
        details::OriginalSupportAuthorityAccess::validate_boundary(authority);
        if (details::active_work)
            details::active_work->retain_identity(authority);
    }
}
} // namespace
PlaneAuthority details::OriginalSupportAuthorityAccess::issue(
    const PlaneSupportIdentity<ArithmeticMode::Ordinary>& support)
{
    const auto bytes = canonical_key_size(support);
    if (!bytes)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Empty immutable source support.");
    if (bytes > authority_key_length_mask)
        throw GeometryError(
            GeometryErrorCode::ResourceLimit, "Original support key length exceeds its tagged namespace.");
    return PlaneAuthority(support);
}
PlaneAuthority details::OriginalSupportAuthorityAccess::issue(
    const PlaneSupportIdentity<ArithmeticMode::Exact>& support)
{
    return issue(GeometryIdentityAccess::retained_exact_support(support));
}
const PlaneSupportIdentity<ArithmeticMode::Ordinary>* details::OriginalSupportAuthorityAccess::retained(
    const PlaneAuthority& authority) noexcept
{
    return std::get_if<PlaneAuthority::Native>(&authority.storage_);
}
const PlaneSupportIdentity<ArithmeticMode::Ordinary>* details::OriginalSupportAuthorityAccess::get(
    const PlaneAuthority& authority)
{
    return retained(authority);
}
void details::OriginalSupportAuthorityAccess::validate_boundary(const PlaneAuthority& authority)
{
    if (!authority.empty() &&
        static_cast<std::uint8_t>(authority.word(0) >> 56) ==
            static_cast<std::uint8_t>(AuthorityKeyTag::OriginalSupport) &&
        !retained(authority))
        throw GeometryError(
            GeometryErrorCode::ProvenanceMismatch, "Original source authority requires typed immutable support.");
}
void details::OriginalSupportAuthorityAccess::merge(PlaneAuthority& destination, const PlaneAuthority& source)
{
    if (destination != source)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Cannot merge different source authority keys.");
    validate_boundary(destination);
    validate_boundary(source);
}
PlaneAuthority details::OriginalSupportAuthorityAccess::authenticate(
    std::span<const std::uint64_t> words, const PlaneAuthorityRegistry& registry)
{
    if (words.empty())
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Empty serialized support authority.");
    const auto found = registry.find(words);
    if (found == registry.end())
        throw GeometryError(
            GeometryErrorCode::ProvenanceMismatch, "Foreign or absent original source support authority.");
    validate_boundary(*found);
    return *found;
}

PlaneAuthority original_support_authority(const PlanarFace<ArithmeticMode::Ordinary>& face)
{
    return details::original_support_authority(planar_support<ArithmeticMode::Ordinary>(face));
}

OrdinaryBoundary::OrdinaryBoundary(FaceConstraint<ArithmeticMode::Ordinary> boundary) : boundary_(std::move(boundary))
{
    if (!boundary_.geometry || boundary_.face >= boundary_.geometry->num_faces())
        throw GeometryError(GeometryErrorCode::InvalidInput, "Invalid Ordinary original boundary constraint.");
    const auto storage = details::checked_geometry_storage<ArithmeticMode::Ordinary>(
        DynamicGeometryAccess::implementation(*boundary_.geometry->dynamic_geometry()));
    const auto& plane = *storage->surfaces.at(boundary_.face).plane;
    if (plane.authorities.empty() && plane.native_support.has_value())
        native_authority_ = details::original_support_authority(*plane.native_support);
}

std::span<const PlaneAuthority> OrdinaryBoundary::authorities() const
{
    const auto storage = details::checked_geometry_storage<ArithmeticMode::Ordinary>(
        DynamicGeometryAccess::implementation(*boundary_.geometry->dynamic_geometry()));
    const auto& keys = storage->surfaces.at(boundary_.face).plane->authorities;
    if (keys.empty() && !native_authority_.empty())
        return {&native_authority_, 1};
    return keys;
}

OrdinaryBoundaryClass OrdinaryBoundary::classify(const Point<ArithmeticMode::Ordinary>& point) const
{
    namespace predicates = details::ordinary_predicates;
    const auto storage = details::checked_geometry_storage<ArithmeticMode::Ordinary>(
        DynamicGeometryAccess::implementation(*boundary_.geometry->dynamic_geometry()));
    const auto& surface = storage->surfaces.at(boundary_.face);
    const auto& plane = *surface.plane;
    if (has_any_support(point.dynamic(), authorities()))
        return OrdinaryBoundaryClass::ConstructionContact;
    std::array<long double, 3> normal{}, anchor{};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        normal[axis] = plane.normal[axis].value;
        anchor[axis] = plane.anchor[axis].value;
    }
    predicates::Result distance;
    try
    {
        distance = predicates::plane_distance(predicates::FramedPoint{point.value, {}, 0},
            predicates::FramedPoint{anchor, plane.frame.origin, plane.frame.exponent}, normal, plane.offset.value);
    }
    catch (const GeometryError& error)
    {
        if (error.code() == GeometryErrorCode::UnresolvedArithmetic)
            return OrdinaryBoundaryClass::Unresolved;
        throw;
    }
    if (distance.sign == predicates::Sign::Unresolved)
        return OrdinaryBoundaryClass::Unresolved;
    if (distance.sign == predicates::Sign::Zero)
        return OrdinaryBoundaryClass::CoordinateContact;
    return static_cast<int>(distance.sign) * surface.orientation < 0 ? OrdinaryBoundaryClass::Inside
                                                                     : OrdinaryBoundaryClass::Outside;
}

const std::vector<PlaneAuthority>& DynamicPoint::supports() const
{
    static const std::vector<PlaneAuthority> empty;
    if (!provenance_ || provenance_->value != value || provenance_->absolute_error != absolute_error ||
        provenance_->arithmetic_mode != arithmetic_mode)
        return empty;
    return provenance_->supports();
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC::details
{
constinit thread_local GeometryStatistics statistics;

[[noreturn]] void resource_failure(ArithmeticMode mode, std::string_view reason)
{
    ++statistics.rejections;
    const char* prefix = mode == ArithmeticMode::Ordinary ? "Ordinary FP geometry: " : "Bounded FP geometry: ";
    throw GeometryError(GeometryErrorCode::ResourceLimit, std::string(prefix) + std::string(reason));
}

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
constinit thread_local Work* active_work = nullptr;
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
std::size_t DynamicGeometryAccess::identity_bytes(
    const std::shared_ptr<const DynamicPointProvenance>& p, bool include_support_payloads)
{
    if (!p)
        return 0;
    std::size_t bytes = 0, retained_bytes = 0;
    GeometryAllocationLedger attachments;
    std::set<const DynamicPointProvenance*> visited;
    std::set<const DynamicPointSupportPayload*> visited_supports;
    std::vector<std::shared_ptr<const DynamicPointProvenance>> pending{p};
    for (std::size_t i = 0; i < pending.size(); ++i)
    {
        const auto point = pending[i];
        if (!point || !visited.insert(point.get()).second)
            continue;
        bytes += sizeof(DynamicPointProvenance) + point->source_edges.capacity() * sizeof(point->source_edges.front());
        if (const auto& payload = point->support_payload; payload && visited_supports.insert(payload.get()).second)
        {
            bytes += sizeof(DynamicPointSupportPayload) + payload->authorities.capacity() * sizeof(PlaneAuthority);
            for (const auto& key : payload->authorities)
            {
                if (include_support_payloads)
                    attachments.retain(key, retained_bytes, std::numeric_limits<std::size_t>::max());
            }
        }
        if (point->source_origin)
            pending.push_back(point->source_origin);
        for (const auto& edge : point->source_edges)
        {
            pending.push_back(edge[0]);
            pending.push_back(edge[1]);
        }
    }
    return bytes + attachments.payload_bytes();
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC
{
void DynamicGeometryAccess::retain_supports(const std::shared_ptr<const DynamicPointProvenance>& root,
    GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    if (!root)
        return;
    std::set<const DynamicPointProvenance*> visited;
    std::vector<std::shared_ptr<const DynamicPointProvenance>> pending{root};
    for (std::size_t i = 0; i < pending.size(); ++i)
    {
        const auto point = pending[i];
        if (!point || !visited.insert(point.get()).second)
            continue;
        for (const auto& authority : point->supports())
            ledger.retain(authority, used, limit);
        if (point->source_origin)
            pending.push_back(point->source_origin);
        for (const auto& edge : point->source_edges)
        {
            pending.push_back(edge[0]);
            pending.push_back(edge[1]);
        }
    }
}

std::size_t DynamicGeometryAccess::storage_bytes(const DynamicPoint& p, bool include_support_payloads)
{
    std::size_t bytes = sizeof(DynamicPoint);
    if (p.provenance_)
    {
        bytes += identity_bytes(p.provenance_, include_support_payloads);
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
namespace
{
std::shared_ptr<const DynamicPointProvenance> point_provenance(const DynamicPoint& point,
    std::shared_ptr<const DynamicPointSupportPayload> payload,
    const std::shared_ptr<const DynamicPointProvenance>& source, bool export_seal, bool numerical_only)
{
    auto record = std::make_shared<DynamicPointProvenance>(
        DynamicPointProvenance{point.value, point.absolute_error, std::move(payload), point.arithmetic_mode});
    if (export_seal)
    {
        record->numerical_only = numerical_only || (source && source->numerical_only);
        record->constructed = record->numerical_only || !source || source->constructed ||
                              source->value != point.value || source->absolute_error != point.absolute_error;
        if (!record->numerical_only)
            record->source_origin = source;
    }
    return record;
}

std::shared_ptr<const DynamicPointSupportPayload> support_payload(std::vector<PlaneAuthority> supports)
{
    if (supports.empty())
        return {};
    if (details::active_work)
        details::active_work->charge(1, sizeof(DynamicPointSupportPayload));
    if (supports.capacity() > supports.size())
    {
        // Account the replacement while the old outer buffer remains alive.
        // Moving authority handles preserves their immutable shared payloads.
        if (details::active_work)
            details::active_work->charge(supports.size(), sizeof(PlaneAuthority));
        std::vector<PlaneAuthority> compact(supports.size());
        std::move(supports.begin(), supports.end(), compact.begin());
        supports = std::move(compact);
    }
    return std::make_shared<DynamicPointSupportPayload>(std::move(supports));
}
} // namespace

void DynamicGeometryAccess::seal(DynamicPoint& point, std::vector<PlaneAuthority> supports)
{
    accept_authorities(supports);
    std::sort(supports.begin(), supports.end());
    supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
    point.provenance_ = point_provenance(point, support_payload(std::move(supports)), {}, false, false);
}

const std::shared_ptr<const DynamicPointProvenance>& DynamicGeometryAccess::borrowed_original_source(
    const std::shared_ptr<const DynamicPointProvenance>& point)
{
    static const std::shared_ptr<const DynamicPointProvenance> empty;
    // The caller's owner pins every immutable source_origin slot. Packet replay
    // validates dependency order before any such links are created.
    const auto* current = &point;
    while (*current && (*current)->source_origin)
    {
        if ((*current)->numerical_only || (*current)->source_origin->arithmetic_mode != (*current)->arithmetic_mode)
            return empty;
        if (active_work)
            active_work->pair();
        current = &(*current)->source_origin;
    }
    return *current && !(*current)->numerical_only ? *current : empty;
}

std::shared_ptr<const DynamicPointProvenance> DynamicGeometryAccess::original_source(
    const std::shared_ptr<const DynamicPointProvenance>& point)
{
    return borrowed_original_source(point);
}

void DynamicGeometryAccess::seal_export(DynamicPoint& point,
    const std::shared_ptr<const DynamicPointProvenance>& source, std::vector<PlaneAuthority> supports,
    bool numerical_only)
{
    if (numerical_only || (source && source->numerical_only))
    {
        seal_numerical(point);
        return;
    }
    accept_authorities(supports);
    std::sort(supports.begin(), supports.end());
    supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
    // Identical immutable support sequences may share storage. Point identity,
    // numerical enclosures and source ancestry remain separate records.
    auto payload =
        source && source->supports() == supports ? source->support_payload : support_payload(std::move(supports));
    point.provenance_ = point_provenance(point, std::move(payload), source, true, false);
}

void DynamicGeometryAccess::seal_generated(DynamicPoint& point,
    std::vector<std::array<std::shared_ptr<const DynamicPointProvenance>, 2>> edges,
    std::vector<PlaneAuthority> supports, bool numerical_only)
{
    if (numerical_only)
    {
        seal_numerical(point);
        return;
    }
    accept_authorities(supports);
    auto record = std::make_shared<DynamicPointProvenance>(DynamicPointProvenance{
        point.value, point.absolute_error, support_payload(std::move(supports)), point.arithmetic_mode});
    record->constructed = true;
    record->source_edges = std::move(edges);
    point.provenance_ = std::move(record);
}

void DynamicGeometryAccess::seal_numerical(DynamicPoint& point)
{
    auto record = std::make_shared<DynamicPointProvenance>(
        DynamicPointProvenance{point.value, point.absolute_error, {}, point.arithmetic_mode});
    record->constructed = true;
    record->numerical_only = true;
    point.provenance_ = std::move(record);
}

const std::shared_ptr<const DynamicConvexGeometry::Impl>& DynamicGeometryAccess::implementation(
    const DynamicConvexGeometry& geometry)
{
    return geometry.impl_;
}

DynamicConvexGeometryPtr DynamicGeometryAccess::geometry(std::shared_ptr<const DynamicConvexGeometry::Impl> impl)
{
    return DynamicConvexGeometryPtr(new DynamicConvexGeometry(std::move(impl)));
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
                resource_failure(limits.arithmetic_mode, "derived scalar-work guard overflow.");
            return a + b;
        };
        const auto checked_multiply = [this](std::size_t a, std::size_t b)
        {
            if (b && a > std::numeric_limits<std::size_t>::max() / b)
                resource_failure(limits.arithmetic_mode, "derived scalar-work guard overflow.");
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
        resource_failure(limits.arithmetic_mode, "construction storage cap exhausted.");
    bytes += count * size;
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC::details
{
void Work::pair()
{
    if (++pairs > limits.maximum_pair_tests)
        resource_failure(limits.arithmetic_mode, "pair-test cap exhausted.");
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
        resource_failure(limits.arithmetic_mode, "zero resource cap.");
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
    seal_represented_source(point, point.supports());
}
void DynamicGeometryAccess::seal_represented_source(DynamicPoint& point, std::vector<PlaneAuthority> supports)
{
    accept_authorities(supports);
    std::sort(supports.begin(), supports.end());
    supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
    const bool represented =
        (point.arithmetic_mode == ArithmeticMode::Ordinary ||
            point.arithmetic_mode == ArithmeticMode::BoundedCertification) &&
        point.absolute_error == std::array<long double, 3>{} &&
        std::all_of(point.value.begin(), point.value.end(), [](long double x) { return std::isfinite(x); });
    point.provenance_ = std::make_shared<DynamicPointProvenance>(DynamicPointProvenance{
        point.value, point.absolute_error, support_payload(std::move(supports)), point.arithmetic_mode, represented});
}
bool DynamicGeometryAccess::same_represented_source(
    const std::shared_ptr<const DynamicPointProvenance>& a, const std::shared_ptr<const DynamicPointProvenance>& b)
{
    // Explicit exact represented-input triples identify a point, not a native
    // support, ownership, epoch or trial. Generated/uncertain inputs do not
    // acquire this relation merely because their nominal coordinates agree.
    if ((a && a->numerical_only) || (b && b->numerical_only) || (a && b && a->arithmetic_mode != b->arithmetic_mode))
        return false;
    const auto& first = borrowed_original_source(a);
    const auto& second = borrowed_original_source(b);
    return first && second &&
           (first == second ||
               (first->represented_native_source && second->represented_native_source && !first->constructed &&
                   !second->constructed && first->arithmetic_mode == second->arithmetic_mode &&
                   first->absolute_error == std::array<long double, 3>{} &&
                   second->absolute_error == std::array<long double, 3>{} && first->value == second->value));
}
} // namespace SimpleFluid::PLIC

namespace SimpleFluid::PLIC::details
{
namespace
{
std::vector<PlaneAuthority> copied_vertex_authorities(
    const std::vector<PlaneAuthority>& source, const PlaneAuthority* appended = nullptr)
{
    // Precharge every new buffer before allocating. The immutable owner stays
    // pinned by the caller, including when appended aliases one of its keys.
    if (active_work)
    {
        active_work->charge(source.size(), sizeof(PlaneAuthority));
        if (appended)
            active_work->charge(1, sizeof(PlaneAuthority));
        for (const auto& authority : source)
            active_work->retain_identity(authority);
        if (appended)
            active_work->retain_identity(*appended);
    }
    std::vector<PlaneAuthority> result;
    result.reserve(source.size() + (appended ? 1 : 0));
    for (const auto& authority : source)
        result.push_back(authority);
    if (appended)
        result.push_back(*appended);
    return result;
}
} // namespace

void StoredVertexAuthorities::share_if_equal(const std::shared_ptr<const DynamicPointSupportPayload>& payload)
{
    if (payload && values() == payload->authorities)
        storage_ = payload;
}

void StoredVertexAuthorities::detach()
{
    if (const auto& shared = shared_payload())
    {
        const auto owner = shared;
        storage_ = copied_vertex_authorities(owner->authorities);
    }
}

void StoredVertexAuthorities::append(const PlaneAuthority& authority)
{
    if (auto* owned = std::get_if<std::vector<PlaneAuthority>>(&storage_))
        owned->push_back(authority);
    else
    {
        const auto owner = shared_payload();
        storage_ = copied_vertex_authorities(owner ? owner->authorities : values(), &authority);
    }
}

void StoredVertexAuthorities::clear()
{
    if (auto* owned = std::get_if<std::vector<PlaneAuthority>>(&storage_))
        owned->clear();
    else
        storage_ = std::vector<PlaneAuthority>{};
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC
{
std::span<const DynamicPoint> DynamicPointExport::points() const& noexcept
{
    return payload_ ? std::span<const DynamicPoint>(payload_->points) : std::span<const DynamicPoint>{};
}
template<ArithmeticMode Mode> std::span<const Point<Mode>> PointExport<Mode>::points() const& noexcept
{
    return payload_ ? std::span<const Point<Mode>>(payload_->points) : std::span<const Point<Mode>>{};
}
template class PointExport<ArithmeticMode::Ordinary>;
template class PointExport<ArithmeticMode::BoundedCertification>;

void DynamicGeometryAccess::attach_identity(DynamicPoint& point, std::shared_ptr<const DynamicPointProvenance> identity)
{
    point.provenance_ = std::move(identity);
}

namespace
{
/** Candidates are reserved before allocation and charged exactly once by final
 * owner admission. Canonical comparison scratch consumes the same ceiling. */
struct ExportReservation
{
    std::size_t& used;
    std::size_t limit, reserved;
    details::Work& work;
    void add(std::size_t count, std::size_t size)
    {
        used = std::max(used, work.bytes);
        const auto remaining = limit - std::min(used, limit);
        if (reserved > remaining || (size && count > (remaining - reserved) / size))
            throw GeometryError(GeometryErrorCode::ResourceLimit, "Point export candidate exceeds caller byte budget.");
        reserved += count * size;
        // Scratch preflights must leave every live candidate allocation covered.
        work.limits.byte_budget = limit - reserved;
    }
};
struct ExportArithmetic
{
    details::Work work;
    details::Work* previous;
    std::size_t& used;
    ExportArithmetic(GeometryLimits limits, std::size_t& cumulative, std::size_t limit)
        : work(
              [&]
              {
                  limits.byte_budget = limit;
                  return limits;
              }()),
          previous(details::active_work), used(cumulative)
    {
        if (limits.arithmetic_mode == ArithmeticMode::BoundedCertification && !certification_environment())
            details::reject(limits.arithmetic_mode, "unsupported floating-point environment; no exact fallback.");
        work.bytes = used;
        // Always bind this explicit caller scope, including inside another operation.
        details::active_work = &work;
    }
    void sync() noexcept { used = std::max(used, work.bytes); }
    ~ExportArithmetic()
    {
        sync();
        details::active_work = previous;
    }
};
bool same_support_set(const std::vector<PlaneAuthority>& a, const std::vector<PlaneAuthority>& b)
{
    return std::all_of(
               a.begin(), a.end(), [&](const auto& key) { return std::find(b.begin(), b.end(), key) != b.end(); }) &&
           std::all_of(
               b.begin(), b.end(), [&](const auto& key) { return std::find(a.begin(), a.end(), key) != a.end(); });
}
std::shared_ptr<const DynamicPointSupportPayload> export_supports(const std::vector<PlaneAuthority>& authorities,
    const std::shared_ptr<const DynamicPointProvenance>& source, ExportReservation& budget)
{
    if (source && same_support_set(authorities, source->supports()))
        return source->support_payload;
    if (authorities.empty())
        return {};
    budget.add(1, sizeof(DynamicPointSupportPayload));
    budget.add(authorities.size(), sizeof(PlaneAuthority));
    std::vector<PlaneAuthority> copied;
    copied.reserve(authorities.size());
    if (copied.capacity() > authorities.size())
        budget.add(copied.capacity() - authorities.size(), sizeof(PlaneAuthority));
    copied.insert(copied.end(), authorities.begin(), authorities.end());
    std::sort(copied.begin(), copied.end());
    copied.erase(std::unique(copied.begin(), copied.end()), copied.end());
    return std::make_shared<DynamicPointSupportPayload>(std::move(copied));
}
template<ArithmeticMode Mode>
std::shared_ptr<const DynamicPointExportPayload> materialize_dynamic(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>& original, const PointExportCache& cache,
    ExportReservation& budget)
{
    const auto canonical = original->export_source ? original->export_source : original;
    const auto geometry = details::checked_geometry_storage<Mode>(canonical);
    budget.add(1, sizeof(DynamicPointExportPayload));
    budget.add(geometry->vertices.size(), sizeof(DynamicPoint));
    auto result = std::make_shared<DynamicPointExportPayload>();
    result->mode = Mode;
    result->points.reserve(geometry->vertices.size());
    if (result->points.capacity() > geometry->vertices.size())
        budget.add(result->points.capacity() - geometry->vertices.size(), sizeof(DynamicPoint));
    for (std::size_t i = 0; i < geometry->vertices.size(); ++i)
    {
        if (const auto existing = cache.vertex_provenance[i].lock())
        {
            DynamicPoint point(existing->value, existing->absolute_error, existing->arithmetic_mode);
            DynamicGeometryAccess::attach_identity(point, existing);
            result->points.push_back(std::move(point));
            continue;
        }
        const auto& vertex = geometry->vertices[i];
        DynamicPoint point;
        if constexpr (Mode == ArithmeticMode::Ordinary)
            point = details::ordinary_kernel::bounded<Mode>(details::ordinary_kernel::operator+(
                details::ordinary_kernel::scaled<Mode>(vertex.coordinates, geometry->frame.exponent),
                details::ordinary_kernel::point<Mode>(geometry->frame.origin)));
        else
            point = details::certification_kernel::bounded<Mode>(details::certification_kernel::operator+(
                details::certification_kernel::scaled<Mode>(vertex.coordinates, geometry->frame.exponent),
                details::certification_kernel::point<Mode>(geometry->frame.origin)));
        budget.add(1, sizeof(DynamicPointProvenance));
        const auto& source = vertex.source_identity;
        const bool export_seal = bool(source) || geometry->export_seal;
        const bool numerical = export_seal && (geometry->numerical_only || (source && source->numerical_only));
        auto supports = numerical
                            ? std::shared_ptr<const DynamicPointSupportPayload>{}
                            : export_supports(vertex.authorities.values(), export_seal ? source : nullptr, budget);
        DynamicGeometryAccess::attach_identity(
            point, point_provenance(point, std::move(supports), source, export_seal, numerical));
        result->points.push_back(std::move(point));
    }
    return result;
}
} // namespace

template<ArithmeticMode Mode, bool Typed>
auto DynamicGeometryAccess::export_impl(const std::shared_ptr<const DynamicConvexGeometry::Impl>& geometry,
    GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    try
    {
        ledger.begin_admission(used, limit);
        details::require_mode(geometry->limits.arithmetic_mode, Mode);
        ExportArithmetic arithmetic(geometry->limits, used, limit);
        if (!geometry->point_export_cache)
            throw GeometryError(
                GeometryErrorCode::InvalidTopology, "Point export requires a finished finite geometry.");
        auto& cache = *geometry->point_export_cache;
        std::lock_guard guard(cache.mutex);
        auto& typed_root = [&]() -> auto&
        {
            if constexpr (Mode == ArithmeticMode::Ordinary)
                return cache.ordinary_root;
            else
                return cache.bounded_root;
        }();
        if constexpr (Typed)
            if (auto existing = typed_root.lock())
            {
                PointExport<Mode> result(std::move(existing));
                ledger.retain(result, used, limit);
                return result;
            }
        auto dynamic = cache.dynamic_root.lock();
        ExportReservation budget{used, limit, 0, arithmetic.work};
        if (!dynamic)
        {
            details::FrozenExportEnvironment environment(cache);
            dynamic = materialize_dynamic<Mode>(geometry, cache, budget);
        }
        if constexpr (Typed)
        {
            budget.add(1, sizeof(PointExportPayload<Mode>));
            budget.add(dynamic->points.size(), sizeof(Point<Mode>));
            auto payload = std::make_shared<PointExportPayload<Mode>>();
            payload->dynamic = dynamic;
            payload->points.reserve(dynamic->points.size());
            if (payload->points.capacity() > dynamic->points.size())
                budget.add(payload->points.capacity() - dynamic->points.size(), sizeof(Point<Mode>));
            for (const auto& point : dynamic->points)
                payload->points.emplace_back(point);
            PointExport<Mode> result(payload);
            arithmetic.sync();
            ledger.retain(result, used, limit);
            // Publication is the only cache mutation and follows complete admission.
            cache.dynamic_root = dynamic;
            typed_root = payload;
            for (std::size_t i = 0; i < dynamic->points.size(); ++i)
                cache.vertex_provenance[i] = identity(dynamic->points[i]);
            return result;
        }
        else
        {
            DynamicPointExport result(dynamic);
            arithmetic.sync();
            ledger.retain(result, used, limit);
            cache.dynamic_root = dynamic;
            for (std::size_t i = 0; i < dynamic->points.size(); ++i)
                cache.vertex_provenance[i] = identity(dynamic->points[i]);
            return result;
        }
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}

template<ArithmeticMode Mode>
PointExport<Mode> DynamicGeometryAccess::export_typed(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>& geometry, GeometryAllocationLedger& ledger,
    std::size_t& used, std::size_t limit)
{
    return export_impl<Mode, true>(geometry, ledger, used, limit);
}
template<ArithmeticMode Mode>
DynamicPointExport DynamicGeometryAccess::export_dynamic(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>& geometry, GeometryAllocationLedger& ledger,
    std::size_t& used, std::size_t limit)
{
    return export_impl<Mode, false>(geometry, ledger, used, limit);
}
template PointExport<ArithmeticMode::Ordinary> DynamicGeometryAccess::export_typed<ArithmeticMode::Ordinary>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&, GeometryAllocationLedger&, std::size_t&, std::size_t);
template PointExport<ArithmeticMode::BoundedCertification>
DynamicGeometryAccess::export_typed<ArithmeticMode::BoundedCertification>(
    const std::shared_ptr<const DynamicConvexGeometry::Impl>&, GeometryAllocationLedger&, std::size_t&, std::size_t);

DynamicPointExport DynamicConvexGeometry::export_points(
    GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit) const
{
    if (arithmetic_mode() == ArithmeticMode::Ordinary)
        return DynamicGeometryAccess::export_dynamic<ArithmeticMode::Ordinary>(impl_, ledger, used, limit);
    return DynamicGeometryAccess::export_dynamic<ArithmeticMode::BoundedCertification>(impl_, ledger, used, limit);
}
} // namespace SimpleFluid::PLIC
