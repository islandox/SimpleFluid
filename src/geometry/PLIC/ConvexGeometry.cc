#include "geometry/PLIC/GeometryError.hh"
#include "geometry/PLIC/SupportClassification.hh"
#include "geometry/PLIC/details/GeometryKernels.hh"
#include "geometry/PLIC/details/ordinary/Predicates.hh"

#include "geometry/PLIC/details/bounded/ConvexGeometry.tcc"
#include "geometry/PLIC/details/ordinary/ConvexGeometry.tcc"

namespace SimpleFluid::PLIC
{
bool has_support(const DynamicPoint& point, const PlaneAuthority& authority)
{
    if (authority.key.empty())
        return false;
    const auto& supports = point.supports();
    return std::find(supports.begin(), supports.end(), authority) != supports.end();
}

bool has_any_support(const DynamicPoint& point, std::span<const PlaneAuthority> authorities)
{
    return std::any_of(
        authorities.begin(), authorities.end(), [&](const auto& authority) { return has_support(point, authority); });
}

namespace
{
PlaneAuthority pack_original_support(const PlaneSupportIdentity<ArithmeticMode::Ordinary>& support)
{
    const auto bytes = details::GeometryIdentityAccess::key(support);
    if (bytes.empty())
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Empty immutable source support.");
    if (bytes.size() > authority_key_length_mask)
        throw GeometryError(
            GeometryErrorCode::ResourceLimit, "Original support key length exceeds its tagged namespace.");
    PlaneAuthority authority;
    authority.key.push_back(original_support_key_header(bytes.size()));
    authority.key.resize(1 + (bytes.size() + 7) / 8, 0);
    for (std::size_t i = 0; i < bytes.size(); ++i)
        authority.key[1 + i / 8] |= std::uint64_t(static_cast<unsigned char>(bytes[i])) << (8 * (i % 8));
    return authority;
}
bool matches_original_support(
    const PlaneAuthority& authority, const PlaneSupportIdentity<ArithmeticMode::Ordinary>& support)
{
    const auto bytes = details::GeometryIdentityAccess::key(support);
    if (bytes.empty())
        return false;
    if (authority.key.size() != 1 + (bytes.size() + 7) / 8 ||
        authority.key.front() != original_support_key_header(bytes.size()))
        return false;
    for (std::size_t word = 1; word < authority.key.size(); ++word)
    {
        std::uint64_t expected = 0;
        for (unsigned byte = 0; byte < 8 && 8 * (word - 1) + byte < bytes.size(); ++byte)
            expected |= std::uint64_t(static_cast<unsigned char>(bytes[8 * (word - 1) + byte])) << (8 * byte);
        if (expected != authority.key[word])
            return false;
    }
    return true;
}
// Classify source-authority tokens only at acceptance boundaries. This does
// not import numeric coefficients or create native attachments from key words.
bool recognized_original_support(const PlaneAuthority& authority)
{
    return !authority.key.empty() && static_cast<std::uint8_t>(authority.key.front() >> 56) ==
                                         static_cast<std::uint8_t>(AuthorityKeyTag::OriginalSupport);
}
void accept_authorities(const std::vector<PlaneAuthority>& authorities)
{
    for (const auto& authority : authorities)
    {
        details::OriginalSupportAuthorityAccess::validate_boundary(authority);
        if (details::active_work)
            if (const auto* support = details::OriginalSupportAuthorityAccess::retained(authority))
                details::active_work->retain_identity(*support);
    }
}
} // namespace
PlaneAuthority details::OriginalSupportAuthorityAccess::issue(
    const PlaneSupportIdentity<ArithmeticMode::Ordinary>& support)
{
    auto authority = pack_original_support(support);
    authority.original_support.support_ = support;
    return authority;
}
PlaneAuthority details::OriginalSupportAuthorityAccess::issue(
    const PlaneSupportIdentity<ArithmeticMode::Exact>& support)
{
    return issue(GeometryIdentityAccess::retained_exact_support(support));
}
const PlaneSupportIdentity<ArithmeticMode::Ordinary>* details::OriginalSupportAuthorityAccess::retained(
    const PlaneAuthority& authority) noexcept
{
    return authority.original_support.support_ ? &*authority.original_support.support_ : nullptr;
}
const PlaneSupportIdentity<ArithmeticMode::Ordinary>* details::OriginalSupportAuthorityAccess::get(
    const PlaneAuthority& authority)
{
    const auto* support = retained(authority);
    if (support && !matches_original_support(authority, *support))
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch,
            "Original support attachment does not match its frozen authority key.");
    return support;
}
void details::OriginalSupportAuthorityAccess::validate_boundary(const PlaneAuthority& authority)
{
    if (!get(authority) && recognized_original_support(authority))
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch,
            "Original source authority requires its typed immutable support attachment.");
}
void details::OriginalSupportAuthorityAccess::merge(PlaneAuthority& destination, const PlaneAuthority& source)
{
    if (destination.key != source.key)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "Cannot merge different source authority keys.");
    (void) get(destination);
    validate_boundary(source);
    if (!retained(destination) && retained(source))
        destination.original_support = source.original_support;
    validate_boundary(destination);
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
    if (keys.empty() && !native_authority_.key.empty())
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
                bytes += key.key.capacity() * sizeof(std::uint64_t);
                if (include_support_payloads)
                    if (const auto* support = details::OriginalSupportAuthorityAccess::retained(key))
                        attachments.retain(*support, retained_bytes, std::numeric_limits<std::size_t>::max());
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
            if (const auto* support = details::OriginalSupportAuthorityAccess::retained(authority))
                ledger.retain(*support, used, limit);
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
std::shared_ptr<const DynamicPointSupportPayload> support_payload(std::vector<PlaneAuthority> supports)
{
    if (supports.empty())
        return {};
    if (details::active_work)
        details::active_work->charge(1, sizeof(DynamicPointSupportPayload));
    if (supports.capacity() > supports.size())
    {
        // Account the replacement while the old outer buffer remains alive.
        // Moving authorities retains each key buffer and its original capacity.
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
    point.provenance_ = std::make_shared<DynamicPointProvenance>(DynamicPointProvenance{
        point.value, point.absolute_error, support_payload(std::move(supports)), point.arithmetic_mode});
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
    auto payload = source && source->supports() == supports ? source->support_payload
                                                          : support_payload(std::move(supports));
    auto record = std::make_shared<DynamicPointProvenance>(
        DynamicPointProvenance{point.value, point.absolute_error, std::move(payload), point.arithmetic_mode});
    record->constructed = !source || source->constructed || source->value != point.value ||
                          source->absolute_error != point.absolute_error;
    record->source_origin = source;
    point.provenance_ = std::move(record);
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
        (point.arithmetic_mode == ArithmeticMode::Ordinary || point.arithmetic_mode == ArithmeticMode::BoundedCertification) &&
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
    if ((a && a->numerical_only) || (b && b->numerical_only) ||
        (a && b && a->arithmetic_mode != b->arithmetic_mode))
        return false;
    const auto& first = borrowed_original_source(a);
    const auto& second = borrowed_original_source(b);
    return first && second &&
           (first == second || (first->represented_native_source && second->represented_native_source &&
                                   !first->constructed && !second->constructed &&
                                   first->arithmetic_mode == second->arithmetic_mode &&
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
            active_work->charge(authority.key.size(), sizeof(std::uint64_t));
        if (appended)
            active_work->charge(appended->key.size(), sizeof(std::uint64_t));
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
