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
const PlaneAuthority& PointAuthorityRange::Iterator::operator*() const
{
    return lookup ? lookup(source, position) : static_cast<const PlaneAuthority*>(source)[position];
}
const PlaneAuthority& PointAuthorityRange::operator[](std::size_t index) const
{
    return *Iterator{source_, lookup_, first_ + index};
}
bool PointAuthorityRange::operator==(const PointAuthorityRange& other) const
{
    return size() == other.size() && std::equal(begin(), end(), other.begin());
}
bool PointAuthorityRange::operator==(const std::vector<PlaneAuthority>& other) const
{
    return size() == other.size() && std::equal(begin(), end(), other.begin());
}

PointProvenanceRef::PointProvenanceRef(std::shared_ptr<const PointSnapshotContextBase> owner, DynamicStoredPointRef ref)
{
    const PointProvenanceView resolved(owner.get(), ref);
    if (!resolved)
        return;
    if (!owner->frozen)
        throw GeometryError(GeometryErrorCode::ProvenanceMismatch, "An unfinished point snapshot cannot escape.");
    if (ref.dependency_slot)
        owner = owner->dependencies.at(ref.dependency_slot - 1);
    owner_ = std::move(owner);
    ref_ = resolved.local_ref();
}
PointProvenanceView PointProvenanceRef::view() const
{
    return {owner_.get(), ref_};
}
template class TypedPointProvenanceRef<ArithmeticMode::Ordinary>;
template class TypedPointProvenanceRef<ArithmeticMode::BoundedCertification>;

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

bool has_any_support(const DynamicPoint& point, PointAuthorityRange authorities)
{
    return std::any_of(
        authorities.begin(), authorities.end(), [&](const auto& authority) { return has_support(point, authority); });
}
bool has_any_support(const DynamicPoint& point, std::span<const PlaneAuthority> keys)
{
    return has_any_support(point, DynamicGeometryAccess::boundary_keys(keys));
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

PointAuthorityRange OrdinaryBoundary::authorities() const
{
    const auto storage = details::checked_geometry_storage<ArithmeticMode::Ordinary>(
        DynamicGeometryAccess::implementation(*boundary_.geometry->dynamic_geometry()));
    const auto& plane = *storage->surfaces.at(boundary_.face).plane;
    const auto keys = resolve_authorities(plane.point_context, plane.authorities);
    if (keys.empty() && !native_authority_.empty())
        return DynamicGeometryAccess::boundary_keys({&native_authority_, 1});
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

namespace
{
bool provenance_matches(const DynamicPoint& point, PointProvenanceView provenance) noexcept
{
    return provenance && provenance.value() == point.value && provenance.absolute_error() == point.absolute_error &&
           provenance.arithmetic_mode() == point.arithmetic_mode;
}
} // namespace

PointAuthorityRange DynamicPoint::supports() const
{
    return provenance_matches(*this, provenance_.view()) ? provenance_.view().supports() : PointAuthorityRange{};
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

constinit thread_local Work* active_work = nullptr;

GeometryLimits mode_limits(ArithmeticMode mode)
{
    GeometryLimits result;
    result.arithmetic_mode = mode;
    return result;
}

void require_mode(ArithmeticMode a, ArithmeticMode b)
{
    if (a == ArithmeticMode::Exact || b == ArithmeticMode::Exact || a != b)
        reject(a, "incompatible arithmetic modes in finite geometry.");
}

void operation()
{
    ++statistics.floating_operations;
}
void charge_point_snapshot(std::size_t count, std::size_t size)
{
    if (active_work)
        active_work->charge(count, size);
}
} // namespace SimpleFluid::PLIC::details

namespace SimpleFluid::PLIC
{
PointProvenanceRef DynamicGeometryAccess::identity(const DynamicPoint& point)
{
    return provenance_matches(point, point.provenance_.view()) ? point.provenance_ : PointProvenanceRef{};
}
std::size_t DynamicGeometryAccess::identity_bytes(const PointProvenanceRef& point, bool include_support_payloads)
{
    return identity_bytes(point.view(), include_support_payloads);
}
std::size_t DynamicGeometryAccess::identity_bytes(PointProvenanceView point, bool include_support_payloads)
{
    if (!point)
        return 0;
    GeometryAllocationLedger ledger;
    std::size_t used = 0;
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (point.arithmetic_mode() == ArithmeticMode::Ordinary)
        ledger.retain_snapshot<ArithmeticMode::Ordinary>(point.owning().context_pin(), used, limit);
    else
        ledger.retain_snapshot<ArithmeticMode::BoundedCertification>(point.owning().context_pin(), used, limit);
    if (!include_support_payloads)
    {
        GeometryAllocationLedger attachments;
        std::size_t retained = 0;
        retain_supports(point, attachments, retained, limit);
        return ledger.payload_bytes() - attachments.payload_bytes();
    }
    return ledger.payload_bytes();
}
void DynamicGeometryAccess::retain_supports(
    PointProvenanceView root, GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    if (!root)
        return;
    // Context dependencies are a frozen DAG. Visit each context once; a copied
    // point owns its complete snapshot, including all registered key payloads.
    std::set<const PointSnapshotContextBase*> seen;
    std::vector<const PointSnapshotContextBase*> pending{root.context()};
    for (std::size_t i = 0; i < pending.size(); ++i)
    {
        const auto* context = pending[i];
        if (!seen.insert(context).second)
            continue;
        snapshot_detail::dispatch(context,
            [&](const auto& typed)
            {
                for (const auto& key : typed.authority_table)
                    ledger.retain(key, used, limit);
            });
        for (const auto& parent : context->dependencies)
            pending.push_back(parent.get());
    }
}
std::size_t DynamicGeometryAccess::storage_bytes(const DynamicPoint& point, bool include_support_payloads)
{
    return sizeof(DynamicPoint) + identity_bytes(point.provenance_, include_support_payloads);
}
bool DynamicGeometryAccess::same_source(const DynamicPoint& a, const DynamicPoint& b)
{
    return a.provenance_ == b.provenance_ && provenance_matches(a, a.provenance_.view()) &&
           provenance_matches(b, b.provenance_.view());
}
namespace
{
void normalize_supports(std::vector<PlaneAuthority>& supports)
{
    accept_authorities(supports);
    std::sort(supports.begin(), supports.end());
    supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
}
template<ArithmeticMode Mode>
PointProvenanceRef single_source(const DynamicPoint& point, std::span<const PlaneAuthority> supports)
{
    PointSnapshotBuilder<Mode> builder;
    const auto previous = DynamicGeometryAccess::identity(point);
    const auto facts = previous && previous.view().local_ref().role == PointProvenanceRole::Native
                           ? previous.view()
                           : PointProvenanceView{};
    const auto view = builder.native_source(
        point.value, point.absolute_error, DynamicGeometryAccess::admit_incidence<Mode>(supports), facts);
    const auto ref = builder.import(view);
    return bind_point(builder.freeze(), ref);
}
template<ArithmeticMode Mode>
PointProvenanceRef single_sealed(const DynamicPoint& point, std::span<const PlaneAuthority> supports)
{
    PointSnapshotBuilder<Mode> builder;
    const auto view =
        builder.sealed(point.value, point.absolute_error, DynamicGeometryAccess::admit_incidence<Mode>(supports));
    const auto ref = builder.import(view);
    return bind_point(builder.freeze(), ref);
}
template<ArithmeticMode Mode>
PointProvenanceRef single_export(
    const DynamicPoint& point, PointProvenanceView source, std::span<const PlaneAuthority> supports, bool numerical)
{
    PointSnapshotBuilder<Mode> builder;
    const auto view = builder.export_alias(
        point.value, point.absolute_error, source, DynamicGeometryAccess::admit_incidence<Mode>(supports), numerical);
    const auto ref = builder.import(view);
    return bind_point(builder.freeze(), ref);
}
template<ArithmeticMode Mode>
PointProvenanceRef single_generated(const DynamicPoint& point,
    std::span<const std::array<PointProvenanceView, 2>> edges, std::span<const PlaneAuthority> supports, bool numerical)
{
    PointSnapshotBuilder<Mode> builder;
    const auto view = numerical ? builder.numerical(point.value, point.absolute_error)
                                : builder.generated(point.value, point.absolute_error, edges,
                                      DynamicGeometryAccess::admit_incidence<Mode>(supports));
    const auto ref = builder.import(view);
    return bind_point(builder.freeze(), ref);
}
} // namespace
void DynamicGeometryAccess::seal(DynamicPoint& point, std::vector<PlaneAuthority> supports)
{
    normalize_supports(supports);
    point.provenance_ = point.arithmetic_mode == ArithmeticMode::Ordinary
                            ? single_sealed<ArithmeticMode::Ordinary>(point, supports)
                            : single_sealed<ArithmeticMode::BoundedCertification>(point, supports);
}
PointProvenanceView DynamicGeometryAccess::borrowed_original_source(PointProvenanceView point)
{
    while (point && point.source_origin())
    {
        const auto source = point.source_origin();
        if (point.numerical_only() || source.arithmetic_mode() != point.arithmetic_mode())
            return {};
        if (details::active_work)
            details::active_work->pair();
        point = source;
    }
    return point && !point.numerical_only() ? point : PointProvenanceView{};
}
void DynamicGeometryAccess::seal_export(
    DynamicPoint& point, PointProvenanceView source, std::vector<PlaneAuthority> supports, bool numerical)
{
    normalize_supports(supports);
    point.provenance_ = point.arithmetic_mode == ArithmeticMode::Ordinary
                            ? single_export<ArithmeticMode::Ordinary>(point, source, supports, numerical)
                            : single_export<ArithmeticMode::BoundedCertification>(point, source, supports, numerical);
}
void DynamicGeometryAccess::seal_generated(DynamicPoint& point, std::vector<std::array<PointProvenanceView, 2>> edges,
    std::vector<PlaneAuthority> supports, bool numerical)
{
    normalize_supports(supports);
    point.provenance_ = point.arithmetic_mode == ArithmeticMode::Ordinary
                            ? single_generated<ArithmeticMode::Ordinary>(point, edges, supports, numerical)
                            : single_generated<ArithmeticMode::BoundedCertification>(point, edges, supports, numerical);
}
void DynamicGeometryAccess::seal_numerical(DynamicPoint& point)
{
    seal_generated(point, {}, {}, true);
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

void Work::charge(std::size_t count, std::size_t size)
{
    if (size && count > (limits.byte_budget - bytes) / size)
        resource_failure(limits.arithmetic_mode, "construction storage cap exhausted.");
    bytes += count * size;
}

void Work::pair()
{
    if (++pairs > limits.maximum_pair_tests)
        resource_failure(limits.arithmetic_mode, "pair-test cap exhausted.");
}

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
    const auto supports = point.supports();
    seal_represented_source(point, {supports.begin(), supports.end()});
}
void DynamicGeometryAccess::seal_represented_source(DynamicPoint& point, std::vector<PlaneAuthority> supports)
{
    normalize_supports(supports);
    point.provenance_ = point.arithmetic_mode == ArithmeticMode::Ordinary
                            ? single_source<ArithmeticMode::Ordinary>(point, supports)
                            : single_source<ArithmeticMode::BoundedCertification>(point, supports);
}
bool DynamicGeometryAccess::same_represented_source(PointProvenanceView a, PointProvenanceView b)
{
    if ((a && a.numerical_only()) || (b && b.numerical_only()) ||
        (a && b && a.arithmetic_mode() != b.arithmetic_mode()))
        return false;
    const auto first = borrowed_original_source(a), second = borrowed_original_source(b);
    return first && second &&
           (first == second ||
               (first.represented_native_source() && second.represented_native_source() && !first.constructed() &&
                   !second.constructed() && first.arithmetic_mode() == second.arithmetic_mode() &&
                   first.absolute_error() == std::array<long double, 3>{} &&
                   second.absolute_error() == std::array<long double, 3>{} && first.value() == second.value()));
}
} // namespace SimpleFluid::PLIC

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

template<ArithmeticMode Mode> std::size_t PointSnapshot<Mode>::size() const noexcept
{
    return payload_ ? payload_->refs.size() : 0;
}
template<ArithmeticMode Mode> const void* PointSnapshot<Mode>::context_identity() const noexcept
{
    return payload_ ? payload_->context.get() : nullptr;
}
template<ArithmeticMode Mode> Point<Mode> PointSnapshot<Mode>::point(std::size_t index) const
{
    if (!payload_)
        throw std::out_of_range("Point snapshot index exceeds its empty batch.");
    const auto ref = payload_->refs.at(index);
    const auto view = resolve_point(payload_->context, ref);
    DynamicPoint result(view.value(), view.absolute_error(), Mode);
    DynamicGeometryAccess::attach_identity(result, bind_point(payload_->context, ref));
    return Point<Mode>(result);
}
template class PointSnapshot<ArithmeticMode::Ordinary>;
template class PointSnapshot<ArithmeticMode::BoundedCertification>;

template<ArithmeticMode Mode>
PointSnapshot<Mode> DynamicGeometryAccess::capture_snapshot(
    std::span<const Point<Mode>> points, GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    try
    {
        ledger.begin_admission(used, limit, !points.empty());
        if (points.empty())
            return {};
        std::shared_ptr<const PointSnapshotContext<Mode>> context;
        // Validate the whole caller span before allocating any candidate. Only
        // exact existing tokens are copied; no context or authority is issued.
        for (const auto& point : points)
        {
            const auto token = identity(point.dynamic());
            if (!token || !token.context_pin()->frozen || token.view().arithmetic_mode() != Mode)
                throw GeometryError(GeometryErrorCode::ProvenanceMismatch,
                    "Point snapshot requires sealed, unmodified points in its arithmetic mode.");
            if (!context)
                context = std::static_pointer_cast<const PointSnapshotContext<Mode>>(token.context_pin());
            else if (context.get() != token.context_identity())
                throw GeometryError(GeometryErrorCode::ProvenanceMismatch,
                    "Point snapshot cannot combine independently owned provenance contexts.");
        }
        // Admit the existing context first. A foreign ledger mode or an
        // exhausted context budget rejects before any compact payload exists.
        ledger.retain_snapshot<Mode>(context, used, limit);
        const auto remaining = limit - std::min(used, limit);
        constexpr auto owner_bytes = sizeof(PointSnapshotPayload<Mode>) + 2 * sizeof(void*);
        if (owner_bytes > remaining || points.size() > (remaining - owner_bytes) / sizeof(StoredPointRef<Mode>))
            throw GeometryError(
                GeometryErrorCode::ResourceLimit, "Point snapshot candidate exceeds caller byte budget.");
        auto payload = std::make_shared<PointSnapshotPayload<Mode>>();
        payload->context = std::move(context);
        payload->refs.reserve(points.size());
        if (payload->refs.capacity() > (remaining - owner_bytes) / sizeof(StoredPointRef<Mode>))
            throw GeometryError(
                GeometryErrorCode::ResourceLimit, "Point snapshot reference capacity exceeds caller byte budget.");
        for (const auto& point : points)
        {
            const auto token = identity(point.dynamic());
            const auto ref = token.local_ref();
            payload->refs.push_back(StoredPointRef<Mode>{ref.dependency_slot, ref.index, ref.role});
        }
        PointSnapshot<Mode> result(payload);
        ledger.retain(result, used, limit);
        return result;
    }
    catch (...)
    {
        ledger.failed_ = true;
        throw;
    }
}
template<ArithmeticMode Mode>
PointSnapshot<Mode> snapshot_points(
    std::span<const Point<Mode>> points, GeometryAllocationLedger& ledger, std::size_t& used, std::size_t limit)
{
    return DynamicGeometryAccess::capture_snapshot<Mode>(points, ledger, used, limit);
}
template PointSnapshot<ArithmeticMode::Ordinary> snapshot_points<ArithmeticMode::Ordinary>(
    std::span<const Point<ArithmeticMode::Ordinary>>, GeometryAllocationLedger&, std::size_t&, std::size_t);
template PointSnapshot<ArithmeticMode::BoundedCertification> snapshot_points<ArithmeticMode::BoundedCertification>(
    std::span<const Point<ArithmeticMode::BoundedCertification>>, GeometryAllocationLedger&, std::size_t&, std::size_t);

void DynamicGeometryAccess::attach_identity(DynamicPoint& point, PointProvenanceRef identity)
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
    std::size_t retained_exclusive = 0;
    bool handed_off = false;
    ~ExportReservation()
    {
        if (!handed_off)
        {
            // Every admitted candidate becomes retired if construction fails
            // before final-ledger handoff. Preflights reserve this complete
            // prefix alongside all arithmetic scratch, including on unwind.
            used = std::max(used, work.bytes);
            assert(used <= limit && reserved <= limit - used);
            used += reserved;
            work.bytes = used;
            work.limits.byte_budget = limit;
        }
    }
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
    void keep(std::size_t count, std::size_t size)
    {
        if (size && count > (reserved - std::min(retained_exclusive, reserved)) / size)
            throw GeometryError(GeometryErrorCode::ResourceLimit,
                "Retained point export candidate exceeds its allocation reservation.");
        retained_exclusive += count * size;
    }
    void debit_retired()
    {
        used = std::max(used, work.bytes);
        const auto retired = reserved - retained_exclusive;
        if (retired > limit - std::min(used, limit))
            throw GeometryError(
                GeometryErrorCode::ResourceLimit, "Retired point export storage exceeds caller budget.");
        used += retired;
        work.bytes = used;
        reserved = retained_exclusive;
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
    budget.keep(1, sizeof(DynamicPointExportPayload));
    budget.keep(result->points.capacity(), sizeof(DynamicPoint));
    if (const auto existing = cache.provenance_root.lock())
    {
        if (cache.vertex_provenance.size() != geometry->vertices.size())
            throw GeometryError(GeometryErrorCode::InvalidTopology, "Point cache slot count differs from geometry.");
        result->point_context = existing;
        for (const auto ref : cache.vertex_provenance)
        {
            const auto identity = bind_point(existing, ref);
            const auto view = identity.view();
            DynamicPoint point(view.value(), view.absolute_error(), view.arithmetic_mode());
            DynamicGeometryAccess::attach_identity(point, identity);
            result->points.push_back(std::move(point));
        }
        return result;
    }
    PointSnapshotBuilder<Mode> builder(
        PointSnapshotAllocation{&budget, [](void* state, std::size_t count, std::size_t size)
            { static_cast<ExportReservation*>(state)->add(count, size); }});
    budget.add(geometry->vertices.size(), sizeof(StoredPointRef<Mode>));
    std::vector<StoredPointRef<Mode>> refs;
    refs.reserve(geometry->vertices.size());
    if (refs.capacity() > geometry->vertices.size())
        budget.add(refs.capacity() - geometry->vertices.size(), sizeof(StoredPointRef<Mode>));
    for (const auto& vertex : geometry->vertices)
    {
        DynamicPoint point;
        if constexpr (Mode == ArithmeticMode::Ordinary)
            point = details::ordinary_kernel::bounded<Mode>(details::ordinary_kernel::operator+(
                details::ordinary_kernel::scaled<Mode>(vertex.coordinates, geometry->frame.exponent),
                details::ordinary_kernel::point<Mode>(geometry->frame.origin)));
        else
            point = details::certification_kernel::bounded<Mode>(details::certification_kernel::operator+(
                details::certification_kernel::scaled<Mode>(vertex.coordinates, geometry->frame.exponent),
                details::certification_kernel::point<Mode>(geometry->frame.origin)));
        const auto source = resolve_point(geometry->point_context, vertex.source_identity);
        const bool export_seal = bool(source) || geometry->export_seal;
        const bool numerical = export_seal && (geometry->numerical_only || (source && source.numerical_only()));
        const auto admitted = resolve_authorities(geometry->point_context, vertex.authorities);
        budget.add(admitted.size(), sizeof(PlaneAuthority));
        std::vector<PlaneAuthority> supports(admitted.begin(), admitted.end());
        if (supports.capacity() > admitted.size())
            budget.add(supports.capacity() - admitted.size(), sizeof(PlaneAuthority));
        // These keys already belong to admitted geometry CSR rows. Preserve
        // canonical ordering without a second Work ownership walk over their
        // borrowed backing payloads; builder issuance validates boundaries.
        std::sort(supports.begin(), supports.end());
        supports.erase(std::unique(supports.begin(), supports.end()), supports.end());
        const auto proof = DynamicGeometryAccess::admit_incidence<Mode>(std::span<const PlaneAuthority>(supports));
        const auto alias = export_seal
                               ? builder.export_alias(point.value, point.absolute_error, source, proof, numerical)
                               : builder.sealed(point.value, point.absolute_error, proof);
        refs.push_back(builder.import(alias));
        result->points.push_back(std::move(point));
    }
    const auto context = builder.freeze();
    budget.keep(1, context->exclusive_storage_bytes());
    result->point_context = context;
    for (std::size_t index = 0; index < result->points.size(); ++index)
        DynamicGeometryAccess::attach_identity(result->points[index], bind_point(context, refs[index]));
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
            budget.keep(1, sizeof(PointExportPayload<Mode>));
            budget.keep(payload->points.capacity(), sizeof(Point<Mode>));
            for (const auto& point : dynamic->points)
                payload->points.emplace_back(point);
            PointExport<Mode> result(payload);
            arithmetic.sync();
            budget.debit_retired();
            budget.handed_off = true;
            ledger.retain(result, used, limit);
            // Publication is the only cache mutation and follows complete admission.
            cache.dynamic_root = dynamic;
            typed_root = payload;
            cache.provenance_root = dynamic->point_context;
            for (std::size_t i = 0; i < dynamic->points.size(); ++i)
                cache.vertex_provenance[i] = identity(dynamic->points[i]).local_ref();
            return result;
        }
        else
        {
            DynamicPointExport result(dynamic);
            arithmetic.sync();
            budget.debit_retired();
            budget.handed_off = true;
            ledger.retain(result, used, limit);
            cache.dynamic_root = dynamic;
            cache.provenance_root = dynamic->point_context;
            for (std::size_t i = 0; i < dynamic->points.size(); ++i)
                cache.vertex_provenance[i] = identity(dynamic->points[i]).local_ref();
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
