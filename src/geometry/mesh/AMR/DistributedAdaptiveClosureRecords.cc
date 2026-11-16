/** @file DistributedAdaptiveClosureRecords.cc
 * @brief Shared immutable closure records and deterministic grading predicates.
 */
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include "geometry/PLIC/GeometryIdentityCodec.hh"
#include "geometry/PLIC/ScalarPacket.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
std::size_t key_capacity(const AdaptiveCellKey& key)
{
    return std::visit(
        [](const auto& value) -> std::size_t
        {
            if constexpr (requires { value.path; })
                return value.path.capacity() * sizeof(std::uint64_t);
            else if constexpr (requires {
                                   value.base;
                                   value.axial;
                               })
                return (value.base.path.capacity() + value.axial.path.capacity()) * sizeof(std::uint64_t);
            else
                return 0;
        },
        key.ancestry);
}
std::size_t key_wire_size(const AdaptiveCellKey& key)
{
    const auto path_bytes = std::visit([](const auto& value) -> std::size_t {
        if constexpr (requires { value.path; }) return value.path.size();
        else if constexpr (requires { value.base; value.axial; }) return value.base.path.size() + value.axial.path.size();
        else return 0;
    }, key.ancestry);
    if (path_bytes > (std::numeric_limits<std::size_t>::max() - 80) / 8)
        throw std::invalid_argument("Adaptive key size overflows.");
    const std::size_t fixed[] = {72, 56, 80, 56};
    return fixed[key.ancestry.index()] + 8 * path_bytes;
}
AdaptiveCellKey decode_bounded_key(std::span<const char> bytes, Budget& budget)
{
    // Path arrays reserve their validated lengths. Keep a conservative envelope
    // for the decoded ancestry value and its variant-copy temporaries; the
    // returned owner's actual capacity is retained after those temporaries die.
    if (bytes.size() > (budget.limit - budget.used) / 3)
        throw std::invalid_argument("Adaptive key decode exceeds live byte budget.");
    const auto admitted = 3 * bytes.size();
    budget.add(admitted);
    auto result = decode_adaptive_key(bytes);
    const auto retained = key_capacity(result);
    if (retained > admitted)
        throw std::invalid_argument("Adaptive key decode capacity exceeds admission.");
    budget.used -= admitted - retained;
    return result;
}
namespace
{
template<class Factor, class Admit>
AdaptiveBytes write_contact(const AdaptiveCellKey& key, std::span<const char> domain,
    std::span<const AdaptiveTraceFace> traces, const std::array<long double, 3>& spans, Factor&& factor, Budget& budget,
    Admit&& add_transient, bool bounded = false)
{
    Writer w;
    std::size_t reservation = 0;
    if (bounded)
    {
        auto add = [&](std::size_t n) {
            if (n > std::numeric_limits<std::size_t>::max() - reservation)
                throw std::invalid_argument("Closure record size overflows.");
            reservation += n;
        };
        add(geometry_record_header_bytes + 8 + key_wire_size(key) + 8 + domain.size());
        add(16 + 3 * PLIC::ScalarPacketWords{}.size() * 8 + 24 + 8);
        for (const auto& face : traces)
        {
            add(8); add(face.stem.size()); add(32);
            if (face.face) { add(8); add(face.face->encoded_incidence_size()); }
        }
        budget.add(reservation);
        w.data.reserve(reservation);
        if (w.data.capacity() != reservation)
            throw std::invalid_argument("Closure record capacity differs from admission.");
    }
    write_geometry_record_header(w, GeometryRecordKind::ClosureSpanContact);
    if (bounded)
    {
        const auto size = key_wire_size(key);
        if (size > (budget.limit - budget.used) / 3) throw std::invalid_argument("Closure key encoding exceeds budget.");
        budget.add(3 * size);
        { auto encoded = encode_adaptive_key(key); w.bytes(encoded); }
        budget.used -= 3 * size;
    }
    else w.bytes(encode_adaptive_key(key));
    w.bytes(domain);
    w.word(std::numeric_limits<long double>::digits);
    w.word(std::numeric_limits<long double>::max_exponent);
    for (const auto span : spans)
        for (const auto word : PLIC::scalar_packet(span))
            w.word(word);
    for (std::size_t a = 0; a < 3; ++a)
        w.word(factor(a));
    w.word(traces.size());
    for (const auto& face : traces)
    {
        if (face.stem.empty())
            throw std::invalid_argument("Comparable trace requires explicit provenance.");
        w.bytes(face.stem);
        w.word(face.depth);
        w.word(std::uint64_t(face.split));
        w.word(face.terminal);
        w.word(bool(face.face));
        if (face.face)
        {
            if (bounded)
            {
                const auto size = face.face->encoded_incidence_size();
                const auto scratch = face.face->encoding_scratch_bytes();
                if (scratch > budget.limit - budget.used || size > (budget.limit - budget.used - scratch) / 6)
                    throw std::invalid_argument("Closure face encoding exceeds live byte budget.");
                const auto admitted = scratch + 6 * size;
                budget.add(admitted);
                { auto encoded = face.face->encode_incidence(admitted); w.bytes(encoded); }
                budget.used -= admitted;
            }
            else
            {
                add_transient(face.face->encode_incidence(budget.limit).size());
                w.bytes(face.face->encode_incidence(budget.limit));
            }
        }
    }
    if (bounded && w.data.size() != reservation)
        throw std::invalid_argument("Closure record size differs from admission.");
    if (!bounded) budget.add(w.data.capacity());
    return std::move(w.data);
}
/** Keep the full source shell, trace owners and certificate scratch in one
 * transient scope. A consumer declares only its separately admitted survivors. */
template<class Result, class Consumer>
Result prepare_contact_scope(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, Budget& budget,
    std::size_t& trace_records, std::size_t& transient_geometry_peak, Consumer&& consume)
{
    const auto before = budget.used;
    std::size_t retained = 0;
    Result result;
    {
        // This ledger pins only one encoder's full cell and trace supports.
        // Source geometry has a separate phase-long ledger. Conservatively
        // charging shared owners in both scopes avoids ownership inference.
        IdentityBudget identities;
        auto observe_transient = [&]()
        { transient_geometry_peak = std::max(transient_geometry_peak, budget.used - before); };
        auto add_transient = [&](std::size_t bytes)
        {
            budget.add(bytes);
            observe_transient();
        };
        auto shape = kernel.geometry(leaf);
        auto trace = kernel.traces(leaf);
        if (trace.key != leaf.key)
            throw std::invalid_argument("Provider trace identifies another leaf.");
        trace_records += trace.faces.size();
        identities.retain(budget, shape);
        observe_transient();
        // Trace buffers are already live during certificate validation. Admit
        // them before measuring that simultaneous scratch estimate.
        add_transient(trace.faces.capacity() * sizeof(AdaptiveTraceFace) + key_capacity(trace.key));
        for (const auto& face : trace.faces)
        {
            add_transient(face.stem.capacity());
            if (face.face)
            {
                add_transient(sizeof(*face.face) + face.face->exclusive_storage_bytes());
                identities.retain(budget, PLIC::planar_support<ArithmeticMode::Ordinary>(*face.face));
                observe_transient();
            }
        }
        std::array<long double, 3> spans;
        {
            // Preserve the original receiver's certificate validation and its
            // decoded coordinate frame before discarding the transient shell.
            // Closure authorities need only these lossless physical axis spans.
            auto certificate = shape->encode_certificate(budget.limit);
            add_transient(certificate.capacity());
            auto decoded =
                PLIC::ConvexCellGeometry<ArithmeticMode::Ordinary>::decode_certificate(certificate, budget.limit);
            const auto decoded_bytes = decoded->storage_bytes();
            add_transient(decoded_bytes);
            const auto measures = decoded->axis_spans();
            for (std::size_t axis = 0; axis < 3; ++axis)
                spans[axis] = measures[axis].value;
            const auto released = certificate.capacity() + decoded_bytes;
            decoded.reset();
            AdaptiveBytes().swap(certificate);
            budget.used -= released;
        }
        result = consume(trace, spans, add_transient, retained);
    }
    // Geometry, temporary trace owners and ledger pins have been destroyed.
    budget.used -= budget.used - before - retained;
    return result;
}
} // namespace
AdaptiveBytes encode_contact(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, Budget& budget,
    std::size_t& trace_records, std::size_t& transient_geometry_peak)
{
    return prepare_contact_scope<AdaptiveBytes>(leaf, kernel, budget, trace_records, transient_geometry_peak,
        [&](const AdaptiveTrace& trace, const std::array<long double, 3>& spans, auto&& add_transient,
            std::size_t& retained)
        {
            auto result = write_contact(
                leaf.key, leaf.domain, trace.faces, spans,
                [&](std::size_t axis)
                {
                    return leaf.provider == AdaptiveProviderKind::Static
                               ? 3
                               : std::uint64_t(kernel.closure_split(leaf, axis));
                },
                budget, add_transient);
            retained = result.capacity();
            return result;
        });
}
AdaptiveBytes encode_contact(const ContactRecord& contact, Budget& budget)
{
    return encode_contact(contact.key, contact.domain, contact, budget);
}
AdaptiveBytes encode_contact(
    const AdaptiveCellKey& key, std::span<const char> domain, const ContactBody& contact, Budget& budget)
{
    const auto before = budget.used;
    auto result = write_contact(
        key, domain, contact.trace_faces(), contact.spans,
        [&](std::size_t axis) { return contact.factors[axis]; }, budget, [&](std::size_t bytes) { budget.add(bytes); }, true);
    budget.used -= budget.used - before - result.capacity();
    return result;
}
namespace
{
ContactRecord read_contact(std::span<const char> bytes, std::size_t limit, Budget* budget, IdentityBudget* identities)
{
    if (bytes.size() > limit)
        throw std::invalid_argument("Closure contact exceeds byte budget.");
    Reader r{bytes};
    read_geometry_record_header(r, GeometryRecordKind::ClosureSpanContact);
    if (budget)
        budget->add(sizeof(ContactRecord));
    ContactRecord contact;
    if (budget)
    {
        const auto key = r.bytes_view();
        // Canonical ancestry paths contain one encoded word per retained word;
        // the complete key is a conservative preallocation bound.
        contact.key = decode_bounded_key(key, *budget);
    }
    else
        contact.key = decode_adaptive_key(r.bytes());
    auto read_bytes = [&]()
    {
        if (!budget)
            return r.bytes();
        const auto field = r.bytes_view();
        budget->add(field.size());
        AdaptiveBytes result(field.begin(), field.end());
        if (result.capacity() > field.size())
            budget->add(result.capacity() - field.size());
        return result;
    };
    contact.domain = read_bytes();
    if (contact.domain.empty())
        throw std::invalid_argument("Closure contact requires connected-domain provenance.");
    if (r.word() != std::numeric_limits<long double>::digits ||
        r.word() != std::numeric_limits<long double>::max_exponent)
        throw std::invalid_argument("Incompatible closure span scalar representation.");
    for (auto& span : contact.spans)
    {
        PLIC::ScalarPacketWords words;
        for (auto& word : words)
            word = r.word();
        span = PLIC::scalar_from_packet(words);
        if (!(span > 0))
            throw std::invalid_argument("Invalid closure physical axis span.");
    }
    for (auto& factor : contact.factors)
    {
        factor = r.word();
        if (factor > 3)
            throw std::invalid_argument("Invalid closure refinement factor.");
    }
    auto n = r.word();
    if (n > bytes.size() / 32)
        throw std::invalid_argument("Invalid comparable trace count.");
    if (budget)
    {
        if (n > (budget->limit - budget->used) / sizeof(AdaptiveTraceFace))
            throw std::invalid_argument("Closure decoded trace capacity exceeds live byte budget.");
        budget->add(n * sizeof(AdaptiveTraceFace));
        contact.traces.reserve(n);
        if (contact.traces.capacity() > n)
            budget->add((contact.traces.capacity() - n) * sizeof(AdaptiveTraceFace));
    }
    for (std::uint64_t i = 0; i < n; ++i)
    {
        AdaptiveTraceFace face;
        face.stem = read_bytes();
        auto depth = r.word(), split = r.word(), terminal = r.word(), geometry = r.word();
        if (face.stem.empty() || depth > 64 || split > 2 || terminal > 1 || geometry > 1)
            throw std::invalid_argument("Invalid comparable trace.");
        face.depth = unsigned(depth);
        face.split = AdaptiveSplit(split);
        face.terminal = terminal;
        if (geometry)
        {
            if (budget)
            {
                const auto encoded = r.bytes_view();
                // The face decoder admits its native polygon/support before
                // allocation. Its cap is the remaining enclosing live credit.
                face.face = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(
                    encoded, budget->limit - budget->used);
                budget->add(sizeof(*face.face));
                identities->face(*budget, *face.face);
            }
            else
                face.face = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(r.bytes(), limit);
            contact.traces_point_independent = false;
        }
        contact.traces.push_back(std::move(face));
    }
    r.done();
    return contact;
}
} // namespace
ContactRecord decode_contact(std::span<const char> bytes, std::size_t limit)
{
    return read_contact(bytes, limit, nullptr, nullptr);
}
ContactRecord decode_contact(std::span<const char> bytes, Budget& budget, IdentityBudget& identities)
{
    return read_contact(bytes, budget.limit - budget.used, &budget, &identities);
}
namespace
{
template<class Result>
Result prepare_contact_impl(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, Budget& budget,
    IdentityBudget& retained_identities, std::size_t& trace_records, std::size_t& transient_geometry_peak)
{
    return prepare_contact_scope<Result>(leaf, kernel, budget, trace_records, transient_geometry_peak,
        [&](const AdaptiveTrace& trace, const std::array<long double, 3>& spans, auto&& add_transient,
            std::size_t& retained)
        {
            const auto before = budget.used;
            budget.add(sizeof(Result));
            Result result;
            if (leaf.domain.empty())
                throw std::invalid_argument("Closure contact requires connected-domain provenance.");
            if constexpr (std::is_same_v<Result, ContactRecord>)
            {
                const auto bytes = key_capacity(leaf.key);
                budget.add(bytes);
                result.key = leaf.key;
                budget.used -= bytes - key_capacity(result.key);
                budget.add(leaf.domain.size());
                result.domain = leaf.domain;
                if (result.domain.capacity() > leaf.domain.size())
                    budget.add(result.domain.capacity() - leaf.domain.size());
            }
            result.spans = spans;
            for (const auto span : result.spans)
                if (!(span > 0))
                    throw std::invalid_argument("Invalid closure physical axis span.");
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                result.factors[axis] =
                    leaf.provider == AdaptiveProviderKind::Static ? 3 : std::uint64_t(kernel.closure_split(leaf, axis));
                if (result.factors[axis] > 3)
                    throw std::invalid_argument("Invalid closure refinement factor.");
            }
            if (trace.faces.size() > budget.limit / sizeof(AdaptiveTraceFace))
                throw std::invalid_argument("Invalid comparable trace count.");
            budget.add(trace.faces.size() * sizeof(AdaptiveTraceFace));
            result.traces.reserve(trace.faces.size());
            if (result.traces.capacity() > trace.faces.size())
                budget.add((result.traces.capacity() - trace.faces.size()) * sizeof(AdaptiveTraceFace));
            for (const auto& source : trace.faces)
            {
                if (source.stem.empty() || source.depth > 64 || std::uint64_t(source.split) > 2)
                    throw std::invalid_argument("Invalid comparable trace.");
                budget.add(source.stem.size());
                AdaptiveTraceFace face{source.stem, source.depth, source.split, {}, source.terminal};
                if (face.stem.capacity() > source.stem.size())
                    budget.add(face.stem.capacity() - source.stem.size());
                if (source.face)
                {
                    // Preserve the receiver's face validation and physical
                    // coordinate representation without a contact-wire roundtrip.
                    // Count the returned packet without allocating it. Nested
                    // geometry-codec scratch retains its existing inner cap.
                    auto admitted_incidence = source.face->encoded_incidence_size();
                    add_transient(admitted_incidence);
                    auto incidence = source.face->encode_incidence(budget.limit);
                    if (incidence.capacity() > admitted_incidence)
                    {
                        add_transient(incidence.capacity() - admitted_incidence);
                        admitted_incidence = incidence.capacity();
                    }
                    budget.add(sizeof(*face.face));
                    face.face = PLIC::PlanarFace<ArithmeticMode::Ordinary>::decode(incidence, budget.limit);
                    retained_identities.face(budget, *face.face);
                    AdaptiveBytes().swap(incidence);
                    budget.used -= admitted_incidence;
                    result.traces_point_independent = false;
                }
                result.traces.push_back(std::move(face));
            }
            retained = budget.used - before;
            return result;
        });
}
} // namespace
ContactRecord prepare_contact(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, Budget& budget,
    IdentityBudget& retained_identities, std::size_t& trace_records, std::size_t& transient_geometry_peak)
{
    return prepare_contact_impl<ContactRecord>(
        leaf, kernel, budget, retained_identities, trace_records, transient_geometry_peak);
}
ContactBody prepare_contact_body(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, Budget& budget,
    IdentityBudget& retained_identities, std::size_t& trace_records, std::size_t& transient_geometry_peak)
{
    return prepare_contact_impl<ContactBody>(
        leaf, kernel, budget, retained_identities, trace_records, transient_geometry_peak);
}
bool span_exceeds_ratio(long double first, long double second, double ratio)
{
    return first > ratio * second;
}
ContactClosureView closure_view(const ContactBody& contact, int owner)
{
    return {owner, std::span<const long double>(contact.spans), std::span<const std::uint64_t>(contact.factors),
        contact.trace_faces(), contact.traces_point_independent};
}
void join_closure_requirement(
    std::span<const std::uint64_t> factors, std::optional<AdaptiveSplit>& required, std::uint64_t split)
{
    if (split > 2 || std::all_of(factors.begin(), factors.end(), [](auto factor) { return factor == 3; }))
        throw std::invalid_argument("Closed-contact closure needs an unsupported static refinement.");
    auto kind = AdaptiveSplit(split);
    if (!required)
        required = kind;
    else if (*required != kind)
        required = AdaptiveSplit::Isotropic;
}
void join_closure_requirement(const ContactBody& contact, std::optional<AdaptiveSplit>& required, std::uint64_t split)
{
    join_closure_requirement(std::span<const std::uint64_t>(contact.factors), required, split);
}
void grade_contact_spans(const ContactClosureView& first, const ContactClosureView& second, double maximum_span_ratio,
    int local_rank, ClosureRequirements& required)
{
    for (std::size_t axis = 0; axis != 3; ++axis)
    {
        if (first.owner == local_rank && span_exceeds_ratio(first.spans[axis], second.spans[axis], maximum_span_ratio))
            join_closure_requirement(first.factors, required.first, first.factors[axis]);
        if (second.owner == local_rank && span_exceeds_ratio(second.spans[axis], first.spans[axis], maximum_span_ratio))
            join_closure_requirement(second.factors, required.second, second.factors[axis]);
    }
}
void grade_contact_trace_point(const ContactClosureView& first, const ContactClosureView& second,
    unsigned maximum_trace_jump, int local_rank, const PLIC::PointIdentity<ArithmeticMode::Ordinary>& first_physical,
    const PLIC::PointIdentity<ArithmeticMode::Ordinary>& second_physical, ClosureRequirements& required)
{
    for (const auto& first_face : first.trace_faces)
        for (const auto& second_face : second.trace_faces)
        {
            const auto request =
                trace_pair_requirements(first_face, second_face, first_physical, second_physical, maximum_trace_jump);
            if (first.owner == local_rank && request.first)
                join_closure_requirement(first.factors, required.first, std::uint64_t(*request.first));
            if (second.owner == local_rank && request.second)
                join_closure_requirement(second.factors, required.second, std::uint64_t(*request.second));
        }
}
ClosureRequirements trace_pair_requirements(const AdaptiveTraceFace& first, const AdaptiveTraceFace& second,
    const PLIC::PointIdentity<ArithmeticMode::Ordinary>& first_point,
    const PLIC::PointIdentity<ArithmeticMode::Ordinary>& second_point, unsigned maximum_trace_jump)
{
    ClosureRequirements requirements;
    if (first.stem != second.stem || (!first.terminal && !second.terminal) ||
        (first.face && !first.face->contains_point(first_point)) ||
        (second.face && !second.face->contains_point(second_point)))
        return requirements;
    if (first.depth < second.depth && second.depth - first.depth > maximum_trace_jump)
        requirements.first = first.split;
    if (second.depth < first.depth && first.depth - second.depth > maximum_trace_jump)
        requirements.second = second.split;
    return requirements;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
