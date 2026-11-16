/** @file DistributedAdaptiveClosureRecords.hh
 * @brief Private immutable closure records and shared grading predicates.
 */
#pragma once
#include "utils/export_helper.hh"
#include "geometry/mesh/AMR/AdaptiveProviderKernel.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveIdentityBudget.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct SIMPLEFLUID_LOCAL ContactBody
{
    std::array<long double, 3> spans{};
    std::array<std::uint64_t, 3> factors{};
    std::vector<AdaptiveTraceFace> traces;
    bool traces_point_independent = true;
    std::shared_ptr<const std::vector<AdaptiveTraceFace>> shared_traces;
    [[nodiscard]] std::span<const AdaptiveTraceFace> trace_faces() const noexcept
    {
        return shared_traces ? std::span<const AdaptiveTraceFace>(*shared_traces) :
                               std::span<const AdaptiveTraceFace>(traces);
    }
};
/** Owning wire and compatibility record. The spatial cache uses only its
 * physical body and resolves identity through the authenticated occurrence. */
struct SIMPLEFLUID_LOCAL ContactRecord : ContactBody
{
    AdaptiveCellKey key;
    std::size_t identity = 0;
    AdaptiveBytes domain;
};
struct SIMPLEFLUID_LOCAL ClosureRequirements
{
    std::optional<AdaptiveSplit> first, second;
};
/** Non-owning grading view over one side of a contact pair.
 *
 * The explicit backend borrows the retained ContactRecord fields; a Cartesian
 * backend will supply the same facts from its retained source and topology.
 * No pointer or span may outlive the pair evaluation. `spans` and `factors`
 * always carry exactly three elements. */
struct SIMPLEFLUID_LOCAL ContactClosureView
{
    int owner = -1;
    std::span<const long double> spans;
    std::span<const std::uint64_t> factors;
    std::span<const AdaptiveTraceFace> trace_faces;
    bool traces_point_independent = true;
};
[[nodiscard]] SIMPLEFLUID_LOCAL ContactClosureView closure_view(const ContactBody&, int owner);
[[nodiscard]] SIMPLEFLUID_LOCAL std::size_t key_capacity(const AdaptiveCellKey&);
[[nodiscard]] SIMPLEFLUID_LOCAL std::size_t key_wire_size(const AdaptiveCellKey&);
[[nodiscard]] SIMPLEFLUID_LOCAL AdaptiveCellKey decode_bounded_key(std::span<const char>, Budget&);
/** Legacy record codec: its payload, field order and admission order are fixed. */
[[nodiscard]] SIMPLEFLUID_LOCAL AdaptiveBytes encode_contact(const AdaptiveLeaf&, const AdaptiveProviderKernel&,
    Budget&, std::size_t& trace_records, std::size_t& transient_geometry_peak);
[[nodiscard]] SIMPLEFLUID_LOCAL AdaptiveBytes encode_contact(const ContactRecord&, Budget&);
[[nodiscard]] SIMPLEFLUID_LOCAL AdaptiveBytes encode_contact(
    const AdaptiveCellKey&, std::span<const char> domain, const ContactBody&, Budget&);
[[nodiscard]] SIMPLEFLUID_LOCAL ContactRecord decode_contact(std::span<const char>, std::size_t limit);
/** Owning decode with preallocation admission against the caller's simultaneous
 * live state. Encoded input remains charged by the caller until return. */
[[nodiscard]] SIMPLEFLUID_LOCAL ContactRecord decode_contact(std::span<const char>, Budget&, IdentityBudget&);
/** Construct an owning record without contact serialization. The caller keeps
 * retained_identities alive as long as the record and retains the returned
 * budget charges until retirement. Includes sizeof(ContactRecord), key/domain,
 * trace capacities and optional face owners; callers admit outer containers. */
[[nodiscard]] SIMPLEFLUID_LOCAL ContactRecord prepare_contact(const AdaptiveLeaf&, const AdaptiveProviderKernel&,
    Budget&, IdentityBudget& retained_identities, std::size_t& trace_records, std::size_t& transient_geometry_peak);
[[nodiscard]] SIMPLEFLUID_LOCAL ContactBody prepare_contact_body(const AdaptiveLeaf&, const AdaptiveProviderKernel&,
    Budget&, IdentityBudget& retained_identities, std::size_t& trace_records, std::size_t& transient_geometry_peak);
[[nodiscard]] SIMPLEFLUID_LOCAL bool span_exceeds_ratio(long double first, long double second, double ratio);
SIMPLEFLUID_LOCAL void join_closure_requirement(
    const ContactBody&, std::optional<AdaptiveSplit>& required, std::uint64_t split);
/** Factor-only join for grading views; same policy as the record overload. */
SIMPLEFLUID_LOCAL void join_closure_requirement(
    std::span<const std::uint64_t> factors, std::optional<AdaptiveSplit>& required, std::uint64_t split);
/** Shared span grading over two closure views. Each owner's requirement is
 * joined only for the local rank, preserving the existing pair policy. */
SIMPLEFLUID_LOCAL void grade_contact_spans(const ContactClosureView& first, const ContactClosureView& second,
    double maximum_span_ratio, int local_rank, ClosureRequirements& required);
/** Shared trace grading for one common contact point. The caller owns work
 * charging and statistics; this applies the span-ratio/trace-jump policy only. */
SIMPLEFLUID_LOCAL void grade_contact_trace_point(const ContactClosureView& first, const ContactClosureView& second,
    unsigned maximum_trace_jump, int local_rank, const PLIC::PointIdentity<ArithmeticMode::Ordinary>& first_physical,
    const PLIC::PointIdentity<ArithmeticMode::Ordinary>& second_physical, ClosureRequirements& required);
[[nodiscard]] SIMPLEFLUID_LOCAL ClosureRequirements trace_pair_requirements(const AdaptiveTraceFace& first,
    const AdaptiveTraceFace& second, const PLIC::PointIdentity<ArithmeticMode::Ordinary>& first_point,
    const PLIC::PointIdentity<ArithmeticMode::Ordinary>& second_point, unsigned maximum_trace_jump);
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
