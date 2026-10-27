/** @file FaceWorkspaceBudget.cc
 * @brief Conservative overlapping lifetime admission at the face boundary.
 */
#include "FVM/details/GeneralVoF/FaceWorkspaceBudget.hh"
#include <cstdio>

namespace SimpleFluid::VoF::general_vof_detail
{
namespace
{
std::size_t saturated_add(std::size_t a, std::size_t b) noexcept
{
    return a + std::min(b, std::numeric_limits<std::size_t>::max() - a);
}
} // namespace
FaceWorkspaceBudget::FaceWorkspaceBudget(geometry::swept_volume::Work& retained,
    const geometry::swept_volume::OrdinaryAggregateContext& progress, FaceWorkspaceTotals& totals)
    : retained_(retained), progress_(progress), totals_(totals), retained_entry_(retained.bytes),
      setup_{retained.limits}
{
    setup_.arithmetic_mode = retained.arithmetic_mode;
    setup_.volume_tolerance = retained.volume_tolerance;
    setup_.limits.byte_budget -= std::min(retained.bytes, setup_.limits.byte_budget);
    setup_.limits.maximum_pair_tests -= std::min(retained.pair_tests, setup_.limits.maximum_pair_tests);
}
FaceWorkspaceBudget::~FaceWorkspaceBudget() noexcept
{
    totals_.peak_bytes = peak_bytes();
    totals_.scratch_construction_bytes = saturated_add(totals_.scratch_construction_bytes, scratch_bytes());
    totals_.operations = operations();
    retained_.pair_tests = saturated_add(retained_.pair_tests, saturated_add(setup_.pair_tests, progress_.pairs));
    const auto pending_bytes = retained_.bytes - retained_entry_;
    totals_.retired_retained_bytes = saturated_add(totals_.retired_retained_bytes, pending_bytes - committed_bytes_);
    // All temporary and unpublished owners must already have been destroyed.
    retained_.bytes = retained_after();
}
std::size_t FaceWorkspaceBudget::scratch_bytes() const noexcept
{
    return saturated_add(setup_.bytes, progress_.bytes);
}
std::size_t FaceWorkspaceBudget::remaining_bytes() const noexcept
{
    const auto used = saturated_add(retained_.bytes, scratch_bytes());
    return retained_.limits.byte_budget - std::min(used, retained_.limits.byte_budget);
}
std::size_t FaceWorkspaceBudget::peak_bytes() const noexcept
{
    // Both ledgers are monotone while this scope exists.
    return std::max(totals_.peak_bytes, saturated_add(retained_.bytes, scratch_bytes()));
}
std::size_t FaceWorkspaceBudget::construction_bytes() const noexcept
{
    return saturated_add(retained_.bytes, saturated_add(totals_.retired_retained_bytes,
                                              saturated_add(totals_.scratch_construction_bytes, scratch_bytes())));
}
std::size_t FaceWorkspaceBudget::operations() const noexcept
{
    return saturated_add(totals_.operations, progress_.operations);
}
void FaceWorkspaceBudget::charge_retained(std::size_t bytes)
{
    if (bytes > remaining_bytes())
    {
        std::array<char, 320> message{};
        (void) std::snprintf(message.data(), message.size(),
            "Finite VoF storage budget exhausted: retained_bytes=%zu face_setup_bytes=%zu "
            "aggregate_scratch_bytes=%zu byte_budget=%zu requested_retained_bytes=%zu.",
            retained_.bytes, setup_.bytes, progress_.bytes, retained_.limits.byte_budget, bytes);
        throw PLIC::GeometryError(PLIC::GeometryErrorCode::ResourceLimit, message.data());
    }
    retained_.charge(bytes);
}
void FaceWorkspaceBudget::validate_retained_commit(std::size_t bytes) const
{
    if (bytes > retained_.bytes - retained_entry_)
        throw std::invalid_argument("Finite VoF retained output commit exceeds its admitted storage.");
}
void FaceWorkspaceBudget::rethrow_resource(const PLIC::GeometryError& error) const
{
    namespace failure = ::SimpleFluid::detail::vof_failure;
    auto& state = failure::context_state();
    auto* record = state.captured ? &state.unwound : state.active;
    if (record)
    {
        record->bytes_used = saturated_add(retained_.bytes, scratch_bytes());
        record->byte_budget = retained_.limits.byte_budget;
        record->pair_tests = saturated_add(retained_.pair_tests, saturated_add(setup_.pair_tests, progress_.pairs));
        record->pair_budget = retained_.limits.maximum_pair_tests;
        for (const auto field : {failure::Field::BytesUsed, failure::Field::ByteBudget, failure::Field::PairTests,
                 failure::Field::PairBudget})
            record->mark(field);
    }
    std::array<char, 256> prefix{};
    (void) std::snprintf(prefix.data(), prefix.size(),
        "VoF face resources: retained_bytes=%zu setup_bytes=%zu aggregate_bytes=%zu storage_limit=%zu; ",
        retained_.bytes, setup_.bytes, progress_.bytes, retained_.limits.byte_budget);
    std::throw_with_nested(PLIC::GeometryError(error.code(), std::string(prefix.data()) + error.what()));
}
} // namespace SimpleFluid::VoF::general_vof_detail
