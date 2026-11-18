/** @file DistributedAdaptiveIdentityBudget.hh
 * @brief Alias-aware identity admission alongside live packet budgets.
 */
#pragma once
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/details/GeometryAllocationOwnerAccess.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** The ledger counter is monotonic even when packet allocations are released
 * from Budget::used. Pins remain alive for this admission scope. */
class IdentityBudget
{
    PLIC::GeometryAllocationLedger allocations_;
    std::size_t charged_ = 0;
    unsigned multiplier_;
    std::size_t native_limit_ = 0;

public:
    explicit IdentityBudget(bool reserve_growth = false) : multiplier_(reserve_growth ? 4 : 1) {}
    void bind_native(const geometry_detail::NativeConstructionAdmission& admission, std::size_t logical_limit)
    {
        if (charged_ || native_limit_)
            throw std::invalid_argument("Native identity bookkeeping must start fresh.");
        PLIC::details::GeometryAllocationOwnerAccess::begin_native_bookkeeping(
            allocations_, admission, charged_, logical_limit);
        native_limit_ = logical_limit;
    }
    [[nodiscard]] std::size_t admitted_bytes() const noexcept { return charged_ * multiplier_; }
    [[nodiscard]] std::size_t metadata_bytes() const noexcept
    {
        return (charged_ - allocations_.payload_bytes()) * multiplier_;
    }
    [[nodiscard]] std::size_t payload_bytes() const noexcept { return allocations_.payload_bytes(); }
    /** Separate PLIC retention-census root; borrowed, with no pin/counter mutation. */
    [[nodiscard]] const PLIC::GeometryAllocationLedger& borrowed_geometry_ledger() const noexcept
    {
        return allocations_;
    }
    template<class Value> void retain(Budget& budget, const Value& value)
    {
        const auto available = (budget.limit - std::min(budget.used, budget.limit)) / multiplier_;
        const auto maximum = std::numeric_limits<std::size_t>::max();
        const auto limit = native_limit_                    ? native_limit_
                           : available > maximum - charged_ ? maximum
                                                            : charged_ + available;
        const auto before = charged_;
        allocations_.retain(value, charged_, limit);
        budget.add((charged_ - before) * multiplier_);
    }
    void face(Budget& budget, const PLIC::PlanarFace<ArithmeticMode::Ordinary>& face)
    {
        const auto bytes = face.exclusive_storage_bytes();
        if (bytes > (budget.limit - std::min(budget.used, budget.limit)) / multiplier_)
            throw std::invalid_argument("Shared face admission exceeds adaptive byte budget.");
        budget.add(bytes * multiplier_);
        retain(budget, PLIC::planar_support<ArithmeticMode::Ordinary>(face));
    }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
