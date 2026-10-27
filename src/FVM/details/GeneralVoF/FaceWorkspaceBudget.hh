/** @file FaceWorkspaceBudget.hh
 * @brief Preparation-retained and face-local conservative storage allowances.
 */
#pragma once
#include "geometry/swept_volume/FiniteVoFTransport.hh"
#include "geometry/swept_volume/OrdinaryAggregate.hh"
#include <cassert>

namespace SimpleFluid::VoF::general_vof_detail
{
struct FaceWorkspaceTotals
{
    std::size_t scratch_construction_bytes = 0, retired_retained_bytes = 0, operations = 0, peak_bytes = 0;
};
/** Declare before every container belonging to a face, so its accounting ends
 * only after those containers (including the returned aggregate) are destroyed.
 * Pending retained charges remain conservative until destruction. Only an
 * explicitly committed surviving delta is retained after the face. Aggregate
 * construction traffic remains reserved for the entire face, including copies.
 */
class FaceWorkspaceBudget
{
public:
    SIMPLEFLUID_FVM_LOCAL FaceWorkspaceBudget(
        geometry::swept_volume::Work&, const geometry::swept_volume::OrdinaryAggregateContext&, FaceWorkspaceTotals&);
    SIMPLEFLUID_FVM_LOCAL ~FaceWorkspaceBudget() noexcept;
    FaceWorkspaceBudget(const FaceWorkspaceBudget&) = delete;
    FaceWorkspaceBudget& operator=(const FaceWorkspaceBudget&) = delete;
    [[nodiscard]] geometry::swept_volume::Work& setup_work() noexcept { return setup_; }
    [[nodiscard]] std::size_t retained_entry() const noexcept { return retained_entry_; }
    [[nodiscard]] std::size_t retained_bytes() const noexcept { return retained_.bytes; }
    [[nodiscard]] std::size_t retained_after() const noexcept { return retained_entry_ + committed_bytes_; }
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t scratch_bytes() const noexcept;
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t remaining_bytes() const noexcept;
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t peak_bytes() const noexcept;
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t construction_bytes() const noexcept;
    /** Aggregate-context operations only; reach and setup predicate work are
     * not included in this counter. */
    [[nodiscard]] SIMPLEFLUID_FVM_LOCAL std::size_t operations() const noexcept;
    SIMPLEFLUID_FVM_LOCAL void charge_retained(std::size_t);
    /** Call from a ResourceLimit catch while this workspace is alive. Preserve
     * the original error as a nested cause and the existing failure scope. */
    [[noreturn]] SIMPLEFLUID_FVM_LOCAL void rethrow_resource(const PLIC::GeometryError&) const;
    template<class Function> decltype(auto) checked_action(Function&& action) const
    {
        try
        {
            return std::forward<Function>(action)();
        }
        catch (const PLIC::GeometryError& error)
        {
            if (error.code() != PLIC::GeometryErrorCode::ResourceLimit)
                throw;
            rethrow_resource(error);
        }
    }
    /** Validate before publication. commit_retained is the final, nonthrowing
     * step after publishing via a nonthrowing move into admitted capacity. */
    SIMPLEFLUID_FVM_LOCAL void validate_retained_commit(std::size_t) const;
    void commit_retained(std::size_t actual_surviving_delta) noexcept
    {
        assert(actual_surviving_delta <= retained_.bytes - retained_entry_);
        committed_bytes_ = actual_surviving_delta;
    }

    template<class T> void reserve_retained_slot(std::vector<T>& values)
    {
        if (values.size() >= retained_.limits.maximum_atoms)
            throw PLIC::GeometryError(
                PLIC::GeometryErrorCode::ResourceLimit, "Finite VoF atom buffer capacity is exhausted.");
        if (values.size() == values.capacity())
        {
            const auto capacity = values.capacity();
            if (capacity > std::numeric_limits<std::size_t>::max() / 2)
                throw PLIC::GeometryError(
                    PLIC::GeometryErrorCode::ResourceLimit, "Finite VoF atom capacity overflows.");
            const auto next = capacity ? 2 * capacity : 1;
            if (next > std::numeric_limits<std::size_t>::max() / sizeof(T))
                throw PLIC::GeometryError(
                    PLIC::GeometryErrorCode::ResourceLimit, "Finite VoF atom byte capacity overflows.");
            // Preserve Work::push's conservative whole-capacity growth charge:
            // the old and new buffers coexist during reserve.
            charge_retained(next * sizeof(T));
            values.reserve(next);
        }
    }
    /** Populate a face-local, unpublished owner. This ledger cannot undo an
     * external vector mutation: surviving output must use the final reserved,
     * validated, nonthrowing publication/commit sequence instead. */
    template<class T> void push_retained(std::vector<T>& values, T value)
    {
        reserve_retained_slot(values);
        values.push_back(std::move(value));
    }

private:
    geometry::swept_volume::Work& retained_;
    const geometry::swept_volume::OrdinaryAggregateContext& progress_;
    FaceWorkspaceTotals& totals_;
    std::size_t retained_entry_, committed_bytes_ = 0;
    geometry::swept_volume::Work setup_;
};
} // namespace SimpleFluid::VoF::general_vof_detail
