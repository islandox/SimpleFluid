/** @file CellWorkspace.hh
 * @brief Scoped active-shell materialization with caller remaining admission.
 */
#pragma once
#include "FVM/details/GeneralVoFInternal.hh"
#include <algorithm>
#include <cstddef>
#include <optional>
namespace SimpleFluid::VoF::general_vof_detail
{
/** This scope owns temporary shell roots, never the caller's persistent Work
 * ledger. Surviving outputs must enter the established output provenance ledger. */
class CellWorkspace
{
public:
    CellWorkspace(ArithmeticMode mode, geometry::swept_volume::Work& work, std::size_t maximum_operations = 0)
        : mode_(mode), work_(&work), workspace_(std::in_place, limits(mode, work, maximum_operations))
    {
    }
    CellWorkspace(ArithmeticMode mode, wire::Budget& budget, std::size_t maximum_operations = 0)
        : mode_(mode), budget_(&budget), maximum_operations_(maximum_operations)
    {
    }
    template<ArithmeticMode Mode>
    PLIC::ConvexGeometryPtr<Mode> materialize(const PLIC::CellGeometryView<Mode>& view)
        requires(Mode != ArithmeticMode::Exact)
    {
        if (Mode != mode_)
            throw std::invalid_argument("VoF active cell workspace arithmetic mode differs.");
        if (budget_)
        {
            // wire::Budget is byte-only: numerical guards belong to each
            // factory as before native-view migration. A fresh scoped workspace
            // preserves those per-factory guards and releases discarded raw
            // scratch roots immediately. All spent bytes remain monotone; a
            // surviving output must be admitted immediately by its owner.
            auto limits = geometry_limits(Mode, budget_->limit - budget_->used);
            limits.maximum_operations = maximum_operations_;
            PLIC::CellGeometryOperationWorkspace temporary(limits);
            PLIC::ConvexGeometryPtr<Mode> result;
            try
            {
                result = view.materialize(temporary);
            }
            catch (...)
            {
                budget_->add(temporary.bytes_used());
                throw;
            }
            budget_->add(temporary.bytes_used());
            return result;
        }
        // Work-backed operations explicitly own cumulative numerical quotas.
        auto& workspace = *workspace_;
        auto remaining = workspace.remaining_limits(Mode);
        if (work_)
        {
            remaining.byte_budget = work_->limits.byte_budget - std::min(work_->bytes, work_->limits.byte_budget);
            remaining.maximum_pair_tests =
                work_->limits.maximum_pair_tests - std::min(work_->pair_tests, work_->limits.maximum_pair_tests);
        }
        else
            remaining.byte_budget = budget_->limit - budget_->used;
        workspace.restrict_remaining(remaining);
        PLIC::ConvexGeometryPtr<Mode> geometry;
        try
        {
            geometry = view.materialize(workspace);
        }
        catch (...)
        {
            // Optional candidate callers can recover from arithmetic failure;
            // their already consumed construction prefix remains charged.
            publish_charge();
            throw;
        }
        publish_charge();
        return geometry;
    }
    [[nodiscard]] PLIC::CellGeometryOperationWorkspace& operation() { return workspace_.value(); }
    void publish_charge()
    {
        if (!workspace_)
            return;
        const auto bytes = workspace_->bytes_used(), pairs = workspace_->pair_tests_used();
        if (work_)
        {
            if (pairs - charged_pairs_ >
                work_->limits.maximum_pair_tests - std::min(work_->pair_tests, work_->limits.maximum_pair_tests))
                throw PLIC::GeometryError(
                    PLIC::GeometryErrorCode::ResourceLimit, "VoF cell materialization exhausts remaining pair work.");
            work_->charge(bytes - charged_bytes_);
            work_->pair_tests += pairs - charged_pairs_;
        }
        else
            budget_->add(bytes - charged_bytes_);
        charged_bytes_ = bytes;
        charged_pairs_ = pairs;
    }

private:
    static PLIC::GeometryLimits limits(
        ArithmeticMode mode, const geometry::swept_volume::Work& work, std::size_t maximum_operations)
    {
        auto limits = geometry_limits(mode, work.limits.byte_budget - std::min(work.bytes, work.limits.byte_budget));
        limits.maximum_operations = maximum_operations;
        limits.maximum_pair_tests =
            work.limits.maximum_pair_tests - std::min(work.pair_tests, work.limits.maximum_pair_tests);
        return limits;
    }
    ArithmeticMode mode_;
    geometry::swept_volume::Work* work_ = nullptr;
    wire::Budget* budget_ = nullptr;
    std::optional<PLIC::CellGeometryOperationWorkspace> workspace_;
    std::size_t maximum_operations_ = 0;
    std::size_t charged_bytes_ = 0, charged_pairs_ = 0;
};
} // namespace SimpleFluid::VoF::general_vof_detail
