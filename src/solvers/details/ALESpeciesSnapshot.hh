#pragma once

#include "solvers/details/ALESpeciesConvergence.hh"

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail::ale_species
{
struct PhaseOmissions
{
    std::size_t liquid_only_species{}, nonliquid_events{}, gas_subsets{};
};

struct SnapshotLayout
{
    std::size_t cell_values{}, values{}, allowances{};
    std::size_t logical_cell_values{}, logical_event_end{}, logical_values{};

    static SnapshotLayout typed(std::size_t owned_cells, std::size_t species, std::size_t populations,
        std::size_t events, std::size_t subsets, PhaseOmissions omitted = {})
    {
        const auto add = [](std::size_t left, std::size_t right) {
            if (right > std::numeric_limits<std::size_t>::max() - left)
                throw std::length_error("ALE species snapshot size overflow.");
            return left + right;
        };
        const auto multiply = [](std::size_t left, std::size_t right) {
            if (left && right > std::numeric_limits<std::size_t>::max() / left)
                throw std::length_error("ALE species snapshot size overflow.");
            return left * right;
        };
        if (omitted.liquid_only_species > species || omitted.nonliquid_events > events || omitted.gas_subsets > subsets)
            throw std::logic_error("ALE species phase omissions exceed the convergence schema.");
        const auto compartments = add(4, populations);
        const auto inventories = multiply(compartments, species);
        const auto carried_and_heat = add(multiply(8, compartments), 3);
        const auto per_cell = add(add(multiply(7, inventories), add(populations, 4)), multiply(2, carried_and_heat));
        const auto omitted_per_cell = multiply(7, multiply(omitted.liquid_only_species, compartments - 1));
        const auto logical_cells = multiply(owned_cells, per_cell);
        const auto logical_event_end = add(logical_cells, multiply(8, events));
        const auto subset_slots = multiply(2, subsets);
        const auto cell_values = multiply(owned_cells, per_cell - omitted_per_cell);
        const auto retained_events = events - omitted.nonliquid_events;
        return {cell_values, add(add(cell_values, multiply(8, retained_events)), subset_slots - omitted.gas_subsets),
            add(retained_events, subsets), logical_cells, logical_event_end, add(logical_event_end, subset_slots)};
    }

    bool operator==(const SnapshotLayout&) const = default;

    bool signed_slot(std::size_t index) const noexcept
    {
        if (index < logical_cell_values) return false;
        if (index < logical_event_end) return (index - logical_cell_values) % 8 == 7;
        return (index - logical_event_end) % 2 == 1;
    }
};

struct SnapshotMaximum
{
    double residual{};
    std::size_t index = std::numeric_limits<std::size_t>::max();
    ConvergenceEntry before{}, after{};
};

class SnapshotReference
{
  public:
    SnapshotReference() = default;

    template<class Visit>
    static SnapshotReference capture(const SnapshotLayout& layout, Visit&& visit)
    {
        SnapshotReference ref;
        ref.layout_ = layout;
        if (layout.values) ref.values_ = std::make_unique_for_overwrite<double[]>(layout.values);
        if (layout.allowances) ref.allowances_ = std::make_unique_for_overwrite<long double[]>(layout.allowances);
        Cursor<false> cursor(ref);
        visit(cursor);
        cursor.finish();
        return ref;
    }

    template<class Visit>
    static SnapshotReference stage(SnapshotReference ref, const SnapshotLayout& layout, Visit&& visit,
        SnapshotMaximum& maximum)
    {
        if (ref.layout_ != layout || (layout.values && !ref.values_) || (layout.allowances && !ref.allowances_))
            throw std::logic_error("ALE species/event schema changed during a trial.");
        Cursor<true> cursor(ref);
        visit(cursor);
        cursor.finish();
        maximum = cursor.maximum;
        return ref;
    }

    bool empty() const noexcept { return !values_; }

  private:
    template<bool Compare>
    class Cursor
    {
      public:
        explicit Cursor(SnapshotReference& reference) : ref(reference) {}
        void skip(std::size_t count)
        {
            if (count > ref.layout_.logical_values - ordinal) schema_error();
            ordinal += count;
        }
        void ordinary(double value)
        {
            if (index >= ref.layout_.values || ordinal >= ref.layout_.logical_values || ref.layout_.signed_slot(ordinal))
                schema_error();
            if constexpr (Compare) observe({ref.values_[index]}, {value});
            ref.values_[index++] = value;
            ++ordinal;
        }
        void signed_net(double value, long double allowance)
        {
            if (index >= ref.layout_.values || ordinal >= ref.layout_.logical_values || !ref.layout_.signed_slot(ordinal)
                || signed_index >= ref.layout_.allowances)
                schema_error();
            if constexpr (Compare) observe({ref.values_[index], ref.allowances_[signed_index]}, {value, allowance});
            ref.values_[index++] = value;
            ref.allowances_[signed_index++] = allowance;
            ++ordinal;
        }
        void finish() const
        {
            if (index != ref.layout_.values || signed_index != ref.layout_.allowances ||
                ordinal != ref.layout_.logical_values) schema_error();
        }
        SnapshotMaximum maximum;

      private:
        [[noreturn]] static void schema_error() { throw std::logic_error("ALE species/event snapshot schema mismatch."); }
        void observe(ConvergenceEntry before, ConvergenceEntry after)
        {
            const auto residual = relative_change(before, after);
            if (residual > maximum.residual) maximum = {residual, ordinal, before, after};
        }
        SnapshotReference& ref;
        std::size_t index{}, signed_index{}, ordinal{};
    };

    SnapshotLayout layout_;
    std::unique_ptr<double[]> values_;
    std::unique_ptr<long double[]> allowances_;
};
}
