#include "PooledCellPoints.hh"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
constexpr std::size_t node_bytes = sizeof(IncidencePools::Point) + 4 * sizeof(void*);
}
PooledCellPoints::PooledCellPoints(Budget& budget, IncidencePools& pools) : budget_(&budget), pools_(&pools) {}
PooledCellPoints::PooledCellPoints(const PooledCellPoints& source, Budget& budget, IncidencePools& pools)
    : budget_(&budget), pools_(&pools)
{
    try
    {
        admit(source.counts_.size() * sizeof(decltype(counts_)::value_type));
        admit(source.before_.size() * sizeof(Slot));
        counts_.reserve(source.counts_.size());
        before_.reserve(source.before_.size());
        if (counts_.capacity() != source.counts_.size() || before_.capacity() != source.before_.size())
            throw std::invalid_argument("Cell point fork capacity differs from admission.");
        counts_.assign(source.counts_.begin(), source.counts_.end());
        before_.assign(source.before_.begin(), source.before_.end());
    }
    catch (...)
    {
        decltype(counts_)().swap(counts_);
        decltype(before_)().swap(before_);
        budget_->used -= charged_;
        charged_ = 0;
        throw;
    }
}
PooledCellPoints::~PooledCellPoints()
{
    decltype(counts_)().swap(counts_);
    decltype(before_)().swap(before_);
    budget_->used -= charged_;
}
void PooledCellPoints::admit(std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() - charged_)
        throw std::invalid_argument("Cell point charge overflow.");
    budget_->add(n);
    charged_ += n;
}
void PooledCellPoints::release(std::size_t n)
{
    charged_ -= n;
    budget_->used -= n;
}
void PooledCellPoints::begin_delta()
{
    if (before_.capacity() < counts_.size())
    {
        const auto previous = before_.capacity() * sizeof(Slot);
        admit(counts_.size() * sizeof(Slot));
        before_.reserve(counts_.size());
        if (before_.capacity() != counts_.size())
            throw std::invalid_argument("Cell point snapshot capacity differs from admission.");
        release(previous);
    }
    before_.clear();
    for (const auto& [slot, count] : counts_)
        before_.push_back(slot);
}
void PooledCellPoints::add(Slot slot)
{
    (void) pools_->points.get(slot);
    auto found = std::lower_bound(
        counts_.begin(), counts_.end(), slot, [](const auto& entry, Slot value) { return entry.first < value; });
    if (found != counts_.end() && found->first == slot)
    {
        if (found->second == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("Physical point contribution count overflow.");
        ++found->second;
        return;
    }
    const auto offset = found - counts_.begin();
    if (counts_.size() == counts_.capacity())
    {
        const auto previous = counts_.capacity();
        if (previous > budget_->limit / sizeof(decltype(counts_)::value_type) / 2)
            throw std::invalid_argument("Cell point capacity overflow.");
        const auto next = previous ? previous * 2 : 1;
        admit(next * sizeof(decltype(counts_)::value_type));
        counts_.reserve(next);
        if (counts_.capacity() != next)
            throw std::invalid_argument("Cell point capacity differs from admission.");
        release(previous * sizeof(decltype(counts_)::value_type));
    }
    counts_.insert(counts_.begin() + offset, {slot, 1});
}
void PooledCellPoints::finish_delta()
{
    const auto bytes = before_.capacity() * sizeof(Slot);
    std::vector<Slot>().swap(before_);
    release(bytes);
}
void PooledCellPoints::remove(Slot slot)
{
    auto found = std::lower_bound(
        counts_.begin(), counts_.end(), slot, [](const auto& entry, Slot value) { return entry.first < value; });
    if (found == counts_.end() || found->first != slot)
        throw std::invalid_argument("Unknown retired physical point contribution.");
    if (!--found->second)
        counts_.erase(found);
}
bool PooledCellPoints::changed() const
{
    if (counts_.size() != before_.size())
        return true;
    auto previous = before_.begin();
    for (const auto& [slot, count] : counts_)
        if (slot != *previous++)
            return true;
    return false;
}
ContactIncidenceRow PooledCellPoints::row(const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel)
{
    return make_row(leaf, kernel, nullptr, true);
}
ContactIncidenceRow PooledCellPoints::row(
    const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, ContactPointBatch& batch, bool keep_identity)
{
    return make_row(leaf, kernel, &batch, keep_identity);
}
ContactIncidenceRow PooledCellPoints::make_row(
    const AdaptiveLeaf& leaf, const AdaptiveProviderKernel& kernel, ContactPointBatch* batch, bool keep_identity)
{
    if (leaf.domain.empty() || counts_.empty())
        throw std::invalid_argument("Incomplete spatial contact incidence row.");
    // Transfer only row-exclusive bytes to its owner. Pool pages and the
    // shared batch directory retain their independent admission credits.
    struct RowAdmission
    {
        Budget& budget;
        std::size_t bytes = 0;
        bool transferred = false;
        ~RowAdmission()
        {
            if (!transferred)
                budget.used -= bytes;
        }
        void add(std::size_t n)
        {
            budget.add(n);
            bytes += n;
        }
    } admission{*budget_};
    if (keep_identity)
    {
        admission.add(key_capacity(leaf.key));
        admission.add(leaf.domain.size());
    }
    auto base_admission = admission.bytes;
    const auto row_element_bound = batch ? 2 * sizeof(std::uint64_t) : sizeof(ContactPointIncidence);
    if (counts_.size() > budget_->limit / row_element_bound)
        throw std::invalid_argument("Contact incidence row exceeds budget.");
    if (!batch)
        admission.add(counts_.size() * sizeof(ContactPointIncidence));
    ContactIncidenceRow result;
    result.revision = leaf.revision;
    if (keep_identity)
    {
        result.key = leaf.key;
        result.domain = leaf.domain;
    }
    const auto copied_base = key_capacity(result.key) + result.domain.capacity();
    if (copied_base > base_admission)
        throw std::invalid_argument("Contact row key capacity exceeds its construction admission.");
    // Copies can discard unused ancestry capacity. Transfer only the buffers
    // actually owned by the row; the caller's source keeps its own allowance.
    budget_->used -= base_admission - copied_base;
    admission.bytes -= base_admission - copied_base;
    base_admission = copied_base;
    if (batch)
        result.points = ContactPointArray::pooled(*batch, counts_.size(), *budget_);
    else
        result.points.reserve(counts_.size());
    if (batch)
        admission.bytes = base_admission + result.points.storage_bytes();
    if (result.points.capacity() != counts_.size())
        throw std::invalid_argument("Contact incidence row capacity differs from admission.");
    // Scoped transient storage is admitted before allocation and retired after
    // the containers die. Ordinary point identities, including contact_point's
    // result, carry their payload inline in the admitted row and set nodes.
    const auto count = counts_.size();
    if (count > budget_->limit / (sizeof(Slot) + node_bytes))
        throw std::invalid_argument("Contact row temporary budget exhausted.");
    const auto scratch = count * (sizeof(Slot) + node_bytes);
    admit(scratch);
    try
    {
        std::vector<Slot> order;
        order.reserve(count);
        for (const auto& [slot, refs] : counts_)
            order.push_back(slot);
        std::sort(
            order.begin(), order.end(), [&](Slot a, Slot b) { return pools_->points.get(a) < pools_->points.get(b); });
        std::set<IncidencePools::Point> normalized;
        for (const auto slot : order)
        {
            const auto& point = pools_->points.get(slot);
            auto contact = kernel.contact_point(leaf, point);
            if (!normalized.insert(contact).second)
                throw std::invalid_argument("Duplicate cell incidence at a normalized closure point.");
            if (batch)
            {
                // Preserve physical sorting and contact_point call order. Pool
                // lookup work restores the caller's floating-point environment.
                const auto contact_slot = pools_->points.intern(std::move(contact));
                if (batch->source_backed())
                    result.points.push_reference(
                        batch->capture_source(slot), batch->capture_source(contact_slot), *budget_);
                else
                    result.points.push_reference(batch->capture(slot), batch->capture(contact_slot), *budget_);
            }
            else
                result.points.push_back({std::move(contact), point});
        }
    }
    catch (...)
    {
        release(scratch);
        if (batch)
            admission.bytes = base_admission + result.points.storage_bytes();
        throw;
    }
    release(scratch);
    if (batch)
    {
        admission.transferred = true;
        return result;
    }
    const auto point_reservation = counts_.size() * sizeof(ContactPointIncidence);
    if (result.points.storage_bytes() > point_reservation)
        throw std::invalid_argument("Contact point arrays exceed their construction admission.");
    budget_->used -= point_reservation - result.points.storage_bytes();
    admission.transferred = true;
    return result;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
