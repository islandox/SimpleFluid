#include "ContactTracePool.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
std::size_t sum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::invalid_argument("Contact trace pool size overflow.");
    return a + b;
}
std::size_t product(std::size_t a, std::size_t b)
{
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Contact trace pool capacity overflow.");
    return a * b;
}
std::size_t buffer_bytes(const std::vector<AdaptiveTraceFace>& faces)
{
    auto bytes = product(faces.capacity(), sizeof(AdaptiveTraceFace));
    for (const auto& face : faces)
        bytes = sum(bytes, face.stem.capacity());
    return bytes;
}
bool less(std::span<const AdaptiveTraceFace> a, std::span<const AdaptiveTraceFace> b) noexcept
{
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
        [](const auto& first, const auto& second)
        {
            if (first.stem != second.stem)
                return first.stem < second.stem;
            if (first.depth != second.depth)
                return first.depth < second.depth;
            if (first.split != second.split)
                return first.split < second.split;
            return first.terminal < second.terminal;
        });
}
constexpr std::size_t owner_bytes = sizeof(std::vector<AdaptiveTraceFace>) + 128;
} // namespace

bool ContactTracePool::Less::operator()(const Owner& a, const Owner& b) const noexcept
{
    return less(*a, *b);
}
bool ContactTracePool::Less::operator()(const Owner& a, std::span<const AdaptiveTraceFace> b) const noexcept
{
    return less(*a, b);
}
bool ContactTracePool::Less::operator()(std::span<const AdaptiveTraceFace> a, const Owner& b) const noexcept
{
    return less(a, *b);
}
ContactTracePool::ContactTracePool(Budget& budget) : budget_(&budget) {}
ContactTracePool::ContactTracePool(const ContactTracePool& source, Budget& budget) : budget_(&budget)
{
    try
    {
        for (const auto& [owner, buffers] : source.inventories_)
        {
            // Charge actual retained capacities, never the source arena's
            // historical used/cumulative counters or immutable geometry pins.
            const auto bytes = sum(sum(owner_bytes, node_bytes()), buffer_bytes(*owner));
            admit(bytes);
            inventories_.emplace(owner, buffers);
        }
    }
    catch (...)
    {
        clear();
        throw;
    }
}
ContactTracePool::~ContactTracePool()
{
    clear();
}
void ContactTracePool::admit(std::size_t bytes)
{
    const auto next = sum(charged_, bytes);
    budget_->add(bytes);
    charged_ = next;
}
void ContactTracePool::clear() noexcept
{
    inventories_.clear();
    budget_->used -= charged_;
    charged_ = 0;
}
std::size_t ContactTracePool::metadata_bytes() const noexcept
{
    return inventories_.size() * node_bytes();
}
std::size_t ContactTracePool::adopt(ContactRecord& record)
{
    if (record.shared_traces)
        throw std::invalid_argument("Contact trace inventory was already adopted.");
    if (record.traces.empty() ||
        std::any_of(record.traces.begin(), record.traces.end(), [](const auto& face) { return bool(face.face); }))
        return 0;
    const auto bytes = buffer_bytes(record.traces);
    if (bytes > budget_->used)
        throw std::invalid_argument("Contact trace buffers have no live admission.");
    const auto found = inventories_.find(std::span<const AdaptiveTraceFace>(record.traces));
    if (found != inventories_.end())
    {
        std::vector<AdaptiveTraceFace>().swap(record.traces);
        budget_->used -= bytes;
        record.shared_traces = found->first;
        return bytes;
    }
    const auto overhead = sum(owner_bytes, node_bytes());
    // Both transferred credit and new allocations must fit the pool counter
    // before allocating or mutating the builder's ownership.
    (void) sum(sum(charged_, overhead), bytes);
    admit(overhead);
    std::shared_ptr<Inventory> owner;
    try
    {
        owner = std::make_shared<Inventory>();
        owner->swap(record.traces);
        const auto [position, inserted] = inventories_.emplace(owner, bytes);
        if (!inserted)
            throw std::invalid_argument("Duplicate contact trace inventory insertion.");
        charged_ += bytes; // Transfer existing buffer credit, without admitting it twice.
        record.shared_traces = position->first;
    }
    catch (...)
    {
        if (owner)
            owner->swap(record.traces);
        budget_->used -= overhead;
        charged_ -= overhead;
        throw;
    }
    return bytes;
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
