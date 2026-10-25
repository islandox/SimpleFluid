/** @file ExactLiveStorage.cc
 * @brief Compiled live exact-payload leases and bounded scalar operations.
 */
#include "FVM/details/ExactLiveStorage.hh"
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
namespace SimpleFluid::VoF::detail
{
namespace
{
constexpr std::size_t slot_bytes =
    sizeof(std::pair<const ExactLiveStorage::M*, ExactLiveStorage::Lease>) + 4 * sizeof(void*);
}
ExactLiveStorage::Lease::Lease(ExactLiveStorage* owner, const void* identity) noexcept
    : owner_(owner), identity_(identity)
{
}
ExactLiveStorage::Lease::Lease(Lease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), identity_(other.identity_)
{
}
ExactLiveStorage::Lease& ExactLiveStorage::Lease::operator=(Lease&& other) noexcept
{
    if (this != &other)
    {
        if (owner_)
            owner_->release(identity_);
        owner_ = std::exchange(other.owner_, nullptr);
        identity_ = other.identity_;
    }
    return *this;
}
ExactLiveStorage::Lease::~Lease()
{
    if (owner_)
        owner_->release(identity_);
}
ExactLiveStorage::Scope::Scope(ExactLiveStorage& owner, std::initializer_list<M*> values) : owner_(owner)
{
    if (values.size() > values_.size())
        throw std::invalid_argument("Exact live scope has too many slots.");
    try
    {
        for (auto* value : values)
        {
            if (owner_.slots_.contains(value))
                throw std::invalid_argument("Exact live scope reuses a tracked slot.");
            owner_.track(*value);
            values_[count_++] = value;
        }
    }
    catch (...)
    {
        while (count_)
            owner_.untrack(*values_[--count_]);
        throw;
    }
}
ExactLiveStorage::Scope::~Scope()
{
    while (count_)
        owner_.untrack(*values_[--count_]);
}
ExactLiveStorage::ExactLiveStorage(Budget& budget, std::size_t operation_bytes)
    : budget_(budget), operation_bytes_(operation_bytes)
{
    if (!operation_bytes)
        throw std::invalid_argument("Exact live arithmetic lease is empty.");
    charge(operation_bytes);
}
ExactLiveStorage::~ExactLiveStorage()
{
    std::vector<Lease>().swap(retained_);
    const auto slot_count = slots_.size();
    slots_.clear();
    budget_.used -= slot_count * slot_bytes + retained_metadata_ + operation_bytes_;
}
void ExactLiveStorage::charge(std::size_t bytes)
{
    if (bytes > budget_.limit - budget_.used)
        throw std::invalid_argument(
            "General VoF donor live exact budget exhausted: used=" + std::to_string(budget_.used) +
            ", requested=" + std::to_string(bytes) + ", limit=" + std::to_string(budget_.limit));
    budget_.add(bytes);
}
ExactLiveStorage::Lease ExactLiveStorage::pin(const M& value)
{
    const auto identity = value.storage_identity();
    auto it = records_.find(identity);
    if (it == records_.end())
    {
        const auto payload = value.storage_bytes();
        const auto metadata = sizeof(std::pair<const void* const, Record>) + 4 * sizeof(void*);
        if (payload > std::numeric_limits<std::size_t>::max() - metadata)
            throw std::invalid_argument("Exact live record capacity overflows.");
        const auto bytes = payload + metadata;
        charge(bytes);
        try
        {
            it = records_.emplace(identity, Record{value, 0, bytes}).first;
        }
        catch (...)
        {
            budget_.used -= bytes;
            throw;
        }
    }
    if (it->second.refs == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Exact live reference count overflows.");
    ++it->second.refs;
    return Lease(this, identity);
}
void ExactLiveStorage::release(const void* identity) noexcept
{
    auto it = records_.find(identity);
    if (--it->second.refs == 0)
    {
        const auto bytes = it->second.bytes;
        records_.erase(it); // destroy the retained exact handle before releasing its charge
        budget_.used -= bytes;
    }
}
void ExactLiveStorage::track(M& value)
{
    if (slots_.contains(&value))
        return;
    charge(slot_bytes);
    try
    {
        auto lease = pin(value);
        slots_.emplace(&value, std::move(lease));
    }
    catch (...)
    {
        budget_.used -= slot_bytes;
        throw;
    }
}
void ExactLiveStorage::untrack(M& value) noexcept
{
    if (slots_.erase(&value))
        budget_.used -= slot_bytes;
}
void ExactLiveStorage::retain(const M& value)
{
    // Three slots per append cover retained capacity plus old/new arrays at growth.
    charge(3 * sizeof(Lease));
    retained_metadata_ += 3 * sizeof(Lease);
    auto lease = pin(value);
    retained_.push_back(std::move(lease));
}
void ExactLiveStorage::retain(const SimpleFluid::PLIC::PhaseMeasures<SimpleFluid::ArithmeticMode::Exact>& value)
{
    retain(value.bulk);
    retain(value.gas);
    retain(value.carrier);
}
void ExactLiveStorage::replace(M& slot, const M& value)
{
    track(slot);
    auto pending = pin(value); // old source and pending replacement coexist
    slot = value;
    slots_.at(&slot) = std::move(pending); // releases old payload only after replacement
}
std::size_t ExactLiveStorage::preflight(const M& a, const M* b) const
{
    const auto x = a.storage_bytes(), y = b ? b->storage_bytes() : 0;
    if (x > operation_bytes_ / 8 || y > operation_bytes_ / 8 - x)
        throw std::invalid_argument("General VoF donor scalar operands exceed reserved operation lease.");
    return 8 * (x + y);
}
ExactLiveStorage::M ExactLiveStorage::checked(M value, std::size_t bound) const
{
    if (value.storage_bytes() > bound)
        throw std::invalid_argument("General VoF donor pending scalar exceeds admitted result lease.");
    return value;
}
ExactLiveStorage::M ExactLiveStorage::sum(const M& a, const M& b) const
{
    const auto bound = preflight(a, &b);
    return checked(a + b, bound);
}
ExactLiveStorage::M ExactLiveStorage::difference(const M& a, const M& b) const
{
    const auto bound = preflight(a, &b);
    return checked(a - b, bound);
}
ExactLiveStorage::M ExactLiveStorage::scaled(const M& a, double factor) const
{
    const auto bound = preflight(a);
    return checked(a.scaled(factor), bound);
}
void ExactLiveStorage::add(M& slot, const M& value)
{
    track(slot);
    auto pending = sum(slot, value);
    replace(slot, pending);
}
void ExactLiveStorage::subtract(M& slot, const M& value)
{
    track(slot);
    auto pending = difference(slot, value);
    replace(slot, pending);
}

} // namespace SimpleFluid::VoF::detail
