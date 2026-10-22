/** @file GeneralVoFExactStorage.cc
 * @brief Shared-payload and explicit-capacity accounting for flow certificates.
 */
#include "solvers/details/GeneralVoFExactStorage.hh"
#include <limits>
#include <stdexcept>
namespace SimpleFluid::detail
{
namespace
{
std::size_t sum(std::size_t a, std::size_t b, std::size_t limit)
{
    if (a > limit || b > limit - a)
        throw std::runtime_error("General VoF cumulative exact storage budget exceeded.");
    return a + b;
}
constexpr std::size_t record_metadata = 2 * (sizeof(void*) + sizeof(std::size_t) * 4 + 4 * sizeof(void*));
unsigned index(GeneralVoFStorageDomain domain)
{
    const auto i = static_cast<unsigned>(domain);
    if (i >= 3)
        throw std::runtime_error("Invalid exact storage domain.");
    return i;
}
} // namespace
std::size_t GeneralVoFExactStorageReport::total() const
{
    std::size_t value = audit_scratch;
    for (auto amount : bytes)
        value = sum(value, amount, std::numeric_limits<std::size_t>::max());
    return value;
}
void GeneralVoFExactStorage::capacity(GeneralVoFStorageDomain domain, std::size_t count, std::size_t width)
{
    const auto i = index(domain);
    if (width && count > budget_ / width)
        throw std::runtime_error("General VoF container capacity exceeds budget.");
    const auto amount = count * width;
    const auto next = sum(used_, amount, budget_);
    capacity_[i] = sum(capacity_[i], amount, budget_);
    used_ = next;
}
void GeneralVoFExactStorage::measure(GeneralVoFStorageDomain domain, const Meshes::ExactGeometryMeasure& value)
{
    const auto i = index(domain);
    const auto identity = value.storage_identity();
    auto found = records_.find(identity);
    if (found == records_.end())
    {
        const auto bytes = value.storage_bytes();
        const auto next = sum(sum(used_, bytes, budget_), record_metadata, budget_);
        Record record;
        record.bytes = bytes;
        record.references[i] = 1;
        records_.emplace(identity, record);
        used_ = next;
    }
    else
    {
        if (found->second.bytes != value.storage_bytes() ||
            found->second.references[i] == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("Exact shared storage changed or reference count overflow.");
        ++found->second.references[i];
    }
}
void GeneralVoFExactStorage::release(GeneralVoFStorageDomain domain, const Meshes::ExactGeometryMeasure& value)
{
    const auto i = index(domain);
    const auto found = records_.find(value.storage_identity());
    if (found == records_.end() || !found->second.references[i])
        throw std::runtime_error("Untracked exact storage release.");
    --found->second.references[i];
    const auto& refs = found->second.references;
    if (!refs[0] && !refs[1] && !refs[2])
    {
        used_ -= found->second.bytes + record_metadata;
        records_.erase(found);
    }
}
GeneralVoFExactStorageReport GeneralVoFExactStorage::finish() const
{
    GeneralVoFExactStorageReport result;
    result.bytes = capacity_;
    result.audit_scratch = records_.size() * record_metadata;
    for (const auto& [identity, record] : records_)
    {
        for (unsigned i = 0; i < 3; ++i)
            if (record.references[i])
            {
                result.bytes[i] += record.bytes;
                break;
            }
    }
    if (result.total() != used_ || used_ > budget_)
        throw std::runtime_error("Inconsistent exact storage accounting.");
    return result;
}
} // namespace SimpleFluid::detail
