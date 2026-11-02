/** @file RetainedInterfaces.cc
 * @brief Owned sparse reconstruction evidence and publication-copy preflight.
 */
#include "FVM/details/GeneralVoFInternal.hh"
namespace SimpleFluid::VoF::general_vof_detail
{
std::size_t RetainedInterfaces::storage_bytes() const
{
    wire::Budget bytes{std::numeric_limits<std::size_t>::max()};
    bytes.add(records_.capacity() * sizeof(Record));
    for (const auto& record : records_)
        bytes.add(ancestry_bytes(record.key));
    return bytes.used;
}

std::size_t RetainedInterfaces::copy_bytes() const
{
    wire::Budget bytes{std::numeric_limits<std::size_t>::max()};
    bytes.add(records_.size() * sizeof(Record));
    for (const auto& record : records_)
        bytes.add(topology_key_wire_bytes(record.key));
    return bytes.used;
}

void RetainedInterfaces::reserve(std::size_t count, wire::Budget& budget)
{
    if (count <= records_.capacity())
        return;
    if (count > records_.max_size() || count > (budget.limit - budget.used) / sizeof(Record))
        throw PLIC::GeometryError(
            PLIC::GeometryErrorCode::ResourceLimit, "General VoF retained interface capacity exceeds its byte budget.");
    // Do not refund the old buffer: both allocations coexist during growth.
    budget.add(count * sizeof(Record));
    records_.reserve(count);
    if (records_.capacity() > count)
        budget.add((records_.capacity() - count) * sizeof(Record));
}

void RetainedInterfaces::append(
    const Key& key, const PLIC::DynamicPolyhedralReconstructionResult& result, wire::Budget& budget)
{
    // Canonical encoding bounds the copied key's dynamic ancestry allocation.
    const auto key_bytes = topology_key_wire_bytes(key);
    budget.add(key_bytes);
    if (records_.size() == records_.capacity())
    {
        const auto old = records_.capacity();
        if (old > records_.max_size() / 2)
            throw PLIC::GeometryError(
                PLIC::GeometryErrorCode::ResourceLimit, "General VoF retained interface capacity overflows.");
        reserve(old ? 2 * old : 1, budget);
    }
    records_.push_back(Record{key, result});
    const auto actual = ancestry_bytes(records_.back().key);
    if (actual > key_bytes)
        budget.add(actual - key_bytes);
}

void RetainedInterfaces::finish()
{
    std::sort(records_.begin(), records_.end(),
        [](const Record& first, const Record& second) { return first.key < second.key; });
    if (std::adjacent_find(records_.begin(), records_.end(),
            [](const Record& first, const Record& second) { return first.key == second.key; }) != records_.end())
        throw std::invalid_argument("General VoF retained interface batch has duplicate canonical keys.");
}

std::size_t RetainedInterfaces::ancestry_bytes(const Key& key)
{
    return std::visit(
        [](const auto& ancestry) -> std::size_t
        {
            using T = std::decay_t<decltype(ancestry)>;
            if constexpr (std::is_same_v<T, AMR::ConvexCellAncestry>)
                return ancestry.path.capacity() * sizeof(std::uint64_t);
            else if constexpr (std::is_same_v<T, AMR::ProductCellAncestry>)
                return (ancestry.base.path.capacity() + ancestry.axial.path.capacity()) * sizeof(std::uint64_t);
            else
                return 0;
        },
        key.ancestry);
}

std::shared_ptr<GeneralVoF::Data> copy_state(const GeneralVoF::Data& source)
{
    wire::Budget budget{source.options.byte_budget};
    // The source and prospective copied sparse buffers coexist. Admit before
    // copying; the normal retained-state audit still covers the complete result.
    budget.add(source.retained_interfaces.storage_bytes());
    const auto expected = source.retained_interfaces.copy_bytes();
    budget.add(expected);
    auto result = std::make_shared<GeneralVoF::Data>(source);
    const auto actual = result->retained_interfaces.storage_bytes();
    if (actual > expected)
        budget.add(actual - expected);
    return result;
}

void invalidate_owned_interfaces(GeneralVoF::Data& data)
{
    const auto rank = data.hierarchy->communicator()->getRank();
    data.retained_interfaces.erase_if([&](const Key& key) { return data.cells.at(key).owner == rank; });
}
} // namespace SimpleFluid::VoF::general_vof_detail
