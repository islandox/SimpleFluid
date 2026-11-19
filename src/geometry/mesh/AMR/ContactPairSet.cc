#include "ContactPairSet.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveWire.hh"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
namespace
{
using Pair = ContactPairSet::Pair;
std::size_t product(std::size_t count, std::size_t width)
{
    if (count > std::numeric_limits<std::size_t>::max() / width)
        throw std::invalid_argument("Contact pair storage size overflows.");
    return count * width;
}
std::uint64_t mix(std::uint64_t value) noexcept
{
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}
std::size_t hash(Pair pair) noexcept
{
    return static_cast<std::size_t>(mix(pair.first) ^ mix(mix(pair.second) + UINT64_C(0x9e3779b97f4a7c15)));
}
bool fits_compact(Pair pair) noexcept
{
    return pair.second <= std::numeric_limits<std::uint32_t>::max();
}
std::uint64_t pack(Pair pair) noexcept
{
    return (static_cast<std::uint64_t>(pair.first) << 32) | static_cast<std::uint64_t>(pair.second);
}
Pair unpack(std::uint64_t pair) noexcept
{
    return {static_cast<std::size_t>(pair >> 32), static_cast<std::size_t>(pair & UINT64_C(0xffffffff))};
}
std::size_t slot(const std::uint64_t* table, std::size_t capacity, Pair pair) noexcept
{
    auto index = hash(pair) & (capacity - 1);
    const auto packed = pack(pair);
    while (table[index] && table[index] != packed)
        index = (index + 1) & (capacity - 1);
    return index;
}
std::size_t slot(const Pair* table, std::size_t capacity, Pair pair) noexcept
{
    auto index = hash(pair) & (capacity - 1);
    while (table[index].second && table[index] != pair)
        index = (index + 1) & (capacity - 1);
    return index;
}
/** Own only the temporary live admission. Never refund cumulative admissions. */
class Admission
{
public:
    Admission(Budget& budget, std::size_t bytes) : budget_(budget), bytes_(bytes) { budget_.add(bytes_); }
    ~Admission() { budget_.used -= bytes_; }
    void transfer() noexcept { bytes_ = 0; }

private:
    Budget& budget_;
    std::size_t bytes_;
};
} // namespace

ContactPairSet::ContactPairSet(Budget& budget) noexcept : budget_(&budget) {}
ContactPairSet::~ContactPairSet()
{
    clear();
}
void ContactPairSet::clear() noexcept
{
    compact_.reset();
    wide_.reset();
    budget_->used -= charged_;
    size_ = capacity_ = charged_ = 0;
}
void ContactPairSet::rebuild(std::size_t capacity, bool wide)
{
    const auto bytes = product(capacity, wide ? sizeof(Pair) : sizeof(std::uint64_t));
    if (bytes > std::numeric_limits<std::size_t>::max() - charged_)
        throw std::invalid_argument("Contact pair reallocation overlap overflows.");
    Admission admission(*budget_, bytes);
    // Declare arrays after admission so failed construction destroys temporary
    // buffers before the admission guard returns their live credit.
    ContactArray<std::uint64_t> compact;
    ContactArray<Pair> expanded;
    if (wide)
        expanded = contact_array<Pair>(capacity);
    else
        compact = contact_array<std::uint64_t>(capacity);
    for (std::size_t i = 0; i < capacity_; ++i)
    {
        if (wide_ ? !wide_[i].second : !compact_[i])
            continue;
        const auto pair = wide_ ? wide_[i] : unpack(compact_[i]);
        if (wide)
            expanded[slot(expanded.get(), capacity, pair)] = pair;
        else
            compact[slot(compact.get(), capacity, pair)] = pack(pair);
    }
    compact_.reset();
    wide_.reset();
    budget_->used -= charged_;
    compact_ = std::move(compact);
    wide_ = std::move(expanded);
    capacity_ = capacity;
    charged_ = bytes;
    admission.transfer();
}
bool ContactPairSet::insert(Pair pair)
{
    if (pair.first >= pair.second)
        throw std::invalid_argument("Contact pairs require ascending distinct identities.");
    if (capacity_)
    {
        if (wide_)
        {
            if (wide_[slot(wide_.get(), capacity_, pair)].second)
                return false;
        }
        else if (fits_compact(pair) && compact_[slot(compact_.get(), capacity_, pair)])
            return false;
    }
    auto capacity = capacity_;
    // Power-of-two capacity >= 8 makes this exact 7/8 admission threshold
    // computable without an overflowing multiplication or size+1 expression.
    if (!capacity || size_ >= capacity - capacity / 8)
    {
        if (capacity > std::numeric_limits<std::size_t>::max() / 2)
            throw std::invalid_argument("Contact pair table capacity overflows.");
        capacity = capacity ? 2 * capacity : 8;
    }
    const bool wide = bool(wide_) || !fits_compact(pair);
    if (capacity != capacity_ || (wide && !wide_))
        rebuild(capacity, wide);
    if (wide_)
        wide_[slot(wide_.get(), capacity_, pair)] = pair;
    else
        compact_[slot(compact_.get(), capacity_, pair)] = pack(pair);
    ++size_;
    return true;
}
ContactPairSet::SortedPairs ContactPairSet::take_sorted()
{
    SortedPairs result;
    if (!size_)
        return result;
    const auto bytes = product(size_, wide_ ? sizeof(Pair) : sizeof(std::uint64_t));
    if (bytes > std::numeric_limits<std::size_t>::max() - charged_)
        throw std::invalid_argument("Contact pair sorting overlap overflows.");
    // Declare temporary arrays after admission so allocation failure keeps the
    // table unchanged and releases only this temporary live admission.
    Admission admission(*budget_, bytes);
    ContactArray<std::uint64_t> compact;
    ContactArray<Pair> expanded;
    if (wide_)
        expanded = contact_array<Pair>(size_);
    else
        compact = contact_array<std::uint64_t>(size_);
    std::size_t next = 0;
    for (std::size_t i = 0; i < capacity_; ++i)
        if (wide_)
        {
            if (wide_[i].second)
                expanded[next++] = wide_[i];
        }
        else if (compact_[i])
            compact[next++] = compact_[i];
    if (wide_)
        std::sort(expanded.get(), expanded.get() + size_);
    else
        std::sort(compact.get(), compact.get() + size_);
    result.budget_ = budget_;
    result.size_ = size_;
    result.charged_ = bytes;
    result.compact_ = std::move(compact);
    result.wide_ = std::move(expanded);
    admission.transfer();
    clear();
    return result;
}
ContactPairSet::SortedPairs::~SortedPairs()
{
    clear();
}
ContactPairSet::SortedPairs::SortedPairs(SortedPairs&& other) noexcept
{
    *this = std::move(other);
}
ContactPairSet::SortedPairs& ContactPairSet::SortedPairs::operator=(SortedPairs&& other) noexcept
{
    if (this != &other)
    {
        clear();
        budget_ = std::exchange(other.budget_, nullptr);
        size_ = std::exchange(other.size_, 0);
        charged_ = std::exchange(other.charged_, 0);
        compact_ = std::move(other.compact_);
        wide_ = std::move(other.wide_);
    }
    return *this;
}
void ContactPairSet::SortedPairs::clear() noexcept
{
    compact_.reset();
    wide_.reset();
    if (budget_)
        budget_->used -= charged_;
    budget_ = nullptr;
    size_ = charged_ = 0;
}
ContactPairSet::Pair ContactPairSet::SortedPairs::const_iterator::operator*() const noexcept
{
    return owner_->wide_ ? owner_->wide_[index_] : unpack(owner_->compact_[index_]);
}
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
