/** @file PointProvenanceRef.hh
 * @brief Opaque owning point tokens and nonowning support-incidence ranges.
 */
#pragma once
#include "utils/ArithmeticMode.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace SimpleFluid::PLIC
{
class PlaneAuthority;
class PointProvenanceView;
struct PointSnapshotContextBase;
template<ArithmeticMode> struct PointSnapshotContext;
template<ArithmeticMode> class TypedPointProvenanceRef;
enum class PointProvenanceRole : std::uint8_t
{
    Native,
    Generated,
    Export,
    Numerical
};
struct DynamicStoredPointRef
{
    std::uint32_t dependency_slot = 0;
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();
    PointProvenanceRole role = PointProvenanceRole::Native;
    [[nodiscard]] explicit operator bool() const noexcept { return index != std::numeric_limits<std::uint32_t>::max(); }
    bool operator==(const DynamicStoredPointRef&) const = default;
};
/** Keys name registered authorities; this range contains only independently
 * admitted incidence rows. Registration alone creates no point membership. */
template<ArithmeticMode> class PointSnapshotBuilder;
template<ArithmeticMode Mode> class StoredPointRef
{
public:
    StoredPointRef() = default;
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return index_ != std::numeric_limits<std::uint32_t>::max();
    }
    [[nodiscard]] std::uint32_t dependency_slot() const noexcept { return dependency_slot_; }
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }
    [[nodiscard]] PointProvenanceRole role() const noexcept { return role_; }
    bool operator==(const StoredPointRef&) const = default;

private:
    friend class DynamicGeometryAccess;
    template<ArithmeticMode> friend class PointSnapshotBuilder;
    template<ArithmeticMode> friend class TypedPointProvenanceRef;
    StoredPointRef(std::uint32_t dependency, std::uint32_t index, PointProvenanceRole role)
        : dependency_slot_(dependency), index_(index), role_(role)
    {
    }
    std::uint32_t dependency_slot_ = 0;
    std::uint32_t index_ = std::numeric_limits<std::uint32_t>::max();
    PointProvenanceRole role_ = PointProvenanceRole::Native;
};
class PointAuthorityRange
{
public:
    using Lookup = const PlaneAuthority& (*)(const void*, std::size_t);
    struct Iterator
    {
        const void* source = nullptr;
        Lookup lookup = nullptr;
        std::size_t position = 0;
        using value_type = PlaneAuthority;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using reference = const PlaneAuthority&;
        using pointer = const PlaneAuthority*;
        const PlaneAuthority& operator*() const;
        Iterator& operator++()
        {
            ++position;
            return *this;
        }
        Iterator operator++(int)
        {
            auto copy = *this;
            ++*this;
            return copy;
        }
        bool operator==(const Iterator&) const = default;
    };
    PointAuthorityRange() = default;
    [[nodiscard]] Iterator begin() const { return {source_, lookup_, first_}; }
    [[nodiscard]] Iterator end() const { return {source_, lookup_, first_ + count_}; }
    [[nodiscard]] bool empty() const { return count_ == 0; }
    [[nodiscard]] std::size_t size() const { return count_; }
    const PlaneAuthority& operator[](std::size_t) const;
    const PlaneAuthority& front() const { return (*this)[0]; }
    bool operator==(const PointAuthorityRange&) const;
    bool operator==(const std::vector<PlaneAuthority>&) const;

private:
    friend class PointProvenanceView;
    friend class DynamicGeometryAccess;
    friend class PointAuthorityRangeAccess;
    PointAuthorityRange(const void* source, std::size_t first, std::size_t count, Lookup lookup)
        : source_(source), first_(first), count_(count), lookup_(lookup)
    {
    }
    const void* source_ = nullptr;
    std::size_t first_ = 0;
    Lookup lookup_ = nullptr;
    std::size_t count_ = 0;
};
/** Owning API-boundary identity. Copies retain the exact frozen snapshot and
 * role/slot; equality never merges independently issued equal-valued points. */
class PointProvenanceRef
{
public:
    PointProvenanceRef() = default;
    PointProvenanceRef(std::nullptr_t) {}
    PointProvenanceRef(const PointProvenanceRef&) = default;
    PointProvenanceRef& operator=(const PointProvenanceRef&) = default;
    PointProvenanceRef(PointProvenanceRef&& other) noexcept
        : owner_(std::move(other.owner_)), ref_(std::exchange(other.ref_, {}))
    {
    }
    PointProvenanceRef& operator=(PointProvenanceRef&& other) noexcept
    {
        owner_ = std::move(other.owner_);
        ref_ = std::exchange(other.ref_, {});
        return *this;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return bool(owner_) && bool(ref_); }
    [[nodiscard]] PointProvenanceView view() const;
    [[nodiscard]] const auto& context_pin() const noexcept { return owner_; }
    [[nodiscard]] const void* context_identity() const noexcept { return owner_.get(); }
    [[nodiscard]] DynamicStoredPointRef local_ref() const noexcept { return ref_; }
    void reset() noexcept
    {
        owner_.reset();
        ref_ = {};
    }
    bool operator==(const PointProvenanceRef& other) const noexcept
    {
        return (!*this && !other) || (owner_ == other.owner_ && ref_ == other.ref_);
    }
    bool operator==(std::nullptr_t) const noexcept { return !bool(*this); }

private:
    friend class DynamicGeometryAccess;
    friend class PointProvenanceView;
    template<ArithmeticMode> friend class PointSnapshotBuilder;
    friend PointProvenanceRef bind_point(std::shared_ptr<const PointSnapshotContextBase>, DynamicStoredPointRef);
    PointProvenanceRef(std::shared_ptr<const PointSnapshotContextBase>, DynamicStoredPointRef);
    std::shared_ptr<const PointSnapshotContextBase> owner_;
    DynamicStoredPointRef ref_;
};
/** Typed owning token retained by Point<Mode>. Runtime erasure occurs only
 * at an explicit DynamicPoint conversion. */
template<ArithmeticMode Mode> class TypedPointProvenanceRef
{
public:
    TypedPointProvenanceRef() = default;
    explicit TypedPointProvenanceRef(const PointProvenanceRef&);
    TypedPointProvenanceRef& operator=(const PointProvenanceRef& source)
    {
        *this = TypedPointProvenanceRef(source);
        return *this;
    }
    TypedPointProvenanceRef(const TypedPointProvenanceRef&) = default;
    TypedPointProvenanceRef& operator=(const TypedPointProvenanceRef&) = default;
    TypedPointProvenanceRef(TypedPointProvenanceRef&& source) noexcept
        : owner_(std::move(source.owner_)), ref_(std::exchange(source.ref_, {}))
    {
    }
    TypedPointProvenanceRef& operator=(TypedPointProvenanceRef&& source) noexcept
    {
        owner_ = std::move(source.owner_);
        ref_ = std::exchange(source.ref_, {});
        return *this;
    }
    [[nodiscard]] explicit operator bool() const noexcept { return bool(owner_) && bool(ref_); }
    [[nodiscard]] PointProvenanceView view() const;
    [[nodiscard]] operator PointProvenanceRef() const;
    bool operator==(const TypedPointProvenanceRef& other) const noexcept
    {
        return (!*this && !other) || (owner_ == other.owner_ && ref_ == other.ref_);
    }

private:
    std::shared_ptr<const PointSnapshotContext<Mode>> owner_;
    StoredPointRef<Mode> ref_;
};
extern template class TypedPointProvenanceRef<ArithmeticMode::Ordinary>;
extern template class TypedPointProvenanceRef<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::PLIC
