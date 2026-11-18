/** @file NativePublicationView.hh
 * @brief Borrowed typed-key lookup over current native publication routes.
 */
#pragma once
#include "geometry/MeshHandle.hh"
#include "geometry/mesh/AMR/DistributedAdaptiveTypes.hh"
#include "parallel/CompositePartition.hh"
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <variant>

namespace SimpleFluid::geometry_detail
{
class NativeSchemaOwner;
class NativeAllocationEnvelope;
} // namespace SimpleFluid::geometry_detail
namespace SimpleFluid::Meshes::AMR
{
class DistributedAdaptiveHierarchy;
struct AdaptiveNativePublication;
namespace adaptive_detail
{
struct SIMPLEFLUID_LOCAL NativePublicationViewData;
class NativePublicationSession;
class NativeSparseAssembler;
class NativeKernelAuthority;
} // namespace adaptive_detail

/** Immutable detached schema input. The qualified getter borrows its actual
 * built-in factory owner; custom kernels use an explicit full metadata export.
 * No kernel, hierarchy, Data or partition controller is retained.
 */
class SIMPLEFLUID_PUBLIC_TYPE AdaptiveNativeSchemaView
{
public:
    [[nodiscard]] std::span<const char> span() const noexcept;

private:
    friend class adaptive_detail::NativePublicationSession;
    friend class adaptive_detail::NativeSparseAssembler;
    friend AdaptiveNativeSchemaView native_publication_schema(const DistributedAdaptiveHierarchy&);
    AdaptiveNativeSchemaView(std::shared_ptr<const geometry_detail::NativeSchemaOwner> owner,
        std::weak_ptr<const adaptive_detail::NativeKernelAuthority> authority,
        std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation)
        : owner_(std::move(owner)), authority_(std::move(authority)), authority_allocation_(std::move(allocation))
    {
    }
    std::shared_ptr<const geometry_detail::NativeSchemaOwner> owner_;
    std::weak_ptr<const adaptive_detail::NativeKernelAuthority> authority_;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> authority_allocation_;
};
/** Collective, including custom-kernel export failure handling. */
[[nodiscard]] AdaptiveNativeSchemaView native_publication_schema(const DistributedAdaptiveHierarchy&);

class SIMPLEFLUID_PUBLIC_TYPE OwnedNativeGidsView
{
public:
    class const_iterator
    {
    public:
        using value_type = std::pair<const AdaptiveCellKey&, std::uint64_t>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        const_iterator() = default;
        [[nodiscard]] reference operator*() const;
        struct Arrow
        {
            reference value;
            [[nodiscard]] const reference* operator->() const noexcept { return &value; }
        };
        [[nodiscard]] Arrow operator->() const { return {**this}; }
        const_iterator& operator++() noexcept
        {
            if (auto* ordinal = std::get_if<std::size_t>(&position_))
                ++*ordinal;
            else
                ++std::get<CompatibilityIterator>(position_);
            return *this;
        }
        const_iterator operator++(int) noexcept
        {
            auto previous = *this;
            ++*this;
            return previous;
        }
        [[nodiscard]] bool operator==(const const_iterator& other) const noexcept
        {
            return data_.get() == other.data_.get() && position_ == other.position_;
        }

    private:
        friend class OwnedNativeGidsView;
        using CompatibilityIterator = std::map<AdaptiveCellKey, std::uint64_t>::const_iterator;
        const_iterator(std::shared_ptr<const adaptive_detail::NativePublicationViewData>, std::size_t);
        const_iterator(std::shared_ptr<const adaptive_detail::NativePublicationViewData> data, CompatibilityIterator at)
            : data_(std::move(data)), position_(at)
        {
        }
        std::shared_ptr<const adaptive_detail::NativePublicationViewData> data_;
        std::variant<std::size_t, CompatibilityIterator> position_;
    };
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return !size(); }
    [[nodiscard]] std::uint64_t at(const AdaptiveCellKey&) const;
    [[nodiscard]] bool contains(const AdaptiveCellKey&) const;
    [[nodiscard]] const_iterator find(const AdaptiveCellKey&) const;
    [[nodiscard]] const_iterator begin() const;
    [[nodiscard]] const_iterator end() const;

private:
    friend class AdaptiveNativePublicationView;
    explicit OwnedNativeGidsView(std::shared_ptr<const adaptive_detail::NativePublicationViewData> data)
        : data_(std::move(data))
    {
    }
    std::shared_ptr<const adaptive_detail::NativePublicationViewData> data_;
};

class SIMPLEFLUID_PUBLIC_TYPE LocalNativeKeysView
{
public:
    class const_iterator
    {
    public:
        using value_type = std::pair<std::uint64_t, const AdaptiveCellKey&>;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        const_iterator() = default;
        [[nodiscard]] reference operator*() const;
        struct Arrow
        {
            reference value;
            [[nodiscard]] const reference* operator->() const noexcept { return &value; }
        };
        [[nodiscard]] Arrow operator->() const { return {**this}; }
        const_iterator& operator++() noexcept
        {
            ++ordinal_;
            return *this;
        }
        const_iterator operator++(int) noexcept
        {
            auto previous = *this;
            ++*this;
            return previous;
        }
        [[nodiscard]] bool operator==(const const_iterator& other) const noexcept
        {
            return data_.get() == other.data_.get() && ordinal_ == other.ordinal_;
        }

    private:
        friend class LocalNativeKeysView;
        const_iterator(std::shared_ptr<const adaptive_detail::NativePublicationViewData>, std::size_t);
        std::shared_ptr<const adaptive_detail::NativePublicationViewData> data_;
        std::size_t ordinal_ = 0;
    };
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return !size(); }
    [[nodiscard]] const AdaptiveCellKey& at(std::uint64_t) const;
    [[nodiscard]] bool contains(std::uint64_t) const;
    [[nodiscard]] const_iterator find(std::uint64_t) const;
    [[nodiscard]] const_iterator begin() const;
    [[nodiscard]] const_iterator end() const;

private:
    friend class AdaptiveNativePublicationView;
    explicit LocalNativeKeysView(std::shared_ptr<const adaptive_detail::NativePublicationViewData> data)
        : data_(std::move(data))
    {
    }
    std::shared_ptr<const adaptive_detail::NativePublicationViewData> data_;
};

/** Owns current membership and scalar routes; key access/iteration borrows from
 * immutable source/remote pages. It never implicitly creates owning key maps.
 */
class SIMPLEFLUID_PUBLIC_TYPE AdaptiveNativePublicationView
{
public:
    AdaptiveNativePublicationView() = default;
    [[nodiscard]] const std::shared_ptr<const CompositePartition<>>& partition() const;
    [[nodiscard]] const std::shared_ptr<const MeshHandle<>>& mesh() const;
    [[nodiscard]] OwnedNativeGidsView owned_gids() const { return OwnedNativeGidsView(data_); }
    [[nodiscard]] LocalNativeKeysView local_keys() const { return LocalNativeKeysView(data_); }
    struct KeyRegistryStorage
    {
        std::size_t scalar_route_bytes = 0;
        bool owns_key_maps = false;
    };
    /** Storage retained for typed keys. Compatibility maps need a separate rich-key estimate. */
    [[nodiscard]] KeyRegistryStorage key_registry_storage() const noexcept;
    /** Rich retained owners are currently unified in indexing. Numeric
     * output_only and explicit required_maps_estimate remain separate; zero
     * geometry/objects columns are not independent measured-zero claims. */
    [[nodiscard]] MeshStorageReport storage() const;

private:
    friend class adaptive_detail::NativePublicationSession;
    friend class adaptive_detail::NativeSparseAssembler;
    friend AdaptiveNativePublication export_adaptive_native_publication(
        const AdaptiveNativePublicationView&, std::size_t);
    explicit AdaptiveNativePublicationView(std::shared_ptr<const adaptive_detail::NativePublicationViewData> data)
        : data_(std::move(data))
    {
    }
    std::shared_ptr<const adaptive_detail::NativePublicationViewData> data_;
};

[[nodiscard]] AdaptiveNativePublicationView publish_adaptive_native(const DistributedAdaptiveHierarchy&,
    std::span<const char> root_schema, std::span<const AdaptiveCellKey> additional_full_ghosts = {});
[[nodiscard]] AdaptiveNativePublicationView publish_adaptive_native(const DistributedAdaptiveHierarchy&,
    const AdaptiveNativeSchemaView&, std::span<const AdaptiveCellKey> additional_full_ghosts = {});
/** Explicit O(N) rich metadata export. Complete map/key overlap is admitted
 * before allocation; qualified publication and normal native consumers use
 * the borrowed view API instead.
 */
[[nodiscard]] AdaptiveNativePublication export_adaptive_native_publication(
    const AdaptiveNativePublicationView&, std::size_t byte_budget);
} // namespace SimpleFluid::Meshes::AMR
