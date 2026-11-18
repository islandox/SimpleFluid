/** @file AdaptiveNativePublicationAdapter.hh
 * @brief Borrowed hierarchy routes with an explicit FrozenNative owning mode.
 */
#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveNative.hh"
#include <variant>
namespace SimpleFluid::VoF::details
{
namespace native_publication_adapter_detail
{
namespace AMR = ::SimpleFluid::Meshes::AMR;
using Key = AMR::AdaptiveCellKey;
template<class Map, class View, class Reference> class Iterator
{
    std::variant<typename Map::const_iterator, typename View::const_iterator> value_;

public:
    explicit Iterator(typename Map::const_iterator i) : value_(i) {}
    explicit Iterator(typename View::const_iterator i) : value_(i) {}
    Reference operator*() const
    {
        return std::visit([](const auto& i) -> Reference { return {(*i).first, (*i).second}; }, value_);
    }
    Iterator& operator++()
    {
        std::visit([](auto& i) { ++i; }, value_);
        return *this;
    }
    bool operator==(const Iterator& other) const { return value_ == other.value_; }
};
class OwnedRoutes
{
    using Map = std::map<Key, std::uint64_t>;
    std::variant<Map, AMR::OwnedNativeGidsView> value_;

public:
    using const_iterator = Iterator<Map, AMR::OwnedNativeGidsView, std::pair<const Key&, std::uint64_t>>;
    OwnedRoutes() = default;
    explicit OwnedRoutes(Map value) : value_(std::move(value)) {}
    explicit OwnedRoutes(AMR::OwnedNativeGidsView value) : value_(std::move(value)) {}
    std::size_t size() const
    {
        return std::visit([](const auto& v) { return v.size(); }, value_);
    }
    std::uint64_t at(const Key& key) const
    {
        return std::visit([&](const auto& v) { return v.at(key); }, value_);
    }
    auto begin() const
    {
        return std::visit([](const auto& v) { return const_iterator(v.begin()); }, value_);
    }
    auto end() const
    {
        return std::visit([](const auto& v) { return const_iterator(v.end()); }, value_);
    }
};
class LocalRoutes
{
    using Map = std::map<std::uint64_t, Key>;
    std::variant<Map, AMR::LocalNativeKeysView> value_;

public:
    using const_iterator = Iterator<Map, AMR::LocalNativeKeysView, std::pair<std::uint64_t, const Key&>>;
    LocalRoutes() = default;
    explicit LocalRoutes(Map value) : value_(std::move(value)) {}
    explicit LocalRoutes(AMR::LocalNativeKeysView value) : value_(std::move(value)) {}
    std::size_t size() const
    {
        return std::visit([](const auto& v) { return v.size(); }, value_);
    }
    const Key& at(std::uint64_t gid) const
    {
        return std::visit([&](const auto& v) -> const Key& { return v.at(gid); }, value_);
    }
    auto begin() const
    {
        return std::visit([](const auto& v) { return const_iterator(v.begin()); }, value_);
    }
    auto end() const
    {
        return std::visit([](const auto& v) { return const_iterator(v.end()); }, value_);
    }
};
} // namespace native_publication_adapter_detail
/** Normal hierarchy publication retains borrowed persistent views. FrozenNative
 * continues to own its supported compatibility maps and accepted-copy proof. */
struct AdaptiveNativePublicationAdapter
{
    std::shared_ptr<const CompositePartition<>> partition;
    std::shared_ptr<const MeshHandle<>> mesh;
    native_publication_adapter_detail::OwnedRoutes owned_gids;
    native_publication_adapter_detail::LocalRoutes local_keys;
    Meshes::MeshStorageReport storage;
    Meshes::AMR::AdaptiveNativePublicationView::KeyRegistryStorage key_registry{0, true};
    AdaptiveNativePublicationAdapter() = default;
    AdaptiveNativePublicationAdapter(Meshes::AMR::AdaptiveNativePublicationView view)
        : partition(view.partition()), mesh(view.mesh()), owned_gids(view.owned_gids()), local_keys(view.local_keys()),
          storage(view.storage()), key_registry(view.key_registry_storage())
    {
    }
    AdaptiveNativePublicationAdapter(Meshes::AMR::AdaptiveNativePublication publication)
        : partition(std::move(publication.partition)), mesh(std::move(publication.mesh)),
          owned_gids(std::move(publication.owned_gids)), local_keys(std::move(publication.local_keys)),
          storage(publication.storage)
    {
    }
    AdaptiveNativePublicationAdapter(std::shared_ptr<const CompositePartition<>> partition_,
        std::shared_ptr<const MeshHandle<>> mesh_, std::map<Meshes::AMR::AdaptiveCellKey, std::uint64_t> owned_,
        std::map<std::uint64_t, Meshes::AMR::AdaptiveCellKey> local_, Meshes::MeshStorageReport storage_)
        : partition(std::move(partition_)), mesh(std::move(mesh_)), owned_gids(std::move(owned_)),
          local_keys(std::move(local_)), storage(storage_)
    {
    }
};
} // namespace SimpleFluid::VoF::details
