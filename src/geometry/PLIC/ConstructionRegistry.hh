/** @file ConstructionRegistry.hh
 * @brief Independently authenticated roots for finite geometry construction replay.
 */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include <compare>

namespace SimpleFluid::PLIC
{
struct ConstructionRootKey
{
    enum class Kind : std::uint64_t
    {
        Native = 1,
        Sweep = 2,
        InputPlane = 3,
        LocalSource = 4
    };
    Kind kind = Kind::Native;
    std::uint64_t entity = 0, revision = 0, slot = 0;
    std::vector<char> identity;
    auto operator<=>(const ConstructionRootKey&) const = default;
};

/** Registration is an explicit trust boundary. Roots must be obtained from
 * native source authority or independently validated trial controls, never
 * from numerical assertions in the packet being decoded. */
template<ArithmeticMode Mode> class ConstructionRegistry
{
    static_assert(Mode == ArithmeticMode::Ordinary || Mode == ArithmeticMode::BoundedCertification);

public:
    struct Impl;
    explicit ConstructionRegistry(GeometryPacketContext, GeometryLimits = {});
    void register_native(std::uint64_t gid, std::uint64_t revision, ConvexGeometryPtr<Mode>);
    void register_sweep(std::uint64_t face, std::uint64_t ordinal, ConvexGeometryPtr<Mode>);
    void register_plane(std::uint64_t identity, const GeometryHalfspace&);
    [[nodiscard]] bool contains(const ConstructionRootKey&) const;
    /** Subset-by-construction in the immutable clip/rebase lineage. This does
     * not replace packet context validation or application route ownership. */
    [[nodiscard]] bool derived_from(const ConvexCellGeometry<Mode>&, const ConstructionRootKey&) const;
    [[nodiscard]] const GeometryPacketContext& context() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    /** Validates the v4 envelope and capped dependency-root manifest without
     * constructing geometry. Returned roots are declarations, not authority. */
    [[nodiscard]] static std::vector<ConstructionRootKey> required_roots(
        std::span<const char>, const GeometryPacketContext&, GeometryLimits = {});

private:
    template<ArithmeticMode> friend class ConvexCellGeometry;
    void register_geometry(ConstructionRootKey, ConvexGeometryPtr<Mode>);
    std::shared_ptr<Impl> impl_;
};

extern template class ConstructionRegistry<ArithmeticMode::Ordinary>;
extern template class ConstructionRegistry<ArithmeticMode::BoundedCertification>;
} // namespace SimpleFluid::PLIC
