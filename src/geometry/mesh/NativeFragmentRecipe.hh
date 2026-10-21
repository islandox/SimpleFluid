/** @file NativeFragmentRecipe.hh
 * @brief Detached procedural provider recipes and state-local part interning.
 */
#pragma once
#include "geometry/mesh/MultiRegionMesh.hh"
#include <memory>
#include <span>
#include <vector>
namespace SimpleFluid::Meshes
{
using NativeFragmentBytes = std::vector<char>;
/** Immutable through shared_ptr<const>. Slots are canonical whole-byte parts:
 * Cartesian: topology + three axes; ExactProduct: topology base, exact base,
 * axial values, axial selection. The view holds name/layers/aliases/run address.
 * Legacy procedural providers use one detached descriptor part. */
struct NativeFragmentDescriptor
{
    enum class Kind : std::uint64_t
    {
        Cartesian,
        ExactProduct,
        LegacyProcedural
    };
    Kind kind = Kind::Cartesian;
    std::vector<std::shared_ptr<const NativeFragmentBytes>> parts;
    NativeFragmentBytes view;
};
/** Transaction/hierarchy-local cache, never a process-global strong registry. */
class NativeFragmentCache
{
public:
    NativeFragmentCache();
    ~NativeFragmentCache();
    NativeFragmentCache(NativeFragmentCache&&) noexcept;
    NativeFragmentCache& operator=(NativeFragmentCache&&) noexcept;
    NativeFragmentCache(const NativeFragmentCache&) = delete;
    struct Impl;

private:
    friend MultiRegionMesh::Region decode_native_fragment(std::span<const char>, NativeFragmentCache&);
    friend std::shared_ptr<const NativeFragmentDescriptor> extract_native_fragment(
        const MultiRegionMesh::Region&, NativeFragmentCache&);
    friend std::shared_ptr<const NativeFragmentDescriptor> decode_native_fragment_descriptor(
        std::span<const char>, NativeFragmentCache&);
    friend MultiRegionMesh::Region materialize_native_fragment(const NativeFragmentDescriptor&, NativeFragmentCache&);
    std::unique_ptr<Impl> impl_;
};
[[nodiscard]] std::shared_ptr<const NativeFragmentDescriptor> extract_native_fragment(
    const MultiRegionMesh::Region&, NativeFragmentCache&);
[[nodiscard]] NativeFragmentBytes encode_native_fragment_descriptor(const NativeFragmentDescriptor&);
[[nodiscard]] std::shared_ptr<const NativeFragmentDescriptor> decode_native_fragment_descriptor(
    std::span<const char>, NativeFragmentCache&);
[[nodiscard]] MultiRegionMesh::Region materialize_native_fragment(
    const NativeFragmentDescriptor&, NativeFragmentCache&);
/** Encodes one native procedural fragment, not a global region catalog. The
 * leaf's native ordinal is separate from these internable descriptor bytes.
 * Cartesian coordinates and exact product constructions remain authoritative. */
[[nodiscard]] std::vector<char> encode_native_fragment(const MultiRegionMesh::Region&);
[[nodiscard]] MultiRegionMesh::Region decode_native_fragment(std::span<const char>, NativeFragmentCache&);
[[nodiscard]] MultiRegionMesh::Region decode_native_fragment(std::span<const char>);
} // namespace SimpleFluid::Meshes
