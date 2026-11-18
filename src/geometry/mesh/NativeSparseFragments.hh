/** @file NativeSparseFragments.hh
 * @brief Immutable fragment table pages and borrowed ordinal dispatch. */
#pragma once
#include "geometry/mesh/AMR/NativePublicationSession.hh"
#include "geometry/mesh/NativeFragmentAccess.hh"
namespace SimpleFluid::Meshes
{
/** The actual Region values live in source fragment records. Pages share those
 * records, and current ordinal dispatch copies only scalar pointers. */
class SIMPLEFLUID_LOCAL NativeSparseFragments final
{
public:
    struct Row
    {
        std::shared_ptr<const NativeFragmentRecord> fragment;
    };
    struct Less
    {
        bool operator()(const Row& a, const Row& b) const
        {
            return std::less<const void*>{}(a.fragment.get(), b.fragment.get());
        }
    };
    using Pages = AMR::adaptive_detail::persistent_ordered_detail::OrderedPages<Row, Less>;
    explicit NativeSparseFragments(const geometry_detail::NativeConstructionAdmission& a)
        : route(a.allocator<const NativeFragmentRecord*>())
    {
    }
    Pages::Ptr root;
    std::shared_ptr<const geometry_detail::NativeAllocationEnvelope> allocation;
    geometry_detail::NativeConstructorVector<const NativeFragmentRecord*> route;
    [[nodiscard]] const MultiRegionMesh::Region& region(std::size_t ordinal) const
    {
        return route.at(ordinal)->region();
    }
    [[nodiscard]] std::size_t size() const noexcept { return route.size(); }
};
} // namespace SimpleFluid::Meshes
