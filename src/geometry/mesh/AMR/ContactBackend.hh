#pragma once
#include "geometry/mesh/AMR/DistributedAdaptiveClosureRecords.hh"
#include <cstdint>
#include <memory>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Internal contact backend behind DistributedAdaptiveContactHalo. The hierarchy
 * transaction API does not change; this seam decides whether contact is answered
 * by the explicit incidence machinery or derived from logical topology. */
enum class ContactBackendKind : std::uint8_t
{
    Explicit,
    Cartesian
};
class ContactBackend
{
public:
    virtual ~ContactBackend() = default;
    [[nodiscard]] virtual ContactBackendKind kind() const noexcept = 0;
    /** Non-owning closure views for an owned contact pair.
     *
     * The explicit backend always returns false: the retained ContactRecords own
     * the facts. A Cartesian backend returns true and fills both views from its
     * retained source/topology without constructing records. Either way the
     * returned views stay valid only for the duration of the pair evaluation
     * and must never be stored. */
    virtual bool closure_views(
        const AdaptiveCellKey&, const AdaptiveCellKey&, ContactClosureView&, ContactClosureView&) const noexcept
    {
        return false;
    }
};
/** Current default: the existing explicit support/line incidence machinery. */
class ExplicitContactBackend final : public ContactBackend
{
public:
    [[nodiscard]] ContactBackendKind kind() const noexcept override { return ContactBackendKind::Explicit; }
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
