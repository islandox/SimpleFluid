/** @file GeometryOwnerImportAccess.hh
 * @brief Internal installation of an immutable authenticated owner set into a
 * production allocation ledger without re-enumerating retained geometry.
 */
#pragma once
#include "geometry/PLIC/GeometryAllocationLedger.hh"
#include "geometry/PLIC/details/GeometryOwnerManifest.hh"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace SimpleFluid::PLIC::details
{
/** A manifest row names a ledger allocation kind in its `kind` byte and the
 * arithmetic channel 0 (Ordinary) or 1 (BoundedCertification). One import
 * must have a uniform channel, which selects/checks the ledger numerical mode. */
class GeometryOwnerImportAccess
{
public:
    /** Install the owner set into a ledger. A fresh, unfailed ledger shares the
     * immutable root and charges its cached union: no retained geometry is
     * enumerated. A populated ledger requires the exact union and returns the
     * number of imported rows it had to enumerate. Any failure poisons the
     * ledger, including failures after a partial union was admitted. */
    static std::size_t import(GeometryAllocationLedger&, const geometry_demand::GeometryOwnerManifest&,
        std::size_t& cumulative_bytes, std::size_t byte_limit);
    /** Rows installed as the immutable import root, distinct from the mutable
     * map rows this ledger admitted itself. */
    [[nodiscard]] static std::size_t imported_owners(const GeometryAllocationLedger&) noexcept;
    /** Export one ledger's admitted owner set as one immutable manifest. Rows
     * keep the ledger allocation kind and use arithmetic channel 0 or 1. The
     * temporary contiguous row buffer and retained index allocations are
     * charged before allocation. Includes any immutable imported root. */
    [[nodiscard]] static geometry_demand::GeometryOwnerManifest export_manifest(const GeometryAllocationLedger&,
        geometry_demand::Admission, geometry_demand::PageWork* = nullptr, std::uint8_t channel = 0);
    /** Borrowed diagnostic traversal of the imported rows only. */
    static void visit_imported(const GeometryAllocationLedger&,
        const std::function<void(std::uint8_t kind, const void* owner, std::size_t bytes)>&);
};
} // namespace SimpleFluid::PLIC::details
