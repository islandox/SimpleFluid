/** @file AdaptiveContactSnapshot.hh
 * @brief Immutable closed-contact queries for one complete local AMR selection.
 */
#pragma once
#include "AdaptiveContactQuery.hh"
#include "ContactGroupAllocation.hh"
#include "ContactHaloContext.hh"
#include "ContactPayloadCensus.hh"
#include "utils/export_helper.hh"
#include <memory>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
class DistributedAdaptiveContactHalo;

/** Pins an immutable leaf selection with explicit point/fan pages or a
 * certified procedural Cartesian state. Queries are local and independent of
 * any hierarchy, provider or mutable candidate overlay. */
class SIMPLEFLUID_LOCAL AdaptiveContactSnapshot
{
public:
    [[nodiscard]] const ContactHaloContext& context() const noexcept { return context_; }
    void visit_closed_contacts(const AdaptiveCellKey&, const ContactWitnessVisitor&) const;
    void visit_borrowed_payload(const std::function<void(const ContactPayloadAllocation&)>&) const;

private:
    friend class DistributedAdaptiveContactHalo;
    struct Storage;
    AdaptiveContactSnapshot(ContactGroupAllocation charge, std::shared_ptr<const Storage> storage,
        ContactHaloContext context)
        : admission_(std::move(charge)), storage_(std::move(storage)), context_(context) {}
    ContactGroupAllocation admission_;
    std::shared_ptr<const Storage> storage_;
    ContactHaloContext context_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
