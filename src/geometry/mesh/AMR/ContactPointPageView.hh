#pragma once
#include "IdentityPool.hh"
#include "geometry/PLIC/GeometryIdentity.hh"
#include <cstddef>
#include <memory>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
using ContactPointIdentity = PLIC::PointIdentity<ArithmeticMode::Ordinary>;
using ContactPointPageView = IdentityPool<ContactPointIdentity>::PageView;
using CapturedPointReference = IdentityPool<ContactPointIdentity>::CapturedReference;

/** One directory for a changed-row batch. Capturing records does not seal a
 * partial page: all normalizations can be interned before freeze publishes it. */
class ContactPointBatch
{
public:
    explicit ContactPointBatch(IdentityPool<ContactPointIdentity>& pool) : pool_(&pool), view_(pool.page_view()) {}
    [[nodiscard]] CapturedPointReference capture(std::size_t logical_slot)
    {
        return view_->capture(*pool_, logical_slot);
    }
    [[nodiscard]] std::shared_ptr<const ContactPointPageView> view() const { return view_; }
    void freeze() { view_->freeze(); }
    [[nodiscard]] std::size_t page_count() const noexcept { return view_->page_count(); }

private:
    IdentityPool<ContactPointIdentity>* pool_;
    std::shared_ptr<ContactPointPageView> view_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
