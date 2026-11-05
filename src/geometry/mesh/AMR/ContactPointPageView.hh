#pragma once
#include "ContactSourcePointView.hh"
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
    explicit ContactPointBatch(IncidencePointDirectory<>& pool)
    {
        if (pool.source())
        {
            directory_ = &pool;
            source_view_ =
                ContactSourcePointView::create(pool.source(), pool.allocation(), pool.keeper(), pool.source_arena());
        }
        else
        {
            pool_ = &pool.physical_pool();
            view_ = pool_->page_view();
        }
    }
    [[nodiscard]] CapturedPointReference capture(std::size_t logical_slot)
    {
        if (!pool_)
            throw std::invalid_argument("Physical capture used for a source namespace.");
        return view_->capture(*pool_, logical_slot);
    }
    [[nodiscard]] CapturedSourcePointReference capture_source(std::size_t logical_slot)
    {
        if (!directory_)
            throw std::invalid_argument("Source capture used for a physical namespace.");
        return source_view_->capture(*directory_, logical_slot);
    }
    [[nodiscard]] bool source_backed() const noexcept { return bool(source_view_); }
    [[nodiscard]] std::shared_ptr<const ContactSourcePointView> source_view() const { return source_view_; }
    [[nodiscard]] std::shared_ptr<const ContactPointPageView> view() const { return view_; }
    void freeze()
    {
        if (source_view_)
            source_view_->freeze();
        else
            view_->freeze();
    }
    [[nodiscard]] std::size_t page_count() const noexcept
    {
        return source_view_ ? source_view_->page_count() : view_->page_count();
    }

private:
    IdentityPool<ContactPointIdentity>* pool_ = nullptr;
    std::shared_ptr<ContactPointPageView> view_;
    IncidencePointDirectory<>* directory_ = nullptr;
    std::shared_ptr<ContactSourcePointView> source_view_;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
