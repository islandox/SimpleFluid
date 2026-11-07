#pragma once
#include "ContactSourcePoints.hh"
#include "IdentityPool.hh"
#include <memory>
#include <optional>
#include <type_traits>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
/** Stable logical point slots. The immutable certified prefix is a separate
 * value namespace; the explicit suffix retains original16-record physical
 * pages. Generic IdentityPool policies and old captured refs are unchanged. */
template<class Hasher = IdentityPoolHasher<ContactSourcePoint>, class Payload = IdentityPoolPayload<ContactSourcePoint>>
class IncidencePointDirectory
{
public:
    using Point = ContactSourcePoint;
    using Store = IdentityPool<Point, Hasher, Payload>;
    using HashPolicy = Hasher;
    using PayloadPolicy = Payload;
    static constexpr bool uses_hashing = Hasher::enabled;
    IncidencePointDirectory(Budget&, std::shared_ptr<Budget> = {}, Hasher = {}, Payload = {},
        IdentityPoolStorage = IdentityPoolStorage::Flat);
    IncidencePointDirectory(const IncidencePointDirectory&, Budget&, std::shared_ptr<Budget> = {},
        std::shared_ptr<CartesianAuthorityArena> = {});
    ~IncidencePointDirectory();
    IncidencePointDirectory(const IncidencePointDirectory&) = delete;
    IncidencePointDirectory& operator=(const IncidencePointDirectory&) = delete;
    [[nodiscard]] Point get(std::size_t) const;
    [[nodiscard]] std::optional<std::size_t> find(const Point&) const;
    [[nodiscard]] std::size_t intern(const Point&);
    [[nodiscard]] std::size_t intern(Point&&);
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t source_size() const noexcept { return source_ ? source_->size() : 0; }
    [[nodiscard]] const std::shared_ptr<const ContactSourcePointSnapshot>& source() const noexcept { return source_; }
    [[nodiscard]] Store& physical_pool() noexcept { return *store_; }
    [[nodiscard]] const Store& physical_pool() const noexcept { return *store_; }
    [[nodiscard]] std::shared_ptr<typename Store::PageView> page_view()
    {
        if (source_)
            throw std::invalid_argument("Source points require their own captured namespace.");
        return store_->page_view();
    }
    operator const Store&() const
    {
        if (source_)
            throw std::invalid_argument("Source slots are not physical insertion slots.");
        return *store_;
    }
    operator Store&()
    {
        if (source_)
            throw std::invalid_argument("Source slots are not physical insertion slots.");
        return *store_;
    }
    [[nodiscard]] const std::shared_ptr<Budget>& keeper() const noexcept { return keeper_; }
    [[nodiscard]] Budget& allocation() const noexcept { return *budget_; }
    [[nodiscard]] const std::shared_ptr<CartesianAuthorityArena>& source_arena() const noexcept { return arena_; }
    class PreparedSource
    {
        friend class IncidencePointDirectory;
        PreparedSource() = default;
        std::shared_ptr<const ContactSourcePointSnapshot> snapshot_, prefix_;
        const void* prefix_namespace_ = nullptr;
        bool trusted_prefix_ = false;

    public:
        [[nodiscard]] const std::shared_ptr<const ContactSourcePointSnapshot>& snapshot() const noexcept
        {
            return snapshot_;
        }
    };
    [[nodiscard]] PreparedSource prepare_source(
        const ContactSourcePointAdmission&, std::shared_ptr<CartesianAuthorityArena>);
    template<class Retire = std::nullptr_t>
    [[nodiscard]] bool publish_source(const PreparedSource&, Retire retire = nullptr);
    /** Only private, unexposed initial rows may be rebound by the caller. Old
     * escaped physical views remain attached to their original page owners. */
    template<class Retire = std::nullptr_t>
    [[nodiscard]] bool publish_source(std::shared_ptr<const ContactSourcePointSnapshot>, Retire retire = nullptr);
    [[nodiscard]] std::size_t retained_bytes() const;
    [[nodiscard]] std::size_t metadata_bytes() const;
    [[nodiscard]] std::size_t source_only_metadata_bytes() const noexcept
    {
        return charge_ + store_->source_only_metadata_bytes();
    }
    [[nodiscard]] std::size_t overlap_bytes() const noexcept { return replaced_overlap_ + store_->overlap_bytes(); }
    void finish_overlap() noexcept
    {
        budget_->used -= std::exchange(replaced_overlap_, 0);
        store_->finish_overlap();
    }
    [[nodiscard]] std::size_t page_credit_overhead_bytes() const noexcept
    {
        return store_->page_credit_overhead_bytes();
    }
    [[nodiscard]] bool owns_page_credit(const void* identity) const noexcept
    {
        if (store_->owns_page_credit(identity) || (source_ && identity == source_->admission_identity()))
            return true;
        for (const auto& lease : leases_)
            if (lease.get() == identity)
                return true;
        return false;
    }
    [[nodiscard]] bool owns_page_admission(const void* body, const void* metadata) const noexcept
    {
        if (store_->owns_page_admission(body, metadata) ||
            (source_ && body == source_.get() && source_->admission_identity() == metadata))
            return true;
        for (const auto& lease : leases_)
            if (lease.get() == metadata && lease->owner.lock().get() == body)
                return true;
        return false;
    }
    template<class Visitor> void visit_borrowed_payload(Visitor&&) const;
    template<class Visitor> void visit_page_admissions(Visitor&& visitor) const
    {
        store_->visit_page_admissions(visitor);
    }
    [[nodiscard]] std::size_t page_count() const noexcept { return store_->owner_pages(); }
    [[nodiscard]] std::size_t record_capacity() const noexcept { return store_->record_capacity(); }
    [[nodiscard]] const void* metadata_identity() const noexcept { return &charge_; }
    template<class Visitor> void visit_borrowed_pages(Visitor&& visitor) const
    {
        store_->visit_borrowed_pages(visitor);
    }

private:
    template<class Retire>
    [[nodiscard]] bool publish_source_from(std::shared_ptr<const ContactSourcePointSnapshot>, std::size_t, Retire);
    [[nodiscard]] std::unique_ptr<Store> make_store(const Store* copy, cartesian_detail::Allocation&);
    void credit_source(cartesian_detail::CreditConstruction&, cartesian_detail::Allocation&);
    Budget* budget_;
    std::shared_ptr<Budget> keeper_;
    Hasher hasher_;
    Payload payload_;
    std::unique_ptr<Store> store_;
    std::shared_ptr<const ContactSourcePointSnapshot> source_;
    std::shared_ptr<CartesianAuthorityArena> arena_;
    std::vector<std::shared_ptr<cartesian_detail::AuthorityOwnerCredit>> leases_;
    IdentityPoolStorage storage_;
    std::size_t charge_ = 0, replaced_overlap_ = 0;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
#include "IncidencePointDirectory.ipp"
