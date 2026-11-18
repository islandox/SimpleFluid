/** @file NativeSplitCppProfile.hh
 * @brief Compiler-profiled Ordinary split workspace before typed child construction.
 */
#pragma once
#include "geometry/PLIC/details/PointSnapshot.hh"
#include "geometry/details/NativeConstructionAdmission.hh"
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace SimpleFluid::PLIC::details::ordinary_kernel
{
/** Accumulated upper extent for legacy requests. An independently admitted
 * typed child may replace a legacy child while this phase is still live. */
class SIMPLEFLUID_LOCAL NativeSplitCppProfile final
{
public:
    explicit NativeSplitCppProfile(const geometry_detail::NativeConstructionAdmission& admission)
        : state_(admission.state()), previous_(current_)
    {
        current_ = this;
    }
    NativeSplitCppProfile(const NativeSplitCppProfile&) = delete;
    NativeSplitCppProfile& operator=(const NativeSplitCppProfile&) = delete;
    ~NativeSplitCppProfile()
    {
        current_ = previous_;
        if (bytes_)
            state_->finish_profile(bytes_, false);
    }

    [[nodiscard]] static bool active() noexcept { return current_ != nullptr; }
    [[nodiscard]] static geometry_detail::NativeConstructionAdmission admission()
    {
        if (!current_)
            throw std::logic_error("Ordinary split has no active native admission.");
        return geometry_detail::NativeConstructionAdmission::from_state(current_->state_);
    }
    class Suspension
    {
    public:
        Suspension() noexcept : saved_(std::exchange(current_, nullptr)) {}
        Suspension(const Suspension&) = delete;
        Suspension& operator=(const Suspension&) = delete;
        ~Suspension() { current_ = saved_; }

    private:
        NativeSplitCppProfile* saved_;
    };
    static void request(std::size_t count, std::size_t width)
    {
        if (!current_ || !count)
            return;
        if (width > (std::numeric_limits<std::size_t>::max() - current_->bytes_) / count)
            throw std::invalid_argument("Native Ordinary split request extent overflows.");
        const auto bytes = count * width;
        current_->state_->reserve(bytes, false);
        current_->bytes_ += bytes;
    }

    // Keep the existing PointSnapshot logical Work charge. Its append path
    // requests the next full vector capacity before destroying the old buffer.
    [[nodiscard]] static PointSnapshotAllocation point_requests() noexcept
    {
        if (!active())
            return {};
        return {nullptr, [](void*, std::size_t count, std::size_t width)
            {
                charge_point_snapshot(count, width);
                request(count, width);
            }};
    }
    template<class T> static void reserve(std::vector<T>& values, std::size_t size)
    {
        if (size > values.capacity())
        {
            request(size, sizeof(T));
            values.reserve(size);
            if (active() && values.capacity() != size)
                throw std::invalid_argument("Ordinary split vector capacity differs from its admitted profile.");
        }
    }
    template<class C> static void next(C& values)
    {
        if (!active() || values.size() != values.capacity())
            return;
        if (values.size() > values.max_size() / 2)
            throw std::invalid_argument("Native Ordinary split vector extent overflows.");
        reserve(values, values.empty() ? 1 : values.size() * 2);
    }
    template<class Value> static void node()
    {
#if defined(__GLIBCXX__)
        request(1, sizeof(std::_Rb_tree_node<Value>));
#else
        if (active())
            throw std::invalid_argument("Unsupported native Ordinary split tree allocation profile.");
#endif
    }
    template<class T> static void combined()
    {
#if defined(__GLIBCXX__)
        request(1, sizeof(std::_Sp_counted_ptr_inplace<T, std::allocator<void>, __gnu_cxx::__default_lock_policy>));
#else
        if (active())
            throw std::invalid_argument("Unsupported native Ordinary split control allocation profile.");
#endif
    }
    template<class T> static void separately_controlled()
    {
#if defined(__GLIBCXX__)
        request(1, sizeof(T) + sizeof(std::_Sp_counted_ptr<T*, __gnu_cxx::__default_lock_policy>));
#else
        if (active())
            throw std::invalid_argument("Unsupported native Ordinary split control allocation profile.");
#endif
    }

private:
    static inline thread_local NativeSplitCppProfile* current_ = nullptr;
    std::shared_ptr<const geometry_detail::NativeAllocationState> state_;
    NativeSplitCppProfile* previous_;
    std::size_t bytes_ = 0;
};
} // namespace SimpleFluid::PLIC::details::ordinary_kernel
