/** @file NativeSchemaOwner.hh
 * @brief Detached immutable schema bytes, with admitted copy or input adoption.
 */
#pragma once
#include "geometry/details/NativeAllocationEnvelope.hh"
#include <algorithm>
#include <span>
#include <variant>

namespace SimpleFluid::geometry_detail
{
class NativeSchemaOwnerAccess;
namespace native_schema_detail
{
/** Existing input bytes are admitted before adoption. This is retained-input
 * credit, not a claim that this domain performed their earlier allocation.
 */
class InputCharge
{
public:
    InputCharge(NativeConstructionAdmission admission, std::size_t bytes)
        : extent_(admission.reserve_profile({bytes}).retain_profile(bytes)) {}
    InputCharge(InputCharge&&) noexcept = default;
    InputCharge(const InputCharge&) = delete;
    [[nodiscard]] const NativeRetainedExtent& extent() const noexcept { return extent_; }
private:
    NativeRetainedExtent extent_;
};
struct AdoptedBytes
{
    // Buffer destruction precedes charge retirement, including every throw.
    InputCharge charge;
    std::vector<char> bytes;
    AdoptedBytes(std::vector<char>&& input, const NativeConstructionAdmission& admission)
        : charge(admission, input.capacity()), bytes(std::move(input))
    {
    }
    AdoptedBytes(AdoptedBytes&&) noexcept = default;
    AdoptedBytes(const AdoptedBytes&) = delete;
};
using Bytes = std::variant<AdoptedBytes, NativeConstructorVector<char>>;
} // namespace native_schema_detail

/** No kernel, hierarchy, source layout or configuration controller is retained.
 * Normal publication borrows span(), while descriptor() remains an explicit
 * allocating compatibility export.
 */
class SIMPLEFLUID_LOCAL NativeSchemaOwner final
{
    class ConstructionKey
    {
        friend class NativeSchemaOwnerAccess;
        ConstructionKey() = default;

    public:
        ConstructionKey(const ConstructionKey&) = default;
        ConstructionKey(ConstructionKey&&) = default;
    };

public:
    NativeSchemaOwner(ConstructionKey, native_schema_detail::Bytes&& bytes) : bytes_(std::move(bytes)) {}
    NativeSchemaOwner(const NativeSchemaOwner&) = delete;
    NativeSchemaOwner& operator=(const NativeSchemaOwner&) = delete;
    NativeSchemaOwner(NativeSchemaOwner&&) = delete;
    NativeSchemaOwner& operator=(NativeSchemaOwner&&) = delete;
    [[nodiscard]] std::span<const char> span() const noexcept
    {
        return std::visit(
            [](const auto& bytes) -> std::span<const char>
            {
                if constexpr (requires { bytes.bytes; })
                    return bytes.bytes;
                else
                    return bytes;
            },
            bytes_);
    }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept
    {
        return std::visit(
            [](const auto& bytes) -> std::size_t
            {
                if constexpr (requires { bytes.bytes; })
                    return bytes.bytes.capacity();
                else
                    return bytes.capacity();
            },
            bytes_);
    }
    /** Adopting an already allocated caller buffer is a retained-input upper
     * extent, distinct from native allocator request statistics. */
    [[nodiscard]] const NativeRetainedExtent* retained_input_extent() const noexcept
    {
        const auto* adopted = std::get_if<native_schema_detail::AdoptedBytes>(&bytes_);
        return adopted ? &adopted->charge.extent() : nullptr;
    }
    [[nodiscard]] bool matches(std::span<const char> bytes) const noexcept { return std::ranges::equal(span(), bytes); }
    [[nodiscard]] std::vector<char> descriptor() const
    {
        const auto bytes = span();
        return {bytes.begin(), bytes.end()};
    }
    [[nodiscard]] const std::shared_ptr<const NativeAllocationEnvelope>& allocation() const noexcept
    {
        return allocation_;
    }

private:
    friend class NativeSchemaOwnerAccess;
    native_schema_detail::Bytes bytes_;
    std::shared_ptr<const NativeAllocationEnvelope> allocation_;
};
class SIMPLEFLUID_LOCAL NativeSchemaOwnerAccess
{
public:
    [[nodiscard]] static std::shared_ptr<const NativeSchemaOwner> copy(
        std::span<const char> bytes, const NativeConstructionAdmission& admission)
    {
        NativeConstructorVector<char> input(admission.allocator<char>());
        if (!bytes.empty())
        {
            input.reserve(bytes.size());
            input.assign(bytes.begin(), bytes.end());
        }
        native_schema_detail::Bytes body(std::in_place_type<NativeConstructorVector<char>>, std::move(input));
        return seal(std::move(body), admission);
    }
    [[nodiscard]] static std::shared_ptr<const NativeSchemaOwner> adopt(
        std::vector<char>&& bytes, const NativeConstructionAdmission& admission)
    {
        native_schema_detail::Bytes body(
            std::in_place_type<native_schema_detail::AdoptedBytes>, std::move(bytes), admission);
        return seal(std::move(body), admission);
    }

private:
    [[nodiscard]] static std::shared_ptr<const NativeSchemaOwner> seal(
        native_schema_detail::Bytes&& bytes, const NativeConstructionAdmission& admission)
    {
        auto tracked = NativeAllocationEnvelopeAccess::make<NativeSchemaOwner>(
            admission, NativeSchemaOwner::ConstructionKey{}, std::move(bytes));
        tracked.owner->allocation_ = tracked.envelope;
        return std::move(tracked.owner);
    }
};
} // namespace SimpleFluid::geometry_detail
