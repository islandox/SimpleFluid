/** Private direct Ordinary cell-certificate encoding in one native domain. */
#pragma once
#include "geometry/PLIC/ConvexGeometry.hh"
#include "geometry/details/NativeConstructorStorage.hh"
#include <stdexcept>
namespace SimpleFluid::PLIC::details
{
class SIMPLEFLUID_LOCAL NativeCellCodecAccess final
{
public:
    [[nodiscard]] static ConvexGeometryPtr<ArithmeticMode::Ordinary> decode(
        std::span<const char>, std::size_t, const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static std::size_t size(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&, const geometry_detail::NativeConstructionAdmission&);
    static void append(std::vector<char>&, const ConvexCellGeometry<ArithmeticMode::Ordinary>&,
        const geometry_detail::NativeConstructionAdmission&);
    [[nodiscard]] static std::vector<char> encode(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&, const geometry_detail::NativeConstructorStorageOwner&);
    /** Private split intake holds all profiled numerical requests live while
     * its callback retires the legacy child bodies. */
    static void with_split_profile(const geometry_detail::NativeConstructionAdmission&, void*, void (*)(void*));
    [[nodiscard]] static bool split_profile_active() noexcept;
    [[nodiscard]] static geometry_detail::NativeConstructionAdmission split_profile_admission();
    static void charge_split_profile_request(std::size_t count, std::size_t width);
    template<class T> static void reserve_split_profile(std::vector<T>& values, std::size_t size)
    {
        if (size <= values.capacity())
            return;
        charge_split_profile_request(size, sizeof(T));
        values.reserve(size);
        if (split_profile_active() && values.capacity() != size)
            throw std::invalid_argument("Ordinary split vector capacity differs from its admitted profile.");
    }
    template<class T> static void reserve_next_split_profile(std::vector<T>& values)
    {
        if (!split_profile_active() || values.size() != values.capacity())
            return;
        if (values.size() > values.max_size() / 2)
            throw std::invalid_argument("Native Ordinary split vector extent overflows.");
        reserve_split_profile(values, values.empty() ? 1 : values.size() * 2);
    }
    [[nodiscard]] static std::vector<ConvexOctantPiece<ArithmeticMode::Ordinary>> profiled_octants(
        const ConvexCellGeometry<ArithmeticMode::Ordinary>&, double);
};
} // namespace SimpleFluid::PLIC::details
