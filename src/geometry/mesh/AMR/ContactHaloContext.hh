#pragma once
#include <compare>
#include <cstdint>
namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ContactHaloContext
{
    std::uint64_t epoch = 0, topology = 0, partition = 0, round = 0, transaction = 0;
    auto operator<=>(const ContactHaloContext&) const = default;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
