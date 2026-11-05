#pragma once
#include <cstddef>
#include <cstdint>

namespace SimpleFluid::Meshes::AMR::adaptive_detail
{
struct ContactPointRef
{
    std::size_t slot;
    std::uint64_t generation;
    bool operator==(const ContactPointRef&) const = default;
};
} // namespace SimpleFluid::Meshes::AMR::adaptive_detail
