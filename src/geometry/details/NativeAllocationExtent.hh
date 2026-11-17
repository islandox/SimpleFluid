#pragma once
#include "utils/export_helper.hh"
#include <array>
#include <cstddef>
namespace SimpleFluid::geometry_detail
{
struct SIMPLEFLUID_LOCAL NativeAllocationExtent
{
    const void* address = nullptr;
    std::size_t bytes = 0;
};
struct SIMPLEFLUID_LOCAL NativeCounterAllocationEnvelope
{
    NativeAllocationExtent object, control;
    NativeAllocationExtent most_recent_allocation;
    bool admission_state_created = false;
};
} // namespace SimpleFluid::geometry_detail
