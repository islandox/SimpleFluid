#pragma once

namespace SimpleFluid::PLIC::details
{
template<class Visitor> void visit_provenance_children(const DynamicPointProvenance& point, Visitor&& visitor)
{
    if (point.source_origin)
        visitor(point.source_origin);
    for (const auto& edge : point.source_edges)
    {
        visitor(edge[0]);
        visitor(edge[1]);
    }
}
} // namespace SimpleFluid::PLIC::details
