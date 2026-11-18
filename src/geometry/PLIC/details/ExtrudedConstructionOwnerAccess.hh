#pragma once
#include "geometry/PLIC/ExtrudedConstruction.hh"
#include "geometry/details/NativeConstructorOwnerSink.hh"
namespace SimpleFluid::PLIC::details
{
class ExtrudedConstructionOwnerAccess
{
public:
    using Support = PlaneSupportIdentity<ArithmeticMode::Ordinary>;
    using IdentityVisit = void (*)(void*, const Support&);
    using TopologyVisit = void (*)(void*, const std::shared_ptr<const Meshes::ExtrudedTopology>&);
    static void visit(const std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Ordinary>>&,
        const geometry_detail::NativeConstructorOwnerSink&, IdentityVisit, TopologyVisit);
};
} // namespace SimpleFluid::PLIC::details
