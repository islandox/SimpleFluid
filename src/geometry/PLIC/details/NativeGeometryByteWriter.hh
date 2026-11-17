/** Private bounded borrowed byte destination; owns no storage. */
#pragma once
#include "geometry/details/NativeConstructionAdmission.hh"
#include "geometry/PLIC/details/GeometryIdentityAccess.hh"
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace SimpleFluid::PLIC::details
{
struct NativeGeometryByteWriter
{
    std::vector<char>* output;
    const geometry_detail::NativeConstructionAdmission* admission;
    template<class Identity> void native_identity(const Identity& identity)
    {
        const GeometryIdentityCanonicalWriter writer{this, [](void* context, std::span<const char> bytes)
            {
                auto& destination = *static_cast<NativeGeometryByteWriter*>(context);
                destination.insert(destination.end(), bytes.data(), bytes.data() + bytes.size());
            }};
        if constexpr (Identity::shared_storage)
        {
            const auto& owner = GeometryIdentityAccess::backing(identity);
            if (!owner)
                return;
            if (!admission)
                throw std::invalid_argument("Native geometry identity needs its exact allocation domain.");
            charge_geometry_identity_scratch(owner->depth, sizeof(const GeometryIdentityBacking*));
            geometry_detail::NativeConstructorVector<const GeometryIdentityBacking*> ancestry(
                admission->allocator<const GeometryIdentityBacking*>());
            ancestry.reserve(owner->depth);
            for (auto* node = owner.get(); node; node = node->parent.get())
            {
                node->native->emit_prefix(writer, node->parent ? node->parent->canonical_size : 0);
                ancestry.push_back(node);
            }
            for (auto node = ancestry.rbegin(); node != ancestry.rend(); ++node)
                (*node)->native->emit_suffix(writer);
        }
        else
            GeometryIdentityAccess::emit_canonical(identity, writer.context, writer.append);
    }
    void push_back(char value)
    {
        if (output->size() == output->capacity())
            throw std::invalid_argument("Native geometry destination exceeded its exact admission.");
        output->push_back(value);
    }
    std::vector<char>::iterator end() const noexcept { return output->end(); }
    void insert(std::vector<char>::iterator position, const char* first, const char* last)
    {
        const auto size = static_cast<std::size_t>(last - first);
        if (size > output->capacity() - output->size())
            throw std::invalid_argument("Native geometry identity exceeded its exact admission.");
        output->insert(position, first, last);
    }
};
} // namespace SimpleFluid::PLIC::details
