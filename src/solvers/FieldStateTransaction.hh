/** @file FieldStateTransaction.hh @brief Lightweight owned-field snapshots. */

#pragma once

#include <stdexcept>
#include <utility>
#include <vector>

namespace SimpleFluid
{

/**
 * @brief Snapshot authoritative owned values of one FieldStored instance.
 *
 * The snapshot is deliberately tied to the exact field object and topology.
 * Geometry may move because fixed-topology local identifiers and maps remain
 * unchanged. Restoring also synchronizes overlap storage before stencil reads.
 */
template<class Field> class FieldStateSnapshot
{
public:
    using value_type = typename Field::value_type;
    using local_ordinal_type = typename Field::local_ordinal_type;

    explicit FieldStateSnapshot(const Field& field) : d_field(&field)
    {
        if constexpr (requires { field.owned_face_ids(); })
        {
            d_local_ids = field.owned_face_ids();
        }
        else
        {
            d_local_ids.reserve(field.num_owned_entries());
            for (size_t owned = 0; owned < field.num_owned_entries(); ++owned)
            {
                d_local_ids.push_back(static_cast<local_ordinal_type>(owned));
            }
        }
        d_values.reserve(d_local_ids.size());
        for (const auto local_id : d_local_ids)
        {
            d_values.push_back(field.value(local_id));
        }
    }

    void restore(Field& field) const
    {
        if (&field != d_field || d_local_ids.size() != d_values.size() || field.num_owned_entries() != d_values.size())
        {
            throw std::invalid_argument("FieldStateSnapshot belongs to another field or topology.");
        }
        for (size_t entry = 0; entry < d_values.size(); ++entry)
        {
            field.set_owned_value(d_local_ids[entry], d_values[entry]);
        }
        field.sync_ghosts();
    }

private:
    const Field* d_field = nullptr;
    std::vector<local_ordinal_type> d_local_ids;
    std::vector<value_type> d_values;
};

} // namespace SimpleFluid
