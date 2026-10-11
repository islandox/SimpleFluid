/**
 * @file FieldStateTransaction.hh
 * @brief  Lightweight owned-field snapshots.
 */

#pragma once

#include <stdexcept>
#include <Teuchos_CommHelpers.hpp>
#include "fields/Field.hh"
#include <type_traits>
#include <functional>
#include <memory>
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

    explicit FieldStateSnapshot(const Field& field)
        : d_field(&field), d_mesh(&field.mesh()),
          d_owned_map(field.owned_data().getMap()),
          d_overlap_map(field.overlap_data().getMap()),
          d_columns(field.owned_data().getNumVectors())
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
        const int local_valid = !(&field != d_field || &field.mesh() != d_mesh
            || field.owned_data().getMap().getRawPtr() != d_owned_map.getRawPtr()
            || field.overlap_data().getMap().getRawPtr() != d_overlap_map.getRawPtr()
            || !mesh_maps_match(field)
            || d_local_ids.size() != d_values.size()
            || field.num_owned_entries() != d_values.size()
            || field.owned_data().getNumVectors() != d_columns);
        int all_valid = 0;
        Teuchos::reduceAll(*d_owned_map->getComm(), Teuchos::REDUCE_MIN, 1,
                          &local_valid, &all_valid);
        if (!all_valid)
            throw std::invalid_argument("FieldStateSnapshot belongs to another field or topology.");
        for (size_t entry = 0; entry < d_values.size(); ++entry)
        {
            field.set_owned_value(d_local_ids[entry], d_values[entry]);
        }
        field.sync_ghosts();
    }

private:
    bool mesh_maps_match(const Field& field) const
    {
        if constexpr (std::is_same_v<typename Field::location_type, CellLocation>)
            return field.mesh().owned_cell_map().getRawPtr() == d_owned_map.getRawPtr()
                && field.mesh().overlap_cell_map().getRawPtr() == d_overlap_map.getRawPtr();
        else if constexpr (std::is_same_v<typename Field::location_type, FaceLocation>)
            return field.mesh().owned_face_map().getRawPtr() == d_owned_map.getRawPtr()
                && field.mesh().overlap_face_map().getRawPtr() == d_overlap_map.getRawPtr();
        else
            return field.mesh().boundary_face_map().getRawPtr() == d_owned_map.getRawPtr()
                && field.mesh().boundary_face_map().getRawPtr() == d_overlap_map.getRawPtr();
    }

    const Field* d_field = nullptr;
    const typename Field::mesh_type* d_mesh = nullptr;
    Teuchos::RCP<const typename Field::map_type> d_owned_map, d_overlap_map;
    size_t d_columns = 0;
    std::vector<local_ordinal_type> d_local_ids;
    std::vector<value_type> d_values;
};

/** Snapshot actions are captured only for an explicitly enabled transaction. */
class AcceptedStateRollback
{
public:
    template<class Field> void capture(Field& field)
    {
        actions.emplace_back([snapshot = FieldStateSnapshot(field), &field] { snapshot.restore(field); });
    }
    template<class Model> void capture_model(Model& model)
    {
        auto saved = std::make_shared<typename Model::StateSnapshot>(model.snapshot());
        actions.emplace_back([saved = std::move(saved), &model] { model.restore(*saved); });
    }
    void add(std::function<void()> action) { actions.push_back(std::move(action)); }
    void restore() const { for (const auto& action : actions) action(); }
private:
    std::vector<std::function<void()>> actions;
};

} // namespace SimpleFluid
