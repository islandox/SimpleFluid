/**
 * @file FieldStateTransaction.hh
 * @brief  Lightweight owned-field snapshots.
 */

#pragma once

#include <stdexcept>
#include <Tpetra_Access.hpp>
#include <Teuchos_CommHelpers.hpp>
#include "fields/Field.hh"
#include <functional>
#include <memory>
#include <utility>
#include <type_traits>
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
          d_overlap_map(field.overlap_data().getMap())
    {
        d_rows = field.owned_data().getLocalLength();
        d_columns = field.owned_data().getNumVectors();
        d_values.resize(d_rows * d_columns);
        if (field.owned_data().isConstantStride())
        {
            const auto values = field.owned_data().getLocalViewHost(Tpetra::Access::ReadOnly);
            for (size_t column = 0; column < d_columns; ++column)
                for (size_t row = 0; row < d_rows; ++row)
                    d_values[column * d_rows + row] = values(row, column);
        }
        else
        {
            for (size_t column = 0; column < d_columns; ++column)
            {
                const auto values = field.owned_data().getData(column);
                for (size_t row = 0; row < d_rows; ++row)
                    d_values[column * d_rows + row] = values[row];
            }
        }
    }

    void restore(Field& field) const
    {
        const int local_valid = !(&field != d_field || &field.mesh() != d_mesh
            || field.owned_data().getMap().getRawPtr() != d_owned_map.getRawPtr()
            || field.overlap_data().getMap().getRawPtr() != d_overlap_map.getRawPtr()
            || !mesh_maps_match(field)
            || field.num_owned_entries() != d_rows
            || field.owned_data().getNumVectors() != d_columns);
        int all_valid = 0;
        Teuchos::reduceAll(*d_owned_map->getComm(), Teuchos::REDUCE_MIN, 1,
                          &local_valid, &all_valid);
        if (!all_valid)
            throw std::invalid_argument("FieldStateSnapshot belongs to another field or topology.");
        if (field.owned_data().isConstantStride())
        {
            auto values = field.owned_data().getLocalViewHost(Tpetra::Access::OverwriteAll);
            for (size_t column = 0; column < d_columns; ++column)
                for (size_t row = 0; row < d_rows; ++row)
                    values(row, column) = d_values[column * d_rows + row];
        }
        else
        {
            // Logical column access preserves unselected backing columns.
            for (size_t column = 0; column < d_columns; ++column)
            {
                auto values = field.owned_data().getDataNonConst(column);
                for (size_t row = 0; row < d_rows; ++row)
                    values[row] = d_values[column * d_rows + row];
            }
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
    size_t d_rows = 0, d_columns = 0;
    std::vector<typename Field::scalar_type> d_values;
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
