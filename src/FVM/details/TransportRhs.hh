/** @file FVM/details/TransportRhs.hh
 * @brief Typed native vector RHS adapter for basic transport assembly.
 */
#pragma once
#include "utils/export_helper.hh"

#include "FVM/AssemblyCallbacks.hh"
#include "utils/export_helper.hh"
#include "fields/FieldStored.hh"

#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <optional>
#include <utility>

namespace SimpleFluid::FVM::detail
{

/** Vector RHS staging with component old-value reads after numerical row construction. */
template<TpetraTypePack Pack, SimpleFluid::MeshType MeshType> struct SIMPLEFLUID_LOCAL BasicVectorTransportRhs
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename VectorCellFieldStored<Pack, MeshType>::vec_type;
    using old_view_type = decltype(std::declval<const VectorCellFieldStored<Pack, MeshType>&>().owned_read_view());
    static constexpr size_t num_components = 3;

    inline BasicVectorTransportRhs(const VectorCellFieldStored<Pack, MeshType>& field,
        const VectorBoundaryConditionProvider<Pack>& condition,
        const VectorBoundaryValueProvider<Pack>& boundary_value, const VectorCellValueProvider<Pack>& source);

    void initialize(const MeshType& mesh);
    inline void begin_cell(local_ordinal_type cell, scalar_type, scalar_type);
    inline void finish_cell(local_ordinal_type cell, scalar_type transient, scalar_type volume) const;
    inline BoundaryConditionType begin_boundary(int batch, size_t face);
    inline void add_boundary_value(local_ordinal_type owner, scalar_type factor) const;

    template<class Metrics>
    inline void add_neumann(
        local_ordinal_type owner, scalar_type diffusivity, const Metrics& metrics, local_ordinal_type face) const;

    bool is_dirichlet(BoundaryConditionType type) const
    {
        return type == BoundaryConditionType::Dirichlet || type == BoundaryConditionType::NoSlip;
    }
    [[noreturn]] void throw_robin() const;

    const VectorCellFieldStored<Pack, MeshType>& field;
    const VectorBoundaryConditionProvider<Pack>& condition_provider;
    const VectorBoundaryValueProvider<Pack>& boundary_provider;
    const VectorCellValueProvider<Pack>& source_provider;
    Teuchos::RCP<typename Pack::multi_vector_type> values;
    std::optional<old_view_type> old_data;
    vec_type cell_source{};
    vec_type boundary_value{};
    VectorBoundaryCondition condition{};
};

} // namespace SimpleFluid::FVM::detail

#include "FVM/details/TransportRhs.ipp"
