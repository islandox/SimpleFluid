/** @file FVM/details/TransportRhs.hh
 * @brief Typed scalar/vector RHS adapters for basic transport assembly.
 */
#pragma once

#include "FVM/AssemblyCallbacks.hh"
#include "SimpleFluidExport.hh"
#include "fields/CellField.hh"
#include "fields/VectorCellField.hh"

#include <Teuchos_RCP.hpp>

#include <cstddef>
#include <optional>
#include <utility>

namespace SimpleFluid::FVM::detail
{

// Declare hot updates inline before the default-pack extern instantiations;
// their .ipp definitions remain available for optimization in consumers.

/** Scalar RHS state with source and old-value sampling before face assembly. */
template<TpetraTypePack Pack> struct SIMPLEFLUID_LOCAL BasicScalarTransportRhs
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using old_view_type = decltype(std::declval<const CellField<Pack>&>().owned_read_view());

    inline BasicScalarTransportRhs(const CellField<Pack>& field, const ScalarBoundaryConditionProvider<Pack>& condition,
        const ScalarBoundaryValueProvider<Pack>& boundary_value, const ScalarCellValueProvider<Pack>& source);

    void initialize(const Mesh<Pack>& mesh);
    inline void begin_cell(local_ordinal_type cell, scalar_type transient, scalar_type volume);
    inline void finish_cell(local_ordinal_type cell, scalar_type, scalar_type) const;
    inline BoundaryConditionType begin_boundary(int batch, size_t face);
    inline void add_boundary_value(local_ordinal_type owner, scalar_type factor) const;

    template<class Metrics>
    inline void add_neumann(
        local_ordinal_type owner, scalar_type diffusivity, const Metrics& metrics, local_ordinal_type face) const;

    bool is_dirichlet(BoundaryConditionType type) const { return type == BoundaryConditionType::Dirichlet; }
    [[noreturn]] void throw_robin() const;

    const CellField<Pack>& field;
    const ScalarBoundaryConditionProvider<Pack>& condition_provider;
    const ScalarBoundaryValueProvider<Pack>& boundary_provider;
    const ScalarCellValueProvider<Pack>& source_provider;
    Teuchos::RCP<typename Pack::vector_type> values;
    std::optional<old_view_type> old_data;
    scalar_type cell_rhs{};
    scalar_type boundary_value{};
    BoundaryCondition condition{};
};

/** Vector RHS state with component old-value reads after matrix-row insertion. */
template<TpetraTypePack Pack> struct SIMPLEFLUID_LOCAL BasicVectorTransportRhs
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using vec_type = typename VectorCellField<Pack>::vec_type;
    using old_view_type = decltype(std::declval<const VectorCellField<Pack>&>().owned_read_view());
    static constexpr size_t num_components = 3;

    inline BasicVectorTransportRhs(const VectorCellField<Pack>& field,
        const VectorBoundaryConditionProvider<Pack>& condition,
        const VectorBoundaryValueProvider<Pack>& boundary_value, const VectorCellValueProvider<Pack>& source);

    void initialize(const Mesh<Pack>& mesh);
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

    const VectorCellField<Pack>& field;
    const VectorBoundaryConditionProvider<Pack>& condition_provider;
    const VectorBoundaryValueProvider<Pack>& boundary_provider;
    const VectorCellValueProvider<Pack>& source_provider;
    Teuchos::RCP<typename Pack::multi_vector_type> values;
    std::optional<old_view_type> old_data;
    vec_type cell_source{};
    vec_type boundary_value{};
    VectorBoundaryCondition condition{};
};

extern template struct BasicScalarTransportRhs<DefaultTpetraTypes>;
extern template struct BasicVectorTransportRhs<DefaultTpetraTypes>;

} // namespace SimpleFluid::FVM::detail

#include "FVM/details/TransportRhs.ipp"
