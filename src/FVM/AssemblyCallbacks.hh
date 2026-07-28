/**
 * @file FVM/AssemblyCallbacks.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Type-erased callback signatures used by compiled FVM assembly templates.
 * @version 0.1
 * @date 2026-07-29
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once

#include "equations/BoundaryConditions.hh"
#include "fields/VectorCellField.hh"

#include <cstddef>
#include <functional>

namespace SimpleFluid::FVM
{

/**
 * @brief Scalar boundary-condition lookup for one boundary-batch face.
 * @tparam Pack Tpetra type pack used by the assembled operator.
 */
template<TpetraTypePack Pack> using ScalarBoundaryConditionProvider = std::function<BoundaryCondition(int, size_t)>;

/**
 * @brief Vector boundary-condition lookup for one boundary-batch face.
 * @tparam Pack Tpetra type pack used by the assembled operator.
 */
template<TpetraTypePack Pack>
using VectorBoundaryConditionProvider = std::function<VectorBoundaryCondition(int, size_t)>;

/**
 * @brief Scalar boundary-value lookup for one boundary-batch face.
 * @tparam Pack Tpetra type pack defining the scalar type.
 */
template<TpetraTypePack Pack>
using ScalarBoundaryValueProvider = std::function<typename Pack::scalar_type(int, size_t)>;

/**
 * @brief Vector boundary-value lookup for one boundary-batch face.
 * @tparam Pack Tpetra type pack defining the vector-field type.
 */
template<TpetraTypePack Pack>
using VectorBoundaryValueProvider = std::function<typename VectorCellField<Pack>::vec_type(int, size_t)>;

/**
 * @brief Scalar cell-value lookup for an owned local cell.
 * @tparam Pack Tpetra type pack defining scalar and ordinal types.
 */
template<TpetraTypePack Pack>
using ScalarCellValueProvider = std::function<typename Pack::scalar_type(typename Pack::local_ordinal_type)>;

/**
 * @brief Vector cell-value lookup for an owned local cell.
 * @tparam Pack Tpetra type pack defining vector and ordinal types.
 */
template<TpetraTypePack Pack>
using VectorCellValueProvider =
    std::function<typename VectorCellField<Pack>::vec_type(typename Pack::local_ordinal_type)>;

/** @brief Select boundary-batch faces that receive an operator term. */
using BoundaryFaceSelector = std::function<bool(int, size_t)>;

/**
 * @brief Boundary coefficient lookup with an owner-cell fallback value.
 * @tparam Pack Tpetra type pack defining the coefficient scalar type.
 */
template<TpetraTypePack Pack>
using BoundaryCoefficientProvider = std::function<typename Pack::scalar_type(int, size_t, typename Pack::scalar_type)>;

} // namespace SimpleFluid::FVM
