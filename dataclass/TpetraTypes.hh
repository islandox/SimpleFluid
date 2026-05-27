/**
 * @file TpetraTypes.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-22
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "typedefs.hh"

#include <limits>

#include <Tpetra_Map.hpp>
#include <Teuchos_Comm.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Kokkos_Core.hpp>

namespace SimpleFluid
{

/**
 * @brief concept for Tpetra type packs, which group together related Tpetra types for convenience.
 * 
 * @details the pack shall satify:
 * - define all the required types (scalar_type, local_ordinal_type, global_ordinal_type, comm_type, node_type, map_type, graph_type, device_type, execution_space, memory_space, size_type, vector_type)
 * - have local_ordinal_type and global_ordinal_type such that all local ordinals can be represented as global ordinals.
 * 
 * @tparam Pack 
 */
template <class Pack>
concept TpetraTypePack = requires {
    typename Pack::scalar_type;
    typename Pack::local_ordinal_type;
    typename Pack::global_ordinal_type;
    typename Pack::comm_type;
    typename Pack::node_type;
    typename Pack::map_type;
    typename Pack::graph_type;
    typename Pack::device_type;
    typename Pack::execution_space;
    typename Pack::memory_space;
    typename Pack::size_type;

    typename Pack::vector_type;

} && (std::numeric_limits<typename Pack::local_ordinal_type>::max() <= 
    std::numeric_limits<typename Pack::global_ordinal_type>::max());

template <class Scalar = real_t,
          class LO = local_index_t,
          class GO = global_index_t,
          class Node = typename Tpetra::Map<LO, GO>::node_type>
struct TpetraTypes
{
    using Map = Tpetra::Map<LO, GO, Node>;
    using Graph = Tpetra::CrsGraph<LO, GO, Node>;

    using Comm = Teuchos::Comm<int>;
    using Vector = Tpetra::Vector<Scalar, LO, GO, Node>;

    using scalar_type = Scalar;
    using local_ordinal_type = LO;
    using global_ordinal_type = GO;
    using node_type = Node;
    using comm_type = Teuchos::Comm<int>;

    using map_type = Tpetra::Map<LO, GO, Node>;
    using graph_type = Tpetra::CrsGraph<LO, GO, Node>;

    using device_type = typename Node::device_type;
    using execution_space = typename device_type::execution_space;
    using memory_space = typename device_type::memory_space;

    using size_type = typename Kokkos::View<LO*, device_type>::size_type;

    using vector_type = Tpetra::Vector<Scalar, LO, GO, Node>;
};

using DefaultTpetraTypes = TpetraTypes<>;

}