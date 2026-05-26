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

#include <Tpetra_Map.hpp>
#include <Teuchos_Comm.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Kokkos_Core.hpp>

namespace SimpleFluid
{

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
};

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