/**
 * @file MeshIndexTypes.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Type-pack definitions and constraints for CRTP meshes.
 * @version 0.1
 * @date 2026-06-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/typedefs.hh"

#include <concepts>
#include <limits>

namespace SimpleFluid
{

/**
 * @brief Require the entity IDs and compatible integral ordinals of a mesh pack.
 * @tparam Pack Candidate mesh index type pack.
 */
template <class Pack>
concept MeshIndexTypePack = requires {
    typename Pack::cell_id_t;
    typename Pack::face_id_t;
    typename Pack::node_id_t;
    typename Pack::local_cell_id_t;
    typename Pack::local_face_id_t;
    typename Pack::local_node_id_t;
    typename Pack::global_cell_id_t;
    typename Pack::global_face_id_t;
    typename Pack::global_node_id_t;
    typename Pack::local_ordinal_type;
    typename Pack::global_ordinal_type;
} && std::integral<typename Pack::local_ordinal_type>
  && std::integral<typename Pack::global_ordinal_type>
  && (std::numeric_limits<typename Pack::local_ordinal_type>::max()
      <= std::numeric_limits<typename Pack::global_ordinal_type>::max());

/**
 * @brief Require reversible cell, face, and node ID-to-ordinal conversions.
 * @tparam Indexer Candidate mesh indexer type.
 */
template<class Indexer>
concept MeshIndexer = requires {
    typename Indexer::cell_id_t;
    typename Indexer::face_id_t;
    typename Indexer::node_id_t;
    typename Indexer::ordinal_t;
} && std::integral<typename Indexer::ordinal_t>
  && requires(const Indexer& indexer,
              typename Indexer::ordinal_t ordinal,
              const typename Indexer::cell_id_t& cell_id,
              const typename Indexer::face_id_t& face_id,
              const typename Indexer::node_id_t& node_id)
{
    { indexer.cell_ordinal(cell_id) }
        -> std::same_as<typename Indexer::ordinal_t>;
    { indexer.face_ordinal(face_id) }
        -> std::same_as<typename Indexer::ordinal_t>;
    { indexer.node_ordinal(node_id) }
        -> std::same_as<typename Indexer::ordinal_t>;
    { indexer.cell_id(ordinal) }
        -> std::same_as<typename Indexer::cell_id_t>;
    { indexer.face_id(ordinal) }
        -> std::same_as<typename Indexer::face_id_t>;
    { indexer.node_id(ordinal) }
        -> std::same_as<typename Indexer::node_id_t>;
};

/**
 * @brief Type pack aggregating the five index types used by CRTP meshes.
 *
 * @tparam CID Cell ID type.
 * @tparam FID Face ID type.
 * @tparam NID Node ID type.
 * @tparam LO Local ordinal type (index within a partition).
 * @tparam GO Global ordinal type (index across all partitions).
 * @tparam LCID Rank-local cell ID type.
 * @tparam LFID Rank-local face ID type.
 * @tparam LNID Rank-local node ID type.
 */
template <class CID,
          class FID,
          class NID,
          class LO,
          class GO,
          class LCID = CID,
          class LFID = FID,
          class LNID = NID>
struct MeshIndexTypes
{
    using cell_id_t = CID;
    using face_id_t = FID;
    using node_id_t = NID;
    using local_cell_id_t = LCID;
    using local_face_id_t = LFID;
    using local_node_id_t = LNID;
    using global_cell_id_t = CID;
    using global_face_id_t = FID;
    using global_node_id_t = NID;
    using local_ordinal_type = LO;
    using global_ordinal_type = GO;

    template<class NewLocalOrdinal, class NewGlobalOrdinal>
    using rebind_ordinals = MeshIndexTypes<
        CID,
        FID,
        NID,
        NewLocalOrdinal,
        NewGlobalOrdinal,
        LCID,
        LFID,
        LNID>;
};

} // namespace SimpleFluid
