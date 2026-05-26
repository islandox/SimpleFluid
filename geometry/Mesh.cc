/**
 * @file Mesh.cc
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "Mesh.hh"
#include "Mesh.tcc"

namespace SimpleFluid
{
    template class Mesh<DefaultTpetraTypes>;
//    template class Mesh<TpetraTypes<real_t, local_index_t, global_index_t, 
//        Tpetra::KokkosCompat::KokkosDeviceWrapperNode<Kokkos::OpenMP> > >;
}
