/** @brief Implementation provider for the symbolic-plan fixture's custom mesh. */
#include "FVM/details/FieldStoredTransportSystem.tcc"
#include "TestMeshes.hh"

namespace SimpleFluid::FVM::detail
{
using Pack = DefaultTpetraTypes;
using EpochMesh = test::transport_providers::EpochMesh;
using Row = StoredTransportMatrixRow<Pack>;

template bool StoredTransportSymbolicPlan<Pack>::matches<EpochMesh>(
    const EpochMesh&, const Teuchos::RCP<Pack::matrix_type>&, const std::vector<Row>&) const;
template StoredPreparedTransportMatrix<Pack> prepare_stored_transport_matrix<Pack, EpochMesh>(const EpochMesh&,
    Teuchos::RCP<Pack::matrix_type>, size_t, const std::vector<Row>&, StoredTransportSymbolicPlan<Pack>*);
template Teuchos::RCP<Pack::matrix_type> finish_stored_transport_matrix<Pack, EpochMesh>(const EpochMesh&,
    Teuchos::RCP<Pack::matrix_type>, size_t, const std::vector<Row>&, StoredTransportSymbolicPlan<Pack>*);
} // namespace SimpleFluid::FVM::detail
