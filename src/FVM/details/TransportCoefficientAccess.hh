/** @file FVM/details/TransportCoefficientAccess.hh
 * @brief Direct coefficient views shared by weighted and physical scalar transport.
 */
#pragma once

#include "fields/FieldStored.hh"

#include <utility>

namespace SimpleFluid::FVM::detail
{

/** An overlap coefficient is either a direct field value or a two-field product.
 * Selection is stable for an assembly and uses inline control flow. No virtual
 * call, function pointer, or materialized coefficient field is needed per face.
 */
template<TpetraTypePack Pack> class TransportCoefficientView
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using view_type =
        decltype(std::declval<const typename Pack::vector_type&>().getLocalViewHost(Tpetra::Access::ReadOnly));

    explicit TransportCoefficientView(view_type values) : d_values(std::move(values)) {}
    TransportCoefficientView(view_type values, view_type factor)
        : d_values(std::move(values)), d_factor(std::move(factor)), d_product(true)
    {
    }

    scalar_type operator()(local_ordinal_type cell) const
    {
        const auto value = d_values(cell, 0);
        return d_product ? value * d_factor(cell, 0) : value;
    }

private:
    view_type d_values;
    view_type d_factor;
    bool d_product = false;
};

/** A concrete interface keeps coefficient expression types out of the kernel's
 * template arguments. Accepted-old storage remains an independent channel.
 */
template<TpetraTypePack Pack> struct ScalarTransportCoefficientAccess
{
    TransportCoefficientView<Pack> storage;
    TransportCoefficientView<Pack> old_storage;
    TransportCoefficientView<Pack> advection;
    TransportCoefficientView<Pack> diffusion;
};

} // namespace SimpleFluid::FVM::detail
