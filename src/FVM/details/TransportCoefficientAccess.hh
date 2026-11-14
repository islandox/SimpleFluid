/** @file FVM/details/TransportCoefficientAccess.hh
 * @brief Direct and product coefficient views shared by scalar transport.
 */
#pragma once

#include "fields/FieldStored.hh"

#include <utility>
#include <variant>

namespace SimpleFluid::FVM::detail
{

/** Direct overlap coefficient access without per-read expression selection. */
template<TpetraTypePack Pack> class TransportCoefficientView
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using view_type =
        decltype(std::declval<const typename Pack::vector_type&>().getLocalViewHost(Tpetra::Access::ReadOnly));

    explicit TransportCoefficientView(view_type values) : d_values(std::move(values)) {}

    scalar_type operator()(local_ordinal_type cell) const { return d_values(cell, 0); }

private:
    view_type d_values;
};

/** Product access preserves factor order without a temporary coefficient field. */
template<TpetraTypePack Pack> class TransportProductCoefficientView
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using view_type = typename TransportCoefficientView<Pack>::view_type;

    TransportProductCoefficientView(view_type values, view_type factor)
        : d_values(std::move(values)), d_factor(std::move(factor))
    {
    }

    scalar_type operator()(local_ordinal_type cell) const
    {
        const auto value = d_values(cell, 0);
        return value * d_factor(cell, 0);
    }

private:
    view_type d_values;
    view_type d_factor;
};

/** Weighted transport keeps independent direct storage and advection channels. */
template<TpetraTypePack Pack> struct WeightedScalarTransportWeights
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    TransportCoefficientView<Pack> storage;
    TransportCoefficientView<Pack> old_storage;
    TransportCoefficientView<Pack> advection;

    scalar_type cell_advection(local_ordinal_type cell, scalar_type) const { return advection(cell); }
};

/** Physical temperature uses capacity for storage and advection. */
template<TpetraTypePack Pack> struct PhysicalTemperatureTransportWeights
{
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;

    TransportProductCoefficientView<Pack> storage;
    TransportProductCoefficientView<Pack> old_storage;

    scalar_type advection(local_ordinal_type cell) const { return storage(cell); }
    scalar_type cell_advection(local_ordinal_type, scalar_type current_storage) const { return current_storage; }
};

/** Only coefficient validation and numeric rows dispatch on the two weight
 * policies. Direct diffusion and the rest of assembly remain shared.
 */
template<TpetraTypePack Pack> struct ScalarTransportCoefficientAccess
{
    std::variant<WeightedScalarTransportWeights<Pack>, PhysicalTemperatureTransportWeights<Pack>> weights;
    TransportCoefficientView<Pack> diffusion;
};

} // namespace SimpleFluid::FVM::detail
