/** @file FissionPowerSource.tcc @brief Compiled template implementations. */
#pragma once

#include "equations/FissionPowerSource.hh"

namespace SimpleFluid
{

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::configure(const FissionPowerSourceOptions& options)
{
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission power configuration",
        [&]
        {
            detail::validate_fission_power_options(options);
        });
    require_uniform_configuration(options);
    switch (options.profile)
    {
        case FissionPowerProfile::Disabled:
            initialize_constant(0.0);
            break;
        case FissionPowerProfile::Constant:
            initialize_constant(options.power_density);
            break;
        case FissionPowerProfile::Gaussian:
            initialize_gaussian(
                options.total_power,
                options.center,
                options.standard_deviation);
            break;
    }
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::initialize_constant(scalar_type power_density)
{
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission constant-power initialization",
        [&]
        {
            require_non_negative(
                power_density, "power density");
        });
    require_uniform_value(power_density, "power density");
    d_base_profile.put_scalar(power_density);
    apply_base_profile(scalar_type{1});
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::initialize_gaussian(
    scalar_type total_power,
    const vec3<scalar_type>& center,
    const vec3<scalar_type>& standard_deviation)
{
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission Gaussian configuration",
        [&]
        {
            require_non_negative(total_power, "total power");
            for (size_t component = 0;
                 component < 3;
                 ++component)
            {
                if (!std::isfinite(center.component(component)))
                {
                    throw std::invalid_argument(
                        "Fission Gaussian center must be finite.");
                }
                const auto width =
                    standard_deviation.component(component);
                if (!std::isfinite(width)
                    || width <= scalar_type{})
                {
                    throw std::invalid_argument(
                        "Fission Gaussian standard deviations must be "
                        "finite and positive.");
                }
            }
        });
    require_uniform_value(total_power, "total power");
    for (size_t component = 0; component < 3; ++component)
    {
        require_uniform_value(
            center.component(component), "Gaussian center");
        require_uniform_value(
            standard_deviation.component(component),
            "Gaussian standard deviation");
    }

    scalar_type local_integral{};
    for (size_t owned = 0;
         owned < d_mesh->num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        const auto centroid = d_mesh->cell_centroid(cell_lid);
        scalar_type exponent{};
        for (size_t component = 0; component < 3; ++component)
        {
            const auto scaled =
                (centroid.component(component)
                 - center.component(component))
              / standard_deviation.component(component);
            exponent += scaled * scaled;
        }
        local_integral +=
            std::exp(-scalar_type{0.5} * exponent)
          * d_mesh->cell_volume(cell_lid);
    }
    const auto profile_integral = global_sum(local_integral);
    scalar_type scale{};
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission Gaussian normalization",
        [&]
        {
            if (total_power > scalar_type{}
                && (!std::isfinite(profile_integral)
                    || profile_integral <= scalar_type{}))
            {
                throw std::invalid_argument(
                    "Positive fission total power requires a positive "
                    "profile integral.");
            }
            scale = total_power == scalar_type{}
                        ? scalar_type{}
                        : checked_scale(
                              total_power, profile_integral);
        });

    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission Gaussian field initialization",
        [&]
        {
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto centroid =
                    d_mesh->cell_centroid(cell_lid);
                scalar_type exponent{};
                for (size_t component = 0;
                     component < 3;
                     ++component)
                {
                    const auto scaled =
                        (centroid.component(component)
                         - center.component(component))
                      / standard_deviation.component(component);
                    exponent += scaled * scaled;
                }
                const auto value =
                    scale
                  * std::exp(
                        -scalar_type{0.5} * exponent);
                require_non_negative(
                    value, "Gaussian field value");
                d_base_profile.set_owned_value(
                    cell_lid, value);
            }
        });
    d_base_profile.sync_ghosts();
    apply_base_profile(scalar_type{1});
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::initialize_from_shape(
    const field_type& shape,
    scalar_type total_power)
{
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission shape initialization",
        [&]
        {
            require_same_mesh(shape);
            require_non_negative(total_power, "total power");
            validate_non_negative_field(shape);
        });
    require_uniform_value(total_power, "total power");

    const auto profile_integral = integrate(shape);
    scalar_type scale{};
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission shape normalization",
        [&]
        {
            if (total_power > scalar_type{}
                && (!std::isfinite(profile_integral)
                    || profile_integral <= scalar_type{}))
            {
                throw std::invalid_argument(
                    "Positive fission total power requires a positive "
                    "profile integral.");
            }
            scale = total_power == scalar_type{}
                        ? scalar_type{}
                        : checked_scale(
                              total_power, profile_integral);
        });
    copy_scaled_field(shape, scale);
    apply_base_profile(scalar_type{1});
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::set_time_multiplier(multiplier_type multiplier)
{
    if (!multiplier)
    {
        throw std::invalid_argument(
            "Fission power time multiplier must be callable.");
    }
    d_source->set_updater(
        [this](const context_type& context, field_type& field)
        {
            apply_time_multiplier(context, field);
        });
    d_time_multiplier = std::move(multiplier);
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::require_uniform_value(
    scalar_type local_value,
    const std::string& label) const
{
    scalar_type minimum_value{};
    scalar_type maximum_value{};
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        1,
        &local_value,
        &minimum_value);
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_value,
        &maximum_value);
    if (minimum_value != maximum_value)
    {
        throw std::invalid_argument(
            "Fission " + label + " must agree on every rank.");
    }
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::require_uniform_configuration(
    const FissionPowerSourceOptions& options) const
{
    const auto communicator = d_mesh->owned_cell_map()->getComm();
    const int local_profile = static_cast<int>(options.profile);
    int minimum_profile = 0;
    int maximum_profile = 0;
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MIN,
        1,
        &local_profile,
        &minimum_profile);
    Teuchos::reduceAll(
        *communicator,
        Teuchos::REDUCE_MAX,
        1,
        &local_profile,
        &maximum_profile);
    if (minimum_profile != maximum_profile)
    {
        throw std::invalid_argument(
            "Fission power profile must agree on every rank.");
    }

    switch (options.profile)
    {
        case FissionPowerProfile::Disabled:
            return;
        case FissionPowerProfile::Constant:
            require_uniform_value(
                options.power_density, "power density");
            return;
        case FissionPowerProfile::Gaussian:
            require_uniform_value(
                options.total_power, "total power");
            for (size_t component = 0; component < 3; ++component)
            {
                require_uniform_value(
                    options.center.component(component),
                    "Gaussian center");
                require_uniform_value(
                    options.standard_deviation.component(component),
                    "Gaussian standard deviation");
            }
            return;
    }
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::copy_scaled_field(
    const field_type& field,
    scalar_type scale)
{
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission field copy",
        [&]
        {
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto value =
                    field.value(cell_lid) * scale;
                require_non_negative(
                    value, "normalized field value");
                d_base_profile.set_owned_value(
                    cell_lid, value);
            }
        });
    d_base_profile.sync_ghosts();
}

template<TpetraTypePack Pack, class MeshType>
auto FissionPowerSource<Pack, MeshType>::integrate(const field_type& field) const -> scalar_type
{
    scalar_type local_power{};
    for (size_t owned = 0;
         owned < d_mesh->num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        local_power +=
            field.value(cell_lid)
          * d_mesh->cell_volume(cell_lid);
    }

    return global_sum(local_power);
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::apply_base_profile(scalar_type multiplier)
{
    auto& applied = d_source->field();
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission applied-field update",
        [&]
        {
            for (size_t owned = 0;
                 owned < d_mesh->num_owned_cells();
                 ++owned)
            {
                const auto cell_lid =
                    static_cast<local_ordinal_type>(owned);
                const auto value =
                    d_base_profile.value(cell_lid) * multiplier;
                require_non_negative(
                    value, "applied field value");
                applied.set_owned_value(cell_lid, value);
            }
        });
    applied.sync_ghosts();
}

template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::apply_time_multiplier(
    const context_type& context,
    field_type& field)
{
    scalar_type multiplier = scalar_type{1};
    if (d_time_multiplier)
    {
        multiplier = d_time_multiplier(context);
    }
    require_non_negative(multiplier, "time multiplier");
    for (size_t owned = 0;
         owned < d_mesh->num_owned_cells();
         ++owned)
    {
        const auto cell_lid =
            static_cast<local_ordinal_type>(owned);
        field.set_owned_value(
            cell_lid,
            d_base_profile.value(cell_lid) * multiplier);
    }
}


template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::initialize_from_power_density(const field_type& power_density)
{
    collective_detail::collective_local_validation(
        *d_mesh,
        "Fission power-density initialization",
        [&] { require_same_mesh(power_density); });
    copy_non_negative_field(power_density);
    apply_base_profile(scalar_type{1});
}


template<TpetraTypePack Pack, class MeshType>
auto FissionPowerSource<Pack, MeshType>::integrated_power() const -> scalar_type
{
    return integrate(d_source->field());
}


template<TpetraTypePack Pack, class MeshType>
void FissionPowerSource<Pack, MeshType>::copy_non_negative_field(const field_type& field)
{
    copy_scaled_field(field, scalar_type{1});
}

} // namespace SimpleFluid
