/**
 * @file ScalarVoidFractionModel.hh
 * @brief Low-order bounded scalar void-fraction update.
 */
#pragma once

#include "equations/BoussinesqModel.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace SimpleFluid
{

/**
 * @brief Runtime controls for the bounded scalar void-fraction model.
 */
struct ScalarVoidFractionOptions
{
    real_t alpha_min = 0.0;
    real_t alpha_max = 0.95;
    real_t initial_alpha = 0.0;
    real_t alpha_collapse_time =
        std::numeric_limits<real_t>::infinity();
    real_t alpha_diffusivity = 0.0;
    real_t constant_slip_velocity = 0.0;
};

/**
 * @brief Validate scalar void-fraction bounds and transport parameters.
 *
 * @param options Candidate scalar-void configuration.
 */
inline void validate_scalar_void_fraction_options(
    const ScalarVoidFractionOptions& options)
{
    if (!std::isfinite(options.alpha_min)
        || !std::isfinite(options.alpha_max)
        || options.alpha_min < 0.0
        || options.alpha_max <= options.alpha_min
        || options.alpha_max >= 1.0)
    {
        throw std::invalid_argument(
            "Scalar void bounds require 0 <= alpha_min < alpha_max < 1.");
    }
    if (!std::isfinite(options.initial_alpha)
        || options.initial_alpha < options.alpha_min
        || options.initial_alpha > options.alpha_max)
    {
        throw std::invalid_argument(
            "Initial scalar void fraction must be finite and inside bounds.");
    }
    if (std::isnan(options.alpha_collapse_time)
        || options.alpha_collapse_time <= 0.0)
    {
        throw std::invalid_argument(
            "Alpha collapse time must be positive or infinite.");
    }
    if (!std::isfinite(options.alpha_diffusivity)
        || options.alpha_diffusivity < 0.0)
    {
        throw std::invalid_argument(
            "Alpha diffusivity must be finite and non-negative.");
    }
    if (!std::isfinite(options.constant_slip_velocity))
    {
        throw std::invalid_argument(
            "Scalar void slip velocity must be finite.");
    }
}

/**
 * @brief Parse scalar void-fraction options from a flat database.
 *
 * @param database Database containing optional scalar-void keys.
 * @return Validated scalar void-fraction options.
 */
inline ScalarVoidFractionOptions scalar_void_fraction_options_from_database(
    const Database& database)
{
    ScalarVoidFractionOptions options;
    options.alpha_min = detail::database_value_or<real_t>(
        database, "alpha_min", options.alpha_min);
    options.alpha_max = detail::database_value_or<real_t>(
        database, "alpha_max", options.alpha_max);
    options.initial_alpha = detail::database_value_or<real_t>(
        database, "initial_alpha_g", options.initial_alpha);
    options.alpha_collapse_time = detail::database_value_or<real_t>(
        database,
        "alpha_collapse_time",
        options.alpha_collapse_time);
    options.alpha_diffusivity = detail::database_value_or<real_t>(
        database, "alpha_diffusivity", options.alpha_diffusivity);
    options.constant_slip_velocity = detail::database_value_or<real_t>(
        database,
        "constant_slip_velocity",
        options.constant_slip_velocity);
    validate_scalar_void_fraction_options(options);
    return options;
}

/**
 * @brief Low-order bounded alpha-g/alpha-l model for source aggregation.
 *
 * @tparam Pack Tpetra type pack used for mesh and field storage.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class ScalarVoidFractionModel
{
public:
    using scalar_type = typename Pack::scalar_type;
    using local_ordinal_type = typename Pack::local_ordinal_type;
    using mesh_type = Mesh<Pack>;
    using field_type = CellField<Pack>;

    /**
     * @brief Construct a scalar void-fraction model on a mesh.
     */
    ScalarVoidFractionModel(
        SP<const mesh_type> mesh,
        ScalarVoidFractionOptions options = {})
        : d_mesh(std::move(mesh)),
          d_options(std::move(options)),
          d_alpha_g(d_mesh, "alpha_g"),
          d_alpha_l(d_mesh, "alpha_l"),
          d_source_alpha_total(d_mesh, "S_alpha_total")
    {
        if (!d_mesh)
        {
            throw std::invalid_argument(
                "ScalarVoidFractionModel requires a non-null mesh.");
        }
        configure(d_options);
    }

    /**
     * @brief Replace model options and reset alpha/source fields.
     */
    void configure(const ScalarVoidFractionOptions& options)
    {
        validate_scalar_void_fraction_options(options);
        d_options = options;
        d_alpha_g.put_scalar(d_options.initial_alpha);
        d_alpha_l.put_scalar(1.0 - d_options.initial_alpha);
        d_source_alpha_total.put_scalar(0.0);
        register_output_fields();
    }

    /**
     * @brief Return the active scalar void-fraction options.
     */
    const ScalarVoidFractionOptions& options() const noexcept
    {
        return d_options;
    }

    /**
     * @brief Gas void-fraction field.
     */
    const field_type& alpha_g() const noexcept { return d_alpha_g; }
    /**
     * @brief Liquid fraction field equal to one minus gas alpha after updates.
     */
    const field_type& alpha_l() const noexcept { return d_alpha_l; }
    /**
     * @brief Realized total alpha source after bounds are applied.
     */
    const field_type& source_alpha_total() const noexcept
    {
        return d_source_alpha_total;
    }

    /**
     * @brief Return collapse realizable without crossing alpha_min this step.
     */
    scalar_type bounded_collapse_rate(
        local_ordinal_type cell_lid,
        scalar_type time_step) const
    {
        if (!std::isfinite(time_step) || time_step <= scalar_type{})
        {
            throw std::invalid_argument(
                "Scalar void collapse requires a positive finite timestep.");
        }
        if (!std::isfinite(d_options.alpha_collapse_time))
        {
            return scalar_type{};
        }
        const auto alpha = d_alpha_g.value(cell_lid);
        const auto available_alpha = std::max(
            alpha - d_options.alpha_min, scalar_type{});
        return std::min(
            alpha / d_options.alpha_collapse_time,
            available_alpha / time_step);
    }

    /**
     * @brief Fields that can be published to solution output.
     */
    const std::map<std::string, const field_type*>& output_fields() const
        noexcept
    {
        return d_output_fields;
    }

    /**
     * @brief Apply explicit radiolysis and boiling alpha sources.
     */
    void update_explicit(
        scalar_type time_step,
        const field_type* source_alpha_rad,
        const field_type* source_alpha_boil)
    {
        if (!std::isfinite(time_step) || time_step <= 0.0)
        {
            throw std::invalid_argument(
                "Scalar void update requires a positive finite time step.");
        }
        check_source(source_alpha_rad, "radiolytic alpha source");
        check_source(source_alpha_boil, "boiling alpha source");

        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto old_alpha = d_alpha_g.value(cell_lid);
            scalar_type raw_rate{};
            if (source_alpha_rad)
                raw_rate += source_alpha_rad->value(cell_lid);
            if (source_alpha_boil)
                raw_rate += source_alpha_boil->value(cell_lid);
            raw_rate -= bounded_collapse_rate(cell_lid, time_step);
            const auto new_alpha = std::clamp(
                old_alpha + time_step * raw_rate,
                d_options.alpha_min,
                d_options.alpha_max);
            d_alpha_g.set_owned_value(cell_lid, new_alpha);
            d_alpha_l.set_owned_value(cell_lid, 1.0 - new_alpha);
            d_source_alpha_total.set_owned_value(
                cell_lid, (new_alpha - old_alpha) / time_step);
        }
        sync_fields();
    }

    /**
     * @brief Mirror externally supplied alpha/source fields into this model.
     */
    void mirror(
        const field_type& alpha_g,
        const field_type& alpha_l,
        const field_type& source_alpha_total)
    {
        check_source(&alpha_g, "alpha_g");
        check_source(&alpha_l, "alpha_l");
        check_source(&source_alpha_total, "S_alpha_total");
        for (size_t owned = 0; owned < d_mesh->num_owned_cells(); ++owned)
        {
            const auto cell_lid =
                static_cast<local_ordinal_type>(owned);
            const auto alpha = std::clamp(
                alpha_g.value(cell_lid),
                d_options.alpha_min,
                d_options.alpha_max);
            d_alpha_g.set_owned_value(cell_lid, alpha);
            d_alpha_l.set_owned_value(cell_lid, alpha_l.value(cell_lid));
            d_source_alpha_total.set_owned_value(
                cell_lid, source_alpha_total.value(cell_lid));
        }
        sync_fields();
    }

private:
    void register_output_fields()
    {
        d_output_fields = {
            {"alpha_g", &d_alpha_g},
            {"alpha_l", &d_alpha_l},
            {"S_alpha_total", &d_source_alpha_total}};
    }

    void check_source(
        const field_type* source,
        const std::string& label) const
    {
        if (source != nullptr && &source->mesh() != d_mesh.get())
        {
            throw std::invalid_argument(
                "ScalarVoidFractionModel received a " + label
                + " on the wrong mesh.");
        }
    }

    void sync_fields()
    {
        d_alpha_g.sync_ghosts();
        d_alpha_l.sync_ghosts();
        d_source_alpha_total.sync_ghosts();
    }

    SP<const mesh_type> d_mesh;
    ScalarVoidFractionOptions d_options;
    field_type d_alpha_g;
    field_type d_alpha_l;
    field_type d_source_alpha_total;
    std::map<std::string, const field_type*> d_output_fields;
};

} // namespace SimpleFluid
