/**
 * @file Problem.hh
 * @brief Lifetime owner and typed registry for a finite-volume problem.
 */

#pragma once

#include "equations/BoundaryConditions.hh"
#include "equations/TimeStepperOptions.hh"
#include "fields/FieldStored.hh"
#include "geometry/MeshHandle.hh"
#include "solvers/BelosLinearSolver.hh"

#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>

namespace SimpleFluid
{

/**
 * @brief Lifetime owner and type-safe named registry for a simulation.
 *
 * A problem keeps the mesh, physical options, fields, equations, and assembled
 * systems alive under unique names. Retrieval verifies the exact registered
 * C++ type before returning an object.
 */
template<TpetraTypePack Pack = DefaultTpetraTypes>
class Problem
{
public:
    /**
     * @brief Construct a problem around a shared mesh and solver options.
     * @throws std::invalid_argument If @p mesh is null.
     */
    explicit Problem(
        SP<const MeshHandle<Pack>> mesh,
        BoundaryConditionSet boundary_conditions = {},
        TimeStepperOptions time_options = {},
        LinearSolverOptions linear_options = {})
        : d_mesh(require_mesh(std::move(mesh))),
          d_boundary_conditions(std::move(boundary_conditions)),
          d_time_options(time_options),
          d_linear_options(linear_options)
    {
    }

    const MeshHandle<Pack>& mesh() const noexcept { return *d_mesh; }
    SP<const MeshHandle<Pack>> mesh_ptr() const noexcept { return d_mesh; }

    BoundaryConditionSet& boundary_conditions() noexcept
    {
        return d_boundary_conditions;
    }

    const BoundaryConditionSet& boundary_conditions() const noexcept
    {
        return d_boundary_conditions;
    }

    TimeStepperOptions& time_options() noexcept { return d_time_options; }
    const TimeStepperOptions& time_options() const noexcept
    {
        return d_time_options;
    }

    LinearSolverOptions& linear_options() noexcept
    {
        return d_linear_options;
    }

    const LinearSolverOptions& linear_options() const noexcept
    {
        return d_linear_options;
    }

    bool contains(const std::string& name) const noexcept
    {
        return d_objects.contains(name);
    }

    /**
     * @brief Construct and register an arbitrary typed object.
     * @return Reference valid for the lifetime of the registry entry.
     * @throws std::invalid_argument If the name is empty or already used.
     */
    template<class Object, class... Args>
    Object& emplace_object(const std::string& name, Args&&... args)
    {
        if (name.empty())
        {
            throw std::invalid_argument(
                "Problem object name cannot be empty.");
        }
        if (contains(name))
        {
            throw std::invalid_argument(
                "Problem already contains an object named '" + name + "'.");
        }

        auto object = std::make_shared<Object>(
            std::forward<Args>(args)...);
        auto* result = object.get();
        d_objects.emplace(
            name,
            ObjectEntry{
                std::type_index(typeid(Object)),
                std::move(object)});
        return *result;
    }

    template<class Object>
    Object& object(const std::string& name)
    {
        return const_cast<Object&>(
            std::as_const(*this).template object<Object>(name));
    }

    /**
     * @brief Retrieve a registered object by exact type and name.
     * @throws std::out_of_range If no object has the requested name.
     * @throws std::invalid_argument If the registered type differs.
     */
    template<class Object>
    const Object& object(const std::string& name) const
    {
        const auto iter = d_objects.find(name);
        if (iter == d_objects.end())
        {
            throw std::out_of_range(
                "Problem does not contain object '" + name + "'.");
        }
        if (iter->second.type != std::type_index(typeid(Object)))
        {
            throw std::invalid_argument(
                "Problem object '" + name + "' has the wrong type.");
        }
        return *static_cast<const Object*>(iter->second.object.get());
    }

    /** @brief Allocate and register a distributed field. */
    template<class Descriptor>
    FieldStored<Descriptor, Pack>& add_field(
        Descriptor descriptor,
        bool zero_out = true)
    {
        const auto name = descriptor.name();
        return add_field_impl(
            name,
            std::make_shared<FieldStored<Descriptor, Pack>>(
                std::move(descriptor), d_mesh, zero_out));
    }

    template<class Descriptor>
    FieldStored<Descriptor, Pack>& add_field(
        Descriptor descriptor,
        const typename Descriptor::value_type& initial_value)
    {
        const auto name = descriptor.name();
        return add_field_impl(
            name,
            std::make_shared<FieldStored<Descriptor, Pack>>(
                std::move(descriptor), d_mesh, initial_value));
    }

    /** @brief Retrieve a field and verify that it belongs to this mesh. */
    template<class StoredField>
    StoredField& field(const std::string& name)
    {
        auto& result = object<StoredField>(name);
        if (result.mesh_ptr().get() != d_mesh.get())
        {
            throw std::invalid_argument(
                "Problem field mesh mismatch.");
        }
        return result;
    }

    /** @brief Register an already constructed equation under a unique name. */
    template<class EquationType>
    EquationType& add_equation(const std::string& name,
                               EquationType equation)
    {
        return emplace_object<EquationType>(
            name, std::move(equation));
    }

    /** @brief Assemble a registered equation and retain the resulting system. */
    template<class EquationType>
    auto& assemble(const std::string& equation_name,
                   const std::string& assembled_name)
    {
        auto assembled =
            object<EquationType>(equation_name).assemble(d_linear_options);
        using AssembledType = decltype(assembled);
        return emplace_object<AssembledType>(
            assembled_name, std::move(assembled));
    }

    /**
     * @brief Solve a registered assembled system.
     * @return Whether its solver reports convergence.
     */
    template<class AssembledType>
    bool solve(const std::string& assembled_name)
    {
        return object<AssembledType>(assembled_name).solve();
    }

    const std::unordered_map<std::string, AnyFieldStored<Pack>>&
    fields() const noexcept
    {
        return d_fields;
    }

private:
    struct ObjectEntry
    {
        std::type_index type;
        std::shared_ptr<void> object;
    };

    static SP<const MeshHandle<Pack>> require_mesh(
        SP<const MeshHandle<Pack>> mesh)
    {
        if (!mesh)
        {
            throw std::invalid_argument(
                "Problem requires a non-null mesh.");
        }
        return mesh;
    }

    template<class StoredField>
    StoredField& add_field_impl(
        const std::string& name,
        SP<StoredField> field)
    {
        if (contains(name))
        {
            throw std::invalid_argument(
                "Problem already contains an object named '" + name + "'.");
        }
        static_assert(
            std::is_constructible_v<AnyFieldStored<Pack>, SP<StoredField>>,
            "Problem field type is not part of AnyFieldStored.");

        auto* result = field.get();
        d_fields.emplace(name, AnyFieldStored<Pack>{field});
        d_objects.emplace(
            name,
            ObjectEntry{
                std::type_index(typeid(StoredField)),
                std::move(field)});
        return *result;
    }

    SP<const MeshHandle<Pack>> d_mesh;
    BoundaryConditionSet d_boundary_conditions;
    TimeStepperOptions d_time_options;
    LinearSolverOptions d_linear_options;
    std::unordered_map<std::string, ObjectEntry> d_objects;
    std::unordered_map<std::string, AnyFieldStored<Pack>> d_fields;
};

} // namespace SimpleFluid
