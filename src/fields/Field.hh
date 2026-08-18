/**
 * @file Field.hh
 * @author islandox(59904740+islandox@users.noreply.github.com)
 * @brief Lightweight typed field descriptors.
 * @version 0.1
 * @date 2026-07-21
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include "dataclass/TpetraTypes.hh"
#include "dataclass/vec3.hh"

#include <array>
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace SimpleFluid
{

/** @brief Location tag for cell-centered fields. */
struct CellLocation
{
};

/** @brief Location tag for face-centered fields. */
struct FaceLocation
{
};

/** @brief Location tag for values stored only on boundary faces. */
struct BoundaryFaceLocation
{
};

/**
 * @brief Named, uniquely identified descriptor for a typed field.
 *
 * A descriptor carries no numerical storage. FieldStored combines it with a
 * mesh and distributed Tpetra storage.
 *
 * @tparam Value Value stored at each mesh location.
 * @tparam Location CellLocation, FaceLocation, or BoundaryFaceLocation.
 */
template<class Value, class Location>
class Field
{
public:
    using value_type = Value;
    using location_type = Location;

    /**
     * @brief Construct a field descriptor with a process-local unique ID.
     * @param name Non-empty name used by problem registries and equations.
     * @throws std::invalid_argument If @p name is empty.
     */
    explicit Field(std::string name)
        : d_id(next_id()),
          d_name(std::move(name))
    {
        if (d_name.empty())
        {
            throw std::invalid_argument(
                "Field requires a non-empty name.");
        }
    }

    size_t id() const noexcept { return d_id; }
    const std::string& name() const noexcept { return d_name; }

    friend bool operator==(const Field&, const Field&) = default;

private:
    static size_t next_id() noexcept
    {
        static std::atomic_size_t id{0};
        return id.fetch_add(1, std::memory_order_relaxed);
    }

    size_t d_id;
    std::string d_name;
};

template<TpetraTypePack Pack = DefaultTpetraTypes>
using ScalarCellFieldDescriptor =
    Field<typename Pack::scalar_type, CellLocation>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using VectorCellFieldDescriptor =
    Field<vec3<typename Pack::scalar_type>, CellLocation>;

/** @brief Row-major 3x3 tensor descriptor stored at cell centers. */
template<TpetraTypePack Pack = DefaultTpetraTypes>
using TensorCellFieldDescriptor =
    Field<std::array<vec3<typename Pack::scalar_type>, 3>, CellLocation>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using ScalarFaceFieldDescriptor =
    Field<typename Pack::scalar_type, FaceLocation>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using VectorFaceFieldDescriptor =
    Field<vec3<typename Pack::scalar_type>, FaceLocation>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using ScalarBoundaryFaceFieldDescriptor =
    Field<typename Pack::scalar_type, BoundaryFaceLocation>;

template<TpetraTypePack Pack = DefaultTpetraTypes>
using VectorBoundaryFaceFieldDescriptor =
    Field<vec3<typename Pack::scalar_type>, BoundaryFaceLocation>;

} // namespace SimpleFluid
