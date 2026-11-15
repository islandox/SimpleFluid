/** @file CollectiveConstruction.hh
 * @brief Agree local allocation before constructors which enter collectives.
 */
#pragma once
#include <Teuchos_CommHelpers.hpp>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

namespace SimpleFluid::detail
{
/**
 * @brief Allocate shared storage collectively, then construct and publish T.
 *
 * T remains responsible for agreeing its own local preparation between its
 * internal collectives. This helper closes the allocation-before-constructor
 * gap in make_shared; it cannot repair a rank-divergent third-party collective.
 */
template<class T, class... Args>
std::shared_ptr<T> collective_make_shared(const Teuchos::Comm<int>& comm, Args&&... args)
{
    struct Storage
    {
        alignas(T) unsigned char bytes[sizeof(T)];
        bool constructed = false;
        T* get() noexcept { return std::launder(reinterpret_cast<T*>(bytes)); }
        ~Storage()
        {
            if (constructed)
                std::destroy_at(get());
        }
    };
    const auto agree = [&](const std::exception_ptr& error, const char* message)
    {
        const int local = error ? 1 : 0;
        int failed = 0;
        Teuchos::reduceAll(comm, Teuchos::REDUCE_MAX, 1, &local, &failed);
        if (failed)
        {
            if (error)
                std::rethrow_exception(error);
            throw std::runtime_error(message);
        }
    };
    std::shared_ptr<Storage> storage;
    std::exception_ptr error;
    try
    {
        storage = std::make_shared<Storage>();
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(error, "Collective object storage allocation failed on another rank.");
    error = {};
    try
    {
        std::construct_at(reinterpret_cast<T*>(storage->bytes), std::forward<Args>(args)...);
        storage->constructed = true;
    }
    catch (...)
    {
        error = std::current_exception();
    }
    agree(error, "Collective object construction failed on another rank.");
    return std::shared_ptr<T>(storage, storage->get());
}
} // namespace SimpleFluid::detail
