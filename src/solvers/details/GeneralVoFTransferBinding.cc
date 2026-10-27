/** @file GeneralVoFTransferBinding.cc
 * @brief Sparse owned-row verification of raw geometric-donor face coefficients.
 */
#include "solvers/details/GeneralVoFTransferBinding.hh"
#include "geometry/swept_volume/SweptGeometry.hh"
#include "parallel/SparseOwnerExchange.hh"
#include "solvers/details/ConservativeVoFExactArithmetic.hh"
#include "solvers/details/GeneralVoFExactStorage.hh"
#include "utils/CompensatedSum.hh"
#include <Teuchos_CommHelpers.hpp>
#include <bit>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
namespace SimpleFluid::detail
{
namespace
{
using Pack = DefaultTpetraTypes;
using GO = Pack::global_ordinal_type;
using LO = Pack::local_ordinal_type;
void word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i)
        bytes.push_back(char(value >> (8 * i)));
}
std::uint64_t word(const std::vector<char>& bytes, std::size_t offset)
{
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= std::uint64_t(static_cast<unsigned char>(bytes[offset + i])) << (8 * i);
    return value;
}
void require(const Teuchos::Comm<int>& comm, bool valid, const char* message, std::exception_ptr local_error = {})
{
    int local = valid, global = 0;
    Teuchos::reduceAll(comm, Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
    {
        if (local_error)
        {
            try
            {
                std::rethrow_exception(local_error);
            }
            catch (const std::exception& error)
            {
                throw std::runtime_error(std::string(message) + " Local cause: " + error.what());
            }
            catch (...)
            {
                throw std::runtime_error(std::string(message) + " Local cause: nonstandard exception.");
            }
        }
        throw std::runtime_error(message);
    }
}
} // namespace
std::vector<GeneralVoFOriginalRow> verify_general_vof_transfer_binding(const SP<const MeshHandle<>>& mesh,
    const std::vector<double>& outgoing, const std::vector<std::map<GO, double>>& incoming,
    const std::vector<ConservativeVoFTransport::FaceContribution>& contributions, std::size_t byte_budget,
    const std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>* exact_outgoing,
    const std::vector<std::map<GO, SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>>*
        exact_incoming,
    bool exact_only)
{
    const auto map = mesh->owned_cell_map();
    const auto comm = map->getComm();
    require(*comm,
        bool(exact_outgoing) == bool(exact_incoming) &&
            (!exact_outgoing || (exact_outgoing->size() == mesh->num_owned_cells() &&
                                    exact_incoming->size() == mesh->num_owned_cells())),
        "General VoF exact route dimensions differ.");
    using Domain = GeneralVoFStorageDomain;
    GeneralVoFExactStorage storage(byte_budget);
    const auto operation_budget = [&]
    {
        // ONE synchronous operation includes its returned result. Internal
        // arithmetic temporaries die before immediate ledger admission, so the
        // transfer is S+W <= B, not S+W+result. Keep a conservative1024 bytes
        // outsideW for one tracking-record insertion (currently144 bytes).
        constexpr std::size_t tracking_reserve = 1024;
        const auto remaining = storage.remaining();
        if (remaining <= tracking_reserve || remaining - tracking_reserve < 1024)
            throw std::runtime_error("General VoF binding has no bounded scalar operation/result reserve.");
        return remaining - tracking_reserve;
    };
    bool storage_valid = true;
    try
    {
        storage.capacity(Domain::Input, contributions.capacity(), sizeof(ConservativeVoFTransport::FaceContribution));
        storage.capacity(Domain::Scratch, contributions.size(),
            2 * (sizeof(GO) + 4 * sizeof(void*)) + 2 * (sizeof(std::pair<const GO, int>) + 4 * sizeof(void*)) +
                2 * (sizeof(GO) + sizeof(int)));
        for (const auto& face : contributions)
            if (face.exact_amount)
                storage.measure(Domain::Input, *face.exact_amount);
        if (exact_outgoing)
            for (const auto& value : *exact_outgoing)
                storage.measure(Domain::Input, value);
        if (exact_incoming)
            for (const auto& row : *exact_incoming)
                for (const auto& [donor, value] : row)
                    storage.measure(Domain::Input, value);
    }
    catch (...)
    {
        storage_valid = false;
    }
    require(*comm, storage_valid, "General VoF binding exact inputs exceed cumulative storage.");
    const auto packet_budget = storage.remaining() / 4;
    require(*comm, packet_budget > 0, "General VoF binding has no packet reserve.");
    int local_exact = bool(exact_outgoing), exact_mode = 0;
    for (const auto& face : contributions)
        local_exact = local_exact || bool(face.exact_amount);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_exact, &exact_mode);
    // Resolve only actual endpoint owners, never a global cell/rank catalog.
    std::set<GO> endpoint_set;
    {
        const auto lease = mesh->acquire_execution_view();
        for (const auto& face : contributions)
        {
            endpoint_set.insert(mesh->cell_global_id(mesh->owner_cell(face.face)));
            endpoint_set.insert(mesh->cell_global_id(mesh->neighbor_cell(face.face)));
        }
    }
    Teuchos::Array<GO> endpoints(endpoint_set.begin(), endpoint_set.end());
    Teuchos::Array<int> owners(endpoints.size(), -1);
    (void) map->getRemoteIndexList(endpoints(), owners());
    bool valid = true;
    std::map<GO, int> owner;
    for (std::size_t i = 0; i < static_cast<std::size_t>(endpoints.size()); ++i)
    {
        valid = valid && owners[i] >= 0;
        owner.emplace(endpoints[i], owners[i]);
    }
    require(*comm, valid, "General VoF raw face coefficient references an absent endpoint.");
    require(*comm, contributions.size() <= packet_budget / (2 * (sizeof(OwnerPacket) + 40)),
        "General VoF raw coefficient verification exceeds its packet budget.");
    std::vector<OwnerPacket> packets;
    packets.reserve(2 * contributions.size());
    try
    {
        std::size_t used = 0;
        const auto lease = mesh->acquire_execution_view();
        for (std::size_t index = 0; index < contributions.size(); ++index)
            for (int sign : {1, -1})
            {
                const auto& face = contributions[index];
                const auto row =
                    mesh->cell_global_id(sign == 1 ? mesh->owner_cell(face.face) : mesh->neighbor_cell(face.face));
                OwnerPacket packet;
                packet.peer = owner.at(row);
                word(packet.bytes, std::uint64_t(row));
                word(packet.bytes, std::uint64_t(face.donor));
                word(packet.bytes, std::uint64_t(mesh->face_global_id(face.face)));
                word(packet.bytes, std::bit_cast<std::uint64_t>(sign * face.amount));
                const bool self_reference = packet.peer == comm->getRank() && bool(face.exact_amount);
                word(packet.bytes, self_reference ? 3 : exact_only ? 2 : bool(face.exact_amount));
                if (self_reference)
                    word(packet.bytes, index);
                else if (face.exact_amount)
                {
                    const auto encoded = face.exact_amount->scaled(sign).encode(packet_budget);
                    word(packet.bytes, encoded.size());
                    packet.bytes.insert(packet.bytes.end(), encoded.begin(), encoded.end());
                }
                if (packet.bytes.size() > packet_budget - used ||
                    sizeof(OwnerPacket) > packet_budget - used - packet.bytes.size())
                    throw std::runtime_error("General VoF coefficient certificates exceed their byte budget.");
                used += sizeof(OwnerPacket) + packet.bytes.size();
                packets.push_back(std::move(packet));
            }
    }
    catch (...)
    {
        valid = false;
    }
    require(*comm, valid, "General VoF raw exact coefficient packet is invalid or exceeds its budget.");
    const auto received = exchange_owner_packets(packets, comm, packet_budget);
    struct Coefficient
    {
        CompensatedSum<> value, magnitude;
        SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> exact;
        std::size_t terms = 0;
    };
    std::vector<std::map<GO, Coefficient>> rows;
    std::vector<GeneralVoFOriginalRow> original;
    try
    {
        storage.capacity(Domain::Scratch, mesh->num_owned_cells(), sizeof(std::map<GO, Coefficient>));
        if (exact_mode)
            storage.capacity(Domain::Result, mesh->num_owned_cells(), sizeof(GeneralVoFOriginalRow));
        for (const auto* group : {static_cast<const std::vector<OwnerPacket>*>(&packets), &received.packets})
        {
            storage.capacity(Domain::Scratch, group->capacity(), sizeof(OwnerPacket));
            for (const auto& packet : *group)
                storage.capacity(Domain::Scratch, packet.bytes.capacity(), sizeof(char));
        }
        storage.capacity(Domain::Scratch, received.packets.size(),
            4 * sizeof(GO) + 3 * sizeof(int) + sizeof(std::pair<const GO, int>) + sizeof(GO) + 8 * sizeof(void*));
        rows.resize(mesh->num_owned_cells());
        if (exact_mode)
            original.resize(mesh->num_owned_cells());
    }
    catch (...)
    {
        valid = false;
    }
    require(*comm, valid, "General VoF binding row/packet storage exceeds its cumulative budget.");
    std::vector<GO> face_ids;
    std::vector<int> face_senders;
    std::exception_ptr recipient_error;
    try
    {
        face_ids.reserve(received.packets.size());
        face_senders.reserve(received.packets.size());
        for (const auto& packet : received.packets)
        {
            if (packet.bytes.size() < 40)
                throw std::runtime_error("Truncated general VoF raw coefficient packet.");
            const auto lid = map->getLocalElement(GO(word(packet.bytes, 0)));
            const auto donor = GO(word(packet.bytes, 8));
            const auto face_id = GO(word(packet.bytes, 16));
            const auto amount = std::bit_cast<double>(word(packet.bytes, 24));
            const auto has_exact = word(packet.bytes, 32);
            if (lid < 0 || !std::isfinite(amount) || has_exact > 3)
                throw std::runtime_error("Invalid general VoF raw coefficient recipient.");
            const auto face_lid = mesh->overlap_face_map()->getLocalElement(face_id);
            if (face_lid < 0)
                throw std::runtime_error("General VoF raw coefficient face is absent from its endpoint.");
            {
                const auto lease = mesh->acquire_execution_view();
                if (mesh->owner_cell(face_lid) != lid && mesh->neighbor_cell(face_lid) != lid)
                    throw std::runtime_error("General VoF raw face coefficient has a wrong endpoint.");
            }
            auto& row = rows[static_cast<std::size_t>(lid)];
            auto found = row.find(donor);
            if (found == row.end())
            {
                storage.capacity(Domain::Scratch, 1, sizeof(std::pair<const GO, Coefficient>) + 4 * sizeof(void*));
                found = row.emplace(donor, Coefficient{}).first;
                if (exact_mode)
                    storage.measure(Domain::Scratch, found->second.exact);
            }
            auto& coefficient = found->second;
            coefficient.value += amount;
            coefficient.magnitude += std::abs(amount);
            ++coefficient.terms;
            face_ids.push_back(face_id);
            face_senders.push_back(packet.peer);
            if (!exact_mode)
            {
                if (has_exact || packet.bytes.size() != 40)
                    throw std::runtime_error(
                        "Finite general VoF coefficient carries an exact payload or trailing bytes.");
                // Finite verification uses the compensated coefficient and its
                // counted roundoff gate below. It retains no rational witness.
                continue;
            }
            SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact> measure;
            if (has_exact == 3)
            {
                if (packet.peer != comm->getRank() || packet.bytes.size() != 48 ||
                    word(packet.bytes, 40) >= contributions.size())
                    throw std::runtime_error("Invalid general VoF local coefficient reference.");
                const auto& source = contributions[std::size_t(word(packet.bytes, 40))];
                int sign = 0;
                {
                    const auto lease = mesh->acquire_execution_view();
                    if (mesh->owner_cell(source.face) == lid)
                        sign = 1;
                    else if (mesh->neighbor_cell(source.face) == lid)
                        sign = -1;
                }
                if (!source.exact_amount || source.donor != donor || mesh->face_global_id(source.face) != face_id ||
                    !sign || amount != sign * source.amount)
                    throw std::runtime_error("General VoF local coefficient metadata mismatch.");
                // The immutable caller contribution survives this collective.
                // Reuse its exact payload for positive incidence; only the exact
                // sign reversal needs bounded arithmetic for the other row.
                measure =
                    sign == 1 ? *source.exact_amount : vof_exact::scale(*source.exact_amount, -1, operation_budget());
            }
            else if (has_exact)
            {
                if (packet.bytes.size() < 48 || word(packet.bytes, 40) != packet.bytes.size() - 48)
                    throw std::runtime_error("Invalid general VoF exact coefficient framing.");
                measure = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::decode(
                    std::span<const char>(packet.bytes.data() + 48, packet.bytes.size() - 48), storage.remaining() / 8);
            }
            else
            {
                if (packet.bytes.size() != 40)
                    throw std::runtime_error("Trailing general VoF coefficient bytes.");
                measure = SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>::from_value(amount);
            }
            // Pin the pending decoded/signed payload before any face-vector
            // capacity growth can consume its reserved retained-result share.
            storage.measure(Domain::Scratch, measure);
            auto& terms = original[static_cast<std::size_t>(lid)].faces;
            if (terms.size() == terms.capacity())
            {
                const auto next = terms.capacity() ? 2 * terms.capacity() : std::size_t(1);
                if (next < terms.capacity())
                    throw std::runtime_error("Binding face capacity overflow.");
                storage.capacity(Domain::Result, next - terms.capacity(), sizeof(GeneralVoFOriginalFaceTerm));
                terms.reserve(next);
            }
            storage.measure(Domain::Result, measure);
            storage.release(Domain::Scratch, measure);
            auto aggregate = vof_exact::add(coefficient.exact, measure, operation_budget());
            storage.measure(Domain::Scratch, aggregate);
            storage.release(Domain::Scratch, coefficient.exact);
            coefficient.exact = std::move(aggregate);
            terms.push_back({face_id, donor, measure});
        }
    }
    catch (...)
    {
        valid = false;
        recipient_error = std::current_exception();
    }
    require(*comm, valid, "General VoF raw face coefficient reached an invalid recipient.", recipient_error);
    // A face can arrive repeatedly (both local endpoints, several donors or
    // opposing groups). Query each directory key once: sparse Tpetra directory
    // implementations need not populate every duplicate query position.
    const std::set<GO> unique_face_set(face_ids.begin(), face_ids.end());
    Teuchos::Array<GO> unique_faces(unique_face_set.begin(), unique_face_set.end());
    Teuchos::Array<int> face_owners(unique_faces.size(), -1);
    (void) mesh->owned_face_map()->getRemoteIndexList(unique_faces(), face_owners());
    std::map<GO, int> authenticated_owner;
    for (std::size_t i = 0; i < static_cast<std::size_t>(unique_faces.size()); ++i)
        authenticated_owner.emplace(unique_faces[i], face_owners[i]);
    for (std::size_t i = 0; i < face_ids.size(); ++i)
        valid = valid && authenticated_owner.at(face_ids[i]) == face_senders[i];
    require(*comm, valid, "General VoF raw face coefficient has an unauthenticated face owner.");
    try
    {
        for (std::size_t c = 0; c < rows.size(); ++c)
        {
            const auto self = map->getGlobalElement(LO(c));
            storage.capacity(Domain::Scratch, 1 + (exact_incoming ? (*exact_incoming)[c].size() : incoming[c].size()),
                sizeof(GO) + 4 * sizeof(void*));
            std::set<GO> required{self};
            if (exact_incoming)
                for (const auto& [donor, amount] : (*exact_incoming)[c])
                    required.insert(donor);
            else
                for (const auto& [donor, amount] : incoming[c])
                    required.insert(donor);
            for (const auto donor : required)
                if (!rows[c].contains(donor))
                {
                    try
                    {
                        storage.capacity(
                            Domain::Scratch, 1, sizeof(std::pair<const GO, Coefficient>) + 4 * sizeof(void*));
                        auto inserted = rows[c].emplace(donor, Coefficient{}).first;
                        if (exact_mode)
                            storage.measure(Domain::Scratch, inserted->second.exact);
                    }
                    catch (...)
                    {
                        valid = false;
                    }
                }
            for (const auto& [donor, coefficient] : rows[c])
            {
                long double expected = 0;
                if (!exact_only)
                {
                    const auto found = incoming[c].find(donor);
                    expected = donor == self ? outgoing[c] : found == incoming[c].end() ? 0 : -found->second;
                }
                if (exact_mode)
                {
                    try
                    {
                        storage.capacity(Domain::Result, 1,
                            sizeof(std::pair<const GO,
                                SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>) +
                                4 * sizeof(void*));
                        storage.measure(Domain::Result, coefficient.exact);
                        original[c].coefficients.emplace(donor, coefficient.exact);
                    }
                    catch (...)
                    {
                        valid = false;
                    }
                }
                if (exact_outgoing)
                {
                    const auto found_exact = (*exact_incoming)[c].find(donor);
                    const auto expected_exact =
                        donor == self ? (*exact_outgoing)[c]
                        : found_exact == (*exact_incoming)[c].end()
                            ? SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>{}
                            : SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>{} -
                                  found_exact->second;
                    valid = valid && coefficient.exact.compare(expected_exact) == 0;
                    continue;
                }
                if (exact_mode)
                {
                    // Exact signed face groups must reproduce the unique routed
                    // coefficient with ONE checked narrowing. A large cancelling
                    // display pair must not hide a materially wrong tiny transfer.
                    try
                    {
                        const double represented =
                            coefficient.exact.sign() ? coefficient.exact.certified().as_double() : 0;
                        valid = valid && represented == static_cast<double>(expected);
                    }
                    catch (...)
                    {
                        valid = false;
                    }
                    continue;
                }
                // Each input group has at most two upstream metric/publication
                // conversions. Neumaier accumulation uses at most
                // five arithmetic operations per term and one final addition;
                // the remaining conversion/subtraction/envelope operations fit
                // within k=8*N+16. Use epsilon as a conservative upper bound on
                // unit roundoff, with explicit gradual-underflow terms. No physical
                // residual tolerance or inventory-dependent floor enters this gate.
                const long double epsilon = std::numeric_limits<double>::epsilon();
                const long double eta = std::numeric_limits<double>::denorm_min();
                const long double n = coefficient.terms;
                const long double magnitude_operations = 5 * n + 1;
                const long double operations = 8 * n + 16;
                const long double magnitude_product = magnitude_operations * epsilon;
                const long double product = operations * epsilon;
                if (!(product < .25L) || !(magnitude_product < .25L))
                {
                    valid = false;
                    continue;
                }
                const long double magnitude_gamma = magnitude_product / (1 - magnitude_product);
                const long double gamma = product / (1 - product);
                const long double magnitude = coefficient.magnitude.value();
                const long double scale =
                    (magnitude + magnitude_operations * eta) / (1 - magnitude_gamma) + std::abs(expected);
                const long double bound = gamma * scale + operations * eta;
                valid = valid && std::isfinite(magnitude) && std::isfinite(coefficient.value.value()) &&
                        std::isfinite(scale) && std::isfinite(bound) &&
                        std::abs(coefficient.value.value() - expected) <= bound;
            }
        }
    }
    catch (...)
    {
        valid = false;
    }
    require(*comm, valid, "General VoF positive transfers disagree with their original face donor equation.");
    try
    {
        (void) storage.finish();
    }
    catch (...)
    {
        valid = false;
    }
    require(*comm, valid, "General VoF binding cumulative result storage exceeds budget.");
    return original;
}
std::vector<GeneralVoFOriginalRow> verify_general_vof_exact_transfer_binding(const SP<const MeshHandle<>>& mesh,
    const std::vector<SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>& outgoing,
    const std::vector<std::map<GO, SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>>& incoming,
    const std::vector<ConservativeVoFTransport::ExactFaceContribution>& faces, std::size_t budget)
{
    // The absent scalar view is encoded explicitly (flag2), never used as physics.
    std::vector<ConservativeVoFTransport::FaceContribution> records;
    for (const auto& face : faces)
        records.push_back({face.face, face.donor, 0,
            std::make_shared<const SimpleFluid::PLIC::GeometryMeasure<SimpleFluid::ArithmeticMode::Exact>>(
                face.amount)});
    return verify_general_vof_transfer_binding(mesh, {}, {}, records, budget, &outgoing, &incoming, true);
}
} // namespace SimpleFluid::detail
