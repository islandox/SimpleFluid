/** @file ConservativeVoFFiniteTransport.tcc
 * @brief Compiled finite arithmetic specialization.
 */
template<ArithmeticMode Mode>
ConservativeVoFTransport::Result ConservativeVoFTransport::solve_finite(const std::vector<long double>& old,
    const std::vector<long double>& source, const std::vector<long double>& capacity,
    const std::vector<long double>& outgoing, const std::vector<FiniteTransfer>& incoming,
    const std::vector<FiniteFaceContribution>& contributions, bool nonnegative, int maximum_iterations,
    double tolerance, std::size_t byte_budget, std::span<const long double> old_error,
    std::span<const long double> source_error, std::span<const long double> capacity_error,
    std::span<const long double> outgoing_error, ArithmeticMode arithmetic_mode)
{
    namespace fp = VoF::finite_detail;
    using A = fp::Arithmetic<Mode>;
    using Value = Meshes::BoundedGeometryValue;
    const auto map = mesh_->owned_cell_map();
    const auto comm = map->getComm();
    const auto n = mesh_->num_owned_cells();
    constexpr bool bounded = Mode == ArithmeticMode::BoundedCertification;
    const auto arithmetic_value = [&](long double value, long double error = 0) { return A::make_value(value, error); };
    const auto safe = [&](auto&& action)
    {
        bool ok = true;
        try
        {
            action();
        }
        catch (...)
        {
            ok = false;
        }
        require(*map, ok, "Finite VoF transport rejected its bounded local arithmetic or metadata.");
    };
    require(*map,
        old.size() == n && source.size() == n && capacity.size() == n && outgoing.size() == n &&
            maximum_iterations > 0 && std::isfinite(tolerance) && tolerance > 0 && byte_budget > 0,
        "Finite VoF transport has invalid dimensions or controls.");
    const auto error_at = [](std::span<const long double> values, std::size_t i)
    { return values.empty() ? 0.L : values[i]; };
    require(*map,
        (old_error.empty() || old_error.size() == n) && (source_error.empty() || source_error.size() == n) &&
            (capacity_error.empty() || capacity_error.size() == n) &&
            (outgoing_error.empty() || outgoing_error.size() == n),
        "Finite momentum input error shape mismatch.");
    std::vector<std::map<GO, Value>> routes(n);
    std::set<GO> donor_ids;
    bool valid = true;
    std::size_t bytes = n * (sizeof(long double) * 12 + sizeof(std::map<GO, Value>));
    safe(
        [&]
        {
            for (const auto& errors : {old_error, source_error, capacity_error, outgoing_error})
                for (auto error : errors)
                    valid = valid && std::isfinite(error) && error >= 0;
            for (std::size_t c = 0; c < n; ++c)
                valid = valid && std::isfinite(old[c]) && std::isfinite(source[c]) && std::isfinite(capacity[c]) &&
                        capacity[c] > 0 && std::isfinite(outgoing[c]) && outgoing[c] >= 0;
            for (const auto& route : incoming)
            {
                const auto row = map->getLocalElement(route.recipient);
                valid = valid && row >= 0 && route.donor != route.recipient && std::isfinite(route.amount) &&
                        route.amount >= 0 && std::isfinite(route.absolute_error) && route.absolute_error >= 0;
                if (row >= 0)
                    valid =
                        routes[row].emplace(route.donor, arithmetic_value(route.amount, route.absolute_error)).second &&
                        valid;
                donor_ids.insert(route.donor);
                bytes += sizeof(route) + sizeof(std::pair<const GO, Value>) + 8 * sizeof(void*);
            }
            for (const auto& face : contributions)
            {
                valid = valid && face.face >= 0 && std::size_t(face.face) < mesh_->num_owned_faces() &&
                        std::isfinite(face.amount) && std::isfinite(face.absolute_error) && face.absolute_error >= 0;
                if (face.face >= 0 && std::size_t(face.face) < mesh_->num_owned_faces())
                    valid = valid && !mesh_->is_geometry_exterior_face(face.face);
                donor_ids.insert(face.donor);
                bytes += sizeof(face) + 128;
            }
            if (bytes > byte_budget)
                throw std::invalid_argument("Finite VoF momentum input capacity exceeds its unchanged byte budget.");
        });
    require(*map, valid, "Finite VoF momentum requires positive routes and canonical internal face streams.");
    // Route raw signed face coefficients once to their incident owned rows.
    // The wire is hi/lo significand plus exponent, independent of ABI padding.
    const auto word = [](std::vector<char>& b, std::uint64_t v)
    {
        for (unsigned a = 0; a < 8; ++a)
            b.push_back(char(v >> (8 * a)));
    };
    const auto scalar = [&](std::vector<char>& b, long double v)
    {
        int exponent = 0;
        const auto m = std::frexp(v, &exponent);
        const auto hi = static_cast<double>(m), lo = static_cast<double>(m - hi);
        if (!std::isfinite(v) || static_cast<long double>(hi) + lo != m)
            throw std::invalid_argument("Finite momentum scalar exceeds its fixed wire representation.");
        word(b, std::bit_cast<std::uint64_t>(hi));
        word(b, std::bit_cast<std::uint64_t>(lo));
        word(b, std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(exponent)));
    };
    const auto read = [](const std::vector<char>& b, std::size_t& offset)
    {
        std::uint64_t v = 0;
        for (unsigned a = 0; a < 8; ++a)
            v |= std::uint64_t(static_cast<unsigned char>(b.at(offset++))) << (8 * a);
        return v;
    };
    const auto read_scalar = [&](const std::vector<char>& b, std::size_t& offset)
    {
        const auto hi = std::bit_cast<double>(read(b, offset)), lo = std::bit_cast<double>(read(b, offset));
        const auto exponent = std::bit_cast<std::int64_t>(read(b, offset));
        if (exponent < std::numeric_limits<int>::min() || exponent > std::numeric_limits<int>::max())
            throw std::invalid_argument("Finite momentum exponent is malformed.");
        const auto value = std::scalbn(static_cast<long double>(hi) + lo, int(exponent));
        if (!std::isfinite(value))
            throw std::invalid_argument("Finite momentum wire value is nonfinite.");
        return value;
    };
    std::set<GO> endpoints;
    safe(
        [&]
        {
            const auto lease = mesh_->acquire_execution_view();
            for (const auto& face : contributions)
            {
                endpoints.insert(mesh_->cell_global_id(mesh_->owner_cell(face.face)));
                endpoints.insert(mesh_->cell_global_id(mesh_->neighbor_cell(face.face)));
            }
        });
    Teuchos::Array<GO> endpoint_ids(endpoints.begin(), endpoints.end());
    Teuchos::Array<int> endpoint_owners(endpoint_ids.size(), -1);
    (void) map->getRemoteIndexList(endpoint_ids(), endpoint_owners());
    std::map<GO, int> owner;
    for (std::size_t i = 0; i < std::size_t(endpoint_ids.size()); ++i)
    {
        valid = valid && endpoint_owners[i] >= 0;
        owner.emplace(endpoint_ids[i], endpoint_owners[i]);
    }
    require(*map, valid, "Finite momentum face references a foreign endpoint.");
    std::vector<OwnerPacket> packets;
    safe(
        [&]
        {
            const auto lease = mesh_->acquire_execution_view();
            for (const auto& face : contributions)
                for (int sign : {1, -1})
                {
                    const auto row = mesh_->cell_global_id(
                        sign > 0 ? mesh_->owner_cell(face.face) : mesh_->neighbor_cell(face.face));
                    OwnerPacket packet;
                    packet.peer = owner.at(row);
                    word(packet.bytes, std::uint64_t(row));
                    word(packet.bytes, std::uint64_t(face.donor));
                    word(packet.bytes, std::uint64_t(mesh_->face_global_id(face.face)));
                    scalar(packet.bytes, sign * face.amount);
                    scalar(packet.bytes, face.absolute_error);
                    packets.push_back(std::move(packet));
                }
            if (packets.size() > (byte_budget - std::min(bytes, byte_budget)) / 128)
                throw std::invalid_argument("Finite momentum packet capacity exceeds its unchanged budget.");
        });
    const auto received = exchange_owner_packets(packets, comm, byte_budget - bytes);
    std::vector<std::map<GO, fp::SumFor<Mode>>> raw_sums(n);
    safe(
        [&]
        {
            std::set<std::tuple<int, GO, GO, GO, long double>> seen;
            for (const auto& packet : received.packets)
            {
                std::size_t offset = 0;
                const auto row_gid = GO(read(packet.bytes, offset)), donor = GO(read(packet.bytes, offset));
                const auto face = GO(read(packet.bytes, offset));
                const auto amount = read_scalar(packet.bytes, offset), error = read_scalar(packet.bytes, offset);
                const auto row = map->getLocalElement(row_gid);
                if (row < 0 || offset != packet.bytes.size() || error < 0 || !std::isfinite(amount) ||
                    !seen.emplace(packet.peer, row_gid, donor, face, amount).second)
                    throw std::invalid_argument("Finite momentum raw coefficient is duplicate or malformed.");
                raw_sums[row][donor].add(arithmetic_value(amount, error));
                donor_ids.insert(donor);
            }
            for (std::size_t c = 0; c < n; ++c)
            {
                const auto self = map->getGlobalElement(LO(c));
                std::set<GO> columns{self};
                for (const auto& [donor, value] : raw_sums[c])
                    columns.insert(donor);
                for (const auto& [donor, value] : routes[c])
                    columns.insert(donor);
                for (auto donor : columns)
                {
                    const auto observed =
                        raw_sums[c].contains(donor) ? raw_sums[c].at(donor).value() : arithmetic_value(0);
                    const auto expected = donor == self ? arithmetic_value(outgoing[c], error_at(outgoing_error, c))
                                          : routes[c].contains(donor) ? A::negate(routes[c].at(donor))
                                                                      : arithmetic_value(0);
                    const auto defect = A::subtract(observed, expected);
                    // Routes supply the positive preconditioner. Ordinary FP
                    // compares their coefficients on the original equation scale;
                    // the raw-face operator below still owns physical acceptance.
                    const auto allowed = bounded ? defect.absolute_error
                                                 : static_cast<long double>(tolerance) *
                                                       (capacity[c] + outgoing[c] + std::abs(expected.value));
                    if (std::abs(defect.value) > allowed)
                        throw std::invalid_argument(
                            "Finite momentum positive routes differ from original raw face coefficients.");
                }
            }
        });
    Teuchos::Array<GO> ids(donor_ids.begin(), donor_ids.end());
    const auto donor_map = Teuchos::rcp(
        new Pack::map_type(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), ids(), map->getIndexBase(), comm));
    Pack::import_type donor_import(map, donor_map);
    Pack::vector_type marker(map), donor_trace(donor_map);
    marker.putScalar(1);
    donor_trace.putScalar(0);
    donor_trace.doImport(marker, donor_import, Tpetra::INSERT);
    for (auto value : donor_trace.getData())
        valid = valid && value == 1;
    require(*map, valid, "Finite VoF momentum references an absent geometric donor.");
    std::vector<std::map<GO, Value>> raw(n);
    for (std::size_t c = 0; c < n; ++c)
        for (const auto& [donor, sum] : raw_sums[c])
            raw[c].emplace(donor, sum.value());
    std::size_t width = 1;
    for (const auto& row : routes)
        width = std::max(width, row.size() + 1);
    auto matrix = Teuchos::rcp(new Pack::matrix_type(map, width));
    std::vector<long double> diagonal(n);
    safe(
        [&]
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                diagonal[c] = capacity[c] + outgoing[c];
                if (!(diagonal[c] > 0) || !std::isfinite(diagonal[c]))
                    throw std::invalid_argument("Finite momentum has a nonpositive row diagonal.");
                Teuchos::Array<GO> columns;
                Teuchos::Array<double> values;
                columns.push_back(map->getGlobalElement(LO(c)));
                values.push_back(1);
                for (const auto& [donor, amount] : routes[c])
                {
                    const auto value = static_cast<double>(-amount.value / diagonal[c]);
                    if (!std::isfinite(value) || (amount.value != 0 && value == 0))
                        throw std::invalid_argument("Finite momentum row coefficient is outside representable range.");
                    columns.push_back(donor);
                    values.push_back(value);
                }
                matrix->insertGlobalValues(map->getGlobalElement(LO(c)), columns(), values());
                const auto value = static_cast<double>((old[c] + source[c]) / diagonal[c]);
                if (!std::isfinite(value))
                    throw std::invalid_argument("Finite momentum normalized RHS overflows.");
                rhs_.replaceLocalValue(LO(c), value);
            }
        });
    matrix->fillComplete();
    solver_.notify_operator_values_changed();
    LinearSolverOptions linear;
    linear.backend = LinearSolverBackend::Gmres;
    linear.preconditioner = LinearPreconditioner::Jacobi;
    linear.reuse_preconditioner = true;
    linear.tolerance = std::max(tolerance, 64 * std::numeric_limits<double>::epsilon());
    linear.max_iterations = maximum_iterations;
    linear.verbosity = 0;
    Result result;
    std::vector<double> trace(n);
    const auto solve = [&]
    {
        linear.max_iterations = maximum_iterations - result.iterations;
        const auto stats = solver_.solve_from_zero_with_statistics(
            Teuchos::rcp_implicit_cast<const Pack::operator_type>(matrix), rhs_, solution_, linear);
        require(*map, stats.converged, "Finite momentum inner solve failed its true residual.");
        result.iterations += stats.iterations;
    };
    solve();
    std::copy(solution_.getData().begin(), solution_.getData().end(), trace.begin());
    int local_exponent = std::numeric_limits<int>::min(), exponent = 0;
    for (std::size_t c = 0; c < n; ++c)
        if (old[c] + source[c] != 0)
        {
            int e = 0;
            (void) std::frexp(std::abs(old[c] + source[c]), &e);
            local_exponent = std::max(local_exponent, e);
        }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_exponent, &exponent);
    const auto reference_scale = exponent == std::numeric_limits<int>::min() ? 1.L : std::scalbn(1.L, exponent - 1);
    std::vector<long double> defect(n);
    std::vector<CompensatedSum<>> accumulated(n);
    for (std::size_t c = 0; c < n; ++c)
        accumulated[c] += trace[c];
    for (;;)
    {
        {
            auto values = cell_owned_.getDataNonConst();
            std::copy(trace.begin(), trace.end(), values.begin());
        }
        donor_trace.doImport(cell_owned_, donor_import, Tpetra::INSERT);
        const auto donor_values = donor_trace.getData();
        double local_norm[2]{}, global_norm[2]{};
        safe(
            [&]
            {
                fp::SumFor<Mode> residual_squared(arithmetic_mode), reference_squared(arithmetic_mode);
                for (std::size_t c = 0; c < n; ++c)
                {
                    fp::SumFor<Mode> equation(arithmetic_mode);
                    equation.add(A::multiply(
                        arithmetic_value(capacity[c], error_at(capacity_error, c)), arithmetic_value(trace[c], 0)));
                    equation.add(arithmetic_value(-old[c], error_at(old_error, c)));
                    equation.add(arithmetic_value(-source[c], error_at(source_error, c)));
                    for (const auto& [donor, amount] : raw[c])
                    {
                        const auto donor_lid = donor_map->getLocalElement(donor);
                        if (donor_lid < 0)
                            throw std::invalid_argument("Finite momentum raw row references absent trace.");
                        equation.add(A::multiply(amount, arithmetic_value(donor_values[donor_lid], 0)));
                    }
                    const auto residual = equation.value();
                    defect[c] = -residual.value;
                    const auto upper =
                        (std::abs(residual.value) + (bounded ? residual.absolute_error : 0)) / reference_scale;
                    const auto rhs = (old[c] + source[c]) / reference_scale;
                    residual_squared.add(A::multiply(arithmetic_value(upper, 0), arithmetic_value(upper, 0)));
                    reference_squared.add(A::multiply(arithmetic_value(rhs, 0), arithmetic_value(rhs, 0)));
                }
                local_norm[0] = bounded ? fp::upper_double(fp::upward(
                                              residual_squared.value().value + residual_squared.value().absolute_error))
                                        : static_cast<double>(residual_squared.value().value);
                const auto lower = std::max(
                    0.L, reference_squared.value().value - (bounded ? reference_squared.value().absolute_error : 0));
                local_norm[1] = static_cast<double>(lower);
                if (bounded && static_cast<long double>(local_norm[1]) > lower)
                    local_norm[1] = std::nextafter(local_norm[1], 0.);
                if (!std::isfinite(local_norm[0]) || !std::isfinite(local_norm[1]))
                    throw std::invalid_argument("Finite momentum physical residual reduction overflows.");
            });
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 2, local_norm, global_norm);
        const auto reduction_u = bounded ? (8.L * comm->getSize() + 16) * std::numeric_limits<double>::epsilon() : 0;
        const auto residual_upper = global_norm[0] * (1 + reduction_u);
        const auto reference_lower = global_norm[1] * (1 - reduction_u);
        result.physical_relative_residual = std::sqrt(residual_upper / (reference_lower > 0 ? reference_lower : 1));
        if (std::isfinite(residual_upper) &&
            ((reference_lower > 0 &&
                 residual_upper <= static_cast<long double>(tolerance) * tolerance * reference_lower) ||
                (reference_lower == 0 && residual_upper == 0)))
            break;
        require(*map, result.physical_refinements < 4 && result.iterations < maximum_iterations,
            "Finite momentum failed its unchanged original physical residual.");
        int local_defect_exponent = std::numeric_limits<int>::min(), defect_exponent = 0;
        safe(
            [&]
            {
                for (std::size_t c = 0; c < n; ++c)
                {
                    defect[c] /= diagonal[c];
                    if (!std::isfinite(defect[c]))
                        throw std::invalid_argument("Finite momentum defect is outside qualified range.");
                    if (defect[c] != 0)
                    {
                        int e = 0;
                        (void) std::frexp(std::abs(defect[c]), &e);
                        local_defect_exponent = std::max(local_defect_exponent, e);
                    }
                }
            });
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_defect_exponent, &defect_exponent);
        require(*map, defect_exponent != std::numeric_limits<int>::min(),
            "Finite momentum arithmetic bound cannot establish the physical residual target.");
        const auto defect_scale = std::scalbn(1.L, defect_exponent - 1);
        for (std::size_t c = 0; c < n; ++c)
            rhs_.replaceLocalValue(LO(c), static_cast<double>(defect[c] / defect_scale));
        solve();
        const auto correction = solution_.getData();
        for (std::size_t c = 0; c < n; ++c)
        {
            accumulated[c] += correction[c] * defect_scale;
            trace[c] = static_cast<double>(accumulated[c].value());
        }
        ++result.physical_refinements;
    }
    result.cell_trace = trace;
    result.amount.resize(n);
    safe(
        [&]
        {
            for (std::size_t c = 0; c < n; ++c)
            {
                const auto amount = capacity[c] * trace[c];
                result.amount[c] = static_cast<double>(amount);
                if (!std::isfinite(trace[c]) || !std::isfinite(result.amount[c]) ||
                    (amount != 0 && result.amount[c] == 0) || (nonnegative && amount < 0))
                    throw std::invalid_argument("Finite momentum final publication is outside qualified range.");
            }
        });
    numeric_valid_ = false;
    exact_preparation_.reset();
    return result;
}

template ConservativeVoFTransport::Result ConservativeVoFTransport::solve_finite<ArithmeticMode::Ordinary>(
    const std::vector<long double>& old, const std::vector<long double>& source,
    const std::vector<long double>& capacity, const std::vector<long double>& outgoing,
    const std::vector<ConservativeVoFTransport::FiniteTransfer>& incoming,
    const std::vector<ConservativeVoFTransport::FiniteFaceContribution>& contributions, bool nonnegative,
    int maximum_iterations, double tolerance, std::size_t byte_budget, std::span<const long double> old_error,
    std::span<const long double> source_error, std::span<const long double> capacity_error,
    std::span<const long double> outgoing_error, ArithmeticMode arithmetic_mode);

template ConservativeVoFTransport::Result ConservativeVoFTransport::solve_finite<ArithmeticMode::BoundedCertification>(
    const std::vector<long double>& old, const std::vector<long double>& source,
    const std::vector<long double>& capacity, const std::vector<long double>& outgoing,
    const std::vector<ConservativeVoFTransport::FiniteTransfer>& incoming,
    const std::vector<ConservativeVoFTransport::FiniteFaceContribution>& contributions, bool nonnegative,
    int maximum_iterations, double tolerance, std::size_t byte_budget, std::span<const long double> old_error,
    std::span<const long double> source_error, std::span<const long double> capacity_error,
    std::span<const long double> outgoing_error, ArithmeticMode arithmetic_mode);
