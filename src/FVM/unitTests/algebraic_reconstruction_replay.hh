/** @file algebraic_reconstruction_replay.hh
 * @brief Local replay of captured accepted-source reconstruction inputs; no physical qualification claim.
 */
namespace algebraic_replay
{
constexpr auto mode = ArithmeticMode::Ordinary;
using Sample = PLIC::PolyhedralPhaseSample<mode>;
struct CapturedCell
{
    int owner = -1;
    AMR::AdaptiveCellKey key;
    bool inventory = false;
    double gas = 0, liquid = 0;
    std::vector<std::uint64_t> stencil;
    std::filesystem::path certificate;
};
struct Capture
{
    std::map<std::uint64_t, CapturedCell> cells;
    std::map<std::uint64_t, std::set<std::uint64_t>> adjacency;
    PLIC::PolyhedralReconstructionOptions options;
    unsigned rings = 2;
    std::vector<std::string> metadata;
};
AMR::AdaptiveCellKey read_key(const std::string& text)
{
    std::vector<std::string> parts;
    std::istringstream in(text);
    std::string part;
    while (std::getline(in, part, ':'))
        parts.push_back(part);
    if (parts.size() == 4 && parts[1] == "0")
    {
        AMR::CartesianCellAncestry cartesian;
        cartesian.level = unsigned(std::stoul(parts[2].substr(6)));
        std::istringstream indices(parts[3].substr(6));
        for (auto& index : cartesian.index)
        {
            std::string value;
            if (!std::getline(indices, value, ','))
                throw std::runtime_error("Malformed Cartesian capture key");
            index = std::stoull(value);
        }
        return {std::stoull(parts[0]), cartesian};
    }
    if (parts.size() != 7 || parts[1] != "2")
        throw std::runtime_error("Replay requires captured CartesianAMR or ExtrudedAMR keys");
    const auto path = [](const std::string& value)
    {
        std::vector<std::uint64_t> result;
        std::istringstream stream(value.substr(5));
        std::string component;
        while (std::getline(stream, component, ','))
            if (!component.empty())
                result.push_back(std::stoull(component));
        return result;
    };
    AMR::ProductCellAncestry product;
    product.base.root = std::stoull(parts[2].substr(5));
    product.base.root_template = std::stoull(parts[3]);
    product.base.path = path(parts[4]);
    product.axial.root = std::stoull(parts[5].substr(2));
    product.axial.path = path(parts[6]);
    return {std::stoull(parts[0]), product};
}
void read_options(std::istream& in, PLIC::PolyhedralReconstructionOptions& o)
{
    std::string key;
    double value;
    while (in >> key >> value)
    {
        if (key == "maximum_offset_iterations")
            o.maximum_offset_iterations = unsigned(value);
        else if (key == "maximum_fit_iterations")
            o.maximum_fit_iterations = unsigned(value);
        else if (key == "maximum_stencil_cells")
            o.maximum_stencil_cells = std::size_t(value);
        else if (key == "maximum_clip_evaluations")
            o.maximum_clip_evaluations = std::size_t(value);
        else if (key == "volume_tolerance")
            o.volume_tolerance = value;
        else if (key == "fit_tolerance")
            o.fit_tolerance = value;
        else if (key == "maximum_condition")
            o.maximum_condition = value;
        else if (key == "orientation_tolerance")
            o.orientation_tolerance = value;
        else if (key == "maximum_model_residual")
            o.maximum_model_residual = value;
        else if (key == "minimum_phase_support_fraction")
            o.minimum_phase_support_fraction = value;
        else if (key == "minimum_optimized_phase_length_ratio")
            o.minimum_optimized_phase_length_ratio = value;
    }
    // GeometricFace explicitly enables this for candidate reconstruction.
    o.measure_only = true;
}
Capture load(const std::filesystem::path& directory, const std::string& prefix)
{
    Capture result;
    std::set<std::string> groups;
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        const auto filename = entry.path().filename().string();
        if (!filename.ends_with(".state.txt") || (!prefix.empty() && !filename.starts_with(prefix)))
            continue;
        const auto rank_suffix = filename.rfind("_rank");
        if (rank_suffix == std::string::npos)
            throw std::runtime_error("State filename lacks rank suffix");
        groups.insert(filename.substr(0, rank_suffix));
        if (groups.size() > 1)
            throw std::runtime_error("Multiple capture groups: set SIMPLEFLUID_REPLAY_CAPTURE to one filename prefix");
        std::ifstream in(entry.path());
        if (!in)
            throw std::runtime_error("Could not open captured state");
        std::string line;
        while (std::getline(in, line))
        {
            std::istringstream row(line);
            std::string kind;
            row >> kind;
            if (kind == "owned" || kind == "support")
            {
                std::string token;
                std::uint64_t gid;
                row >> token >> gid;
                auto& cell = result.cells[gid];
                double gas = 0, liquid = 0;
                while (row >> token)
                {
                    if (token == "owner")
                    {
                        int owner;
                        row >> owner;
                        if (cell.owner != -1 && cell.owner != owner)
                            throw std::runtime_error("Captured owner disagrees");
                        cell.owner = owner;
                    }
                    else if (token == "key")
                    {
                        std::string key;
                        row >> key;
                        cell.key = read_key(key);
                    }
                    else if (token == "gas")
                        row >> gas;
                    else if (token == "liquid")
                        row >> liquid;
                    else if (token == "stencil_gids")
                    {
                        std::uint64_t neighbor;
                        while (row >> neighbor)
                            cell.stencil.push_back(neighbor);
                    }
                }
                if (kind == "owned")
                {
                    if (cell.inventory)
                        throw std::runtime_error("Duplicate captured owned inventory");
                    cell.inventory = true;
                    cell.gas = gas;
                    cell.liquid = liquid;
                }
            }
            else if (kind == "certificate")
            {
                std::uint64_t gid;
                std::string path;
                row >> gid >> path;
                result.cells[gid].certificate = directory / path;
            }
            else if (kind == "tile")
            {
                std::string token;
                std::uint64_t first = 0, second = 0;
                bool have_first = false, have_second = false;
                while (row >> token)
                {
                    if (token == "first")
                    {
                        row >> first;
                        have_first = true;
                    }
                    else if (token == "second")
                    {
                        row >> second;
                        have_second = true;
                    }
                    else if (token == "nodes")
                        break;
                }
                if (have_first && have_second)
                {
                    result.adjacency[first].insert(second);
                    result.adjacency[second].insert(first);
                }
            }
            else if (kind == "reconstruction")
                read_options(row, result.options);
            else if (kind == "byte_budget")
            {
                std::string token;
                while (row >> token)
                    if (token == "support_rings")
                        row >> result.rings;
            }
            else if (kind == "source_accepted_control" || kind == "accepted_steps" || kind == "failure_record")
                result.metadata.push_back(line);
        }
    }
    if (groups.empty())
        throw std::runtime_error("No matching captured state files");
    return result;
}
Sample fit_sample(const CapturedCell& c)
{
    std::ifstream in(c.certificate, std::ios::binary);
    if (!in)
        throw std::runtime_error("Missing geometry certificate: " + c.certificate.string());
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto geometry = PLIC::ConvexCellGeometry<mode>::decode_certificate(bytes);
    const auto volume = geometry->volume().value;
    // Independent physical inventories remain untouched. Only the geometric
    // fitting label complements the smaller independently stored amount.
    const bool gas_minor = c.gas <= c.liquid;
    const auto minority = gas_minor ? c.gas : c.liquid;
    if (volume < minority)
        throw std::runtime_error("Captured minority exceeds original native geometry");
    const auto dominant = double(volume - minority);
    return {geometry, gas_minor ? minority : dominant, gas_minor ? dominant : minority};
}
std::set<std::uint64_t> expected_stencil(const Capture& capture, std::uint64_t target)
{
    std::set<std::uint64_t> reached{target}, frontier{target};
    for (unsigned ring = 0; ring < capture.rings; ++ring)
    {
        std::set<std::uint64_t> next;
        for (const auto gid : frontier)
            if (const auto found = capture.adjacency.find(gid); found != capture.adjacency.end())
                for (const auto neighbor : found->second)
                    if (reached.insert(neighbor).second)
                        next.insert(neighbor);
        frontier = std::move(next);
    }
    return reached;
}
std::vector<std::uint64_t> production_order(const Capture& capture, std::uint64_t target)
{
    // Match documented breadth-first ring discovery in persistent key order,
    // independently of transient gid numbering and the sorted support listing.
    std::set<std::uint64_t> reached{target};
    std::map<AMR::AdaptiveCellKey, std::uint64_t> frontier{{capture.cells.at(target).key, target}};
    std::vector<std::uint64_t> ordered;
    for (unsigned ring = 0; ring < capture.rings; ++ring)
    {
        std::map<AMR::AdaptiveCellKey, std::uint64_t> next;
        for (const auto& [key, gid] : frontier)
            if (const auto found = capture.adjacency.find(gid); found != capture.adjacency.end())
            {
                std::map<AMR::AdaptiveCellKey, std::uint64_t> neighbors;
                for (const auto neighbor : found->second)
                    neighbors.emplace(capture.cells.at(neighbor).key, neighbor);
                for (const auto& [neighbor_key, neighbor] : neighbors)
                    if (reached.insert(neighbor).second)
                    {
                        ordered.push_back(neighbor);
                        next.emplace(neighbor_key, neighbor);
                    }
            }
        frontier = std::move(next);
    }
    return ordered;
}
void report_sample(std::ostream& out, std::uint64_t gid, const CapturedCell& c, const Sample& sample)
{
    const auto center = sample.geometry->centroid();
    const auto volume = sample.geometry->volume().value;
    out << "sample gid=" << gid << " owner=" << c.owner << " center=" << center[0] << ',' << center[1] << ','
        << center[2] << " native_volume=" << volume << " physical_G=" << c.gas << " physical_L=" << c.liquid
        << " fit_G=" << sample.gas_volume << " fit_L=" << sample.carrier_volume
        << " gas_support_fraction=" << sample.gas_volume / volume
        << " liquid_support_fraction=" << sample.carrier_volume / volume << '\n';
}
void report_assessment(std::ostream& out, const PLIC::PolyhedralResolutionAssessment& a)
{
    out << "assessment reason=" << int(a.reason) << " admitted=" << a.admitted() << " stencil_cells=" << a.stencil_cells
        << " gas_support=" << a.gas_support_fraction << " liquid_support=" << a.carrier_support_fraction
        << " gradient_strength=" << a.directional_strength << " gradient_uncertainty=" << a.directional_uncertainty
        << " condition=" << a.stencil_condition << " angular_eligible=" << a.angular_fit_eligible
        << " local_span=" << a.local_cell_span << " requested_span=" << a.requested_cell_span << '\n';
}
void report_plane(std::ostream& out, const char* label, const PLIC::PolyhedralInterface& plane)
{
    out << label << " orientation_method=" << int(plane.orientation_method) << " normal=" << plane.gas.normal[0] << ','
        << plane.gas.normal[1] << ',' << plane.gas.normal[2] << " offset=" << plane.gas.offset
        << " anchor=" << plane.gas.anchor[0] << ',' << plane.gas.anchor[1] << ',' << plane.gas.anchor[2]
        << " target_residual=" << plane.target_volume_residual
        << " finite_target_difference=" << plane.finite_target_volume_difference.value
        << " target_numerical_error=" << plane.target_volume_numerical_error_estimate
        << " neighbor_rms_measured=" << plane.stencil_volume_residual_measured
        << " neighbor_rms=" << plane.stencil_volume_residual << " angular_converged=" << plane.fit_converged << '\n';
}
PLIC::GeometryPoint diagnostic_gradient(const Sample& target, const std::vector<Sample>& stencil)
{
    // Independent long-double inverse-distance weighted least squares. This is
    // a diagnostic normal, never a substitute for the production optimizer.
    const bool gas = target.gas_volume <= target.carrier_volume;
    const auto fraction = [&](const Sample& s)
    { return (gas ? s.gas_volume : s.carrier_volume) / (static_cast<long double>(s.gas_volume) + s.carrier_volume); };
    const auto center = target.geometry->centroid();
    std::array<std::array<long double, 4>, 3> matrix{};
    for (const auto& sample : stencil)
    {
        const auto neighbor = sample.geometry->centroid();
        std::array<long double, 3> d;
        long double radius2 = 0;
        for (unsigned i = 0; i < 3; ++i)
        {
            d[i] = neighbor[i] - center[i];
            radius2 += d[i] * d[i];
        }
        if (!(radius2 > 0))
            throw std::runtime_error("Diagnostic LS repeats target center");
        const auto delta = (gas ? -1 : 1) * (fraction(sample) - fraction(target));
        for (unsigned i = 0; i < 3; ++i)
        {
            for (unsigned j = 0; j < 3; ++j)
                matrix[i][j] += d[i] * d[j] / radius2;
            matrix[i][3] += d[i] * delta / radius2;
        }
    }
    for (unsigned column = 0; column < 3; ++column)
    {
        unsigned pivot = column;
        for (unsigned row = column + 1; row < 3; ++row)
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column]))
                pivot = row;
        std::swap(matrix[pivot], matrix[column]);
        const auto diagonal = matrix[column][column];
        if (diagonal == 0)
            throw std::runtime_error("Diagnostic LS is rank deficient");
        for (unsigned j = column; j < 4; ++j)
            matrix[column][j] /= diagonal;
        for (unsigned row = 0; row < 3; ++row)
            if (row != column)
            {
                const auto factor = matrix[row][column];
                for (unsigned j = column; j < 4; ++j)
                    matrix[row][j] -= factor * matrix[column][j];
            }
    }
    const auto length = std::hypot(matrix[0][3], matrix[1][3], matrix[2][3]);
    if (!(length > 0))
        throw std::runtime_error("Diagnostic LS has zero direction");
    return {double(matrix[0][3] / length), double(matrix[1][3] / length), double(matrix[2][3] / length)};
}
void diagnostic_offset(std::ostream& out, const Sample& target, const std::vector<Sample>& stencil,
    const PLIC::PolyhedralReconstructionOptions& options)
{
    const auto normal = diagnostic_gradient(target, stencil);
    const auto plane = PLIC::PolyhedralReconstruction<mode>::solve_offset(target, normal, options);
    report_plane(out, "diagnostic_LS_strict_target_offset_only", plane);
    out << "diagnostic_strict_relative_minority_residual="
        << std::abs(plane.finite_target_volume_difference.value) / std::min(target.gas_volume, target.carrier_volume)
        << " original_volume_tolerance=" << options.volume_tolerance << '\n';
    auto measured_plane = plane.gas;
    const bool gas = target.gas_volume <= target.carrier_volume;
    if (!gas)
    {
        for (auto& component : measured_plane.normal)
            component = -component;
        measured_plane.offset = -measured_plane.offset;
    }
    long double squared = 0;
    for (const auto& sample : stencil)
    {
        const auto measured = PLIC::OrdinaryPlaneMeasure(*sample.geometry).below(measured_plane);
        const auto reference = gas ? sample.gas_volume : sample.carrier_volume;
        const auto residual = (measured.volume - reference) / sample.geometry->volume().value;
        squared += residual * residual;
        out << "diagnostic_neighbor residual_fraction=" << residual << " measure_status=" << int(measured.status)
            << " numerical_error=" << measured.numerical_error_estimate << '\n';
    }
    out << "diagnostic_LS_neighbor_RMS=" << std::sqrt(squared / stencil.size())
        << " measured=true production_optimizer_result=false\n";
}
TEST(GeneralVoFMotionAccuracy, ReplayCapturedAlgebraicReconstruction)
{
    const auto* directory = std::getenv("SIMPLEFLUID_REPLAY_ALGEBRAIC_STATE");
    if (!directory)
        GTEST_SKIP() << "Set captured-state directory and SIMPLEFLUID_REPLAY_CELL for local diagnostic replay";
    const auto* requested = std::getenv("SIMPLEFLUID_REPLAY_CELL");
    ASSERT_NE(requested, nullptr);
    const auto target_gid = std::stoull(requested);
    const auto* prefix = std::getenv("SIMPLEFLUID_REPLAY_CAPTURE");
    const auto capture = load(directory, prefix ? prefix : "");
    const auto found = capture.cells.find(target_gid);
    ASSERT_NE(found, capture.cells.end()) << "Requested gid absent from accepted-source capture";
    ASSERT_TRUE(found->second.inventory) << "Target's owner file is missing";
    const auto expected = expected_stencil(capture, target_gid);
    const std::set<std::uint64_t> captured(found->second.stencil.begin(), found->second.stencil.end());
    std::ostringstream report;
    report << std::setprecision(std::numeric_limits<long double>::max_digits10)
           << "snapshot=accepted_source unpublished_candidate_not_captured=true physical_qualification=false\n";
    for (const auto& line : capture.metadata)
        report << line << '\n';
    bool complete = expected == captured;
    for (const auto gid : expected)
    {
        const auto cell = capture.cells.find(gid);
        if (!captured.contains(gid))
            report << "missing_stencil_gid=" << gid
                   << " owner=" << (cell == capture.cells.end() ? -1 : cell->second.owner) << '\n';
        if (cell == capture.cells.end() || !cell->second.inventory || cell->second.certificate.empty())
        {
            complete = false;
            report << "missing_sample_gid=" << gid
                   << " owner=" << (cell == capture.cells.end() ? -1 : cell->second.owner) << '\n';
        }
    }
    for (const auto gid : captured)
        if (!expected.contains(gid))
            report << "unexpected_stencil_gid=" << gid << '\n';
    report << "support_rings=" << capture.rings << " expected_count=" << expected.size()
           << " captured_count=" << captured.size() << " halo_complete=" << complete << '\n';
    ASSERT_TRUE(complete) << report.str();
    const auto target = fit_sample(found->second);
    ASSERT_GT(target.gas_volume, 0) << "Requested reconstruction target must be mixed";
    ASSERT_GT(target.carrier_volume, 0) << "Requested reconstruction target must be mixed";
    report_sample(report, target_gid, found->second, target);
    std::vector<Sample> stencil;
    const auto ordered = production_order(capture, target_gid);
    report << "production_bfs_stencil_order=";
    for (const auto gid : ordered)
        report << gid << ',';
    report << '\n';
    for (const auto gid : ordered)
    {
        const auto& cell = capture.cells.at(gid);
        stencil.push_back(fit_sample(cell));
        report_sample(report, gid, cell, stencil.back());
    }
    const auto assessment = PLIC::PolyhedralReconstruction<mode>::assess_resolution(target, stencil, capture.options);
    report_assessment(report, assessment);
    report << "production_orientation_before_reconstruct=unpublished angular_fit_eligible="
           << assessment.angular_fit_eligible << " measure_only=" << capture.options.measure_only << '\n';
    bool rejected = false;
    int rejection_reason = -1;
    try
    {
        report_plane(report, "production_replay",
            PLIC::PolyhedralReconstruction<mode>::reconstruct(target, stencil, capture.options));
    }
    catch (const PLIC::PolyhedralResolutionRequired& error)
    {
        rejected = true;
        rejection_reason = int(error.assessment().reason);
        report << "production_rejected=true error=" << error.what() << '\n';
        report_assessment(report, error.assessment());
    }
    catch (const std::exception& error)
    {
        rejected = true;
        report << "production_rejected=true non_resolution_error=" << error.what() << '\n';
    }
    try
    {
        diagnostic_offset(report, target, stencil, capture.options);
    }
    catch (const std::exception& error)
    {
        report << "diagnostic_LS_offset_rejected=" << error.what() << '\n';
    }
    if (const auto* expected_reason = std::getenv("SIMPLEFLUID_REPLAY_EXPECTED_REASON"))
    {
        const std::string value(expected_reason);
        if (value == "accepted")
            EXPECT_FALSE(rejected);
        else if (value == "any_rejection")
            EXPECT_TRUE(rejected);
        else
        {
            EXPECT_TRUE(rejected);
            EXPECT_EQ(rejection_reason, std::stoi(value));
        }
    }
    RecordProperty("replay_diagnostics", report.str());
    const auto rank = Tpetra::getDefaultComm()->getRank();
    const auto path = std::filesystem::path(directory) /
                      ("replay-gid" + std::to_string(target_gid) + "-rank" + std::to_string(rank) + ".txt");
    std::ofstream out(path);
    ASSERT_TRUE(out.good());
    out << report.str();
    out.close();
    ASSERT_FALSE(out.fail());
    if (rank == 0)
        std::cout << report.str();
}
} // namespace algebraic_replay
