/** @file Support.tcc
 * @brief Compiled helpers shared by the selected General VoF solver kernels.
 */
namespace SimpleFluid
{
namespace
{
using Pack = DefaultTpetraTypes;
using LO = Pack::local_ordinal_type;
using GO = Pack::global_ordinal_type;
using Vector = VoF::Vector;
double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
void append_word(std::vector<char>& bytes, std::uint64_t value)
{
    for (unsigned a = 0; a < 8; ++a)
        bytes.push_back(char(value >> (8 * a)));
}
std::uint64_t read_word(const std::vector<char>& bytes, std::size_t offset)
{
    std::uint64_t value = 0;
    for (unsigned a = 0; a < 8; ++a)
        value |= std::uint64_t(static_cast<unsigned char>(bytes.at(offset + a))) << (8 * a);
    return value;
}
std::shared_ptr<const MeshHandle<>> validated_phase_mesh(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    const auto mesh = phases->mesh();
    int local = phases->hierarchy()->geometry_unchanged(), global = 0;
    Teuchos::reduceAll(*mesh->owned_cell_map()->getComm(), Teuchos::REDUCE_MIN, 1, &local, &global);
    if (!global)
        throw std::runtime_error("General VoF source geometry is stale before workspace construction.");
    return mesh;
}
template<ArithmeticMode Mode>
std::vector<Vector> finite_spans_mode(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    std::vector<Vector> result;
    for (const auto& geometry : phases->owned_finite_geometry<Mode>())
    {
        const auto spans = geometry->axis_spans();
        result.push_back({spans[0].as_double(), spans[1].as_double(), spans[2].as_double()});
    }
    return result;
}
std::vector<Vector> finite_spans(const std::shared_ptr<const VoF::GeneralVoF>& phases)
{
    if (phases->arithmetic_mode() == ArithmeticMode::Exact)
        return {};
    if (phases->arithmetic_mode() == ArithmeticMode::Ordinary)
        return finite_spans_mode<ArithmeticMode::Ordinary>(phases);
    return finite_spans_mode<ArithmeticMode::BoundedCertification>(phases);
}
} // namespace
} // namespace SimpleFluid
