#pragma once
namespace SimpleFluid::PLIC
{
template<> class ExtrudedConstruction<ArithmeticMode::Exact>
{
public:
    struct Impl;
    /** make/decode require a valid embedded base supplied by the caller. They
     * validate the DAG, layout, individual convex cells and native metrics;
     * ExtrudedAMR certifies global root embedding, preserved by its exact templates.
     */
    [[nodiscard]] static std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> make(
        std::shared_ptr<const ExtrudedTopology> topology, std::vector<PointRecipe> points,
        std::vector<std::size_t> active_points, std::vector<ScalarRecipe> axial, std::vector<std::size_t> active_axial);
    [[nodiscard]] static std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> decode(
        std::shared_ptr<const ExtrudedTopology> topology, std::span<const char> packet);
    [[nodiscard]] std::vector<char> encode() const;
    [[nodiscard]] std::vector<char> encode_base() const;
    [[nodiscard]] std::vector<char> encode_axis() const;
    [[nodiscard]] std::vector<char> encode_axis_values() const;
    [[nodiscard]] std::vector<char> encode_axis_selection() const;
    [[nodiscard]] static std::vector<char> combine_axis_parts(
        std::span<const char> values, std::span<const char> selection);
    [[nodiscard]] std::vector<char> encode_view() const;
    [[nodiscard]] static std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> decode_shared(
        std::shared_ptr<const ExtrudedTopology> topology, std::span<const char> base, std::span<const char> axis,
        std::span<const char> view, std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> base_anchor = {},
        std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> axis_anchor = {});
    [[nodiscard]] static std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> shared_view(
        std::shared_ptr<const ExtrudedTopology> topology,
        std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> base,
        std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> axis, std::size_t first_layer,
        std::size_t layers);
    [[nodiscard]] static std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> shared_axis_selection(
        std::shared_ptr<const ExtrudedTopology> topology,
        std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> base,
        std::shared_ptr<const ExtrudedConstruction<ArithmeticMode::Exact>> coordinate_anchor,
        std::vector<std::size_t> axial_points);
    [[nodiscard]] const void* base_storage_identity() const noexcept;
    [[nodiscard]] const void* axis_storage_identity() const noexcept;
    [[nodiscard]] const void* axis_values_storage_identity() const noexcept;
    [[nodiscard]] std::size_t base_storage_bytes() const;
    [[nodiscard]] std::size_t axis_storage_bytes() const;
    [[nodiscard]] std::size_t axis_values_storage_bytes() const;
    [[nodiscard]] std::size_t view_storage_bytes() const noexcept;
    [[nodiscard]] const std::vector<MeshUtils::Vec3>& display_points_ref() const noexcept;
    [[nodiscard]] std::span<const double> z_view() const noexcept;
    [[nodiscard]] std::size_t first_layer() const noexcept;
    [[nodiscard]] std::vector<MeshUtils::Vec3> display_points() const;
    [[nodiscard]] std::vector<double> display_z() const;
    [[nodiscard]] double cell_volume(std::size_t) const;
    [[nodiscard]] MeshUtils::Vec3 cell_centroid(std::size_t) const;
    [[nodiscard]] double face_area(std::size_t) const;
    [[nodiscard]] MeshUtils::Vec3 face_centroid(std::size_t) const;
    [[nodiscard]] MeshUtils::Vec3 face_area_vector(std::size_t) const;
    [[nodiscard]] ConvexGeometryPtr<ArithmeticMode::Exact> certified_cell_geometry(std::size_t) const;
    [[nodiscard]] ConvexGeometryPtr<ArithmeticMode::Exact> atomic_cell_geometry(std::size_t) const;
    [[nodiscard]] PlanarFacePtr<ArithmeticMode::Exact> certified_face_geometry(std::size_t) const;
    [[nodiscard]] std::size_t storage_bytes() const;
    [[nodiscard]] double scaled_base_area(std::size_t base_cell) const;
    [[nodiscard]] bool base_quality_above(std::size_t base_cell, double minimum) const;
    /** Root-only planar embedding admission using a compact broad phase and
     * exact intersections; retained descendants inherit this certificate. */
    void certify_base_embedding() const;
    [[nodiscard]] bool matches_topology(const std::shared_ptr<const ExtrudedTopology>& topology) const noexcept;

private:
    explicit ExtrudedConstruction(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<const Impl> impl_;
};
} // namespace SimpleFluid::PLIC
