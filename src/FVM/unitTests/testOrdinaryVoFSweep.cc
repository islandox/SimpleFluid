/** @file testOrdinaryVoFSweep.cc
 * @brief Ordinary signed sweep defects use independent physical cell budgets.
 */
#include "geometry/swept_volume/FiniteVoFTransport.hh"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <map>

namespace
{
using SimpleFluid::ArithmeticMode;
namespace fp = SimpleFluid::VoF::finite_detail;
using Point = SimpleFluid::PLIC::Point<ArithmeticMode::Ordinary>;

struct StationarySquare
{
    static constexpr long double height = 1e12L;
    const std::array<std::uint64_t, 4> nodes{0, 1, 2, 3};
    std::map<std::uint64_t, std::pair<Point, Point>> traces;

    StationarySquare()
    {
        const std::array<std::array<long double, 3>, 4> points{
            {{0, 0, height}, {1, 0, height}, {1, 1, height}, {0, 1, height}}};
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            const Point point(points[i]);
            traces.emplace(nodes[i], std::pair{point, point});
        }
    }
};

SimpleFluid::geometry::swept_volume::Work ordinary_work()
{
    SimpleFluid::geometry::swept_volume::Work work;
    work.arithmetic_mode = ArithmeticMode::Ordinary;
    return work;
}

// Allocate the unchanged physical volume tolerance equally to the six faces
// of a unit-area box. The budget depends on cell volume, never observed defect.
long double face_budget(long double cell_volume, const SimpleFluid::geometry::swept_volume::Work& work)
{
    return cell_volume * static_cast<long double>(work.volume_tolerance) / 6;
}

TEST(OrdinaryVoFSweep, CellBudgetRetainsQuantizedSignedDefectAndCanonicalControl)
{
    const StationarySquare face;
    for (const long double sign : {-1.L, 1.L})
    {
        auto work = ordinary_work();
        const auto budget = face_budget(1, work);
        const auto control = fp::make_value(ArithmeticMode::Ordinary, sign * budget / 2);
        ASSERT_LT(std::abs(control.value),
            std::nextafter(StationarySquare::height, std::numeric_limits<long double>::infinity()) -
                StationarySquare::height);
        const auto sweep = SimpleFluid::geometry::swept_volume::make_sweep(7, face.nodes, face.traces, control, work, budget);
        EXPECT_EQ(sweep.face, 7u);
        EXPECT_EQ(sweep.backcenter.value, sweep.center.value);
        EXPECT_EQ(sweep.backcenter.value[2], StationarySquare::height);
        ASSERT_TRUE(sweep.pieces.empty());
        EXPECT_EQ(sweep.constructed_volume.value, 0);
        EXPECT_EQ(sweep.constructed_volume.absolute_error, 0);
        EXPECT_EQ(sweep.constructed_volume.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(sweep.control.value, control.value);
        EXPECT_EQ(sweep.control.absolute_error, control.absolute_error);
        EXPECT_EQ(sweep.control.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_EQ(sweep.signed_control_difference.value, -control.value);
        EXPECT_EQ(sweep.signed_control_difference.absolute_error, 0);
        EXPECT_EQ(sweep.signed_control_difference.arithmetic_mode, ArithmeticMode::Ordinary);
        EXPECT_NE(sweep.signed_control_difference.value, 0);
        EXPECT_EQ(sweep.ordinary_volume_budget, budget);
        EXPECT_LE(std::abs(sweep.constructed_volume.value - sweep.control.value), budget);
        for (auto node : face.nodes)
        {
            EXPECT_EQ(face.traces.at(node).first.value, face.traces.at(node).second.value);
            EXPECT_EQ(face.traces.at(node).first.value[2], StationarySquare::height);
        }
    }
}

TEST(OrdinaryVoFSweep, SmallerIndependentCellBudgetRejectsTheSameSignedDefect)
{
    const StationarySquare face;
    for (const long double sign : {-1.L, 1.L})
    {
        auto work = ordinary_work();
        const auto control = fp::make_value(ArithmeticMode::Ordinary, sign * face_budget(1, work) / 2);
        // A unit-area box one quarter as deep has a smaller independent quota;
        // its stationary face and requested canonical control stay identical.
        const auto smaller_budget = face_budget(.25L, work);
        ASSERT_GT(std::abs(control.value), smaller_budget);
        EXPECT_THROW(
            (void) SimpleFluid::geometry::swept_volume::make_sweep(7, face.nodes, face.traces, control, work, smaller_budget), std::invalid_argument);
        EXPECT_EQ(control.value, sign * face_budget(1, work) / 2);
    }
}

TEST(OrdinaryVoFSweep, InvalidIndependentBudgetsReject)
{
    const StationarySquare face;
    const auto control = fp::make_value(ArithmeticMode::Ordinary, 0);
    for (const auto budget :
        {0.L, -1.L, std::numeric_limits<long double>::infinity(), std::numeric_limits<long double>::quiet_NaN()})
    {
        auto work = ordinary_work();
        EXPECT_THROW((void) SimpleFluid::geometry::swept_volume::make_sweep(7, face.nodes, face.traces, control, work, budget), std::invalid_argument);
    }
    auto work = ordinary_work();
    EXPECT_THROW((void) SimpleFluid::geometry::swept_volume::make_sweep(7, face.nodes, face.traces, control, work), std::invalid_argument);
}

TEST(OrdinaryVoFSweep, CanonicalTriangleTraversalBorrowsRecipeAndPreservesMixedNodeOrder)
{
    const std::array<std::uint64_t, 4> nodes{9, 3, 7, 2};
    const std::vector<Point> current{Point({0, 0, 0}), Point({1, 0, 0}), Point({1, 1, 0}), Point({0, 1, 0})};
    const std::vector<Point> departure{
        Point({0, 0, -.25L}), Point({1, 0, -.25L}), Point({1, 1, -.25L}), Point({0, 1, -.25L})};
    const Point backcenter({.5L, .5L, -.25L});
    const std::array<const Point*, 9> points{&current[0], &current[1], &current[2], &current[3],
        &departure[0], &departure[1], &departure[2], &departure[3], &backcenter};
    // Frozen canonical recipe for this mixed ascending/descending node sequence.
    const std::array<std::array<std::size_t, 3>, 12> expected{{
        {8, 5, 4}, {1, 0, 4}, {1, 4, 5}, {8, 6, 5}, {1, 5, 6}, {1, 6, 2},
        {8, 7, 6}, {3, 2, 6}, {3, 6, 7}, {8, 4, 7}, {3, 7, 4}, {3, 4, 0}}};
    std::size_t ordinal = 0;
    SimpleFluid::geometry::swept_volume::visit_sweep_triangles(nodes, current, departure, backcenter,
        [&](const Point& a, const Point& b, const Point& c)
        {
            ASSERT_LT(ordinal, expected.size());
            EXPECT_EQ(&a, points[expected[ordinal][0]]);
            EXPECT_EQ(&b, points[expected[ordinal][1]]);
            EXPECT_EQ(&c, points[expected[ordinal][2]]);
            ++ordinal;
        });
    EXPECT_EQ(ordinal, expected.size());
}

TEST(OrdinaryVoFSweep, TranslatedUnitSquareRetainsAnalyticSignedPrismVolume)
{
    const std::array<std::uint64_t, 4> nodes{9, 3, 7, 2};
    for (const long double distance : {-.25L, 0.L, .25L})
    {
        auto work = ordinary_work();
        std::map<std::uint64_t, std::pair<Point, Point>> traces;
        const std::array<std::array<long double, 3>, 4> coordinates{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}};
        for (std::size_t i = 0; i < nodes.size(); ++i)
        {
            auto displaced = coordinates[i];
            displaced[2] -= distance;
            traces.emplace(nodes[i], std::pair{Point(coordinates[i]), Point(displaced)});
        }
        const auto sweep = SimpleFluid::geometry::swept_volume::make_sweep(
            17, nodes, traces, fp::make_value(ArithmeticMode::Ordinary, distance), work, face_budget(1, work));
        EXPECT_EQ(sweep.control.value, distance);
        EXPECT_LE(std::abs(sweep.constructed_volume.value - distance), face_budget(1, work));
        if (distance == 0)
            EXPECT_TRUE(sweep.pieces.empty());
        else
        {
            ASSERT_FALSE(sweep.pieces.empty());
            long double volume = 0;
            for (const auto& piece : sweep.pieces)
                volume += piece.sign * piece.geometry->volume().value;
            EXPECT_LE(std::abs(volume - distance), face_budget(1, work));
        }
    }
}
} // namespace
