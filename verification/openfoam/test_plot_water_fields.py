#!/usr/bin/env python3
"""Focused checks for physical cell matching and zero-reference error plots."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from compare_verification import ComparisonError
from plot_water_fields import COLUMNS, errors, generate, load_fields, relative_error


class WaterFieldFigures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {"schema_version": 1, "case": "planarALE", "mode": "transient",
                         "expected_times_s": [0, 1], "time_tolerance_s": 1e-10,
                         "spatial_output": {"cell_samples": ["0", "1"], "width_m": 1}}
        self.rows = [{"time_s": t, "sample": str(i), "z_lower_m": i/2, "z_upper_m": (i+1)/2,
                      "temperature_K": 300+t, "density_kg_m3": 996-t,
                      "alpha_g": 0, "ux_m_s": 0, "uy_m_s": 0, "uz_m_s": 0}
                     for t in (0, 1) for i in (0, 1)]
        self.of = self.write("openfoam.csv", self.rows)
        self.sf = self.write("simplefluid.csv", self.rows)

    def write(self, name, rows):
        path = self.root/name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_zero_reference_is_undefined_even_when_both_zero(self):
        self.assertIsNone(relative_error(0, 0, 1e-12))
        self.assertIsNone(relative_error(1e-13, 2e-13, 1e-12))
        self.assertEqual(relative_error(2, 1, 1e-12), -50)

    def test_reversed_velocity_is_not_hidden_by_equal_speed(self):
        reference = {"temperature_K": 300, "density_kg_m3": 996, "alpha_g": 0,
                     "speed_m_s": 1, "ux_m_s": 0, "uy_m_s": 0, "uz_m_s": 1}
        actual = {**reference, "uz_m_s": -1}
        self.assertEqual(errors(reference, actual)["speed_m_s"]["relative_error_percent"], 200)

    def test_two_truncated_histories_cannot_pass(self):
        self.write("openfoam.csv", self.rows[:-1])
        self.write("simplefluid.csv", self.rows[:-1])
        with self.assertRaisesRegex(ComparisonError, "missing"):
            load_fields(self.manifest, self.of, self.sf)

    def test_cell_gap_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["z_lower_m"] += 0.1
        self.write("simplefluid.csv", rows)
        with self.assertRaisesRegex(ComparisonError, "gap"):
            load_fields(self.manifest, self.of, self.sf)

    def test_different_mesh_extent_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["z_upper_m"] += 0.1
        self.write("simplefluid.csv", rows)
        with self.assertRaisesRegex(ComparisonError, "extents"):
            load_fields(self.manifest, self.of, self.sf)

    def test_physical_gas_fraction_is_checked(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["alpha_g"] = 1.1
        self.write("simplefluid.csv", rows)
        with self.assertRaisesRegex(ComparisonError, "gas volume"):
            load_fields(self.manifest, self.of, self.sf)

    def test_zero_fields_produce_valid_figures_and_blank_relative_csv(self):
        reference, actual = load_fields(self.manifest, self.of, self.sf)
        output = self.root/"figures"
        result = generate(self.manifest, reference, actual, output, ["svg"])
        self.assertEqual(result["fields"]["alpha_g"]["undefined_relative_samples"], 4)
        self.assertIsNone(result["fields"]["alpha_g"]["max_absolute_relative_error_percent"])
        for name in ["distribution", "relative", "absolute", "history_distribution", "history_relative"]:
            ET.parse(output/f"{name}.svg")
        with (output/"matched_fields.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["alpha_g_relative_error_percent"], "")
        self.assertEqual(json.loads((output/"statistics.json").read_text())["status"], "diagnostic")

    def planar_rows(self):
        self.manifest["case"] = "bottomHeatedBubblyConvection"
        self.manifest["spatial_output"].update(layout="xz", cell_samples=["0", "1", "2", "3"])
        return [{"time_s": t, "sample": str(i), "x_lower_m": (i%2)/2, "x_upper_m": (i%2+1)/2,
                 "z_lower_m": (i//2)/2, "z_upper_m": (i//2+1)/2, "temperature_K": 300+t+i,
                 "density_kg_m3": 996-t, "alpha_g": .001*i, "ux_m_s": .001, "uy_m_s": 0,
                 "uz_m_s": .01 if i%2 else -.01} for t in (0,1) for i in range(4)]

    def test_planar_circulation_fields_render_all_cells(self):
        rows = self.planar_rows()
        self.write("openfoam.csv", rows)
        self.write("simplefluid.csv", rows)
        reference, actual = load_fields(self.manifest, self.of, self.sf)
        result = generate(self.manifest, reference, actual, self.root/"planar", ["svg"])
        self.assertEqual(result["matched_cells"], 8)
        self.assertEqual(result["fields"]["speed_m_s"]["max_absolute_error"], 0)
        for path in (self.root/"planar").glob("*.svg"):
            ET.parse(path)

    def test_horizontal_hole_is_rejected(self):
        rows = self.planar_rows()
        for row in rows:
            if row["x_upper_m"] == .5:
                row["x_upper_m"] = .4
        self.write("openfoam.csv", rows)
        self.write("simplefluid.csv", rows)
        with self.assertRaisesRegex(ComparisonError, "horizontal gap"):
            load_fields(self.manifest, self.of, self.sf)

    def test_matching_wrong_meshes_cannot_bypass_declared_coordinates(self):
        self.manifest['mesh_edges_m']={'x':[0,1],'y':[0,1],'z':[0,.4,1]}
        with self.assertRaisesRegex(ComparisonError,'shared reference mesh'):
            load_fields(self.manifest,self.of,self.sf)


if __name__ == "__main__":
    unittest.main()
