# tests/test_scad_generation.py

import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Corrected solid2 imports
from solid2 import sphere, cylinder, polyhedron, union, translate, rotate, color
from solid2.core.object_base import OpenSCADObject # OpenSCADObject is in object_base
from solid2.core.object_base import module, ref, comment, empty # If you need to patch these specifically
from solid2.core.utils import Code # If you need to patch Code specifically


# Updated imports: Now import from your package
from scad import export_projections_to_scad, export_cross_sections_to_scad
from structure import Hypercube # Import Hypercube for type hinting and potential direct use


class TestScadGeneration(unittest.TestCase):

    def setUp(self):
        # Define dummy filenames for testing
        self.proj_output_file = "test_projections.scad"
        self.cross_output_file = "test_cross_sections.scad"

        # Ensure no test files exist from previous runs
        if os.path.exists(self.proj_output_file):
            os.remove(self.proj_output_file)
        if os.path.exists(self.cross_output_file):
            os.remove(self.cross_output_file)

    def tearDown(self):
        # Clean up generated files after each test
        if os.path.exists(self.proj_output_file):
            os.remove(self.proj_output_file)
        if os.path.exists(self.cross_output_file):
            os.remove(self.cross_output_file)

    @patch('solid2.core.object_base.OpenSCADObject.save_as_scad')
    def test_export_projections_creates_file(self, mock_save_as_scad):
        """
        Test that export_projections_to_scad attempts to save an SCAD file.
        We mock save_as_scad to prevent actual file creation during this test,
        but confirm it was called.
        """
        export_projections_to_scad(
            output_filename=self.proj_output_file,
            hypercube_dimension=3,
            num_frames=2
        )
        mock_save_as_scad.assert_called_once_with(self.proj_output_file)
        # Verify that the print statement for success is called
        # (This is more for confirming the flow than strict testing)
        # Note: Capturing stdout for print statements is more complex with unittest,
        # so we'll rely on the mock call for file saving.

    @patch('solid2.core.object_base.OpenSCADObject.save_as_scad')
    def test_export_cross_sections_creates_file(self, mock_save_as_scad):
        """
        Test that export_cross_sections_to_scad attempts to save an SCAD file.
        """
        export_cross_sections_to_scad(
            output_filename=self.cross_output_file,
            hypercube_dimension=3,
            num_frames=2
        )
        mock_save_as_scad.assert_called_once_with(self.cross_output_file)

    def test_generated_projection_scad_content_basics(self):
        """
        Test that the generated projection SCAD file contains expected basic elements
        like global parameters and module definitions. This requires actual file generation.
        """
        export_projections_to_scad(
            output_filename=self.proj_output_file,
            hypercube_dimension=3,
            num_frames=2,
            projection_radius=0.01
        )
        self.assertTrue(os.path.exists(self.proj_output_file))

        with open(self.proj_output_file, 'r') as f:
            content = f.read()

        self.assertIn("$fn = 50;", content)
        self.assertIn("wireframe_radius = 0.01;", content) # Check for radius being correctly inserted
        self.assertIn("module projection_frame_1() {", content)
        self.assertIn("module projection_frame_2() {", content)
        self.assertIn("current_frame = 1;", content)
        self.assertIn("render_all_projections = true;", content)
        self.assertIn("if (render_all_projections) {", content)

    def test_generated_cross_section_scad_content_basics(self):
        """
        Test that the generated cross-section SCAD file contains expected basic elements
        like global parameters and module definitions.
        """
        export_cross_sections_to_scad(
            output_filename=self.cross_output_file,
            hypercube_dimension=3,
            num_frames=2
        )
        self.assertTrue(os.path.exists(self.cross_output_file))

        with open(self.cross_output_file, 'r') as f:
            content = f.read()

        self.assertIn("$fn = 50;", content)
        self.assertIn("render_color = [0.8, 0.4, 0.2, 1.0];", content)
        self.assertIn("module cross_section_frame_1() {", content)
        self.assertIn("module cross_section_frame_2() {", content)
        self.assertIn("current_frame = 1;", content)
        self.assertIn("render_all_cross_sections = true;", content)
        self.assertIn("if (render_all_cross_sections) {", content)
        # Check that the color is applied within the module
        self.assertIn("color(render_color) {", content)

    @patch('structure.Hypercube.project')
    def test_projection_calls_hypercube_project(self, mock_project):
        """
        Verify that export_projections_to_scad correctly calls Hypercube.project.
        """
        # Mock the return of project to avoid complex geometry generation for this test
        mock_project.return_value = {
            0: MagicMock(coordinates=np.array([0.0, 0.0, 0.0])),
            1: MagicMock(coordinates=np.array([1.0, 0.0, 0.0]))
        }
        export_projections_to_scad(
            output_filename=self.proj_output_file,
            hypercube_dimension=4,
            num_frames=1,
            target_projection_dimension=3,
            projection_type='orthogonal'
        )
        mock_project.assert_called_once()
        args, kwargs = mock_project.call_args
        self.assertEqual(args[0], 3) # target_dimension
        self.assertEqual(args[1], 'orthogonal') # projection_type

    @patch('structure.Hypercube.cross_section')
    def test_cross_section_calls_hypercube_cross_section(self, mock_cross_section):
        """
        Verify that export_cross_sections_to_scad correctly calls Hypercube.cross_section.
        """
        # Mock the return of cross_section for simplicity
        # Need to return valid data for polyhedron to not fail immediately in solid2
        mock_cross_section.return_value = (
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), # vertices
            np.array([[0, 1, 2]]) # faces (a single triangle)
        )
        export_cross_sections_to_scad(
            output_filename=self.cross_output_file,
            hypercube_dimension=4,
            num_frames=1,
            slice_dimension=3,
            min_slice_value=0.0,
            max_slice_value=0.0
        )
        mock_cross_section.assert_called_once()
        args, kwargs = mock_cross_section.call_args
        self.assertEqual(args[0], 3) # hyperplane_dim
        # Check if hyperplane_value is within expected range (close to 0.0 for 1 frame)
        self.assertTrue(np.isclose(args[1], 0.0))

    def test_cross_section_handles_empty_result(self):
        """
        Test that cross-section generation handles cases where no geometry is found.
        This should result in an 'empty' solid2 object.
        """
        # Mock the cross_section method to return None, None
        with patch('structure.Hypercube.cross_section', return_value=(None, None)):
            export_cross_sections_to_scad(
                output_filename=self.cross_output_file,
                hypercube_dimension=4,
                num_frames=1,
                slice_dimension=3,
                min_slice_value=0.0,
                max_slice_value=0.0
            )
            self.assertTrue(os.path.exists(self.cross_output_file))
            with open(self.cross_output_file, 'r') as f:
                content = f.read()
            self.assertIn("Empty Cross-section Frame 1/1", content) # Check for the comment indicating empty

    def test_cross_section_handles_points_only_result(self):
        """
        Test that cross-section generation handles cases where only points/lines are found (no faces).
        This should result in spheres being rendered.
        """
        # Mock the cross_section method to return points but no faces
        with patch('structure.Hypercube.cross_section', return_value=(np.array([[0.0, 0.0, 0.0]]), None)):
            export_cross_sections_to_scad(
                output_filename=self.cross_output_file,
                hypercube_dimension=4,
                num_frames=1,
                slice_dimension=3,
                min_slice_value=0.0,
                max_slice_value=0.0
            )
            self.assertTrue(os.path.exists(self.cross_output_file))
            with open(self.cross_output_file, 'r') as f:
                content = f.read()
            self.assertIn("Points/Lines Cross-section Frame 1/1", content) # Check for the comment
            self.assertIn("sphere", content) # Check for sphere generation


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)