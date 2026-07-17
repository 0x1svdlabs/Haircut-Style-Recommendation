import unittest
import sys
import os

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from feature_extractor import extract_features
from recommender import recommend

class TestComponents(unittest.TestCase):
    
    def test_recommender_mapping(self):
        """
        Verify that correct hairstyles are recommended for known face shapes
        and that case insensitivity works.
        """
        # Test lowercase
        recos_oval = recommend("oval")
        self.assertEqual(len(recos_oval), 5)
        self.assertIn("Crew Cut", recos_oval)
        self.assertIn("Fade Cut", recos_oval)
        
        # Test case insensitivity
        recos_oval_caps = recommend("OVAL")
        self.assertEqual(recos_oval, recos_oval_caps)
        
        # Test fallback for unknown shapes
        recos_fallback = recommend("nonexistent_shape")
        self.assertTrue(len(recos_fallback) > 0)
        
    def test_feature_extractor_validation(self):
        """
        Verify that feature extractor returns None for invalid or incomplete landmark arrays.
        """
        # Test empty landmark list
        self.assertIsNone(extract_features([]))
        
        # Test list with insufficient landmarks (less than 468)
        self.assertIsNone(extract_features([(100, 100)] * 10))

if __name__ == '__main__':
    unittest.main()
    print("All component tests passed successfully!")
