import sys
sys.path.append('src')

import unittest
import polars as pl
import numpy as np
from metrics import calculate_zone_collapse_speed, calculate_defensive_reaction_time

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        # 3 Defenders, moving inward (collapsing)
        self.defenders_df = pl.DataFrame({
            "frame_id": [1, 1, 1, 2, 2, 2],
            "nfl_id": [1, 2, 3, 1, 2, 3],
            "club": ["DEF", "DEF", "DEF", "DEF", "DEF", "DEF"],
            "std_x": [0, 10, 5, 2, 8, 5], # Frame 1: Width 10. Frame 2: Width 6. Area shrinks.
            "std_y": [0, 0, 10, 2, 2, 8],
            "a": [1.0, 1.0, 1.0, 5.0, 5.0, 5.0] # High acceleration change
        })
        
    def test_zone_collapse_speed(self):
        res = calculate_zone_collapse_speed(self.defenders_df, "DEF")
        print(f"Collapse Result: {res}")
        
        # Frame 1 to 2
        # Area 1: Triangle (0,0), (10,0), (5,10) -> Base=10, Height=10 -> Area=50
        # Area 2: Triangle (2,2), (8,2), (5,8) -> Base=6, Height=6 -> Area=18
        # Diff = 18 - 50 = -32
        # Rate = -32 / 0.1 = -320
        
        rate = res.filter(pl.col("frame_id") == 2)["hull_area_rate"][0]
        self.assertAlmostEqual(rate, -320.0, places=1)
        
    def test_reaction_time(self):
        # Peak jerk at Frame 2 (accel goes 1 -> 5)
        # release at frame 1
        res = calculate_defensive_reaction_time(self.defenders_df, 1, "DEF")
        print(f"Reaction Time: {res}")
        
        # Frame 2 is max diff. Frame 2 - Frame 1 = 1 frame = 0.1s
        self.assertAlmostEqual(res, 0.1, places=2)

if __name__ == '__main__':
    unittest.main()
