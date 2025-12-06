import sys
sys.path.append('src')

import unittest
import polars as pl
import matplotlib.pyplot as plt
from visualization import create_football_field, animate_play
import os

class TestViz(unittest.TestCase):
    def test_field_plot(self):
        fig, ax = create_football_field()
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
    def test_animation_function(self):
        # Create dummy play data
        data = {
            "frame_id": [1, 1, 2, 2],
            "std_x": [50, 60, 51, 61],
            "std_y": [20, 30, 21, 31],
            "player_name": ["P1", "P2", "P1", "P2"],
            "club": ["A", "B", "A", "B"]
        }
        df = pl.DataFrame(data)
        
        # Test if it runs without error (mock save?)
        # For CI/CD usually we mock Animation.save
        # Here we can just try running it to a temp file
        
        try:
            output = animate_play(df, "test_anim.mp4")
            self.assertTrue(os.path.exists(output))
            os.remove(output)
        except Exception as e:
            # FFMpeg might not be installed in this env
            print(f"Animation failed: {e}")
            if "ffmpeg" not in str(e).lower():
                self.fail(f"Animation code error: {e}")

if __name__ == '__main__':
    unittest.main()
