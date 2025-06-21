# src/core/utils/performance_analyzer.py

import time
import matplotlib.pyplot as plt
import pandas as pd
import os


class PerformanceAnalyzer:
    """
    A utility class to analyze and visualize performance improvements from caching.
    """

    def __init__(self):
        self.results = []

    def add_result(self, video_name, cached, total_frames, total_time, detection_time):
        """
        Add a performance result.

        Args:
            video_name: Name of the processed video
            cached: Whether cache was used (True/False)
            total_frames: Total number of frames processed
            total_time: Total processing time in seconds
            detection_time: Time spent on YOLO detection in seconds
        """
        self.results.append({
            'video_name': video_name,
            'cached': cached,
            'total_frames': total_frames,
            'total_time': total_time,
            'detection_time': detection_time,
            'fps': total_frames / total_time,
            'detection_percentage': (detection_time / total_time) * 100 if total_time > 0 else 0
        })

    def get_results_df(self):
        """
        Get the results as a DataFrame.
        """
        return pd.DataFrame(self.results)

    def print_summary(self):
        """
        Print a summary of the performance results.
        """
        df = self.get_results_df()

        if len(df) == 0:
            print("No performance data available.")
            return

        # Group by whether cache was used, only compute mean for numeric columns
        try:
            # Use numeric_only=True to avoid errors with string columns
            grouped = df.groupby('cached').mean(numeric_only=True)

            print("\n===== Performance Summary =====")
            if True in grouped.index:
                print(f"With cache:")
                print(f"  Average FPS: {grouped.loc[True, 'fps']:.2f}")
                print(f"  Average processing time per frame: {1000 / grouped.loc[True, 'fps']:.2f} ms")

            if False in grouped.index:
                print(f"Without cache:")
                print(f"  Average FPS: {grouped.loc[False, 'fps']:.2f}")
                print(f"  Average processing time per frame: {1000 / grouped.loc[False, 'fps']:.2f} ms")

            # Calculate improvement if both cached and non-cached results exist
            if True in grouped.index and False in grouped.index:
                speedup = grouped.loc[True, 'fps'] / grouped.loc[False, 'fps']
                time_saved_percent = (1 - grouped.loc[True, 'detection_time'] / grouped.loc[
                    False, 'detection_time']) * 100

                print(f"\nPerformance improvement:")
                print(f"  Speed-up factor: {speedup:.2f}x")
                print(f"  Time saved on detection: {time_saved_percent:.2f}%")
                print(f"  Processing time reduction: {(1 - 1 / speedup) * 100:.2f}%")
        except Exception as e:
            print(f"Error generating performance summary: {str(e)}")

            # Fallback to simple summary
            cached_results = df[df['cached'] == True]
            non_cached_results = df[df['cached'] == False]

            if not cached_results.empty:
                print("\nWith cache:")
                for _, row in cached_results.iterrows():
                    print(f"  Video: {row['video_name']}")
                    print(f"  Frames: {row['total_frames']}")
                    print(f"  Total time: {row['total_time']:.2f} seconds")
                    print(f"  FPS: {row['fps']:.2f}")

            if not non_cached_results.empty:
                print("\nWithout cache:")
                for _, row in non_cached_results.iterrows():
                    print(f"  Video: {row['video_name']}")
                    print(f"  Frames: {row['total_frames']}")
                    print(f"  Total time: {row['total_time']:.2f} seconds")
                    print(f"  FPS: {row['fps']:.2f}")

    def plot_comparison(self, output_path=None):
        """
        Generate comparison plots.

        Args:
            output_path: Path to save the plot. If None, the plot will be displayed.
        """
        df = self.get_results_df()

        if len(df) == 0:
            print("No performance data available for plotting.")
            return

        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # For pivot tables, we need to ensure the video_name is used as index
            # and we only plot numeric columns

            # FPS comparison
            fps_data = df.pivot_table(index='video_name', columns='cached', values='fps')
            fps_data.plot(kind='bar', ax=ax1)
            ax1.set_title('FPS Comparison: Cached vs. Non-cached')
            ax1.set_ylabel('Frames Per Second')
            ax1.set_xlabel('Video')
            ax1.legend(['Non-cached', 'Cached'])

            # Detection time comparison
            time_data = df.pivot_table(index='video_name', columns='cached', values='detection_time')
            time_data.plot(kind='bar', ax=ax2)
            ax2.set_title('Detection Time (seconds)')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_xlabel('Video')
            ax2.legend(['Non-cached', 'Cached'])

            plt.tight_layout()

            if output_path:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                print(f"Plot saved to {output_path}")
            else:
                plt.show()
        except Exception as e:
            print(f"Error generating performance plot: {str(e)}")
