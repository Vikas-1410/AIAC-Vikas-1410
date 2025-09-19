import csv
import io
from typing import Dict, Tuple, List
from collections import defaultdict

def parse_streaming_telemetry(raw_text: str) -> Tuple[Dict[str, float], float]:
    """
    Parse raw streaming telemetry CSV data and compute per-viewer averages.
    
    Handles flaky connectivity issues:
    - Truncated lines
    - Non-numeric values
    - Missing fields
    - Malformed CSV entries
    
    Args:
        raw_text (str): Raw CSV text with multiple lines in format 'id,timestamp,watch_time'
        
    Returns:
        Tuple[Dict[str, float], float]: 
            - Dictionary mapping viewer_id to average watch_time
            - Overall average watch_time across all viewers
    """
    
    # Track data per viewer
    viewer_data = defaultdict(list)
    total_watch_time = 0.0
    total_valid_entries = 0
    
    # Split into lines and process each
    lines = raw_text.strip().split('\n')
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Split by comma, handling potential issues
        try:
            parts = line.split(',')
            
            # Check if we have at least 3 parts (id, timestamp, watch_time)
            if len(parts) < 3:
                print(f"Warning: Line {line_num} truncated or incomplete: '{line}'")
                continue
                
            viewer_id = parts[0].strip()
            timestamp = parts[1].strip()
            watch_time_str = parts[2].strip()
            
            # Validate viewer_id is not empty
            if not viewer_id:
                print(f"Warning: Line {line_num} has empty viewer ID: '{line}'")
                continue
            
            # Try to parse watch_time as float
            try:
                watch_time = float(watch_time_str)
                
                # Validate watch_time is reasonable (non-negative)
                if watch_time < 0:
                    print(f"Warning: Line {line_num} has negative watch_time: {watch_time}")
                    continue
                    
                # Store the valid data
                viewer_data[viewer_id].append(watch_time)
                total_watch_time += watch_time
                total_valid_entries += 1
                
            except ValueError:
                print(f"Warning: Line {line_num} has non-numeric watch_time '{watch_time_str}': '{line}'")
                continue
                
        except Exception as e:
            print(f"Warning: Line {line_num} caused parsing error: {e} - '{line}'")
            continue
    
    # Compute per-viewer averages
    viewer_averages = {}
    for viewer_id, watch_times in viewer_data.items():
        if watch_times:  # Safety check
            viewer_averages[viewer_id] = sum(watch_times) / len(watch_times)
    
    # Compute overall average
    overall_average = total_watch_time / total_valid_entries if total_valid_entries > 0 else 0.0
    
    return viewer_averages, overall_average


def generate_dashboard_summary(viewer_averages: Dict[str, float], overall_average: float) -> Dict:
    """
    Generate dashboard-ready summary statistics.
    
    Args:
        viewer_averages: Dictionary of viewer_id to average watch_time
        overall_average: Overall average watch_time
        
    Returns:
        Dictionary with summary statistics for dashboards and alerts
    """
    
    if not viewer_averages:
        return {
            "total_viewers": 0,
            "overall_average": 0.0,
            "min_average": 0.0,
            "max_average": 0.0,
            "alert_thresholds": {
                "low_engagement": 0,
                "high_engagement": 0
            }
        }
    
    watch_times = list(viewer_averages.values())
    
    summary = {
        "total_viewers": len(viewer_averages),
        "overall_average": round(overall_average, 2),
        "min_average": round(min(watch_times), 2),
        "max_average": round(max(watch_times), 2),
        "median_average": round(sorted(watch_times)[len(watch_times)//2], 2),
        "alert_thresholds": {
            "low_engagement": len([wt for wt in watch_times if wt < overall_average * 0.5]),
            "high_engagement": len([wt for wt in watch_times if wt > overall_average * 1.5])
        }
    }
    
    return summary


# Example usage and test cases
if __name__ == "__main__":
    # Test with sample data including various error conditions
    sample_data = """user1,2024-01-01T10:00:00,120.5
user2,2024-01-01T10:01:00,85.2
user1,2024-01-01T10:02:00,95.0
user3,2024-01-01T10:03:00,200.1
user2,2024-01-01T10:04:00,invalid_time
user4,2024-01-01T10:05:00,150.0
user1,2024-01-01T10:06:00,110.3
truncated_line,2024-01-01T10:07:00
user5,2024-01-01T10:08:00,-50.0
user6,2024-01-01T10:09:00,75.8
user2,2024-01-01T10:10:00,90.5
user7,2024-01-01T10:11:00,180.2
user8,2024-01-01T10:12:00,abc
user9,2024-01-01T10:13:00,160.7
user10,2024-01-01T10:14:00,140.0"""
    
    print("=== Streaming Telemetry Parser Test ===\n")
    
    # Parse the data
    viewer_averages, overall_average = parse_streaming_telemetry(sample_data)
    
    print("Per-viewer averages:")
    for viewer_id, avg_time in sorted(viewer_averages.items()):
        print(f"  {viewer_id}: {avg_time:.2f} seconds")
    
    print(f"\nOverall average: {overall_average:.2f} seconds")
    
    # Generate dashboard summary
    summary = generate_dashboard_summary(viewer_averages, overall_average)
    
    print("\n=== Dashboard Summary ===")
    print(f"Total viewers: {summary['total_viewers']}")
    print(f"Overall average: {summary['overall_average']} seconds")
    print(f"Min average: {summary['min_average']} seconds")
    print(f"Max average: {summary['max_average']} seconds")
    print(f"Median average: {summary['median_average']} seconds")
    print(f"Low engagement viewers (< 50% of avg): {summary['alert_thresholds']['low_engagement']}")
    print(f"High engagement viewers (> 150% of avg): {summary['alert_thresholds']['high_engagement']}")
