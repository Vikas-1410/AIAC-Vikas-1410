from typing import Dict, Optional, Tuple, Union
from collections import defaultdict


class StreamWindow:
    """
    A data structure for managing a sliding window of key-value pairs.
    
    This class provides functionality to add, remove, and analyze data points
    in a stream processing context. It maintains a dictionary of data points
    and provides statistical summaries.
    
    Attributes:
        data (Dict[str, Union[float, int]]): Dictionary storing the data points
            with string IDs as keys and numeric values.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty StreamWindow.
        
        Creates a new instance with an empty data dictionary.
        """
        self.data: Dict[str, Union[float, int]] = {}
    
    def add(self, id: str, value: Union[float, int]) -> None:
        """
        Add or update a data point in the stream window.
        
        Args:
            id (str): Unique identifier for the data point.
            value (Union[float, int]): Numeric value to store.
            
        Raises:
            ValueError: If the value is not numeric or if id is empty.
        """
        if not isinstance(id, str) or not id.strip():
            raise ValueError("ID must be a non-empty string")
        
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be a number (int or float)")
        
        self.data[id] = value
    
    def remove(self, id: str) -> bool:
        """
        Remove a data point from the stream window.
        
        Args:
            id (str): Unique identifier of the data point to remove.
            
        Returns:
            bool: True if the data point was removed, False if it didn't exist.
        """
        if id in self.data:
            del self.data[id]
            return True
        return False
    
    def get(self, id: str) -> Optional[Union[float, int]]:
        """
        Retrieve a data point by its ID.
        
        Args:
            id (str): Unique identifier of the data point.
            
        Returns:
            Optional[Union[float, int]]: The value if found, None otherwise.
        """
        return self.data.get(id)
    
    def contains(self, id: str) -> bool:
        """
        Check if a data point exists in the stream window.
        
        Args:
            id (str): Unique identifier to check.
            
        Returns:
            bool: True if the data point exists, False otherwise.
        """
        return id in self.data
    
    def summary(self) -> Tuple[int, Optional[float]]:
        """
        Calculate summary statistics for the current data.
        
        Returns:
            Tuple[int, Optional[float]]: A tuple containing:
                - count (int): Number of data points
                - average (Optional[float]): Average value, or None if empty
        """
        count = len(self.data)
        if count == 0:
            return (0, None)
        
        total = sum(self.data.values())
        average = total / count
        return (count, average)
    
    def clear(self) -> None:
        """
        Remove all data points from the stream window.
        """
        self.data.clear()
    
    def size(self) -> int:
        """
        Get the current number of data points.
        
        Returns:
            int: Number of data points in the stream window.
        """
        return len(self.data)
    
    def keys(self) -> list[str]:
        """
        Get all data point IDs.
        
        Returns:
            list[str]: List of all data point identifiers.
        """
        return list(self.data.keys())
    
    def values(self) -> list[Union[float, int]]:
        """
        Get all data point values.
        
        Returns:
            list[Union[float, int]]: List of all data point values.
        """
        return list(self.data.values())


# Usage Example
if __name__ == "__main__":
    # Create a new stream window
    window = StreamWindow()
    
    print("=== StreamWindow Usage Example ===\n")
    
    # Add some data points
    print("Adding data points...")
    window.add("sensor_1", 25.5)
    window.add("sensor_2", 30.0)
    window.add("sensor_3", 22.8)
    window.add("sensor_4", 28.3)
    
    # Display current state
    print(f"Current size: {window.size()}")
    print(f"Data points: {window.keys()}")
    print(f"Values: {window.values()}")
    
    # Get summary statistics
    count, average = window.summary()
    print(f"\nSummary - Count: {count}, Average: {average:.2f}")
    
    # Check for specific data points
    print(f"\nContains 'sensor_1': {window.contains('sensor_1')}")
    print(f"Value of 'sensor_2': {window.get('sensor_2')}")
    
    # Remove a data point
    print(f"\nRemoving 'sensor_3'...")
    removed = window.remove("sensor_3")
    print(f"Removed successfully: {removed}")
    
    # Updated summary
    count, average = window.summary()
    print(f"Updated summary - Count: {count}, Average: {average:.2f}")
    
    # Error handling example
    print("\n=== Error Handling Examples ===")
    try:
        window.add("", 10.0)  # Empty ID
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        window.add("sensor_5", "invalid")  # Non-numeric value
    except ValueError as e:
        print(f"Error: {e}")
    
    # Clear all data
    print(f"\nClearing all data...")
    window.clear()
    count, average = window.summary()
    print(f"After clear - Count: {count}, Average: {average}")
