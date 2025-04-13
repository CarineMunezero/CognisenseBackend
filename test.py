import random
import time

def generate_sample_data():
    """
    Continuously generate lines of sample data.
    Each line contains 5 floating-point numbers formatted to 6 decimal places,
    separated by tabs.
    """
    while True:
        # Generate 5 random values.
        # Adjust the range as needed to mimic your real sensor data.
        values = [random.uniform(-150, 350) for _ in range(5)]
        # Format each value to 6 decimal places and join them with a tab.
        line = "\t".join(f"{v:.6f}" for v in values)
        print(line)
        # Adjust sleep time for your desired update frequency.
        time.sleep(0.1)

if __name__ == "__main__":
    generate_sample_data()

