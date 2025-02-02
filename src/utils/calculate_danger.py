import os
from dotenv import load_dotenv


def calculate_danger(tracking_data, danger_threshold=5, standup_threshold=3, help_distance_threshold=50):
    """
    Calculate the danger score based on tracking data.

    Parameters:
    tracking_data: List[List[int]] - The tracking data for each frame.
    danger_threshold: int - The number of frames a person needs to be on the ground to count as a high danger situation.
    standup_threshold: int - The number of frames a person needs to stay not falling to reduce danger.
    help_distance_threshold: int - The distance threshold between a falling person and others to consider it as help.

    Returns:
    int - The calculated danger score.
    """

    danger_score = 0
    fall_duration = []  # Track how long each person has been falling
    standup_duration = []  # Track how long each person has been standing
    max_people = max([len(frame) for frame in tracking_data])  # Find the maximum number of people detected

    # Initialize fall and standup duration counters for each person
    for _ in range(max_people):
        fall_duration.append(0)
        standup_duration.append(0)

    for frame in tracking_data:
        num_falling = sum(frame)

        # Rule 1: If more than one person is falling, add more danger
        if num_falling == 1:
            danger_score += 1
        elif num_falling > 1:
            danger_score += 2

        # Rule 2: Check fall duration for each person
        for i in range(len(frame)):
            if frame[i] == 1:
                fall_duration[i] += 1
                standup_duration[i] = 0  # Reset standup duration if falling

                # If person has been falling for too long, increase danger score
                if fall_duration[i] >= danger_threshold:
                    danger_score += 1
            else:
                standup_duration[i] += 1  # Increment standup duration if not falling
                fall_duration[i] = 0  # Reset fall duration if standing

                # If person has been standing for a while, reduce danger score
                if standup_duration[i] >= standup_threshold:
                    danger_score = max(0, danger_score - 1)  # Danger score can't go below 0

        # Rule 3: Check for help (if there are other people close to the falling person)
        for i in range(len(frame)):
            if frame[i] == 1:  # Only check for help if the person is falling
                for j in range(len(frame)):
                    if i != j and frame[j] == 0:  # Another person is nearby and not falling
                        # Here you can add logic to check the distance between person i and j (e.g., using bbox coordinates)
                        # For now, we'll assume if another person is present and not falling, they are close enough
                        danger_score = max(0, danger_score - 1)  # Reduce danger score if help is detected

    return danger_score


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Example of loading configuration values from environment variables
    DANGER_THRESHOLD = int(os.getenv("DANGER_THRESHOLD", 5))
    STANDUP_THRESHOLD = int(os.getenv("STANDUP_THRESHOLD", 3))
    HELP_DISTANCE_THRESHOLD = int(os.getenv("HELP_DISTANCE_THRESHOLD", 50))

    # Example tracking data
    example_tracking_data = [
        [1, 0, 0],  # Frame 1
        [1, 1, 0],  # Frame 2
        [0, 1, 0],  # Frame 3
    ]

    # Calculate danger score
    score = calculate_danger(example_tracking_data, DANGER_THRESHOLD, STANDUP_THRESHOLD, HELP_DISTANCE_THRESHOLD)
    print(f"Calculated Danger Score: {score}")
