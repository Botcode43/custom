import pickle
import os

DB_FILE = "gestures.pkl"  # Database file to store gestures

def save_gesture(name, landmarks):
    """
    Saves a new gesture to the database.
    :param name: Name of the gesture (e.g., "Hi", "Thumbs Up").
    :param landmarks: Hand landmark coordinates.
    """
    gestures = load_gestures()  # Load existing gestures
    gestures[name] = landmarks  # Add new gesture

    with open(DB_FILE, "wb") as f:
        pickle.dump(gestures, f)  # Save to file

    print(f"Gesture '{name}' saved successfully!")


def load_gestures():
    """
    Loads saved gestures from the database.
    :return: Dictionary of stored gestures.
    """
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}  # Return empty dict if no file exists


def clear_gestures():
    """
    Clears all stored gestures.
    """
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("All gestures have been deleted.")
    else:
        print("No stored gestures found.")

