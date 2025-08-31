# Camera Debug and Detection Utilities

import cv2
import platform
import sys


class CameraDebugUtils:
    """Utility class for debugging camera issues and finding available cameras."""

    @staticmethod
    def list_all_cameras(max_cameras=10):
        """
        List all available cameras on the system.

        Args:
            max_cameras (int): Maximum number of cameras to check

        Returns:
            list: List of working camera indices
        """
        available_cameras = []

        print("Scanning for available cameras...")
        print("-" * 50)

        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        available_cameras.append({
                            'index': i,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'working': True
                        })

                        print(f"Camera {i}: ✓ Working - Resolution: {width}x{height}, FPS: {fps}")
                    else:
                        print(f"Camera {i}: ✗ Found but not working (no frame)")
                else:
                    print(f"Camera {i}: ✗ Cannot open")

                cap.release()

            except Exception as e:
                print(f"Camera {i}: ✗ Error - {str(e)}")

        print("-" * 50)
        print(f"Total working cameras found: {len(available_cameras)}")

        return available_cameras

    @staticmethod
    def test_camera_with_backends(camera_index):
        """
        Test camera with different backends to find the working one.

        Args:
            camera_index (int): Camera index to test

        Returns:
            dict: Test results for different backends
        """
        backends = [
            (cv2.CAP_DSHOW, "DirectShow (Windows)"),
            (cv2.CAP_MSMF, "Microsoft Media Foundation"),
            (cv2.CAP_V4L2, "Video4Linux2 (Linux)"),
            (cv2.CAP_ANY, "Any available backend")
        ]

        results = {}

        print(f"Testing camera {camera_index} with different backends...")
        print("-" * 50)

        for backend_id, backend_name in backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        results[backend_name] = "✓ Working"
                        print(f"{backend_name}: ✓ Working")
                    else:
                        results[backend_name] = "✗ Opens but no frame"
                        print(f"{backend_name}: ✗ Opens but no frame")
                else:
                    results[backend_name] = "✗ Cannot open"
                    print(f"{backend_name}: ✗ Cannot open")

                cap.release()

            except Exception as e:
                results[backend_name] = f"✗ Error: {str(e)}"
                print(f"{backend_name}: ✗ Error - {str(e)}")

        return results

    @staticmethod
    def get_system_info():
        """Get system information that might affect camera access."""
        info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': sys.version,
            'opencv_version': cv2.__version__
        }

        print("System Information:")
        print("-" * 30)
        for key, value in info.items():
            print(f"{key}: {value}")

        return info

    @staticmethod
    def improved_camera_test(camera_index, timeout=5):
        """
        Improved camera test with timeout and multiple attempts.

        Args:
            camera_index (int): Camera index to test
            timeout (int): Timeout in seconds

        Returns:
            dict: Test results
        """
        result = {
            'success': False,
            'error': None,
            'properties': {},
            'frame_captured': False
        }

        try:
            # Try with DirectShow backend first (Windows)
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(camera_index)

            if not cap.isOpened():
                result['error'] = "Cannot open camera"
                return result

            # Set timeout for frame capture
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Try to read frame multiple times
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    result['frame_captured'] = True
                    break

                # Wait a bit between attempts
                cv2.waitKey(100)

            if result['frame_captured']:
                # Get camera properties
                result['properties'] = {
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
                    'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
                    'saturation': cap.get(cv2.CAP_PROP_SATURATION)
                }
                result['success'] = True
            else:
                result['error'] = "Camera opens but cannot capture frame"

            cap.release()

        except Exception as e:
            result['error'] = f"Exception: {str(e)}"

        return result


# Enhanced camera test function for integration
def enhanced_camera_test(camera_index=0):
    """
    Enhanced camera test function that can be integrated into the main application.

    Args:
        camera_index (int): Camera index to test

    Returns:
        bool: True if camera is working, False otherwise
    """
    debug_utils = CameraDebugUtils()

    # Get system info
    debug_utils.get_system_info()
    print()

    # Test specific camera
    result = debug_utils.improved_camera_test(camera_index)

    print(f"Camera {camera_index} test result:")
    print(f"Success: {result['success']}")
    if result['error']:
        print(f"Error: {result['error']}")
    if result['properties']:
        print(f"Properties: {result['properties']}")

    return result['success']


# Function to find the best camera
def find_best_camera():
    """
    Find the best available camera.

    Returns:
        int: Index of the best camera, or -1 if none found
    """
    debug_utils = CameraDebugUtils()
    available_cameras = debug_utils.list_all_cameras()

    if available_cameras:
        # Return the first working camera
        return available_cameras[0]['index']

    return -1


# Usage example
if __name__ == "__main__":
    # List all available cameras
    debug_utils = CameraDebugUtils()
    available_cameras = debug_utils.list_all_cameras()

    if available_cameras:
        print(f"\nTesting first available camera (index {available_cameras[0]['index']}):")
        debug_utils.test_camera_with_backends(available_cameras[0]['index'])
    else:
        print("No cameras found!")

    # Test specific camera
    print(f"\nTesting camera 2 specifically:")
    enhanced_camera_test(2)