import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector_node')
        
        # 1. Load the AI Model (Nano version for speed)
        self.get_logger().info("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt') 
        self.get_logger().info("YOLO Model Loaded!")

        # 2. Create the subscriber (Listen to the camera)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # This must match your topic name
            self.listener_callback,
            10)
        
        # 3. Create the translator
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            # 4. Convert ROS Image -> OpenCV Image
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 5. Run the AI on the frame
            # We use verbose=False to keep the terminal clean
            results = self.model(frame, verbose=False)

            # 6. Process results
            for result in results:
                # Get the list of detected classes (names of objects)
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Print what we found
                    self.get_logger().info(f"I see a {class_name} ({confidence:.2f})")

            # Optional: Show what the robot sees in a window
            annotated_frame = results[0].plot()
            cv2.imshow("Robot View", annotated_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
