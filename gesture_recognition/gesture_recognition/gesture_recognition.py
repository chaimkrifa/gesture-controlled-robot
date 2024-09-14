import pickle
import cv2
import mediapipe as mp
import numpy as np
import rclpy  # ROS 2 client library for Python
from rclpy.node import Node
from std_msgs.msg import String  # Standard ROS 2 message type
from geometry_msgs.msg import Twist  # Import the Twist message type

# Load gesture model
model_dict = pickle.load(open('/home/vboxuser/ros2_ws/src/gesture_recognition/gesture_recognition/model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Gesture labels
labels_SHOW = {0: 'Stop', 1: 'Left', 2: 'Right', 3: 'Forward', 4: 'Backward'}
labels_MCU = {0: 's', 1: 'l', 2: 'r', 3: 'f', 4: 'b'}

linearX = {0: 0, 1: 0, 2: 0, 3: 0.5, 4: -0.5}
linearY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
linearZ = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
angularX = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
angularY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
angularZ = {0: 0, 1: -0.5, 2: 0.5, 3: 0, 4: 0}

# ROS 2 Node for publishing gestures
class GesturePublisher(Node):
    def __init__(self):
        super().__init__('gesture_publisher')
        # Create a publisher for Twist messages
        self.publisher_ = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
        self.cap = cv2.VideoCapture(0)

    def publish_gesture(self):
        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = self.cap.read()
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character_show = labels_SHOW[int(prediction[0])]
                predicted_character_MCU = labels_MCU[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character_show, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                print(predicted_character_MCU)

                # Publish gesture to ROS 2 topic
                twist_msg = Twist()
                twist_msg.linear.x = linearX[int(prediction[0])]
                twist_msg.linear.y = linearY[int(prediction[0])]
                twist_msg.linear.z = linearZ[int(prediction[0])]
                twist_msg.angular.x = angularX[int(prediction[0])]
                twist_msg.angular.y = angularY[int(prediction[0])]
                twist_msg.angular.z = angularZ[int(prediction[0])]

                # Publish the Twist message to ROS 2 topic
                self.publisher_.publish(twist_msg)
                self.get_logger().info(f'Published Twist message: {twist_msg}')

            cv2.imshow('Gesture Recognition', frame)
            cv2.waitKey(1)

        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    gesture_publisher = GesturePublisher()

    # Keep the node running
    rclpy.spin(gesture_publisher)

    # Clean up and shutdown
    gesture_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

