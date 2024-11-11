So, I started by conducting research on how to approach the task. After gaining a solid understanding, I wrote my Python code using Visual Studio Code (VS Code). I installed the required extensions for Jupyter Notebook and Python in VS Code and then used the command prompt to download essential Python libraries such as numpy, pandas, opencv-python, tensorflow, and keras.

Next, I downloaded 12 videos from YouTube, representing 3 different gestures: clap, walk, and run (4 videos per gesture: 3 for training and 1 for testing). After downloading the videos, I converted them into AVI format for easy processing. From these videos, I created my dataset, normalized it, split the data for training and testing, trained the model, and finally tested it with an accuracy of 100%.

Here's how the code works:

Capturing frames and extracting pose landmarks:
We use OpenCV to open the video files and MediaPipe to detect pose landmarks. As each video frame is processed, the detected landmarks (such as key body joints) are saved into a CSV file along with the frame number and a label (e.g., clap, walk, or run). MediaPipe Pose detects 33 landmarks (e.g., elbows, knees), and each landmark's x, y, z coordinates, and visibility score are stored.

cap = cv2.VideoCapture('clap1.avi') ... csv_writer.writerow(headers)

Normalizing the data:
Once the raw landmark data is collected, we normalize it by referencing the left hip's position to ensure that the model focuses on relative body movements, ignoring the person's absolute position.

for i in range(33): df[f'x_{i}'] -= df['x_23'] # Normalize by left hip x-coordinate

Splitting the dataset:
The normalized data is split into training and testing sets using the train_test_split() method. This allows the model to learn from one portion of the data and test its accuracy on the other portion.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

Training the model:
A neural network using TensorFlow/Keras was built, which consists of multiple layers: Two dense layers with 64 and 32 neurons respectively, both using the ReLU activation function. The output layer uses the softmax activation function to predict one of the 3 gestures (clap, walk, run). After compiling the model, I trained it on the landmark data and achieved 100% accuracy on the test data.

model = tf.keras.models.Sequential([ tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(len(set(y)), activation='softmax') ])

Saving the model:
After training, the model is saved to a file, making it reusable for later testing or deployment.

model.save(model_path)

Testing the model:
In the final step, we load the trained model and use it to make predictions on new video data. For each frame in the test video, we detect the pose landmarks, prepare them for the model, and predict the gesture (clap, walk, or run). The predicted gesture is displayed on the video screen in real-time.

predicted_label = np.argmax(prediction) label_text = label_map[predicted_label] cv2.putText(image, f"Pose: {label_text}", (10, 40), ...)


Output:
A new window comes out and shows the output. The new window which was open or which was the testing runs can be closeed by pressing the Q button to stop the video. In summary, after setting up the environment and downloading the necessary videos, I created a dataset of pose landmarks from different gestures. Then, I trained a neural network model, tested it on unseen video data, and achieved 100% accuracy in recognizing the gestures.

For Climb: -
![image](https://github.com/user-attachments/assets/d9b97e58-eb54-41ca-851f-7b3e92f00011)

For Kick: -
![image](https://github.com/user-attachments/assets/60b92597-f75a-467c-805d-0ba77458103d)

Terminating the output window:
The output window can be terminated by simply pressing Q.
