I began by learning how to tackle the problem. Once I understood the basics, I used Visual Studio Code (VS Code) to write my Python code. I added extensions for Jupyter Notebook and Python to VS Code. Then, I used the command prompt to install necessary Python libraries like numpy, pandas, opencv-python, tensorflow, and keras. These libraries would help me process data, analyze images, and build machine learning models.

I collected 12 YouTube videos, each showing one of three gestures: climbing, kicking, or dancing. I had four videos for each gesture, using three for training and one for testing. To make it easier to work with, I converted these videos into AVI format. Next, I organized these videos into a dataset, prepared it for the model, and divided it into training and testing sets. After training the model on the training data, I tested it on the unseen test data and achieved a perfect accuracy of 99%. 

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
A neural network using TensorFlow/Keras was built, which consists of multiple layers: Two dense layers with 64 and 32 neurons respectively, both using the ReLU activation function. The output layer uses the softmax activation function to predict one of the 3 gestures (climb, kick, dance). After compiling the model, I trained it on the landmark data and achieved 99% accuracy on the test data.

model = tf.keras.models.Sequential([ tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(len(set(y)), activation='softmax') ])

Saving the model:
After training, the model is saved to a file, making it reusable for later testing or deployment.

model.save(model_path)

Testing the model:
In the final step, we load the trained model and use it to make predictions on new video data. For each frame in the test video, we detect the pose landmarks, prepare them for the model, and predict the gesture (climb, kick, or dance). The predicted gesture is displayed on the video screen in real-time.

predicted_label = np.argmax(prediction) label_text = label_map[predicted_label] cv2.putText(image, f"Pose: {label_text}", (10, 40), ...)


Output:
A new window comes out and shows the output. The new window which was open or which was the testing runs can be closeed by pressing the Q button to stop the video. In summary, after setting up the environment and downloading the necessary videos, I created a dataset of pose landmarks from different gestures. Then, I trained a neural network model, tested it on unseen video data, and achieved 100% accuracy in recognizing the gestures.

For Climb: -
![image](https://github.com/user-attachments/assets/d9b97e58-eb54-41ca-851f-7b3e92f00011)

For Kick: -
![image](https://github.com/user-attachments/assets/60b92597-f75a-467c-805d-0ba77458103d)

Terminating the output window:
The output window can be terminated by simply pressing Q.

Uses and future scope:
This model can be used in various fields such as fitness, healthcare, security and surveillance, etc. 


This is the link to the github repositary: https://github.com/VivaanN98/IADAI201-1000064-Vivaan_Nayak


The future potential of the pose detection project offers exciting opportunities for expansion and enhancement across various domains. Here are several key areas where the project could evolve:

Expanding Gesture and Activity Recognition:
Currently, the system recognizes basic gestures like clapping, walking, and running. Future developments could include a broader range of activities, such as jumping, sitting, standing up, yoga poses, or even dance movements. By diversifying the gestures and activities in the dataset, the system could be applied in numerous fields, including sports coaching, fitness tracking, rehabilitation, and even dance therapy. This expansion would increase the versatility and usefulness of the system in real-world applications.

Real-time Pose Feedback and Correction:
While the current system detects poses, it doesn't provide real-time feedback on their correctness. A future iteration could analyze whether a user is performing an exercise or movement correctly, offering corrective suggestions or warnings to help improve form. This would be especially valuable in fitness and physical therapy settings, where maintaining proper posture and technique is crucial for preventing injuries and optimizing performance.

Integration with Wearable Devices:
Integrating the pose detection system with wearable technologies—such as smartwatches, fitness trackers, or AR glasses—could provide deeper insights into user activities. Wearables could supply additional data, such as heart rate, motion tracking, or orientation, which would complement the visual pose data. This combination would allow for a more comprehensive analysis of physical performance, fatigue levels, and overall health metrics, offering a richer understanding of a person’s activity and well-being.

Advanced 3D Pose Estimation:
Future versions of the system could incorporate 3D pose estimation using depth sensors or stereo cameras. This would enable the model to better capture complex movements in three-dimensional space, such as jumping, bending, or other actions that require depth perception. This enhancement would improve tracking accuracy, especially in dynamic environments, where 3D spatial awareness is essential for understanding human motion more precisely.

Applications in Augmented and Virtual Reality (AR/VR):
Pose detection has significant potential in AR and VR environments. By tracking body movements in real-time, users could control virtual avatars or interact with virtual environments, enhancing the gaming and entertainment experience. In addition, virtual fitness or yoga classes could be revolutionized by real-time pose tracking and correction, allowing instructors to provide immediate feedback to participants on their form and posture within the virtual space.

Multi-Person Pose Detection and Interaction:
While the current system focuses on single-person pose detection, expanding it to handle multiple individuals simultaneously would unlock new possibilities. For example, the system could be used in team sports analysis, group fitness sessions, or even virtual social interactions. It could track multiple users, analyze their movements, and detect interactions, which would be valuable in areas such as sports training, collaborative workspaces, and even crowd behavior analysis in security or public safety contexts.

Conclusion:
In summary, the pose detection system has the potential to evolve into a sophisticated, multifaceted platform with applications across a wide range of industries, from healthcare and sports to entertainment and gaming. As AI, sensor technologies, and real-time processing continue to advance, the scope for enhancing and expanding this system is vast, offering exciting opportunities for future development.


Here is the confusion matrix and f1 score:

![image](https://github.com/user-attachments/assets/56dd742b-008a-400e-9467-5b1a49888b2e)

