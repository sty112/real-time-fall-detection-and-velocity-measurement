![image](https://github.com/sty112/real-time-fall-detection-and-velocity-measurement/assets/84215617/0c0ee7d2-9119-4fbf-bf1b-220844f344ff)# real-time-fall-detection-and-velocity-measurement

Introduction:
This is an individual project done for my capstone course,my purpose of building this project is because many people especially the older generation, miss the best rescue period after having no one at home to support them, resulting in permanent injuries or casualties after a serious fall at home. So my purpose is to use machine vision and deep learning to identify whether the detected character has fallen, and calculate the speed of the fall to determine the degree of injury.

procedure:
1. I used the yolov8 object detection model (yolov8n.pt) to predict if the person in the camera has fallen or not. To let the yolov8.net model know what falling looks like i trained the model with around 10000 images of people falling.
  
2. Then to calculate the speed of the fall, I used the pretrained pose estimation provided by ultralytics. pose estimation models are models where the model identifies and tracks the position of a person or object in an image or video. It does this by detecting key points, such as joints in the human body (e.g., elbows, knees, and shoulders), and estimating their spatial coordinates.

3. I then extract the y coordinate of each keypoints from the pose estimation model frame by frame to calculate the velocity of each keypoints when the person is falling and when the person has completely fell flat we will see which part has the maximum fall velocity and we will use it as the fall velocity of the person. If the fall velocity exceeds where the human bone can withstand an ambulance will be called. The v-t graph is also being further analyzed for when we need more precise fall detection as models aren't always perfect. Then after able to calculate the fall valocity, I then combine the fall detection model and the pose estimation model to do real time fall detection and fall velocity measurement.

Testing:
To test if the models are working as intended, I conducted a small experiment by recording various fall patterns for example front fall,back fall and going to fall but balanced ones self, and I threw the videos to the model for it to analyze.

front fall
![image](https://github.com/sty112/real-time-fall-detection-and-velocity-measurement/assets/84215617/5ea2c713-e311-469b-924b-26d09ea21c86)

back fall
![image](https://github.com/sty112/real-time-fall-detection-and-velocity-measurement/assets/84215617/e43482ac-a4c4-44c3-88f1-49f5edc7a357)

going to fall but balanced
![image](https://github.com/sty112/real-time-fall-detection-and-velocity-measurement/assets/84215617/746362f2-b478-4de9-afe4-f85ce394c840)




