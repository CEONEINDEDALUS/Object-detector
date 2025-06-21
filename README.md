# Video Object Tracker with YOLOv8
![11](https://github.com/user-attachments/assets/09f845be-05a0-4da9-bdfa-e71fc302440d)

A PyQt5-based desktop application for object detection and tracking in videos using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).  
It enables users to load a video, perform object detection and tracking, select a target object, customize highlight colors, adjust confidence threshold, and export videos with tracking overlays.

---

## Features

- **Video Playback Controls**: Play, pause, seek, and frame navigation.
- **Object Detection & Tracking**: Utilizes YOLOv8 for high-performance detection and ByteTrack for tracking.
- **Target Selection**: Click on detected objects to select the tracking target.
- **Customizable Highlight Colors**: Choose colors for target and non-target bounding boxes.
- **Adjustable Confidence Threshold**: Fine-tune which detections are shown.
- **Export Video**: Save the annotated/tracked video.
- **Tracking Logs**: View detection and tracking logs in real-time.

## Usage

1. **Run the application:**
   ```sh
   python yolo_object_tracker.py
   ```

2. **Load a video:**
   - Use the `File > Open Video` menu or the button to select your video file (`.mp4`, `.avi`, `.mov`, etc.).

3. **Play and analyze:**
   - Play the video. Detections and tracked objects will be shown.
   - Use the `Select Target` button and click on a bounding box to choose the object to track.

4. **Customize:**
   - Change target/non-target colors.
   - Adjust the confidence threshold for detection.

5. **Export:**
   - Click `Export Video` to save the video with bounding boxes and tracking overlays.

---

## Requirements

See [requirements.txt](requirements.txt).  
Key dependencies:
- PyQt5
- opencv-python
- numpy
- ultralytics

---

## Notes

- The first time you run the app, YOLOv8 weights (`yolov8n.pt`) will be downloaded automatically.
- For best performance, use a machine with a CUDA-capable GPU.
- If you encounter video loading issues, try re-encoding your video file.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- 
