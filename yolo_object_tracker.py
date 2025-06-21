
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QSlider, QTextEdit, QMessageBox, QColorDialog, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor
from ultralytics import YOLO
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Object Tracker")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize variables
        self.video_path = None
        self.cap = None
        self.frame = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.is_playing = False
        self.target_id = None
        self.results = None
        self.output_video = None
        self.selecting_target = False
        self.skipped_frames = 0
        self.max_skipped_frames = 10
        self.target_color = (0, 255, 0)  # Default: Green for target
        self.non_target_color = (255, 0, 0)  # Default: Red for non-target
        self.conf_threshold = 0.25  # Default YOLOv8 confidence threshold

        # Initialize YOLOv8 model
        try:
            self.model = YOLO('yolov8n.pt')
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load YOLO model: {e}")
            sys.exit(1)

        # Setup GUI
        self.setup_ui()
        
        # Timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Left panel: Video display and controls
        left_panel = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.mousePressEvent = self.select_target
        left_panel.addWidget(self.video_label)

        # Playback controls
        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_pause)
        controls_layout.addWidget(self.play_btn)

        self.select_target_btn = QPushButton("Select Target")
        self.select_target_btn.clicked.connect(self.enable_target_selection)
        controls_layout.addWidget(self.select_target_btn)

        self.target_color_btn = QPushButton("Set Target Color")
        self.target_color_btn.clicked.connect(self.set_target_color)
        controls_layout.addWidget(self.target_color_btn)

        self.non_target_color_btn = QPushButton("Set Non-Target Color")
        self.non_target_color_btn.clicked.connect(self.set_non_target_color)
        controls_layout.addWidget(self.non_target_color_btn)

        self.export_btn = QPushButton("Export Video")
        self.export_btn.clicked.connect(self.export_video)
        controls_layout.addWidget(self.export_btn)

        # Confidence threshold control
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence Threshold:")
        conf_layout.addWidget(conf_label)
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 0.9)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(self.conf_threshold)
        self.conf_spinbox.setDecimals(2)
        self.conf_spinbox.valueChanged.connect(self.set_conf_threshold)
        conf_layout.addWidget(self.conf_spinbox)
        
        left_panel.addLayout(controls_layout)
        left_panel.addLayout(conf_layout)

        # Frame slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.seek_frame)
        left_panel.addWidget(self.slider)

        # Right panel: Log area
        right_panel = QVBoxLayout()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        right_panel.addWidget(QLabel("Tracking Log"))
        right_panel.addWidget(self.log_area)

        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open Video")
        open_action.triggered.connect(self.open_video)

    def set_target_color(self):
        color = QColorDialog.getColor(initial=QColor(*self.target_color), parent=self, title="Select Target Color")
        if color.isValid():
            self.target_color = (color.red(), color.green(), color.blue())
            logger.info(f"Target color set to RGB: {self.target_color}")
            self.log_area.append(f"Target color set to RGB: {self.target_color}")
            self.update_frame()  # Force frame refresh
        else:
            logger.warning("Target color selection cancelled or invalid")
            self.log_area.append("Target color selection cancelled")

    def set_non_target_color(self):
        color = QColorDialog.getColor(initial=QColor(*self.non_target_color), parent=self, title="Select Non-Target Color")
        if color.isValid():
            self.non_target_color = (color.red(), color.green(), color.blue())
            logger.info(f"Non-target color set to RGB: {self.non_target_color}")
            self.log_area.append(f"Non-target color set to RGB: {self.non_target_color}")
            self.update_frame()  # Force frame refresh
        else:
            logger.warning("Non-target color selection cancelled or invalid")
            self.log_area.append("Non-target color selection cancelled")

    def set_conf_threshold(self, value):
        self.conf_threshold = value
        logger.info(f"Confidence threshold set to: {self.conf_threshold:.2f}")
        self.log_area.append(f"Confidence threshold set to: {self.conf_threshold:.2f}")
        self.update_frame()  # Refresh frame to apply new threshold
        if self.results and self.results[0].boxes:
            self.log_area.append(f"Detected {len(self.results[0].boxes)} objects")

    def open_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", 
                                                        "Video Files (*.mp4 *.avi *.mov)")
        if self.video_path:
            try:
                self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    raise ValueError("Failed to open video file")
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                if self.total_frames <= 0 or self.fps <= 0:
                    raise ValueError("Invalid video metadata: frames or FPS not detected")
                self.slider.setMaximum(self.total_frames - 1)
                self.current_frame = 0
                self.skipped_frames = 0
                self.log_area.append(f"Loaded video: {self.video_path}")
                self.log_area.append(f"Total frames: {self.total_frames}, FPS: {self.fps:.2f}")
                self.update_frame()
            except Exception as e:
                logger.error(f"Failed to load video: {e}")
                QMessageBox.critical(self, "Error", f"Failed to load video: {e}. Try re-encoding the video.")
                self.cap = None

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.skipped_frames += 1
                if self.skipped_frames >= self.max_skipped_frames:
                    logger.warning(f"Too many consecutive frame read failures ({self.skipped_frames})")
                    self.is_playing = False
                    self.timer.stop()
                    self.play_btn.setText("Play")
                    self.log_area.append(f"Error: Too many frame read failures at frame {self.current_frame}")
                    return
                logger.warning(f"Failed to read frame {self.current_frame}, skipping")
                self.log_area.append(f"Warning: Skipped frame {self.current_frame} due to read error")
                self.current_frame += 1
                self.slider.setValue(self.current_frame)
                return

            self.skipped_frames = 0
            self.frame = frame

            if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                logger.error(f"Invalid frame dimensions at frame {self.current_frame}")
                self.log_area.append(f"Error: Invalid frame dimensions at frame {self.current_frame}")
                return

            # Perform object detection and tracking
            try:
                self.results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf_threshold, verbose=False)
                logger.info(f"Processed frame {self.current_frame}, detected {len(self.results[0].boxes) if self.results[0].boxes else 0} objects")
            except Exception as e:
                logger.error(f"YOLO tracking failed at frame {self.current_frame}: {e}")
                self.log_area.append(f"Error: YOLO tracking failed at frame {self.current_frame}: {e}")
                return

            # Draw bounding boxes
            annotated_frame = self.draw_bounding_boxes(frame)
            
            # Convert frame to QImage
            height, width, channel = annotated_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            q_image = q_image.rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(q_image))

            # Update log
            if self.target_id is not None and self.results and self.results[0].boxes:
                for box in self.results[0].boxes:
                    if box.id is not None and int(box.id) == self.target_id:
                        conf = box.conf.item() if box.conf is not None else 0
                        cls_name = self.model.names[int(box.cls)] if box.cls is not None else "Unknown"
                        self.log_area.append(f"Frame {self.current_frame}: Target {cls_name} (ID: {self.target_id}), Confidence: {conf:.2f}")

            if self.is_playing:
                self.current_frame += 1
                self.slider.setValue(self.current_frame)
                if self.output_video:
                    try:
                        self.output_video.write(annotated_frame)
                    except Exception as e:
                        logger.error(f"Failed to write frame to output video: {e}")
                        self.log_area.append(f"Error: Failed to write frame to output video: {e}")

            if self.current_frame >= self.total_frames:
                self.is_playing = False
                self.timer.stop()
                self.play_btn.setText("Play")
                self.log_area.append("Reached end of video")

        except Exception as e:
            logger.error(f"Error in update_frame: {e}")
            self.log_area.append(f"Error in frame {self.current_frame}: {e}")

    def draw_bounding_boxes(self, frame):
        frame_copy = frame.copy()
        if self.results and self.results[0].boxes:
            for box in self.results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item() if box.conf is not None else 0
                cls_name = self.model.names[int(box.cls)] if box.cls is not None else "Unknown"
                track_id = int(box.id) if box.id is not None else -1
                
                color = self.target_color if track_id == self.target_id else self.non_target_color
                thickness = 3 if track_id == self.target_id else 1
                
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
                label = f"{cls_name} {conf:.2f}"
                if track_id == self.target_id:
                    label += f" (Target ID: {track_id})"
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame_copy

    def play_pause(self):
        if not self.cap:
            QMessageBox.warning(self, "Error", "No video loaded")
            return

        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.is_playing = True
            self.timer.start(int(1000 / self.fps))
            self.play_btn.setText("Pause")

    def enable_target_selection(self):
        if not self.cap:
            QMessageBox.warning(self, "Error", "No video loaded")
            return
        self.selecting_target = True
        self.video_label.setCursor(Qt.PointingHandCursor)
        self.log_area.append("Click on an object to select as target")

    def select_target(self, event):
        if not self.selecting_target or not self.results or not self.frame.any():
            return

        try:
            x, y = event.pos().x(), event.pos().y()
            for box in self.results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.target_id = int(box.id) if box.id is not None else None
                    cls_name = self.model.names[int(box.cls)] if box.cls is not None else "Unknown"
                    self.log_area.append(f"Selected target: {cls_name} (ID: {self.target_id})")
                    self.selecting_target = False
                    self.video_label.setCursor(Qt.ArrowCursor)
                    self.update_frame()
                    break
        except Exception as e:
            logger.error(f"Error selecting target: {e}")
            self.log_area.append(f"Error selecting target: {e}")

    def seek_frame(self, value):
        if not self.cap:
            return
        self.current_frame = value
        if not self.is_playing:
            self.update_frame()

    def export_video(self):
        if not self.cap:
            QMessageBox.warning(self, "Error", "No video loaded")
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video Files (*.mp4)")
        if output_path:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output_video = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                                  (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                                   int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                self.is_playing = True
                self.timer.start(int(1000 / self.fps))
                self.log_area.append("Exporting video with tracking overlay...")
            except Exception as e:
                logger.error(f"Failed to initialize video writer: {e}")
                self.log_area.append(f"Error: Failed to initialize video writer: {e}")

    def closeEvent(self, event):
        if self.output_video:
            self.output_video.release()
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoTracker()
    window.show()
    sys.exit(app.exec_())
