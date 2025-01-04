import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import sqlite3
from datetime import datetime
import json

from VolumeStoneNew_v3_4_LatestUpdate import VideoProcessor, VideoConfig

class RTSPStream(QThread):
    frame_ready = pyqtSignal(np.ndarray)  # Signal untuk mengirim frame ke GUI

    def __init__(self):
        super().__init__()
        self.cap = None
        self.is_running = False

    def start_stream(self, ip, port):
        rtsp_url = f"rtsp://admin:@{ip}:{port}/stream"
        try:
            self.cap = cv2.VideoCapture(rtsp_url)
            if not self.cap.isOpened():
                raise Exception("Failed to connect to RTSP stream")
            self.is_running = True
            self.start()  # Mulai thread untuk menangani stream
        except Exception as e:
            print(f"Error connecting to stream: {e}")

    def stop_stream(self):
        if self.cap:
            self.cap.release()
        self.is_running = False
        self.quit()  # Menghentikan thread saat stream dihentikan

    def run(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)  # Emit signal dengan frame baru

class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('stone_detection.db')
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS detection_sessions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            roi_points TEXT,
                            config TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS frame_analysis (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id INTEGER,
                            frame_number INTEGER,
                            timestamp TEXT,
                            stones_detected INTEGER,
                            coverage_percentage REAL,
                            analysis_data TEXT,
                            FOREIGN KEY (session_id) REFERENCES detection_sessions (id))''')
        self.conn.commit()

    def save_session(self, roi_points, config):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO detection_sessions (timestamp, roi_points, config)
                          VALUES (?, ?, ?)''',
                       (datetime.now().isoformat(), json.dumps(roi_points), json.dumps(config.__dict__)))
        self.conn.commit()
        return cursor.lastrowid

    def save_frame_analysis(self, session_id, frame_data):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO frame_analysis (session_id, frame_number, timestamp, stones_detected,
                                                      coverage_percentage, analysis_data)
                          VALUES (?, ?, ?, ?, ?, ?)''',
                       (session_id, frame_data['frame_number'], frame_data['timestamp'], frame_data['stones_detected'],
                        frame_data['coverage_percentage'], json.dumps(frame_data)))
        self.conn.commit()

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid gray;")
        self.layout.addWidget(self.video_label)

        # Control buttons
        self.button_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.analyze_btn = QPushButton("Analyze")
        self.button_layout.addWidget(self.preview_btn)
        self.button_layout.addWidget(self.analyze_btn)
        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

class DashboardWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        # Status display
        self.status_label = QLabel("Status: Idle")
        self.stones_label = QLabel("Detected Stones: 0")
        self.coverage_label = QLabel("Coverage: 0%")

        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.stones_label)
        self.layout.addWidget(self.coverage_label)

        self.setLayout(self.layout)

    def update_stats(self, stones, coverage):
        self.stones_label.setText(f"Detected Stones: {stones}")
        self.coverage_label.setText(f"Coverage: {coverage:.2f}%")

    def set_status(self, status):
        self.status_label.setText(f"Status: {status}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stone Detection System")
        self.setMinimumSize(800, 600)

        # Initialize components
        self.rtsp_stream = RTSPStream()
        self.db_manager = DatabaseManager()
        self.video_processor = None
        self.config = VideoConfig()
        self.roi_points = []
        self.is_selecting_roi = False
        self.current_session_id = None

        # Setup UI
        self.setup_ui()

        # Connect signals
        self.rtsp_stream.frame_ready.connect(self.update_frame)

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Connection controls
        conn_layout = QHBoxLayout()
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("IP Address")
        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("Port")
        self.stream_btn = QPushButton("Stream")
        self.stream_btn.clicked.connect(self.toggle_stream)

        conn_layout.addWidget(self.ip_input)
        conn_layout.addWidget(self.port_input)
        conn_layout.addWidget(self.stream_btn)
        main_layout.addLayout(conn_layout)

        # Video widget
        self.video_widget = VideoWidget()
        self.video_widget.preview_btn.clicked.connect(self.start_roi_selection)
        self.video_widget.analyze_btn.clicked.connect(self.toggle_analysis)
        main_layout.addWidget(self.video_widget)

        # Dashboard
        self.dashboard = DashboardWidget()
        main_layout.addWidget(self.dashboard)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def toggle_stream(self):
        if not self.rtsp_stream.is_running:
            ip = self.ip_input.text()
            port = self.port_input.text()

            if ip and port:
                self.rtsp_stream.start_stream(ip, port)
                self.stream_btn.setText("Stop")
                self.dashboard.set_status("Streaming")
            else:
                QMessageBox.critical(self, "Error", "Please enter IP and Port")
        else:
            self.rtsp_stream.stop_stream()
            self.stream_btn.setText("Stream")
            self.dashboard.set_status("Idle")

    def update_frame(self, frame):
        # Convert frame to QImage and display it
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_widget.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_widget.video_label.setPixmap(scaled_pixmap)

        # Process frame if analysis is active
        if self.video_processor and not self.is_selecting_roi:
            self.process_frame(frame)

    def process_frame(self, frame):
        if self.video_processor and self.current_session_id:
            # Detect stones
            stones = self.video_processor.stone_detector.detect_stones(frame,
                                                                     self.video_processor.roi_mask)

            # Calculate coverage
            coverage = self.video_processor.grid_analyzer.calculate_coverage()

            # Update dashboard
            self.dashboard.update_stats(len(stones), coverage)

            # Save frame analysis to database
            frame_data = {
                'frame_number': self.video_processor.frame_count,
                'timestamp': datetime.now().isoformat(),
                'stones_detected': len(stones),
                'coverage_percentage': coverage,
                'analysis_data': {
                    'stone_areas': [stone['area'] for stone in stones],
                    'grid_coverage': coverage
                }
            }
            self.db_manager.save_frame_analysis(self.current_session_id, frame_data)

    def start_roi_selection(self):
        if self.rtsp_stream.is_running:
            self.is_selecting_roi = True
            self.dashboard.set_status("Selecting ROI - Click 4 points")
            self.roi_points = []
            self.temp_frame = None
            self.video_widget.preview_btn.setEnabled(False)
            self.video_widget.analyze_btn.setEnabled(False)
            self.video_widget.video_label.mousePressEvent = self.roi_mouse_press

            # Tambahkan instruksi visual
            QMessageBox.information(self, "ROI Selection",
                "Click 4 points on the video to select the region of interest.\n"
                "Points will be connected to form a polygon.")

    def roi_mouse_press(self, event):
        if self.is_selecting_roi:
            pos = event.pos()
            # Get current frame for drawing
            if self.temp_frame is None:
                self.temp_frame = self.rtsp_stream.read_frame().copy()

            # Convert coordinates to original frame size
            label_size = self.video_widget.video_label.size()
            pixmap_size = self.video_widget.video_label.pixmap().size()

            # Calculate scaling factors and offsets
            x_scale = self.temp_frame.shape[1] / pixmap_size.width()
            y_scale = self.temp_frame.shape[0] / pixmap_size.height()

            # Calculate label position offsets
            x_offset = (label_size.width() - pixmap_size.width()) // 2
            y_offset = (label_size.height() - pixmap_size.height()) // 2

            # Adjust click coordinates
            x = int((pos.x() - x_offset) * x_scale)
            y = int((pos.y() - y_offset) * y_scale)

            # Ensure coordinates are within frame boundaries
            x = max(0, min(x, self.temp_frame.shape[1] - 1))
            y = max(0, min(y, self.temp_frame.shape[0] - 1))

            self.roi_points.append((x, y))

            # Draw point and lines on frame
            frame_copy = self.temp_frame.copy()

            # Draw all points
            for point in self.roi_points:
                cv2.circle(frame_copy, point, 5, (0, 255, 0), -1)

            # Draw lines between points
            if len(self.roi_points) > 1:
                for i in range(len(self.roi_points) - 1):
                    cv2.line(frame_copy, self.roi_points[i], self.roi_points[i + 1],
                            (0, 255, 0), 2)

                # Close polygon if all points selected
                if len(self.roi_points) == 4:
                    cv2.line(frame_copy, self.roi_points[-1], self.roi_points[0],
                            (0, 255, 0), 2)

            # Update display
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line,
                            QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_widget.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            self.video_widget.video_label.setPixmap(scaled_pixmap)

            # Update status
            self.dashboard.set_status(f"Selecting ROI - {len(self.roi_points)}/4 points")

            if len(self.roi_points) >= 4:
                self.finish_roi_selection()

    def finish_roi_selection(self):
        self.is_selecting_roi = False
        self.video_widget.video_label.mousePressEvent = None
        self.dashboard.set_status("ROI Selected")

        # Create mask from ROI points
        frame_shape = self.temp_frame.shape
        self.roi_mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
        roi_points_arr = np.array(self.roi_points, dtype=np.int32)
        cv2.fillPoly(self.roi_mask, [roi_points_arr], 255)

        # Initialize video processor with selected ROI
        self.video_processor = VideoProcessor(self.config)
        self.video_processor.roi_mask = self.roi_mask
        self.video_processor.roi_points = self.roi_points

        # Initialize grid analyzer
        self.video_processor.grid_analyzer.initialize_grid(self.roi_mask, self.roi_points)

        # Save session to database
        self.current_session_id = self.db_manager.save_session(
            self.roi_points, self.config)

        # Re-enable buttons
        self.video_widget.preview_btn.setEnabled(True)
        self.video_widget.analyze_btn.setEnabled(True)

        # Show confirmation
        QMessageBox.information(self, "ROI Selection",
            "Region of Interest has been selected successfully.\n"
            "You can now start the analysis.")

    def toggle_analysis(self):
        if not self.video_processor or not self.roi_points:
            QMessageBox.warning(self, "Warning", "Please select ROI first")
            return

        if self.video_widget.analyze_btn.text() == "Analyze":
            self.video_widget.analyze_btn.setText("Stop Analysis")
            self.dashboard.set_status("Analyzing")
        else:
            self.video_widget.analyze_btn.setText("Analyze")
            self.dashboard.set_status("Streaming")

    def process_frame(self, frame):
        if self.video_processor and self.current_session_id:
            # Detect stones
            stones = self.video_processor.stone_detector.detect_stones(frame,
                                                                     self.video_processor.roi_mask)

            # Calculate coverage
            coverage = self.video_processor.grid_analyzer.calculate_coverage()

            # Update dashboard
            self.dashboard.update_stats(len(stones), coverage)

            # Save frame analysis to database
            frame_data = {
                'frame_number': self.video_processor.frame_count,
                'timestamp': datetime.now().isoformat(),
                'stones_detected': len(stones),
                'coverage_percentage': coverage,
                'analysis_data': {
                    'stone_areas': [stone['area'] for stone in stones],
                    'grid_coverage': coverage
                }
            }
            self.db_manager.save_frame_analysis(self.current_session_id, frame_data)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
