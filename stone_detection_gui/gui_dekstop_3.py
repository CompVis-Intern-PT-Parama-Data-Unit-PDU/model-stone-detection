import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import cv2
import numpy as np
import threading
import mysql.connector
from PIL import Image, ImageTk
import time
import os
import logging
import json
from datetime import datetime

class VolumeStoneAdvancedApp:
    def __init__(self, root):
        self.setup_logging()

        # Main window configuration
        self.root = root
        self.root.title("Volume Stone Advanced CCTV Analysis")
        self.root.geometry("1920x1080")
        self.root.configure(bg='#f0f0f0')

        # State variables
        self.rtsp_url = None
        self.video_capture = None
        self.is_streaming = False
        self.is_analyzing = False
        self.roi_points = []
        self.current_frame = None
        self.is_selecting_roi = False
        self.analysis_results = []
        self.last_analysis_time = 0
        self.analysis_interval = 1.0  # seconds

        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'volume_stone_db',
            'auth_plugin': 'mysql_native_password'
        }

        # Initialize processing components
        self.setup_processing()
        self.setup_database()
        self.create_main_layout()
        self.create_menu_bar()

    def setup_processing(self):
        """Initialize processing components"""
        from VolumeStoneNew_v3_4_LatestUpdate import VideoConfig, StoneDetector, GridAnalyzer

        self.config = VideoConfig()
        self.stone_detector = StoneDetector(self.config)
        self.grid_analyzer = GridAnalyzer((self.config.grid_cell_width,
                                         self.config.grid_cell_height))

    def setup_logging(self):
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('logs/volume_stone.log'),
                logging.StreamHandler()
            ]
        )

    def create_menu_bar(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Grid",
                                variable=tk.BooleanVar(value=True),
                                command=self.toggle_grid)

        # Analysis Menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Settings", command=self.show_settings)

    def create_main_layout(self):
        # Style configuration
        style = ttk.Style()
        style.configure("TFrame", background='#f0f0f0')
        style.configure("TButton", padding=6, relief="flat", background="#4CAF50")
        style.configure("TLabel", background='#f0f0f0')

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (Video and Controls)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Connection panel
        self.create_connection_panel(left_frame)

        # Video display
        self.create_video_panel(left_frame)

        # Control panel
        self.create_control_panel(left_frame)

        # Right panel (Results)
        self.create_results_panel(main_frame)

        # Status bar
        self.create_status_bar()

    def create_connection_panel(self, parent):
        conn_frame = ttk.Frame(parent)
        conn_frame.pack(fill='x', pady=5)

        # IP input
        ttk.Label(conn_frame, text="IP CCTV:").pack(side='left')
        self.ip_entry = ttk.Entry(conn_frame, width=20)
        self.ip_entry.pack(side='left', padx=5)

        # Port input
        ttk.Label(conn_frame, text="Port:").pack(side='left')
        self.port_entry = ttk.Entry(conn_frame, width=10)
        self.port_entry.pack(side='left', padx=5)

        # Stream button
        self.stream_button = ttk.Button(conn_frame,
                                      text="Stream",
                                      command=self.start_stream,
                                      style="Accent.TButton")
        self.stream_button.pack(side='left', padx=5)

    def reset_roi(self):
        """Reset ROI selection"""
        self.roi_points = []
        self.analyze_btn.config(state=tk.DISABLED)
        self.roi_btn.config(text="Select ROI")
        self.roi_btn.config(command=self.start_roi_selection)
        self.status_var.set("ROI selection reset")

    def create_video_panel(self, parent):
        video_frame = ttk.Frame(parent)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ROI selection bindings
        self.video_label.bind('<Button-1>', self.on_mouse_click)
        self.video_label.bind('<Button-3>', self.complete_roi_selection)
        self.video_label.bind('<Return>', self.complete_roi_selection)
        self.video_label.bind('<r>', self.reset_roi)
        self.video_label.focus_set()

    def create_control_panel(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', pady=5)

        # ROI selection button
        self.roi_button = ttk.Button(control_frame,
                                   text="Select ROI",
                                   command=self.start_roi_selection)
        self.roi_button.pack(side='left', padx=5)

        # Analysis button
        self.analyze_button = ttk.Button(control_frame,
                                       text="Start Analysis",
                                       command=self.toggle_analysis,
                                       state=tk.DISABLED)
        self.analyze_button.pack(side='left', padx=5)

        # Save button
        self.save_button = ttk.Button(control_frame,
                                    text="Save Frame",
                                    command=self.save_current_frame,
                                    state=tk.DISABLED)
        self.save_button.pack(side='left', padx=5)

    def create_results_panel(self, parent):
        right_frame = ttk.Frame(parent, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Results title
        ttk.Label(right_frame,
                 text="Analysis Results",
                 font=('Helvetica', 12, 'bold')).pack(pady=5)

        # Results tree
        self.results_tree = ttk.Treeview(right_frame,
                                       columns=('Time', 'Coverage', 'Stones'),
                                       show='headings',
                                       height=20)

        self.results_tree.heading('Time', text='Time')
        self.results_tree.heading('Coverage', text='Coverage %')
        self.results_tree.heading('Stones', text='Stones')

        self.results_tree.column('Time', width=100)
        self.results_tree.column('Coverage', width=80)
        self.results_tree.column('Stones', width=60)

        self.results_tree.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(right_frame,
                                orient=tk.VERTICAL,
                                command=self.results_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

    def create_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root,
                             textvariable=self.status_var,
                             relief=tk.SUNKEN,
                             anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def start_stream(self):
        ip = self.ip_entry.get()
        port = self.port_entry.get()

        if not ip or not port:
            messagebox.showerror("Error", "Please enter IP and Port")
            return

        self.rtsp_url = f"rtsp://admin:@{ip}:{port}/stream"
        threading.Thread(target=self.stream_video, daemon=True).start()

    def stream_video(self):
        try:
            self.video_capture = cv2.VideoCapture(self.rtsp_url)
            if not self.video_capture.isOpened():
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", "Could not connect to stream"))
                return

            self.is_streaming = True
            self.stream_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.NORMAL)

            while self.is_streaming:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                self.current_frame = frame.copy()

                # Process frame if analyzing
                if self.is_analyzing and \
                   time.time() - self.last_analysis_time >= self.analysis_interval:
                    self.process_current_frame()

                # Update display with overlays
                self.update_display(frame)

            self.video_capture.release()

        except Exception as e:
            logging.error(f"Streaming error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Streaming error: {str(e)}"))

    def process_current_frame(self):
        """Process the current frame for analysis"""
        try:
            if self.current_frame is None or not self.roi_points:
                return

            # Create ROI mask
            mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.roi_points)], 255)

            # Detect stones
            stones = self.stone_detector.detect_stones(self.current_frame, mask)

            # Update grid
            self.grid_analyzer.update_grid(stones, mask)
            coverage = self.grid_analyzer.calculate_coverage()

            # Save frame if significant detection
            if coverage > 1.0:  # More than 1% coverage
                self.save_detection_frame(coverage, len(stones))

            # Update UI
            self.update_results(coverage, len(stones))
            self.last_analysis_time = time.time()

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            self.status_var.set(f"Processing error: {str(e)}")

    def update_display(self, frame):
        """Update video display with overlays"""
        if frame is None:
            logging.error("Received None frame in update_display")
            return  # Exit if the frame is invalid

        display_frame = frame.copy()

        # Draw ROI
        if self.roi_points and len(self.roi_points) > 0:
            cv2.polylines(display_frame,
                        [np.array(self.roi_points, dtype=np.int32)],
                        True, (0, 255, 0), 2)

        # Draw grid if analyzing
        if self.is_analyzing and hasattr(self.grid_analyzer, 'draw_grid_overlay'):
            display_frame = self.grid_analyzer.draw_grid_overlay(
                display_frame, self.roi_points)
            if display_frame is None:
                logging.error("draw_grid_overlay returned None")

        # Convert to RGB and display
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

    def update_results(self, coverage: float, stone_count: int):
        """Update results display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_tree.insert('', 0, values=(
            timestamp,
            f"{coverage:.2f}",
            stone_count
        ))

        self.status_var.set(
            f"Analysis: {coverage:.2f}% coverage, {stone_count} stones detected")

    def save_detection_frame(self, coverage: float, stone_count: int):
        """Save frame with significant detection"""
        try:
            # Create detection directory
            os.makedirs('detections', exist_ok=True)

            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = f"detections/detection_{timestamp}.jpg"
            cv2.imwrite(frame_path, self.current_frame)

            # Save to database
            self.save_to_database({
                'coverage': coverage,
                'stone_count': stone_count,
                'frame_path': frame_path
            })

        except Exception as e:
            logging.error(f"Error saving detection: {str(e)}")

    def save_current_frame(self):
        """Save current frame manually"""
        if self.current_frame is None:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                initialfile=f"frame_{timestamp}.jpg",
                filetypes=[("JPEG files", "*.jpg"),
                            ("PNG files", "*.png"),
                            ("All files", "*.*")])

            if filename:
                cv2.imwrite(filename, self.current_frame)
                messagebox.showinfo("Success", f"Frame saved to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save frame: {str(e)}")

    def start_roi_selection(self):
        """Start ROI selection mode"""
        self.roi_points = []
        self.is_selecting_roi = True
        self.analyze_button.config(state=tk.DISABLED)
        self.roi_button.config(text="Reset ROI")
        self.status_var.set("Click to select ROI points (minimum 3 points)")

    def on_mouse_click(self, event):
        """Handle mouse clicks for ROI selection"""
        if not self.is_streaming or not self.is_selecting_roi:
            return

        # Add point
        self.roi_points.append((event.x, event.y))

        # Enable analysis if we have enough points
        if len(self.roi_points) >= 3:
            self.analyze_button.config(state=tk.NORMAL)

        # Update display
        self.update_display(self.current_frame)

    def complete_roi_selection(self, event=None):
        """Complete ROI selection"""
        if len(self.roi_points) < 3:
            messagebox.showwarning(
                "Warning",
                "Please select at least 3 points for ROI"
            )
            return

        self.is_selecting_roi = False
        self.roi_button.config(text="Select ROI")

        # Initialize grid analyzer with ROI
        mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.roi_points)], 255)
        self.grid_analyzer.initialize_grid(mask, self.roi_points)

        self.status_var.set("ROI selected. Ready for analysis.")

    def toggle_analysis(self):
        """Toggle analysis state"""
        if not self.is_analyzing:
            self.start_analysis()
        else:
            self.stop_analysis()

    def start_analysis(self):
        """Start analysis process"""
        if len(self.roi_points) < 3:
            messagebox.showwarning("Warning", "Please select ROI first")
            return

        self.is_analyzing = True
        self.analyze_button.config(text="Stop Analysis")
        self.last_analysis_time = 0  # Force immediate analysis
        self.status_var.set("Analysis started...")

    def stop_analysis(self):
        """Stop analysis process"""
        self.is_analyzing = False
        self.analyze_button.config(text="Start Analysis")
        self.status_var.set("Analysis stopped")

    def save_to_database(self, analysis_result):
        """Save analysis result to database"""
        try:
            query = """
                INSERT INTO analysis_results
                (timestamp, frame_path, coverage_percentage, stone_count)
                VALUES (NOW(), %s, %s, %s)
            """

            self.db_cursor.execute(query, (
                analysis_result['frame_path'],
                analysis_result['coverage'],
                analysis_result['stone_count']
            ))

            self.db_connection.commit()
            logging.info(f"Analysis result saved to database")

        except mysql.connector.Error as e:
            logging.error(f"Database error: {str(e)}")

    def setup_database(self):
        """Setup database connection and tables"""
        try:
            self.db_connection = mysql.connector.connect(**self.db_config)
            self.db_cursor = self.db_connection.cursor()

            # Create database if not exists
            self.db_cursor.execute(
                "CREATE DATABASE IF NOT EXISTS volume_stone_db")
            self.db_cursor.execute("USE volume_stone_db")

            # Create tables
            self.db_cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    frame_path VARCHAR(255),
                    coverage_percentage FLOAT,
                    stone_count INT
                )
            """)

            self.db_connection.commit()
            logging.info("Database setup complete")

        except mysql.connector.Error as e:
            logging.error(f"Database setup error: {str(e)}")
            messagebox.showwarning(
                "Database Warning",
                "Failed to connect to database. Analysis results won't be saved."
            )

    def export_results(self):
        """Export analysis results"""
        try:
            # Get results from database
            self.db_cursor.execute("""
                SELECT timestamp, coverage_percentage, stone_count, frame_path
                FROM analysis_results
                ORDER BY timestamp DESC
            """)

            results = [{
                'timestamp': row[0].strftime("%Y-%m-%d %H:%M:%S"),
                'coverage': float(row[1]),
                'stone_count': int(row[2]),
                'frame_path': row[3]
            } for row in self.db_cursor.fetchall()]

            # Save to file
            export_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )

            if export_path:
                with open(export_path, 'w') as f:
                    json.dump({
                        'analysis_results': results,
                        'export_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'total_records': len(results)
                    }, f, indent=4)

                messagebox.showinfo(
                    "Success",
                    f"Results exported to {export_path}"
                )

        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export results: {str(e)}"
            )

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Analysis Settings")
        settings_window.geometry("400x300")

        # Grid settings
        ttk.Label(settings_window, text="Grid Settings").pack(pady=10)

        # Grid cell size
        grid_frame = ttk.Frame(settings_window)
        grid_frame.pack(fill='x', padx=20)

        ttk.Label(grid_frame, text="Cell Width:").pack(side='left')
        width_var = tk.StringVar(value=str(self.config.grid_cell_width))
        ttk.Entry(grid_frame, textvariable=width_var, width=10).pack(side='left', padx=5)

        ttk.Label(grid_frame, text="Cell Height:").pack(side='left', padx=5)
        height_var = tk.StringVar(value=str(self.config.grid_cell_height))
        ttk.Entry(grid_frame, textvariable=height_var, width=10).pack(side='left')

        # Analysis interval
        interval_frame = ttk.Frame(settings_window)
        interval_frame.pack(fill='x', padx=20, pady=10)

        ttk.Label(interval_frame, text="Analysis Interval (s):").pack(side='left')
        interval_var = tk.StringVar(value=str(self.analysis_interval))
        ttk.Entry(interval_frame, textvariable=interval_var, width=10).pack(side='left', padx=5)

        # Save button
        def save_settings():
            try:
                self.config.grid_cell_width = int(width_var.get())
                self.config.grid_cell_height = int(height_var.get())
                self.analysis_interval = float(interval_var.get())
                settings_window.destroy()
                messagebox.showinfo("Success", "Settings saved")
            except ValueError as e:
                messagebox.showerror("Error", "Please enter valid numbers")

        ttk.Button(settings_window,
                  text="Save Settings",
                  command=save_settings).pack(pady=20)

    def toggle_grid(self):
        """Toggle grid visibility"""
        self.config.show_grid = not self.config.show_grid
        if self.current_frame is not None:
            self.update_display(self.current_frame)

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.is_streaming:
                self.is_streaming = False
                if self.video_capture:
                    self.video_capture.release()
            self.root.destroy()

def main():
    root = tk.Tk()
    app = VolumeStoneAdvancedApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()