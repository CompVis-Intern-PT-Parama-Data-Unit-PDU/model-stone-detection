import cv2
import os
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import logging
from datetime import datetime
from tqdm import tqdm
import time

# Configure logging dengan format yang lebih informatif
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Konfigurasi parameter untuk pemrosesan video"""
    # Interval analisis
    analysis_interval: int = 5  # seconds

    # Parameter area batuan
    min_stone_area: int = 30
    max_stone_area: int = 5000

    # Parameter perubahan area
    max_area_change: float = 0.5

    # Parameter deteksi
    detection_sensitivity: float = 0.6
    min_circularity: float = 0.3
    max_circularity: float = 0.85
    min_aspect_ratio: float = 0.4
    max_aspect_ratio: float = 0.9

    # Parameter output
    output_width: int = 640

    # Parameter enhancement
    blur_kernel_size: int = 7
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    bilateral_sigma_color: int = 50
    bilateral_sigma_space: int = 50

    # Parameter density
    density_weight: float = 0.3
    distance_normalization: float = 50.0

    # Parameter percentage
    min_percentage: float = 0.5
    max_percentage: float = 40.0
    smoothing_factor: float = 0.3

class ProcessingProgress:
    """Class untuk tracking dan menampilkan progress pemrosesan"""
    def __init__(self, total_frames: int, description: str = "Processing"):
        self.progress_bar = tqdm(total=total_frames,
                               desc=description,
                               unit="frames")
        self.start_time = time.time()
        self.total_frames = total_frames

    def update(self, frames: int = 1):
        self.progress_bar.update(frames)

    def get_eta(self) -> str:
        elapsed = time.time() - self.start_time
        frames_processed = self.progress_bar.n
        if frames_processed == 0:
            return "Calculating..."

        frames_remaining = self.total_frames - frames_processed
        time_per_frame = elapsed / frames_processed
        eta_seconds = frames_remaining * time_per_frame

        return time.strftime('%H:%M:%S', time.gmtime(eta_seconds))

    def close(self):
        self.progress_bar.close()

class InteractiveROISelector:
    """Class untuk pemilihan ROI secara interaktif"""
    def __init__(self):
        self.points = []
        self.original_frame = None
        self.window_name = "Select ROI - Click points, Press 'C' to complete, 'R' to reset"

    def mouse_callback(self, event, x, y, flags, param):
        display_frame = param['frame']
        scale = display_frame.shape[1] / self.original_frame.shape[1]

        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(x / scale)
            original_y = int(y / scale)
            self.points.append((original_x, original_y))

            # Update display
            self._update_display(display_frame, scale)

    def _update_display(self, display_frame: np.ndarray, scale: float):
        frame_copy = display_frame.copy()
        if len(self.points) > 0:
            points = np.array(self.points, np.int32)
            scaled_points = (points * scale).astype(np.int32)

            # Draw polygon
            if len(self.points) > 2:
                cv2.fillPoly(frame_copy, [scaled_points], (0, 255, 0, 50))
            cv2.polylines(frame_copy, [scaled_points],
                         len(self.points) > 2, (0, 255, 0), 2)

            # Draw points
            for pt in scaled_points:
                cv2.circle(frame_copy, tuple(pt), 3, (0, 0, 255), -1)

            # Draw point numbers
            for i, pt in enumerate(scaled_points):
                cv2.putText(frame_copy, str(i+1),
                          (pt[0]+5, pt[1]-5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 1)

        # Show instructions
        cv2.putText(frame_copy, "Press 'C' to complete, 'R' to reset",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.imshow(self.window_name, frame_copy)

    def select_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Memungkinkan user untuk memilih ROI dengan mouse.
        Returns:
            mask: Binary mask untuk ROI
            points: List koordinat points ROI
        """
        self.original_frame = frame.copy()
        display_width = 1280
        scale = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * scale)
        display_frame = cv2.resize(frame, (display_width, display_height))

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, display_width, display_height)
        cv2.setMouseCallback(self.window_name,
                           self.mouse_callback,
                           {'frame': display_frame})

        while True:
            self._update_display(display_frame, scale)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and len(self.points) > 2:
                break
            elif key == ord('r'):
                self.points = []

        cv2.destroyWindow(self.window_name)

        # Create mask
        mask = np.zeros(self.original_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.points, np.int32)], 255)

        logger.info(f"ROI selected with {len(self.points)} points")
        return mask, self.points

class StoneDetector:
    """Class untuk deteksi dan analisis batuan"""
    def __init__(self, config: VideoConfig):
        self.config = config

        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=20,
            varThreshold=int(16 * config.detection_sensitivity),
            detectShadows=True
        )

        # Initialize tracking variables
        self.last_total_area = 0
        self.last_normalized_percentage = 0.0
        self.frame_count = 0

        logger.info("Stone Detector initialized with configuration:")
        logger.info(f"- Detection sensitivity: {config.detection_sensitivity}")
        logger.info(f"- Area range: {config.min_stone_area}-{config.max_stone_area} pixels")
        logger.info(f"- Percentage range: {config.min_percentage}-{config.max_percentage}%")

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Meningkatkan kualitas frame untuk deteksi yang lebih baik
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        enhanced = clahe.apply(gray)

        # Apply bilateral filter for noise reduction
        denoised = cv2.bilateralFilter(
            enhanced,
            self.config.blur_kernel_size,
            self.config.bilateral_sigma_color,
            self.config.bilateral_sigma_space
        )

        return denoised

    def filter_contours(self, contours: List[np.ndarray]) -> List[dict]:
        """
        Filter kontur berdasarkan kriteria bentuk dan ukuran
        """
        valid_stones = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.config.min_stone_area < area < self.config.max_stone_area:
                # Calculate shape metrics
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Calculate aspect ratio
                _, (width, height), _ = cv2.minAreaRect(cnt)
                aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0

                # Apply strict filtering
                if (self.config.min_circularity < circularity < self.config.max_circularity and
                    self.config.min_aspect_ratio < aspect_ratio < self.config.max_aspect_ratio):

                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        valid_stones.append({
                            'area': area,
                            'centroid': (cx, cy),
                            'contour': cnt,
                            'circularity': circularity,
                            'aspect_ratio': aspect_ratio
                        })

        return valid_stones

    def detect_stones(self, frame: np.ndarray, roi_mask: np.ndarray) -> List[dict]:
        """
        Deteksi batuan dalam frame
        """
        self.frame_count += 1

        # Enhance and mask frame
        enhanced_frame = self.enhance_frame(frame)
        masked_frame = cv2.bitwise_and(enhanced_frame, enhanced_frame, mask=roi_mask)

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(masked_frame)

        # Post-process mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find and filter contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stones = self.filter_contours(contours)

        # Handle sudden area changes
        current_total_area = sum(stone['area'] for stone in stones)
        if self.last_total_area > 0:
            area_change = abs(current_total_area - self.last_total_area) / self.last_total_area

            if area_change > self.config.max_area_change:
                # Apply smoothing
                current_total_area = (self.config.smoothing_factor * current_total_area +
                                    (1 - self.config.smoothing_factor) * self.last_total_area)

                # Adjust stone areas
                if stones:
                    adjustment_factor = current_total_area / sum(stone['area'] for stone in stones)
                    for stone in stones:
                        stone['area'] *= adjustment_factor

        self.last_total_area = current_total_area
        return stones

    def calculate_coverage_density(self, stones: List[dict], roi_area: int) -> float:
        """
        Hitung density coverage dengan normalisasi yang lebih akurat
        """
        if not stones or roi_area <= 0:
            return 0.0

        # Calculate total stone area
        total_stone_area = sum(stone['area'] for stone in stones)

        # Calculate base percentage
        try:
            base_percentage = (total_stone_area / roi_area) * 100
        except (ZeroDivisionError, ValueError):
            base_percentage = 0.0

        # Calculate spatial distribution
        centroids = np.array([stone['centroid'] for stone in stones])
        if len(centroids) > 1:
            distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    normalized_dist = dist / np.sqrt(roi_area)
                    distances.append(normalized_dist)

            avg_distance = np.mean(distances) if distances else 0
            density_factor = 1.0 / (1.0 + avg_distance / self.config.distance_normalization)
        else:
            density_factor = 0.5

        # Calculate final percentage
        final_percentage = base_percentage * (1 + (density_factor * self.config.density_weight))

        # Apply bounds
        final_percentage = max(self.config.min_percentage,
                             min(final_percentage, self.config.max_percentage))

        # Apply temporal smoothing
        smoothed_percentage = (self.config.smoothing_factor * final_percentage +
                             (1 - self.config.smoothing_factor) * self.last_normalized_percentage)

        self.last_normalized_percentage = smoothed_percentage
        return smoothed_percentage

class VideoProcessor:
    """Class untuk pemrosesan video dan analisis batuan"""
    def __init__(self, config: VideoConfig):
        self.config = config
        self.roi_selector = InteractiveROISelector()
        self.stone_detector = StoneDetector(config)
        self.roi_mask = None
        self.roi_points = []
        self.roi_area = 0

        logger.info("Video Processor initialized")

    def get_roi_area(self, frame: np.ndarray, roi_points: List[Tuple[int, int]]) -> np.ndarray:
        """Extract ROI area dari frame"""
        x_coords = [p[0] for p in roi_points]
        y_coords = [p[1] for p in roi_points]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        return frame[y1:y2, x1:x2]

    def calculate_total_stone_area(self, stones: List[dict]) -> int:
        """Hitung total area batuan dengan menghindari overlap"""
        stone_mask = np.zeros_like(self.roi_mask)

        # Gambar semua kontur batu pada mask
        for stone in stones:
            cv2.drawContours(stone_mask, [stone['contour']], -1, 255, -1)

        # Hitung area total
        return cv2.countNonZero(stone_mask)

    def _visualize_frame(self, frame: np.ndarray, roi_points: List[Tuple[int, int]],
                        stones: List[dict], roi_area: int) -> np.ndarray:
        """Visualisasi frame dengan anotasi"""
        frame_vis = frame.copy()

        # Draw ROI polygon
        cv2.polylines(frame_vis, [np.array(roi_points, np.int32)], True, (0, 255, 0), 2)

        # Calculate statistics
        total_stone_area = self.calculate_total_stone_area(stones)
        coverage_percentage = self.stone_detector.calculate_coverage_density(stones, roi_area)

        # Add overlay for statistics
        overlay = frame_vis.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame_vis, 0.5, 0, frame_vis)

        # Add statistics text
        cv2.putText(frame_vis, f"Stones Detected: {len(stones)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_vis, f"Coverage: {coverage_percentage:.2f}%", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_vis, f"Total Area: {total_stone_area} px", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw stones with information
        for stone in stones:
            # Draw contour
            cv2.drawContours(frame_vis, [stone['contour']], -1, (0, 0, 255), 2)

            # Draw centroid
            cx, cy = stone['centroid']
            cv2.circle(frame_vis, (cx, cy), 4, (255, 0, 0), -1)

            # Add stone information
            label = f"{int(stone['area'])}px"
            cv2.putText(frame_vis, label, (cx + 5, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return frame_vis

    def process_video(self, video_path: str, output_dir: str):
        """Process video dan generate analisis"""
        logger.info("Starting video processing")
        logger.info(f"Input video: {video_path}")
        logger.info(f"Output directory: {output_dir}")

        # Setup directories
        dirs = {
            'frames': os.path.join(output_dir, 'frames'),
            'roi_frames': os.path.join(output_dir, 'roi_frames'),
            'periodic_frames': os.path.join(output_dir, 'periodic_frames')
        }

        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_analysis = fps * self.config.analysis_interval

        logger.info(f"Video properties:")
        logger.info(f"- FPS: {fps}")
        logger.info(f"- Resolution: {frame_size}")
        logger.info(f"- Total frames: {total_frames}")
        logger.info(f"- Duration: {total_frames/fps:.2f} seconds")

        # Select ROI
        logger.info("Select ROI area - Click points and press 'C' when done")
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")

        self.roi_mask, self.roi_points = self.roi_selector.select_roi(first_frame)
        self.roi_area = cv2.countNonZero(self.roi_mask)
        logger.info(f"ROI area selected: {self.roi_area} pixels")

        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Full video output
        out_full = cv2.VideoWriter(
            os.path.join(output_dir, 'processed_full.avi'),
            fourcc, fps, frame_size, isColor=True
        )

        # ROI video output
        roi_width = max(p[0] for p in self.roi_points) - min(p[0] for p in self.roi_points)
        roi_height = max(p[1] for p in self.roi_points) - min(p[1] for p in self.roi_points)
        out_roi = cv2.VideoWriter(
            os.path.join(output_dir, 'processed_roi.avi'),
            fourcc, fps, (roi_width, roi_height), isColor=True
        )

        if not out_full.isOpened() or not out_roi.isOpened():
            raise ValueError("Failed to create video writers")

        # Initialize progress tracking
        progress = ProcessingProgress(total_frames, "Processing video")
        analysis_data = {}

        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Update progress
                progress.update()

                # Detect stones
                stones = self.stone_detector.detect_stones(frame, self.roi_mask)

                # Visualize and save
                frame_vis = self._visualize_frame(frame, self.roi_points, stones, self.roi_area)
                roi_area = self.get_roi_area(frame_vis, self.roi_points)

                # Write to videos
                out_full.write(frame_vis)
                out_roi.write(roi_area)

                # Periodic analysis
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_count % frames_per_analysis == 0:
                    timestamp = frame_count / fps

                    # Calculate metrics
                    total_stone_area = self.calculate_total_stone_area(stones)
                    coverage_percentage = self.stone_detector.calculate_coverage_density(
                        stones, self.roi_area)

                    # Save analysis data
                    analysis_data[int(timestamp)] = {
                        'timestamp': timestamp,
                        'stones_detected': len(stones),
                        'coverage_percentage': round(coverage_percentage, 2),
                        'total_stone_area': round(total_stone_area, 2),
                        'roi_area': self.roi_area,
                        'individual_areas': [round(stone['area'], 2) for stone in stones]
                    }

                    # Save periodic frame
                    cv2.imwrite(
                        os.path.join(dirs['periodic_frames'], f'periodic_{int(timestamp)}s.jpg'),
                        roi_area
                    )

                # Save frames with detections
                if stones:
                    cv2.imwrite(
                        os.path.join(dirs['frames'], f'frame_{frame_count:04d}.jpg'),
                        frame_vis
                    )
                    cv2.imwrite(
                        os.path.join(dirs['roi_frames'], f'roi_{frame_count:04d}.jpg'),
                        roi_area
                    )

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise
        finally:
            # Cleanup
            progress.close()
            cap.release()
            out_full.release()
            out_roi.release()

        # Save analysis data
        analysis_path = os.path.join(output_dir, 'analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=4)

        logger.info("Processing complete!")
        logger.info(f"Analysis data saved to: {analysis_path}")
        logger.info(f"Results saved to: {output_dir}")

        # Return summary
        return {
            'total_frames_processed': total_frames,
            'analysis_intervals': len(analysis_data),
            'detection_frames': len(os.listdir(dirs['frames'])),
            'output_directory': output_dir
        }

def main():
    """Main function untuk menjalankan program"""
    try:
        # Print welcome message
        print("\n" + "="*50)
        print("Stone Detection System v3.4")
        print("="*50 + "\n")

        # Initialize configuration
        config = VideoConfig()

        # Get video path
        default_video_path = r"D:\Mbkm\Magang\PDU\CompVis\Preparing Dataset\23-11-2024\Trial Dataset\dataset_asli_pdu_2.mp4"
        video_path = input(f"Enter video path (press Enter for default: {default_video_path}): ").strip()
        if not video_path:
            video_path = default_video_path

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output_dir = os.path.join(
            os.path.dirname(video_path),
            f"analysis_results_{timestamp}"
        )

        output_dir = input(f"Enter output directory (press Enter for default: {default_output_dir}): ").strip()
        if not output_dir:
            output_dir = default_output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize processor
        processor = VideoProcessor(config)

        # Process video
        logger.info("Starting processing pipeline...")
        results = processor.process_video(video_path, output_dir)

        # Print summary
        print("\n" + "="*50)
        print("Processing Summary:")
        print("="*50)
        print(f"Total frames processed: {results['total_frames_processed']}")
        print(f"Analysis intervals: {results['analysis_intervals']}")
        print(f"Frames with detections: {results['detection_frames']}")
        print(f"\nResults saved to: {results['output_directory']}")
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        input("\nPress Enter to exit...")