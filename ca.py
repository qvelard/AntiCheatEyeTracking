import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

from eyetrax import GazeEstimator, run_9_point_calibration
import cv2
import pyautogui
import numpy as np
import time
from collections import deque

# Create estimator and calibrate
print("🎯 Initializing Eye Tracking System...")
estimator = GazeEstimator()
print("📐 Starting 9-point calibration...")
run_9_point_calibration(estimator)
print("✅ Calibration complete!")

# # Save model
# estimator.save_model("gaze_model.pkl")

# # Load model
# estimator = GazeEstimator()
# estimator.load_model("gaze_model.pkl")

# UI Configuration
size = pyautogui.size()
cap = cv2.VideoCapture(0)
print(f"🖥️  Screen size detected: {size.width}x{size.height}")
print("🎥 Starting video capture...")

# Tracking variables
positions = deque(maxlen=20)  # For suspicious behavior detection (increased)
smoothed_positions = deque(maxlen=15)  # For moving average (large window for ultra-smooth)
warning_count = 0
start_time = time.time()
frame_count = 0

# Smoothing parameters (adjustable in real-time)
smoothing_window_size = 15  # Can be adjusted with + and - keys
smoothing_alpha = 0.3  # Exponential smoothing factor

# Create UI window
cv2.namedWindow('Eye Tracking Anti-Cheat', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Eye Tracking Anti-Cheat', 800, 600)

class ObjectiveCheatDetector:
    """
    Objective, measurable cheat detection with multiple behavioral indicators
    """
    def __init__(self):
        self.baseline_metrics = {}
        self.detection_history = deque(maxlen=100)
        self.calibration_period = 30.0  # seconds to establish baseline
        self.start_time = time.time()
        
        # Metrics to track
        self.metrics = {
            'off_screen_duration': 0,
            'fixation_clusters': [],
            'saccade_velocities': [],
            'attention_dispersion': [],
            'temporal_patterns': [],
            'gaze_entropy': []
        }
        
    def calculate_objective_metrics(self, positions_window, current_time):
        """Calculate objective, measurable behavioral metrics"""
        if len(positions_window) < 5:
            return {}
        
        positions_array = np.array(list(positions_window))
        
        # 1. Spatial Dispersion (Standard Deviation of positions)
        spatial_dispersion = np.std(positions_array, axis=0).mean()
        
        # 2. Temporal Consistency (Movement velocity variance)
        velocities = []
        for i in range(1, len(positions_array)):
            dx = positions_array[i][0] - positions_array[i-1][0]
            dy = positions_array[i][1] - positions_array[i-1][1]
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        velocity_variance = np.var(velocities) if len(velocities) > 1 else 0
        avg_velocity = np.mean(velocities) if velocities else 0
        
        # 3. Screen Coverage (How much of screen is being used)
        x_range = np.max(positions_array[:, 0]) - np.min(positions_array[:, 0])
        y_range = np.max(positions_array[:, 1]) - np.min(positions_array[:, 1])
        screen_coverage = (x_range * y_range) / (size.width * size.height)
        
        # 4. Off-screen percentage
        on_screen_positions = [
            pos for pos in positions_window 
            if 0 <= pos[0] <= size.width and 0 <= pos[1] <= size.height
        ]
        off_screen_percentage = (len(positions_window) - len(on_screen_positions)) / len(positions_window)
        
        # 5. Fixation clustering (detect if staring at specific areas)
        fixation_score = self._calculate_fixation_clustering(positions_array)
        
        # 6. Gaze entropy (randomness of gaze patterns)
        gaze_entropy = self._calculate_gaze_entropy(positions_array)
        
        return {
            'spatial_dispersion': spatial_dispersion,
            'velocity_variance': velocity_variance,
            'avg_velocity': avg_velocity,
            'screen_coverage': screen_coverage,
            'off_screen_percentage': off_screen_percentage,
            'fixation_clustering': fixation_score,
            'gaze_entropy': gaze_entropy,
            'timestamp': current_time
        }
    
    def _calculate_fixation_clustering(self, positions):
        """Calculate how clustered the fixations are (suspicious if too clustered off-screen)"""
        if len(positions) < 3:
            return 0
        
        # Use k-means like clustering to find fixation centers
        from collections import defaultdict
        clusters = defaultdict(list)
        cluster_threshold = 50  # pixels
        
        for pos in positions:
            found_cluster = False
            for center, cluster_positions in clusters.items():
                distance = np.sqrt((pos[0] - center[0])**2 + (pos[1] - center[1])**2)
                if distance < cluster_threshold:
                    cluster_positions.append(pos)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters[tuple(pos)] = [pos]
        
        # Calculate clustering score
        if len(clusters) == 0:
            return 0
        
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        max_cluster_size = max(cluster_sizes)
        clustering_score = max_cluster_size / len(positions)
        
        return clustering_score
    
    def _calculate_gaze_entropy(self, positions):
        """Calculate entropy of gaze distribution"""
        if len(positions) < 5:
            return 0
        
        # Create grid and calculate distribution
        grid_size = 10
        x_bins = np.linspace(0, size.width, grid_size)
        y_bins = np.linspace(0, size.height, grid_size)
        
        hist, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=[x_bins, y_bins])
        hist = hist.flatten()
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def update_baseline(self, current_metrics):
        """Update baseline during calibration period"""
        runtime = time.time() - self.start_time
        
        if runtime < self.calibration_period:
            # Still in calibration period
            for key, value in current_metrics.items():
                if key != 'timestamp':
                    if key not in self.baseline_metrics:
                        self.baseline_metrics[key] = []
                    self.baseline_metrics[key].append(value)
            return False  # Not ready for detection yet
        
        # Calibration complete - calculate baseline averages
        if not hasattr(self, 'baseline_calculated'):
            for key, values in self.baseline_metrics.items():
                self.baseline_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            self.baseline_calculated = True
            print("🎯 Baseline calibration complete - objective detection active")
        
        return True
    
    def calculate_cheat_probability(self, current_metrics):
        """Calculate objective cheat probability based on deviation from baseline"""
        if not hasattr(self, 'baseline_calculated'):
            return 0.0, {}
        
        risk_factors = {}
        total_risk = 0
        
        # Define weights for different factors
        weights = {
            'off_screen_percentage': 0.30,  # Highest weight - looking away
            'fixation_clustering': 0.20,    # High clustering off-screen
            'spatial_dispersion': 0.15,     # Unusual movement patterns
            'velocity_variance': 0.15,      # Erratic movements
            'screen_coverage': 0.10,        # Too limited or too wide coverage
            'gaze_entropy': 0.10           # Unnatural gaze patterns
        }
        
        for metric, weight in weights.items():
            if metric in current_metrics and metric in self.baseline_metrics:
                current_val = current_metrics[metric]
                baseline = self.baseline_metrics[metric]
                
                # Calculate z-score (standard deviations from baseline)
                if baseline['std'] > 0:
                    z_score = abs(current_val - baseline['mean']) / baseline['std']
                    
                    # Convert z-score to risk (sigmoid function)
                    risk = 1 / (1 + np.exp(-2 * (z_score - 2)))  # Risk increases after 2 std devs
                    
                    risk_factors[metric] = {
                        'current': current_val,
                        'baseline_mean': baseline['mean'],
                        'z_score': z_score,
                        'risk': risk,
                        'weight': weight
                    }
                    
                    total_risk += risk * weight
        
        return min(total_risk * 100, 100), risk_factors

# Global detector instance
cheat_detector = ObjectiveCheatDetector()

def check_sus():
    global warning_count, cheat_detector
    
    if len(positions) < 10:
        return False, 0.0, {}
    
    # Calculate objective metrics
    current_metrics = cheat_detector.calculate_objective_metrics(positions, time.time())
    
    # Update baseline (returns False during calibration period)
    if not cheat_detector.update_baseline(current_metrics):
        return False, 0.0, {}  # Still calibrating
    
    # Calculate objective cheat probability
    cheat_prob, risk_factors = cheat_detector.calculate_cheat_probability(current_metrics)
    
    # Determine if this is a warning
    is_warning = cheat_prob > 70  # Threshold for warning
    
    if is_warning:
        warning_count += 1
        print(f"⚠️  WARNING #{warning_count}: Cheat probability: {cheat_prob:.1f}%")
        
        # Print detailed breakdown
        print("📊 Risk Factor Breakdown:")
        for metric, data in risk_factors.items():
            if data['risk'] > 0.1:  # Only show significant risks
                print(f"   - {metric}: {data['current']:.2f} (baseline: {data['baseline_mean']:.2f}) "
                      f"Z-score: {data['z_score']:.1f}, Risk: {data['risk']*100:.1f}%")
    
    return is_warning, cheat_prob, risk_factors

def calculate_moving_average(positions_deque, alpha=None):
    """Calculate advanced moving average of positions for ultra-smooth tracking"""
    global smoothing_alpha
    if alpha is None:
        alpha = smoothing_alpha
        
    if len(positions_deque) == 0:
        return None
    
    if len(positions_deque) == 1:
        return positions_deque[0]
    
    # Multi-layer smoothing approach
    # Layer 1: Simple moving average for base smoothing
    simple_avg_x = sum(pos[0] for pos in positions_deque) / len(positions_deque)
    simple_avg_y = sum(pos[1] for pos in positions_deque) / len(positions_deque)
    
    # Layer 2: Exponential weighted moving average for trend following
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    
    for i, (x, y) in enumerate(positions_deque):
        # Exponential weight: newer positions get much higher weight
        weight = alpha * ((1 - alpha) ** (len(positions_deque) - 1 - i))
        
        weighted_x += x * weight
        weighted_y += y * weight
        total_weight += weight
    
    if total_weight > 0:
        exp_avg_x = weighted_x / total_weight
        exp_avg_y = weighted_y / total_weight
        
        # Layer 3: Gaussian-weighted average for ultra-smooth center
        gaussian_weighted_x = 0
        gaussian_weighted_y = 0
        gaussian_total_weight = 0
        
        center_index = len(positions_deque) // 2
        sigma = len(positions_deque) / 4  # Standard deviation for Gaussian
        
        for i, (x, y) in enumerate(positions_deque):
            # Gaussian weight centered on the middle of the window
            gaussian_weight = np.exp(-0.5 * ((i - center_index) / sigma) ** 2)
            gaussian_weighted_x += x * gaussian_weight
            gaussian_weighted_y += y * gaussian_weight
            gaussian_total_weight += gaussian_weight
        
        if gaussian_total_weight > 0:
            gaussian_avg_x = gaussian_weighted_x / gaussian_total_weight
            gaussian_avg_y = gaussian_weighted_y / gaussian_total_weight
        else:
            gaussian_avg_x, gaussian_avg_y = exp_avg_x, exp_avg_y
        
        # Combine all three methods with adaptive weights based on movement stability
        positions_array = np.array(list(positions_deque))
        movement_variance = np.var(positions_array, axis=0).mean()
        
        # High variance = more movement = use exponential (more responsive)
        # Low variance = stable = use Gaussian (ultra-smooth)
        variance_threshold = 1000
        if movement_variance > variance_threshold:
            # High movement: favor exponential average
            final_x = exp_avg_x * 0.7 + simple_avg_x * 0.2 + gaussian_avg_x * 0.1
            final_y = exp_avg_y * 0.7 + simple_avg_y * 0.2 + gaussian_avg_y * 0.1
        else:
            # Low movement: favor Gaussian smoothing
            final_x = gaussian_avg_x * 0.6 + exp_avg_x * 0.3 + simple_avg_x * 0.1
            final_y = gaussian_avg_y * 0.6 + exp_avg_y * 0.3 + simple_avg_y * 0.1
        
        # Outlier detection and correction
        if len(positions_deque) >= 5:
            recent_positions = list(positions_deque)[-5:]
            recent_avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
            recent_avg_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
            
            # Calculate distance from recent trend
            distance = ((final_x - recent_avg_x)**2 + (final_y - recent_avg_y)**2)**0.5
            
            # Adaptive outlier threshold based on recent movement
            base_threshold = 80
            movement_factor = min(movement_variance / 500, 2.0)
            outlier_threshold = base_threshold * (1 + movement_factor)
            
            if distance > outlier_threshold:
                # Blend with recent average to reduce outlier influence
                blend_factor = min(distance / (outlier_threshold * 1.5), 0.8)
                final_x = final_x * (1 - blend_factor) + recent_avg_x * blend_factor
                final_y = final_y * (1 - blend_factor) + recent_avg_y * blend_factor
        
        return (final_x, final_y)
    
    return positions_deque[-1]  # Fallback to last position

def draw_ui_overlay(frame, current_pos, smoothed_pos, is_warning, fps, cheat_prob=0, risk_factors={}):
    """Draw UI overlay with tracking information"""
    h, w = frame.shape[:2]
    
    # Create overlay
    overlay = frame.copy()
    
    # Background for stats (enlarged for more info)
    stats_height = 300 if risk_factors else 200
    cv2.rectangle(overlay, (10, 10), (w-10, stats_height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (w-10, stats_height), (255, 255, 255), 2)
    
    # Title
    cv2.putText(overlay, "EYE TRACKING ANTI-CHEAT SYSTEM", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Stats
    runtime = time.time() - start_time
    cv2.putText(overlay, f"Runtime: {runtime:.1f}s | FPS: {fps:.1f} | Warnings: {warning_count}", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Smoothing parameters display
    cv2.putText(overlay, f"Smoothing: Window={len(smoothed_positions)}/{smoothing_window_size} | Alpha={smoothing_alpha:.2f}", 
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    # Current position (raw)
    if current_pos:
        x, y = current_pos
        cv2.putText(overlay, f"Raw Position: ({x:.0f}, {y:.0f})", 
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Smoothed position
    if smoothed_pos:
        sx, sy = smoothed_pos
        cv2.putText(overlay, f"Smoothed Position: ({sx:.0f}, {sy:.0f})", 
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Calculate smoothing distance (how much smoothing applied)
        if current_pos:
            smooth_distance = ((sx - x)**2 + (sy - y)**2)**0.5
            cv2.putText(overlay, f"Smoothing Distance: {smooth_distance:.1f}px", 
                        (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        # Screen bounds status (using smoothed position)
        on_screen = (0 <= sx <= size.width and 0 <= sy <= size.height)
        status_color = (0, 255, 0) if on_screen else (0, 0, 255)
        status_text = "ON SCREEN" if on_screen else "OFF SCREEN"
        cv2.putText(overlay, f"Status: {status_text}", 
                    (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Controls help
    cv2.putText(overlay, "Controls: [+/-] Window Size | [w/s] Alpha | [r] Reset", 
                (20, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Objective cheat probability display
    runtime = time.time() - cheat_detector.start_time if 'cheat_detector' in globals() else 0
    if runtime < 30:
        # During calibration
        calibration_progress = (runtime / 30) * 100
        cv2.putText(overlay, f"CALIBRATING BASELINE: {calibration_progress:.1f}%", 
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        # Show objective cheat probability
        prob_color = (0, 0, 255) if cheat_prob > 70 else (0, 255, 255) if cheat_prob > 40 else (0, 255, 0)
        cv2.putText(overlay, f"CHEAT PROBABILITY: {cheat_prob:.1f}%", 
                    (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, prob_color, 2)
        
        # Show top risk factors
        if risk_factors:
            y_offset = 240
            cv2.putText(overlay, "TOP RISK FACTORS:", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            y_offset += 20
            
            # Sort by risk level and show top 3
            sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1]['risk'], reverse=True)[:3]
            for metric, data in sorted_risks:
                if data['risk'] > 0.1:  # Only show significant risks
                    risk_text = f"{metric}: {data['risk']*100:.1f}% (Z: {data['z_score']:.1f})"
                    cv2.putText(overlay, risk_text, (25, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 150), 1)
                    y_offset += 15
    
    # Warning indicator
    if is_warning:
        cv2.putText(overlay, "🚨 OBJECTIVE CHEAT DETECTION!", 
                    (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Flash effect
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 10)
    
    # Position history visualization (raw trajectory)
    if len(positions) > 1:
        # Draw raw trajectory (thin line)
        for i in range(1, len(positions)):
            prev_x, prev_y = positions[i-1]
            curr_x, curr_y = positions[i]
            
            # Map to frame coordinates
            prev_frame_x = int((prev_x / size.width) * w) if size.width > 0 else w//2
            prev_frame_y = int((prev_y / size.height) * h) if size.height > 0 else h//2
            curr_frame_x = int((curr_x / size.width) * w) if size.width > 0 else w//2
            curr_frame_y = int((curr_y / size.height) * h) if size.height > 0 else h//2
            
            # Clamp to frame bounds
            prev_frame_x = max(0, min(w-1, prev_frame_x))
            prev_frame_y = max(0, min(h-1, prev_frame_y))
            curr_frame_x = max(0, min(w-1, curr_frame_x))
            curr_frame_y = max(0, min(h-1, curr_frame_y))
            
            # Thin gray line for raw data
            cv2.line(overlay, (prev_frame_x, prev_frame_y), (curr_frame_x, curr_frame_y), (100, 100, 100), 1)
    
    # Smoothed trajectory (thick colored line)
    if len(smoothed_positions) > 1:
        for i in range(1, len(smoothed_positions)):
            prev_x, prev_y = smoothed_positions[i-1]
            curr_x, curr_y = smoothed_positions[i]
            
            # Map to frame coordinates
            prev_frame_x = int((prev_x / size.width) * w) if size.width > 0 else w//2
            prev_frame_y = int((prev_y / size.height) * h) if size.height > 0 else h//2
            curr_frame_x = int((curr_x / size.width) * w) if size.width > 0 else w//2
            curr_frame_y = int((curr_y / size.height) * h) if size.height > 0 else h//2
            
            # Clamp to frame bounds
            prev_frame_x = max(0, min(w-1, prev_frame_x))
            prev_frame_y = max(0, min(h-1, prev_frame_y))
            curr_frame_x = max(0, min(w-1, curr_frame_x))
            curr_frame_y = max(0, min(h-1, curr_frame_y))
            
            # Color based on position in history (newer = brighter)
            alpha = i / len(smoothed_positions)
            color = (int(255 * alpha), int(150 * alpha), 0)
            
            cv2.line(overlay, (prev_frame_x, prev_frame_y), (curr_frame_x, curr_frame_y), color, 3)
    
    # Draw position dots
    # Raw position (small red dot)
    if current_pos:
        x, y = current_pos
        frame_x = int((x / size.width) * w) if size.width > 0 else w//2
        frame_y = int((y / size.height) * h) if size.height > 0 else h//2
        frame_x = max(0, min(w-1, frame_x))
        frame_y = max(0, min(h-1, frame_y))
        cv2.circle(overlay, (frame_x, frame_y), 4, (0, 0, 255), -1)  # Small red dot
    
    # Smoothed position (large green dot)
    if smoothed_pos:
        sx, sy = smoothed_pos
        smooth_frame_x = int((sx / size.width) * w) if size.width > 0 else w//2
        smooth_frame_y = int((sy / size.height) * h) if size.height > 0 else h//2
        smooth_frame_x = max(0, min(w-1, smooth_frame_x))
        smooth_frame_y = max(0, min(h-1, smooth_frame_y))
        cv2.circle(overlay, (smooth_frame_x, smooth_frame_y), 8, (0, 255, 0), -1)  # Large green dot
        cv2.circle(overlay, (smooth_frame_x, smooth_frame_y), 12, (255, 255, 255), 2)  # White border
    
    return overlay

print("🚀 Starting real-time tracking... Press 'q' to quit")

while True:
    frame_count += 1
    
    # Extract features from frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break
    
    # Calculate FPS
    if frame_count % 30 == 0:  # Update every 30 frames
        current_time = time.time()
        fps = 30 / (current_time - start_time + 1e-6) if frame_count > 30 else 0
    else:
        fps = 0
    
    features, blink = estimator.extract_features(frame)
    current_pos = None
    is_warning = False

    # Predict screen coordinates
    smoothed_pos = None
    if features is not None and not blink:
        x, y = estimator.predict([features])[0]
        current_pos = (x, y)
        
        # Add to positions for suspicious behavior detection
        positions.append((x, y))
        
        # Add to smoothed positions for visualization
        smoothed_positions.append((x, y))
        
        # Adjust smoothed_positions window size if needed
        if len(smoothed_positions) > smoothing_window_size:
            # Create new deque with adjusted size
            new_positions = deque(list(smoothed_positions)[-smoothing_window_size:], maxlen=smoothing_window_size)
            smoothed_positions = new_positions
        
        # Calculate moving average
        smoothed_pos = calculate_moving_average(smoothed_positions)
        
        # Check for suspicious behavior (using raw positions)
        is_warning, cheat_probability, risk_factors = check_sus()
    elif blink:
        # Show blink detection
        cv2.putText(frame, "BLINK DETECTED", (50, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    else:
        # No face/features detected
        is_warning, cheat_probability, risk_factors = False, 0.0, {}
    
    # Draw UI overlay with both raw and smoothed positions
    display_frame = draw_ui_overlay(frame, current_pos, smoothed_pos, is_warning, fps, 
                                  cheat_probability, risk_factors)
    
    # Show frame
    cv2.imshow('Eye Tracking Anti-Cheat', display_frame)
    
    # Handle keyboard input for real-time smoothing control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):  # Increase smoothing window
        smoothing_window_size = min(smoothing_window_size + 1, 25)
        print(f"Smoothing window increased to {smoothing_window_size}")
    elif key == ord('-'):  # Decrease smoothing window
        smoothing_window_size = max(smoothing_window_size - 1, 3)
        print(f"Smoothing window decreased to {smoothing_window_size}")
    elif key == ord('w'):  # Increase alpha (more responsive)
        smoothing_alpha = min(smoothing_alpha + 0.05, 0.8)
        print(f"Smoothing alpha increased to {smoothing_alpha:.2f} (more responsive)")
    elif key == ord('s'):  # Decrease alpha (more smooth)
        smoothing_alpha = max(smoothing_alpha - 0.05, 0.1)
        print(f"Smoothing alpha decreased to {smoothing_alpha:.2f} (more smooth)")
    elif key == ord('r'):  # Reset to defaults
        smoothing_window_size = 15
        smoothing_alpha = 0.3
        smoothed_positions.clear()
        print("Smoothing parameters reset to defaults")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print(f"\n📊 SESSION SUMMARY:")
print(f"   Total Runtime: {time.time() - start_time:.1f} seconds")
print(f"   Total Warnings: {warning_count}")
print(f"   Frames Processed: {frame_count}")
if warning_count > 0:
    print(f"   ⚠️  Suspicious activity detected {warning_count} times!")
else:
    print(f"   ✅ No suspicious activity detected")
print("👋 Eye tracking session ended.")
