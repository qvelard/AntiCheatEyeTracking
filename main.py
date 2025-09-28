
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

from eyetrax import GazeEstimator, run_9_point_calibration
import cv2
import pyautogui
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
# Flask server
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)  # Allow all origins; restrict in production
from flask_socketio import SocketIO, emit
import base64
import io
socketio = SocketIO(app, cors_allowed_origins="*")
from flask_socketio import SocketIO, emit
import base64
import io
latest_frame_path = 'latest_frame.png'
tracking_thread = None
tracking_active = False
session_summary = {}
socketio = SocketIO(app, cors_allowed_origins="*")


# Global variables for tracking
estimator = None
size = None
cap = None
positions = deque(maxlen=20)
smoothed_positions = deque(maxlen=15)
warning_count = 0
start_time = None
frame_count = 0
smoothing_window_size = 15
smoothing_alpha = 0.3
control_points_enabled = False
control_validator = None
cheat_detector = None

@dataclass
class ControlPoint:
    """Structure for control point validation"""
    timestamp: float
    expected_position: Tuple[int, int]
    predicted_position: Tuple[int, int]
    distance_error: float
    is_valid: bool

class ControlPointValidator:
    """
    Validates eye-tracking accuracy by injecting hidden control points
    and comparing predicted gaze with ground-truth locations
    """
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.control_history = deque(maxlen=50)
        self.last_control_time = 0
        self.control_interval = 15.0  # Inject control point every 15 seconds
        self.tolerance_pixels = 100  # Acceptable error in pixels
        self.current_control_point = None
        self.control_display_duration = 0.5  # Show control point for 0.5 seconds
        
        # Statistics
        self.total_tests = 0
        self.failed_tests = 0
        self.accuracy_threshold = 0.7  # 70% accuracy required
        
    def should_inject_control_point(self) -> bool:
        """Determine if it's time to inject a new control point"""
        current_time = time.time()
        return (current_time - self.last_control_time) >= self.control_interval
    
    def generate_control_point(self) -> Tuple[int, int]:
        """Generate a random control point position on screen"""
        margin_x = int(self.screen_width * 0.1)
        margin_y = int(self.screen_height * 0.1)
        
        x = np.random.randint(margin_x, self.screen_width - margin_x)
        y = np.random.randint(margin_y, self.screen_height - margin_y)
        return (x, y)
    
    def start_control_test(self) -> Tuple[int, int]:
        """Start a new control point test"""
        control_pos = self.generate_control_point()
        self.current_control_point = {
            'position': control_pos,
            'start_time': time.time(),
            'completed': False
        }
        self.last_control_time = time.time()
        return control_pos
    
    def check_control_point(self, predicted_gaze: Tuple[int, int]) -> bool:
        """Check if current control point test is active and validate prediction"""
        if self.current_control_point is None:
            return False
            
        current_time = time.time()
        elapsed = current_time - self.current_control_point['start_time']
        
        # Control point display finished
        if elapsed >= self.control_display_duration and not self.current_control_point['completed']:
            expected_pos = self.current_control_point['position']
            
            # Calculate distance error
            distance_error = ((predicted_gaze[0] - expected_pos[0])**2 + 
                            (predicted_gaze[1] - expected_pos[1])**2)**0.5
            
            # Create control point record
            is_valid = distance_error <= self.tolerance_pixels
            control_point = ControlPoint(
                timestamp=current_time,
                expected_position=expected_pos,
                predicted_position=predicted_gaze,
                distance_error=distance_error,
                is_valid=is_valid
            )
            
            self.control_history.append(control_point)
            self.total_tests += 1
            if not is_valid:
                self.failed_tests += 1
                
            self.current_control_point['completed'] = True
            print(f"🎯 Control Point: Distance={distance_error:.1f}px {'✅' if is_valid else '❌'}")
            return True
            
        return False
    
    def should_display_control_point(self) -> Tuple[bool, Tuple[int, int]]:
        """Check if control point should be displayed"""
        if self.current_control_point is None:
            return False, (0, 0)
            
        elapsed = time.time() - self.current_control_point['start_time']
        if elapsed < self.control_display_duration:
            return True, self.current_control_point['position']
        else:
            return False, (0, 0)
    
    def get_accuracy_stats(self) -> Dict:
        """Get current accuracy statistics"""
        if self.total_tests == 0:
            return {'accuracy': 1.0, 'total_tests': 0, 'failed_tests': 0, 'avg_error': 0}
            
        accuracy = 1.0 - (self.failed_tests / self.total_tests)
        avg_error = np.mean([cp.distance_error for cp in self.control_history]) if self.control_history else 0
        
        return {
            'accuracy': accuracy,
            'total_tests': self.total_tests,
            'failed_tests': self.failed_tests,
            'avg_error': avg_error
        }
    
    def is_session_compromised(self) -> bool:
        """Determine if session should be marked as cheating"""
        if self.total_tests < 3:
            return False
            
        stats = self.get_accuracy_stats()
        return stats['accuracy'] < self.accuracy_threshold


def initialize_tracking():
    global estimator, size, cap, control_validator, cheat_detector, positions, smoothed_positions, warning_count, start_time, frame_count
    print("🎯 Initializing Eye Tracking System...")
    estimator = GazeEstimator()
    print("📐 Starting 9-point calibration...")
    run_9_point_calibration(estimator)
    print("✅ Calibration complete!")
    size = pyautogui.size()
    cap = cv2.VideoCapture(0)
    print(f"🖥️  Screen size detected: {size.width}x{size.height}")
    print("🎥 Starting video capture...")
    positions = deque(maxlen=20)
    smoothed_positions = deque(maxlen=15)
    warning_count = 0
    start_time = time.time()
    frame_count = 0
    cheat_detector = ObjectiveCheatDetector()
    if control_points_enabled:
        control_validator = ControlPointValidator(size.width, size.height)
    else:
        control_validator = None

class ObjectiveCheatDetector:
    """Objective, measurable cheat detection with multiple behavioral indicators"""
    def __init__(self):
        self.baseline_metrics = {}
        self.detection_history = deque(maxlen=100)
        self.calibration_period = 30.0  # seconds to establish baseline
        self.start_time = time.time()
        
        # Time series data for plotting
        self.cheat_probability_history = []
        self.timestamps = []
        self.risk_factors_history = []
        
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
        
        # 1. Spatial Dispersion
        spatial_dispersion = np.std(positions_array, axis=0).mean()
        
        # 2. Temporal Consistency
        velocities = []
        for i in range(1, len(positions_array)):
            dx = positions_array[i][0] - positions_array[i-1][0]
            dy = positions_array[i][1] - positions_array[i-1][1]
            velocity = np.sqrt(dx*dx + dy*dy)
            velocities.append(velocity)
        
        velocity_variance = np.var(velocities) if velocities else 0
        
        # 3. Off-screen percentage
        on_screen_count = sum(1 for x, y in positions_array 
                             if 0 <= x <= size.width and 0 <= y <= size.height)
        off_screen_percentage = (len(positions_array) - on_screen_count) / len(positions_array) * 100
        
        # 4. Fixation clustering
        if len(positions_array) > 3:
            distances = []
            for i in range(len(positions_array)):
                for j in range(i+1, len(positions_array)):
                    dist = np.sqrt((positions_array[i][0] - positions_array[j][0])**2 + 
                                 (positions_array[i][1] - positions_array[j][1])**2)
                    distances.append(dist)
            fixation_clustering = np.mean(distances) if distances else 0
        else:
            fixation_clustering = 0
        
        # 5. Screen coverage
        if len(positions_array) > 1:
            x_range = np.max(positions_array[:, 0]) - np.min(positions_array[:, 0])
            y_range = np.max(positions_array[:, 1]) - np.min(positions_array[:, 1])
            screen_coverage = (x_range * y_range) / (size.width * size.height) * 100
        else:
            screen_coverage = 0
        
        # 6. Gaze entropy (randomness measure)
        gaze_entropy = spatial_dispersion / max(1, np.mean(velocities)) if velocities else 0
        
        return {
            'spatial_dispersion': spatial_dispersion,
            'velocity_variance': velocity_variance,
            'off_screen_percentage': off_screen_percentage,
            'fixation_clustering': fixation_clustering,
            'screen_coverage': screen_coverage,
            'gaze_entropy': gaze_entropy,
            'timestamp': current_time
        }
    
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
            return False
        
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
        
        # Use static weights (simplified from RLHF)
        weights = {
            'off_screen_percentage': 0.40,  # Highest weight - looking away
            'fixation_clustering': 0.20,    # High clustering off-screen
            'spatial_dispersion': 0.15,     # Unusual movement patterns
            'velocity_variance': 0.15,      # Erratic movements
            'screen_coverage': 0.00,        # Too limited or too wide coverage
            'gaze_entropy': 0.10           # Unnatural gaze patterns
        }
        
        for metric, weight in weights.items():
            if metric in current_metrics and metric in self.baseline_metrics:
                current_val = current_metrics[metric]
                baseline = self.baseline_metrics[metric]
                
                # Calculate z-score (standard deviations from baseline)
                if baseline['std'] > 0:
                    z_score = abs(current_val - baseline['mean']) / baseline['std']
                    
                    # Convert z-score to risk (clamped at 3 std devs)
                    risk_score = min(100, (z_score / 3.0) * 100) * weight
                    total_risk += risk_score
                    
                    risk_factors[metric] = {
                        'current': current_val,
                        'baseline_mean': baseline['mean'],
                        'z_score': z_score,
                        'risk_score': risk_score,
                        'weight': weight
                    }
        
        # Store for plotting
        self.cheat_probability_history.append(total_risk)
        self.timestamps.append(time.time())
        self.risk_factors_history.append(risk_factors.copy())
        
        return total_risk, risk_factors

# Initialize detector
cheat_detector = ObjectiveCheatDetector()

def check_sus():
    """Check for suspicious behavior using objective metrics"""
    global warning_count
    
    if len(positions) < 5:
        return False, 0.0, {}
    
    # Calculate current metrics
    current_metrics = cheat_detector.calculate_objective_metrics(positions, time.time())
    
    # Update baseline (returns True when ready for detection)
    if not cheat_detector.update_baseline(current_metrics):
        return False, 0.0, {}
    
    # Calculate cheat probability
    cheat_prob, risk_factors = cheat_detector.calculate_cheat_probability(current_metrics)
    
    # Determine if warning
    is_warning = cheat_prob > 50.0
    if is_warning:
        warning_count += 1
    
    return is_warning, cheat_prob, risk_factors

def calculate_moving_average(positions_deque):
    """Calculate moving average with multiple smoothing methods"""
    if len(positions_deque) < 2:
        return positions_deque[-1] if positions_deque else (0, 0)
    
    positions_list = list(positions_deque)
    
    # Simple moving average
    avg_x = sum(pos[0] for pos in positions_list) / len(positions_list)
    avg_y = sum(pos[1] for pos in positions_list) / len(positions_list)
    
    # Exponential smoothing (recent positions have more weight)
    if len(positions_list) > 1:
        exp_x, exp_y = positions_list[0]
        for i in range(1, len(positions_list)):
            exp_x = smoothing_alpha * positions_list[i][0] + (1 - smoothing_alpha) * exp_x
            exp_y = smoothing_alpha * positions_list[i][1] + (1 - smoothing_alpha) * exp_y
        
        # Blend simple and exponential smoothing
        final_x = 0.7 * avg_x + 0.3 * exp_x
        final_y = 0.7 * avg_y + 0.3 * exp_y
        
        return (final_x, final_y)
    
    return positions_deque[-1]

def plot_session_analysis():
    """Plot comprehensive session analysis"""
    if len(cheat_detector.cheat_probability_history) < 10:
        print("⚠️ Not enough data for meaningful analysis (need at least 10 data points)")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Eye Tracking Session Analysis', fontsize=16, fontweight='bold')
    
    # Convert timestamps to relative time
    start_time_session = min(cheat_detector.timestamps)
    relative_times = [(t - start_time_session) / 60 for t in cheat_detector.timestamps]  # Minutes
    
    # 1. Cheat Probability Over Time
    ax1.plot(relative_times, cheat_detector.cheat_probability_history, 
            linewidth=2, color='#e74c3c', alpha=0.8)
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    ax1.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Cheat Probability (%)')
    ax1.set_title('Cheat Detection Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # 2. Distribution of Cheat Probabilities
    ax2.hist(cheat_detector.cheat_probability_history, bins=20, 
            color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(cheat_detector.cheat_probability_history), 
               color='red', linestyle='--', label=f'Mean: {np.mean(cheat_detector.cheat_probability_history):.1f}%')
    ax2.set_xlabel('Cheat Probability (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Cheat Probabilities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Risk Factor Contributions
    if cheat_detector.risk_factors_history:
        risk_factor_names = ['off_screen_percentage', 'fixation_clustering', 'spatial_dispersion', 
                           'velocity_variance', 'screen_coverage', 'gaze_entropy']
        avg_contributions = []
        
        for factor in risk_factor_names:
            contributions = []
            for risk_dict in cheat_detector.risk_factors_history:
                if factor in risk_dict:
                    contributions.append(risk_dict[factor].get('risk_score', 0))
            avg_contributions.append(np.mean(contributions) if contributions else 0)
        
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#27ae60', '#3498db', '#9b59b6']
        bars = ax3.bar(range(len(risk_factor_names)), avg_contributions, color=colors, alpha=0.7)
        ax3.set_xlabel('Risk Factors')
        ax3.set_ylabel('Average Contribution (%)')
        ax3.set_title('Average Risk Factor Contributions')
        ax3.set_xticks(range(len(risk_factor_names)))
        ax3.set_xticklabels([name.replace('_', ' ').title() for name in risk_factor_names], 
                           rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_contributions):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. Session Summary Statistics
    session_stats = {
        'Duration': f"{(cheat_detector.timestamps[-1] - cheat_detector.timestamps[0]) / 60:.1f} min",
        'Avg Cheat Prob': f"{np.mean(cheat_detector.cheat_probability_history):.1f}%",
        'Max Cheat Prob': f"{np.max(cheat_detector.cheat_probability_history):.1f}%",
        'Warnings': f"{warning_count}",
        'High Risk Events': f"{sum(1 for p in cheat_detector.cheat_probability_history if p > 70)}"
    }
    
    if control_validator is not None:
        control_stats = control_validator.get_accuracy_stats()
        session_stats.update({
            'Control Tests': f"{control_stats['total_tests']}",
            'Control Accuracy': f"{control_stats['accuracy']:.1%}",
            'Avg Control Error': f"{control_stats.get('avg_error', 0):.1f}px"
        })
    
    ax4.axis('off')
    stats_text = "Session Summary\n" + "="*20 + "\n"
    for key, value in session_stats.items():
        stats_text += f"{key}: {value}\n"
    
    # Add recommendation
    avg_prob = np.mean(cheat_detector.cheat_probability_history)
    if avg_prob < 30:
        recommendation = "✅ Session appears legitimate"
        rec_color = 'green'
    elif avg_prob < 60:
        recommendation = "⚠️ Some suspicious activity detected"
        rec_color = 'orange'
    else:
        recommendation = "❌ High probability of cheating"
        rec_color = 'red'
    
    stats_text += f"\nRecommendation:\n{recommendation}"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Session analysis saved as: {filename}")
    plt.show()

def draw_ui_overlay(frame, current_pos, smoothed_pos, is_warning, fps, cheat_prob=0, risk_factors={}):
    """Modern, clean UI overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Modern dark panel with rounded corners effect
    panel_height = 140
    panel_color = (20, 20, 20)  # Dark background
    
    # Main panel
    cv2.rectangle(overlay, (15, 15), (w-15, panel_height), panel_color, -1)
    cv2.rectangle(overlay, (15, 15), (w-15, panel_height), (100, 100, 100), 2)
    
    # Title with modern font
    cv2.putText(overlay, "EYE TRACKING ANTI-CHEAT", (30, 45), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # Status indicators (modern circular design)
    runtime = time.time() - start_time
    
    # Status row
    y_pos = 70
    cv2.putText(overlay, f"Runtime: {runtime:.1f}s", (30, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(overlay, f"FPS: {fps:.1f}", (200, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(overlay, f"Warnings: {warning_count}", (300, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Control Points status
    if control_validator is not None:
        control_stats = control_validator.get_accuracy_stats()
        control_color = (0, 255, 0) if control_stats['accuracy'] >= 0.7 else (0, 100, 255)
        cv2.putText(overlay, f"Control: {control_stats['accuracy']:.0%}", (450, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, control_color, 1)
    
    # Position display
    y_pos = 90
    if current_pos:
        cv2.putText(overlay, f"Gaze: ({current_pos[0]:.0f}, {current_pos[1]:.0f})", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
    
    if smoothed_pos:
        cv2.putText(overlay, f"Smooth: ({smoothed_pos[0]:.0f}, {smoothed_pos[1]:.0f})", (250, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
    
    # Cheat probability with modern progress bar
    y_pos = 110
    runtime = time.time() - cheat_detector.start_time if 'cheat_detector' in globals() else 0
    if runtime < 30:
        # Calibration progress
        progress = (runtime / 30) * 100
        cv2.putText(overlay, f"CALIBRATING: {progress:.0f}%", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        # Cheat probability with color coding
        prob_color = (0, 100, 255) if cheat_prob > 70 else (0, 255, 255) if cheat_prob > 40 else (0, 255, 0)
        cv2.putText(overlay, f"THREAT LEVEL: {cheat_prob:.0f}%", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, prob_color, 2)
        
        # Progress bar for threat level
        bar_width = 200
        bar_height = 8
        bar_x, bar_y = 250, y_pos - 8
        
        # Background bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Progress bar
        progress_width = int((cheat_prob / 100) * bar_width)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), prob_color, -1)
    
    # Draw gaze trail (simplified)
    if len(smoothed_positions) > 1:
        trail_positions = list(smoothed_positions)[-10:]  # Last 10 positions
        for i in range(1, len(trail_positions)):
            prev_pos = trail_positions[i-1]
            curr_pos = trail_positions[i]
            
            # Convert to frame coordinates
            prev_x = int((prev_pos[0] / size.width) * w) if size.width > 0 else w//2
            prev_y = int((prev_pos[1] / size.height) * h) if size.height > 0 else h//2
            curr_x = int((curr_pos[0] / size.width) * w) if size.width > 0 else w//2
            curr_y = int((curr_pos[1] / size.height) * h) if size.height > 0 else h//2
            
            # Clamp coordinates
            prev_x, prev_y = max(0, min(w-1, prev_x)), max(0, min(h-1, prev_y))
            curr_x, curr_y = max(0, min(w-1, curr_x)), max(0, min(h-1, curr_y))
            
            # Draw trail line with fade effect
            alpha = i / len(trail_positions)
            color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
            cv2.line(overlay, (prev_x, prev_y), (curr_x, curr_y), color, 2)
    
    # Current gaze position (enhanced)
    if smoothed_pos:
        sx, sy = smoothed_pos
        smooth_x = int((sx / size.width) * w) if size.width > 0 else w//2
        smooth_y = int((sy / size.height) * h) if size.height > 0 else h//2
        smooth_x, smooth_y = max(0, min(w-1, smooth_x)), max(0, min(h-1, smooth_y))
        
        # Modern gaze indicator
        cv2.circle(overlay, (smooth_x, smooth_y), 12, (0, 255, 0), -1)  # Green center
        cv2.circle(overlay, (smooth_x, smooth_y), 16, (255, 255, 255), 2)  # White ring
        cv2.circle(overlay, (smooth_x, smooth_y), 20, (100, 255, 100), 1)  # Outer glow
    
    # Control point display (enhanced)
    if control_validator is not None:
        should_display, control_pos = control_validator.should_display_control_point()
        if should_display:
            ctrl_x = int((control_pos[0] / size.width) * w) if size.width > 0 else w//2
            ctrl_y = int((control_pos[1] / size.height) * h) if size.height > 0 else h//2
            ctrl_x, ctrl_y = max(0, min(w-1, ctrl_x)), max(0, min(h-1, ctrl_y))
            
            # Pulsing control point
            pulse = abs(np.sin(time.time() * 8))
            radius = int(20 + 10 * pulse)
            cv2.circle(overlay, (ctrl_x, ctrl_y), radius, (0, 255, 255), 3)  # Cyan ring
            cv2.circle(overlay, (ctrl_x, ctrl_y), 8, (255, 255, 0), -1)  # Yellow center
            
            # Modern "FOCUS" text
            cv2.putText(overlay, "FOCUS HERE", (ctrl_x - 60, ctrl_y - 35), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
    
    # Controls help (modern style)
    help_y = h - 30
    cv2.putText(overlay, "Controls: [Q] Quit  [P] Plot Analysis  [+/-] Smoothing", 
                (20, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return overlay


def tracking_loop():
    global tracking_active, session_summary
    global smoothed_positions, positions, frame_count, warning_count, smoothing_window_size, start_time, control_validator, cheat_detector
    print("🚀 Starting real-time tracking (server mode)...")
    initialize_tracking()
    cv2.namedWindow('Eye Tracking Anti-Cheat', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Eye Tracking Anti-Cheat', 800, 600)
    while tracking_active:
        try:
            frame_count += 1
            while tracking_active:
                try:
                    frame_count += 1
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Failed to capture frame")
                        tracking_active = False
                        continue
                    fps = 0
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        fps = 30 / (current_time - start_time + 1e-6) if frame_count > 30 else 0
                    features, blink = estimator.extract_features(frame)
                    current_pos = None
                    is_warning = False
                    cheat_probability = 0.0
                    risk_factors = {}
                    smoothed_pos = None
                    if features is not None and not blink:
                        x, y = estimator.predict([features])[0]
                        current_pos = (x, y)
                        positions.append((x, y))
                        smoothed_positions.append((x, y))
                        if len(smoothed_positions) > smoothing_window_size:
                            new_positions = deque(list(smoothed_positions)[-smoothing_window_size:], maxlen=smoothing_window_size)
                            smoothed_positions = new_positions
                        smoothed_pos = calculate_moving_average(smoothed_positions)
                        is_warning, cheat_probability, risk_factors = check_sus()
                        if control_validator is not None:
                            if control_validator.should_inject_control_point():
                                control_pos = control_validator.start_control_test()
                                print(f"🎯 Control Point Test Started at ({control_pos[0]}, {control_pos[1]})")
                            if control_validator.check_control_point(current_pos):
                                if control_validator.is_session_compromised():
                                    print("🚨 SESSION COMPROMISED - Control point validation failed!")
                                    is_warning = True
                                    cheat_probability = max(cheat_probability, 85.0)
                    elif blink:
                        cv2.putText(frame, "BLINK DETECTED", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    display_frame = draw_ui_overlay(frame, current_pos, smoothed_pos, is_warning, fps, cheat_probability, risk_factors)
                    # Encode frame as JPEG and emit via WebSocket
                    _, buffer = cv2.imencode('.jpg', display_frame)
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    socketio.emit('frame', {'image': jpg_as_text})
                    cv2.imwrite(latest_frame_path, display_frame)
                    cv2.imshow('Eye Tracking Anti-Cheat', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        tracking_active = False
                    elif key == ord('p') or key == ord('P'):
                        print("📊 Generating session analysis...")
                        plot_session_analysis()
                    elif key == ord('+') or key == ord('='):
                        smoothing_window_size = min(smoothing_window_size + 1, 25)
                        print(f"Smoothing window increased to {smoothing_window_size}")
                    elif key == ord('-'):
                        smoothing_window_size = max(smoothing_window_size - 1, 3)
                        print(f"Smoothing window decreased to {smoothing_window_size}")
                except Exception as e:
                    print(f"Error in tracking loop: {e}")
                    tracking_active = False
            # Add control point stats if available
            if control_validator is not None and hasattr(control_validator, 'get_stats'):
                stats = control_validator.get_stats() if callable(getattr(control_validator, 'get_stats', None)) else None
                if stats:
                    session_summary['control_accuracy'] = stats.get('accuracy', None)
                    session_summary['control_tests'] = stats.get('total_tests', None)
            # Add recommendation based on avg_prob
            avg_prob = session_summary.get('average_threat', None)
            if avg_prob is not None:
                if avg_prob < 30:
                    session_summary['recommendation'] = "Session appears legitimate"
                elif avg_prob < 60:
                    session_summary['recommendation'] = "Some suspicious activity detected"
                else:
                    session_summary['recommendation'] = "High probability of cheating detected"
            else:
                session_summary['recommendation'] = "High probability of cheating detected"
        except Exception as e:
            print(f"Critical error in tracking loop: {e}")
            tracking_active = False
    print("👋 Eye tracking session ended")

# Flask endpoints
@app.route('/start', methods=['POST'])
def start_tracking():
    global tracking_thread, tracking_active, control_points_enabled
    if tracking_active:
        return jsonify({'status': 'already running'}), 400
    control_points_enabled = request.json.get('control_points_enabled', False)
    tracking_active = True
    tracking_thread = threading.Thread(target=tracking_loop)
    tracking_thread.start()
    return jsonify({'status': 'started'}), 200

@app.route('/stop', methods=['POST'])
def stop_tracking():
    global tracking_active
    tracking_active = False
    return jsonify({'status': 'stopping'}), 200

@app.route('/status', methods=['GET'])
def get_status():
    global tracking_active, session_summary
    return jsonify({'tracking_active': tracking_active, 'session_summary': session_summary}), 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)