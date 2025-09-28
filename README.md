# Eye Tracking Anti-Cheat System - Mathematical Documentation

## Getting Started

### Installation

1. **Backend (Python Server)**:
   ```bash
   cd EyeTracking
   pip install -r ../requirements.txt
   ```


### Running the System

1. **Start the Python Backend Server**:
   ```bash
   cd EyeTracking
   # Launch the server
   python main.py
   ```
   The server will run on `http://localhost:5000`

    Without the control point method :
   ```
   Invoke-RestMethod -Uri "http://localhost:5000/start" -Method POST -ContentType "application/json" -Body '{"control_points_enabled": false}'
   ```
    With the control point method : 
    ```
    Invoke-RestMethod -Uri "http://localhost:5000/start" -Method POST -ContentType "application/json" -Body '{"control_points_enabled": true}'
    ```
## Overview

This document provides a comprehensive mathematical analysis of the eye tracking anti-cheat system implemented in `ca_clean.py`. The system uses objective behavioral metrics, statistical analysis, and control point validation to detect cheating attempts in real-time.

## Table of Contents

1. [Core Mathematical Concepts](#core-mathematical-concepts)
2. [Behavioral Metrics](#behavioral-metrics)
3. [Statistical Analysis](#statistical-analysis)
4. [Control Point Validation](#control-point-validation)
5. [Smoothing Algorithms](#smoothing-algorithms)
6. [Risk Assessment](#risk-assessment)
7. [Implementation Details](#implementation-details)

---

## Core Mathematical Concepts

### Coordinate System

The system operates in a 2D screen coordinate system where:
- **Origin**: Top-left corner (0, 0)
- **X-axis**: Horizontal, increasing rightward
- **Y-axis**: Vertical, increasing downward
- **Bounds**: [0, screen_width] × [0, screen_height]

Each gaze position is represented as: **P(t) = (x(t), y(t))**

### Time Series Analysis

Gaze data forms a discrete time series:
```
G = {P(t₀), P(t₁), P(t₂), ..., P(tₙ)}
```

Where each position P(tᵢ) represents the gaze coordinates at time tᵢ.

---

## Behavioral Metrics

### 1. Spatial Dispersion

**Purpose**: Measures the spread of gaze positions to detect unnatural clustering or excessive wandering.

**Mathematical Definition**:
```
σ_spatial = √[(σₓ² + σᵧ²) / 2]
```

Where:
- `σₓ = std([x₁, x₂, ..., xₙ])` - Standard deviation of X coordinates
- `σᵧ = std([y₁, y₂, ..., yₙ])` - Standard deviation of Y coordinates

**Interpretation**:
- Low values (< 50px): Excessive focus, possible bot behavior
- High values (> 500px): Erratic movement, possible distraction
- Normal range: 100-300px for typical reading/viewing tasks

### 2. Velocity Variance

**Purpose**: Analyzes the consistency of eye movement velocities to detect artificial patterns.

**Step 1 - Calculate instantaneous velocities**:
```
v(tᵢ) = √[(x(tᵢ) - x(tᵢ₋₁))² + (y(tᵢ) - y(tᵢ₋₁))²] / Δt
```

**Step 2 - Compute velocity variance**:
```
σ²_velocity = (1/n) Σ[v(tᵢ) - v̄]²
```

Where `v̄` is the mean velocity.

**Biological Context**:
- Human saccades: 30-500°/second (highly variable)
- Smooth pursuit: 1-30°/second (consistent)
- Artificial movements often show abnormally low variance

### 3. Off-Screen Percentage

**Purpose**: Quantifies how much time is spent looking away from the screen.

**Mathematical Definition**:
```
P_offscreen = (N_offscreen / N_total) × 100%
```

Where:
- `N_offscreen = |{P(tᵢ) : x(tᵢ) ∉ [0, W] ∨ y(tᵢ) ∉ [0, H]}|`
- `N_total` = Total number of gaze samples
- `W, H` = Screen width and height

**Cheating Indicators**:
- > 30%: High likelihood of looking at external resources
- > 50%: Strong indicator of cheating behavior

### 4. Fixation Clustering

**Purpose**: Analyzes the spatial distribution of fixation points using pairwise distances.

**Mathematical Definition**:
```
d̄_fixation = (2/[n(n-1)]) ΣΣ √[(xᵢ - xⱼ)² + (yᵢ - yⱼ)²]
```

Where the double sum covers all unique pairs (i,j) with i < j.

**Interpretation**:
- Low clustering (high d̄): Normal scanning behavior
- High clustering (low d̄): Possible focus on specific areas or artificial patterns

### 5. Screen Coverage

**Purpose**: Measures the area of screen actively viewed.

**Mathematical Definition**:
```
Coverage = [(max(X) - min(X)) × (max(Y) - min(Y))] / (W × H) × 100%
```

**Edge Cases**:
- Very low coverage (< 10%): Tunnel vision or bot behavior
- Very high coverage (> 90%): Possible rapid scanning or artificial movement

### 6. Gaze Entropy

**Purpose**: Measures the unpredictability of gaze patterns using information theory concepts.

**Mathematical Definition**:
```
H_gaze = σ_spatial / max(1, v̄_velocity)
```

This simplified entropy measure relates spatial randomness to movement consistency.

**Interpretation**:
- Low entropy: Predictable, possibly artificial patterns
- High entropy: Natural human variability

---

## Statistical Analysis

### Baseline Calibration

During the first 30 seconds, the system establishes individual baselines for each metric.

**For each metric M**:
```
μₘ = (1/n) Σ Mᵢ                    (Sample mean)
σₘ = √[(1/(n-1)) Σ(Mᵢ - μₘ)²]     (Sample standard deviation)
```

**Baseline Structure**:
```python
baseline_metrics[M] = {
    'mean': μₘ,
    'std': σₘ, 
    'min': min(M₁, M₂, ..., Mₙ),
    'max': max(M₁, M₂, ..., Mₙ)
}
```

### Z-Score Analysis

**Purpose**: Standardizes current behavior against individual baseline.

**Mathematical Definition**:
```
Z_M = |M_current - μₘ| / σₘ
```

**Interpretation**:
- Z < 1: Within normal range (68% confidence interval)
- 1 ≤ Z < 2: Moderate deviation (95% confidence interval)
- 2 ≤ Z < 3: Significant deviation (99.7% confidence interval)  
- Z ≥ 3: Extreme deviation (> 99.7% confidence interval)

### Risk Score Calculation

**Step 1 - Convert Z-score to risk percentage**:
```
Risk_M = min(100, (Z_M / 3) × 100) × W_M
```

Where `W_M` is the weight assigned to metric M.

**Step 2 - Aggregate total risk**:
```
Risk_total = Σ Risk_M
```

---

## Control Point Validation

### Control Point Generation

**Random Uniform Distribution**:
```
x_control ~ U(0.1W, 0.9W)
y_control ~ U(0.1H, 0.9H)
```

This ensures control points stay within the central 80% of the screen, avoiding edge effects.

### Distance Error Calculation

**Euclidean Distance**:
```
d_error = √[(x_predicted - x_control)² + (y_predicted - y_control)²]
```

### Accuracy Assessment

**Validation Criteria**:
```
Valid = {
    True,  if d_error ≤ tolerance_pixels
    False, otherwise
}
```

**Session Accuracy**:
```
Accuracy = N_valid / N_total
```

**Compromise Detection**:
```
Compromised = {
    True,  if Accuracy < 0.7 AND N_total ≥ 3
    False, otherwise
}
```

---

## Smoothing Algorithms

### Simple Moving Average

**Mathematical Definition**:
```
P̄_SMA(t) = (1/n) Σ P(t-i+1)    for i = 1 to n
```

### Exponential Smoothing

**Recursive Formula**:
```
P̄_EXP(t) = α × P(t) + (1-α) × P̄_EXP(t-1)
```

Where `α ∈ [0,1]` is the smoothing parameter.

### Hybrid Smoothing (Used in System)

**Blended Approach**:
```
P̄_final(t) = 0.7 × P̄_SMA(t) + 0.3 × P̄_EXP(t)
```

This combination provides both stability (SMA) and responsiveness (EXP).

---

## Risk Assessment

### Weighted Risk Model

**Weight Assignment**:
```python
weights = {
    'off_screen_percentage': 0.40,  # Highest priority
    'fixation_clustering': 0.20,    # Important for focus analysis
    'spatial_dispersion': 0.15,     # Movement pattern analysis  
    'velocity_variance': 0.15,      # Consistency checking
    'screen_coverage': 0.00,        # Disabled (edge case metric)
    'gaze_entropy': 0.10           # Randomness analysis
}
```

**Final Risk Calculation**:
```
Risk_final = Σ (Weight_i × Risk_i)
```

### Threat Level Classification

```
Classification = {
    "Low Risk":      Risk_final < 30%
    "Medium Risk":   30% ≤ Risk_final < 60%  
    "High Risk":     60% ≤ Risk_final < 80%
    "Critical":      Risk_final ≥ 80%
}
```

---

## Implementation Details

### Data Structures

**Position Buffer**:
```python
positions = deque(maxlen=20)  # Last 20 raw positions
smoothed_positions = deque(maxlen=15)  # Last 15 smoothed positions
```

**Control Point Structure**:
```python
@dataclass
class ControlPoint:
    timestamp: float
    expected_position: Tuple[int, int]
    predicted_position: Tuple[int, int] 
    distance_error: float
    is_valid: bool
```

### Performance Optimizations

**Circular Buffers**: Use `collections.deque` with fixed `maxlen` for O(1) append operations.

**Incremental Statistics**: Calculate running statistics without storing all historical data.

**Lazy Evaluation**: Compute expensive metrics only when sufficient data is available.

### Numerical Stability

**Division by Zero Protection**:
```python
denominator = max(baseline['std'], 1e-6)  # Prevents division by zero
z_score = abs(current_val - baseline['mean']) / denominator
```

**Boundary Clamping**:
```python
risk_score = min(100, max(0, calculated_risk))  # Clamp to [0, 100]
```

---

## Mathematical Assumptions

### 1. Gaussian Distribution
The system assumes baseline metrics follow approximately normal distributions, justifying the use of Z-scores.

### 2. Independence  
Metrics are assumed to be conditionally independent given the user's behavior state.

### 3. Stationarity
User behavior patterns are assumed to be relatively stable during the calibration period.

### 4. Linear Risk Aggregation
Individual risk scores are combined linearly, which may not capture complex interaction effects.

---

## Validation and Calibration

### Statistical Validation

**Confidence Intervals**: The system uses 99.7% confidence intervals (3σ rule) for anomaly detection.

**Sample Size**: Minimum 5 data points required for meaningful statistical analysis.

**Temporal Window**: 30-second calibration period balances accuracy and responsiveness.

### Empirical Tuning

**Threshold Selection**: Risk thresholds (30%, 60%, 80%) derived from empirical testing.

**Weight Optimization**: Metric weights optimized based on false positive/negative rates.

**Control Point Tolerance**: 100-pixel tolerance accounts for natural eye-tracking inaccuracy.

---

## Limitations and Future Improvements

### Current Limitations

1. **Linear Risk Model**: May miss complex behavioral patterns
2. **Fixed Weights**: Not adaptive to individual differences
3. **Temporal Independence**: Doesn't model temporal correlations
4. **Limited Context**: No consideration of task-specific normal behavior

### Potential Improvements

1. **Machine Learning**: Replace linear model with trained classifiers
2. **Temporal Modeling**: Add time-series analysis for sequential patterns
3. **Adaptive Thresholds**: Personalize thresholds based on user history
4. **Multi-Modal**: Integrate additional biometric signals

---

## References and Further Reading

1. Duchowski, A. T. (2017). *Eye Tracking Methodology: Theory and Practice*
2. Holmqvist, K. et al. (2011). *Eye Tracking: A Comprehensive Guide to Methods and Measures*
3. Rayner, K. (2009). "Eye movements and attention in reading, scene perception, and visual search"
4. Young, L. R. & Sheena, D. (1975). "Survey of eye movement recording methods"

---

*This document provides the mathematical foundation for understanding and extending the eye tracking anti-cheat system. For implementation details, refer to the source code in `main.py`.*