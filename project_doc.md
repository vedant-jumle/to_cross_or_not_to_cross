# Counterfactual Explanations for Pedestrian Crossing Prediction
## Project Specification Document

---

## 1. PROJECT OVERVIEW

### 1.1 Problem Statement
Current pedestrian crossing prediction models in autonomous driving provide predictions (will cross / won't cross) but lack interpretable explanations about which factors drive those decisions. This creates a trust and safety validation gap for deployment in autonomous vehicles.

### 1.2 Proposed Solution
Develop a feature-based counterfactual explanation framework that:
1. Takes existing pedestrian crossing prediction models
2. Systematically perturbs input features (traffic signals, vehicle proximity, pedestrian gaze)
3. Generates minimal counterfactuals showing which factors causally influence predictions
4. Validates faithfulness (predictions flip as expected) and minimality (fewest changes)

### 1.3 Expected Outcome
A systematic analysis identifying which contextual factors (traffic signals, vehicle proximity, pedestrian attention) are most critical for crossing decisions, providing interpretable insights for autonomous vehicle safety validation.

---

## 2. DATASETS

### 2.1 JAAD (Joint Attention in Autonomous Driving)
**Download:** https://data.nvision2.eecs.yorku.ca/JAAD_dataset/
**GitHub:** https://github.com/ykotseruba/JAAD

**Specifications:**
- 346 video clips (5-15 seconds each)
- 2,793 annotated pedestrians (686 with full behavioral annotations)
- Locations: North America + Europe

**Key Annotations:**
- Bounding boxes per frame (x1, y1, x2, y2)
- Occlusion flags (0=none, 1=partial, 2=full)
- Behavioral labels per frame:
  - `crossing`: 1/0/-1 (crossing / not crossing / no intention)
  - `looking`: whether pedestrian is looking at traffic
  - `walking`, `standing`: motion state
- Attributes (per pedestrian):
  - Demographics: `age`, `gender`
  - `group_size`: number of people in group
  - `direction`: LAT (crossing), LONG (along road), n/a
  - `designated`: D (crosswalk), ND (not crosswalk)
  - `signalized`: S (traffic signal), NS (no signal)
- Scene attributes (per video):
  - `weather`: clear, rain, snow
  - `time_of_day`: day, night
- Traffic elements (per frame):
  - Traffic lights, stop signs, crosswalks

**Data Structure:**
```
JAAD_dataset/
├── annotations/
│   └── JAAD_DATA.pkl  # Generated via generate_database()
├── clips/
│   ├── video_0001.mp4
│   ├── video_0002.mp4
│   └── ...
└── splits/
    ├── default/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
```

### 2.2 PIE (Pedestrian Intention Estimation)
**Download:** https://data.nvision2.eecs.yorku.ca/PIE_dataset/
**GitHub:** https://github.com/aras62/PIE

**Specifications:**
- 6 hours of video footage (300K+ frames)
- 1,842 pedestrian samples
- Location: Toronto streets
- 30 FPS, HD resolution

**Key Annotations:**
- Bounding boxes + occlusion
- **Crossing intention probability** (from human experiment!)
- Actions: `walking`, `standing`, `looking`, `not_looking`, `crossing`, `not_crossing`
- **Vehicle data from OBD sensor:**
  - GPS coordinates
  - Vehicle speed (crucial feature!)
  - Heading direction
- Traffic infrastructure:
  - Traffic lights (state + position)
  - Signs
  - Zebra crossings

**Data Structure:**
```
PIE_dataset/
├── annotations/
│   └── PIE_DATA.pkl  # Generated via generate_database()
├── clips/
│   ├── set_01/
│   ├── set_02/
│   └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

**Data Split:**
- Train: 50%
- Test: 40%
- Validation: 10%

### 2.3 Recommended Choice
**Start with JAAD** because:
- Smaller, easier to work with
- Simpler feature set
- Behavioral annotations are sufficient
- PIE can be added later if time permits

---

## 3. BASELINE MODEL

### 3.1 Model Selection Strategy

**Option A: Use Existing Pretrained Model (RECOMMENDED)**
- **Repository:** https://github.com/ykotseruba/JAAD (has pretrained models)
- **Repository:** https://github.com/aras62/PIEPredict (PIE models)
- **Advantages:**
  - No training time needed
  - Proven to work
  - Can immediately start on counterfactual generation
- **Model Types Available:**
  - LSTM-based models
  - GNN-based models (encode pedestrian-vehicle relationships)
  - Intention estimation models

**Option B: Train Simple Baseline (If needed)**
- Use feature-based classifier (Random Forest, XGBoost, or simple MLP)
- Input features: See Section 3.2
- Target: Binary classification (will cross / won't cross)

### 3.2 Feature Space

**Required Features (Available in Both Datasets):**

**Pedestrian Features:**
1. `bbox_trajectory`: Last N frames of bounding box positions (x, y, w, h)
2. `looking`: Is pedestrian looking at traffic? (binary)
3. `walking`: Is pedestrian walking? (binary)
4. `standing`: Is pedestrian standing? (binary)
5. `distance_to_road`: Distance from pedestrian to road edge (pixels or meters)

**Vehicle Features:**
6. `vehicle_speed`: Current vehicle speed (m/s or km/h) [PIE has this directly, JAAD may need proxy]
7. `vehicle_distance`: Distance between vehicle and pedestrian (can be computed from bbox)
8. `time_to_collision`: Estimated TTC if pedestrian crosses

**Scene Context:**
9. `traffic_light`: State of traffic light (red / green / yellow / none)
10. `crosswalk`: Is pedestrian at designated crossing? (binary)
11. `signalized`: Is location signalized? (binary)

**Additional Features (Optional):**
12. `group_size`: Number of pedestrians in group
13. `age`: Estimated age category (child / young / adult / senior)
14. `weather`: Weather condition
15. `time_of_day`: Day or night

**Feature Extraction Pipeline:**
```python
# Pseudocode for feature extraction
def extract_features(pedestrian_id, video_id, frame_range):
    features = {
        'bbox_trajectory': get_bbox_history(ped_id, video_id, frames=16),  # Last 16 frames
        'looking': get_behavior(ped_id, 'looking', current_frame),
        'walking': get_behavior(ped_id, 'walking', current_frame),
        'standing': get_behavior(ped_id, 'standing', current_frame),
        'distance_to_road': compute_distance_to_road(bbox, frame),
        'vehicle_speed': get_vehicle_speed(video_id, current_frame),  # From OBD or proxy
        'vehicle_distance': compute_vehicle_distance(ped_bbox, vehicle_bbox),
        'traffic_light': get_traffic_light_state(video_id, current_frame),
        'crosswalk': get_attribute(ped_id, 'designated'),
        'signalized': get_attribute(ped_id, 'signalized'),
        'group_size': get_attribute(ped_id, 'group_size'),
    }
    return features
```

### 3.3 Model Architecture Options

**Simple Baseline (If training from scratch):**
```python
# MLP Classifier
Input: Feature vector (11-15 dims)
Hidden: [64, 32, 16]
Output: Sigmoid(will_cross probability)
Loss: Binary Cross-Entropy
Optimizer: Adam
```

**Temporal Baseline (If using trajectory):**
```python
# LSTM
Input: Sequence of features (T=16 frames, F=11-15 features)
LSTM: Hidden size 64, 2 layers
FC: [64, 32, 1]
Output: Sigmoid(will_cross probability)
```

**Pretrained Model (Recommended):**
- Load from JAAD/PIE repositories
- Use as black box
- Focus effort on counterfactual generation

---

## 4. COUNTERFACTUAL GENERATION FRAMEWORK

### 4.1 Counterfactual Definition

**Formal Definition:**
Given:
- Input features: **x** = [f₁, f₂, ..., fₙ]
- Model prediction: M(**x**) = y (e.g., y = 1 for "will cross")

Find counterfactual **x'** such that:
1. M(**x'**) ≠ y (prediction flips)
2. **Distance(x, x')** is minimal (fewest changes)
3. **x'** is realistic (valid feature values)

**Example:**
- Original: `traffic_light=green, vehicle_distance=30m, looking=True` → Prediction: **Will Cross**
- Counterfactual: `traffic_light=red, vehicle_distance=30m, looking=True` → Prediction: **Won't Cross**
- Explanation: "Pedestrian would NOT cross if traffic light were red"

### 4.2 Perturbation Strategy

**Feature Types & Perturbation Methods:**

**Binary Features** (looking, walking, crosswalk, signalized):
- Flip: 0 → 1 or 1 → 0

**Categorical Features** (traffic_light, age, weather):
- Enumerate all valid values
- Example: traffic_light ∈ {red, green, yellow, none}

**Continuous Features** (vehicle_distance, vehicle_speed, distance_to_road):
- Discretize into bins
- Example: vehicle_distance ∈ [0-10m, 10-20m, 20-30m, 30-50m, 50m+]
- Or: Perturb by ±X% (e.g., ±10%, ±20%, ±50%)

**Sequential Features** (bbox_trajectory):
- Keep fixed (don't perturb trajectory, too complex)
- Or: Interpolate/extrapolate trajectory

### 4.3 Counterfactual Search Algorithms

**Algorithm 1: Brute Force Enumeration (Baseline)**
```python
def brute_force_counterfactual(x, model, target_flip):
    """
    Try all single-feature perturbations.
    """
    counterfactuals = []
    original_pred = model.predict(x)
    
    for feature_idx in range(len(x)):
        for new_value in get_valid_values(feature_idx):
            x_cf = x.copy()
            x_cf[feature_idx] = new_value
            
            new_pred = model.predict(x_cf)
            if new_pred != original_pred:  # Prediction flipped
                counterfactuals.append({
                    'x_cf': x_cf,
                    'changed_features': [feature_idx],
                    'num_changes': 1
                })
    
    return counterfactuals
```

**Algorithm 2: Greedy Search (For Minimality)**
```python
def greedy_minimal_counterfactual(x, model):
    """
    Iteratively add feature changes until prediction flips.
    Prioritize features by importance.
    """
    original_pred = model.predict(x)
    x_cf = x.copy()
    changed_features = []
    
    # Get feature importance (from model or prior analysis)
    feature_order = get_feature_importance_order(model)
    
    for feature_idx in feature_order:
        for new_value in get_valid_values(feature_idx):
            x_cf[feature_idx] = new_value
            
            if model.predict(x_cf) != original_pred:
                changed_features.append(feature_idx)
                return x_cf, changed_features
            
            # Revert if didn't flip
            x_cf[feature_idx] = x[feature_idx]
    
    return None  # No counterfactual found
```

**Algorithm 3: DICE-style Optimization (Advanced, if time permits)**
```python
def dice_counterfactual(x, model, lambda_sparsity=0.1):
    """
    Optimize for counterfactual using gradient descent.
    Loss = prediction_loss + lambda * sparsity_loss
    """
    # This requires differentiable model and features
    # May not be applicable for discrete features
    pass  # Implement if time allows
```

### 4.4 Recommended Approach for 8 Weeks

**Phase 1: Single Feature Counterfactuals (Weeks 3-4)**
- Implement Algorithm 1 (Brute Force)
- For each sample, try changing each feature individually
- Identify which features cause prediction flip
- **Deliverable:** "Feature X caused flip in Y% of cases"

**Phase 2: Multi-Feature Counterfactuals (Weeks 5-6)**
- Implement Algorithm 2 (Greedy)
- Find minimal set of features to change
- **Deliverable:** "Average 1.5 features needed to flip prediction"

**Phase 3: Analysis & Validation (Week 7)**
- See Section 5

---

## 5. EVALUATION METRICS

### 5.1 Faithfulness Metrics

**Definition:** Do predictions actually flip when features are changed?

**Metric 1: Flip Rate**
```
Flip Rate = (# samples where prediction flipped) / (# total samples)
```

**Metric 2: Prediction Change Magnitude**
```
For regression output (probability):
Delta_pred = |M(x') - M(x)|

High delta = high faithfulness
```

**Metric 3: Consistency Check**
```
Flip feature back and forth:
x → x' → x
Should see: y → y' → y

Consistency = (# consistent flips) / (# total)
```

### 5.2 Minimality Metrics

**Definition:** Are we changing the fewest possible features?

**Metric 1: Sparsity**
```
Sparsity = # features changed / # total features

Lower = better (more minimal)
```

**Metric 2: Average Features Changed**
```
Mean # of features changed across all counterfactuals
```

**Metric 3: Comparison to Random Baseline**
```
Random baseline: Change features randomly until flip
Your method should require fewer changes than random
```

### 5.3 Realism Metrics

**Definition:** Are counterfactuals realistic?

**Metric 1: Feature Value Validity**
```
All feature values must be in valid range
Example: vehicle_distance ≥ 0
```

**Metric 2: Feature Correlation Check**
```
Some feature combinations are unrealistic:
- vehicle_speed=0 but vehicle_distance=decreasing (impossible)
- traffic_light=red but crosswalk=none (unlikely)

Check if counterfactual violates common sense
```

**Metric 3: Human Evaluation (Optional)**
- Show counterfactuals to humans
- Ask: "Is this scenario realistic?"
- Compute % of realistic counterfactuals

### 5.4 Feature Importance Ranking

**Goal:** Which features matter most?

**Metric 1: Flip Frequency per Feature**
```
For each feature f:
    Flip_freq(f) = # times changing f caused flip / # total samples

Rank features by flip frequency
```

**Metric 2: Feature Necessity**
```
For multi-feature counterfactuals:
    Necessity(f) = # times f was necessary (couldn't flip without it) / # times f was changed

High necessity = feature is critical
```

**Metric 3: Shapley Values (Advanced)**
- Compute Shapley values for feature importance
- Requires multiple model evaluations
- More theoretically grounded

---

## 6. EXPERIMENTAL DESIGN

### 6.1 Timeline (8 Weeks)

**Week 1-2: Setup & Data Preparation**
- Download JAAD dataset
- Install data interface from GitHub
- Extract features from annotations
- Load pretrained model OR train baseline model
- Create train/val/test splits
- **Milestone:** Feature extraction pipeline working

**Week 3-4: Counterfactual Generation - Single Feature**
- Implement brute force search (Algorithm 1)
- Generate single-feature counterfactuals for test set
- Compute flip rates per feature
- Visualize: Which features cause most flips?
- **Milestone:** Single-feature analysis complete

**Week 5-6: Counterfactual Generation - Multi Feature**
- Implement greedy search (Algorithm 2)
- Generate minimal multi-feature counterfactuals
- Compute minimality metrics (average # features changed)
- Compare to random baseline
- **Milestone:** Multi-feature analysis complete

**Week 7: Evaluation & Analysis**
- Compute all metrics (faithfulness, minimality, realism)
- Feature importance ranking
- Case studies: Interesting examples
- Failure analysis: When do counterfactuals fail?
- **Milestone:** Complete results & analysis

**Week 8: Blog Post Writing**
- Write blog post (max 3000 words)
- Create visualizations
- Prepare presentation
- **Milestone:** Blog post ready for submission

### 6.2 Experiments to Run

**Experiment 1: Feature-Level Analysis**
- **Goal:** Which single features are most important?
- **Method:** For each feature, generate counterfactuals by changing only that feature
- **Metrics:** Flip rate per feature
- **Expected Result:** traffic_light and vehicle_distance likely most important

**Experiment 2: Minimal Counterfactuals**
- **Goal:** What's the minimum # of features to change?
- **Method:** Greedy search for minimal counterfactuals
- **Metrics:** Average # features changed, distribution
- **Expected Result:** 1-2 features sufficient for most cases

**Experiment 3: Baseline Comparison**
- **Goal:** Is our method better than random?
- **Method:** Compare against random feature perturbation
- **Metrics:** # features changed (ours vs. random)
- **Expected Result:** Our method requires fewer changes

**Experiment 4: Prediction Confidence Analysis**
- **Goal:** Are confident predictions harder to flip?
- **Method:** Stratify by prediction confidence, compute flip rate
- **Metrics:** Flip rate vs. confidence
- **Expected Result:** High confidence predictions harder to flip

**Experiment 5: Context Dependency**
- **Goal:** Do important features vary by context?
- **Method:** Stratify by scene context (crosswalk vs. no crosswalk, day vs. night)
- **Metrics:** Feature importance ranking per context
- **Expected Result:** Different features matter in different contexts

**Experiment 6: Case Studies**
- **Goal:** Qualitative understanding
- **Method:** Select 5-10 interesting cases, show counterfactuals
- **Metrics:** None (qualitative)
- **Expected Result:** Interpretable explanations

### 6.3 Evaluation Protocol

**Test Set:**
- Use official JAAD test split
- OR: If using pretrained model, use same test set as original paper
- Size: Aim for 200-500 pedestrian samples

**Metrics to Report:**
1. **Faithfulness:**
   - Flip rate: X%
   - Average prediction change: Δ = Y
2. **Minimality:**
   - Average # features changed: N
   - Sparsity: S
   - Comparison to random baseline: Ours (N₁) vs. Random (N₂)
3. **Feature Importance:**
   - Top 5 features ranked by flip frequency
   - Necessity scores for top features
4. **Qualitative:**
   - 5-10 case study examples
   - Failure cases (when counterfactual not found)

---

## 7. IMPLEMENTATION GUIDE

### 7.1 Code Structure

```
pedestrian_counterfactuals/
├── data/
│   ├── download_jaad.py          # Download JAAD dataset
│   ├── feature_extractor.py      # Extract features from annotations
│   └── dataloader.py              # PyTorch/TF dataloader
├── models/
│   ├── baseline_model.py         # Simple baseline if training
│   ├── pretrained_loader.py      # Load pretrained JAAD/PIE model
│   └── model_wrapper.py          # Unified interface for predictions
├── counterfactuals/
│   ├── perturbations.py          # Feature perturbation logic
│   ├── search_algorithms.py     # Brute force, greedy, etc.
│   └── generator.py              # Main counterfactual generator
├── evaluation/
│   ├── metrics.py                # Faithfulness, minimality metrics
│   ├── feature_importance.py    # Rank features by importance
│   └── visualizer.py             # Plot results
├── experiments/
│   ├── exp1_single_feature.py   # Experiment 1
│   ├── exp2_minimal_cf.py       # Experiment 2
│   └── ...
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_baseline.ipynb
│   └── 03_counterfactual_analysis.ipynb
├── results/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── README.md
```

### 7.2 Key Dependencies

```txt
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Deep learning (choose one)
torch>=1.9.0  # PyTorch
# OR
tensorflow>=2.6.0  # TensorFlow

# JAAD/PIE interface
opencv-python>=4.5.0
tqdm>=4.62.0

# Optional
jupyter>=1.0.0
plotly>=5.0.0  # Interactive plots
```

### 7.3 Data Loading Example

```python
# data/feature_extractor.py
import pickle
import numpy as np
from jaad_data import JAAD  # From JAAD GitHub repo

class JAAdFeatureExtractor:
    def __init__(self, data_path):
        self.jaad = JAAD(data_path=data_path)
        
    def extract_features(self, ped_id, video_id, frame_range):
        """
        Extract features for a single pedestrian sample.
        
        Returns:
            features (dict): {
                'bbox_trajectory': np.array (T, 4),
                'looking': bool,
                'walking': bool,
                'vehicle_distance': float,
                'traffic_light': str,
                'crosswalk': bool,
                ...
            }
            label (int): 1 if crossing, 0 otherwise
        """
        # Get pedestrian data
        ped_data = self.jaad.get_ped_data(ped_id, video_id)
        
        # Extract bbox trajectory (last 16 frames)
        bboxes = ped_data['bbox'][frame_range]
        
        # Extract behavioral annotations
        looking = ped_data['behavior']['looking'][frame_range[-1]]
        walking = ped_data['behavior']['walking'][frame_range[-1]]
        
        # Extract attributes
        crosswalk = ped_data['attributes']['designated'] == 'D'
        signalized = ped_data['attributes']['signalized'] == 'S'
        
        # Get traffic light state (from scene annotations)
        traffic_light = self._get_traffic_light_state(video_id, frame_range[-1])
        
        # Compute vehicle distance (requires vehicle bbox)
        vehicle_distance = self._compute_vehicle_distance(ped_data, frame_range[-1])
        
        features = {
            'bbox_trajectory': bboxes,
            'looking': int(looking),
            'walking': int(walking),
            'crosswalk': int(crosswalk),
            'signalized': int(signalized),
            'traffic_light': traffic_light,
            'vehicle_distance': vehicle_distance,
            # Add more features...
        }
        
        # Label: 1 if pedestrian crosses, 0 otherwise
        label = int(ped_data['behavior']['crossing'][frame_range[-1]] == 1)
        
        return features, label
    
    def extract_dataset(self, split='train'):
        """
        Extract features for entire dataset split.
        """
        samples = []
        labels = []
        
        for ped_id in self.jaad.get_pedestrian_ids(split):
            features, label = self.extract_features(ped_id, ...)
            samples.append(features)
            labels.append(label)
        
        return samples, labels
```

### 7.4 Counterfactual Generator Example

```python
# counterfactuals/generator.py
import numpy as np
from typing import Dict, List, Tuple

class CounterfactualGenerator:
    def __init__(self, model, feature_spec):
        """
        Args:
            model: Trained prediction model
            feature_spec (dict): Specification of features and their valid values
                Example: {
                    'traffic_light': ['red', 'green', 'yellow', 'none'],
                    'looking': [0, 1],
                    'vehicle_distance': [0, 10, 20, 30, 50, 100],  # Discretized
                    ...
                }
        """
        self.model = model
        self.feature_spec = feature_spec
        
    def generate_single_feature_counterfactuals(self, x: Dict) -> List[Dict]:
        """
        Generate counterfactuals by changing one feature at a time.
        
        Args:
            x (dict): Original feature dict
            
        Returns:
            List of dicts, each containing:
                - 'x_cf': Counterfactual features
                - 'changed_feature': Name of changed feature
                - 'original_value': Original value
                - 'new_value': New value
                - 'original_pred': Original prediction
                - 'new_pred': Counterfactual prediction
                - 'flipped': Whether prediction flipped
        """
        original_pred = self.model.predict(x)
        counterfactuals = []
        
        for feature_name in x.keys():
            if feature_name not in self.feature_spec:
                continue  # Skip features we don't want to perturb
                
            valid_values = self.feature_spec[feature_name]
            original_value = x[feature_name]
            
            for new_value in valid_values:
                if new_value == original_value:
                    continue  # Skip original value
                
                # Create counterfactual
                x_cf = x.copy()
                x_cf[feature_name] = new_value
                
                # Get prediction
                new_pred = self.model.predict(x_cf)
                flipped = (new_pred != original_pred)
                
                counterfactuals.append({
                    'x_cf': x_cf,
                    'changed_feature': feature_name,
                    'original_value': original_value,
                    'new_value': new_value,
                    'original_pred': original_pred,
                    'new_pred': new_pred,
                    'flipped': flipped,
                    'num_changes': 1
                })
        
        return counterfactuals
    
    def generate_minimal_counterfactual(self, x: Dict, 
                                       max_features: int = 3) -> Dict:
        """
        Generate minimal counterfactual using greedy search.
        """
        original_pred = self.model.predict(x)
        x_cf = x.copy()
        changed_features = []
        
        # Greedy search: Try features in order of importance
        feature_order = self._get_feature_importance_order()
        
        for feature_name in feature_order:
            if len(changed_features) >= max_features:
                break
                
            valid_values = self.feature_spec[feature_name]
            
            for new_value in valid_values:
                if new_value == x[feature_name]:
                    continue
                    
                x_cf[feature_name] = new_value
                new_pred = self.model.predict(x_cf)
                
                if new_pred != original_pred:
                    # Success! Prediction flipped
                    changed_features.append({
                        'feature': feature_name,
                        'original': x[feature_name],
                        'new': new_value
                    })
                    return {
                        'x_cf': x_cf,
                        'changed_features': changed_features,
                        'num_changes': len(changed_features),
                        'original_pred': original_pred,
                        'new_pred': new_pred
                    }
                else:
                    # Revert if didn't flip
                    x_cf[feature_name] = x[feature_name]
        
        # No counterfactual found
        return None
    
    def _get_feature_importance_order(self) -> List[str]:
        """
        Return features ordered by importance.
        Could be based on prior analysis or model feature importance.
        """
        # Placeholder: Return default order
        # In practice, this would be learned from data
        return ['traffic_light', 'vehicle_distance', 'looking', 
                'walking', 'crosswalk', 'signalized']
```

### 7.5 Evaluation Example

```python
# evaluation/metrics.py
import numpy as np
from typing import List, Dict

class CounterfactualEvaluator:
    def compute_flip_rate(self, counterfactuals: List[Dict]) -> float:
        """
        Compute percentage of counterfactuals that caused prediction flip.
        """
        flipped = sum(1 for cf in counterfactuals if cf['flipped'])
        return flipped / len(counterfactuals) if counterfactuals else 0.0
    
    def compute_sparsity(self, counterfactuals: List[Dict]) -> float:
        """
        Compute average sparsity (% of features changed).
        """
        sparsities = [cf['num_changes'] / total_features 
                     for cf in counterfactuals]
        return np.mean(sparsities)
    
    def compute_feature_importance(self, counterfactuals: List[Dict]) -> Dict:
        """
        Compute flip frequency for each feature.
        """
        feature_counts = {}
        feature_flips = {}
        
        for cf in counterfactuals:
            feature = cf['changed_feature']
            flipped = cf['flipped']
            
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
            if flipped:
                feature_flips[feature] = feature_flips.get(feature, 0) + 1
        
        # Compute flip rate per feature
        importance = {
            feature: feature_flips.get(feature, 0) / count
            for feature, count in feature_counts.items()
        }
        
        # Sort by importance
        return dict(sorted(importance.items(), 
                          key=lambda x: x[1], 
                          reverse=True))
    
    def evaluate_all(self, counterfactuals: List[Dict]) -> Dict:
        """
        Compute all metrics.
        """
        return {
            'flip_rate': self.compute_flip_rate(counterfactuals),
            'average_changes': np.mean([cf['num_changes'] 
                                       for cf in counterfactuals]),
            'sparsity': self.compute_sparsity(counterfactuals),
            'feature_importance': self.compute_feature_importance(counterfactuals)
        }
```

---

## 8. EXPECTED RESULTS & DELIVERABLES

### 8.1 Quantitative Results

**Table 1: Single-Feature Counterfactual Analysis**
| Feature | Flip Rate | Avg. Prediction Change |
|---------|-----------|----------------------|
| traffic_light | 75% | 0.62 |
| vehicle_distance | 58% | 0.48 |
| looking | 42% | 0.31 |
| walking | 35% | 0.25 |
| crosswalk | 28% | 0.19 |

**Table 2: Minimal Counterfactual Statistics**
| Metric | Value |
|--------|-------|
| Average # features changed | 1.8 |
| Sparsity | 0.16 (16% of features) |
| % requiring 1 feature | 62% |
| % requiring 2 features | 28% |
| % requiring 3+ features | 10% |

**Table 3: Comparison to Random Baseline**
| Method | Avg. # Changes | Flip Rate |
|--------|----------------|-----------|
| Our Method | 1.8 | 87% |
| Random Baseline | 3.4 | 85% |

### 8.2 Qualitative Results

**Case Study Examples:**

**Example 1: Traffic Light Dominates**
- Original: `traffic_light=green, vehicle_distance=25m, looking=True` → **Will Cross (95%)**
- CF1: `traffic_light=red, vehicle_distance=25m, looking=True` → **Won't Cross (12%)**
- Explanation: "Traffic light is the dominant factor; even with favorable conditions, red light prevents crossing"

**Example 2: Vehicle Proximity Matters When Light is Ambiguous**
- Original: `traffic_light=none, vehicle_distance=50m, looking=True` → **Will Cross (78%)**
- CF1: `traffic_light=none, vehicle_distance=10m, looking=True` → **Won't Cross (22%)**
- Explanation: "In absence of traffic signal, vehicle proximity becomes critical"

**Example 3: Multiple Factors Required**
- Original: `traffic_light=yellow, vehicle_distance=30m, looking=False` → **Won't Cross (65%)**
- CF1: Need to change both: `traffic_light=green` AND `vehicle_distance=50m` → **Will Cross (85%)**
- Explanation: "Requires both favorable signal AND safe distance"

### 8.3 Final Deliverables

1. **Code Repository:**
   - Clean, documented code
   - README with setup instructions
   - Jupyter notebooks for reproducibility

2. **Blog Post (3000 words max):**
   - Introduction & Motivation
   - Related Work (brief)
   - Method Description
   - Experimental Setup
   - Results & Analysis
   - Discussion & Future Work
   - Conclusion

3. **Visualizations:**
   - Feature importance bar chart
   - Flip rate heatmap
   - Case study examples with before/after
   - Distribution of # features changed

4. **Presentation Slides:**
   - 10-15 slides for poster session
   - Key results highlighted

---

## 9. POTENTIAL CHALLENGES & MITIGATION

### 9.1 Challenge: Model Access/Training

**Problem:** Pretrained models may not be available or compatible.

**Mitigation:**
- Primary: Use JAAD GitHub pretrained models
- Backup: Train simple baseline (MLP or LSTM)
- Budget 1-2 weeks for model issues

### 9.2 Challenge: Feature Engineering

**Problem:** Some features may be missing or hard to extract.

**Mitigation:**
- Focus on features that are directly available in annotations
- If vehicle_speed missing in JAAD, use proxy (e.g., frame-to-frame bbox change)
- Start with minimal feature set, expand if time allows

### 9.3 Challenge: Counterfactual Realism

**Problem:** Generated counterfactuals may be unrealistic.

**Mitigation:**
- Add feature validity checks
- Document limitations in blog post
- Acknowledge this as future work

### 9.4 Challenge: Evaluation Without Ground Truth

**Problem:** No "true" counterfactuals to compare against.

**Mitigation:**
- Use proxy metrics (faithfulness, minimality, consistency)
- Human evaluation on small sample (optional)
- Compare to random baseline
- Qualitative case studies

### 9.5 Challenge: Time Constraints

**Problem:** 8 weeks is tight.

**Mitigation:**
- Stick to feature-based approach (no image editing)
- Use pretrained model (no training)
- Focus on single-feature analysis first (guaranteed results)
- Multi-feature is "bonus" if time allows
- Prepare backup plan: Just single-feature analysis is publishable

---

## 10. RESOURCES

### 10.1 Papers to Read

**Core Papers:**
1. JAAD Dataset: Rasouli et al., "Are they going to cross?", ICCVW 2017
2. PIE Dataset: Rasouli et al., "PIE: A Large-Scale Dataset", ICCV 2019
3. Counterfactual Explanations: Wachter et al., "Counterfactual Explanations Without Opening the Black Box", 2017
4. DICE: Mothilal et al., "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations", FAT* 2020

**Recent Explainability Work:**
5. PedVLM: "Pedestrian Vision Language Model", IEEE 2025
6. ExPedCross: "Explainable Pedestrian Crossing Prediction", ArXiv 2023

### 10.2 Repositories

- JAAD: https://github.com/ykotseruba/JAAD
- PIE: https://github.com/aras62/PIE
- PIE Pretrained Models: https://github.com/aras62/PIEPredict
- DICE (for reference): https://github.com/interpretml/DiCE

### 10.3 Tutorials

- JAAD Interface Tutorial: See JAAD repo README
- PyTorch Basics: https://pytorch.org/tutorials/
- Counterfactual Explanations Overview: Molnar, "Interpretable Machine Learning" (free online book)

---

## 11. BACKUP PLAN (If Things Go Wrong)

### Scenario A: Pretrained Model Doesn't Work
**Fallback:** Train simple MLP baseline on extracted features
**Time Cost:** +1 week
**Impact:** Minimal (analysis still valid)

### Scenario B: Multi-Feature Search Too Complex
**Fallback:** Stick to single-feature analysis only
**Time Saved:** +2 weeks
**Impact:** Still publishable, just narrower scope

### Scenario C: Dataset Issues
**Fallback:** Use only JAAD (skip PIE)
**Time Saved:** +1 week
**Impact:** Smaller dataset but still sufficient

### Scenario D: Running Out of Time
**Minimum Viable Project:**
1. Single-feature counterfactual analysis ✅
2. Feature importance ranking ✅
3. 3-5 case studies ✅
4. Blog post ✅

This is still a complete project even without multi-feature analysis.

---

## 12. SUCCESS CRITERIA

### Must Have (Required for Passing):
✅ Feature extraction pipeline working
✅ Counterfactual generation implemented (at least single-feature)
✅ Evaluation metrics computed
✅ Blog post written (3000 words, follows rubric)
✅ Presentation ready

### Should Have (For Good Grade):
✅ Multi-feature counterfactuals
✅ Comparison to baseline
✅ Feature importance analysis
✅ Multiple visualizations
✅ Case studies with insights

### Nice to Have (For Excellent Grade):
✅ Both JAAD and PIE datasets
✅ Multiple search algorithms compared
✅ Human evaluation study
✅ Code publicly released
✅ Novel insights about pedestrian behavior

---

## 13. CONTACT & SUPPORT

### For Dataset Issues:
- JAAD: aras@eecs.yorku.ca, yulia_k@eecs.yorku.ca
- PIE: arasouli.ai@gmail.com

### For Technical Help:
- TA: (Your assigned TA)
- Course Forum: (If available)

### For Claude Code:
- Use this document as reference
- Break down tasks into smaller chunks
- Test incrementally

---

## FINAL CHECKLIST BEFORE STARTING

Before you begin implementation, confirm:

- [ ] JAAD dataset downloaded (~20-30 GB)
- [ ] JAAD GitHub repo cloned
- [ ] Python environment set up (Python 3.7+)
- [ ] Dependencies installed (see requirements.txt)
- [ ] Data interface working (can load annotations)
- [ ] Pretrained model accessible OR plan to train baseline
- [ ] Understanding of feature space (which features to use)
- [ ] Timeline planned (which weeks for which tasks)
- [ ] Backup plan clear (what to drop if running out of time)

---

**Good luck! This is a solid, achievable project. Focus on getting single-feature analysis working first—everything else is bonus.**