# Controller Hyperparameters Configuration Guide

## Overview
All tunable parameters are now centralized at the top of the `Controller` class in `tools/controller.py`. This makes it easy to adjust the car's behavior without searching through the code.

## Location
```python
class Controller():
    # ==================== HYPERPARAMETERS - TUNE THESE ====================
    # ... all parameters here ...
    # ==================================================================
```

## Parameter Categories

### 1. Signboard Collection Parameters

#### `SAMPLES_REQUIRED = 30`
**What it does**: Number of signboard detections to collect before confirming the majority class.
- **Increase (35-40)**: More stable decision, but slower response
- **Decrease (20-25)**: Faster decision, but more prone to errors
- **Recommended range**: 25-35

#### `EARLY_TRIGGER_SAMPLES = 10`
**What it does**: Minimum samples needed to trigger early turn if intersection is already visible.
- **Increase (15-20)**: More confident early triggers
- **Decrease (5-8)**: Faster reaction to sudden intersections
- **Recommended range**: 8-15

#### `LEFT_SIGN_BOOST = 3`
**What it does**: Multiplier for 'left' sign detection (each 'left' counts as 3 samples).
- **Increase (4-5)**: Prioritize left turns more
- **Decrease (2)**: Treat left equally with other signs
- **Recommended range**: 2-4
- **Note**: Left signs are boosted because they're critical and sometimes harder to detect

### 2. Area Confidence Weighting

#### `AREA_WEIGHT_DIVISOR = 100.0`
**What it does**: Divisor in confidence formula: `(area / divisor) ^ exponent`
- **Increase (150-200)**: Reduce influence of large areas
- **Decrease (50-80)**: Increase influence of area size
- **Recommended range**: 80-120

#### `AREA_WEIGHT_EXPONENT = 1.5`
**What it does**: Exponent in confidence formula - controls how much larger areas matter.
- **Increase (2.0-2.5)**: Much higher weight for larger detections
- **Decrease (1.0-1.2)**: More linear relationship
- **Recommended range**: 1.2-2.0
- **Note**: 1.5 means larger signs have disproportionately more confidence

### 3. Turning Sequence Timing

#### `TURNING_MAX_FRAMES = 25`
**What it does**: Maximum frames to execute a turn before resetting.
- **Increase (30-35)**: Longer turns (for sharper corners)
- **Decrease (20-22)**: Shorter turns (for gentle curves)
- **Recommended range**: 20-30
- **Warning**: Too low may cause incomplete turns; too high may overshoot

### 4. Turn Angles (degrees)

#### `ANGLE_RIGHT = -25`
**What it does**: Steering angle for right turns (negative = right).
- **More negative (-30 to -35)**: Sharper right turns
- **Less negative (-20 to -22)**: Gentler right turns
- **Recommended range**: -30 to -20

#### `ANGLE_LEFT_CASE1 = 32`
**What it does**: Left turn angle when `top_corner > TOP_CORNER_HIGH_THRESH`.
- **Increase (35-40)**: Sharper left turns (high traffic case)
- **Decrease (28-30)**: Gentler left turns
- **Recommended range**: 28-35
- **Note**: Case 1 is for intersections with high top corner sum (more open road ahead)

#### `ANGLE_LEFT_CASE2 = 28`
**What it does**: Default left turn angle.
- **Increase (30-32)**: Sharper default left turns
- **Decrease (25-27)**: Gentler default left turns
- **Recommended range**: 25-32

#### `ANGLE_NO_LEFT = -25`
**What it does**: Turn right when 'no left' sign detected.
- **Same tuning as ANGLE_RIGHT**

#### `ANGLE_NO_RIGHT = 30`
**What it does**: Turn left when 'no right' sign detected.
- **Increase (32-35)**: Sharper left alternative
- **Decrease (25-28)**: Gentler left alternative
- **Recommended range**: 25-35

#### `ANGLE_NO_STRAIGHT_LEFT = 26`
**What it does**: Prefer left when 'no straight' sign detected and left corner favors it.
- **Increase (28-32)**: Sharper left preference
- **Decrease (22-25)**: Gentler left preference
- **Recommended range**: 22-30

#### `ANGLE_NO_STRAIGHT_RIGHT = -24`
**What it does**: Prefer right when 'no straight' sign detected and right corner favors it.
- **More negative (-28 to -30)**: Sharper right preference
- **Less negative (-20 to -22)**: Gentler right preference
- **Recommended range**: -28 to -20

### 5. Corner Sum Thresholds

These control turn decisions based on road geometry detected in corners of the image.

#### `TOP_CORNER_HIGH_THRESH = 17_000`
**What it does**: High threshold for top corner pixel sum (determines left turn case).
- **Increase (18_000-20_000)**: Require more open road for Case 1
- **Decrease (15_000-16_000)**: Use Case 1 more often
- **Recommended range**: 15_000 - 20_000

#### `TOP_CORNER_MID_THRESH = 17_500`
**What it does**: Mid threshold for 'no right' decision logic.
- **Increase (18_000-19_000)**: Stricter conditions for 'no right' turn
- **Decrease (16_000-17_000)**: Easier 'no right' turn trigger
- **Recommended range**: 16_000 - 19_000

#### `LEFT_CORNER_THRESH = 2_000`
**What it does**: Threshold for left corner sum (used in 'no right' logic).
- **Increase (2_500-3_000)**: Require more left visibility
- **Decrease (1_500-1_800)**: Easier left visibility trigger
- **Recommended range**: 1_500 - 3_000

### 6. Intersection Detection Parameters

#### `INTERSECTION_MIN_WIDTH = 140`
**What it does**: Minimum lane width (pixels) to consider it an intersection.
- **Increase (150-170)**: Only detect wider intersections (fewer false positives)
- **Decrease (120-135)**: Detect narrower intersections (more sensitive)
- **Recommended range**: 120-160
- **Warning**: Too low causes premature turns; too high misses intersections

#### `INTERSECTION_LOW_HEIGHT = 48`
**What it does**: Starting height (row) to check for lane width.
- **Increase (50-55)**: Check closer to car
- **Decrease (45-48)**: Check further ahead
- **Recommended range**: 45-55

#### `INTERSECTION_CHECK_WINDOW = 5`
**What it does**: Number of consecutive heights to check.
- **Increase (6-8)**: More thorough checking
- **Decrease (3-4)**: Faster checking
- **Recommended range**: 3-7

#### `INTERSECTION_CHECK_HEIGHT = 62`
**What it does**: Upper band height for intersection verification.
- **Increase (65-70)**: Check further ahead for confirmation
- **Decrease (58-60)**: Check closer for confirmation
- **Recommended range**: 58-70

#### `INTERSECTION_TOP_THRESH = 15_000`
**What it does**: Maximum top region sum to confirm intersection (lower = more road ahead).
- **Increase (16_000-18_000)**: Allow more road pixels for intersection
- **Decrease (12_000-14_000)**: Stricter intersection confirmation
- **Recommended range**: 12_000 - 18_000

### 7. System Parameters

#### `RESET_COUNTER_LIMIT = 200`
**What it does**: Frames before automatic system reset (safety feature).
- **Increase (250-300)**: Longer before reset
- **Decrease (150-180)**: More frequent resets
- **Recommended range**: 150-250
- **Note**: Prevents stuck states if something goes wrong

#### `VERBOSE_LOGGING = True`
**What it does**: Enable/disable detailed console logging.
- **`True`**: Full logs with emojis (debugging/tuning)
- **`False`**: Minimal logs (production)
- **Recommended**: `True` during tuning, `False` for competition

## Quick Tuning Guide

### Problem: Car turns too early
**Solutions:**
1. Increase `INTERSECTION_MIN_WIDTH` to 150-160
2. Increase `SAMPLES_REQUIRED` to 35-40
3. Decrease `EARLY_TRIGGER_SAMPLES` to 5-8 (or disable early trigger)

### Problem: Car misses intersections
**Solutions:**
1. Decrease `INTERSECTION_MIN_WIDTH` to 120-130
2. Enable early trigger: set `EARLY_TRIGGER_SAMPLES` to 10-15
3. Increase `INTERSECTION_TOP_THRESH` to 16_000-18_000

### Problem: Turns are too sharp
**Solutions:**
1. Reduce turn angles by 3-5 degrees
2. Example: `ANGLE_RIGHT = -22` instead of -25

### Problem: Turns are too gentle (doesn't make it around corner)
**Solutions:**
1. Increase turn angles by 3-5 degrees
2. Increase `TURNING_MAX_FRAMES` to 28-30

### Problem: Wrong signboard detected
**Solutions:**
1. Increase `SAMPLES_REQUIRED` to 35-40 (more samples = more stable)
2. Adjust `AREA_WEIGHT_EXPONENT` to 2.0 (prioritize closer/larger signs)
3. Increase `LEFT_SIGN_BOOST` if left signs are being missed

### Problem: Car hesitates at intersection
**Solutions:**
1. Enable/tune early trigger
2. Decrease `SAMPLES_REQUIRED` to 25
3. Increase `INTERSECTION_CHECK_WINDOW` to 6-7

## Tuning Workflow

1. **Start with defaults** - Test one lap
2. **Identify the issue** - Which intersection/turn is problematic?
3. **Tune one parameter at a time** - Don't change multiple things
4. **Test the change** - Run at least 2-3 laps
5. **Document your changes** - Keep notes on what works
6. **Revert if worse** - Go back to previous values if performance decreases

## Testing Commands

```bash
# Full run with verbose logging (default)
python main.py

# To disable verbose logging, edit controller.py:
# VERBOSE_LOGGING = False
```

## Advanced: Formula Explanations

### Area Confidence Formula
```python
area_weight = (area / AREA_WEIGHT_DIVISOR) ** AREA_WEIGHT_EXPONENT
```
- Example with defaults: area=900, divisor=100, exponent=1.5
- Result: (900/100)^1.5 = 9^1.5 = 27.0
- This means a 900 pixel detection contributes 27.0 to confidence

### Intersection Detection Logic
```python
lane_width > INTERSECTION_MIN_WIDTH AND 
sum_top < INTERSECTION_TOP_THRESH
```
- Wide lane + low top sum = intersection
- Narrow lane or high top sum = not intersection

## Backup Your Settings

Before tuning, back up the file:
```bash
cp tools/controller.py tools/controller.py.backup
```

To restore:
```bash
cp tools/controller.py.backup tools/controller.py
```

## Happy Tuning! 🎯

Remember: Small changes (±10%) often work better than large jumps!
