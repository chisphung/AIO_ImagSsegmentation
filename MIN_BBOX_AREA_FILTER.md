# Minimum Bounding Box Area Filter

**Date:** October 16, 2025  
**Change:** Added minimum bounding box area filter to skip small/far signboard detections

---

## 🎯 What Changed

Added logic to filter out signboard detections with bounding box area below a threshold (default: 200 pixels²). This helps ignore:
- Small signboards that are too far away
- Noisy/false detections with tiny bounding boxes
- Low-confidence detections that shouldn't influence decisions

---

## 📝 Code Changes

### 1. New Hyperparameter

**File:** `tools/controller.py` (line ~19)

```python
# Signboard Collection
SAMPLES_REQUIRED = 30              # Number of samples to collect before confirming signboard
EARLY_TRIGGER_SAMPLES = 10         # Minimum samples for early trigger if intersection detected
LEFT_SIGN_BOOST = 3                # Multiplier for 'left' sign detection (boost importance)
MIN_BBOX_AREA = 200                # ⭐ NEW: Minimum bounding box area to accept detection
```

### 2. Area Check Logic

**File:** `tools/controller.py` (line ~260)

```python
for pred in preds:
    try:
        class_id = int(pred[-1])
    except Exception:
        continue
    if class_id < 0 or class_id >= len(self.class_names):
        continue
    
    # ⭐ NEW: Calculate bounding box area and skip if too small
    boxes = pred[:4]
    try:
        bbox_area = float(max(0.0, (boxes[2] - boxes[0])) * max(0.0, (boxes[3] - boxes[1])))
        if bbox_area < self.MIN_BBOX_AREA:
            if self.VERBOSE_LOGGING:
                print(f"[control] ⏭️  Skipping detection with small area: {bbox_area:.1f} < {self.MIN_BBOX_AREA}")
            continue
    except Exception:
        continue
    
    label = self.class_names[class_id]
    if label in self.traffic_lights:
        # ... process valid detection
```

---

## 🔄 How It Works

### Detection Flow

```
┌─────────────────────────────────────┐
│  YOLO Detection                     │
│  (Signboard found in frame)         │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│  Extract bounding box               │
│  [x1, y1, x2, y2]                   │
└──────────────┬──────────────────────┘
               │
               ↓
┌─────────────────────────────────────┐
│  Calculate area                     │
│  area = (x2 - x1) × (y2 - y1)       │
└──────────────┬──────────────────────┘
               │
               ↓
       ┌───────┴───────┐
       │               │
       ↓               ↓
┌─────────────┐  ┌──────────────────┐
│ area < 200? │  │  area >= 200?    │
│    YES      │  │     YES          │
└──────┬──────┘  └────────┬─────────┘
       │                  │
       ↓                  ↓
┌─────────────┐  ┌──────────────────┐
│  ⏭️ SKIP     │  │  ✅ ACCEPT       │
│  (ignore)   │  │  Add to samples  │
└─────────────┘  └──────────────────┘
```

---

## ✅ Benefits

### 1. **Filter Distant Signboards**
- Signboards that are too far away have small bounding boxes
- These shouldn't trigger signboard collection yet
- Wait until car gets closer for more reliable detection

### 2. **Reduce False Positives**
- Small detections are often noise or misclassifications
- Filtering them improves overall accuracy
- Higher confidence in confirmed signboard type

### 3. **Better Sample Quality**
- Only collect samples from clear, close-range detections
- Improves majority voting accuracy
- More consistent turning decisions

### 4. **Reduce Processing Overhead**
- Skip unnecessary processing of tiny detections
- Cleaner logs (fewer irrelevant detections)
- Focus on meaningful signboards

---

## 🎛️ Tuning the Threshold

### Default Value
```python
MIN_BBOX_AREA = 200  # pixels²
```

### When to Adjust

**Increase to 300-400 if:**
- Getting too many false detections from far away
- Want to wait until signboard is very close
- Getting wrong signboard types from small detections

**Decrease to 100-150 if:**
- Missing signboards (not detecting in time)
- Signboards are small in your camera view
- Need earlier detection trigger

### Area Examples

For reference, typical bounding box areas:

```
Small (far away):     100-200 px²   → Might be filtered
Medium (approaching): 200-500 px²   → Accepted ✓
Large (close):        500-2000 px²  → Accepted ✓
Very large (very close): >2000 px² → Accepted ✓
```

The actual area depends on:
- Camera resolution
- Signboard physical size
- Distance from camera
- Zoom level

---

## 🧪 Testing

### What to Watch For

1. **Log Messages:**
   ```
   [control] ⏭️  Skipping detection with small area: 150.2 < 200
   ```
   This shows detections being filtered

2. **Sample Collection:**
   ```
   [control] 📝 Added sign 'left' to stored list (count=1)
   [control] 💾 Stored: left, area=345.6, confidence=11.95
   ```
   Only larger detections should be stored

3. **No Premature Triggers:**
   - Should not start collecting samples from far-away signboards
   - Wait until signboard is closer and clearer

### Test Scenarios

✅ **Far away signboard:** Should be ignored until car gets closer  
✅ **Approaching signboard:** Should start collecting at appropriate distance  
✅ **Close signboard:** Should collect samples reliably  
✅ **Small noise detections:** Should be filtered out  

---

## 📊 Expected Behavior

### Before Filter (Without MIN_BBOX_AREA check)
```
Frame 100: Detected 'left' area=120.5  ← Too small, noise?
Frame 101: Detected 'right' area=95.3  ← Conflicting, tiny
Frame 102: Detected 'left' area=310.2  ← Good detection
Frame 103: Detected 'left' area=85.7   ← Too small again
...
Result: Mixed samples, lower confidence
```

### After Filter (With MIN_BBOX_AREA = 200)
```
Frame 100: Detected area=120.5 → SKIPPED (< 200)
Frame 101: Detected area=95.3  → SKIPPED (< 200)
Frame 102: Detected 'left' area=310.2 → ACCEPTED ✓
Frame 103: Detected area=85.7  → SKIPPED (< 200)
Frame 104: Detected 'left' area=345.1 → ACCEPTED ✓
...
Result: Clean samples, higher confidence
```

---

## 🔍 Troubleshooting

### Problem: Missing signboards entirely

**Symptoms:**
- Car drives past signboards without detecting
- No samples collected
- No log messages about signboards

**Solutions:**
1. Lower `MIN_BBOX_AREA` to 100-150
2. Check YOLO model is detecting signboards (check raw YOLO output)
3. Verify camera view includes signboards

### Problem: Still getting false positives

**Symptoms:**
- Wrong signboard types collected
- Small detections still being accepted
- Area values just above 200

**Solutions:**
1. Increase `MIN_BBOX_AREA` to 300-400
2. Check if YOLO model needs retraining
3. Increase `SAMPLES_REQUIRED` for more robust majority voting

### Problem: Too many "Skipping" messages

**Symptoms:**
- Logs flooded with skip messages
- Performance impact from too many detections

**Solutions:**
1. Keep filtering (it's working!)
2. Set `VERBOSE_LOGGING = False` to reduce log output
3. Consider YOLO confidence threshold adjustment

---

## 📈 Performance Impact

- **CPU:** Minimal - simple area calculation (multiplication)
- **Memory:** None - no additional storage
- **Detection latency:** None - just a comparison
- **Overall:** Positive - fewer samples to process downstream

---

## 🔗 Related Settings

This filter works together with other signboard settings:

```python
MIN_BBOX_AREA = 200              # Area filter (NEW)
SAMPLES_REQUIRED = 30            # How many samples to collect
AREA_WEIGHT_DIVISOR = 100.0      # How area affects confidence
AREA_WEIGHT_EXPONENT = 1.5       # Area weighting power
```

**Flow:**
1. Detection found
2. **Area check** (MIN_BBOX_AREA) ← NEW FILTER
3. Add to samples (SAMPLES_REQUIRED)
4. Weight by area (AREA_WEIGHT_DIVISOR/EXPONENT)
5. Confirm majority → Wait for intersection

---

## 💡 Best Practices

1. **Start with default (200)** and test
2. **Monitor logs** for skip messages
3. **Adjust based on track** - different tracks may need different values
4. **Consider camera setup** - higher resolution = larger areas
5. **Test at competition** - lighting/conditions may affect detection

---

## 📋 Summary

| Aspect | Details |
|--------|---------|
| **Change** | Added MIN_BBOX_AREA filter |
| **Default Value** | 200 pixels² |
| **Location** | Line ~19 (hyperparameters), Line ~260 (logic) |
| **Purpose** | Filter small/far/noisy detections |
| **Impact** | Better sample quality, fewer false positives |
| **Tunable** | Yes, adjust based on testing |

---

**Status:** ✅ Filter Added Successfully  
**Testing:** Ready for testing with `python main.py`  
**Next Steps:** Test on track and tune threshold if needed

