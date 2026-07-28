# TomoTherapy RTPLAN Private DICOM Tag Investigation

**Scope:** all 39 patients in `DA exit data_new/Complete/` (55 beams total: 35
single-beam Helical patients, 3 four-beam TomoDirect patients, 1 eight-beam
TomoDirect patient — `P32`). Every `*.cdms` file was located recursively,
read with `pydicom`, and walked **exhaustively** (every element in every
nested sequence, not just the fields `preprocessing/rtplan_parser.py`
currently extracts) to guarantee no private tag was missed.

**Rule followed throughout:** a "likely purpose" is only stated when the
dataset itself provides direct evidence (an exact string/value match, or a
computed correlation). Where no such evidence exists, that is reported
explicitly as *unconfirmed* rather than guessed.

---

## 1. Full inventory

Every private element found anywhere in the RTPLAN tree, by structural
location:

| Tag | Structural location | VR | Present in | Occurrences |
|---|---|---|---|---|
| `(300D,0010)` | `BeamSequence[*]` | LO | 39/39 patients, 55/55 beams | 55 (private-block creator ID, not data) |
| `(300D,1040)` | `BeamSequence[*]` | UN | 35/39 patients | 35 (only Helical beams) |
| `(300D,1060)` | `BeamSequence[*]` | UN | 35/39 patients | 35 (only Helical beams) |
| `(300D,1080)` | `BeamSequence[*]` | UN | 39/39 patients, 55/55 beams | 55 |
| `(300D,0010)` | `BeamSequence[*].ControlPointSequence[*]` | LO | 39/39 patients | 42,050 (creator ID, repeated per CP) |
| `(300D,10A7)` | `BeamSequence[*].ControlPointSequence[*]` | UN | 39/39 patients | 42,050 / 46,582 control points (90.3%) |
| `(300D,0010)` | `DoseReferenceSequence[*]` | LO | 39/39 patients | 39 (creator ID) |
| `(300D,1010)` | `DoseReferenceSequence[*]` | UN | 39/39 patients | 39 |
| `(300D,1012)` | `DoseReferenceSequence[*]` | UN | 39/39 patients | 39 |
| `(300D,1016)` | `DoseReferenceSequence[*]` | UN | 39/39 patients | 39 |
| `(300D,1017)` | `DoseReferenceSequence[*]` | UN | 39/39 patients | 39 |
| `(300D,101B)` | `DoseReferenceSequence[*]` | UN | 39/39 patients | 39 |

No private tags were found anywhere else in the tree (top-level dataset,
`FractionGroupSequence`, `ReferencedBeamSequence`, `ReferencedDoseReferenceSequence`,
`BeamLimitingDeviceSequence`, `BeamLimitingDevicePositionSequence`). All
private elements belong to a single private creator block, **`TOMO_HA_01`**
(registered at element `0x0010` of group `0x300D`), confirming every one of
them is TomoTherapy/Hi-Art-proprietary rather than a documented DICOM
standard attribute.

`planned.csv` has 64 signal columns (columns 4+) and `D1.csv` has 640, in
**every one of the 39 patients** (`csv_plan_cols` / `csv_det_cols`, 39/39
each) — these two numbers are the yardsticks used below.

---

## 2. Per-tag analysis

### `(300D,0010)` — Private Creator ID

- **Frequency:** every beam, every control point that carries any other
  `300D` element, every dose-reference item — 100% wherever the block is used.
- **Dimensions:** scalar string.
- **Data type:** LO (string), literal value `"TOMO_HA_01"` in all 39 patients, no exceptions.
- **Changes per control point:** no — constant.
- **Matches planned.csv / detector dims:** n/a (not numeric data).
- **Summary statistics:** n/a (single categorical value, zero variance).
- **Likely purpose (confirmed):** this is not a data field. Per the DICOM
  private-tag mechanism, it reserves the `0x10xx` element block for the
  vendor "TomoTherapy Incorporated, Hi-Art" (`TOMO_HA_01`). It only tells
  us *who* owns the other tags. **Not useful for ML.**

### `(300D,1040)` — beam-level numeric

- **Frequency:** 35/39 patients, 35/55 beams. Cross-checked against beam
  type: present in **35/35 Helical beams, 0/20 FIXED_ANGLE (TomoDirect)
  beams** — a clean, exact split by delivery type, not a random subset.
- **Dimensions:** scalar per beam.
- **Data type:** UN (raw bytes) decoding to ASCII float, e.g. `b'13.2'`.
- **Changes per control point:** n/a — one value per beam, not per CP.
- **Matches planned.csv/detector dims:** no (scalar, not an array).
- **Summary statistics** (Helical beams, n=35): mean **15.29**, std **6.03**,
  min **11.8**, median **13.2**, max **39.4** (P5, `H50 HELICAL`).
- **Correlation tests run** (Pearson r against beam-level quantities
  computed independently from the same file):
  - vs. number of gantry rotations (unwrapped angle): r = −0.19
  - vs. declared control-point count: r = −0.19
  - vs. beam meterset (MU): r = 0.21
  - vs. dose-per-fraction (`TargetPrescriptionDose / NumberOfFractionsPlanned`): **r = 0.80**
- **Likely purpose:** **unconfirmed.** The moderate correlation with
  dose-per-fraction is suggestive but far from a clean identity (r=0.80,
  not ≈1.0), and the value range (11.8–39.4) doesn't obviously match any
  standard unit for that quantity. All other tested candidates (rotation
  count, MU, CP count) were weakly or negatively correlated. I did not find
  evidence strong enough to assign this a name. **Recommend excluding from
  the ML pipeline** until the vendor's private-tag dictionary is available
  to confirm it.

### `(300D,1060)` — beam-level numeric — **Pitch (confirmed)**

- **Frequency:** 35/39 patients, 35/55 beams — identical presence pattern
  to `(300D,1040)`: only Helical beams (0/20 FIXED_ANGLE beams).
- **Dimensions:** scalar per beam.
- **Data type:** UN → ASCII float, e.g. `b'0.17500 '`.
- **Changes per control point:** n/a — one value per beam.
- **Matches planned.csv/detector dims:** no (scalar).
- **Summary statistics** (n=35): mean **0.330**, std **0.125**, min
  **0.122**, median **0.430**, max **0.446**.
- **Evidence:** `BeamDescription` contains free text of the form
  `"Pitch=0.175 and field width=50.44"` for every Helical beam. Parsing
  that text and comparing it against `(300D,1060)` **numerically matched
  in all 35/35 beams, to full decimal precision, zero mismatches.**
- **Likely purpose (confirmed by exact match):** this tag **is** the
  plan's Pitch value, stored redundantly outside the free-text
  `BeamDescription`. **Potentially useful for ML** as a clean numeric
  feature (no text parsing required), though it carries no new
  information beyond what `BeamDescription` parsing already gives —
  useful mainly as a more robust/validated source of the same number.

### `(300D,1080)` — beam-level numeric

- **Frequency:** 39/39 patients, 55/55 beams — present on every beam of
  every type (unlike 1040/1060).
- **Dimensions:** scalar per beam.
- **Data type:** UN → ASCII float, e.g. `b'0.66871 '`.
- **Changes per control point:** n/a — one value per beam.
- **Matches planned.csv/detector dims:** no (scalar).
- **Summary statistics:**
  - Helical beams (n=35): mean **0.617**, std **0.274**, min **0.084**,
    max **0.914** — always < 1.
  - FIXED_ANGLE/TomoDirect beams (n=20): mean **3.711**, std **1.911**,
    min **1.826**, max **8.132** — always > 1, a completely different
    range from Helical.
- **Hypothesis tested: Modulation Factor.** Using the per-control-point
  leaf-open-time array `(300D,10A7)` (see below), I computed the two
  standard candidate ratios per Helical beam: `max/mean(nonzero)` (the
  textbook MF definition, range 1.46–2.06 in this data — clinically
  plausible) and its inverse `mean(nonzero)/max` (range 0.49–0.68,
  overlapping `(300D,1080)`'s range). Correlated each against
  `(300D,1080)`:
  - `max/mean(nonzero)` vs `(300D,1080)`: r = 0.30
  - `mean(nonzero)/max` vs `(300D,1080)`: r = −0.32

  Both are weak. **The Modulation Factor hypothesis is not supported by
  this data** despite superficially plausible value ranges — I'm
  reporting the test and its negative result rather than asserting the
  identity anyway.
- **Likely purpose:** **unconfirmed.** The clean bimodal split by delivery
  type (always <1 for Helical, always >1 for TomoDirect) shows it *is*
  delivery-mode-dependent and probably TPS-optimizer-related (e.g. a cost
  function value, convergence metric, or an MF variant computed
  differently from what I tested) — but I could not confirm which.
  **Recommend excluding from the ML pipeline** without vendor
  documentation.

### `(300D,10A7)` — per-control-point 64-element array — the strongest finding

- **Frequency:** present at 42,050 of 46,582 control points across the
  dataset (90.3%). The gap is structural, not random: it is **never**
  present at a beam's first control point (0/55 beams), and for most
  beams (40/55) it first appears at control-point index 1; for the
  remaining 15 beams it first appears slightly later (index 2, 3, 5, 6, 8,
  or 9). From index-of-first-appearance onward it is present at every
  subsequent control point in the beam.
- **Dimensions:** **exactly 64 values, in all 42,050 occurrences, with zero
  exceptions.**
- **Data type:** UN → backslash-separated ASCII floats (DICOM's normal
  multi-value separator, despite the undocumented VR).
- **Changes per control point:** yes — by construction each control point
  carries its own 64-value array, and spot checks (e.g. `P1`) show no two
  consecutive control points share the same pattern.
- **Matches planned.csv dimensions:** **yes, exactly.** `planned.csv` has
  64 signal columns in all 39 patients — identical to this tag's fixed
  length in all 39 patients. This also matches `configs/config.py`'s
  `leaf_mm = (np.arange(64) - 31.5) * LEAF_PITCH`, i.e. the same 64-leaf
  grid this codebase already uses for the planned-fluence side of the
  pipeline.
- **Matches detector projections:** no — `D1.csv` has 640 columns, not 64;
  no private tag of any kind has 640 elements anywhere in the dataset.
- **Summary statistics** (pooled across every individual value in the
  dataset, split by beam type):

  | | Helical (n=2,612,992 values) | FIXED_ANGLE (n=78,208 values) |
  |---|---|---|
  | mean | 0.110 | 0.102 |
  | std | 0.257 | 0.224 |
  | min | 0.000 | 0.000 |
  | max | 0.9999 | 1.0000 |
  | % exactly 0 | 80.9% | 80.1% |
  | % exactly 1 | 0.0% | 0.1% |
  | % strictly between 0 and 1 | 19.1% | 19.8% |
  | 90th / 99th percentile | 0.535 / 0.997 | 0.513 / 0.883 |

- **Likely purpose:** strongly consistent with a **per-projection,
  per-leaf open-time fraction** — i.e. the binary MLC sinogram driving
  delivery at each control point. Evidence: (1) bounded to `[0, 1]` in
  100% of ~2.7M values, matching a physically normalized "open time
  fraction"; (2) ~80% exactly zero, consistent with a sparse, mostly-closed
  binary MLC; (3) exactly 64 elements, matching TomoTherapy's known
  64-leaf design **and** this dataset's own `planned.csv` column count in
  every patient; (4) absent only at each beam's initial control point(s),
  consistent with a ramp-up/reference point that precedes active
  modulation. I have **not** confirmed the leaf ordering matches
  `planned.csv`'s column ordering (no shared index/leaf-ID field exists to
  verify this directly) — that would need to be checked before using it
  as a positional feature.
- **This is the one private tag with a strong, evidence-based case for ML
  relevance** (see §4).

### `DoseReferenceSequence` private block: `(300D,1010)`, `(300D,1012)`, `(300D,1016)`, `(300D,1017)`, `(300D,101B)`

These five live in a completely different part of the DICOM tree —
`DoseReferenceSequence`, a **plan-level prescription/goal structure**
(exactly one item per patient, 39/39, independent of beam count), sitting
alongside standard public fields like `DoseReferenceType`,
`TargetPrescriptionDose`, `TargetMinimumDose`, `ConstraintWeight`.

| Tag | Frequency | Type | Observed values (39 patients) |
|---|---|---|---|
| `(300D,1010)` | 39/39 | categorical string | `NONE` — 39/39 (zero variance) |
| `(300D,1012)` | 39/39 | small integer | `{1:8, 2:13, 3:7, 4:8, 5:1, 6:1, 7:1}` |
| `(300D,1016)` | 39/39 | integer | range 50–650 (`{100:17, 250:7, 150:4, 500:4, ...}`) |
| `(300D,1017)` | 39/39 | integer | range 100–1000 (`{200:8, 250:6, 350:4, 300:4, ...}`) |
| `(300D,101B)` | 39/39 | categorical string | `TO_VOLUME` — 38/39, `MEDIAN_TO_TUMOR` — 1/39 |

- **Changes per control point:** n/a — plan-level, not per-beam or per-CP.
- **Matches planned.csv/detector dims:** no (scalars, one set per patient).
- **Likely purpose:** the values and the surrounding public fields
  (`DoseReferenceType=TARGET`, `TargetPrescriptionDose`,
  `ConstraintWeight`) place this squarely in **dose-prescription/goal
  bookkeeping** — `(300D,101B)`'s two observed values (`TO_VOLUME`,
  `MEDIAN_TO_TUMOR`) read as a dose-reporting-method enum, and
  `(300D,1010)`'s constant `NONE` and `(300D,1012)`'s small integer range
  are consistent with unused/optional-field and priority/type codes in
  the same structure. None of the four numeric fields correlate with
  anything delivery-related by construction — they belong to a different
  part of the plan than the beam/control-point data. **Not useful for
  predicting delivered detector signal from planned fluence** — this
  describes *why* a dose was prescribed, not *how* it was delivered.

---

## 3. Cross-cutting findings (not private tags, but relevant context found during this investigation)

- **Control points per gantry rotation is a tight dataset-wide constant.**
  Independently unwrapping `GantryAngle` across each Helical beam's control
  points and dividing by declared control-point count gives **51.10 ± 0.05
  control points per rotation across all 35 Helical beams** (min 51.03, max
  51.23). This is a genuine structural constant of the delivery system,
  derived from the data itself.
- **`planned.csv`/`D1.csv` row counts don't exactly equal declared control
  points.** For 36/39 patients, `total_declared_control_points −
  csv_row_count = 2 × num_beams` exactly — i.e. each beam contributes
  precisely 2 "extra" control points (almost certainly the same
  start/end reference control points that also lack the `(300D,10A7)`
  leaf array). **3 patients break this pattern**: `P2` (492 more CPs
  declared than CSV rows), `P33` (**796 fewer** CPs declared than CSV
  rows — the CSV has more rows than the plan has control points), `P38`
  (1499 more CPs declared than CSV rows). I did not find a confirmed
  explanation in the DICOM data for these three (candidate explanations —
  multi-fraction concatenation in the CSV, a partial/interrupted
  export, or an adaptive re-plan not reflected in the single RTPLAN file
  audited — are plausible but unverified). **Flag these three patients
  for manual review before using their row-indexed data for
  control-point-level ML supervision.**

---

## 4. What's actually useful for ML

| Tag | Verdict | Reasoning |
|---|---|---|
| `(300D,10A7)` (per-CP, 64-leaf array) | **Worth pursuing** | Only private tag with a confirmed dimensional match to existing pipeline data (64 = `planned.csv` columns = `configs.config` leaf grid), a physically sane bounded/sparse distribution, and a plausible, well-evidenced identity (leaf-open-time sinogram). Before using it as a feature: (a) verify leaf-index ordering against `planned.csv` columns — no shared key currently proves this, (b) resolve the CP-count vs CSV-row-count offset (§3) so control points align to the correct CSV rows, (c) handle the ~10% of control points where it's absent (beam start). |
| `(300D,1060)` (Pitch) | **Low-value but safe** | Confirmed identity, but redundant with `BeamDescription` text already available; only adds robustness, not new information. Helical-only (35/39 patients) — TomoDirect patients would need imputation or a separate delivery-mode feature. |
| `(300D,1040)`, `(300D,1080)` | **Exclude for now** | No confirmed identity despite multiple correlation tests against physically meaningful candidates. Including unlabeled/unconfirmed scalars as model features risks encoding TPS-internal bookkeeping the model can't generalize from, or worse, a label-adjacent leak (e.g. if either turns out to be an optimizer output computed from the eventual dose). |
| `(300D,0010)` (creator ID) | **Exclude** | Constant string, zero information. |
| `DoseReferenceSequence` block (`1010`,`1012`,`1016`,`1017`,`101B`) | **Exclude** | Plan-level prescription/goal bookkeeping, structurally and semantically unrelated to beam delivery physics; no path to affecting the detector signal. |

**Bottom line:** of the seven distinct private data tags found (excluding
the creator-ID marker), only **`(300D,10A7)`** has strong dataset-internal
evidence tying it to the physical quantity this project already models
(planned fluence / leaf modulation). It's the only one I'd recommend
spending integration effort on next — as a genuinely new per-projection,
per-leaf signal, not derivable from `planned.csv`/`D1.csv` alone. Everything
else in the private-tag space is either redundant with data already in the
pipeline (`(300D,1060)`) or unconfirmed/unrelated (everything else).
