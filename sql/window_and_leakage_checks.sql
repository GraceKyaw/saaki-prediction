-- Observation window and leakage-check queries.
-- Original-model phase. Used to decide and verify the 24-hour window
-- length, the AKI-during-window leakage exclusion, and the sliding-
-- window step size.

-- ============================================================
-- 1. Retention at candidate window lengths
-- ============================================================
-- Checked how much of the cohort would be retained at 24h vs 48h
-- observation windows, to inform the window-length decision.

WITH sepsis_onset AS (
  SELECT stay_id, GREATEST(suspected_infection_time, sofa_time) AS sepsis_onset_time
  FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
  WHERE sepsis3 = TRUE
),
aki_onset AS (
  SELECT stay_id, MIN(charttime) AS aki_onset_time
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages`
  WHERE aki_stage >= 1
  GROUP BY stay_id
),
icu_info AS (
  SELECT icu.stay_id, icu.outtime AS icu_outtime,
    TIMESTAMP_DIFF(icu.outtime, icu.intime, HOUR) AS icu_los_hours,
    d.admission_age
  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` AS d ON icu.stay_id = d.stay_id
),
cohort AS (
  SELECT s.stay_id, s.sepsis_onset_time, i.icu_outtime
  FROM sepsis_onset AS s
  INNER JOIN icu_info AS i ON s.stay_id = i.stay_id
  LEFT JOIN aki_onset AS a ON s.stay_id = a.stay_id
  WHERE i.admission_age BETWEEN 18 AND 90
    AND i.icu_los_hours >= 48
    AND (a.aki_onset_time IS NULL OR a.aki_onset_time >= s.sepsis_onset_time)
)
SELECT
  COUNT(*) AS total_stays,
  COUNT(CASE WHEN TIMESTAMP_DIFF(icu_outtime, sepsis_onset_time, HOUR) >= 24 THEN 1 END) AS retained_at_24h,
  COUNT(CASE WHEN TIMESTAMP_DIFF(icu_outtime, sepsis_onset_time, HOUR) >= 48 THEN 1 END) AS retained_at_48h
FROM cohort;
-- Result: 24h retains 99.9% of cohort (14,826 of 14,839); 48h retains
-- 92.9%. 24h chosen: near-total retention, avoids selection bias
-- toward patients with longer stays, matches literature precedent,
-- more clinically actionable lead time.

-- ============================================================
-- 2. AKI-during-window leakage check (single-window models)
-- ============================================================
-- Classified each stay as valid-negative, valid-positive, or
-- excluded-leakage under the single-window rule (AKI must occur
-- strictly after the 24h window closes).

WITH sepsis_onset AS (
  SELECT stay_id, GREATEST(suspected_infection_time, sofa_time) AS sepsis_onset_time
  FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
  WHERE sepsis3 = TRUE
),
aki_onset AS (
  SELECT stay_id, MIN(charttime) AS aki_onset_time
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages`
  WHERE aki_stage >= 1
  GROUP BY stay_id
),
icu_info AS (
  SELECT icu.stay_id, icu.outtime AS icu_outtime,
    TIMESTAMP_DIFF(icu.outtime, icu.intime, HOUR) AS icu_los_hours,
    d.admission_age
  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` AS d ON icu.stay_id = d.stay_id
),
cohort AS (
  SELECT
    s.stay_id, s.sepsis_onset_time,
    TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR) AS window_end,
    a.aki_onset_time
  FROM sepsis_onset AS s
  INNER JOIN icu_info AS i ON s.stay_id = i.stay_id
  LEFT JOIN aki_onset AS a ON s.stay_id = a.stay_id
  WHERE i.admission_age BETWEEN 18 AND 90
    AND i.icu_los_hours >= 48
    AND (a.aki_onset_time IS NULL OR a.aki_onset_time >= s.sepsis_onset_time)
    AND TIMESTAMP_DIFF(i.icu_outtime, s.sepsis_onset_time, HOUR) >= 24
)
SELECT
  CASE
    WHEN aki_onset_time IS NULL THEN 'valid_negative'
    WHEN aki_onset_time > window_end THEN 'valid_positive'
    ELSE 'excluded_leakage'
  END AS category,
  COUNT(*) AS n_stays
FROM cohort
GROUP BY category;
-- Result: valid_negative 2,297, valid_positive 3,601, excluded_leakage
-- 8,928 (60.2% of 14,826). Final single-window cohort: 5,898 stays.

-- ============================================================
-- 3. Sliding-window step size comparison (12h vs 24h)
-- ============================================================
-- Compared total window instances and contributing patients at two
-- candidate step sizes, to decide between them. Uses the strict,
-- corrected boundary rule (aki_onset_time > window_end), matching the
-- single-window definition exactly, so contributing-patient counts
-- are directly comparable to the single-window cohort above.

WITH sepsis_onset AS (
  SELECT stay_id, GREATEST(suspected_infection_time, sofa_time) AS sepsis_onset_time
  FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
  WHERE sepsis3 = TRUE
),
aki_onset AS (
  SELECT stay_id, MIN(charttime) AS aki_onset_time
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages`
  WHERE aki_stage >= 1
  GROUP BY stay_id
),
icu_info AS (
  SELECT icu.stay_id, icu.outtime AS icu_outtime,
    TIMESTAMP_DIFF(icu.outtime, icu.intime, HOUR) AS icu_los_hours,
    d.admission_age
  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` AS d ON icu.stay_id = d.stay_id
),
locked_cohort AS (
  SELECT
    s.stay_id,
    TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 24 HOUR) AS window_end,
    LEAST(i.icu_outtime, TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 30 DAY)) AS extraction_end,
    a.aki_onset_time
  FROM sepsis_onset AS s
  INNER JOIN icu_info AS i ON s.stay_id = i.stay_id
  LEFT JOIN aki_onset AS a ON s.stay_id = a.stay_id
  WHERE i.admission_age BETWEEN 18 AND 90
    AND i.icu_los_hours >= 48
    AND (a.aki_onset_time IS NULL OR a.aki_onset_time >= s.sepsis_onset_time)
    AND TIMESTAMP_DIFF(i.icu_outtime, s.sepsis_onset_time, HOUR) >= 24
),
eligibility AS (
  SELECT
    stay_id, window_end, extraction_end, aki_onset_time,
    (aki_onset_time IS NULL OR aki_onset_time > window_end) AS has_at_least_one_window
  FROM locked_cohort
),
windows_per_stay AS (
  SELECT
    stay_id,
    CASE WHEN has_at_least_one_window THEN
      GREATEST(1,
        CAST(FLOOR(
          TIMESTAMP_DIFF(
            LEAST(extraction_end, COALESCE(TIMESTAMP_SUB(aki_onset_time, INTERVAL 1 SECOND), extraction_end)),
            window_end,
            HOUR
          ) / 24
        ) AS INT64) + 1
      )
    ELSE 0 END AS n_windows
  FROM eligibility
)
SELECT
  COUNT(DISTINCT CASE WHEN n_windows > 0 THEN stay_id END) AS n_stays_contributing,
  SUM(n_windows) AS total_window_instances
FROM windows_per_stay;
-- Result (24h step, strict boundary): 5,898 contributing patients
-- (matches the single-window cohort exactly, as it should), 14,623
-- total window instances. 24h step size locked over 12h: a shorter
-- step increases overlap/correlation between windows without
-- meaningfully expanding the eligible patient pool.
