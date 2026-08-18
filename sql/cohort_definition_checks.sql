-- Cohort definition and verification queries.
-- Original-model phase. Run in order; each was used to confirm a
-- specific cohort decision before it was locked.

-- ============================================================
-- 1. Baseline count: all septic ICU stays, before any filters
-- ============================================================
SELECT
  COUNT(DISTINCT s.stay_id) AS total_sepsis_stays,
  COUNT(DISTINCT CASE WHEN k.max_aki_stage >= 1 THEN s.stay_id END) AS sepsis_with_aki,
  ROUND(
    COUNT(DISTINCT CASE WHEN k.max_aki_stage >= 1 THEN s.stay_id END) * 100.0
    / COUNT(DISTINCT s.stay_id), 1
  ) AS aki_rate_percent
FROM `physionet-data.mimiciv_3_1_derived.sepsis3` AS s
LEFT JOIN (
  SELECT stay_id, MAX(aki_stage) AS max_aki_stage
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages`
  GROUP BY stay_id
) AS k ON s.stay_id = k.stay_id;
-- Result: 32,899 total stays, 77.5% AKI rate (before this phase's
-- additional filters: strict AKI-before-sepsis exclusion, LOS >= 48h,
-- age 18-90, minimum post-onset ICU time)

-- ============================================================
-- 2. Admission type distribution, on the locked cohort
-- ============================================================
-- Used to confirm the reference category for one-hot encoding, and to
-- check whether SA1's reference category ("DIRECT EMER.") still made
-- sense for this phase's differently-filtered cohort. It did not
-- (only 3.1% of this cohort) -- Emergency (grouped) was used instead.

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
  SELECT icu.stay_id, icu.hadm_id, icu.outtime AS icu_outtime,
    TIMESTAMP_DIFF(icu.outtime, icu.intime, HOUR) AS icu_los_hours,
    d.admission_age
  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` AS d ON icu.stay_id = d.stay_id
),
admission_info AS (
  SELECT hadm_id, admission_type
  FROM `physionet-data.mimiciv_3_1_hosp.admissions`
),
cohort AS (
  SELECT s.stay_id, adm.admission_type
  FROM sepsis_onset AS s
  INNER JOIN icu_info AS i ON s.stay_id = i.stay_id
  INNER JOIN admission_info AS adm ON i.hadm_id = adm.hadm_id
  LEFT JOIN aki_onset AS a ON s.stay_id = a.stay_id
  WHERE i.admission_age BETWEEN 18 AND 90
    AND i.icu_los_hours >= 48
    AND (a.aki_onset_time IS NULL OR a.aki_onset_time >= s.sepsis_onset_time)
    AND TIMESTAMP_DIFF(i.icu_outtime, s.sepsis_onset_time, HOUR) >= 24
)
SELECT
  admission_type,
  COUNT(*) AS n_stays,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_cohort
FROM cohort
GROUP BY admission_type
ORDER BY n_stays DESC;
-- Result: 8 categories, 14,826 total. EW EMER. 54.1%, URGENT 22.1%,
-- OBSERVATION ADMIT 8.8%, SURGICAL SAME DAY ADMISSION 8.1%,
-- ELECTIVE 3.7%, DIRECT EMER. 3.1%, DIRECT OBSERVATION <0.1%,
-- EU OBSERVATION <0.1%. Grouped into Emergency (88.2%) / Elective
-- (11.8%) per the community-sourced mapping documented in the
-- methodology (limitation: not officially validated, recommend
-- clinician review before reuse).

-- ============================================================
-- 3. Sepsis onset hour-alignment check
-- ============================================================
-- Checked whether sepsis_onset_time (GREATEST(suspected_infection_time,
-- sofa_time)) is typically exactly on the hour, since this determined
-- whether the extraction's hour-binning needed to be relative to each
-- patient's own onset (rather than calendar clock time).

WITH sepsis_onset AS (
  SELECT stay_id, GREATEST(suspected_infection_time, sofa_time) AS sepsis_onset_time
  FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
  WHERE sepsis3 = TRUE
),
icu_info AS (
  SELECT icu.stay_id, icu.outtime AS icu_outtime,
    TIMESTAMP_DIFF(icu.outtime, icu.intime, HOUR) AS icu_los_hours,
    d.admission_age
  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` AS d ON icu.stay_id = d.stay_id
),
aki_onset AS (
  SELECT stay_id, MIN(charttime) AS aki_onset_time
  FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages`
  WHERE aki_stage >= 1
  GROUP BY stay_id
),
cohort AS (
  SELECT s.stay_id, s.sepsis_onset_time
  FROM sepsis_onset AS s
  INNER JOIN icu_info AS i ON s.stay_id = i.stay_id
  LEFT JOIN aki_onset AS a ON s.stay_id = a.stay_id
  WHERE i.admission_age BETWEEN 18 AND 90
    AND i.icu_los_hours >= 48
    AND (a.aki_onset_time IS NULL OR a.aki_onset_time >= s.sepsis_onset_time)
    AND TIMESTAMP_DIFF(i.icu_outtime, s.sepsis_onset_time, HOUR) >= 24
)
SELECT
  COUNT(*) AS total_stays,
  COUNT(CASE WHEN EXTRACT(MINUTE FROM sepsis_onset_time) = 0
              AND EXTRACT(SECOND FROM sepsis_onset_time) = 0 THEN 1 END) AS exactly_on_the_hour,
  ROUND(100.0 * COUNT(CASE WHEN EXTRACT(MINUTE FROM sepsis_onset_time) = 0
              AND EXTRACT(SECOND FROM sepsis_onset_time) = 0 THEN 1 END) / COUNT(*), 1) AS pct_on_the_hour
FROM cohort;
-- Result: 89.8% exactly on the hour, 10.2% (1,517 stays) not.
-- Confirmed the extraction needed to bin hours relative to each
-- patient's own onset time, not calendar clock time.

-- ============================================================
-- 4. Sepsis onset before ICU admission -- edge case check
-- ============================================================
-- Checked how often sepsis_onset_time falls before icu_intime, since
-- the labs extraction bounds readings to each stay's own ICU window
-- (icu_intime to extraction_end), which could in principle exclude
-- legitimate pre-ICU labs for these patients.

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
  SELECT icu.stay_id, icu.intime AS icu_intime, icu.outtime AS icu_outtime,
    TIMESTAMP_DIFF(icu.outtime, icu.intime, HOUR) AS icu_los_hours,
    d.admission_age
  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` AS d ON icu.stay_id = d.stay_id
),
cohort AS (
  SELECT s.stay_id, s.sepsis_onset_time, i.icu_intime
  FROM sepsis_onset AS s
  INNER JOIN icu_info AS i ON s.stay_id = i.stay_id
  LEFT JOIN aki_onset AS a ON s.stay_id = a.stay_id
  WHERE i.admission_age BETWEEN 18 AND 90
    AND i.icu_los_hours >= 48
    AND (a.aki_onset_time IS NULL OR a.aki_onset_time >= s.sepsis_onset_time)
    AND TIMESTAMP_DIFF(i.icu_outtime, s.sepsis_onset_time, HOUR) >= 24
)
SELECT
  COUNT(*) AS total_stays,
  COUNT(CASE WHEN sepsis_onset_time < icu_intime THEN 1 END) AS onset_before_icu_admission,
  ROUND(100.0 * COUNT(CASE WHEN sepsis_onset_time < icu_intime THEN 1 END) / COUNT(*), 2) AS pct_affected,
  ROUND(AVG(CASE WHEN sepsis_onset_time < icu_intime
            THEN TIMESTAMP_DIFF(icu_intime, sepsis_onset_time, MINUTE) END), 1) AS avg_gap_minutes_when_affected
FROM cohort;
-- Result: 0.47% of cohort (70 stays), average 42-minute gap.
-- Judged negligible; documented as an accepted limitation, not fixed.
