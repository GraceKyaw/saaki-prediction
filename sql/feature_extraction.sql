-- MAIN EXTRACTION (single query, extract once) -- CORRECTED VERSION
-- WIDE format: one row per (stay_id, hour), all 25 features as
-- columns side by side. Produced directly in this query -- no pivot
-- step needed afterward in Python.
-- CORRECTION: hour_bucket is now an integer number of HOURS SINCE
-- SEPSIS ONSET (0, 1, 2, ...), not a calendar clock-hour timestamp.
-- The previous version used TIMESTAMP_TRUNC(charttime, HOUR), which
-- rounds to the calendar hour -- for the 10.2% of stays (1,517 of
-- 14,826) whose sepsis_onset_time does not fall exactly on the hour,
-- this produced a partial, ambiguous first hour. Binning by
-- TIMESTAMP_DIFF(charttime, sepsis_onset_time, HOUR) instead makes
-- "hour 0" unambiguous for every patient: the first full hour after
-- their own onset, regardless of what clock time onset fell on.
-- Static features (age, gender, admission_type) and label-construction
-- timestamps (sepsis_onset_time, extraction_end, aki_onset_time) are
-- included as columns on every row for that stay -- so this one query
-- is the only extraction needed; preprocessing derives whatever each
-- model representation needs from this single output, per the
-- "extract once, derive three ways" plan.
--
-- Scope: locked cohort (14,826 stays), 25 confirmed features,
-- from sepsis onset to (ICU discharge OR sepsis onset + 30 days),
-- whichever comes first. Multiple readings within the same hour are
-- averaged. Temperature is standardized to Celsius before averaging
-- (Fahrenheit itemid 223761 converted via (F-32)*5/9; Celsius itemid
-- 223762 left as-is). Missing hours are simply absent -- NOT filled
-- in here; forward-fill / indicators / imputation all happen in
-- preprocessing.

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
  SELECT
    icu.stay_id, icu.hadm_id, icu.subject_id,
    icu.intime AS icu_intime, icu.outtime AS icu_outtime,
    TIMESTAMP_DIFF(icu.outtime, icu.intime, HOUR) AS icu_los_hours,
    d.admission_age, d.gender
  FROM `physionet-data.mimiciv_3_1_icu.icustays` AS icu
  INNER JOIN `physionet-data.mimiciv_3_1_derived.icustay_detail` AS d ON icu.stay_id = d.stay_id
),
admission_info AS (
  SELECT hadm_id, admission_type
  FROM `physionet-data.mimiciv_3_1_hosp.admissions`
),
cohort AS (
  -- fully locked cohort: 14,826 stays, with static features and
  -- label-timestamps attached, ready to be repeated onto every hourly row
  SELECT
    s.stay_id, i.subject_id,
    s.sepsis_onset_time,
    i.icu_intime, i.icu_outtime,
    LEAST(i.icu_outtime, TIMESTAMP_ADD(s.sepsis_onset_time, INTERVAL 30 DAY)) AS extraction_end,
    i.admission_age, i.gender, adm.admission_type,
    a.aki_onset_time
  FROM sepsis_onset AS s
  INNER JOIN icu_info AS i ON s.stay_id = i.stay_id
  INNER JOIN admission_info AS adm ON i.hadm_id = adm.hadm_id
  LEFT JOIN aki_onset AS a ON s.stay_id = a.stay_id
  WHERE i.admission_age BETWEEN 18 AND 90
    AND i.icu_los_hours >= 48
    AND (a.aki_onset_time IS NULL OR a.aki_onset_time >= s.sepsis_onset_time)
    AND TIMESTAMP_DIFF(i.icu_outtime, s.sepsis_onset_time, HOUR) >= 24
),

-- itemid -> feature_name mapping, vitals (chartevents)
vitals_readings AS (
  SELECT
    c.stay_id,
    CASE
      WHEN ce.itemid = 220045 THEN 'heart_rate'
      WHEN ce.itemid IN (223762, 223761) THEN 'temperature'
      WHEN ce.itemid = 220277 THEN 'spo2'
      WHEN ce.itemid IN (220179, 220050, 225309) THEN 'sbp'
      WHEN ce.itemid IN (220180, 220051, 225310) THEN 'dbp'
      WHEN ce.itemid IN (220052, 220181, 225312) THEN 'map'
    END AS feature_name,
    TIMESTAMP_DIFF(ce.charttime, c.sepsis_onset_time, HOUR) AS hour_bucket,
    -- convert Fahrenheit (223761) to Celsius before it can be mixed
    -- with the Celsius itemid (223762) in the same average
    CASE WHEN ce.itemid = 223761 THEN (ce.valuenum - 32) * 5 / 9 ELSE ce.valuenum END AS valuenum
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` AS ce
  INNER JOIN cohort AS c ON ce.stay_id = c.stay_id
  WHERE ce.itemid IN (220045, 223762, 223761, 220277,
                       220179, 220050, 225309, 220180, 220051, 225310, 220052, 220181, 225312)
    AND ce.valuenum IS NOT NULL
    AND ce.charttime BETWEEN c.sepsis_onset_time AND c.extraction_end
),

-- itemid -> feature_name mapping, labs (labevents)
-- bounded by this specific stay's intime/outtime via charttime, not
-- just hadm_id, to avoid mis-attributing labs across multiple ICU
-- stays within the same hospital admission
labs_readings AS (
  SELECT
    c.stay_id,
    CASE
      WHEN le.itemid IN (50868, 52500) THEN 'anion_gap'
      WHEN le.itemid IN (50912, 52546) THEN 'creatinine'
      WHEN le.itemid = 51222 THEN 'hemoglobin'
      WHEN le.itemid = 51249 THEN 'mchc'
      WHEN le.itemid = 51237 THEN 'inr'
      WHEN le.itemid = 51265 THEN 'platelets'
      WHEN le.itemid IN (50885, 53089) THEN 'total_bilirubin'
      WHEN le.itemid IN (50971, 52610) THEN 'potassium'
      WHEN le.itemid = 50970 THEN 'phosphate'
      WHEN le.itemid IN (50813, 52442) THEN 'lactate'
      WHEN le.itemid = 50820 THEN 'ph'
      WHEN le.itemid = 50821 THEN 'pao2'
      WHEN le.itemid = 50818 THEN 'paco2'
      WHEN le.itemid IN (51006, 52647) THEN 'bun'
      WHEN le.itemid IN (51301, 51755) THEN 'wbc'
    END AS feature_name,
    TIMESTAMP_DIFF(le.charttime, c.sepsis_onset_time, HOUR) AS hour_bucket,
    le.valuenum
  FROM `physionet-data.mimiciv_3_1_hosp.labevents` AS le
  INNER JOIN `physionet-data.mimiciv_3_1_icu.icustays` AS icu ON le.hadm_id = icu.hadm_id
  INNER JOIN cohort AS c ON icu.stay_id = c.stay_id
  WHERE le.itemid IN (50868, 52500, 50912, 52546, 51222, 51249, 51237, 51265,
                       50885, 53089, 50971, 52610, 50970, 50813, 52442,
                       50820, 50821, 50818, 51006, 52647, 51301, 51755)
    AND le.valuenum IS NOT NULL
    AND le.charttime BETWEEN c.sepsis_onset_time AND c.extraction_end
    -- bound to this specific stay's actual ICU window
    AND le.charttime BETWEEN c.icu_intime AND c.extraction_end
),

-- urine output (already correctly combined by the official derived table)
urine_readings AS (
  SELECT
    c.stay_id,
    'urine_output' AS feature_name,
    TIMESTAMP_DIFF(uo.charttime, c.sepsis_onset_time, HOUR) AS hour_bucket,
    uo.urineoutput AS valuenum
  FROM `physionet-data.mimiciv_3_1_derived.urine_output` AS uo
  INNER JOIN cohort AS c ON uo.stay_id = c.stay_id
  WHERE uo.charttime BETWEEN c.sepsis_onset_time AND c.extraction_end
),

all_readings AS (
  SELECT * FROM vitals_readings
  UNION ALL
  SELECT * FROM labs_readings
  UNION ALL
  SELECT * FROM urine_readings
)

SELECT
  ar.stay_id,
  c.subject_id,
  ar.hour_bucket,
  AVG(CASE WHEN ar.feature_name = 'heart_rate' THEN ar.valuenum END) AS heart_rate,
  AVG(CASE WHEN ar.feature_name = 'temperature' THEN ar.valuenum END) AS temperature,
  AVG(CASE WHEN ar.feature_name = 'spo2' THEN ar.valuenum END) AS spo2,
  AVG(CASE WHEN ar.feature_name = 'sbp' THEN ar.valuenum END) AS sbp,
  AVG(CASE WHEN ar.feature_name = 'dbp' THEN ar.valuenum END) AS dbp,
  AVG(CASE WHEN ar.feature_name = 'map' THEN ar.valuenum END) AS map,
  AVG(CASE WHEN ar.feature_name = 'anion_gap' THEN ar.valuenum END) AS anion_gap,
  AVG(CASE WHEN ar.feature_name = 'creatinine' THEN ar.valuenum END) AS creatinine,
  AVG(CASE WHEN ar.feature_name = 'hemoglobin' THEN ar.valuenum END) AS hemoglobin,
  AVG(CASE WHEN ar.feature_name = 'mchc' THEN ar.valuenum END) AS mchc,
  AVG(CASE WHEN ar.feature_name = 'inr' THEN ar.valuenum END) AS inr,
  AVG(CASE WHEN ar.feature_name = 'platelets' THEN ar.valuenum END) AS platelets,
  AVG(CASE WHEN ar.feature_name = 'total_bilirubin' THEN ar.valuenum END) AS total_bilirubin,
  AVG(CASE WHEN ar.feature_name = 'potassium' THEN ar.valuenum END) AS potassium,
  AVG(CASE WHEN ar.feature_name = 'phosphate' THEN ar.valuenum END) AS phosphate,
  AVG(CASE WHEN ar.feature_name = 'lactate' THEN ar.valuenum END) AS lactate,
  AVG(CASE WHEN ar.feature_name = 'ph' THEN ar.valuenum END) AS ph,
  AVG(CASE WHEN ar.feature_name = 'pao2' THEN ar.valuenum END) AS pao2,
  AVG(CASE WHEN ar.feature_name = 'paco2' THEN ar.valuenum END) AS paco2,
  AVG(CASE WHEN ar.feature_name = 'bun' THEN ar.valuenum END) AS bun,
  AVG(CASE WHEN ar.feature_name = 'wbc' THEN ar.valuenum END) AS wbc,
  AVG(CASE WHEN ar.feature_name = 'urine_output' THEN ar.valuenum END) AS urine_output,
  -- static features and label-construction timestamps: one value per
  -- stay, naturally not duplicated since there's only one row per hour
  c.admission_age,
  c.gender,
  c.admission_type,
  c.sepsis_onset_time,
  c.extraction_end,
  c.aki_onset_time
FROM all_readings AS ar
INNER JOIN cohort AS c ON ar.stay_id = c.stay_id
GROUP BY ar.stay_id, c.subject_id, ar.hour_bucket,
  c.admission_age, c.gender, c.admission_type,
  c.sepsis_onset_time, c.extraction_end, c.aki_onset_time
ORDER BY ar.stay_id, ar.hour_bucket;