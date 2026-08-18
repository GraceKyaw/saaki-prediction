-- Feature and itemid verification queries.
-- Original-model phase. Used to confirm itemid mappings before
-- locking the feature list and building the extraction query.

-- ============================================================
-- 1. Temperature unit verification
-- ============================================================
-- Confirmed which chartevents itemid reports Celsius vs Fahrenheit,
-- since mixing them without conversion would corrupt the average.

SELECT
  itemid,
  MIN(valuenum) AS min_value,
  MAX(valuenum) AS max_value,
  AVG(valuenum) AS avg_value,
  COUNT(*) AS n_readings
FROM `physionet-data.mimiciv_3_1_icu.chartevents`
WHERE itemid IN (223761, 223762)
  AND valuenum IS NOT NULL
GROUP BY itemid;
-- Result: 223762 reports Celsius (~35-38 range), 223761 reports
-- Fahrenheit (~95-101 range). 223761 converted via (F-32)*5/9 before
-- averaging with 223762 in the extraction query.

-- ============================================================
-- 2. BUN and WBC dual-itemid verification
-- ============================================================
-- Checked whether the secondary itemid for BUN (52647) and WBC
-- (51755) represent the same underlying test as the primary itemid
-- (51006, 51301 respectively), by finding a patient with both itemids
-- present and checking whether the secondary reading is consistent
-- with the surrounding history from the primary itemid (same
-- system-transition pattern already established for creatinine,
-- anion gap, and potassium in SA3).

-- BUN check (patient 13227028): single 52647 reading embedded in
-- continuous 51006 history, value consistent (20 mg/dL surrounded by
-- 19/19) -- combined.
SELECT charttime, itemid, valuenum
FROM `physionet-data.mimiciv_3_1_hosp.labevents`
WHERE subject_id = 13227028
  AND itemid IN (51006, 52647)
ORDER BY charttime;

-- WBC check (patient 17859352): single 51755 reading embedded in
-- continuous 51301 history, value consistent (5.6 K/uL surrounded by
-- 3.7/3.9) -- combined.
SELECT charttime, itemid, valuenum
FROM `physionet-data.mimiciv_3_1_hosp.labevents`
WHERE subject_id = 17859352
  AND itemid IN (51301, 51755)
ORDER BY charttime;
-- Note: 51756 (a third possible WBC itemid) was not independently
-- verified via this same per-patient check -- it did not appear in
-- the sampled patient's record. Not included in the final mapping;
-- documented as an unconfirmed extension in the methodology.

-- ============================================================
-- 3. Itemid lookup for the 5 features added beyond SA3's base set
-- ============================================================
-- SBP, DBP, MAP, BUN, WBC -- confirmed against d_items and cross-
-- checked against the official vitalsign.sql source for blood
-- pressure specifically (225312, the MAP itemid from arterial lines,
-- was missing from a keyword search and only found via the official
-- source).

SELECT itemid, label, category, unitname
FROM `physionet-data.mimiciv_3_1_icu.d_items`
WHERE itemid IN (220179, 220050, 225309,  -- SBP
                  220180, 220051, 225310,  -- DBP
                  220052, 220181, 225312,  -- MAP
                  220045)                  -- heart rate (existing, for reference)
ORDER BY itemid;

SELECT itemid, label, category, fluid
FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
WHERE itemid IN (51006, 52647,   -- BUN
                  51301, 51755)  -- WBC
ORDER BY itemid;
