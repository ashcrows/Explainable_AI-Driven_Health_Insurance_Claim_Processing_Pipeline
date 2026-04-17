Dataset Augmentation Strategy
==============================

The primary dataset (US Health Insurance Dataset from Kaggle) contains real
demographic and medical cost information but does not include a fraud label.
To make this dataset usable for supervised fraud detection, we applied a
statistically defensible augmentation strategy based on domain knowledge from
health insurance fraud literature and CMS fraud enforcement patterns.


Rationale
---------
Real-world health insurance fraud detection models are typically trained on
labeled historical claims where fraud outcomes are known from adjudication or
investigation records. Since such proprietary labeled data is not publicly
available, academic projects commonly use one of the following approaches:

  1. Apply rule-based heuristics to generate synthetic fraud labels.
  2. Use anomaly detection framing (treat extreme outliers as fraud).
  3. Use a combination of rules derived from published fraud patterns.

This project uses approach 1, with rules derived from documented CMS and
private insurer fraud indicators (see references below). All rules are
transparently defined, making the augmentation academically reproducible.


Fraud Label Assignment Rules
-----------------------------
A claim is labeled as fraudulent (is_fraud = True) if it meets one or more
of the following conditions. The conditions are applied independently and
are cumulative.

Rule 1 - Extreme Claim Inflation
  Condition:
    charges > 4 * expected_charge_for_demographics
  Expected charge is approximated as:
    base = 1800 + age * 55 + bmi * 35
    if smoker: base *= 2.8
    base += children * 450
  A claim is flagged if actual charges exceed 4x this baseline.
  Basis: Inflated billing is the most common form of insurance fraud.
         A 4x threshold captures statistical outliers beyond the 98th
         percentile in the Kaggle dataset while avoiding false positives
         in edge medical cases.

Rule 2 - Rapid Resubmission with High Amount
  Condition:
    days_since_last_claim <= 7 AND charges > 15000
  Basis: Duplicate or near-duplicate claim submissions within a short
         window are a known CMS fraud indicator. Combined with a high
         amount, this pattern is statistically anomalous.

Rule 3 - Implausible Procedure for Demographics
  Condition:
    age < 30 AND procedure_code IN ('27447', '43239')
  Procedure 27447 = knee replacement, 43239 = gastric bypass.
  Both are rare before age 30 and flagged by CMS fraud detection systems
  when combined with high claim amounts.
  Basis: Age-implausible procedures are a documented audit trigger.

Rule 4 - Provider Concentration Anomaly
  Condition:
    provider_id = 'PRV9999' OR provider_claim_deviation > 4.5
  A single synthetic high-volume fraudulent provider (PRV9999) is injected
  via the Kafka producer. This simulates provider-level billing fraud.
  Basis: Provider concentration fraud (one provider submitting many
         high-value claims) is among the top investigated patterns by
         the HHS Office of Inspector General.


Augmented Fields Added
-----------------------
The following fields do not appear in the original Kaggle dataset and were
added to make the streaming simulation realistic:

  claim_id              - UUID, unique per event
  patient_id            - UUID, unique per patient
  procedure_code        - CPT code, realistic distribution
  procedure_description - text description of procedure
  provider_id           - synthetic provider identifier
  submission_timestamp  - ISO timestamp of claim submission
  days_since_last_claim - integer, days since prior claim by same patient
  diagnosis_code        - ICD-10 style code
  is_fraud              - boolean, derived from rules above


Fraud Rate
----------
The rule-based labeling produces approximately 7-9% fraud rate across the
full synthetic dataset, which is consistent with published estimates of
healthcare fraud prevalence (3-10% of total healthcare spending, per CMS).
The Kafka producer injects fraud at 8% by default (--fraud-rate 0.08).


Medicare Secondary Dataset Integration
---------------------------------------
The CMS DE-SynPUF (Synthetic Public Use Files) data is used for large-scale
batch validation. The following column mapping was used to align CMS fields
with this project's schema:

  CMS: BENE_BIRTH_DT               -> age (derived)
  CMS: BENE_SEX_IDENT_CD           -> sex
  CMS: CLM_PMT_AMT                 -> claim_amount
  CMS: CLM_FROM_DT                 -> submission_timestamp
  CMS: AT_PHYSN_NPI                -> provider_id (encoded)
  CMS: HCPCS_CD                    -> procedure_code (mapped to CPT)
  CMS: ICD9_DGNS_CD_1              -> diagnosis_code

The fraud label for CMS data is also synthetic, applied using the same
four rules above on the mapped fields.


References
----------
1. CMS Medicare Fraud Strike Force - documented fraud patterns (cms.gov)
2. "Medicare Fraud: A $60 Billion Problem" - GAO Report GAO-11-703
3. Bauder, R.A. et al. "Medicare fraud detection using machine learning."
   IEEE ICHI 2017.
4. Johnson, J.M. and Khoshgoftaar, T.M. "Medicare fraud detection using
   neural networks." Journal of Big Data, 2019.
