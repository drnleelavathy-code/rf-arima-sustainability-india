
================================================================
DATASET README AND CODEBOOK
================================================================
Title   : Synthetic Consumer Behavior Dataset for Sustainable
          Product Adoption Analysis — India E-Commerce Context
Paper   : A Hybrid Machine Learning Framework for AI-Driven
          Sustainable Development (Technical Note)
Journal : MDPI
Authors : Ritesh Kalidindi (1)
          Leelavathy Narkedamilly (2)
          Uma Meghana S (3)
          (1) International School of Hyderabad
          (2) Godavari Global University, Rajamahendravaram
          (3) Oracle Health Care, WA, USA
Date    : 21 February 2026
Seed    : 42
Python  : 3.9+  |  numpy>=1.21  |  pandas>=1.5
================================================================

IMPORTANT DISCLOSURE
--------------------
This dataset is SYNTHETICALLY GENERATED using domain-calibrated
weights derived from prior empirical literature:
  [1] Deloitte India (2023) — Reverse Logistics Survey
  [2] CSE (2024)            — Environmental Preference Data
  [3] Joshi & Rahman (2015) — Green Purchase Behaviour Study

The adoption_score target variable is constructed via Equation
(1) of the paper (weighted linear combination + Gaussian noise).
RF feature importance scores recover the generative weights by
design — this is a validation approach, not novel discovery.
This is NOT a field-collected transaction dataset.

DATASET OVERVIEW
----------------
Records        : 10,512
Columns        : 21 (including record_id index)
File format    : CSV, UTF-8, comma-delimited
Missing values : ~3% in 5 columns (seed=43, realistic imputation)

COLUMN CODEBOOK
---------------
record_id                   Row index (0-based)

DEMOGRAPHIC — INTEGER CODES (for ML models)
age_group_code              0=18-25, 1=26-35, 2=36-45,
                            3=46-55, 4=55+
city_tier_code              0=Tier-1(Metro), 1=Tier-2,
                            2=Tier-3(Rural)
income_q_code               0=Top 20%, 1=60-80%, 2=40-60%,
                            3=20-40%, 4=Bottom 20%
edu_level_code              0=Below Secondary, 1=Secondary,
                            2=Graduate, 3=Postgraduate

DEMOGRAPHIC — STRING LABELS (human-readable)
age_group_label             e.g. '18-25', '26-35' ...
city_tier_label             'Tier-1', 'Tier-2', 'Tier-3'
income_q_label              'Top_20pct' ... 'Bottom_20pct'
edu_level_label             'Graduate', 'Secondary' ...

HOUSEHOLD & BEHAVIOURAL
household_size              Integer 1-7
purchase_freq_per_month     Exponential(scale=5), clipped [1,30]
transaction_value_inr       Lognormal(mean=6.5, sigma=0.9) INR
digital_literacy_score      Beta distribution [0,1]

ATTITUDINAL — scaled [0,1], ~3% missing
consumer_awareness_index    Primary driver (Eq.1 weight = 0.34)
price_sensitivity           Higher = more sensitive (wt = 0.22)
product_availability        Supply-side score (wt = 0.28)

DERIVED SCALED FEATURES — [0,1], no missing
age_scaled                  1 - age_group_code / 4
tier_scaled                 1 - city_tier_code / 2
income_scaled               1 - income_q_code  / 4
perceived_quality           Normal(mu=0.5, sigma=0.12)

TARGET VARIABLES
adoption_score              Continuous [0,1] — Equation (1)
adoption_binary             1 if adoption_score > 0.5 else 0

SAMPLE PROPORTIONS (designed)
City tier  : Tier-1=42%, Tier-2=33%, Tier-3=25%
Age groups : 18-25=22.3%, 26-35=29.7%, 36-45=27.2%,
             46-55=15.0%, 55+=5.8%
Income     : Top20=20.0%, 60-80=22.0%, 40-60=25.5%,
             20-40=21.3%, Bottom20=11.2%

EQUATION 1 — GENERATIVE WEIGHTS
consumer_awareness_index    0.34
product_availability        0.28
price_competitiveness (inv) 0.22
age_scaled                  0.08
tier_scaled                 0.05
income_scaled               0.02
perceived_quality           0.01
noise epsilon               N(0, 0.05^2)

REPRODUCIBILITY
---------------
Run dataset_generation_script.py with Python 3.9+ and
numpy seed=42 to regenerate the exact same dataset.

LICENSE
-------
Creative Commons Attribution 4.0 International (CC BY 4.0)
Free to use, share, and adapt with attribution to the paper.

CITATION
--------
Kalidindi R.; Narkedamilly L.; Uma Meghana S. Synthetic
Consumer Behavior Dataset — Hybrid RF-ARIMA Sustainable
Development Study. 2026.
DOI: [to be assigned on Zenodo upload]
================================================================
