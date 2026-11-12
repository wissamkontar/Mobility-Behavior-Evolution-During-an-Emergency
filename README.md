# Mobility Behavior Evolution During an Emergency

Code for the analysis in:

> **Mobility Behavior Evolution During an Emergency**  
> Omid Armantalab, Jason Hawkins, and Wissam Kontar  
> Department of Civil & Environmental Engineering, University of Nebraska–Lincoln  
> Department of Civil Engineering, University of Calgary

This repository studies how people shift between two mobility behavior types during a 15-day emergency period in a densely populated Japanese metropolitan area, using the open **YJMob100K** dataset. **Returners** repeatedly visit a small set of familiar locations, while **explorers** travel across a broader range of destinations. The analysis compares a 15-day normal period with the 15-day emergency period, tracks how behavior changes from day to day, separates weekdays from weekends and holidays, and examines mobility inside and outside each person's nearby neighborhood.

---

## Method

### 1. Stops

Each YJMob100K record gives a user's 500 m × 500 m grid cell in a 30-minute time slot. Consecutive records of a user in the same cell are merged into a single **stop**, and the time spent at that location is summed.

### 2. Radius of gyration

The radius of gyration measures how far a user typically travels from the center of their movements:

```text
r_g = sqrt( (1/T) * sum_i t_i * |r_i - r_cm|^2 )
```

where `t_i` is the time spent at location `i`, `T` is the total time, and `r_cm` is the time-weighted center of mass. The **k-radius of gyration** `r_g^(k)` applies the same formula to the `k` locations where the user spent the most time.

### 3. Returners and explorers

```text
S_k = r_g^(k) / r_g

S_k > 0.5   ->  k-returner
S_k <= 0.5  ->  k-explorer
```

Users are classified with `k = 4` in both periods.

### 4. Mobility indicators

Returners and explorers are compared on:

- maximum distance from home;
- non-home dwelling time;
- real entropy of their movement sequences;
- spatial distribution of stops and stay time; and
- time and distance traveled outside their nearby neighborhood.

### 5. Behavioral transitions

Users classified in both periods fall into four groups: returner to returner, returner to explorer, explorer to explorer, and explorer to returner. The normal to emergency transitions are compared against transitions between normal periods to separate the effect of the emergency from ordinary changes.

---

## Repository Files

| File | Purpose |
| --- | --- |
| `preprocess_yjmob100k.py` | Cleans the YJMob100K data, infers home locations, and builds the stop tables for the normal and emergency periods. |
| `mobility_metrics.py` | Study settings and shared functions used by all scripts. |
| `period_analysis.py` | Analysis of a single period; run once for the normal period and once for the emergency period. |
| `observation_window_analysis.py` | Determines how many days of data are needed to distinguish emergency from normal behavior. |
| `transition_analysis.py` | Behavioral transitions between periods, the four transition groups, and the points of interest (POI) comparison. |
| `requirements.txt` | Python packages required to run the code. |

---

## Data

The study uses **Dataset 2** of YJMob100K, which contains the movements of 25,000 anonymized individuals over 75 days. Days 0 to 59 represent normal behavior and days 60 to 74 an emergency period designated by the dataset creators.

The data are available on Zenodo:

<https://zenodo.org/records/10836269>

Download the following files and place them in `data/raw/`:

```text
data/
└── raw/
    ├── yjmob100k-dataset2.csv.gz
    └── cell_POIcat.csv.gz
```

The data are **not included in this repository**. Please cite the data descriptor when using the data:

> Yabe, T., Tsubouchi, K., Shimizu, T., Sekimoto, Y., Sezaki, K., Moro, E., and Pentland, A.  
> *YJMob100K: City-scale and longitudinal dataset of anonymized human mobility trajectories.*  
> Scientific Data, 11, 397 (2024).  
> DOI: 10.1038/s41597-024-03237-9

---

## Installation

Python 3.10 or newer is recommended. Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Preprocessing

```bash
python preprocess_yjmob100k.py
```

This step removes day 27 as advised by the dataset creators, marks weekends and holidays, infers each user's home location from nighttime records during the normal days, and builds the stop tables for the normal period (days 43 to 57) and the emergency period (days 60 to 74).

### 2. Single-period analysis

```bash
python period_analysis.py --period normal
python period_analysis.py --period emergency
```

For each period, this step fits distributions to the radius of gyration, classifies users as returners or explorers, compares the two groups, and analyzes their spatial patterns and mobility outside nearby neighborhoods.

### 3. Observation window analysis

```bash
python observation_window_analysis.py
```

This step classifies users over observation windows of increasing length to find the minimum number of days needed to distinguish emergency from normal behavior, and checks the result on other normal segments and longer windows.

### 4. Transition analysis

```bash
python transition_analysis.py
```

This step computes the transitions between returner and explorer behavior from the normal to the emergency period, compares them with transitions between normal periods, tests their sensitivity to the classification threshold, and compares POI exposure across the four transition groups.

All results are saved as CSV tables in the `results/` folder.

---

## Typical Workflow

```text
1. Download YJMob100K data
            |
            v
2. preprocess_yjmob100k.py
            |
            +------------------+------------------+
            |                  |                  |
            v                  v                  v
3. period_analysis.py   4. observation_       5. transition_
   (normal, emergency)     window_analysis.py    analysis.py
```

In command-line form:

```bash
python preprocess_yjmob100k.py
python period_analysis.py --period normal
python period_analysis.py --period emergency
python observation_window_analysis.py
python transition_analysis.py
```

All study settings, such as the analysis windows, the value of `k`, and the classification threshold, are defined at the top of `mobility_metrics.py`. A fixed random seed is used throughout so results are reproducible.

---

## Contact

For questions about the analysis or repository, please contact the corresponding author through the contact information provided in the manuscript.
