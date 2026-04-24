# Forest Health Analysis
This project analyses forest health using Copernicus Sentinel-2 satellite data (2020–2025). Rather than a supervised classification, it uses unsupervised clustering on remote-sensing vegetation indices to detect and map forest degradation — without labeled ground truth.

The area of interest (AOI) covers a section of the Black Forest National Park, south-east of the Hornisgrinde, chosen for its documented degradation patterns and deliberate non-intervention management policy.

**🌐 Full write-up and results:** [https://www.peer-schlieker.de/]

---

## What This Project Does

1. **Fetches multi-spectral Sentinel-2 data** via the SentinelHub API across multiple timestamps, with temporal interpolation to handle cloud cover
2. **Masks forest pixels** using ESA WorldCover land cover data
3. **Computes vegetation indices** (NDVI, NDWI, SAVI, EVI, NDRE variants, NBR) and derives temporal and spatial features from them
4. **Clusters forest pixels** using k-Means (k=4) into degradation states ranging from severely degraded to healthy
5. **Analyses topographic relationships** between cluster labels and DEM-derived features (elevation, slope, aspect, TWI, TPI, etc.)

The clustering results were externally validated against [deadtrees.earth](https://deadtrees.earth), a University of Freiburg project that maps standing dead trees using supervised ML on aerial imagery — two independent methods, consistent conclusions.

---

## Repository Contents

```
notebooks/          # Exploratory analysis and methodology notebooks
  band_eda.ipynb    # Band and feature analysis
  clustering.ipynb  # k-Means clustering methodology
  dem_analysis.ipynb# Topographic feature analysis
src/                # Pipeline source code
  api/              # SentinelHub data fetching and tiling logic
  features/         # Feature calculators (Template Method Pattern)
  clustering/       # Clustering and evaluation utilities
  dem/              # DEM processing and topographic feature extraction
```

---

## Environment Setup

1. **Create poetry environment:**
```bash
poetry install
poetry env activate
```

2. **Copy the template:**
```bash
cp template.env .env
```

3. **Add your credentials to `.env`:**  
Required only if you want to download data from SentinelHub.
```
SENTINELHUB_CLIENT_ID=your-actual-client-id
SENTINELHUB_CLIENT_SECRET=your-actual-secret
```

---

## Data

Due to Copernicus API rate limits, the multi-year dataset is not included in this repository. The dataset covers January 2020 to September 2025 at monthly resolution for a focused AOI in the Black Forest National Park. Visit the write-up above to see the results and methodology in detail.

---

## License Notice

**Code License:**  
The code in this project is available for educational and demonstration purposes only.

- ✅ View and study the code
- ✅ Run it locally for learning
- ✅ Use it as inspiration for your own projects
- ❌ Copy or redistribute this code without permission
- ❌ Use it for commercial purposes

For questions about code usage, contact: github@peer-schlieker.de

---

## Data Attribution

**Sentinel-2 Data:**  
This project uses modified Copernicus Sentinel data from 2020–2025. Original Copernicus Sentinel data are freely available for reproduction, distribution, public communication, and adaptation under EU law (Regulation (EU) No 377/2014) through the Copernicus Programme.  
Source: Copernicus Open Access Hub / Copernicus APIs.

**Land Cover Mask:**  
This project incorporates ESA WorldCover data for forest pixel masking.  
© ESA WorldCover Project 2021. Contains modified Copernicus Sentinel data (2021) processed by the European Space Agency WorldCover consortium.

**Digital Elevation Model:**  
Elevation data are derived from the ASTER Global Digital Elevation Model (V003), provided by NASA and distributed through the NASA Land Processes Distributed Active Archive Center (LP DAAC).  
Dataset DOI: https://doi.org/10.5067/ASTER/ASTGTM.003

**External Validation:**  
Cluster results were compared against [deadtrees.earth](https://deadtrees.earth), a University of Freiburg project mapping standing dead trees via supervised ML on aerial imagery.

---

## Acknowledgments

This project uses several open-source libraries. See `environment.yml` for the full list.
