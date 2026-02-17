# Forest Health Classification

This machine learning project classifies forest health using Copernicus Sentinel satellite data (2024-2025). The model analyzes multi-spectral satellite imagery to detect and classify forest conditions.

**🌐 See the live results:** [https://www.peer-schlieker.de/]

## Repository Contents
- Model training and evaluation code
- Data preprocessing utilities from Sentinel Hub API
- Visualization and analysis scripts
- Documentation and methodology

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
3. **Add your credentials to .env:**  
If you want to download data from sentinelhub. 
```
SENTINELHUB_CLIENT_ID=your-actual-client-id
SENTINELHUB_CLIENT_SECRET=your-actual-secret
```

## Data
Due to Copernicus API rate limits (1 year per month per account), the multi-year dataset is not included in this repository. Visit the live demo above to see the classification results and model performance.

**License Notice**

**Code License:**
The code in this project is available for educational and demonstration purposes only.
- ✅ View and study the code
- ✅ Run it locally for learning  
- ✅ Use it as inspiration for your own projects
- ❌ Copy or redistribute this code without permission
- ❌ Use it for commercial purposes

For questions about code usage, contact: github@peer-schlieker.de

## Data Attribution

This project uses modified Copernicus Sentinel data from 2020–2025.  
Original Copernicus Sentinel data are freely available for reproduction, distribution, public communication, and adaptation under EU law (Regulation (EU) No 377/2014) through the Copernicus Programme.

Source access: Copernicus Open Access Hub / Copernicus APIs.

This project also incorporates ESA WorldCover data.  
© ESA WorldCover Project 2021. Contains modified Copernicus Sentinel data (2021) processed by the European Space Agency WorldCover consortium.

---

## Digital Elevation Model

Elevation data are derived from the ASTER Global Digital Elevation Model (V003), provided by NASA and distributed through the NASA Land Processes Distributed Active Archive Center (LP DAAC).  
Dataset DOI: https://doi.org/10.5067/ASTER/ASTGTM.003

**Acknowledgments**
This project uses several open-source libraries. See environment.yml for the full list.
