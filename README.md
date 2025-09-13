# Forest Health Classification

This machine learning project classifies forest health using Copernicus Sentinel satellite data (2023-2025). The model analyzes multi-spectral satellite imagery to detect and classify forest conditions.

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

**Data Attribution:**
This project's results are based on modified Copernicus Sentinel data 2023-2025. 
The original Copernicus Sentinel data is freely available under EU law (Regulation EU No 377/2014) 
for reproduction, distribution, communication to the public, and adaptation.

Original data source: Copernicus Open Access Hub / Copernicus API

**Acknowledgments**
This project uses several open-source libraries. See environment.yml for the full list.