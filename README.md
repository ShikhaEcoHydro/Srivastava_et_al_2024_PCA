# Srivastava_et_al_2024_PCA

Python implementation of Principal Component Analysis (PCA) applied to bulk-sediment geochemical datasets from two distinct sedimentary systems — the Shimla and Chail metasediments (SCM) of the Lesser Himalaya, Himachal Pradesh, and the mudflat sediments of Diu Island (DMS), southern Saurashtra, Gujarat. The analysis reproduces and validates the results published in Srivastava et al. (2024), Journal of Earth System Science, 133(3), 122. The code is validated against the original MATLAB outputs and produces scree plots, RQ-mode biplots, and loadings heatmaps identical to the published figures.


## Author
Deepshikha Srivastava  
Central University of Karnataka, Kadaganchi Campus, Gulbarga, Karnataka


## Requirements
See `requirements.txt` or install directly:


## Datasets
- **SCM** - Shimla & Chail Metasediments, Lesser Himalaya, Himachal Pradesh (Joshi et al. 2021)
- **DMS** - Diu Island Mudflat Sediments, southern Saurashtra, Gujarat (Banerji et al. 2021)


## Data Format
Each Excel file must have three sheets:
- **Data**- first column = sample ID, remaining columns = geochemical variables
- **Scores** - MATLAB-computed scores for validation
- **Loadings** - MATLAB-computed eigenvectors and variance for validation


## Usage
1. Place your Excel files in the `data/` folder
2. Update `DATA_DIR` and `OUTPUT_DIR` in `src/main_v2.py`
3. Run:python src/main\_v2.py


## Output
| File | Description |
|---|---|
| `Figure4_SCM_PCA.png` | Scree plot + biplots for SCM |
| `Figure5_DMS_PCA.png` | Scree plot + biplots for DMS |
| `FigureS1_Loadings_Heatmap.png` | Loadings heatmap for both datasets |
| `PCA_Results.xlsx` | All scores, loadings and variance tables |


## Key Results
- SCM: PC1=20.86%, PC2=19.75%, PC3=11.90% → cumulative 52.51%
- DMS: PC1=48.94%, PC2=15.64%, PC3=14.72% → cumulative 79.30%


## Reference
Srivastava, D., Dubey, C.P., Banerji, U.S., & Joshi, K.B. (2024). Geochemical trends in sedimentary environments using PCA approach. *Journal of Earth System Science*, 133(3), 122.  
DOI: [10.1007/s12040-024-02306-2](https://doi.org/10.1007/s12040-024-02306-2)

## Acknowledgements
The authors acknowledge the support received from the Director of the National Centre for Earth Science Studies (NCESS), Thiruvananthapuram.
