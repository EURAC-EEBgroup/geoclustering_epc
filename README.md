# GeoClustering EPC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![EURAC Research](https://img.shields.io/badge/EURAC-Research-green.svg)](https://www.eurac.edu/)

> **Geographic Clustering and Analysis of Energy Performance Certificates (EPC) for Building Stock Assessment**

A comprehensive Python framework for geographic clustering and energy performance analysis of building stocks using Energy Performance Certificate (EPC) data. This tool enables researchers and practitioners to identify building clusters based on geographic proximity and energy characteristics, supporting data-driven decision making for large-scale energy efficiency interventions.

## 🏢 Overview

The GeoClustering EPC tool provides advanced analytics for building energy performance data by combining:

- **Geographic Information Systems (GIS)** analysis
- **Machine Learning clustering** algorithms  
- **Energy Performance Certificate (EPC)** data processing
- **Statistical analysis** and visualization
- **Building stock characterization** and segmentation

### Key Features

- 🗺️ **Geographic Clustering**: Spatial analysis using coordinates and administrative boundaries
- 📊 **Energy Performance Analysis**: EPC data processing and energy indicator calculations
- 🤖 **Machine Learning**: K-means, DBSCAN, and hierarchical clustering algorithms
- 📈 **Advanced Visualization**: Interactive maps, plots, and energy performance dashboards
- 🔬 **Statistical Analysis**: Comprehensive statistical characterization of building clusters
- 📋 **Automated Reporting**: HTML reports with cluster analysis and energy insights

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/EURAC-EEBgroup/geoclustering_epc.git
cd geoclustering_epc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import pandas as pd
from geoclustering_epc import GeoClusteringAnalyzer

# Load your EPC dataset
data = pd.read_csv('your_epc_data.csv')

# Initialize the analyzer
analyzer = GeoClusteringAnalyzer(
    data=data,
    lat_col='latitude',
    lon_col='longitude', 
    energy_cols=['primary_energy', 'heating_demand']
)

# Perform geographic clustering
clusters = analyzer.fit_geographic_clusters(
    n_clusters=5,
    method='kmeans'
)

# Analyze energy performance by cluster
energy_analysis = analyzer.analyze_energy_performance()

# Generate comprehensive report
analyzer.generate_report(output_path='cluster_analysis_report.html')
```

## 📁 Project Structure

```
geoclustering_epc/
├── src/
│   ├── geoclustering_epc/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── clustering.py          # Clustering algorithms
│   │   │   ├── geographic.py          # Geographic analysis
│   │   │   └── energy_analysis.py     # Energy performance analysis
│   │   ├── data/
│   │   │   ├── preprocessing.py       # Data cleaning and preparation
│   │   │   └── validation.py          # Data quality checks
│   │   ├── visualization/
│   │   │   ├── maps.py               # Interactive mapping
│   │   │   ├── plots.py              # Statistical plots
│   │   │   └── reports.py            # HTML report generation
│   │   └── utils/
│   │       ├── config.py             # Configuration management
│   │       └── helpers.py            # Utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Exploratory data analysis
│   ├── 02_geographic_clustering.ipynb # Geographic clustering examples
│   ├── 03_energy_analysis.ipynb     # Energy performance analysis
│   └── 04_case_studies.ipynb        # Real-world case studies
├── data/
│   ├── sample/                       # Sample datasets
│   └── processed/                    # Processed data outputs
├── tests/
│   ├── test_clustering.py
│   ├── test_geographic.py
│   └── test_energy_analysis.py
├── docs/
│   ├── methodology.md                # Technical methodology
│   ├── api_reference.md              # API documentation
│   └── examples/                     # Usage examples
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## 🔧 Core Components

### Geographic Clustering

```python
from geoclustering_epc.core.clustering import GeographicClustering

# Initialize geographic clustering
geo_cluster = GeographicClustering()

# Cluster buildings by location
clusters = geo_cluster.cluster_by_coordinates(
    data=buildings_df,
    lat_col='latitude',
    lon_col='longitude',
    method='kmeans',
    n_clusters=8
)

# Cluster by administrative boundaries
admin_clusters = geo_cluster.cluster_by_admin_boundaries(
    data=buildings_df,
    admin_level='municipality'
)
```

### Energy Performance Analysis

```python
from geoclustering_epc.core.energy_analysis import EnergyPerformanceAnalyzer

# Initialize energy analyzer
energy_analyzer = EnergyPerformanceAnalyzer()

# Calculate energy indicators
indicators = energy_analyzer.calculate_energy_indicators(
    data=clustered_buildings,
    primary_energy_col='ep_h_nd',
    floor_area_col='useful_floor_area'
)

# Perform statistical analysis by cluster
cluster_stats = energy_analyzer.analyze_by_cluster(
    data=clustered_buildings,
    cluster_col='cluster_id'
)
```

### Visualization and Mapping

```python
from geoclustering_epc.visualization.maps import InteractiveMap
from geoclustering_epc.visualization.plots import EnergyPlots

# Create interactive map
map_viz = InteractiveMap()
cluster_map = map_viz.plot_clusters(
    data=clustered_buildings,
    lat_col='latitude',
    lon_col='longitude',
    cluster_col='cluster_id',
    color_by='energy_class'
)

# Generate energy performance plots
energy_plots = EnergyPlots()
energy_plots.plot_cluster_comparison(
    data=clustered_buildings,
    energy_col='primary_energy_demand',
    cluster_col='cluster_id'
)
```

## 📊 Data Requirements

### Input Data Format

Your EPC dataset should include the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `latitude` | Building latitude coordinates | ✅ |
| `longitude` | Building longitude coordinates | ✅ |
| `primary_energy` | Primary energy demand [kWh/m²/year] | ✅ |
| `heating_demand` | Heating energy demand [kWh/m²/year] | ✅ |
| `floor_area` | Useful floor area [m²] | ✅ |
| `building_type` | Building type category | ⭕ |
| `construction_year` | Year of construction | ⭕ |
| `energy_class` | Energy performance class (A-G) | ⭕ |
| `municipality` | Administrative municipality | ⭕ |

### Sample Data

```python
# Example data structure
import pandas as pd

sample_data = pd.DataFrame({
    'latitude': [46.4982, 46.5018, 46.4951],
    'longitude': [11.3548, 11.3421, 11.3667],
    'primary_energy': [120.5, 95.2, 180.7],
    'heating_demand': [85.3, 62.1, 142.8],
    'floor_area': [150, 200, 120],
    'building_type': ['Residential', 'Office', 'Residential'],
    'energy_class': ['C', 'B', 'D']
})
```

## 🛠️ Advanced Features

### Custom Clustering Algorithms

```python
# Use custom distance metrics for geographic clustering
from geoclustering_epc.core.clustering import CustomGeoClustering

custom_cluster = CustomGeoClustering()
clusters = custom_cluster.cluster_with_constraints(
    data=buildings_df,
    max_distance_km=2.0,
    min_cluster_size=50,
    energy_weight=0.3,
    geographic_weight=0.7
)
```

### Multi-criteria Analysis

```python
# Combine geographic and energy performance criteria
from geoclustering_epc.core.analysis import MultiCriteriaAnalysis

mca = MultiCriteriaAnalysis()
optimal_clusters = mca.optimize_clustering(
    data=buildings_df,
    criteria=['geographic_compactness', 'energy_homogeneity', 'building_type_similarity'],
    weights=[0.4, 0.4, 0.2]
)
```

### Automated Report Generation

```python
# Generate comprehensive analysis report
from geoclustering_epc.visualization.reports import AnalysisReport

report = AnalysisReport()
report.generate_full_report(
    data=clustered_buildings,
    output_path='energy_cluster_analysis.html',
    include_maps=True,
    include_statistics=True,
    include_recommendations=True
)
```

## 📈 Methodology

The GeoClustering EPC methodology combines several analytical approaches:

1. **Data Preprocessing**: Quality checks, outlier detection, and standardization
2. **Geographic Analysis**: Spatial proximity analysis using haversine distance
3. **Energy Characterization**: Statistical analysis of energy performance indicators
4. **Clustering Algorithms**: K-means, DBSCAN, hierarchical clustering
5. **Validation**: Silhouette analysis, cluster stability assessment
6. **Visualization**: Interactive maps and statistical plots

### Clustering Methods

- **Geographic K-means**: Distance-based clustering using coordinates
- **Administrative Clustering**: Grouping by administrative boundaries
- **Energy-informed Clustering**: Combined geographic and energy performance criteria
- **Density-based Clustering**: DBSCAN for irregular cluster shapes

## 🔬 Scientific Applications

This tool supports research in:

- **Urban Energy Planning**: Large-scale building stock assessment
- **Policy Development**: Evidence-based energy efficiency policies
- **Renovation Strategies**: Targeted building renovation programs
- **Energy Modeling**: District-level energy system modeling
- **Sustainability Assessment**: Environmental impact evaluation

## 📚 Examples and Tutorials

Explore the `notebooks/` directory for comprehensive examples:

- **Data Exploration**: Understanding your EPC dataset
- **Geographic Clustering**: Step-by-step clustering analysis
- **Energy Performance Analysis**: Statistical energy characterization
- **Case Studies**: Real-world applications

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏛️ Institution

**EURAC Research - Institute for Renewable Energy**
- **Website**: https://www.eurac.edu/en/institutes-centers/institute-for-renewable-energy
- **Address**: Viale Druso 1, 39100 Bolzano, Italy

### Authors

- **Daniele Antonucci** - Lead Developer - [daniele.antonucci@eurac.edu](mailto:daniele.antonucci@eurac.edu)
- **Research Team** - Energy Efficient Buildings Group

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/EURAC-EEBgroup/geoclustering_epc/issues)
- **Documentation**: [API Reference](docs/api_reference.md)
- **Email**: [daniele.antonucci@eurac.edu](mailto:daniele.antonucci@eurac.edu)

## 🙏 Acknowledgments

This research is supported by:
- European Union's Horizon 2020 research and innovation programme
---

**Keywords**: Energy Performance Certificate, Geographic Clustering, Building Stock Analysis, Energy Efficiency, GIS, Machine Learning, Urban Energy Planning