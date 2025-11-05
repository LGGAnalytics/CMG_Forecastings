# AI Agent Instructions for Analysis Project

## Project Overview
This is a data analysis project focused on statistical analysis and machine learning using Python. The project is structured as a Jupyter notebook-based workflow with data processing and analysis capabilities.

## Key Components
- `sth.ipynb`: Main Jupyter notebook containing analysis workflows
- `data/`: Directory containing data files for analysis
- `requirements.txt`: Project dependencies

## Development Environment
### Required Dependencies
The project relies on the following key packages (see `requirements.txt` for specific versions):
- `pandas` & `numpy`: Data manipulation and numerical operations
- `matplotlib` & `seaborn`: Data visualization
- `statsmodels` & `scikit-learn`: Statistical analysis and machine learning
- `boto3` & `pyathena`: AWS integration for data access

### Setup
1. Create a Python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Development Patterns
### Data Analysis Workflow
- Jupyter notebooks are the primary development environment
- Data files are stored in the `data/` directory
- AWS services (via boto3/pyathena) are used for data access

### Best Practices
1. Package Management:
   - Always update `requirements.txt` when adding new dependencies
   - Use exact versions in `requirements.txt` to ensure reproducibility

2. Notebook Organization:
   - Keep data loading and preprocessing steps at the beginning
   - Document analysis steps with markdown cells
   - Include visualizations to support findings

3. AWS Integration:
   - Use boto3 for AWS service interactions
   - PyAthena is used for querying AWS Athena

## Common Tasks
1. Running Analysis:
   - Open Jupyter notebooks in VS Code
   - Run cells sequentially from top to bottom
   - Ensure all dependencies are installed before running

2. Adding New Dependencies:
   ```powershell
   pip install <package>
   pip freeze > requirements.txt
   ```

## Important Notes
- The project uses specific versions of packages to maintain consistency
- AWS credentials must be properly configured for data access features
- Keep data files organized in the `data/` directory