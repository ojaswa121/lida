# 📊 Enhanced LIDA Visualization Assistant

A powerful Streamlit application that combines LIDA (Language Interface for Data Analysis) with local LLM capabilities to generate intelligent data visualizations and insights from CSV files.

## 🌟 Features

### 🔍 **Data Analysis & Insights**
- **Basic LIDA Summary**: Automated column analysis and data profiling
- **Detailed Insightful Summary**: Advanced statistical analysis with correlation insights
- **Data Quality Assessment**: Missing values, duplicates, and outlier detection
- **Smart Recommendations**: Actionable insights for data preprocessing

### 📊 **Visualization Capabilities**
- **Question-Based Visualization**: Natural language queries to generate charts
- **Chart Type Explorer**: Browse and generate specific chart types
- **Multiple Libraries**: Support for Seaborn, Matplotlib, and Plotly
- **Comprehensive Chart Support**: 20+ Seaborn charts and 30+ Matplotlib chart types

### 🧠 **AI-Powered Features**
- **Local LLM Integration**: Uses LM Studio for privacy-focused AI processing
- **Enhanced Query Processing**: Automatically optimizes visualization requests
- **Code Generation**: Generates Python code for every visualization
- **Interactive Examples**: Quick-start templates for common visualizations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- [LM Studio](https://lmstudio.ai/) running locally on port 1234

### Setup

1. **Clone or download the application**
```bash
# Save the code as lida_app.py
```

2. **Install required dependencies**
```bash
pip install streamlit lida pandas pillow requests python-dotenv openai
```

3. **Set up LM Studio**
   - Download and install LM Studio
   - Load a compatible language model (e.g., Llama 2, Code Llama, or similar)
   - Start the local server on `http://127.0.0.1:1234`

4. **Create environment file (optional)**
```bash
# Create .env file for additional configuration
touch .env
```

## 🚀 Usage

### Starting the Application
```bash
streamlit run lida_app.py
```

### Basic Workflow

1. **Upload CSV File**
   - Navigate to the file upload section
   - Select your CSV file (max 10MB)
   - Preview your data in the expandable sections

2. **Choose Analysis Type**
   - **Summarize**: Get comprehensive data insights and visualization goals
   - **Question-based Graph**: Ask natural language questions about your data
   - **Chart Type Explorer**: Browse and generate specific chart types

3. **Generate Visualizations**
   - Enter your visualization request
   - Select preferred library (Seaborn/Matplotlib/Plotly)
   - Download generated code and images

## 📊 Supported Chart Types

### Seaborn Charts (20+ types)
- `scatter` - Scatter plots for relationships
- `line` - Line plots for trends
- `bar` / `barh` - Bar charts (vertical/horizontal)
- `hist` - Histograms for distributions
- `box` / `violin` - Distribution summaries
- `heatmap` - Correlation matrices
- `pair` - Pairwise relationships
- `reg` - Regression plots
- `kde` - Kernel density estimates
- And many more...

### Matplotlib Charts (30+ types)
- `plot` - Basic line plots
- `scatter` - Scatter plots
- `bar` / `barh` - Bar charts
- `hist` - Histograms
- `pie` - Pie charts
- `contour` - Contour plots
- `polar` - Polar coordinates
- `errorbar` - Error bars
- `hexbin` - Hexagonal binning
- And many more...

## 💡 Example Queries

### Natural Language Questions
```
"Show a violin plot of sales by region"
"Create a heatmap of correlations between all numeric columns"
"Make a scatter plot of price vs quantity with regression line"
"Generate a box plot showing distribution of scores by category"
"Display a bar chart comparing revenue across different months"
```

### Chart Type Explorer
- Select any chart type from the dropdown
- Choose X and Y axes from your data columns
- Generate with one click

## 🔧 Configuration

### LM Studio Setup
```python
# The app automatically configures these settings:
OPENAI_API_KEY = "lm-studio"
OPENAI_BASE_URL = "http://127.0.0.1:1234/v1"
```

### Text Generation Settings
```python
textgen_config = TextGenerationConfig(
    n=1,
    temperature=0.1,        # Low temperature for consistent results
    max_tokens=2048,        # Maximum response length
    use_cache=True          # Enable caching for better performance
)
```

## 🏗️ Project Structure

```
lida_app.py
├── Configuration & Setup
├── Utility Functions
│   ├── Image processing
│   ├── File handling
│   ├── Data validation
│   └── Summary generation
├── UI Components
│   ├── Sidebar configuration
│   ├── File upload
│   └── Data preview
└── Main Features
    ├── Data Summarization
    ├── Question-based Visualization
    └── Chart Type Explorer
```

## 🔒 Privacy & Security

- **Local Processing**: All AI processing happens locally via LM Studio
- **No Data Upload**: Your data never leaves your machine
- **Temporary Files**: CSV files are processed in temporary storage and cleaned up
- **No API Keys**: No external API keys required

## 🐛 Troubleshooting

### Common Issues

1. **LM Studio Connection Failed**
   - Ensure LM Studio is running on port 1234
   - Check that a model is loaded in LM Studio
   - Verify firewall settings

2. **CSV Upload Issues**
   - Check file size (must be < 10MB)
   - Ensure proper CSV format
   - Verify no special characters in column names

3. **Visualization Generation Failed**
   - Try simpler, more specific queries
   - Check if your data has the required columns
   - Ensure sufficient data for the requested chart type

### Performance Tips
- Use smaller datasets for faster processing
- Enable caching in text generation config
- Choose appropriate chart types for your data size

## 🤝 Contributing

This is a self-contained application. To extend functionality:

1. Add new chart types to the `CHART_TYPES` dictionaries
2. Implement additional summary analysis in `generate_insightful_summary()`
3. Add support for more file formats beyond CSV
4. Integrate additional visualization libraries

## 📝 License

This project uses the following open-source libraries:
- Streamlit (Apache 2.0)
- LIDA (MIT)
- Pandas (BSD 3-Clause)
- Matplotlib (PSF)
- Seaborn (BSD 3-Clause)

## 🆘 Support

For issues related to:
- **LIDA**: Check the [LIDA GitHub repository](https://github.com/microsoft/lida)
- **LM Studio**: Visit [LM Studio documentation](https://lmstudio.ai/docs)
- **Streamlit**: See [Streamlit documentation](https://docs.streamlit.io/)

## 🎯 Roadmap

Future enhancements:
- [ ] Support for Excel files
- [ ] Real-time data streaming
- [ ] Custom chart templating
- [ ] Advanced statistical analysis
- [ ] Export to various formats (PDF, SVG, etc.)
- [ ] Collaborative features

---

**Built with ❤️ using Streamlit, LIDA, and LM Studio**