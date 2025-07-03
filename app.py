
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from lida import Manager, TextGenerationConfig, llm
import requests
from PIL import Image
from io import BytesIO
import base64
import tempfile
import uuid
import pandas as pd
import llmx.generators.text.openai_textgen
from openai import OpenAI
import openai

# === Configure page ===
st.set_page_config(
    page_title="LIDA Visualization Assistant",
    page_icon="📊",
    layout="wide"
)

# === Configure OpenAI client to use LM Studio's local server ===
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1234/v1"

try:
    llmx.generators.text.openai_textgen.OpenAITextGenerator.client = OpenAI(
        api_key="lm-studio",
        base_url="http://127.0.0.1:1234/v1"
    )
    openai.api_key = "lm-studio"
    openai.api_base = "http://127.0.0.1:1234/v1"
except Exception as e:
    st.error(f"Failed to configure OpenAI client: {e}")
    st.stop()

@st.cache_data(ttl=60)
def check_lm_studio_connection():
    try:
        resp = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
        if resp.status_code == 200:
            models_data = resp.json()
            return True, [m['id'] for m in models_data.get('data', [])]
        return False, []
    except Exception as e:
        return False, str(e)

connection_status, models_or_error = check_lm_studio_connection()
if connection_status:
    st.sidebar.success("LM Studio is connected ✅")
    selected_model = st.sidebar.selectbox("Select Model", models_or_error)
else:
    st.sidebar.error(f"LM Studio is unreachable: {models_or_error}")
    st.stop()

# === Initialize LIDA ===
try:
    lida = Manager(text_gen=llm("openai"))
except Exception as e:
    st.error(f"Failed to initialize LIDA: {e}")
    st.stop()

# === Text generation config ===
textgen_config = TextGenerationConfig(
    n=1,
    model=selected_model,
    use_cache=True,
    temperature=0.1,
    max_tokens=2048
)

# === Chart Type Definitions ===
SEABORN_CHART_TYPES = {
    "scatter": "Scatter plot for relationships between continuous variables",
    "line": "Line plot for trends over time or ordered data",
    "bar": "Bar plot for categorical data comparison",
    "barh": "Horizontal bar plot",
    "hist": "Histogram for distribution of continuous data",
    "box": "Box plot for distribution summary with outliers",
    "violin": "Violin plot for distribution shape",
    "strip": "Strip plot for categorical scatter",
    "swarm": "Swarm plot for non-overlapping categorical scatter",
    "point": "Point plot for categorical data with error bars",
    "count": "Count plot for frequency of categorical data",
    "reg": "Regression plot with fitted line",
    "resid": "Residual plot for regression diagnostics",
    "kde": "Kernel density estimate plot",
    "joint": "Joint plot combining scatter and marginal distributions",
    "pair": "Pairwise relationships in dataset",
    "heatmap": "Heatmap for correlation or 2D data",
    "clustermap": "Clustered heatmap with dendrograms",
    "facet": "Facet grid for multiple subplots",
    "cat": "Categorical plot (combines multiple categorical plots)"
}

MATPLOTLIB_CHART_TYPES = {
    "plot": "Basic line plot",
    "scatter": "Scatter plot",
    "bar": "Vertical bar chart",
    "barh": "Horizontal bar chart", 
    "hist": "Histogram",
    "box": "Box plot",
    "area": "Area plot",
    "pie": "Pie chart",
    "hexbin": "Hexagonal binning plot",
    "stem": "Stem plot",
    "step": "Step plot",
    "fill_between": "Fill area between curves",
    "stackplot": "Stacked area plot",
    "polar": "Polar plot",
    "loglog": "Log-log scale plot",
    "semilogx": "Semi-log x-axis plot",
    "semilogy": "Semi-log y-axis plot",
    "errorbar": "Error bar plot",
    "contour": "Contour plot",
    "contourf": "Filled contour plot",
    "imshow": "Image display plot",
    "matshow": "Matrix display plot",
    "pcolor": "Pseudocolor plot",
    "pcolormesh": "Pseudocolor plot with mesh",
    "quiver": "Arrow/vector field plot",
    "streamplot": "Streamline plot",
    "specgram": "Spectrogram plot",
    "spy": "Spy plot for sparse matrices",
    "tricontour": "Triangular grid contour",
    "tricontourf": "Filled triangular grid contour",
    "tripcolor": "Triangular grid pseudocolor",
    "triplot": "Triangular grid plot"
}

# === Utility Functions ===
def base64_to_image(base64_string):
    try:
        byte_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(byte_data))
    except Exception as e:
        st.error(f"Failed to decode image: {e}")
        return None

def save_uploaded_file(uploaded_file):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.error(f"Failed to save uploaded file: {e}")
        return None

def validate_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            st.warning("The uploaded CSV file is empty.")
            return False
        return True
    except Exception as e:
        st.error(f"Invalid CSV file: {e}")
        return False

def cleanup_temp_file(file_path):
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        st.warning(f"Could not delete temp file: {e}")

def summarize_to_text(summary_json):
    """Convert LIDA JSON summary to a clean text summary."""
    if not isinstance(summary_json, dict) or "fields" not in summary_json:
        return "⚠️ The summary is not in the expected format."

    lines = []
    lines.append(f"### 📁 Dataset Overview")
    lines.append(f"- **File Name:** `{summary_json.get('file_name', 'N/A')}`")
    lines.append(f"- **Number of Columns:** {len(summary_json['fields'])}")

    lines.append("\n### 📊 Column Insights:")
    for field in summary_json["fields"]:
        col = field.get("column", "Unknown")
        props = field.get("properties", {})
        dtype = props.get("dtype", "N/A")
        num_uniques = props.get("num_unique_values", "N/A")
        samples = props.get("samples", [])[:3]
        description = props.get("description", "")

        lines.append(f"- **{col}** ({dtype}) — {num_uniques} unique values")
        if samples:
            lines.append(f"  - Sample values: {', '.join(map(str, samples))}")
        if description:
            lines.append(f"  - Description: {description}")

    return "\n".join(lines)

def generate_insightful_summary(df: pd.DataFrame, file_name: str = "Uploaded Dataset") -> str:
    """Generate a detailed natural-language summary from any CSV dataframe."""
    lines = [f"## 📊 Insightful Data Summary for `{file_name}`\n"]
    
    # General info
    lines.append(f"- **Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Missing values analysis
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        lines.append("- **Data Quality:** ✅ No missing values detected")
    else:
        missing_percent = (missing_count / (df.shape[0] * df.shape[1])) * 100
        lines.append(f"- **Data Quality:** ⚠️ {missing_count} missing values ({missing_percent:.1f}% of total data)")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        lines.append(f"- **Duplicates:** {duplicates} duplicate rows found")
    else:
        lines.append("- **Duplicates:** ✅ No duplicate rows")
    
    # Data type distribution
    dtype_counts = df.dtypes.value_counts()
    lines.append(f"\n### 📋 Data Types Distribution:")
    for dtype, count in dtype_counts.items():
        lines.append(f"- **{dtype}:** {count} columns")
    
    # Column-level insights
    lines.append("\n### 🔍 Detailed Column Analysis:")
    
    for col in df.columns:
        dtype = df[col].dtype
        unique_vals = df[col].nunique()
        missing_vals = df[col].isnull().sum()
        
        lines.append(f"\n#### ➤ `{col}`")
        lines.append(f"- **Type:** {dtype} | **Unique Values:** {unique_vals} | **Missing:** {missing_vals}")
        
        if dtype == "object" or dtype.name == "category":
            # Categorical analysis
            if unique_vals > 0:
                value_counts = df[col].value_counts()
                top_val = value_counts.index[0]
                freq = value_counts.iloc[0]
                freq_percent = (freq / len(df)) * 100
                lines.append(f"- **Most Frequent:** '{top_val}' ({freq} times, {freq_percent:.1f}%)")
                
                if unique_vals <= 10:
                    lines.append(f"- **All Values:** {list(df[col].unique())}")
                else:
                    lines.append(f"- **Sample Values:** {list(df[col].unique())[:5]}... (showing first 5)")
                    
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Numerical analysis
            if not df[col].isnull().all():
                desc = df[col].describe()
                lines.append(f"- **Range:** {desc['min']:.2f} to {desc['max']:.2f}")
                lines.append(f"- **Central Tendency:** Mean = {desc['mean']:.2f}, Median = {desc['50%']:.2f}")
                lines.append(f"- **Spread:** Std = {desc['std']:.2f}")
                
                # Check for potential outliers using IQR method
                Q1 = desc['25%']
                Q3 = desc['75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
                if outliers > 0:
                    lines.append(f"- **Potential Outliers:** {outliers} values outside IQR bounds")
                    
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # DateTime analysis
            if not df[col].isnull().all():
                min_date = df[col].min()
                max_date = df[col].max()
                date_range = max_date - min_date
                lines.append(f"- **Date Range:** {min_date} to {max_date}")
                lines.append(f"- **Time Span:** {date_range.days} days")
    
    # Correlation insights for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        lines.append(f"\n### 🔗 Correlation Insights:")
        corr_matrix = df[numeric_cols].corr()
        
        # Find strongest correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    correlations.append((abs(corr_val), corr_val, col1, col2))
        
        correlations.sort(reverse=True)
        if correlations:
            lines.append("- **Strongest Correlations:**")
            for abs_corr, corr, col1, col2 in correlations[:3]:
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs_corr > 0.7 else "moderate" if abs_corr > 0.3 else "weak"
                lines.append(f"  - `{col1}` ↔ `{col2}`: {corr:.3f} ({strength} {direction})")
    
    # Data insights and recommendations
    lines.append(f"\n### 💡 Key Insights & Recommendations:")
    
    if missing_count > 0:
        lines.append("- Consider handling missing values through imputation or removal")
    
    if duplicates > 0:
        lines.append("- Review and potentially remove duplicate records")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_cardinality:
            lines.append(f"- High cardinality categorical columns detected: {high_cardinality}")
    
    if len(numeric_cols) >= 2:
        lines.append("- Dataset suitable for correlation analysis and regression modeling")
    
    if any(pd.api.types.is_datetime64_any_dtype(df[col]) for col in df.columns):
        lines.append("- Time-series analysis possible with datetime columns")
    
    return "\n".join(lines)

def enhance_query_with_chart_type(query: str, library: str) -> str:
    """Enhance user query with specific chart type instructions."""
    query_lower = query.lower()
    
    # Define chart type mappings
    chart_types = SEABORN_CHART_TYPES if library == "seaborn" else MATPLOTLIB_CHART_TYPES
    
    # Check for explicit chart type mentions
    detected_charts = []
    for chart_type in chart_types.keys():
        if chart_type in query_lower:
            detected_charts.append(chart_type)
    
    if detected_charts:
        chart_type = detected_charts[0]  # Use first detected
        if library == "seaborn":
            query += f" Use seaborn's {chart_type} plot (sns.{chart_type}plot() or appropriate seaborn function)."
        else:
            query += f" Use matplotlib's {chart_type} plot (plt.{chart_type}() or appropriate matplotlib function)."
    else:
        # Add general instruction for comprehensive chart support
        if library == "seaborn":
            query += " Use the most appropriate seaborn visualization function for this data and question."
        else:
            query += " Use the most appropriate matplotlib visualization function for this data and question."
    
    return query

# === UI ===
st.title("📊 Enhanced LIDA Visualization Assistant")
st.markdown("*Support for all Matplotlib & Seaborn chart types with advanced data insights*")

st.sidebar.header("🛠️ Configuration")
visualization_library = st.sidebar.selectbox(
    "Visualization Library", 
    ["seaborn", "matplotlib", "plotly"],
    help="Choose your preferred visualization library"
)

# Chart type selector
if visualization_library in ["seaborn", "matplotlib"]:
    chart_types = SEABORN_CHART_TYPES if visualization_library == "seaborn" else MATPLOTLIB_CHART_TYPES
    
    with st.sidebar.expander(f"📊 Available {visualization_library.title()} Charts"):
        for chart_type, description in chart_types.items():
            st.write(f"**{chart_type}:** {description}")

menu = st.sidebar.selectbox("Choose an option", ["Summarize", "Question based graph", "Chart Type Explorer"])

st.header("📁 Upload Your Data")
file_uploader = st.file_uploader("Upload a CSV file", type="csv")
if not file_uploader:
    st.info("👆 Please upload a CSV file to begin.")
    st.stop()

# Validate file
if file_uploader.size > 10 * 1024 * 1024:
    st.error("File too large. Please upload a file < 10MB.")
    st.stop()

temp_file_path = save_uploaded_file(file_uploader)
if not validate_csv_file(temp_file_path):
    cleanup_temp_file(temp_file_path)
    st.stop()

df = pd.read_csv(temp_file_path)
st.success(f"✅ File uploaded! Shape: {df.shape}")

with st.expander("📋 Data Preview"):
    st.dataframe(df.head(10))

with st.expander("📊 Data Info"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Columns:**")
        st.write(list(df.columns))
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes.to_dict())

# === Main Application Logic ===
if menu == "Summarize":
    st.header("🔎 Enhanced Data Analysis & Visualization Goals")
    
    # Enhanced summary option
    summary_type = st.radio(
        "Choose Summary Type:",
        ["Basic LIDA Summary", "Detailed Insightful Summary", "Both"],
        help="Select the type of data summary you want to generate"
    )

    if st.button("🚀 Generate Analysis", type="primary"):
        with st.spinner("Analyzing your data..."):
            try:
                # Generate LIDA summary
                summary = lida.summarize(temp_file_path, summary_method='default', textgen_config=textgen_config)
                
                if summary_type in ["Basic LIDA Summary", "Both"]:
                    if not summary:
                        st.error("LIDA summary generation failed.")
                    else:
                        summary_text = summarize_to_text(summary)
                        st.subheader("📋 Basic LIDA Data Summary")
                        st.markdown(summary_text)
                
                if summary_type in ["Detailed Insightful Summary", "Both"]:
                    st.subheader("🔍 Detailed Insightful Data Summary")
                    insightful_summary = generate_insightful_summary(df, file_uploader.name)
                    st.markdown(insightful_summary)

                # Generate visualization goals
                if summary:
                    with st.spinner("Generating visualization goals..."):
                        goals = lida.goals(summary=summary, textgen_config=textgen_config)

                    if not goals:
                        st.warning("No visualization goals generated.")
                    else:
                        st.subheader("🎯 Suggested Visualizations")
                        for i, goal in enumerate(goals if isinstance(goals, list) else [goals], 1):
                            text = goal.get('question') if isinstance(goal, dict) else str(goal)
                            st.write(f"**{i}.** {text}")

            except Exception as e:
                st.error(f"Analysis error: {e}")
            finally:
                cleanup_temp_file(temp_file_path)

elif menu == "Question based graph":
    st.header("🧠 Custom Question-Based Visualization")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area(
            "Enter your visualization question", 
            height=100, 
            placeholder="e.g., Show a violin plot of sales by region, Create a heatmap of correlations, Make a scatter plot of price vs quantity"
        )
    
    with col2:
        st.write("**Quick Examples:**")
        example_queries = [
            "violin plot of values by category",
            "heatmap of correlations",
            "scatter plot with regression line", 
            "box plot distribution",
            "bar chart comparison",
            "line plot trends over time",
            "pie chart proportions",
            "histogram distribution"
        ]
        
        for example in example_queries:
            if st.button(f"📝 {example}", key=f"example_{example}"):
                query = f"Create a {example} using the uploaded data"
                st.experimental_rerun()

    if st.button("🎨 Generate Custom Graph", type="primary", disabled=not query.strip()):
        with st.spinner("Processing your request..."):
            try:
                summary = lida.summarize(temp_file_path, summary_method='default', textgen_config=textgen_config)
                
                # Enhance query with chart type information
                enhanced_query = enhance_query_with_chart_type(query, visualization_library)
                
                st.info(f"🔍 **Enhanced Query:** {enhanced_query}")

                charts = lida.visualize(
                    summary=summary,
                    goal=enhanced_query,
                    textgen_config=textgen_config,
                    library=visualization_library
                )

                if charts and len(charts) > 0:
                    chart = charts[0]
                    st.subheader("📊 Generated Visualization")
                    
                    img = base64_to_image(chart.raster)
                    if img:
                        st.image(img, use_column_width=True)
                        
                        # Display code and download options
                        if hasattr(chart, 'code'):
                            with st.expander("📝 Generated Code"):
                                st.code(chart.code, language="python")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        "💾 Download Code", 
                                        data=chart.code, 
                                        file_name=f"visualization_{visualization_library}.py",
                                        mime="text/plain"
                                    )
                                with col2:
                                    st.download_button(
                                        "🖼️ Download Image",
                                        data=base64.b64decode(chart.raster),
                                        file_name="visualization.png",
                                        mime="image/png"
                                    )
                    else:
                        st.error("Failed to display chart image.")
                else:
                    st.warning("No chart was generated. Try rephrasing your question or being more specific about the visualization type.")
                    
            except Exception as e:
                st.error(f"Visualization error: {e}")
                st.info("💡 **Tip:** Try being more specific about the chart type (e.g., 'scatter plot', 'bar chart', 'heatmap')")
            finally:
                cleanup_temp_file(temp_file_path)

elif menu == "Chart Type Explorer":
    st.header("🎨 Chart Type Explorer")
    st.markdown("Explore different chart types available in your selected visualization library.")
    
    # Chart type selection
    chart_types = SEABORN_CHART_TYPES if visualization_library == "seaborn" else MATPLOTLIB_CHART_TYPES
    
    selected_chart = st.selectbox(
        f"Select a {visualization_library.title()} Chart Type:",
        list(chart_types.keys()),
        help="Choose a chart type to generate automatically"
    )
    
    st.info(f"**{selected_chart.title()}:** {chart_types[selected_chart]}")
    
    # Column selection for the chart
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if numeric_cols:
            x_col = st.selectbox("X-axis (Numeric)", numeric_cols + categorical_cols)
        else:
            x_col = st.selectbox("X-axis", df.columns.tolist())
    
    with col2:
        if numeric_cols:
            y_col = st.selectbox("Y-axis (Numeric)", numeric_cols + [None])
        else:
            y_col = st.selectbox("Y-axis", df.columns.tolist() + [None])
    
    # Generate chart type specific query
    if st.button(f"🎨 Generate {selected_chart.title()} Chart", type="primary"):
        with st.spinner(f"Creating {selected_chart} visualization..."):
            try:
                summary = lida.summarize(temp_file_path, summary_method='default', textgen_config=textgen_config)
                
                # Create specific query based on chart type and selected columns
                if y_col:
                    query = f"Create a {selected_chart} plot showing {x_col} vs {y_col}"
                else:
                    query = f"Create a {selected_chart} plot for {x_col}"
                
                # Add library-specific instructions
                if visualization_library == "seaborn":
                    query += f" using seaborn's {selected_chart} plot function"
                else:
                    query += f" using matplotlib's {selected_chart} plot function"
                
                charts = lida.visualize(
                    summary=summary,
                    goal=query,
                    textgen_config=textgen_config,
                    library=visualization_library
                )

                if charts and len(charts) > 0:
                    chart = charts[0]
                    st.subheader(f"📊 {selected_chart.title()} Visualization")
                    
                    img = base64_to_image(chart.raster)
                    if img:
                        st.image(img, use_column_width=True)
                        
                        if hasattr(chart, 'code'):
                            with st.expander("📝 Generated Code"):
                                st.code(chart.code, language="python")
                                st.download_button(
                                    "💾 Download Code", 
                                    data=chart.code, 
                                    file_name=f"{selected_chart}_{visualization_library}.py"
                                )
                    else:
                        st.error("Failed to display chart.")
                else:
                    st.warning(f"Could not generate {selected_chart} chart. Try a different chart type.")
                    
            except Exception as e:
                st.error(f"Chart generation error: {e}")
            finally:
                cleanup_temp_file(temp_file_path)

# === Footer ===
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Enhanced LIDA Visualization Assistant | "
    f"Supporting all {visualization_library.title()} chart types"
    "</div>", 
    unsafe_allow_html=True
)
