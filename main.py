import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Razzle - AI Stock Analysis")

# Custom CSS for modern styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1e3d59 !important;
        margin-bottom: 2rem !important;
    }
    .stSubheader {
        color: #17a2b8 !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title with custom styling
st.markdown('<h1 class="stTitle">Razzle</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.5rem; color: #666;">AI-based Stock Analysis & Prediction</p>', unsafe_allow_html=True)

# Create a modern search bar
with st.container():
    col1, col2, col3 = st.columns([2,6,2])
    with col2:
        stock = st.text_input('Enter Stock Ticker', 'AAPL', 
                            help='Enter the stock symbol (e.g., AAPL for Apple Inc.)')

# Get stock data
try:
    ticker = yf.Ticker(stock)
    # Initial date range (will be updated based on selected period)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    
    df = ticker.history(start=start, end=end)
    info = ticker.info

    # Modern metrics layout
    st.markdown('<h2 class="stSubheader">Key Metrics</h2>', unsafe_allow_html=True)
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric(
            "Current Price",
            f"${info.get('currentPrice', 'N/A')}",
            f"{((info.get('currentPrice', 0) - info.get('previousClose', 0)) / info.get('previousClose', 1) * 100):.2f}%"
        )
    
    with metrics_col2:
        st.metric(
            "Market Cap",
            f"${(info.get('marketCap', 0) / 1e9):.2f}B",
            None
        )
    
    with metrics_col3:
        st.metric(
            "P/E Ratio",
            f"{info.get('trailingPE', 'N/A'):.2f}",
            None
        )
    
    with metrics_col4:
        st.metric(
            "Volume",
            f"{info.get('volume', 'N/A'):,}",
            None
        )

    # Detailed information in expandable sections
    with st.expander("Detailed Stock Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Price Metrics")
            st.write(f"**Previous Close:** ${info.get('previousClose', 'N/A')}")
            st.write(f"**52-Week High:** ${info.get('fiftyTwoWeekHigh', 'N/A')}")
            st.write(f"**52-Week Low:** ${info.get('fiftyTwoWeekLow', 'N/A')}")
            st.write(f"**Day Range:** ${info.get('dayLow', 'N/A')} - ${info.get('dayHigh', 'N/A')}")
        
        with col2:
            st.markdown("### Company Metrics")
            st.write(f"**EPS (TTM):** ${info.get('trailingEps', 'N/A')}")
            st.write(f"**Forward EPS:** ${info.get('forwardEps', 'N/A')}")
            st.write(f"**Price-to-Book:** {info.get('priceToBook', 'N/A'):.2f}")
            st.write(f"**Shares Outstanding:** {(info.get('sharesOutstanding', 0) / 1e9):.2f}B")

    # Technical Analysis Section
    st.markdown('<h2 class="stSubheader">Technical Analysis</h2>', unsafe_allow_html=True)

    # Add time period selector
    time_periods = {
        "1 Day": 1,
        "1 Month": 30,
        "6 Months": 180,
        "1 Year": 365,
        "10 Years": 3650
    }

    selected_period = st.selectbox(
        "Select Time Period",
        options=list(time_periods.keys()),
        index=1  # Default to 1 Month (30 days)
    )

    # Calculate date range based on selected period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=time_periods[selected_period])
    period_start = start_date.strftime('%Y-%m-%d')
    period_end = end_date.strftime('%Y-%m-%d')

    # Get stock data for selected period
    df_period = ticker.history(start=period_start, end=period_end)

    # Create interactive price chart
    fig = make_subplots(rows=2, cols=1, 
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05,
                        subplot_titles=(f'{selected_period} Price Analysis', 'Volume'))

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_period.index,
            open=df_period['Open'],
            high=df_period['High'],
            low=df_period['Low'],
            close=df_period['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Add moving averages
    if time_periods[selected_period] > 100:  # Only show MAs for longer periods
        fig.add_trace(
            go.Scatter(
                x=df_period.index,
                y=df_period['Close'].rolling(100).mean(),
                name='100MA',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )

    if time_periods[selected_period] > 200:  # Only show 200MA for periods longer than 200 days
        fig.add_trace(
            go.Scatter(
                x=df_period.index,
                y=df_period['Close'].rolling(200).mean(),
                name='200MA',
                line=dict(color='green', width=1)
            ),
            row=1, col=1
        )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df_period.index,
            y=df_period['Volume'],
            name='Volume',
            marker=dict(color='rgba(0, 0, 255, 0.5)')
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'{stock} Stock Analysis - {selected_period}',
        yaxis_title='Stock Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,  # Disable rangeslider for primary chart
        template='plotly_white',
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Update Y-axes labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Update X-axes
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

    # Add buttons for zoom levels
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
    )

    # Show the chart
    st.plotly_chart(fig, use_container_width=True)

    # Add price statistics for the selected period
    with st.expander("Price Statistics for Selected Period"):
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric(
                "Period High",
                f"${df_period['High'].max():.2f}",
                None
            )
        
        with stats_col2:
            st.metric(
                "Period Low",
                f"${df_period['Low'].min():.2f}",
                None
            )
        
        with stats_col3:
            period_return = ((df_period['Close'][-1] - df_period['Close'][0]) / df_period['Close'][0] * 100)
            st.metric(
                "Period Return",
                f"{period_return:.2f}%",
                None
            )
        
        with stats_col4:
            st.metric(
                "Average Volume",
                f"{df_period['Volume'].mean():,.0f}",
                None
            )

    # AI Predictions Section
    st.markdown('<h2 class="stSubheader">AI Price Predictions</h2>', unsafe_allow_html=True)
    
    # Add explanation for the prediction graph
    st.markdown("""
    **Understanding the Prediction Graph:**
    - **X-axis (Time)**: Represents the chronological sequence of trading days in the test period
    - **Y-axis (Price)**: Shows the stock price in USD
    - **Blue line**: Actual historical stock prices
    - **Red line**: AI model's predicted prices for the same period
    
    The AI model uses the past 100 days of price data to predict each day's price. This helps in 
    understanding how well the model can capture the stock's price movements.
    """)

    # Data preprocessing for predictions
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    try:
        # Load the model and make predictions
        model = load_model('my_model.keras')
        
        # Prepare test data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test = []
        y_test = []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)

        # Scale back to original range
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Create prediction chart
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            y=y_test,
            name='Actual Price',
            line=dict(color='blue')
        ))
        fig_pred.add_trace(go.Scatter(
            y=y_predicted.flatten(),
            name='Predicted Price',
            line=dict(color='red')
        ))
        
        fig_pred.update_layout(
            title='AI Price Predictions vs Actual Prices',
            xaxis_title='Trading Days (Most Recent 30% of 1-Year Data)',
            yaxis_title='Stock Price (USD)',
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add range slider
        fig_pred.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(fig_pred, use_container_width=True)

        # Show prediction performance metrics
        mse = np.mean((y_test - y_predicted.flatten()) ** 2)
        rmse = np.sqrt(mse)
        st.markdown("### Prediction Performance Metrics")
        st.write(f"Root Mean Square Error (RMSE): ${rmse:.2f}")

    except Exception as e:
        st.error("Error loading the AI model or making predictions.")


    # PDF Export Function
    def create_stock_report_pdf():
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        
        elements.append(Paragraph(f"Stock Analysis Report - {stock}", title_style))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Key Metrics", styles['Heading2']))
        metrics_data = [
            ["Metric", "Value"],
            ["Current Price", f"${info.get('currentPrice', 'N/A')}"],
            ["Market Cap", f"${(info.get('marketCap', 0) / 1e9):.2f}B"],
            ["P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}"],
            ["52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}"],
            ["52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}"],
            ["EPS (TTM)", f"${info.get('trailingEps', 'N/A')}"],
        ]
        
        table = Table(metrics_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        doc.build(elements)
        buffer.seek(0)
        return buffer

    # Add download button
    st.markdown('<h2 class="stSubheader">Export Report</h2>', unsafe_allow_html=True)
    pdf_buffer = create_stock_report_pdf()
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name=f"{stock}_analysis_report.pdf",
        mime="application/pdf",
        key='download_button',
        help="Download a detailed PDF report of the stock analysis"
    )

except Exception as e:
    st.error(f"Error fetching data for {stock}. Please check the ticker symbol and try again.")