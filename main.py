import pandas as pd
import streamlit as st
import pickle
import yfinance as yf
import altair as alt

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Stock Price Prediction</h1>", unsafe_allow_html=True)

# Company options and ticker mapping
companies = ["Apple", 'Microsoft', 'Amazon', 'Google']
company_map = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL"
}

# User selects a company
name = st.selectbox("Choose the company", companies, index=None, placeholder="Select a company")

if st.button("Predict"):
    selected_company = company_map[name]
    st.write(f"You selected: {selected_company}")

    # Download stock data
    data = yf.download(selected_company, start='2014-01-01')

    # Select latest 5 rows for prediction
    input_data = data[['High', 'Low', 'Open', 'Volume']].tail(5).copy()

    # Convert all columns to float
    input_data = input_data.astype(float)

    # Make prediction
    prediction = model.predict(input_data)

    # Prepare results
    result_df = pd.DataFrame({
        'Date': input_data.index,
        'Predicted Close': prediction
    })

    close_data = [x[0] for x in data['Close'].tail(5).values]
    predicted_close = [y for y in prediction]
    print(close_data, predicted_close)

    # Combine into one DataFrame for plotting
    plot_df = pd.DataFrame({
        'Date': data['Close'].tail(5).index.tolist() * 2,
        'Value': close_data + predicted_close,
        'Type': ['Actual'] * 5 + ['Predicted'] * 5
    })

    # Altair line chart
    chart = alt.Chart(plot_df).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Value:Q', title='Stock Price',
                scale=alt.Scale(domain=[min(plot_df['Value']) - 5, max(plot_df['Value']) + 5])),
        color='Type:N'
    ).properties(
        width=600,
        height=400,
        title='Actual vs Predicted Closing Prices'
    )

    st.altair_chart(chart, use_container_width=True)

    st.write("Predicted Closing Prices:")
    st.dataframe(result_df)
