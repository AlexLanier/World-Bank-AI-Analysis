# Quick Start Guide

Get the World Bank Loan Prediction Flask App running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- The data files from your notebook: `worldbank_loans.parquet`, `gdp_total.csv`, `gdp_per_capita.csv`

## Step-by-Step Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

This will create the `trained_model.cbm` file:

```bash
python train_model.py
```

**Note:** This may take several minutes depending on your system. You'll see progress updates as it trains.

### 3. Start the Flask App

```bash
python app.py
```

You should see:
```
🚀 Starting Flask app...
✅ Model loaded successfully
✅ Label encoder loaded successfully
✅ Ready to serve predictions!
 * Running on http://0.0.0.0:5000
```

### 4. Open in Browser

Go to: `http://localhost:5000`

### 5. Make Predictions!

Fill in the form with sample loan data and click "Predict Loan Outcome"

## Troubleshooting

### "Model file not found"
- Run `python train_model.py` first to train the model

### "Module not found" errors
- Make sure you installed requirements: `pip install -r requirements.txt`

### Port already in use
- The app runs on port 5000 by default
- Change it in `app.py`: `app.run(port=5001)`

### Training takes too long
- The default is 500 iterations
- You can reduce it in `train_model.py` for faster training during development

## Sample Data for Testing

Try these values to test the prediction:

```
Loan Type: SCP USD
Region: EAST ASIA AND PACIFIC
Interest Rate: 4.25
Original Principal Amount: 50000000
Borrower's Obligation: 50000000
Due to IBRD: 50000000
GDP Total: 1000000000000
GDP Per Capita: 5000
```

## What's Next?

- Customize the UI in `templates/index.html`
- Add more features to the model
- Deploy to a cloud service like Heroku, AWS, or Azure
- Create an API documentation with tools like Swagger

## Need Help?

Check the main README.md for more details about the project structure and features.

