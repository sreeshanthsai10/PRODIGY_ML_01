import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Load the dataset
data = pd.read_csv(r"C:\PRODIGY_INTERNSHIP TASKS\train.csv")
# Select features and target variable
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)

# Function to predict house price based on user input
def predict_house_price():
    print("Enter the following details to predict the house price:")
    square_footage = float(input("Square Footage: "))
    bedrooms = int(input("Number of Bedrooms: "))
    bathrooms = int(input("Number of Bathrooms: "))

    # Create a DataFrame with the user inputs
    new_data = pd.DataFrame({
        'GrLivArea': [square_footage],
        'BedroomAbvGr': [bedrooms],
        'FullBath': [bathrooms]
    })

    # Predict the price
    predicted_price = model.predict(new_data)
    print(f'\nPredicted House Price: ${predicted_price[0]:,.2f}')

# Run the prediction function
predict_house_price()

