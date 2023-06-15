import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# import scikit-learn


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the feature names
feature_names = [
    "friend_cnt",
    "avg_friend_age",
    "friend_country_cnt",
    "subscriber_friend_cnt",
    "songsListened",
    "subscription_spending_$",
    "posts",
    "shouts",
]
mms = MinMaxScaler()  # Normalization


def pre_process_data(dataset):
    dataset_clean_na = dataset.dropna()
    result = pd.DataFrame(
        mms.fit_transform(dataset_clean_na), columns=dataset_clean_na.columns
    )
    return result


# Create the Streamlit app
def main():
    # Set the app title
    st.title("Music Map Prediction App")

    # Create input fields for the features
    friend_cnt = st.slider("Friend Count", 0, 6000, value=500)
    avg_friend_age = st.slider("Average Friend Age", 0, 80, value=30)
    friend_country_cnt = st.slider("Friend Country Count", 0, 136)
    subscriber_friend_cnt = st.slider("Subscriber Friend Count", 0, 309)
    songs_listened = st.slider("Songs Listened", 0, 100000, value=35000)
    subscription_spending = st.slider(
        "Subscription Spending ($)", 0, 45000, value=11000
    )
    posts = st.slider("Posts", 0, 15185, value=157)
    shouts = st.slider("Shouts", 0, 65000)

    # Create a dictionary with the input data
    input_data = {
        "friend_cnt": [friend_cnt],
        "avg_friend_age": [avg_friend_age],
        "friend_country_cnt": [friend_country_cnt],
        "subscriber_friend_cnt": [subscriber_friend_cnt],
        "songsListened": [songs_listened],
        "subscription_spending_$": [subscription_spending],
        "posts": [posts],
        "shouts": [shouts],
    }
    # Convert the input features into a DataFrame

    input_data_df = pd.DataFrame(input_data)
    # input_data_df
    # Create a prediction button
    if st.button("Predict"):
        # normalize
        pre_process_data(input_data_df)

        # Make the prediction
        prediction = model.predict(input_data_df)
        if prediction[0] == 0:
            results = "Non Adopter"
        else:
            results = "Adopter"

        # Display the prediction result
        st.success(f"This user will be: {results}")


# Run the app
if __name__ == "__main__":
    main()
