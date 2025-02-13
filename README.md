# ETA Prediction Application

## Project Overview
This project estimates the Estimated Time of Arrival (ETA) for trips in Hyderabad. It utilizes machine learning models to analyze trip data and predict arrival times based on various factors.

## Technologies Used
- **Programming Language:** Python
- **Framework:** Streamlit
- **Machine Learning:** Scikit-Learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Altair
- **Web Requests:** Requests

## File Structure
- **app.py**: Main application file for running the Streamlit interface.
- **Estimating the Time of Arrival.ipynb**: Jupyter Notebook containing exploratory data analysis and model training.
- **hyderabad_eta_data.csv**: Dataset containing trip information.
- **requirements.txt**: List of dependencies required for running the project.
- **time_estimater.pickle**: Serialized trained model for predicting ETAs.
- **innomatics-footer-logo.png**: Company branding/logo.

## Installation Guide
1. Clone the repository or download the files.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## How to Use
1. Upload or select a dataset with trip details.
2. Enter trip parameters such as start location, destination, and time.
3. Click on the **Predict ETA** button.
4. The application will display the estimated arrival time based on the trained model.

## Model Details
- The model is trained using Scikit-Learn.
- The dataset contains trip details such as start time, distance, and traffic conditions.
- The trained model is stored as `time_estimater.pickle`.

## Future Enhancements
- Improve prediction accuracy using deep learning models.
- Add real-time traffic data integration.
- Deploy as a web application using cloud services.

## Credits
Knowledge by **Innomatics Research Labs**.
Developed by **Amarendra Nayak**.

For any queries, please reach out to **[toamarendranayak@gmail.com]**.

