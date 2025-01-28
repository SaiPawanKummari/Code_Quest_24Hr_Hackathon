from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the dataset and train the model (this is simplified for the example)
student = pd.read_csv('student_dataset1.csv')

# Prepare the features and target variable
X = student.drop(['Name', 'Core_Course_1_Grade','Overall_Score','Teaching_Assistant' ], axis=1)
y = student['Overall_Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict on the entire dataset for ranking purposes
predicted_scores = model.predict(X)

# Add the predicted scores to the original dataframe for ranking
student['Predicted_Score'] = predicted_scores

# Sort the students based on the predicted scores (descending order)
student_ranked = student.sort_values(by='Predicted_Score', ascending=False)

# Group by 'Year_Of_Admission' and get the top 3 students for each year
top_students = student_ranked.groupby('Admitted_year').head(3)

@app.route('/')
def index():
    admitted_years = student['Admitted_year'].unique()  # Get unique admission years
    return render_template('index.html', admitted_years=admitted_years)

@app.route('/year/<int:year>')
def year_results(year):
    # Filter the top students for the selected year
    year_students = top_students[top_students['Admitted_year'] == year].copy()  # Use .copy() to avoid SettingWithCopyWarning
    year_students['Rank'] = year_students['Predicted_Score'].rank(method='first', ascending=False)

    return render_template('results.html', year=year, top_students=year_students)

@app.route('/all_results')
def all_results():
    # Get top 3 students for each year (already calculated in top_students)
    return render_template('all_results.html', top_students=top_students)

@app.route('/student/<int:student_id>')
def student_details(student_id):
    # Get the student details by filtering the dataframe
    student_detail = student[student['Student_id'] == student_id].iloc[0]

    # Pass the student's details to the template
    return render_template('student_details.html', student=student_detail)

if __name__ == '__main__':
    app.run(debug=True)
