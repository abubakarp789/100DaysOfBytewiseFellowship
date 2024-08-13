from flask import Flask, render_template, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load your cleaned data
data = pd.read_csv('data/cleaned_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data_page():
    return render_template('data.html', tables=[data.to_html(classes='data')], titles=data.columns.values)

@app.route('/analysis')
def analysis():
    # Generate plots (this can be expanded)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['quali_pos'], bins=20, kde=False)
    plt.title('Distribution of Qualifying Positions')
    plt.xlabel('Qualifying Position')
    plt.ylabel('Frequency')
    plt.savefig('static/images/quali_pos_dist.png')
    plt.close()

    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)
