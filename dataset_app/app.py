from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

# Load your dataset (put your CSV file here later)
df = pd.read_csv("websockets.csv")

@app.route("/")
def home():
    return "<h2>Go to <a href='/quantitative'>Quantitative Analysis</a> or <a href='/visualization/1'>Visualization</a></h2>"

@app.route("/quantitative", methods=["GET", "POST"])
def quantitative():
    plot_url1, plot_url2 = None, None
    if request.method == "POST":
        try:
            nrows = int(request.form["nrows"])
            data = df.head(nrows)
        except:
            data = df

        # Example graph 1
        fig, ax = plt.subplots()
        ax.plot(data.index, data.iloc[:, 0])  
        ax.set_title("Coding Effort")
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url1 = base64.b64encode(img.getvalue()).decode()

        # Example graph 2
        fig, ax = plt.subplots()
        ax.plot(data.index, data.iloc[:, 1], label="Score1")
        ax.plot(data.index, data.iloc[:, 2], label="Score2")
        ax.legend()
        ax.set_title("Similarity Scores")
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url2 = base64.b64encode(img.getvalue()).decode()

    return render_template("quantitative.html", plot_url1=plot_url1, plot_url2=plot_url2)

@app.route("/visualization/<int:row_id>")
def visualization(row_id):
    row = df.iloc[row_id]
    return render_template("visualization.html", row=row)

if __name__ == "__main__":
    app.run(debug=True)
