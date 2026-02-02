from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/intrusion_model.pkl")
encoders = joblib.load("model/encoders.pkl")

columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins',
    'logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files',
    'num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        df = pd.read_csv(file, names=columns)
        df.drop(["label", "difficulty"], axis=1, inplace=True)

        # Apply SAME encoders
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        prediction = model.predict(df)
        attacks = sum(prediction)
        result = f"Detected {attacks} suspicious records"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
