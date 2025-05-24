import pandas as pd
from datetime import datetime
import joblib
from scraper_module import scrape_karkidi_jobs  
import smtplib
from email.mime.text import MIMEText

MODEL_PATH = "kmeans_model.pkl"
USER_PREFERRED_CLUSTERS = [0, 2, 4] 
USER_EMAIL = "rejoyjohny@gmail.com"  # email to send alert

def predict_clusters(df):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train the model first.")
        return df
    model, vectorizer = joblib.load(MODEL_PATH)
    X = vectorizer.transform(df['Skills'].fillna(""))
    df['Cluster'] = model.predict(X)
    return df

def send_email_alert(subject, body, to_email):
    # Simple SMTP setup for Gmail example (use app password or another SMTP server)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    from_email = "your_email@gmail.com"
    password = "your_email_password"

    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
    print(f"Alert sent to {to_email}")

def main():
    print(f"Started daily scraping at {datetime.now()}")
    df_new_jobs = scrape_karkidi_jobs(keyword="data science", pages=2, delay=1)
    df_new_jobs = predict_clusters(df_new_jobs)

    matched_jobs = df_new_jobs[df_new_jobs['Cluster'].isin(USER_PREFERRED_CLUSTERS)]

    if not matched_jobs.empty:
        body = "New jobs matching your preferences:\n\n"
        for _, row in matched_jobs.iterrows():
            body += f"{row['Title']} at {row['Company']} (Cluster {row['Cluster']})\nLink: {row['JobURL']}\n\n"

        send_email_alert("New Karkidi Job Alerts", body, USER_EMAIL)
    else:
        print("No new jobs matching preferences today.")

if __name__ == "__main__":
    main()
