import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import os
import schedule
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = "kmeans_model.pkl"
USER_PREFS_FILE = "user_preferences.csv"
DATA_PATH = "karkidi_jobs_daily.csv"
ALERT_FOLDER = "alerts"  # Folder to store individual user alerts

# --- Scrape function ---
def scrape_karkidi_jobs(keyword="data science", pages=1, delay=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")

        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""
                job_url = "https://www.karkidi.com" + job.find("a", href=True)['href']

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Skills": skills,
                    "Summary": summary,
                    "JobURL": job_url,
                    "ScrapedAt": datetime.now().isoformat()
                })
            except:
                continue
        time.sleep(delay)

    return pd.DataFrame(jobs_list)

# --- Classify jobs ---
def classify_jobs(df):
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return df
    model, vectorizer = joblib.load(MODEL_PATH)
    X = vectorizer.transform(df['Skills'].fillna(""))
    df['Cluster'] = model.predict(X)
    return df

# --- Alert users ---
def alert_users(df):
    if not os.path.exists(USER_PREFS_FILE):
        print("No user preferences found.")
        return

    os.makedirs(ALERT_FOLDER, exist_ok=True)
    prefs = pd.read_csv(USER_PREFS_FILE)

    for _, row in prefs.iterrows():
        name = row['Name']
        preferred_clusters = eval(row['PreferredClusters'])  # Convert string list to list
        matched_jobs = df[df['Cluster'].isin(preferred_clusters)]

        if not matched_jobs.empty:
            alert_file = os.path.join(ALERT_FOLDER, f"{name.lower().replace(' ', '_')}_alerts.csv")
            matched_jobs.to_csv(alert_file, index=False)
            print(f"[{datetime.now().isoformat()}] Alert saved for {name}: {len(matched_jobs)} jobs.")
        else:
            print(f"[{datetime.now().isoformat()}] No jobs found for {name} today.")

# --- Main task to run daily ---
def run_daily_scrape():
    print(f"Starting daily scrape: {datetime.now().isoformat()}")
    df = scrape_karkidi_jobs(keyword="data science", pages=2)
    if df.empty:
        print("No jobs scraped.")
        return
    df = classify_jobs(df)
    df.to_csv(DATA_PATH, index=False)
    alert_users(df)
    print("Daily scrape and alert completed.")

# Schedule for once a day (e.g., 9 AM)
schedule.every().day.at("09:00").do(run_daily_scrape)

if __name__ == "__main__":
    print("Scheduler started...")
    while True:
        schedule.run_pending()
        time.sleep(60)
