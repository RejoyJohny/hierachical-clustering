import streamlit as st
import pandas as pd
import os
import joblib
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

MODEL_PATH = "kmeans_model.pkl"
DATA_PATH = "karkidi_jobs.csv"

# --------------------------- Scraper ---------------------------
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

# --------------------------- Clustering ---------------------------
def train_model(df, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Skills'].fillna(""))
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X)
    df['Cluster'] = model.labels_
    joblib.dump((model, vectorizer), MODEL_PATH)
    df.to_csv(DATA_PATH, index=False)
    return df

# --------------------------- Predict New ---------------------------
def predict_clusters(df):
    if not os.path.exists(MODEL_PATH):
        return df
    model, vectorizer = joblib.load(MODEL_PATH)
    X = vectorizer.transform(df['Skills'].fillna(""))
    df['Cluster'] = model.predict(X)
    return df

# --------------------------- Get Cluster Labels ---------------------------
def get_cluster_labels():
    if not os.path.exists(DATA_PATH):
        return {}
    df = pd.read_csv(DATA_PATH)
    cluster_labels = {}
    for i in sorted(df['Cluster'].unique()):
        top_skills = (
            df[df['Cluster'] == i]['Skills']
            .str.split(',')
            .explode()
            .str.strip()
            .value_counts()
            .head(3)
            .index.tolist()
        )
        cluster_labels[i] = ", ".join(top_skills)
    return cluster_labels

# --------------------------- Streamlit UI ---------------------------
st.title(" Karkidi Job Clustering App")
st.markdown("""
Automatically scrape jobs from [karkidi.com](https://www.karkidi.com), cluster them by required skills,
and notify users about matching jobs based on their skill interests.
""")

option = st.sidebar.selectbox("Choose an action", ["Scrape & Train", "Classify New Jobs", "Get Alerts"])

if option == "Scrape & Train":
    keyword = st.text_input("Enter keyword for jobs (e.g., data science)", "data science")
    pages = st.slider("Number of pages to scrape", 1, 5, 2)
    clusters = st.slider("Number of skill clusters", 2, 10, 5)
    if st.button("Scrape and Train Model"):
        df = scrape_karkidi_jobs(keyword=keyword, pages=pages)
        df = train_model(df, num_clusters=clusters)
        st.success(f"Scraped and clustered {len(df)} jobs into {clusters} clusters.")
        st.dataframe(df.head())

elif option == "Classify New Jobs":
    keyword = st.text_input("Keyword to classify new jobs", "data science")
    pages = st.slider("Number of pages to scrape", 1, 5, 1)
    if st.button("Classify Jobs"):
        df = scrape_karkidi_jobs(keyword=keyword, pages=pages)
        df = predict_clusters(df)
        st.dataframe(df.head())

elif option == "Get Alerts":
    st.header(" Get Job Alerts by Preferred Cluster")
    cluster_labels = get_cluster_labels()

    if not cluster_labels:
        st.warning("No trained model found. Please run 'Scrape & Train' first.")
    else:
        selected = st.multiselect(
            "Select your preferred job skill clusters to get alerts:",
            options=list(cluster_labels.keys()),
            format_func=lambda x: f"Cluster {x}: {cluster_labels[x]}"
        )

        if st.button("Show Alerts"):
            df = scrape_karkidi_jobs(pages=1)
            df = predict_clusters(df)
            filtered_df = df[df['Cluster'].isin(selected)]
            st.dataframe(filtered_df)
            if not filtered_df.empty:
                st.success(f"{len(filtered_df)} matching jobs found!")
            else:
                st.info("No matching jobs found right now.")
