#  Karkidi Job Scraper & Skill-Based Classifier

This project scrapes job listings from [Karkidi.com](https://www.karkidi.com/), clusters them using unsupervised learning based on required skills, classifies new job postings into these clusters, and notifies users based on their skill preferences. It also features a Streamlit web interface for interaction.

---

##  Features

-  Scrapes job postings from Karkidi.com
-  Extracts job title, company name, skills, and location
-  Clusters jobs based on required skills using **KMeans**
-  Classifies new jobs into skill clusters
-  Allows users to set skill preferences
-  Sends daily job alerts based on preferences
-  Fully automated daily scraping using `schedule`
-  Interactive web UI built with **Streamlit**

---

##  Data Collected

Each job posting includes:

- **Job Title**
- **Company Name**
- **Required Skills**
- **Location**

---

##  Clustering Approach

- **Algorithm:** KMeans (unsupervised clustering)
- **Number of Clusters:** Determined using manual inspection and elbow method
- **Vectorization:** TF-IDF vectorizer on cleaned skill text

---


##  Streamlit App

Run the web interface:

```bash
streamlit run app.py
