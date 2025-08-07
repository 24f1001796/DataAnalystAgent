from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import requests
from bs4 import BeautifulSoup

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Utility: Scrape Wikipedia Table ========== #
def scrape_wikipedia_table():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # First table with highest grossing films
    table = soup.find("table", class_="wikitable")

    df = pd.read_html(str(table))[0]
    # Clean up column names
    df.columns = [col.replace("\n", " ").strip() for col in df.columns]

    # Remove rows with missing values
    df = df.dropna(subset=["Rank", "Peak"])
    
    # Convert necessary columns
    df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')

    return df

# ========== Utility: Generate Scatterplot as Base64 ========== #
def generate_scatterplot(df):
    df = df.dropna(subset=['Rank', 'Peak'])
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Rank'], df['Peak'], color='blue')
    m, b = np.polyfit(df['Rank'], df['Peak'], 1)
    plt.plot(df['Rank'], m * df['Rank'] + b, color='red', linestyle='dotted')
    plt.xlabel('Rank')
    plt.ylabel('Peak')
    plt.title('Rank vs Peak Scatterplot')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return f"data:image/png;base64,{img_base64}"

# ========== Utility: Answer Questions ========== #
def answer_questions(df, questions):
    answers = []

    for q in questions:
        q_lower = q.lower()

        if "before 2000" in q_lower and "$2 bn" in q_lower:
            df_filtered = df[df['Worldwide gross'].str.contains(r'\$2[\.\,]')]
            df_filtered['Year'] = pd.to_datetime(df_filtered['Title'].str.extract(r'\((\d{4})\)', expand=False), errors='coerce')
            count = df_filtered[df_filtered['Year'].dt.year < 2000].shape[0]
            answers.append(f"{count} movies grossed over $2B before 2000.")

        elif "earliest film" in q_lower and "$1.5 bn" in q_lower:
            df['Gross'] = df['Worldwide gross'].str.replace(r'[^0-9.]', '', regex=True).astype(float)
            filtered = df[df['Gross'] > 1.5e9]
            earliest = filtered.iloc[filtered['Rank'].idxmin()]
            answers.append(f"The earliest $1.5B+ film is {earliest['Title']}.")

        elif "correlation between the rank and peak" in q_lower:
            correlation = df['Rank'].corr(df['Peak'])
            answers.append(f"The correlation between Rank and Peak is {correlation:.2f}.")

        elif "scatterplot" in q_lower:
            image_url = generate_scatterplot(df)
            answers.append(image_url)

        else:
            answers.append("I don't know how to answer that yet.")

    return answers

# ========== Route: /analyze ========== #
@app.post("/analyze")
async def analyze_file(questions: UploadFile = File(...)):
    contents = await questions.read()
    text = contents.decode("utf-8")
    question_list = [q.strip() for q in text.split("\n") if q.strip()]
    
    df = scrape_wikipedia_table()
    answers = answer_questions(df, question_list)

    return JSONResponse(content={"answers": answers})

# ========== Run the Server ========== #
if __name__ == "__main__":
    uvicorn.run("main:app", port=8080, reload=True)
