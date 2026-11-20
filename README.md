
# 🔍 TextGuard
### NLP-Based Document Similarity & Plagiarism Screening System

TextGuard is a Python-based Natural Language Processing (NLP)
application that compares two documents and identifies potentially
similar or duplicated content.

The application uses TF-IDF Cosine Similarity, Jaccard Similarity,
and Bigram Similarity to analyze textual overlap.

Built using Streamlit, TextGuard provides an interactive dashboard
with sentence-level analysis, keyword comparison, similarity
visualizations, downloadable reports, and SQLite-based history
management.

> **Disclaimer:** TextGuard is a document similarity screening tool.
> Similarity scores alone do not establish plagiarism.

---

## 📌 Project Overview

Document similarity analysis is an important application of
Natural Language Processing.

TextGuard helps users compare two documents, measure textual
similarity, identify repeated sentences, and explore matching
keywords and phrases.

The system provides multiple similarity metrics to help users
understand how closely two documents match.

### Project Objectives

- Develop an NLP-based document comparison application.
- Measure document similarity using multiple techniques.
- Detect exact duplicate and potentially similar sentences.
- Analyze common words, keywords, and word sequences.
- Visualize document similarity through interactive charts.
- Generate downloadable comparison reports.
- Store comparison history using SQLite.
- Provide historical analytics and similarity trends.

---

## ✨ Key Features

### 1. Document Input

- Enter two documents manually.
- Upload TXT files.
- Upload PDF files.
- Preview uploaded document content.
- Compare documents using an interactive interface.

### 2. NLP Preprocessing

- Convert text to lowercase.
- Remove punctuation.
- Tokenize text into words.
- Optional English stopword removal.
- Optional Porter stemming.
- Configure minimum sentence length.

### 3. Document Similarity Analysis

TextGuard implements three similarity techniques:

**TF-IDF Cosine Similarity**

Measures similarity between document vectors using
Term Frequency-Inverse Document Frequency.

**Jaccard Similarity**

Measures the overlap between unique word sets.

**Bigram Similarity**

Measures overlap between consecutive two-word sequences.

The application displays similarity scores as percentages
and classifies TF-IDF similarity into three levels:

| Similarity Score | Classification |
|------------------|----------------|
| 70% - 100% | High |
| 40% - below 70% | Moderate |
| Below 40% | Low |

### 4. Sentence-Level Similarity

- Split documents into sentences.
- Compare sentences using TF-IDF cosine similarity.
- Find the closest matching sentence in Document 2.
- Configure the sentence similarity threshold.
- Detect exact duplicate sentences.
- Sort sentence matches by similarity.
- Search matching sentences by keyword.
- Highlight potentially similar sentences.
- Export sentence comparison results as CSV.

### 5. Keyword and Word Analysis

- Identify common keywords.
- Identify unique keywords.
- Calculate word frequencies.
- Compare word frequencies between documents.
- Identify common bigrams.
- Find the longest common consecutive word sequence.
- Filter word-level comparison results.
- Configure the number of displayed top keywords.

### 6. Data Visualizations

- Overall similarity comparison chart.
- Document statistics comparison.
- Sentence similarity distribution chart.
- Sentence similarity heatmap.
- Top keyword frequency charts.
- Bidirectional sentence coverage chart.
- Historical similarity trend chart.
- Historical risk distribution chart.
- Historical similarity level distribution chart.

### 7. Similarity Screening Summary

The application provides:

- Overall document similarity.
- Sentence similarity coverage.
- Number of potentially similar sentences.
- Exact duplicate sentence count.
- Percentage of flagged sentences.
- Potential similarity risk classification.

The screening summary is based on textual similarity
and requires human interpretation.

### 8. Downloadable Reports

TextGuard supports:

| Report | Format |
|--------|--------|
| Document comparison report | TXT |
| Structured analysis report | JSON |
| Formatted similarity report | HTML |
| Sentence similarity results | CSV |
| Similarity heatmap | PNG |
| Comparison history | CSV |
| Comparison history | JSON |

### 9. SQLite Comparison History

- Automatically save completed comparisons.
- Store similarity scores and document statistics.
- Search saved comparisons by ID.
- Filter comparisons by similarity level.
- Filter comparisons by risk level.
- Delete individual comparison records.
- Export complete comparison history.
- Clear history with confirmation.

### 10. History Analytics Dashboard

The history analytics dashboard displays:

- Total saved comparisons.
- Average TF-IDF similarity.
- Highest similarity score.
- Lowest similarity score.
- Average Jaccard similarity.
- Average Bigram similarity.
- Total potentially similar sentences.
- Similarity trends across comparisons.
- Similarity level distribution.
- Risk level distribution.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| Scikit-learn | TF-IDF and cosine similarity |
| NLTK | Porter stemming |
| Pandas | Data processing and tables |
| Matplotlib | Similarity heatmap |
| PyPDF | PDF text extraction |
| SQLite | Local comparison history |
| Git | Version control |
| GitHub | Source code hosting |

---

## 📂 Project Structure

```text
TextGuard/
│
├── app.py
│   └── Streamlit user interface
│
├── nlp_utils.py
│   └── NLP preprocessing, similarity algorithms,
│       keyword analysis, and report generation
│
├── database.py
│   └── SQLite database operations,
│       history management, and analytics
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Files excluded from Git
│
└── textguard.db
    └── SQLite database generated automatically
```

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git (optional)
- Internet connection for installing dependencies

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/TextGuard-NLP-Document-Similarity.git
```

Replace `YOUR_USERNAME` with your GitHub username
and update the repository name if necessary.

### Step 2: Navigate to the Project

```bash
cd TextGuard-NLP-Document-Similarity
```

### Step 3: Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python -m streamlit run app.py
```

### Step 6: Open the Application

Streamlit normally starts the application at:

```text
http://localhost:8501
```

Open this address in your browser if it does not
open automatically.

---

## 🚀 How to Use TextGuard

### Step 1: Provide Documents

Choose one of the following input methods:

- Enter Text
- Upload Documents

Provide both documents before starting the comparison.

### Step 2: Configure NLP Settings

Choose whether to:

- Remove stopwords.
- Apply stemming.
- Adjust the sentence similarity threshold.
- Set the minimum words per sentence.
- Configure the number of top keywords.

### Step 3: Compare Documents

Click:

**Compare Documents**

The application calculates document similarity
and saves the comparison summary to SQLite.

### Step 4: Review Results

Explore the five analysis tabs:

**Overview**

View similarity metrics, document statistics,
longest matching sequence, risk summary, and coverage.

**Sentence Analysis**

View exact duplicates, sentence matches,
similarity scores, search, sorting, and highlighting.

**Visualizations**

Explore similarity distributions, sentence
heatmaps, and keyword frequency charts.

**Keywords**

Review common keywords, unique keywords,
common bigrams, and word frequency comparisons.

**Reports**

Download analysis reports in TXT, JSON,
and HTML formats.

### Step 5: Explore Comparison History

Scroll to the Saved Comparison History section.

You can:

- View previous comparisons.
- Search by comparison ID.
- Apply similarity and risk filters.
- Delete individual records.
- Export history.
- Clear all history after confirmation.

### Step 6: Review History Analytics

Explore historical similarity trends,
average similarity metrics, and distributions.

---

## 🧠 NLP Techniques Explained

### 1. Text Preprocessing

Text preprocessing prepares documents for analysis.

The application performs:

1. Lowercasing
2. Punctuation removal
3. Word tokenization
4. Optional stopword removal
5. Optional stemming

Example:

**Original Text:**

```text
Students are Learning Machine Learning!
```

**After Lowercasing and Punctuation Removal:**

```text
students are learning machine learning
```

**After Optional Stemming:**

```text
student are learn machin learn
```

### 2. TF-IDF

TF-IDF stands for:

**Term Frequency-Inverse Document Frequency**

It represents documents using numerical vectors
based on the importance of words.

TF-IDF helps reduce the influence of words that
occur frequently across documents.

### 3. Cosine Similarity

Cosine similarity measures the angle between
two document vectors.

Formula:

```text
Cosine Similarity = (A · B) / (||A|| × ||B||)
```

Where:

- A = Vector representation of Document 1
- B = Vector representation of Document 2
- A · B = Dot product
- ||A|| = Magnitude of vector A
- ||B|| = Magnitude of vector B

The application converts the result into a percentage.

### 4. Jaccard Similarity

Jaccard similarity compares two sets.

Formula:

```text
Jaccard Similarity = |A ∩ B| / |A ∪ B|
```

Example:

```text
Document 1: machine learning python

Document 2: machine learning java
```

Common words:

```text
machine, learning
```

Unique words across both documents:

```text
machine, learning, python, java
```

Jaccard similarity:

```text
2 / 4 × 100 = 50%
```

### 5. Bigram Similarity

A bigram is a sequence of two consecutive words.

Example:

```text
machine learning is powerful
```

Bigrams:

```text
machine learning
learning is
is powerful
```

TextGuard compares the sets of bigrams
in both documents.

### 6. Sentence Similarity

TextGuard splits documents into sentences.

Each sentence from Document 1 is compared
against sentences from Document 2.

The application identifies the highest-scoring
match for each sentence.

### 7. Longest Common Word Sequence

The application uses dynamic programming
to find the longest consecutive sequence of
matching words between two documents.

This helps identify repeated phrases.

---

## 📊 Example Comparison

**Document 1**

```text
Machine learning is a branch of artificial intelligence.
Python is widely used for machine learning.
```

**Document 2**

```text
Machine learning is a branch of artificial intelligence.
Java is a popular programming language.
```

### Expected Observations

- The first sentence is an exact match.
- Both documents contain common NLP-related words.
- TF-IDF similarity indicates textual overlap.
- The sentence analysis identifies the matching sentence.
- The heatmap highlights the corresponding sentence pair.
- The comparison summary is stored in SQLite.

Exact overall scores depend on the selected
preprocessing settings.

---

## 🗄️ Database Design

TextGuard uses SQLite to store comparison history.

### Table: comparisons

| Column | Description |
|--------|-------------|
| id | Unique comparison identifier |
| created_at | UTC timestamp |
| document1_words | Word count of Document 1 |
| document2_words | Word count of Document 2 |
| tfidf | TF-IDF similarity score |
| jaccard | Jaccard similarity score |
| bigram | Bigram similarity score |
| similarity_level | High, Moderate, or Low |
| similar_sentences | Number of similar sentences |
| risk_level | Potential similarity risk |
| threshold | Selected sentence threshold |
| min_words | Minimum words per sentence |

The database is initialized automatically
when the application starts.

---

## 🔒 Privacy and Data Handling

- Text analysis runs within the Python application.
- SQLite comparison history is stored locally.
- History records contain comparison metrics and settings.
- The current implementation does not store full
  document contents in the history database.
- Uploaded documents are processed for analysis.
- Users can delete individual history records
  or clear the complete history.

When deploying the application online, review
the hosting platform's storage and privacy policies.

---

## ⚠️ Limitations

TextGuard primarily measures lexical similarity.

Current limitations include:

- Limited understanding of semantic meaning.
- May not detect advanced paraphrasing.
- May miss similarity when different words express
  the same meaning.
- PDF extraction depends on whether the PDF
  contains extractable text.
- Scanned PDFs without embedded text may not work.
- Sentence segmentation is rule-based.
- Long documents may take more processing time.
- Similarity scores cannot independently establish plagiarism.
- Local SQLite storage may not persist on all cloud platforms.

---

## 🔮 Future Enhancements

Potential improvements include:

- DOCX file support.
- Semantic similarity using Sentence-BERT.
- Multiple document comparison.
- Improved sentence segmentation.
- More advanced plagiarism screening.
- User authentication.
- Cloud database integration.
- Batch document processing.
- Automated unit testing.
- Performance optimization for large documents.

---

## 📈 Development Milestones

TextGuard was developed incrementally
through 50 Git commits.

Major development stages included:

1. Project initialization and basic interface.
2. Text preprocessing and TF-IDF similarity.
3. File upload and document statistics.
4. Sentence-level similarity and duplicate detection.
5. Keyword analysis and visualizations.
6. Multiple similarity methods.
7. Downloadable reports.
8. Word-level comparison.
9. SQLite comparison history.
10. Historical analytics and final documentation.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your updates.
5. Open a pull request.

---

## 📄 License

No license has been assigned to this repository.

If you want to allow reuse under a standard
open-source license, add a LICENSE file.

---

## ⭐ Support

If you find this project useful,
consider starring the repository.

---

### TextGuard — NLP Document Similarity Analyzer

**Built with Python, Streamlit, Scikit-learn, NLTK, and SQLite.**
