# Study Notes Summarizer & Quiz Generator

A Streamlit app that turns your study notes or PDFs into a bullet point summary and an interactive multiple-choice quiz, using the Groq API.

**Live demo:** YOUR-STREAMLIT-LINK-HERE

## Screenshots
<img width="950" height="437" alt="image" src="https://github.com/user-attachments/assets/8bc8d2f8-da4a-45aa-b145-d17cb062487b" />
<img width="953" height="420" alt="image" src="https://github.com/user-attachments/assets/f8da1a98-e31f-47b6-9398-6bc021530355" />
<img width="953" height="428" alt="image" src="https://github.com/user-attachments/assets/c78604bd-f916-4b84-b8cf-ab7772aa90c3" />


## Features

- Upload a PDF, paste notes, or use both together
- Handles long documents by splitting them into overlapping chunks
- Chunked (map-reduce) summarization: each chunk is summarized, then combined into a final 5-bullet summary
- Generates 3 to 10 multiple-choice questions as structured JSON
- Validates the JSON (4 options, valid answer index) and retries if the output is malformed
- Interactive quiz: answers stay hidden until you submit, then you get a score with explanations
- Download the summary and quiz (with answers) as a text file
- Friendly error messages for a missing API key, rate limits, and connection problems

## How it works

1. Text is extracted from the PDF with `pypdf` and combined with any pasted notes.
2. The text is split into chunks of about 3000 characters with a 200-character overlap.
3. Each chunk is summarized, and the partial summaries are combined into a final summary.
4. The quiz is generated from the condensed summaries, so it covers the whole document, and returned as JSON.
5. The JSON is validated before the quiz is shown.

Notes are wrapped in `<notes>` tags and the system prompt tells the model to treat them as data, as a basic defense against prompt injection.

## Tech stack

Python, Streamlit, Groq API (`openai/gpt-oss-120b`), pypdf, python-dotenv

## Run it locally

1. Clone the repo:
   ```
   git clone https://github.com/YOUR-USERNAME/study-notes-quiz-generator.git
   cd study-notes-quiz-generator
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project folder:
   ```
   GROQ_API_KEY=your_key_here
   ```
   You can get a free key from the Groq console.
4. Start the app:
   ```
   streamlit run app.py
   ```

## Limitations

- Scanned PDFs (images of text) are not supported because there is no OCR yet
- At most 15 chunks (about 45,000 characters) are processed per run
- The quiz is built from summaries, so it can miss fine details
- Summaries and quiz answers come from an LLM and can contain mistakes, so check them against your notes

## Possible improvements

- OCR for scanned PDFs
- Shuffle the answer positions in code
- Save quiz history and scores
- Tests for chunking and JSON validation
