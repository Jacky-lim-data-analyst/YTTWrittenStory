# yt-story

**yt-story** is an automated pipeline that transforms English YouTube video transcripts into high-quality, structured Chinese Markdown articles. It leverages Google's Gemini models to transcribe, translate, polish, and structure content, making it ready for publication or personal archiving.

## Features

- **Ingestion**: Automatically fetches transcripts from YouTube videos using video IDs.
- **Preprocessing**: Cleans and chunks raw transcript text for optimal LLM processing.
- **Translation**: Translates English content into natural, fluent Simplified Chinese using Gemini.
- **Editing**: Polishes the translated text to enhance readability, flow, and engagement (removing "translationese").
- **Structuring**: Generates metadata including catchy titles, concise descriptions, and relevant tags.
- **Output**: Exports the final result as a formatted Markdown file.

## Prerequisites

- **Python**: version 3.12 or higher.
- **Gemini API Key**: A valid API key from Google AI Studio.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd yt-story
    ```

2.  **Set up the environment:**
    This project uses `uv` for dependency management, but can also be installed via `pip`.

    *   **Using `uv` (Recommended):**
        ```bash
        uv sync
        ```

    *   **Using `pip`:**
        ```bash
        pip install -r pyproject.toml
        # Or manually install dependencies:
        # pip install google-genai ollama python-dotenv pyyaml tiktoken youtube-transcript-api
        ```

3.  **Configuration:**
    Create a `.env` file in the project root and add your Gemini API key:
    ```env
    GEMINI_API_KEY=your_actual_api_key_here
    ```

## Usage

Currently, the project is set up to run via the `main.py` script.

1.  Open `main.py` and update the `video_id` and `title` variables with the YouTube video you want to process:
    ```python
    video_id = "<YoutubeVideoID>" # Replace with your YouTube Video ID
    title = "<video_title>" # Optional title context
    ```

2.  Run the script:
    ```bash
    python main.py
    ```

3.  **Check the output:**
    The generated Markdown file will be saved in the `output/` directory, e.g., `output/20260102_095604_Title.md`.

## Configuration

You can customize the behavior of the LLM agents in `config/gemini_settings.yaml`:

-   **translate_agent**: Settings for the English-to-Chinese translation step.
-   **edit_agent**: Settings for the polishing and editing step.
-   **structurer_agent**: Settings for extracting title, description, and tags.

Each agent allows configuration of:
-   `system_prompt`: Instructions for the model.
-   `gemini_settings`: Model name (e.g., `gemini-2.5-flash`), temperature, and top_p.

## Project Structure

```
yt-story/
├── config/                 # Configuration files (LLM settings)
├── src/
│   ├── ingest/             # YouTube transcript fetching
│   ├── preprocess/         # Text cleaning and chunking
│   ├── llm/                # Gemini client and prompts
│   ├── pipeline/           # Orchestration logic (Translate -> Edit -> Structure)
│   ├── output/             # Markdown file writing
│   └── utils/              # Logging and helpers
├── output/                 # Generated Markdown files
├── main.py                 # Entry point script
├── pyproject.toml          # Project dependencies
└── .env                    # Environment variables (API Key)
```

