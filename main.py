"""Entire workflow orchestration"""
import json
from src.ingest.youtube_transcript import fetch_yt_transcript
from src.preprocess.cleaner import TranscriptCleaner
from src.preprocess.chunker import chunk_transcript, try_import_tiktoken
from src.pipeline.translate_edit_structure import translate_edit_structure, tes_async_wrapper
from src.output.markdown_writer import MarkdownWriter
from src.utils.logging import get_logger

logger = get_logger(__name__)

video_id = "6WNwPIvXaj0"
title = "This bus rider RECORDED something EVIL..."

snippet_list, video_metadata = fetch_yt_transcript(video_id=video_id, title=title)
logger.info(f"Retrieved {len(snippet_list)} transcript snippets")
# print(video_metadata)

txt_cleaner = TranscriptCleaner()
cleaned_snippet_list = [txt_cleaner.clean_text(snippet) for snippet in snippet_list]
logger.info("Snippet cleaning done")

# concatenate to big string
cleaned_text = " ".join(cleaned_snippet_list)
chunked = chunk_transcript(cleaned_text, tokenizer_fn=try_import_tiktoken())
logger.info(f"Processed to {len(chunked)} chunks")

# print(chunked)
# synchronous (sequential edit & structure)
# output = translate_edit_structure(chunked_txt=chunked, txt_metadata=video_metadata)

# async edit & structure
output = tes_async_wrapper(chunked_txt=chunked, txt_metadata=video_metadata)

if isinstance(output, dict):
    model_response_filename = "response.json"
    with open(model_response_filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)

    with open(model_response_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # initialize writer
    writer = MarkdownWriter(output_dir="output_markdown")
    
    print("Writing markdown file...")
    md_path = writer.write_content(structured_data=data)
    logger.info(f"Markdown file created: {md_path}")
else:
    logger.error("Something wrong with the edit or structure workflow")
