"""CLI application for YouTube transcript processing and markdown generation"""
import json
import argparse
import sys
from pathlib import Path

from src.ingest.youtube_transcript import fetch_yt_transcript
from src.preprocess.cleaner import TranscriptCleaner
from src.preprocess.chunker import chunk_transcript, try_import_tiktoken
from src.pipeline.translate_edit_structure import translate_edit_structure, tes_async_wrapper
from src.output.markdown_writer import MarkdownWriter
from src.utils.logging import get_logger

logger = get_logger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process YouTube transcripts and generate structured markdown output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        %(prog)s -v F2Wf0qKZ5fg -t "Video Title"
        %(prog)s -v F2Wf0qKZ5fg -t "Video Title" -o ./my_output_markdown --sync
        """
    )

    # required arguments
    parser.add_argument(
        "-v", "--video-id",
        required=True,
        help="YouTube video ID (e.g. 'F2Wf0qKZ5fg')"
    )

    parser.add_argument(
        "-t", "--title",
        required=True,
        help="Video Title for processing"
    )

    # optional argument
    parser.add_argument(
        "-o", "--output-dir",
        default="output_markdown",
        help="Directory for markdown output (default: output_markdown)"
    )

    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous processing instead of async (default: async)"
    )

    return parser.parse_args()

def process_transcript(video_id, title, use_sync=False):
    """
    Process YouTube transcript through the entire pipeline

    Args:
        video_id: Youtube video id
        title: video title
        use_sync: whether to use synchronous processing

    Returns:
        dict: processed and structured data
    """
    # fetch transcript 
    snippet_list, video_metadata = fetch_yt_transcript(video_id=video_id, title=title)
    logger.info(f"Retrieved {len(snippet_list)} transcript snippets")

    # clean transcript
    txt_cleaner = TranscriptCleaner()
    cleaned_snippet_list = [txt_cleaner.clean_text(snippet) for snippet in snippet_list]
    logger.info("Snippet cleaning done")

    # concatenate to big string
    cleaned_text = " ".join(cleaned_snippet_list)
    # chunk transcript
    chunked = chunk_transcript(cleaned_text, tokenizer_fn=try_import_tiktoken())
    logger.info(f"Processed to {len(chunked)} chunks")

    # processing with translate, edit and structure pipeline
    if use_sync:
        logger.info("Using synchronous processing...")
        output = translate_edit_structure(chunked_txt=chunked, txt_metadata=video_metadata)
    else:
        logger.info("Using async processing...")
        output = tes_async_wrapper(chunked_txt=chunked, txt_metadata=video_metadata)

    return output

def write_output_to_file_and_exit(output, filename="error_output.txt"):
    """
    Write non-dictionary output to a text file and exit gracefully

    Args:
        output: The output data to write (any type)
        filename: name of the file to write to
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            if isinstance(output, (list, tuple)):
                f.write(json.dumps(output, indent=4, default=str, ensure_ascii=False))
            elif isinstance(output, str):
                f.write(output)
            else:
                f.write(str(output))

        logger.warning(f"Output was not a dictionary. Written to {filename}")

    except Exception as ex:
        logger.error(f"Failed to write output to file: {str(ex)}")
    finally:
        sys.exit(1)

def main():
    """Main CLI entry point"""
    args = parse_arguments()

    try:
        # process transcript
        print(f"Processing video: {args.title}")
        print(f"\nVideo ID: {args.video_id}\n")

        output = process_transcript(
            video_id=args.video_id,
            title=args.title,
            use_sync=args.sync
        )

        if not isinstance(output, dict):
            write_output_to_file_and_exit(output)

        # generate markdown
        print(f"\nGenerating markdown output...")
        writer = MarkdownWriter(output_dir=args.output_dir)
        md_path = writer.write_content(structured_data=output)

        logger.info(f"Markdown file created: {md_path}")
        
    except KeyboardInterrupt:
        print("\n\n ⚠️  Process interrupted by user")
        sys.exit(130)
    except Exception as ex:
        logger.error(f"Error during processing: {str(ex)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
    