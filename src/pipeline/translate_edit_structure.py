"""
Pipeline:
1. Expects chunked text from chunker.
2. Translate the chunked text from English to Chinese.
3. Edit (fix the grammar and enhance readability) chunks by chunks
4. All the text chunks are concatenated.
5. Input to structurer LLM for structured response
"""

from pathlib import Path
import yaml
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from ..llm.gemini_client import GeminiClient
from ..llm.prompts import PromptTemplates
from ..utils.logging import get_logger

logger = get_logger(__name__)

class TextForMarkdown(BaseModel):
    title: str
    description: str
    tags: list[str]

def get_gemini_config() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    gemini_config_filepath = project_root / "config" / "gemini_settings.yaml"

    with open(gemini_config_filepath, 'r', encoding='utf-8') as f:
        gemini_config = yaml.safe_load(f)
    
    logger.info("Gemini generation configuration loaded")
    return gemini_config

def translate_edit_structure(
    chunked_txt: list[dict],
    # structured_response: BaseModel,
    txt_metadata: dict | None = None,
):
    """The pipeline to orchestrate the translation, edit and structuring of Youtube transcript"""
    if not chunked_txt:
        raise ValueError("The chunked text must not be empty")
    
    logger.info(f"Received {len(chunked_txt)} chunks of text from chunker")
    # API key
    load_dotenv()
    api_key = os.environ["GEMINI_API_KEY"]
    if not api_key:
        logger.critical("GEMINI_API_KEY missing")
        raise Exception("GEMINI API KEY must be defined")
    
    # load gemini config settings
    gemini_config = get_gemini_config()
    
    print("Initializing GEMINI translator client ...")
    # part 1: translation
    translator_client = GeminiClient(
        api_key=api_key,
        model_name=gemini_config["translate_agent"]["gemini_settings"]["model_name"] or "gemini-2.5-flash",
        system_instruction=gemini_config["translate_agent"]["system_prompt"] or None,
    )
    logger.info("Translator client initialized")

    # loop through the chunked text
    responses = []
    for txt_dict in chunked_txt:
        prompt = PromptTemplates.build_translator_prompt(txt_dict["text"])
        response = translator_client.generate(
            prompt=prompt,
            temperature=gemini_config["translate_agent"]["gemini_settings"]["options"]["temperature"],
            top_p=gemini_config["translate_agent"]["gemini_settings"]["options"]["top_p"]
        )
        responses.append(response)
    translator_client.close()
    logger.info("Translator client closed")

    # Join the string list
    translate_out = " ".join(responses)

    # part 2: edit
    edit_client = GeminiClient(
        api_key=api_key,
        model_name=gemini_config["edit_agent"]["gemini_settings"]["model_name"] or "gemini-2.5-flash",
        system_instruction=gemini_config["edit_agent"]["system_prompt"] or None,
    )

    prompt = PromptTemplates.build_editor_prompt(translate_out)
    edit_out = edit_client.generate(
        prompt=prompt,
        temperature=gemini_config["edit_agent"]["gemini_settings"]["options"]["temperature"],
        top_p=gemini_config["edit_agent"]["gemini_settings"]["options"]["top_p"],
        thinking_mode=True
    )

    edit_client.close()

    # part 3: structure
    structure_client = GeminiClient(
        api_key=api_key,
        model_name=gemini_config["structurer_agent"]["gemini_settings"]["model_name"] or "gemini-2.5-flash",
        system_instruction=gemini_config["structurer_agent"]["system_prompt"] or None,
    )

    prompt = PromptTemplates.build_structurer_prompt(edit_out)
    structure_out = structure_client.generate(
        prompt=prompt,
        temperature=gemini_config["structurer_agent"]["gemini_settings"]["options"]["temperature"],
        top_p=gemini_config["structurer_agent"]["gemini_settings"]["options"]["top_p"],
        thinking_mode=True,
        response_schema=TextForMarkdown
    )

    if not isinstance(structure_out, BaseModel):
        logger.warning("The structuring pipeline does not return BaseModel")
        return edit_out
    
    output_dict = structure_out.model_dump()
    output_dict["body"] = edit_out

    if txt_metadata:
        output_dict.update(txt_metadata)

    return output_dict

# if __name__ == "__main__":
#     # load gemini config settings
#     gemini_config = get_gemini_config()

#     print(gemini_config["translate_agent"]["gemini_settings"]["model_name"])
#     print("\n")
#     print(gemini_config["translate_agent"]["gemini_settings"]["options"]["temperature"])
