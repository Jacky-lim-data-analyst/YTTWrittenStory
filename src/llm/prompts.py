
class PromptTemplates:
    """Templates for constructing complete prompts with context."""

    @staticmethod
    def build_translator_prompt(content: str) -> str:
        """Prompt for translation"""
        return f"Translate the following English text to Chinese:\n\n{content}"
    
    @staticmethod
    def build_editor_prompt(content: str) -> str:
        """Prompt for editing"""
        return f"Edit the following Chinese text to make it more engaging and readable:\n\n{content}"
    
    @staticmethod
    def build_structurer_prompt(content: str, video_title: str | None = None) -> str:
        """Prompt for structuring"""
        context = ""
        if video_title:
            context = f"\n\nOriginal video title: {video_title}\n"

        return f"Structure the following Chinese content into JSON format:{context}\n{content}"
    