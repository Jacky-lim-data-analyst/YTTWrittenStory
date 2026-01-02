import re
# import json
import unicodedata
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

class CleaningLevel(Enum):
    """Define different levels of cleaning aggressiveness"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class TranscriptCleaner:
    """Clean and normalize YouTube transcripts for better readability 
    (configurable cleaning options)"""
    # configuration flags
    remove_fillers: bool = True
    remove_sound_effects: bool = True
    normalize_whitespace: bool = True
    fix_punctuation: bool = True
    fix_casing: bool = True
    remove_repetitions: bool = True
    remove_timestamps: bool = True
    clean_numbers: bool = False
    remove_urls_emails: bool = True
    max_line_length: Optional[int] = None
    deduplicate_lines: bool = True

    # customizable patterns and replacements
    filler_words: List[str] = field(default_factory=lambda: [
        "um", "uh", "ah", "er", "hm", "hmm", "you know", "I mean",
        "basically", "sort of", "kinda"
    ])

    sound_effects: List[str] = field(default_factory=lambda: [
        r"\[.*?\]",  # square brackets (e.g. [music], [applause])
        r"\{.*?\}",
        r"<.*?>"
    ])

    timestamp_patterns: List[str] = field(default_factory=lambda: [
        r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{3})?",  # HH:MM:SS or MM:SS
        r"\d{1,2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2},\d{3}",  # SRT format
    ])

    def __post_init__(self):
        """Compile regex patterns after initialization"""
        # compile filler words pattern
        self.filler_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(f) for f in self.filler_words) + r')\b',
            re.IGNORECASE
        )

        # compile sound effects pattern
        self.sound_effects_pattern = re.compile(
            '|'.join(self.sound_effects),
            re.IGNORECASE
        )

        # compile timestamp patterns
        self.timestamp_pattern = re.compile(
            '|'.join(self.timestamp_patterns)
        )

        # URL and email pattern
        self.url_email_pattern = re.compile(
            r'(?:https?://|www\.)\S+|[\w\.-]+@[\w\.-]+\.\w+'
        )

        # Multiple spaces/tabs/newlines
        self.whitespace_pattern = re.compile(r'\s+')

        # multiple punctuations (e.g. ???, ...)
        self.multi_punct_pattern = re.compile(r'([!?.])\1+')

        # repetition pattern (same word 3+ times)
        self.repetition_pattern = re.compile(r'\b(\w+)(?:\s+\1){2,}\b', re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """Main cleaning method"""
        if not text.strip():
            print("Warning: empty string")
            return text
        
        # original_text = text
        text = self._preprocess(text)

        # apply cleaning steps
        if self.remove_sound_effects:
            text = self._remove_sound_effects(text)

        if self.remove_timestamps:
            text = self._remove_timestamps(text)

        if self.remove_urls_emails:
            text = self._remove_urls_emails(text)

        if self.remove_fillers:
            text = self._remove_filler_words(text)

        if self.remove_repetitions:
            text = self._remove_repetitions(text)

        if self.fix_punctuation:
            text = self._fix_punctuation(text)

        if self.fix_casing:
            text = self._fixing_casing(text)

        if self.clean_numbers:
            text = self._clean_numbers(text)

        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.max_line_length:
            text = self._limit_line_length(text)

        if self.deduplicate_lines:
            text = self._deduplicate_lines(text)

        return text.strip()

    def _preprocess(self, text: str) -> str:
        """Initial text processing"""
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def _remove_sound_effects(self, text: str) -> str:
        """Remove sound effects and bracketed content"""
        return self.sound_effects_pattern.sub(' ', text)
    
    def _remove_timestamps(self, text: str) -> str:
        return self.timestamp_pattern.sub(' ', text)
    
    def _remove_filler_words(self, text: str) -> str:
        return self.filler_pattern.sub('', text)
    
    def _remove_urls_emails(self, text: str) -> str:
        return self.url_email_pattern.sub('', text)
    
    def _remove_repetitions(self, text: str) -> str:
        """Remove repeated words (3+ times)"""
        def replace_repetition(match):
            return match.group(1)   # keep only 1 instance
        
        return self.repetition_pattern.sub(replace_repetition, text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces"""
        text = self.whitespace_pattern.sub(' ', text)

        # remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # ensure space after punctuation if followed by letter
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)

        return text.strip()
    
    def _fix_punctuation(self, text: str) -> str:
        """Fix punctuation issues"""
        # replace multiple punctuations with single
        text = self.multi_punct_pattern.sub(r'\1', text)

        # fix missing spaces afetr commas
        text = re.sub(r',(\S)', r', \1', text)

        # fix missing spaces before opening quotes
        text = re.sub(r'(\w)"(\w)', r'\1 "\2', text)

        # remove spaces inside quotes
        text = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', text)

        return text
    
    def _fixing_casing(self, text: str) -> str:
        """Fix capitalization issues"""
        # capitalize first letter of each sentence
        sentences = re.split(r'([.!?]\s+)', text)
        result = ''

        for i, sentence in enumerate(sentences):
            if i % 2 == 0:   # actual sentence content
                if sentence.strip():
                    # capitalize first character
                    sentence = sentence.strip()
                    if sentence:
                        sentence = sentence[0].upper() + sentence[1:]

            result += sentence

        # fix "i" to "I"
        result = re.sub(r'\bi\b', 'I', result)

        return result
    
    def _clean_numbers(self, text: str) -> str:
        """Clean and format numbers"""
        text = re.sub(r'\b\d+\b', '', text)
        return text
    
    def _limit_line_length(self, text: str) -> str:
        """Split long lines"""
        lines = text.split('\n')
        result = []

        for line in lines:
            if len(line) > self.max_line_length:
                # split line at word boundaries
                words = line.split()
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 > self.max_line_length:
                        result.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                
                if current_line:
                    result.append(' '.join(current_line))
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _deduplicate_lines(self, text: str) -> str:
        """Remove duplicate consecutive lines"""
        lines = text.split('\n')
        cleaned_lines = []

        for i, line in enumerate(lines):
            if i == 0 or line.strip() != lines[i - 1].strip():
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)
    
    def print_stats(self, original_text: str, cleaned_text: str) -> None:
        """Print cleaning stats"""
        orig_words = len(original_text.split())
        cleaned_words = len(original_text.split())
        orig_chars = len(original_text)
        cleaned_chars = len(cleaned_text)

        print(f"\n=== Cleaning Statistics ===")
        print(f"Original: {orig_words} words, {orig_chars} characters")
        print(f"Cleaned: {cleaned_words} words, {cleaned_chars} characters")
        print(f"Reduction: {orig_words - cleaned_words} words ({((orig_words - cleaned_words)/orig_words*100):.1f}%)")
        print(f"Character reduction: {orig_chars - cleaned_chars} ({((orig_chars - cleaned_chars)/orig_chars*100):.1f}%)")

    @classmethod
    def create_preset(cls, level: CleaningLevel) -> 'TranscriptCleaner':
        """Create a cleaner with preset configurations"""
        if level == CleaningLevel.MINIMAL:
            return cls(
                remove_sound_effects=True,
                remove_timestamps=True,
                remove_fillers=False,
                remove_repetitions=False,
                remove_urls_emails=True,
                normalize_whitespace=True,
                fix_punctuation=True,
                fix_casing=True,
            )
        elif level == CleaningLevel.MODERATE:
            return cls(
                remove_sound_effects=True,
                remove_timestamps=True,
                remove_fillers=True,
                remove_repetitions=False,
                remove_urls_emails=True,
                normalize_whitespace=True,
                fix_punctuation=True,
                fix_casing=True,
                clean_numbers=False,
                max_line_length=100
            )
        elif level == CleaningLevel.AGGRESSIVE:
            return cls(
                remove_sound_effects=True,
                remove_timestamps=True,
                remove_fillers=True,
                remove_repetitions=True,
                remove_urls_emails=True,
                normalize_whitespace=True,
                fix_punctuation=True,
                fix_casing=True,
                clean_numbers=True,
                deduplicate_lines=True,
                max_line_length=80
            )
        else:
            return cls()

def fix_casing(text: str) -> str:
    # capitalize first letter of each sentence
    sentences = re.split(r'([.!?]\s+)', text)
    # print(sentences)
    result = ''

    for i, sentence in enumerate(sentences):
        if i % 2 == 0:   # actual sentence content
            # print(sentence)
            if sentence.strip():
                # capitalize first character
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]

        result += sentence

    # fix "i" to "I"
    result = re.sub(r'\bi\b', 'I', result)

    return result

def clean_caption_example():
    caption = """
    [Music]
    00:01:23 Welcome everyone um to today's tutorial.
    
    So, like, you know, we're going to um learn Python.
    [Applause]
    (laughter)
    Visit our site: https://example.com for more info.
    
    Umm... let's start. So so so anyway... right?
    """
    print("Original caption:")
    print(caption)
    print("\n" + "="*50 + "\n")

    # create cleaner with custom setting
    cleaner = TranscriptCleaner()

    cleaned = cleaner.clean_text(caption)

    print("Cleaned caption:")
    print(cleaned)

if __name__ == "__main__":
    # s = "how many seconds the connection should wait before raising an OperationalError when a table is locked. If another i connection opens a transaction to modify a table, that table will be locked until the transaction is committed. Default five seconds. go."

    # cleaned_str = fix_casing(s)
    # print(cleaned_str)
    clean_caption_example()
