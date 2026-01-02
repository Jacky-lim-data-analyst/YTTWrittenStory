"""
Utility function for writing structured content to markdown files
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

class MarkdownWriter:
    """Write structured content to markdown files with proper formatting"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the markdown writer

        Args:
            output_dir: Directory to save markdown files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_content(
        self, 
        structured_data: Dict[str, str],
        filename: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Write structured content to a markdown file

        Args:
            structured_data: Dictionary with 'title', 'description', 'tags', 'body'
            filename: Optional custom filename (without extension)
            metadata: Optional metadata to include (video_url, source, etc.)
        """
        # generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # use title as part of filename
            title_slug = self._slugify(structured_data.get('title', 'untitled'))
            filename = f"{timestamp}_{title_slug}"

        filepath = self.output_dir / f"{filename}.md"

        # build markdown content
        md_content = self._build_markdown(structured_data, metadata)

        # write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return str(filepath)

    def _build_markdown(self, data: Dict[str, str], metadata: Optional[Dict] = None):
        """Build the markdown content from structured data"""
        lines = []

        # add YAML frontmatter if metadata provided
        if metadata:
            lines.append("---")
            lines.append(f"title: \"{data.get('title', 'Untitled')}\"")
            lines.append(f"date: {datetime.now().isoformat()}")

            if 'video_url' in metadata:
                lines.append(f"source: {metadata['video_url']}")
            if 'video_id' in metadata:
                lines.append(f"video_id: {metadata['video_id']}")
            if 'original_language' in metadata:
                lines.append(f"original_language: {metadata['original_language']}")

            # add tags from structured data
            tags = data.get('tags', [])
            if tags:
                tags_str = ', '.join(tags)
                lines.append(f"tags: [{tags_str}]")

            lines.append("---")
            lines.append("")

        # add title
        lines.append(f"# {data.get('title', 'Untitled')}")
        lines.append("")

        # add description as blockquote
        description = data.get('description', '')
        if description:
            lines.append(f"> {description}")
            lines.append("")

        # add tags section (if not in frontmatter)
        if not metadata:
            tags = data.get('tags', [])
            if tags:
                tags_md = ' '.join([f"`{tag}`" for tag in tags])
                lines.append(f"**标签:** {tags_md}")
                lines.append("")

        # add horizontal rule
        lines.append("---")
        lines.append("")

        # add body content
        body = data.get('body', '')
        if body:
            lines.append(body)

        # add footer with metadata if available
        if metadata:
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("### 来源信息")
            lines.append("")

            if 'video_url' in metadata:
                lines.append(f"- **原视频:** {metadata['video_url']}")
            if 'video_title' in metadata:
                lines.append(f"- **原标题:** {metadata['video_title']}")
            if 'channel_name' in metadata:
                lines.append(f"- **频道:** {metadata['channel_name']}")

            lines.append(f"- **处理时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return '\n'.join(lines)

    def _slugify(self, text: str, max_length: int = 50) -> str:
        """Convert text to a safe filename slug"""
        # remove or replace special characters
        SPECIAL_CHARS = ' /\\:*?"<>|'
        slug = text.lower()
        slug = slug.translate(str.maketrans({c: '_' for c in SPECIAL_CHARS}))

        # 1. Only allow alphanumeric characters
        # 2. Limit length
        # 3. Remove trailing underscores
        slug = ''.join(c for c in slug if c.isalnum() or c in ('_', '-'))

        if len(slug) > max_length:
            slug = slug[:max_length]

        slug = slug.rstrip('_')
        
        return slug or 'untitled'
    
if __name__ == "__main__":
#     sample_data = {
#         "title": "机器学习入门指南",
#         "description": "从零开始学习机器学习的核心概念,通过实际案例掌握关键技术。",
#         "tags": ["机器学习", "Python", "数据分析", "入门教程", "实战案例"],
#         "body": """机器学习是人工智能的重要分支,它的神奇之处在于:计算机可以自主学习,而不需要我们为每个场景编写详细的指令。

# ## 什么是机器学习?

# 简单来说,机器学习就是让计算机通过数据来学习规律,然后用这些规律来做预测或决策。就像我们人类通过经验学习一样,机器也可以从大量数据中总结出模式。

# ## 核心概念

# 机器学习主要包含三个关键要素:

# 1. **数据** - 机器学习的原料
# 2. **算法** - 学习的方法
# 3. **模型** - 学习的成果

# 掌握这三个要素,你就迈出了机器学习的第一步。

# ## 实际应用

# 机器学习已经渗透到我们生活的方方面面:推荐系统、语音识别、图像识别等等。了解这些应用背后的原理,将帮助你更好地理解这个技术的潜力。"""
#     }

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    print(str(PROJECT_ROOT))
    json_filepath = PROJECT_ROOT / "response.json"

    with open(json_filepath, 'r', encoding="utf-8") as f:
        sample_data = json.load(f)

    print(sample_data)

    # initialize writer
    writer = MarkdownWriter(output_dir="output")
    
    print("Writing markdown file...")
    md_path = writer.write_content(structured_data=sample_data)
    print(f"Markdown file created: {md_path}")
    