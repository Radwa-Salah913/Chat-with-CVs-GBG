
import os
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter
import pymupdf4llm  
root_path = os.path.dirname(os.path.abspath(__file__))
dir_path = os.path.join(root_path,"assets","temp_uploads","Radwa Salah AI Engineer.docx")



markdown_text = pymupdf4llm.to_markdown(dir_path)
def is_heading(line):
    stripped = line.strip()
    
    # فاضي
    if not stripped:
        return False
    
    # طويل جدًا؟ يبقى مش header
    if len(stripped) > 60:
        return False
    
    # فيه نقطة في الآخر؟ غالبًا paragraph
    if stripped.endswith('.'):
        return False
    
    # عدد الكلمات كبير؟ يبقى مش header
    if len(stripped.split()) > 6:
        return False
    
    # فيه أرقام كتير؟ غالبًا مش header
    if sum(c.isdigit() for c in stripped) > 3:
        return False
    
    # شكل Title Case أو ALL CAPS
    if stripped.istitle() or stripped.isupper():
        return True
    
    return False

def auto_convert_to_markdown(text):
    lines = text.split("\n")
    new_lines = []
    
    # أول سطر نخليه عنوان رئيسي
    first_line_added = False

    for line in lines:
        if not first_line_added and line.strip():
            new_lines.append(f"# {line.strip()}")
            first_line_added = True
            continue
        
        if is_heading(line):
            new_lines.append(f"\n## {line.strip()}\n")
        else:
            new_lines.append(line)

    return "\n".join(new_lines)



"""headers_to_split_on = [
    ("#", "title"),
    ("##", "section"),
    ("###", "subsection"),
]"""

markdown_text = auto_convert_to_markdown(markdown_text)

headers_to_split_on = [
    ("#", "candidate"),
    ("##", "section"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

docs = splitter.split_text(markdown_text)

for doc in docs:
    print(doc.metadata,"\n")
    print(doc.page_content,"\n")
    print("--------------------------------------------------------------------------------------\n")