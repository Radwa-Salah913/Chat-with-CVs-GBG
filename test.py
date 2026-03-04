
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_unstructured import UnstructuredLoader

root_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(root_path,"assets","temp_uploads","MennatAllah Ibrahim Kamal Khalil AI .pdf")


"""loader = UnstructuredLoader(
    file_path=path,
    chunking_strategy="by_title",
    # كبري الرقم ده عشان يسحب السيكشن باللي فيه من غير ما يقطعه
    #max_characters=3000, 
    # اطلبي منه يدمج العناصر الصغيرة اللي تحت بعضها (زي الـ Bullet points)
    combine_under_n_chars=600, 
    # ده هيمنع إنه يعتبر كلمة Bold صغيرة سيكشن جديد لوحدها
    new_after_n_chars=5000, 
)

docs = loader.load()







    
for doc in docs:
    print("Metadata:", doc.metadata,"\n\n")
    print("Content:\n", doc.page_content)
    print(len(doc.page_content),"\n")
    print("-" * 50)"""
###############################################################
from unstructured.partition.pdf import partition_pdf

from langchain_core.documents import Document

elements = partition_pdf(path)
for el in elements:
    if type(el).__name__ == "Title":
        print("TITLE →", el.text)
        print("-" * 40)
"""sections = []
current_section = None

i = 0
while i < len(elements):

    element = elements[i]

    if element.category == "Title":

        # نحسب عدد العناصر اللي بعده لحد Title جديد
        content_block = []
        j = i + 1

        while j < len(elements) and elements[j].category != "Title":
            content_block.append(elements[j])
            j += 1

        # لو البلوك اللي بعده كبير → Section حقيقي
        if len(content_block) >= 3:
            if current_section:
                sections.append(current_section)

            current_section = {
                "title": element.text.strip(),
                "content": [e.text.strip() for e in content_block]
            }

        else:
            # غالبًا Subsection → ضيفه داخل نفس القسم
            if current_section:
                current_section["content"].append(element.text.strip())
                current_section["content"].extend(
                    [e.text.strip() for e in content_block]
                )

        i = j

    else:
        i += 1

# حفظ آخر واحد
if current_section:
    sections.append(current_section)

for sec in sections:
    print("Section Title:", sec["title"])
    print("Content:", sec["content"])
    print("-" * 50)"""