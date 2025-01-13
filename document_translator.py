from pdfplumber import PDF
import os
from openai import OpenAI

client = OpenAI()


def load_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    with PDF(open(pdf_file, "rb")) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, max_chars=7000):
    """Divide text into chunks suitable for GPT-4."""
    chunks = []
    while text:
        chunk = text[:max_chars]
        # Ensure chunks do not split mid-sentence or mid-paragraph
        last_period = max(chunk.rfind(". "), chunk.rfind("\n"))
        if last_period == -1 or last_period < int(max_chars * 0.8):
            last_period = len(chunk) - 1
        chunks.append(chunk[: last_period + 1])
        text = text[last_period + 1 :].lstrip()
    return chunks


def translate_chunk(chunk, model="gpt-4o-mini"):
    """Send a chunk to GPT-4 for translation."""
    completion = client.chat.completions.create(
        model=model,
        store=True,
        messages=[
            {
                "role": "system",
                "content": """You are a professional translator. Translate the following text line-by-line.
The passage is a section of a manuscript that you to translate into contemporary English to make it easily readable for everyone. Rephrase the following text into simpler English at an 8th-grade reading level. Please rewrite it and do not plagiarize. Do not give an answer, only give the translated text. 
Make sure that the reading experience “flows”, for example by not using the same words & sentence structures too often. Only translate, do not mention anything else. Please format the text properly without separating lines between the passages and exactly in the same structure it was so I can easily copy/paste it entirely into the manuscript of the book. You can remove numbers if that makes the reading experience better. You must translate/rewrite everything exactly and not shorten it. Keep the quotes in their formatted way if applicable.
This is important for my career: Please try to translate everything sentence-by-sentence and do not make it shorter!
Important: Please Translate everything sentence-by-sentence in contemporary English and do not make it shorter!
So do not skip translating any sentence line-by-line in contemporary English and do not make the translation shorter. This is crucial!
""",
            },
            {"role": "user", "content": chunk},
        ],
    )

    return completion.choices[0].message.content


def translate_manuscript(text):
    """Translate an entire manuscript."""
    chunks = chunk_text(text)
    translations = []
    for i, chunk in enumerate(chunks):
        # break after 10 chunks for testing
        if i == 1:
            break
        print(f"Translating chunk {i+1}/{len(chunks)}...")
        try:
            translation = translate_chunk(chunk)
            translations.append(translation)
        except Exception as e:
            print(f"Error translating chunk {i+1}: {e}")
            translations.append(
                "[ERROR: Translation failed for this section. Please review manually]"
            )
    return "\n".join(translations)


def save_translation(translated_text, output_file):
    """Save the translated text to a file."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(translated_text)


def main():
    print("Manuscript Translation Tool")
    input_file = input("Enter the path to the PDF manuscript file: ")
    output_file = input("Enter the path to save the translated manuscript (TXT): ")

    if input_file.endswith(".pdf"):
        text = load_text_from_pdf(input_file)
    else:
        print("Unsupported file type. Please provide a PDF file.")
        return

    print("File loaded successfully. Starting translation...")
    translated_text = translate_manuscript(text)

    print("Translation completed. Saving output...")
    save_translation(translated_text, output_file)
    print(f"Translated manuscript saved to {output_file}")


if __name__ == "__main__":
    main()
