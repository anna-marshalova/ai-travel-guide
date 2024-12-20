import os
import re
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter


def isvalid_text(text):
    return text and not re.match(r"^\s*$", text)


def remove_metadata(page_data):
    metedata_keys = [
        "Ссылки",
        "Литература",
        "Примечания",
        "См. также",
        "Навигация",
        "Прочее",
    ]
    for key in metedata_keys:
        page_data.pop(key, None)
    return page_data


def get_section_name(page_name, section_name):
    if page_name not in section_name:
        section_name = f"{page_name}: {section_name}"
    return section_name.replace("_", " ")


def flatten_data(data):
    flat_data = {}
    if isinstance(data, dict):
        data = [data]
    for d in data:
        for page_name, page_data in d.items():
            d[page_name] = remove_metadata(page_data)
            for section_name, section_data in page_data.items():
                if section_data:
                    section_name = get_section_name(page_name, section_name)
                    if section_name not in flat_data:
                        flat_data[section_name] = {section_data}
                    else:
                        flat_data[section_name].add(section_data)
    return flat_data


def make_chunks(data, chunk_size=1000, chunk_overlap=200):
    chunked_data = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    for title, texts in data.items():
        if isinstance(texts, str):
            texts = [texts]
        chunks = []
        for text in texts:
            if text:
                chunks.extend(text_splitter.split_text(text))
        if chunks:
            chunked_data[title] = chunks
    return chunked_data

def load_and_preprocess_data(datadir=None, paths=[]):
    assert datadir or paths, "Please provide `datadir` or List[str] of paths"
    data = []
    if len(paths) == 0:
        paths = [os.path.join(datadir, fname) for fname in os.listdir(datadir) if fname.endswith(".json")]
    for path in paths:
        with open(path) as f:
            data.append(json.load(f))

    flat_data = flatten_data(data)
    chunked_data = make_chunks(flat_data)
    return chunked_data

if __name__ == "__main__":
    paths = ["./data/big_cities_data.json"]
    chunked_data = load_and_preprocess_data(paths=paths)
    assert all(
        len(chunk) < 2000 for chunks in chunked_data.values() for chunk in chunks
    )
