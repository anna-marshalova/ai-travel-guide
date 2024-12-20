import json
import re

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import wikipediaapi


class WikiParser:
    def __init__(self, base_url="https://ru.wikipedia.org"):
        self.base_url = base_url
        self.wiki_html = wikipediaapi.Wikipedia(
            "MyProjectName (merlin@example.com)",
            "ru",
            extract_format=wikipediaapi.ExtractFormat.HTML,
        )

    def get_page_url(self, page_name):
        if self.base_url in page_name:
            return page_name
        else:
            page_name = page_name.split("/")[0].replace(" ", "_")
        return f"{self.base_url}/wiki/{page_name.replace(' ', '_')}"

    def get_name_link(self, div):
        a = div.find("a")
        if a:
            return a.get_text().strip(), a.get("href")
        return None, None

    def find_iter_class(self, wrapper, node_class, terminal_class):
        result = []
        for item in wrapper.find_all(node_class):
            terminals = item.find_all(class_=terminal_class)
            result.extend(terminals)
            children = item.find_all(class_=node_class)
            for child in children:
                result.extend(self.find_iter_class(child, node_class, terminal_class))
        return result

    def get_pages_from_table(self, index_page_name, target_column):
        index_page_url = self.get_page_url(index_page_name)
        response = requests.get(index_page_url)
        soup = BeautifulSoup(response.text, "html.parser")

        pages = {}
        # Find tables with city lists
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            all_cols = [col.get_text().strip() for col in rows[0].find_all("th")]
            if target_column in all_cols:
                target_col_num = all_cols.index(target_column)
                for row in rows[1:]:
                    cols = row.find_all("td")
                    if cols:
                        name, link = self.get_name_link(cols[target_col_num])
                        if name and link:
                            pages[name] = link
        return pages

    def get_pages_from_category(self, index_page_name, max_pages=None):
        max_pages = max_pages or float("inf")

        def get_categorymembers(members, result={}, level=0, max_level=1):
            if max_pages is not None and len(result) >= max_pages:
                return result
            if level == 0:
                pbar = tqdm(members.values())
            else:
                pbar = members.values()
            for member in pbar:
                if len(result) > max_pages:
                    return result
                if member.ns != wikipediaapi.Namespace.CATEGORY:
                    result.update({member.title: member.canonicalurl})
                if member.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
                    result.update(
                        get_categorymembers(
                            member.categorymembers,
                            result=result,
                            level=level + 1,
                            max_level=max_level,
                        )
                    )
            return result

        # index_page_url = self.get_index_page_url(index_page_name)
        pages = {}
        # response = requests.get(index_page_url)
        category_page = self.wiki_html.page(index_page_name)
        pages = get_categorymembers(category_page.categorymembers)
        return pages

    def validate_section_title(self, section_title):
        if section_title.get("class") in ["mw-toc-heading", "vector-menu-heading"]:
            return False
        if section_title.get("id") in [
            "mw-toc-heading",
            "Ссылки",
            "Литература",
            "Примечания",
            "См. также",
            "Навигация",
            "Прочее",
        ]:
            return False
        if re.match(r"p(-.*)*", section_title.get("id")):
            return False
        return True

    def validate_content(self, content):
        if content.name not in ["p", "ul", "ol", "section", "div"]:
            return False
        if content.name == "div" and ("vcard" not in content.get("class")):
            return False
        if content.get("class") in ["reference"]:
            return False
        return True

    def parse_page_wikipedia(self, name):
        """Parse Wikipedia page for a specific city"""
        page = self.wiki_html.page(name)
        page_info = {}
        for section in page.sections:
            title = section.title
            page_info.update({title: []})
            soup = BeautifulSoup(section.text, "html.parser")
            for current in soup.find_all(["p", "ol", "ul"]):
                page_info[title].append(current.text)

            for subsection in section.sections:
                subtitle = f"{title}: {subsection.title}"
                page_info.update({subtitle: []})
                soup = BeautifulSoup(subsection.text, "html.parser")
                for current in soup.find_all(["p", "ol", "ul"]):
                    page_info[subtitle].append(current.text)
        return {k: " ".join(v) for k, v in page_info.items() if v}

    def parse_page_wikivoyage(self, name):
        """Parse Wikivoyage page for a specific city"""
        # Encode city name for URL
        url = self.get_page_url(name)

        try:
            response = requests.get(url, timeout=20)
            soup = BeautifulSoup(response.text, "html.parser")
            # Initialize city data dictionary
            page_info = {}
            section_divs = soup.find_all("div", class_="mw-heading mw-heading2")
            if not section_divs:
                snak = soup.find(class_="wikidata-main-snak")
                snak_name = snak.find("a").get_text().strip()
                snak_link = snak.find("a").get("href")
                if snak_name and snak_link:
                    return self.parse_page_wikivoyage(snak_name, snak_link)
                else:
                    return {}

            for section_div in section_divs:
                section = section_div.find("h2")
                title = section.get("id")
                if self.validate_section_title(section) and title:
                    section_content = {title: []}
                    current = section_div.find_next_sibling()
                    while (
                        current
                        and current.name not in ["h2"]
                        and set(current.get("class", [])) & {"mw-heading2"} == set()
                    ):
                        if self.validate_content(current):
                            if current.name in ["p", "ul", "ol"]:
                                section_content[title].append(
                                    current.get_text().strip()
                                )
                            if current.name == "div":
                                section_content[title].append(
                                    current.get_text().strip()
                                )
                            if current.name == "section":
                                current_div = current.find("div")
                                if "mw-heading3" in current_div.get("class", []):
                                    subtitle = current_div.find("h3").get("id")
                                    if (
                                        self.validate_section_title(current_div)
                                        and subtitle
                                    ):
                                        section_content[f"{title}: {subtitle}"] = []
                                        current_div = current_div.find_next_sibling()
                                        while (
                                            current_div
                                            and current_div.name not in ["h2", "h3"]
                                            and set(current_div.get("class", []))
                                            & {"mw-heading2", "mw-heading3"}
                                            == set()
                                        ):
                                            if current_div.name in ["p", "ul", "ol"]:
                                                section_content[
                                                    f"{title}: {subtitle}"
                                                ].append(current_div.get_text().strip())
                                            else:
                                                current_div_div = current_div.find_all(
                                                    "div"
                                                )
                                                for d in current_div_div:
                                                    section_content[
                                                        f"{title}: {subtitle}"
                                                    ].append(d.get_text().strip())
                                            if current_div.find_next_sibling():
                                                current_div = (
                                                    current_div.find_next_sibling()
                                                )
                                            else:
                                                current_div = current_div.parent
                                                if current_div:
                                                    current_div = (
                                                        current_div.parent.parent
                                                    )
                                                break

                        current = current.find_next_sibling()
                    # Combine content if there's any
                    section_content = {
                        k: " ".join(v) for k, v in section_content.items()
                    }
                    page_info.update(section_content)

            return page_info

        except Exception as e:
            print(f"Error parsing {name}: {e}")
            return {}

    def scrape_pages(self, pages, limit=None):
        if limit is not None:
            pages = dict(list(pages.items())[:limit])
        data = {}
        pbar = tqdm(list(pages.items()))
        for name, link in pbar:
            pbar.set_description(name)
            if self.base_url in ["https://ru.wikivoyage.org"]:
                page_data = self.parse_page_wikivoyage(name)
            else:
                page_data = self.parse_page_wikipedia(name)
            if page_data:
                data[name] = page_data

        return data


def save_to_json(data, filename):
    """Save collected data to JSON"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {filename}")


def get_whole_data(big_city_limit=None, russian_city_limit=None, tourism_limit=None):
    wikipedia_parser = WikiParser()
    wikivoyage_parser = WikiParser(base_url="https://ru.wikivoyage.org")

    russian_cities_pages = wikipedia_parser.get_pages_from_table(
        "Список городов России", "Город"
    )
    big_cities_pages = wikipedia_parser.get_pages_from_table(
        "Список городов с населением более миллиона человек", "Город"
    )
    tourism_pages = wikipedia_parser.get_pages_from_category(
        "Категория:Туризм по странам", max_pages=tourism_limit
    )

    russian_cities_data = wikipedia_parser.scrape_pages(
        russian_cities_pages, russian_city_limit
    )
    save_to_json(russian_cities_data, "russian_cities_data.json")
    big_cities_data = wikipedia_parser.scrape_pages(big_cities_pages, big_city_limit)
    save_to_json(big_cities_data, "big_cities_data.json")

    tourism_data = wikipedia_parser.scrape_pages(tourism_pages, tourism_limit)
    save_to_json(tourism_data, "tourism_data.json")

    wikivoyage_russian_cities_data = wikivoyage_parser.scrape_pages(
        russian_cities_pages, russian_city_limit
    )
    save_to_json(wikivoyage_russian_cities_data, "wikivoyage_russian_cities_data.json")

    wikivoyage_big_cities_data = wikivoyage_parser.scrape_pages(
        big_cities_pages, big_city_limit
    )
    save_to_json(wikivoyage_big_cities_data, "wikivoyage_big_cities_data.json")


def get_wikivoyage_data(
    big_city_limit=None, russian_city_limit=None
):
    wikipedia_parser = WikiParser()
    wikivoyage_parser = WikiParser(base_url="https://ru.wikivoyage.org")

    russian_cities_pages = wikipedia_parser.get_pages_from_table(
        "Список городов России", "Город"
    )
    big_cities_pages = wikipedia_parser.get_pages_from_table(
        "Список городов с населением более миллиона человек", "Город"
    )

    wikivoyage_russian_cities_data = wikivoyage_parser.scrape_pages(
        russian_cities_pages, russian_city_limit
    )
    save_to_json(wikivoyage_russian_cities_data, "wikivoyage_russian_cities_data.json")
    wikivoyage_big_cities_data = wikivoyage_parser.scrape_pages(
        big_cities_pages, big_city_limit
    )
    save_to_json(wikivoyage_big_cities_data, "wikivoyage_big_cities_data.json")


if __name__ == "__main__":
    get_whole_data(big_city_limit=None, russian_city_limit=None, tourism_limit=None)
    # get_wikivoyage_data(big_city_limit=None, russian_city_limit=None)
