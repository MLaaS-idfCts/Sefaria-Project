import re
import pickle
from collections import Counter
import itertools
import requests
import html
from tqdm import tqdm


def get_text(book, chapter, verse):
    params = dict(context=0, ven=0)
    url = f'https://www.sefaria.org/api/texts/{book}.{chapter}.{verse}'
    resp = requests.get(url=url, params=params)
    data = resp.json()
    print(data['he'])
    return data['he']


def get_text_by_ref(ref, remove_nikud=True):
    params = dict(context=0, ven=0)
    url = f'https://www.sefaria.org/api/texts/{ref}'
    resp = requests.get(url=url, params=params)
    data = resp.json()['he'].rstrip(" \n")
    if remove_nikud:
        return re.sub(r'[\u0591-\u05BD\u05BF-\u05C2\u05C4-\u05C7]', '', data)
    return data


def get_topics_per_verse_in_book(book, start_chapter=1, verse_topics_dict=None):
    chapter = start_chapter
    if verse_topics_dict is None:
        verse_topics_dict = {}
    is_end_counter = 0
    while True:
        try:
            url = f'https://www.sefaria.org.il/api/related/{book}.{chapter}?with_sheet_links=1'
            resp = requests.get(url=url)
            data = resp.json()
            if "error" in data:
                break
            if len(data["topics"]) == 0:
                is_end_counter += 1
            if is_end_counter > 10:
                break
            for k in data["topics"]:
                for verse in k["anchorRefExpanded"]:
                    is_end_counter = 0
                    if verse not in verse_topics_dict:
                        verse_topics_dict[verse] = set()
                    verse_topics_dict[verse].add(k["topic"])
        except Exception as e:
            print(f"Error: {book}-{chapter} - {e}")
        chapter += 1
        # print(chapter)
    return verse_topics_dict


def get_all_books():
    url = 'https://www.sefaria.org.il/api/index/titles'
    resp = requests.get(url=url)
    data = resp.json()
    # print(data['books'])
    return data['books']


def get_all_book_names(base_url):
    resp = requests.get(url=base_url)
    pattern = re.compile(
        f'<div class="gridBoxItem"><a href="/(.+?)" class="blockLink refLink"')
    data_found = list(re.findall(pattern, html.unescape((resp.text))))
    print(data_found)


def pickle_save_obj(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_obj(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def step1_extract_topics_per_verse(base_path):
    all_topics_per_verse = {}
    # for book in ['Mishnah_Berakhot', 'Mishnah_Peah', 'Mishnah_Demai', 'Mishnah_Kilayim', 'Mishnah_Sheviit',
    #              'Mishnah_Terumot', 'Mishnah_Maasrot', 'Mishnah_Maaser_Sheni', 'Mishnah_Challah', 'Mishnah_Orlah',
    #              'Mishnah_Bikkurim', 'Mishnah_Shabbat', 'Mishnah_Eruvin', 'Mishnah_Pesachim', 'Mishnah_Shekalim',
    #              'Mishnah_Yoma', 'Mishnah_Sukkah', 'Mishnah_Beitzah', 'Mishnah_Rosh_Hashanah', 'Mishnah_Taanit',
    #              'Mishnah_Megillah', 'Mishnah_Moed_Katan', 'Mishnah_Chagigah', 'Mishnah_Yevamot', 'Mishnah_Ketubot',
    #              'Mishnah_Nedarim', 'Mishnah_Nazir', 'Mishnah_Sotah', 'Mishnah_Gittin', 'Mishnah_Kiddushin',
    #              'Mishnah_Bava_Kamma', 'Mishnah_Bava_Metzia', 'Mishnah_Bava_Batra', 'Mishnah_Sanhedrin',
    #              'Mishnah_Makkot', 'Mishnah_Shevuot', 'Mishnah_Eduyot', 'Mishnah_Avodah_Zarah', 'Pirkei_Avot',
    #              'Mishnah_Horayot', 'Mishnah_Zevachim', 'Mishnah_Menachot', 'Mishnah_Chullin', 'Mishnah_Bekhorot',
    #              'Mishnah_Arakhin', 'Mishnah_Temurah', 'Mishnah_Keritot', 'Mishnah_Meilah', 'Mishnah_Tamid',
    #              'Mishnah_Middot', 'Mishnah_Kinnim', 'Mishnah_Kelim', 'Mishnah_Oholot', 'Mishnah_Negaim',
    #              'Mishnah_Parah', 'Mishnah_Tahorot', 'Mishnah_Mikvaot', 'Mishnah_Niddah', 'Mishnah_Makhshirin',
    #              'Mishnah_Zavim', 'Mishnah_Tevul_Yom', 'Mishnah_Yadayim', 'Mishnah_Oktzin']:
    for book in [("Mekhilta_d'Rabbi_Yishmael", 12), ("Mekhilta_DeRabbi_Shimon_Bar_Yochai", 3),
                 "Sifra%2C_Braita_d'Rabbi_Yishmael", 'Sifrei_Bamidbar', 'Sifrei_Devarim', 'Midrash_Mishlei',
                 ('Midrash_Sekhel_Tov%2C_Bereshit', 15), 'Midrash_Tehillim',
                 'Ein_Yaakov%2C_Berakhot', 'Ein_Yaakov_(Glick_Edition)%2C_Berakhot', "Pesikta_D'Rav_Kahanna",
                 'Pesikta_Rabbati', 'Pirkei_DeRabbi_Eliezer', 'Tanna_debei_Eliyahu_Zuta%2C_Seder_Eliyahu_Zuta',
                 'Tanna_Debei_Eliyahu_Rabbah', 'Midrash_Tanchuma_Buber%2C_Bereshit']:
        if type(book) is tuple:
            get_topics_per_verse_in_book(book[0], book[1], verse_topics_dict=all_topics_per_verse)
        else:
            get_topics_per_verse_in_book(book, verse_topics_dict=all_topics_per_verse)
        print(book)
    pickle_save_obj(f'{base_path}/verses.pickle', list(all_topics_per_verse.keys()))
    pickle_save_obj(f'{base_path}/topics.pickle', list(all_topics_per_verse.values()))


def step2_extract_text_per_verse(base_path):
    all_verses = pickle_load_obj(f'{base_path}/verses.pickle')
    all_text = []
    for k in tqdm(all_verses):
        all_text += [get_text_by_ref(k)]
    pickle_save_obj(f'{base_path}/texts.pickle', all_text)


if __name__ == "__main__":
    # get_all_book_names("https://www.sefaria.org.il/texts/Midrash")
    base_path = "/data/database/midrash"
    # Step 1
    step1_extract_topics_per_verse(base_path)
    # Test step 1
    # print(len(pickle_load_obj(f'{base_path}/verses.pickle')))
    # a = pickle_load_obj(f'{base_path}/topics.pickle')
    # print(Counter(itertools.chain.from_iterable(map(lambda x: list(x), a))))
    # Step 2
    step2_extract_text_per_verse(base_path)
    # Test Step 2
    # print(pickle_load_obj(f'{base_path}/texts.pickle')[3000])
