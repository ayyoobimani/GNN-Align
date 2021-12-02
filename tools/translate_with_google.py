import argparse
import six
from google.cloud import translate_v2 as translate
import codecs
import random

SPECIAL_CASES = {
    'ee': 'et',
}

LANGUAGES = {
    'af': 'afrikaans',
    'sq': 'albanian',
    'am': 'amharic',
    'ar': 'arabic',
    'hy': 'armenian',
    'az': 'azerbaijani',
    'eu': 'basque',
    'be': 'belarusian',
    'bn': 'bengali',
    'bs': 'bosnian',
    'bg': 'bulgarian',
    'ca': 'catalan',
    'ceb': 'cebuano',
    'ny': 'chichewa',
    'zh-cn': 'chinese (simplified)',
    'zh-tw': 'chinese (traditional)',
    'co': 'corsican',
    'hr': 'croatian',
    'cs': 'czech',
    'da': 'danish',
    'nl': 'dutch',
    'en': 'english',
    'eo': 'esperanto',
    'et': 'estonian',
    'tl': 'filipino',
    'fi': 'finnish',
    'fr': 'french',
    'fy': 'frisian',
    'gl': 'galician',
    'ka': 'georgian',
    'de': 'german',
    'el': 'greek',
    'gu': 'gujarati',
    'ht': 'haitian creole',
    'ha': 'hausa',
    'haw': 'hawaiian',
    'iw': 'hebrew',
    'he': 'hebrew',
    'hi': 'hindi',
    'hmn': 'hmong',
    'hu': 'hungarian',
    'is': 'icelandic',
    'ig': 'igbo',
    'id': 'indonesian',
    'ga': 'irish',
    'it': 'italian',
    'ja': 'japanese',
    'jw': 'javanese',
    'kn': 'kannada',
    'kk': 'kazakh',
    'km': 'khmer',
    'ko': 'korean',
    'ku': 'kurdish (kurmanji)',
    'ky': 'kyrgyz',
    'lo': 'lao',
    'la': 'latin',
    'lv': 'latvian',
    'lt': 'lithuanian',
    'lb': 'luxembourgish',
    'mk': 'macedonian',
    'mg': 'malagasy',
    'ms': 'malay',
    'ml': 'malayalam',
    'mt': 'maltese',
    'mi': 'maori',
    'mr': 'marathi',
    'mn': 'mongolian',
    'my': 'myanmar (burmese)',
    'ne': 'nepali',
    'no': 'norwegian',
    'or': 'odia',
    'ps': 'pashto',
    'fa': 'persian',
    'pl': 'polish',
    'pt': 'portuguese',
    'pa': 'punjabi',
    'ro': 'romanian',
    'ru': 'russian',
    'sm': 'samoan',
    'gd': 'scots gaelic',
    'sr': 'serbian',
    'st': 'sesotho',
    'sn': 'shona',
    'sd': 'sindhi',
    'si': 'sinhala',
    'sk': 'slovak',
    'sl': 'slovenian',
    'so': 'somali',
    'es': 'spanish',
    'su': 'sundanese',
    'sw': 'swahili',
    'sv': 'swedish',
    'tg': 'tajik',
    'ta': 'tamil',
    'te': 'telugu',
    'th': 'thai',
    'tr': 'turkish',
    'uk': 'ukrainian',
    'ur': 'urdu',
    'ug': 'uyghur',
    'uz': 'uzbek',
    'vi': 'vietnamese',
    'cy': 'welsh',
    'xh': 'xhosa',
    'yi': 'yiddish',
    'yo': 'yoruba',
    'zu': 'zulu'
    }

def read_file(fpath):
    res = {}
    with codecs.open(fpath, 'r', 'utf8') as f:
        for l in f:
            res[l.strip().split('\t')[1]] = l.strip().split('\t')[0]
        
    return res

def write_out(output_path, lines, s_lang, t_lang, translations):
    with codecs.open(f"{output_path}/MT_{s_lang}_{t_lang}.txt", 'w', 'utf8') as f:
        for translation in translations:
            f.write(f"{lines[translation['origin']]}\t{translation['text']}\n")

def create_translation(s_file, output_path, s_lang, t_langs):
    lines = read_file(s_file)
    
    #print(lines)
    
    translate_client = translate.Client()

    if t_langs == None:
        t_langs = list(LANGUAGES.keys())
        t_langs.remove(s_lang)
    
    inp = list(lines.keys())
    for t_lang in t_langs:
        print(f"translating from {s_lang} to {t_lang}")
        translations = []
        for line in inp:
            res = translate_client.translate(line, target_language=t_lang)
            translations.append({'origin':res['input'], 'text':res['translatedText']})

            if random.randint(0, len(lines)) < len(lines)/10:
                print(f"{s_lang}->{t_lang}, {res['input']}->{res['translatedText']}")
        write_out(output_path, lines, s_lang, t_lang, translations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="translates the given file in blinker data format to target language(s).", 
	epilog="example: python -m translate_with_google -sf source_file -sl source_lang -tl en,de,ru -o output_path ")
    
    parser.add_argument('-sf', default=None, type=str, required=True, help="source file path")
    parser.add_argument('-sl', default=None, type=str, required=True, help="source language")    
    parser.add_argument('-tl', default=None, type=str, required=False, help="comma separated"
        "list of target languages, if not provided translates to all available languages") 
    parser.add_argument('-o', default=None, type=str, required=True, help="directory to save translation files") 

    args = parser.parse_args()

    if args.tl != None:
        args.tl = list(args.tl.strip().split(','))
    print("hi")
    
    create_translation(args.sf, args.o, args.sl, args.tl)
