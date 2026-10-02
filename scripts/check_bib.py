# -*- coding: utf-8 -*-
"""refs.bib 의 서지를 CrossRef (DOI/제목 검색) 와 arXiv API 로 대조한다.

  python scripts/check_bib.py

출력: 항목별로 bib 값과 외부 값 (journal/booktitle, volume, issue, pages, year, DOI) 을 나란히 찍고, 다른 곳에 '!!' 를 표시한다.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.path.join(REPO, "paper", "refs.bib")
UA = "subtractive-intelligence-bibcheck/1.0 (mailto:shalooooooooom@gmail.com)"


def parse_bib(path):
    txt = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\{(\w+),(.*?)\n\}", txt, flags=re.S):
        kind, key, body = m.group(1), m.group(2), m.group(3)
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", flags=re.S))
        fields = {k: re.sub(r"[{}]", "", v).replace("\\&", "&").replace("~", " ") for k, v in fields.items()}
        out.append((kind, key, fields))
    return out


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def crossref(doi=None, title=None):
    if doi:
        try:
            return json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
        except Exception as e:
            return {"error": str(e)}
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 1})
    try:
        items = json.loads(get("https://api.crossref.org/works?" + q))["message"]["items"]
        return items[0] if items else {"error": "no result"}
    except Exception as e:
        return {"error": str(e)}


def arxiv(title):
    q = urllib.parse.urlencode({"search_query": 'ti:"%s"' % title.replace("$", "").replace("-", " "), "max_results": 1})
    try:
        xml = get("http://export.arxiv.org/api/query?" + q)
        root = ET.fromstring(xml)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = root.find("a:entry", ns)
        if e is None:
            return {"error": "no result"}
        return {"id": e.find("a:id", ns).text, "title": " ".join(e.find("a:title", ns).text.split()),
                "published": e.find("a:published", ns).text[:10]}
    except Exception as e:
        return {"error": str(e)}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def main():
    entries = parse_bib(BIB)
    print(f"{len(entries)} entries")
    for kind, key, f in entries:
        title = f.get("title", "")
        if "arxiv" in f.get("journal", "").lower():
            r = arxiv(title)
            same = norm(r.get("title", ""))[:40] == norm(title)[:40]
            print(f"\n[{key}] arXiv  bib: {f.get('journal')} {f.get('year')}")
            print(f"   api: {r.get('id')} {r.get('published')} | {r.get('title', r.get('error'))[:90]} {'' if same else '!! title mismatch'}")
            time.sleep(3)
            continue
        r = crossref(doi=f.get("doi"), title=title)
        if "error" in r:
            print(f"\n[{key}] crossref error: {r['error']}")
            continue
        ct = r.get("container-title", [""])
        ct = ct[0] if ct else ""
        ev = r.get("event", {}).get("name", "")
        year = None
        for k in ("published-print", "published-online", "issued", "created"):
            if r.get(k, {}).get("date-parts"):
                year = r[k]["date-parts"][0][0]
                break
        api = {"title": " ".join((r.get("title") or [""])[0].split()), "container": ct or ev, "volume": r.get("volume"),
               "issue": r.get("issue"), "page": r.get("page"), "year": year, "doi": r.get("DOI")}
        same = norm(api["title"])[:40] == norm(title)[:40]
        flags = []
        for bk, ak in (("volume", "volume"), ("number", "issue"), ("pages", "page"), ("year", "year")):
            bv, av = f.get(bk), api.get(ak)
            if bv is not None and av is not None and norm(str(bv).replace("--", "-")) != norm(str(av)):
                flags.append(f"!! {bk}: bib {bv} vs api {av}")
        if not same:
            flags.append("!! title mismatch (check manually)")
        print(f"\n[{key}] bib: {f.get('journal') or f.get('booktitle')} v{f.get('volume')} n{f.get('number')} p{f.get('pages')} {f.get('year')} {f.get('doi', '')}")
        print(f"   api: {api['container'][:60]} v{api['volume']} n{api['issue']} p{api['page']} {api['year']} {api['doi']} | {api['title'][:80]}")
        for fl in flags:
            print("   " + fl)
        time.sleep(1.2)


if __name__ == "__main__":
    main()
