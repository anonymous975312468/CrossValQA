#!/usr/bin/env python3
"""
Download PubMed articles for proteomics pipeline.

Downloads full-text articles from PMC when available, falls back to abstracts.
Saves to category folders and creates a master file.
"""

import os
import json
import time
import requests
import xml.etree.ElementTree as ET

# --- CONFIGURATION ---
EMAIL = "anonymous@example.com"
BASE_DIR = "proteomics_dataset"

# NCBI E-utilities URLs
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Proteomics categories and PMIDs
PROTEOMICS_DATA = {
    "enzymology_and_kinetics": ["41381349", "41096611", "40725096", "40545624", "40349921"],
    "structural_biology_and_motifs": ["40799657", "41538474", "41465416", "40545624", "36362121"],
    "protein_protein_interactions": ["40433662", "40385224", "37812997", "35949491", "35615908"],
    "post_translational_modifications": ["37275247", "31345222", "28159896", "23844161", "39796186"],
    "clinical_proteomics_and_biomarkers": ["40885913", "40618677", "40384977", "41569177", "41524014", "41147948", "40507607"]
}


def get_pmc_id(pmid: str) -> str:
    """Convert PMID to PMC ID if available."""
    params = {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": pmid,
        "email": EMAIL
    }
    response = requests.get(ELINK_URL, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    link = root.find(".//Link/Id")
    if link is not None and link.text:
        return f"PMC{link.text}"
    return None


def fetch_pmc_full_text(pmc_id: str) -> dict:
    """Fetch full-text article from PMC."""
    params = {
        "db": "pmc",
        "id": pmc_id.replace("PMC", ""),
        "rettype": "xml",
        "retmode": "xml",
        "email": EMAIL
    }

    response = requests.get(EFETCH_URL, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    article = root.find(".//article")

    if article is None:
        return None

    # Extract title
    title_elem = article.find(".//article-title")
    title = "".join(title_elem.itertext()) if title_elem is not None else ""

    # Extract abstract
    abstract_parts = []
    for abs_elem in article.findall(".//abstract"):
        for p in abs_elem.findall(".//p"):
            text = "".join(p.itertext())
            if text.strip():
                abstract_parts.append(text.strip())
    abstract = "\n".join(abstract_parts)

    # Extract full body text
    body_sections = []
    for body in article.findall(".//body"):
        for sec in body.findall(".//sec"):
            section = extract_section(sec)
            if section:
                body_sections.append(section)

        # Also get paragraphs directly under body (not in sections)
        for p in body.findall("./p"):
            text = "".join(p.itertext())
            if text.strip():
                body_sections.append({"title": "", "text": text.strip()})

    # Extract authors
    authors = []
    for contrib in article.findall(".//contrib[@contrib-type='author']"):
        surname = contrib.find(".//surname")
        given = contrib.find(".//given-names")
        if surname is not None and surname.text:
            name = surname.text
            if given is not None and given.text:
                name = f"{given.text} {surname.text}"
            authors.append(name)

    # Extract journal
    journal_elem = article.find(".//journal-title")
    journal = journal_elem.text if journal_elem is not None else ""

    # Extract publication date
    pub_date = ""
    year_elem = article.find(".//pub-date/year")
    if year_elem is not None and year_elem.text:
        pub_date = year_elem.text

    # Extract keywords
    keywords = []
    for kw in article.findall(".//kwd"):
        if kw.text:
            keywords.append(kw.text)

    # Extract DOI
    doi = ""
    for article_id in article.findall(".//article-id"):
        if article_id.get("pub-id-type") == "doi":
            doi = article_id.text
            break

    # Combine body text
    full_text_parts = []
    for section in body_sections:
        if section["title"]:
            full_text_parts.append(f"\n## {section['title']}\n")
        full_text_parts.append(section["text"])
    full_text = "\n".join(full_text_parts)

    return {
        "title": title,
        "abstract": abstract,
        "full_text": full_text,
        "sections": body_sections,
        "authors": authors,
        "journal": journal,
        "publication_date": pub_date,
        "keywords": keywords,
        "doi": doi,
    }


def extract_section(sec_elem) -> dict:
    """Extract text from a section element."""
    # Get section title
    title_elem = sec_elem.find("./title")
    title = "".join(title_elem.itertext()) if title_elem is not None else ""

    # Get all paragraphs in this section (not in subsections)
    paragraphs = []
    for p in sec_elem.findall("./p"):
        text = "".join(p.itertext())
        if text.strip():
            paragraphs.append(text.strip())

    # Get subsections recursively
    subsections = []
    for subsec in sec_elem.findall("./sec"):
        sub = extract_section(subsec)
        if sub:
            subsections.append(sub)

    text = "\n\n".join(paragraphs)

    # Append subsection text
    for sub in subsections:
        if sub["title"]:
            text += f"\n\n### {sub['title']}\n{sub['text']}"
        else:
            text += f"\n\n{sub['text']}"

    if not text.strip():
        return None

    return {"title": title, "text": text.strip()}


def fetch_pubmed_metadata(pmid: str) -> dict:
    """Fetch article metadata and abstract from PubMed."""
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "xml",
        "retmode": "xml",
        "email": EMAIL
    }

    response = requests.get(EFETCH_URL, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    article_elem = root.find(".//PubmedArticle")

    if article_elem is None:
        return None

    # Extract title
    title_elem = article_elem.find(".//ArticleTitle")
    title = "".join(title_elem.itertext()) if title_elem is not None else ""

    # Extract abstract (may have multiple parts)
    abstract_parts = []
    for abs_text in article_elem.findall(".//AbstractText"):
        label = abs_text.get("Label", "")
        text = "".join(abs_text.itertext()) if abs_text.text or len(abs_text) > 0 else ""
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    abstract = "\n".join(abstract_parts)

    # Extract authors
    authors = []
    for author in article_elem.findall(".//Author"):
        lastname = author.find("LastName")
        forename = author.find("ForeName")
        if lastname is not None and lastname.text:
            name = lastname.text
            if forename is not None and forename.text:
                name = f"{forename.text} {lastname.text}"
            authors.append(name)

    # Extract journal
    journal_elem = article_elem.find(".//Journal/Title")
    journal = journal_elem.text if journal_elem is not None else ""

    # Extract publication date
    pub_date = ""
    year_elem = article_elem.find(".//PubDate/Year")
    month_elem = article_elem.find(".//PubDate/Month")
    if year_elem is not None:
        pub_date = year_elem.text
        if month_elem is not None and month_elem.text:
            pub_date = f"{month_elem.text} {pub_date}"

    # Extract MeSH terms
    mesh_terms = []
    for mesh in article_elem.findall(".//MeshHeading/DescriptorName"):
        if mesh.text:
            mesh_terms.append(mesh.text)

    # Extract keywords
    keywords = []
    for kw in article_elem.findall(".//Keyword"):
        if kw.text:
            keywords.append(kw.text)

    # Extract DOI
    doi = ""
    for article_id in article_elem.findall(".//ArticleId"):
        if article_id.get("IdType") == "doi":
            doi = article_id.text
            break

    return {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "publication_date": pub_date,
        "mesh_terms": mesh_terms,
        "keywords": keywords,
        "doi": doi,
    }


def fetch_article(pmid: str) -> dict:
    """Fetch article - tries PMC full text first, falls back to PubMed abstract."""

    # First get PubMed metadata
    pubmed_data = fetch_pubmed_metadata(pmid)
    if pubmed_data is None:
        return None

    article = {
        "pmid": pmid,
        "pmc_id": None,
        "has_full_text": False,
        **pubmed_data,
        "full_text": "",
        "sections": [],
    }

    # Try to get PMC full text
    time.sleep(0.35)  # Rate limit
    pmc_id = get_pmc_id(pmid)

    if pmc_id:
        article["pmc_id"] = pmc_id
        time.sleep(0.35)  # Rate limit

        try:
            pmc_data = fetch_pmc_full_text(pmc_id)
            if pmc_data and pmc_data.get("full_text"):
                article["has_full_text"] = True
                article["full_text"] = pmc_data["full_text"]
                article["sections"] = pmc_data.get("sections", [])
                # Prefer PMC abstract if available
                if pmc_data.get("abstract"):
                    article["abstract"] = pmc_data["abstract"]
        except Exception as e:
            print(f"    Warning: Could not fetch PMC full text: {e}")

    return article


def download_pubmed_articles(categories: dict):
    """Download full-text articles and organize by category."""
    os.makedirs(BASE_DIR, exist_ok=True)

    # Track all articles
    all_articles = {}  # pmid -> article data
    skipped_pmids = []  # PMIDs without full text

    # First pass: fetch all unique PMIDs
    unique_pmids = set()
    for pmids in categories.values():
        unique_pmids.update(pmids)

    print(f"Fetching {len(unique_pmids)} unique articles...")
    print("-" * 50)

    for pmid in sorted(unique_pmids):
        try:
            article = fetch_article(pmid)
            if article:
                if article.get("has_full_text"):
                    all_articles[pmid] = article
                    print(f"  {pmid}: [FULL] {article['title'][:45]}...")
                else:
                    skipped_pmids.append(pmid)
                    print(f"  {pmid}: [SKIPPED - no full text] {article['title'][:35]}...")
            else:
                skipped_pmids.append(pmid)
                print(f"  {pmid}: [SKIPPED - no data]")

            time.sleep(0.4)

        except Exception as e:
            skipped_pmids.append(pmid)
            print(f"  {pmid}: [ERROR] {e}")

    # Organize by category and save one JSON per category
    print(f"\nOrganizing into categories...")
    print("-" * 50)

    category_stats = {}

    for category, pmids in categories.items():
        category_path = os.path.join(BASE_DIR, category)
        os.makedirs(category_path, exist_ok=True)

        # Get full-text articles for this category
        category_articles = []
        for pmid in pmids:
            if pmid in all_articles:
                article = all_articles[pmid].copy()
                article["categories"] = [category]
                category_articles.append(article)

        category_stats[category] = len(category_articles)

        # Save single JSON for category
        category_file = os.path.join(category_path, "articles.json")
        category_data = {
            "category": category,
            "article_count": len(category_articles),
            "pmids": [a["pmid"] for a in category_articles],
            "articles": category_articles
        }

        with open(category_file, "w") as f:
            json.dump(category_data, f, indent=2)

        print(f"  {category}: {len(category_articles)} articles")

    # Update category assignments for master file
    for pmid, article in all_articles.items():
        article["categories"] = [
            cat for cat, pmids in categories.items() if pmid in pmids
        ]

    # Save master file with all full-text articles
    master_file = os.path.join(BASE_DIR, "all_articles.json")
    master_data = {
        "metadata": {
            "total_articles": len(all_articles),
            "skipped_no_full_text": len(skipped_pmids),
            "skipped_pmids": skipped_pmids,
            "categories": list(categories.keys()),
            "articles_per_category": category_stats
        },
        "articles": list(all_articles.values())
    }

    with open(master_file, "w") as f:
        json.dump(master_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Full-text articles: {len(all_articles)}")
    print(f"Skipped (no full text): {len(skipped_pmids)}")
    if skipped_pmids:
        print(f"  Skipped PMIDs: {', '.join(skipped_pmids)}")
    print(f"\nMaster file: {master_file}")
    print(f"\nArticles per category:")
    for category, count in category_stats.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    download_pubmed_articles(PROTEOMICS_DATA)