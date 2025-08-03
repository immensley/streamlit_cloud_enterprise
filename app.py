
import streamlit as st
import pandas as pd
from fpdf import FPDF
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from datetime import date, timedelta
from typing import List, Dict

st.set_page_config(page_title="Etsy Bridge (Enterprise+)", layout="wide")
st.title("Etsy Bridge (Enterprise+)")
st.caption("GSC + Trends → Opportunities → Coverage → A/B → UTM → Client Report")

# -------------------- Sample fallback data --------------------
@st.cache_data
def load_sample():
    queries = pd.read_csv("data/queries.csv")
    listings = pd.read_csv("data/listings.csv")
    opps = pd.read_csv("data/opportunities.csv")
    return queries, listings, opps

queries_sample, listings, opps_sample = load_sample()

# -------------------- GSC client --------------------
SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

def gsc_client_from_secrets():
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        sa_info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
        return build("searchconsole", "v1", credentials=creds, cache_discovery=False)
    except Exception:
        return None

def gsc_query(site_url: str, start_date: str, end_date: str, row_limit: int = 2500) -> pd.DataFrame:
    svc = gsc_client_from_secrets()
    if not svc:
        return queries_sample.copy()
    body = {"startDate": start_date, "endDate": end_date, "dimensions": ["query"], "rowLimit": row_limit, "startRow": 0}
    try:
        resp = svc.searchanalytics().query(siteUrl=site_url, body=body).execute()
        rows = resp.get("rows", [])
        data = [{"query": r["keys"][0], "impressions": r.get("impressions", 0), "clicks": r.get("clicks", 0), "ctr": r.get("ctr", 0.0), "position": r.get("position", 0.0)} for r in rows if r.get("keys")]
        df = pd.DataFrame(data)
        return df.sort_values("impressions", ascending=False).reset_index(drop=True) if not df.empty else queries_sample.copy()
    except Exception as e:
        st.error(f"GSC API error: {e}")
        return queries_sample.copy()

# -------------------- Trends (Google via pytrends; Pinterest via CSV/paste) --------------------
def google_trends(seed_terms: list, geo: str = "US", timeframe: str = "today 3-m") -> Dict[str, pd.DataFrame]:
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        cat = 7  # Shopping
        geo_code = "" if geo == "Worldwide" else geo
        pytrends.build_payload(seed_terms, cat=cat, timeframe=timeframe, geo=geo_code, gprop="")
        iot = pytrends.interest_over_time().reset_index()
        related = pytrends.related_queries()
        rising_frames = [rq['rising'].assign(seed=seed) for seed, rq in related.items() if rq and rq.get('rising') is not None and not rq['rising'].empty]
        rising = pd.concat(rising_frames, ignore_index=True) if rising_frames else pd.DataFrame(columns=["query","value","seed"])
        return {"iot": iot, "rising": rising}
    except Exception as e:
        st.error(f"Google Trends error: {e}")
        return {"iot": pd.DataFrame(), "rising": pd.DataFrame(columns=["query","value","seed"])}

# -------------------- Heuristics --------------------
def build_utm_url(url, source, medium, campaign, term=None, content=None):
    parts = list(urlparse(url))
    query = dict(parse_qsl(parts[4], keep_blank_values=True))
    query["utm_source"] = source
    query["utm_medium"] = medium
    query["utm_campaign"] = campaign
    if term: query["utm_term"] = term
    if content: query["utm_content"] = content
    parts[4] = urlencode(query, doseq=True)
    return urlunparse(parts)

def map_to_listing(term: str, listings_df: pd.DataFrame):
    term_l = term.lower()
    best = None; score = 0
    for _, row in listings_df.iterrows():
        t = str(row['title']).lower()
        s = sum(1 for w in term_l.split() if w in t)
        if s > score:
            score, best = s, row
    if score >= 2 and best is not None:
        return {"term": term, "action": "Update existing listing (use exact phrase)", "listing_id": best['listing_id'], "listing_title": best['title']}
    else:
        return {"term": term, "action": "Create new listing targeting phrase", "listing_id": None, "listing_title": None}

def heuristic_title(term: str) -> str:
    return f"{term.title()} – Printable Template (Digital Download)"

def heuristic_tags(term: str):
    base = term.lower().strip()
    words = [w for w in base.split() if len(w) > 2][:5]
    tags = [base, f"{base} printable", f"{base} template", f"digital {words[0] if words else 'download'}", "instant download"]
    # dedupe
    out, seen = [], set()
    for t in tags:
        if t not in seen: out.append(t); seen.add(t)
    return out[:13]

def merge_gsc_trends(gsc_df: pd.DataFrame, rising_df: pd.DataFrame) -> pd.DataFrame:
    g = gsc_df.copy(); r = rising_df.copy()
    if "value" not in r.columns: r["value"] = 0
    g["query_l"] = g["query"].str.lower(); r["query_l"] = r["query"].str.lower()
    rows = []
    for _, gr in g.iterrows():
        q = gr["query_l"]
        matches = r[r["query_l"].str.contains(q)]
        if matches.empty: matches = r[r["query_l"].apply(lambda x: q in x)]
        if matches.empty: continue
        top = matches.sort_values("value", ascending=False).head(1).iloc[0]
        score = (gr.get("impressions", 0) + 1) * (top.get("value", 0) + 1)
        rows.append({"query": gr["query"], "impressions": gr.get("impressions", 0), "position": gr.get("position", 0.0), "trend_term": top["query"], "trend_value": top.get("value", 0), "opportunity_score": score})
    return pd.DataFrame(rows).sort_values("opportunity_score", ascending=False).reset_index(drop=True)

# -------------------- New: Exact Phrase Coverage --------------------
def exact_phrase_coverage(listings_df: pd.DataFrame, phrases: List[str]) -> pd.DataFrame:
    rows = []
    for _, l in listings_df.iterrows():
        title = str(l["title"]).lower()
        tags = str(l.get("tags","")).lower()
        for p in phrases:
            phrase = p.lower().strip()
            covered = (phrase in title) or (phrase in tags)
            rows.append({"listing_id": l["listing_id"], "listing_title": l["title"], "phrase": p, "covered": covered, "where": "title/tags" if covered else "—"})
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["listing_id","listing_title"], values="covered", aggfunc="sum")
    pivot = pivot.rename(columns={"covered":"phrases_covered"}).reset_index()
    return df, pivot

# -------------------- New: Title/Tag A/B Generator --------------------
def generate_ab_variants(term: str) -> Dict[str, Dict[str, str]]:
    t = term.strip().lower()
    a_title = f"{term.title()} – Printable Template (Digital Download)"
    b_title = f"{term.title()} (Instant Download) – Editable PDF/PNG"
    a_tags = ", ".join([t, f"{t} printable", f"{t} template", "instant download", "digital download"])
    b_tags = ", ".join([f"editable {t}", f"{t} pdf", f"{t} png", "printable template", "downloadable"])
    return {"A": {"title": a_title, "tags": a_tags}, "B": {"title": b_title, "tags": b_tags}}

# -------------------- New: One‑Click Client Report --------------------
def pdf_report(opportunities: pd.DataFrame, coverage_pivot: pd.DataFrame, utm_plan: pd.DataFrame, filename: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Etsy Bridge Client Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Date: {date.today().isoformat()}", ln=True)
    def add_table(df: pd.DataFrame, title: str, limit: int = 20):
        pdf.set_font("Helvetica", "B", 13); pdf.ln(4); pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Helvetica", "", 9)
        view = df.head(limit).copy()
        cols = list(view.columns)
        page_width = pdf.w - 2*pdf.l_margin
        col_w = page_width / max(1, len(cols))
        # header
        for c in cols: pdf.cell(col_w, 6, c[:18], border=1)
        pdf.ln(6)
        # rows
        for _, row in view.iterrows():
            for c in cols:
                txt = str(row[c])
                if len(txt) > 24: txt = txt[:21] + "..."
                pdf.cell(col_w, 6, txt, border=1)
            pdf.ln(6)
    if not opportunities.empty: add_table(opportunities, "Prioritized Opportunities")
    if not coverage_pivot.empty: add_table(coverage_pivot, "Exact Phrase Coverage (count per listing)")
    if not utm_plan.empty: add_table(utm_plan, "UTM Plan (first 20)")
    pdf.output(filename)
    return filename

# -------------------- UI --------------------
tabs = st.tabs(["GSC × Trends", "Coverage", "A/B Generator", "UTM (single)", "UTM (bulk)", "Client Report", "Listings", "Sample Exports"])
(t1, t2, t3, t4, t5, t6, t7, t8) = tabs

with t1:
    st.subheader("Prioritize with Opportunity Score")
    site = st.text_input("GSC Property (https://yourdomain.com)", placeholder="https://yourdomain.com")
    seeds = st.text_input("Seed keywords", value="wedding menu, a5 planner inserts, minimalist wall art")
    geo = st.selectbox("Geo", ["US","GB","CA","AU","Worldwide"], index=0)
    tf = st.selectbox("Timeframe", ["now 7-d","today 1-m","today 3-m","today 12-m"], index=2)
    days = st.slider("GSC lookback days", 30, 180, 90)
    if st.button("Run"):
        end = date.today(); start = end - timedelta(days=days)
        gsc = gsc_query(site or "https://example.com", start.isoformat(), end.isoformat())
        trends = google_trends([s.strip() for s in seeds.split(",") if s.strip()], geo=geo, timeframe=tf)
        opps = merge_gsc_trends(gsc, trends["rising"])
        st.dataframe(opps.head(100), use_container_width=True)
        st.download_button("Download Opportunities CSV", opps.to_csv(index=False), file_name="opportunities.csv")

with t2:
    st.subheader("Exact Phrase Coverage")
    targets = st.text_area("Target phrases (one per line)", value="printable wedding menu template\na5 planner inserts printable\nminimalist wall art printable", height=120)
    if st.button("Check Coverage"):
        phrases = [t.strip() for t in targets.splitlines() if t.strip()]
        detail, pivot = exact_phrase_coverage(listings, phrases)
        st.markdown("**Per‑phrase detail**")
        st.dataframe(detail, use_container_width=True)
        st.markdown("**Coverage per listing (count of phrases covered)**")
        st.dataframe(pivot, use_container_width=True)
        st.session_state["coverage_pivot"] = pivot
        st.download_button("Download Coverage Detail CSV", detail.to_csv(index=False), file_name="coverage_detail.csv")
        st.download_button("Download Coverage Summary CSV", pivot.to_csv(index=False), file_name="coverage_summary.csv")

with t3:
    st.subheader("Title/Tag A/B Generator")
    focus = st.text_input("Focus term", value="printable wedding menu template")
    if st.button("Generate A/B"):
        ab = generate_ab_variants(focus)
        st.write("**Variant A — Title**"); st.code(ab["A"]["title"])
        st.write("**Variant A — Tags**"); st.code(ab["A"]["tags"])
        st.write("**Variant B — Title**"); st.code(ab["B"]["title"])
        st.write("**Variant B — Tags**"); st.code(ab["B"]["tags"])
        st.info("Tip: Track CTR in Etsy Shop Stats; pick winner after ~200 impressions.")

with t4:
    st.subheader("UTM Builder (single)")
    etsy_url = st.text_input("Etsy URL", placeholder="https://www.etsy.com/listing/123456/item")
    chan = st.selectbox("Preset", ["Pinterest (organic)","Pinterest Ads","Google (organic)","Google Ads (Search)","Instagram (organic)","Instagram Ads"])
    presets = {"Pinterest (organic)":"pinterest|social","Pinterest Ads":"pinterest|cpc","Google (organic)":"google|organic","Google Ads (Search)":"google|cpc","Instagram (organic)":"instagram|social","Instagram Ads":"instagram|cpc"}
    src, med = presets[chan].split("|")
    camp = st.text_input("utm_campaign", value=f"launch_{date.today().isoformat()}")
    term = st.text_input("utm_term (optional)"); content = st.text_input("utm_content (optional)")
    if st.button("Generate UTM"):
        if not etsy_url: st.warning("Enter a URL")
        else:
            tagged = build_utm_url(etsy_url, src, med, camp, term or None, content or None)
            st.code(tagged); st.download_button("Download .txt", data=tagged.encode(), file_name="utm_link.txt")

with t5:
    st.subheader("UTM Builder (bulk)")
    urls = st.text_area("Etsy URLs (one per line)", height=150)
    chan_b = st.selectbox("Preset (bulk)", ["Pinterest (organic)","Pinterest Ads","Google (organic)","Google Ads (Search)","Instagram (organic)","Instagram Ads"])
    presets = {"Pinterest (organic)":"pinterest|social","Pinterest Ads":"pinterest|cpc","Google (organic)":"google|organic","Google Ads (Search)":"google|cpc","Instagram (organic)":"instagram|social","Instagram Ads":"instagram|cpc"}
    src_b, med_b = presets[chan_b].split("|")
    camp_b = st.text_input("utm_campaign (bulk)", value=f"campaign_{date.today().isoformat()}")
    term_b = st.text_input("utm_term (optional, bulk)"); content_b = st.text_input("utm_content (optional, bulk)")
    if st.button("Generate UTM CSV"):
        url_list = [u.strip() for u in urls.splitlines() if u.strip()]
        if not url_list: st.warning("Paste at least one URL")
        else:
            rows = [{"original_url": u, "utm_source": src_b, "utm_medium": med_b, "utm_campaign": camp_b, "utm_term": term_b, "utm_content": content_b, "tagged_url": build_utm_url(u, src_b, med_b, camp_b, term_b or None, content_b or None)} for u in url_list]
            df = pd.DataFrame(rows); st.dataframe(df, use_container_width=True)
            st.session_state["utm_plan"] = df
            st.download_button("Download UTM CSV", df.to_csv(index=False), file_name="utm_links.csv")

with t6:
    st.subheader("One‑Click Client Report (PDF)")
    opps = opps_sample.copy()  # plug your merged opportunities here after running t1
    pivot = st.session_state.get("coverage_pivot", pd.DataFrame())
    utm_plan = st.session_state.get("utm_plan", pd.DataFrame())
    if st.button("Generate Client PDF"):
        path = pdf_report(opps, pivot, utm_plan, "client_report.pdf")
        with open(path, "rb") as f:
            st.download_button("Download Client Report PDF", data=f, file_name="client_report.pdf")

with t7:
    st.subheader("Listings (sample)")
    st.dataframe(listings, use_container_width=True)

with t8:
    st.subheader("Sample Exports")
    if st.button("Export Opportunities CSV (sample)"):
        path = "opportunities.csv"; opps_sample.to_csv(path, index=False)
        with open(path, "rb") as f: st.download_button("Download", data=f, file_name="opportunities.csv")
