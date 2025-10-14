"""
InsightMate.ai — Sampling-aware CSV upload + Streamlit demo 
(Ask your data anything — private, local, and free.)

MVP feature list:
- 5 MB free cap (MAX_FREE_BYTES)
- Sampling mode for files larger than the cap (DEFAULT_SAMPLE_ROWS)
- Optional stratified sampling
- "Upgrade to analyze full dataset" CTA which triggers a full local read
- No absolute paths: uploaded files stored under tempfile.gettempdir()

Author: InsightMate.ai (Rrahul Ganddhi - rahul.m.gandhi93@gmail.com)
"""

# --- Imports ---
from typing import Optional, Tuple, Dict
import os
import io
import tempfile
import math

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Config ---
MAX_FREE_BYTES = 5 * 1024 * 1024       # 5 MB free cap
DEFAULT_SAMPLE_ROWS = 5000             # default number of rows to analyze in sampling mode
_SAMPLE_CHUNK_MULTIPLIER = 10          # multiplier when reading chunk for stratified sampling
_MAX_CHUNK_ROWS = 100_000              # safety cap to avoid loading huge chunk into memory
_RANDOM_STATE = 42

# --- Sampling helpers ---
def _stratified_sample_df(df: pd.DataFrame, stratify_col: str, sample_n: int) -> pd.DataFrame:
    """
    Proportional stratified sampling from the provided df chunk.
    Ensures at least 1 row per group where feasible.
    """
    if stratify_col not in df.columns:
        raise ValueError(f"Stratify column '{stratify_col}' not found in dataframe chunk.")
    groups = df.groupby(stratify_col)
    group_counts = groups.size().to_dict()
    total = sum(group_counts.values()) or 1

    # compute per-group sample counts (proportional)
    sample_counts = {}
    for g, cnt in group_counts.items():
        n = max(1, int(round(sample_n * (cnt / total))))
        sample_counts[g] = n

    # adjust totals if rounding pushed us over sample_n
    current_total = sum(sample_counts.values())
    if current_total > sample_n:
        # trim from largest sampled groups until we match target
        for g in sorted(sample_counts, key=lambda k: sample_counts[k], reverse=True):
            if current_total <= sample_n:
                break
            reducible = sample_counts[g] - 1
            reduce_by = min(reducible, current_total - sample_n)
            sample_counts[g] -= reduce_by
            current_total -= reduce_by

    # sample from groups
    parts = []
    for g, n in sample_counts.items():
        grp = groups.get_group(g)
        n_use = min(n, len(grp))
        parts.append(grp.sample(n=n_use, replace=False, random_state=_RANDOM_STATE))

    if not parts:
        return df.head(sample_n).reset_index(drop=True)
    return pd.concat(parts).reset_index(drop=True)

# --- Loader: sampling-aware CSV reader ---
def load_csv_with_sampling(path_or_buffer,
                           max_free_bytes: int = MAX_FREE_BYTES,
                           sample_rows: int = DEFAULT_SAMPLE_ROWS,
                           stratify_col: Optional[str] = None
                           ) -> Tuple[pd.DataFrame, Dict]:
    """
    Load CSV with sampling fallback when file size > max_free_bytes.
    Returns (df, meta) where meta contains keys:
      - sampled: bool
      - sample_rows: int
      - full_available: bool
      - file_size: int or None
      - note: human-readable explanation
    Behaviors:
      - If path_or_buffer is a filesystem path and file size <= max_free_bytes -> read full.
      - If file size > max_free_bytes -> read a sample (first N rows by default).
      - If stratify_col provided: read a larger chunk and perform stratified sampling from it.
      - If path_or_buffer is a file-like object: read first sample_rows rows (no file_size).
    """
    meta = {
        "sampled": False,
        "sample_rows": 0,
        "full_available": False,
        "file_size": None,
        "note": ""
    }

    # Case A: filesystem path
    if isinstance(path_or_buffer, str) and os.path.exists(path_or_buffer):
        file_size = os.path.getsize(path_or_buffer)
        meta["file_size"] = file_size
        meta["full_available"] = True

        if file_size <= max_free_bytes:
            # Full load under free cap
            df = pd.read_csv(path_or_buffer)
            meta.update({"sampled": False, "sample_rows": len(df), "note": "Loaded full CSV (within free cap)."})
            return df, meta

        # File exceeds free cap -> sampling mode
        meta["sampled"] = True

        if stratify_col:
            chunk_rows = min(sample_rows * _SAMPLE_CHUNK_MULTIPLIER, _MAX_CHUNK_ROWS)
            try:
                chunk = pd.read_csv(path_or_buffer, nrows=chunk_rows)
            except Exception:
                # fallback to reading sample_rows if chunk read fails
                chunk = pd.read_csv(path_or_buffer, nrows=sample_rows)

            if stratify_col in chunk.columns:
                df_sample = _stratified_sample_df(chunk, stratify_col, sample_rows)
                meta["note"] = f"Stratified sample: {len(df_sample)} rows (from first {len(chunk)} rows)."
            else:
                df_sample = chunk.head(sample_rows)
                meta["note"] = f"Sampling first {len(df_sample)} rows (stratify_col not found in initial chunk)."
            df = df_sample.reset_index(drop=True)
            meta["sample_rows"] = len(df)
            return df, meta
        else:
            # simple first-N rows
            df = pd.read_csv(path_or_buffer, nrows=sample_rows)
            meta["sample_rows"] = len(df)
            meta["note"] = f"Sampling first {len(df)} rows of large file (size={file_size} bytes)."
            return df, meta

    # Case B: file-like buffer / stream (no file size)
    try:
        if hasattr(path_or_buffer, "seek"):
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
        df = pd.read_csv(path_or_buffer, nrows=sample_rows)
        meta["sampled"] = True
        meta["sample_rows"] = len(df)
        meta["file_size"] = None
        meta["full_available"] = False
        meta["note"] = f"Loaded first {len(df)} rows from buffer/stream (no file size metadata)."
        return df, meta
    except Exception as ex:
        raise RuntimeError("Unable to read CSV from provided path_or_buffer: " + str(ex))

# --- Utility: save uploaded file to a temporary path (no absolute paths in code) ---
def save_uploaded_file(uploaded_file) -> str:
    """
    Saves a Streamlit uploaded file (UploadedFile) to a safe temp file and returns the path.
    Note: the caller should manage/clean the file if necessary.
    """
    suffix = ".csv" if uploaded_file.name.lower().endswith(".csv") else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(uploaded_file.getvalue())
        return tf.name  # absolute path under the OS temp dir

# --- Small analysis helpers (demo) ---
def simple_analysis_preview(df: pd.DataFrame) -> Dict:
    """
    Small, non-exhaustive preview analysis used in the demo UI.
    Returns a dict with some useful snippets (head, shape, numeric describe).
    """
    preview = {
        "shape": df.shape,
        "head": df.head(5),
        "dtypes": df.dtypes.to_dict()
    }
    # Try numeric describe
    try:
        preview["describe"] = df.describe().transpose()
    except Exception:
        preview["describe"] = None
    return preview

def generate_pandas_snippet(meta: Dict, sample_rows: int = DEFAULT_SAMPLE_ROWS) -> str:
    """
    Returns a short illustrative pandas code snippet that documents what the app did.
    (This is a human-facing snippet users can copy; not intended for execution as-is.)
    """
    if meta.get("sampled"):
        return (
            "# InsightMate: sampled load\n"
            f"import pandas as pd\n"
            f"df = pd.read_csv('PATH_TO_YOUR_CSV', nrows={meta.get('sample_rows', sample_rows)})\n"
            "# then run your analysis (example)\n"
            "df.head()\n"
        )
    else:
        return (
            "# InsightMate: full load\n"
            "import pandas as pd\n"
            "df = pd.read_csv('PATH_TO_YOUR_CSV')\n"
            "df.head()\n"
        )

# --- Streamlit UI (main) ---
def run_app():
    st.set_page_config(page_title="InsightMate — Upload (Sampling Demo)", layout="wide")
    st.title("InsightMate.ai — Upload (Sampling Demo)")
    st.markdown("Upload a CSV (≤ 5 MB free). Files larger than 5 MB will be analyzed in **sampling mode**.")

    # Upload widget
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    stratify_col_input = st.text_input("Optional: column name to stratify sampling by (leave empty to skip)", value="")

    # persistent storage of uploaded path across reruns
    if 'uploaded_path' not in st.session_state:
        st.session_state['uploaded_path'] = None
        st.session_state['uploaded_name'] = None

    if uploaded is not None:
        # save uploaded file to temp and store path in session_state
        tmp_path = save_uploaded_file(uploaded)
        st.session_state['uploaded_path'] = tmp_path
        st.session_state['uploaded_name'] = uploaded.name
        st.success(f"Saved uploaded file to temporary path (for this session). Filename: {uploaded.name}")

    csv_path = st.session_state.get('uploaded_path', None)

    if csv_path:
        # Load dataset with sampling-aware loader
        stratify_col = stratify_col_input.strip() or None
        df, meta = load_csv_with_sampling(csv_path, max_free_bytes=MAX_FREE_BYTES,
                                          sample_rows=DEFAULT_SAMPLE_ROWS, stratify_col=stratify_col)

        # Attach metadata to df object for downstream use
        try:
            df._insightmate_meta = meta
        except Exception:
            df = df.copy()
            df._insightmate_meta = meta

        # Show meta and preview
        st.subheader("Load details")
        st.write("**Metadata:**")
        st.json(meta)

        st.subheader("Data preview")
        st.dataframe(df.head(100))

        # Show a small analysis preview
        preview = simple_analysis_preview(df)
        st.subheader("Quick preview stats")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Shape:", preview["shape"])
            st.write("Dtypes:")
            st.write(preview["dtypes"])
        with col2:
            st.write("Describe (numeric):")
            if preview["describe"] is not None:
                st.dataframe(preview["describe"].head(20))
            else:
                st.write("No numeric columns or describe failed.")

        # Chart — pick first numeric column and plot a histogram (demo)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.subheader("Auto chart (histogram of first numeric column)")
            fig, ax = plt.subplots()
            ax.hist(df[numeric_cols[0]].dropna(), bins=30)
            ax.set_title(f"Histogram — {numeric_cols[0]}")
            st.pyplot(fig)
        else:
            st.info("No numeric column found for an automatic chart demo.")

        # Provide pandas snippet to reproduce what happened locally
        st.subheader("Copyable pandas snippet")
        st.code(generate_pandas_snippet(meta, sample_rows=DEFAULT_SAMPLE_ROWS), language="python")

        # If sampled, offer Upgrade CTA
        if meta.get("sampled"):
            st.warning(
                f"This dataset was loaded in **sampling mode** ({meta.get('sample_rows')} rows). "
                "Analyze the full dataset only if you accept the local read (may be large)."
            )
            if st.button("Upgrade to analyze full dataset (local read)"):
                # Attempt to load full dataset (local read). This is intentionally explicit action.
                with st.spinner("Loading full dataset into memory..."):
                    try:
                        full_df = pd.read_csv(csv_path)
                        full_df._insightmate_meta = {"sampled": False, "sample_rows": len(full_df),
                                                    "full_available": True, "file_size": meta.get("file_size"),
                                                    "note": "User opted to load full dataset locally."}
                        st.success(f"Full dataset loaded: {full_df.shape[0]} rows, {full_df.shape[1]} columns.")
                        # Replace df with full_df for further exploration in this session
                        st.session_state['_full_df'] = full_df
                        # Display first rows
                        st.dataframe(full_df.head(100))
                    except Exception as e:
                        st.error(f"Failed to load full dataset: {e}")

        # If user previously loaded full_df, show it as option to switch to
        if '_full_df' in st.session_state:
            st.info("Full dataset is available in this session.")
            if st.button("Analyze using full dataset in session"):
                full_df = st.session_state['_full_df']
                st.dataframe(full_df.head(100))
                st.write("Shape:", full_df.shape)

    else:
        st.info("No CSV uploaded yet. Use the uploader above to start.")

    # Footer / notes
    st.markdown("---")
    st.markdown(
        "Notes:\n"
        "- This demo saves uploads to the OS temp directory for the current session (no absolute paths in the code).\n"
        "- Sampling mode provides a fast, privacy-preserving way to explore large files without reading everything.\n"
        "- Integrate this loader into your analysis pipeline and use `df._insightmate_meta` to show provenance in the UI."
    )

# --- Entrypoint ---
if __name__ == "__main__":
    run_app()
