from __future__ import annotations

import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False


def maybe_render_chart(st, df: pd.DataFrame, chart_spec: dict) -> None:
    if df is None or df.empty:
        return
    if not chart_spec or chart_spec.get("type") in (None, "", "none"):
        return

    ctype = chart_spec.get("type")
    x = chart_spec.get("x")
    y = chart_spec.get("y")
    title = chart_spec.get("title") or ""
    bins = chart_spec.get("bins")

    if ctype not in ("bar", "line", "hist"):
        return

    FIG_W, FIG_H = 6.8, 3.2
    DPI = 120

    if ctype in ("bar", "line"):
        if not x or not y:
            return
        if x not in df.columns or y not in df.columns:
            return

        if _HAS_MPL:
            fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
            ax = fig.add_subplot(111)

            if ctype == "bar":
                ax.bar(df[x].astype(str), df[y])
                ax.tick_params(axis="x", labelrotation=30)
            else:
                ax.plot(df[x], df[y])

            if title:
                ax.set_title(title)
            ax.set_xlabel(x)
            ax.set_ylabel(y)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            try:
                series = df.set_index(df[x].astype(str))[y]
                if ctype == "bar":
                    st.bar_chart(series, use_container_width=True)
                else:
                    st.line_chart(series, use_container_width=True)
            except Exception:
                st.info("Chart rendering is unavailable (matplotlib is not installed).")
        return

    if ctype == "hist":
        if not y or y not in df.columns:
            return
        series = pd.to_numeric(df[y], errors="coerce").dropna()
        if series.empty:
            return

        if _HAS_MPL:
            fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
            ax = fig.add_subplot(111)

            if isinstance(bins, int) and bins > 0:
                ax.hist(series, bins=bins)
            else:
                ax.hist(series)

            if title:
                ax.set_title(title)
            ax.set_xlabel(y)
            ax.set_ylabel("count")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Histogram rendering is unavailable (matplotlib is not installed).")
