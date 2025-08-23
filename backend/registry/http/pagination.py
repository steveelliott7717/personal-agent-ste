# backend/registry/http/pagination.py

from __future__ import annotations

from typing import Any, Dict, List, Callable, Optional
from backend.registry.util.jsonptr import json_pointer_get
from backend.registry.util.url import url_with_param, looks_like_url
from backend.registry.http.headers import get_ci, parse_link_next
import time
import random


def _maybe_sleep(delay_ms: int, jitter_ms: int) -> None:
    if delay_ms or jitter_ms:
        d = delay_ms
        if jitter_ms:
            d += random.randint(0, max(0, jitter_ms))
        time.sleep(d / 1000.0)


def _page_meta_from(resp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": int(resp.get("status", 0)),
        "url": str(resp.get("final_url") or resp.get("url") or ""),
        "bytes_len": int(resp.get("bytes_len", 0)),
    }


def paginate_response(
    first_resp: Dict[str, Any], args: Dict[str, Any], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build pagination over an already-executed first response.

    Parameters
    ----------
    first_resp : dict
        The first HTTP response: expects keys status, final_url/url, headers, bytes_len, json.
    args : dict
        Original http.fetch args (used mainly to read paginate config defaults).
    cfg : dict
        Runtime config:
          - fetch: Callable[[str], Dict[str, Any]]  # REQUIRED: fetch a subsequent page via GET
          - mode: "cursor" | "header_link" | "page"
          - json_pointer: str                (cursor)
          - cursor_param: str = "cursor"     (cursor)
          - header: str = "Link"             (header_link)
          - rel: str = "next"                (header_link, for RFC5988 Link)
          - page_param: str = "page"         (page)
          - start_page: int = 1              (page)
          - max_pages: int = 10
          - items_pointer: str | None
          - concat_json: bool = False
          - delay_ms: int = 0
          - delay_jitter_ms: int = 0

    Returns
    -------
    dict with:
      {
        "pages": [{"status","url","bytes_len"}, ...],
        "pages_count": N,
        # when concat_json + items_pointer:
        "items": [...],
        "items_count": M
      }
    """
    fetch: Optional[Callable[[str], Dict[str, Any]]] = cfg.get("fetch")
    if not callable(fetch):
        # No fetcher → nothing to paginate
        return {"pages": [_page_meta_from(first_resp)], "pages_count": 1}

    mode = str(cfg.get("mode") or "cursor").lower()
    max_pages = int(cfg.get("max_pages", 10))
    items_ptr = cfg.get("items_pointer")
    concat_json = bool(cfg.get("concat_json", False))
    delay_ms = int(cfg.get("delay_ms", 0))
    delay_jitter_ms = int(cfg.get("delay_jitter_ms", 0))

    final_url_1 = str(first_resp.get("final_url") or first_resp.get("url") or "")
    pages: List[Dict[str, Any]] = [_page_meta_from(first_resp)]
    items: List[Any] = []

    if concat_json and items_ptr and isinstance(first_resp.get("json"), (dict, list)):
        first_items = json_pointer_get(first_resp["json"], items_ptr)
        if isinstance(first_items, list):
            items.extend(first_items)

    # Helper: early-stop rule while still recording the page (status already captured)
    def _should_stop_after(resp: Dict[str, Any]) -> bool:
        status = int(resp.get("status", 0))
        if status == 204:
            return True
        # Empty body stop:
        if int(resp.get("bytes_len", 0)) == 0:
            return True
        if concat_json and items_ptr and isinstance(resp.get("json"), (dict, list)):
            arr = json_pointer_get(resp["json"], items_ptr)
            if not isinstance(arr, list) or len(arr) == 0:
                return True
        return False

    # ---------- cursor mode ----------
    if mode == "cursor" and isinstance(first_resp.get("json"), (dict, list)):
        ptr = cfg.get("json_pointer")
        cursor_param = cfg.get("cursor_param", "cursor")

        next_ref = json_pointer_get(first_resp["json"], ptr) if ptr else None
        page_idx = 1
        while next_ref and page_idx < max_pages:
            page_idx += 1
            next_url = (
                str(next_ref)
                if looks_like_url(next_ref)
                else url_with_param(final_url_1, cursor_param, str(next_ref))
            )

            # fetch next page
            resp_n = fetch(next_url)
            pages.append(_page_meta_from(resp_n))

            # collect items if configured
            if (
                concat_json
                and items_ptr
                and isinstance(resp_n.get("json"), (dict, list))
            ):
                nxt_items = json_pointer_get(resp_n["json"], items_ptr)
                if isinstance(nxt_items, list):
                    items.extend(nxt_items)

            # decide whether to stop after recording
            if _should_stop_after(resp_n):
                break

            # advance cursor
            next_ref = (
                json_pointer_get(resp_n.get("json"), ptr)
                if ptr and isinstance(resp_n.get("json"), (dict, list))
                else None
            )

            # polite delay before another hop
            if next_ref:
                _maybe_sleep(delay_ms, delay_jitter_ms)

    # ---------- header_link mode ----------
    elif mode == "header_link":
        header_name = str(cfg.get("header") or "Link")
        want_rel = str(cfg.get("rel") or "next")

        page_idx = 1
        # find first "next" from the initial page
        link_val = get_ci(first_resp.get("headers") or {}, header_name)
        if link_val and header_name.lower() == "link":
            next_url = parse_link_next(link_val, want_rel)
        elif link_val and looks_like_url(link_val):
            next_url = link_val
        else:
            next_url = None

        while next_url and page_idx < max_pages:
            page_idx += 1
            resp_n = fetch(next_url)
            pages.append(_page_meta_from(resp_n))

            if (
                concat_json
                and items_ptr
                and isinstance(resp_n.get("json"), (dict, list))
            ):
                nxt_items = json_pointer_get(resp_n["json"], items_ptr)
                if isinstance(nxt_items, list):
                    items.extend(nxt_items)

            if _should_stop_after(resp_n):
                break

            # read next from this page
            link_val = get_ci(resp_n.get("headers") or {}, header_name)
            if link_val and header_name.lower() == "link":
                next_url = parse_link_next(link_val, want_rel)
            elif link_val and looks_like_url(link_val):
                next_url = link_val
            else:
                next_url = None

            if next_url:
                _maybe_sleep(delay_ms, delay_jitter_ms)

    # ---------- page mode ----------
    elif mode == "page":
        page_param = str(cfg.get("page_param", "page"))
        start_page = int(cfg.get("start_page", 1))

        page_idx = 1
        cur_page = start_page + 1  # we assume caller fetched start_page already
        while page_idx < max_pages:
            page_idx += 1
            next_url = url_with_param(final_url_1, page_param, str(cur_page))
            cur_page += 1

            resp_n = fetch(next_url)
            pages.append(_page_meta_from(resp_n))

            if (
                concat_json
                and items_ptr
                and isinstance(resp_n.get("json"), (dict, list))
            ):
                nxt_items = json_pointer_get(resp_n["json"], items_ptr)
                if isinstance(nxt_items, list):
                    items.extend(nxt_items)

            if _should_stop_after(resp_n):
                break

            # polite delay before another numbered page
            if page_idx < max_pages:
                _maybe_sleep(delay_ms, delay_jitter_ms)

    # Package result
    out: Dict[str, Any] = {"pages": pages, "pages_count": len(pages)}
    if concat_json and items_ptr:
        out["items"] = items
        out["items_count"] = len(items)
    return out
