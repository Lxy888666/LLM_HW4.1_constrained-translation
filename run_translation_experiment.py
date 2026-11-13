"""Stratified development/test experiment for constrained English-to-Chinese translation."""
from __future__ import annotations

import concurrent.futures
import json
import math
import random
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import dotenv_values
from matplotlib import font_manager
from openai import OpenAI
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
DATA = ROOT / "data" / "translation_errors.json"
SEED = 20261114
OLD_SEED = 20261113
OLD_PILOT_N = 30
TEMP = 0.7
WORKERS = 4
MIN_DEV_N_FOR_PROMPT_SELECTION = 3
LOCK = threading.Lock()

CATEGORIES = ["术语与指代", "数字与单位", "否定范围", "条件与顺序", "情态与规范强度", "其他语义问题"]
DEV_QUOTA = {"术语与指代": 5, "数字与单位": 0, "否定范围": 5,
             "条件与顺序": 5, "情态与规范强度": 5, "其他语义问题": 9}
TEST_QUOTA = {"术语与指代": 10, "数字与单位": 2, "否定范围": 9,
              "条件与顺序": 9, "情态与规范强度": 9, "其他语义问题": 18}

PROMPTS = {
    "plain": (
        "Translate the English text into accurate, concise Simplified Chinese. Preserve the meaning and return only the translation."
    ),
    "baseline": (
        "Translate the English text into accurate, concise Simplified Chinese. Preserve every stated fact and relation. "
        "Check technical terms and referents; numbers, units, signs, ranges, and their objects; negation and scope; "
        "conditions, exceptions, and order; the strength of obligations, recommendations, permissions, and possibilities; "
        "and all risk, safety, and performance limits. Do not omit restrictions or add claims, guarantees, causes, or actions "
        "that the source does not support. Return only the translation."
    ),
    "fact_map": (
        "Translate the English text into accurate, concise Simplified Chinese. Preserve all source-supported meaning. "
        "Before drafting, identify the source propositions and their links: who does what to which object; each technical term "
        "and referent; each number, sign, unit, range, and the object it describes; negation and quantifier scope; conditions, "
        "exceptions, and sequence; and the strength of requirements, recommendations, permissions, and possibilities. "
        "After drafting, check that every identified link is preserved and that no unsupported claim was added. "
        "Do not show the checklist. Return only the final Simplified Chinese translation."
    ),
    "self_audit": (
        "Translate the English text into accurate, concise Simplified Chinese. Preserve every stated fact, relation, and limit. "
        "Check technical terms and referents; numbers, units, signs, ranges, and their objects; negation and scope; "
        "conditions, exceptions, and order; modality and risk strength; and omissions or unsupported additions. "
        "Then compare your draft against the English source once more and make only corrections supported by that source. "
        "Do not strengthen, weaken, or broaden a claim for stylistic reasons. Return only the final translation."
    ),
}
CHALLENGERS = ["baseline", "fact_map", "self_audit"]

ISSUE_PATTERNS = {
    "数字与单位": re.compile(r"数字(?:错误|遗漏|变化|改变|对应|顺序|不一致|误译)|数值(?:错误|遗漏|变化|改变|对象|对应|单位|范围)|数字单位|数值单位|单位(?:换算|对应|错误|遗漏|误译)|数值范围|计量单位|测量值|量程|阈值|百分比|正负号|小数|温度|电压|压力|分母|分子|%", re.I),
    "否定范围": re.compile(r"否定|否认|否定范围|否定词|并非|不等于|不是所有|没有任何|未能|未曾|不可|不得|不应|无需|不必|no\b|not\b|never\b|neither\b|none\b", re.I),
    "情态与规范强度": re.compile(r"情态|模态|规范强度|义务|建议|许可|强制|可能性|可能|应当|必须|可以|允许|承诺|保证|\bshould\b|\bshall\b|\bmust\b|\bmay\b|\bcan\b", re.I),
    "条件与顺序": re.compile(r"条件|顺序|先后|前提|例外|因果|前件|后件|触发|必要条件|充分条件|\bunless\b|\bonly if\b|\bprovided\b|\bbefore\b|\bafter\b|\bwhen\b", re.I),
    "术语与指代": re.compile(r"术语|指代|修饰|技术对象|专名|译名|词义|概念|学科|类别|含义|对应关系|所指|定义|译法|名词|词缀|词素|语法|介词|动词|\bterm\b|\bword\b|\bpart of speech\b", re.I),
}
PRIORITY = ["数字与单位", "否定范围", "情态与规范强度", "条件与顺序", "术语与指代"]

TARGET_EVAL_PROMPT = (
    "Assess each Chinese translation against its English source and the single annotated issue. "
    "Decide whether it fixes that specific issue while remaining faithful to the source. The reference is an example of a valid "
    "repair, not required wording. Do not use overall fluency to excuse a meaning error. Use FIXED, NOT_FIXED, or UNCERTAIN. "
    "Judge candidates independently and return JSON only: {\"evaluations\":[{\"id\":\"A\",\"status\":\"FIXED|NOT_FIXED|UNCERTAIN\",\"reason\":\"brief source-grounded evidence\"}]}"
)
QUALITY_EVAL_PROMPT = """Evaluate each Chinese translation against its English source, blind to the generation condition. Check technical terms and referents; numbers, units, signs, ranges and their objects; negation and scope; conditions, exceptions and order; modality, risk, safety and performance limits; and omissions or unsupported additions. Mark each dimension PASS, FAIL, NA or UNCERTAIN. Give semantic completeness and Chinese readability integer scores from 1 to 5. A translation passes only if no dimension is FAIL or UNCERTAIN, semantic completeness is at least 4, and readability is at least 3. Return JSON only: {"evaluations":[{"id":"A","dimensions":{"terms_referents":"PASS|FAIL|NA|UNCERTAIN","numbers_units":"PASS|FAIL|NA|UNCERTAIN","negation_scope":"PASS|FAIL|NA|UNCERTAIN","conditions_order":"PASS|FAIL|NA|UNCERTAIN","modality_risk":"PASS|FAIL|NA|UNCERTAIN","omissions_additions":"PASS|FAIL|NA|UNCERTAIN"},"semantic_completeness":1,"readability":1,"notes":"brief"}]}"""


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text))


def classify_issue(note: str, source: str) -> str:
    # Classify from the first diagnosis clause, ignoring unrelated examples or
    # details later in the annotation (for example, a modal error note may also
    # mention that the source number was preserved).
    primary_note = re.split(r"[。；;\n]", note, maxsplit=1)[0]
    numeric_source = re.sub(r"\(\s*\d+(?:\.\d+)*\s*\)|\[\s*\d+\s*\]", "", source)
    numeric_source = re.sub(r"\bISO(?:/IEC)?\s*\d+(?:-\d+)+", "", numeric_source, flags=re.I)
    has_measure = bool(re.search(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|°\s?[CF]|(?:mm|cm|km|kg|g|mg|ml|l|min|s|h|v|a|kpa|mpa|hz|w|kw|mb|gb)\b)",
        numeric_source, re.I))
    matches = []
    for priority, category in enumerate(PRIORITY):
        if category == "数字与单位" and not has_measure:
            continue
        match = ISSUE_PATTERNS[category].search(primary_note)
        if match:
            matches.append((match.start(), priority, category))
    return min(matches)[2] if matches else "其他语义问题"


def load_eligible() -> list[dict]:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    unique: dict[str, dict] = {}
    for row in rows:
        source, incorrect, note, reference = list(row.values())
        if not (8 <= word_count(source) <= 70 and len(source) <= 600 and len(reference) <= 1000):
            continue
        unique.setdefault(source, {"source": source, "known_incorrect": incorrect,
                                   "error_note": note, "reference": reference,
                                   "issue_type": classify_issue(note, source)})
    return list(unique.values())


def make_splits() -> tuple[list[dict], list[dict], dict]:
    eligible = load_eligible()
    legacy_order = eligible.copy()
    random.Random(OLD_SEED).shuffle(legacy_order)
    legacy_sources = {row["source"] for row in legacy_order[:OLD_PILOT_N]}
    remaining = [row.copy() for row in eligible if row["source"] not in legacy_sources]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in remaining:
        buckets[row["issue_type"]].append(row)
    dev: list[dict] = []
    test: list[dict] = []
    counts = {"eligible_unique": len(eligible), "previously_examined_excluded": len(legacy_sources),
              "remaining": len(remaining), "remaining_by_type": {c: len(buckets[c]) for c in CATEGORIES}}
    for index, category in enumerate(CATEGORIES):
        pool = buckets[category].copy()
        random.Random(SEED + index * 997).shuffle(pool)
        need_dev, need_test = DEV_QUOTA[category], TEST_QUOTA[category]
        if len(pool) < need_dev + need_test:
            raise RuntimeError(f"Insufficient examples for {category}: need {need_dev + need_test}, found {len(pool)}")
        for row in pool[:need_dev]:
            row.update({"split": "development", "sample_id": f"D{len(dev)+1:03d}"})
            dev.append(row)
        for row in pool[need_dev:need_dev + need_test]:
            row.update({"split": "test", "sample_id": f"H{len(test)+1:03d}"})
            test.append(row)
    counts["development_by_type"] = dict(Counter(row["issue_type"] for row in dev))
    counts["test_by_type"] = dict(Counter(row["issue_type"] for row in test))
    if {x["source"] for x in dev} & {x["source"] for x in test}:
        raise RuntimeError("Development/test source overlap detected")
    if ({x["source"] for x in dev} | {x["source"] for x in test}) & legacy_sources:
        raise RuntimeError("Previously examined pilot source leaked into a new split")
    return dev, test, counts


def load_client():
    cfg = dotenv_values(ROOT / ".env")
    if not cfg.get("key") or not cfg.get("URL") or not cfg.get("model"):
        raise RuntimeError(".env must contain key, URL, and model")
    base = cfg["URL"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return OpenAI(api_key=cfg["key"], base_url=base, timeout=90, max_retries=0), cfg["model"], base


def request(client, model, messages, temperature, limit, json_mode=False):
    started = time.perf_counter()
    kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": limit,
              "extra_body": {"thinking": {"type": "disabled"}}}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    error = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            usage = response.usage
            text = choice.message.content or ""
            return {"ok": bool(text.strip()), "text": text, "finish": choice.finish_reason,
                    "usage": {"prompt": getattr(usage, "prompt_tokens", 0) if usage else 0,
                              "completion": getattr(usage, "completion_tokens", 0) if usage else 0,
                              "total": getattr(usage, "total_tokens", 0) if usage else 0},
                    "seconds": round(time.perf_counter() - started, 3), "error": None}
        except Exception as exc:
            error = type(exc).__name__
            if attempt == 0:
                time.sleep(1.0)
    return {"ok": False, "text": "", "finish": None,
            "usage": {"prompt": 0, "completion": 0, "total": 0},
            "seconds": round(time.perf_counter() - started, 3), "error": error}


def append_log(path: Path, record: dict):
    with LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate(samples: list[dict], condition_prompts: dict[str, str], phase: str, client, model, log_path: Path) -> dict:
    output = {row["sample_id"]: {} for row in samples}
    jobs = [(row, condition, prompt_id) for row in samples
            for condition, prompt_id in condition_prompts.items()]

    def run(job):
        row, condition, prompt_id = job
        result = request(client, model,
                         [{"role": "system", "content": PROMPTS[prompt_id]},
                          {"role": "user", "content": row["source"]}], TEMP, 1200)
        return row, condition, prompt_id, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, condition, prompt_id, result = future.result()
            sid = row["sample_id"]
            output[sid][condition] = result
            append_log(log_path, {"phase": phase, "id": sid, "issue_type": row["issue_type"],
                                  "condition": condition, "prompt_id": prompt_id, **result})
            if index % 20 == 0 or index == len(futures):
                print(f"{phase} generation: {index}/{len(futures)}", flush=True)
    return output


def evaluate(samples: list[dict], generated: dict, condition_ids: list[str], phase: str,
             client, model, log_path: Path) -> dict:
    output = {}
    jobs = []
    for row in samples:
        sid = row["sample_id"]
        ordered = [(condition, generated[sid][condition]["text"]) for condition in condition_ids]
        random.Random(SEED + sum(ord(char) for char in sid) + (0 if phase == "development" else 50000)).shuffle(ordered)
        mapping = {chr(65 + i): prompt_id for i, (prompt_id, _) in enumerate(ordered)}
        payload = {"source": row["source"], "annotated_issue": row["error_note"],
                   "reference_repair_example": row["reference"],
                   "candidates": [{"id": chr(65 + i), "translation": text}
                                  for i, (_, text) in enumerate(ordered)]}
        jobs.append((row, payload, mapping))

    def run(job):
        row, payload, mapping = job
        response = request(client, model,
                           [{"role": "system", "content": TARGET_EVAL_PROMPT},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                           0.0, 3200, True)
        try:
            parsed = json.loads(response["text"]) if response["ok"] else None
        except json.JSONDecodeError:
            parsed = None
        evaluations = {}
        for item in (parsed or {}).get("evaluations", []) if isinstance(parsed, dict) else []:
            prompt_id = mapping.get(str(item.get("id", "")))
            if prompt_id:
                status = str(item.get("status", "UNCERTAIN")).upper()
                if status not in {"FIXED", "NOT_FIXED", "UNCERTAIN"}:
                    status = "UNCERTAIN"
                evaluations[prompt_id] = {"status": status, "reason": item.get("reason", "")}
        return row, {"response": response, "parsed": parsed, "mapping": mapping, "evaluations": evaluations}

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, record = future.result()
            sid = row["sample_id"]
            output[sid] = record
            append_log(log_path, {"phase": phase, "id": sid, "issue_type": row["issue_type"], **record})
            if index % 20 == 0 or index == len(futures):
                print(f"{phase} targeted evaluation: {index}/{len(futures)}", flush=True)
    return output


def judge_quality(samples: list[dict], generated: dict, condition_ids: list[str], phase: str,
                  client, model, log_path: Path) -> dict:
    output = {}
    jobs = []
    for row in samples:
        sid = row["sample_id"]
        ordered = [(condition, generated[sid][condition]["text"]) for condition in condition_ids]
        random.Random(SEED + sum(ord(char) for char in sid) + 90000).shuffle(ordered)
        mapping = {chr(65 + i): condition for i, (condition, _) in enumerate(ordered)}
        payload = {"source": row["source"], "candidates": [
            {"id": chr(65 + i), "translation": text} for i, (_, text) in enumerate(ordered)]}
        jobs.append((row, payload, mapping))

    def run(job):
        row, payload, mapping = job
        response = request(client, model,
                           [{"role": "system", "content": QUALITY_EVAL_PROMPT},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                           0.0, 5000, True)
        try:
            parsed = json.loads(response["text"]) if response["ok"] else None
        except json.JSONDecodeError:
            parsed = None
        evaluations = {}
        for item in (parsed or {}).get("evaluations", []) if isinstance(parsed, dict) else []:
            condition = mapping.get(str(item.get("id", "")))
            if not condition:
                continue
            dimensions = item.get("dimensions") or {}
            values = [str(dimensions.get(k, "UNCERTAIN")).upper() for k in
                      ["terms_referents", "numbers_units", "negation_scope",
                       "conditions_order", "modality_risk", "omissions_additions"]]
            try:
                semantic = int(item.get("semantic_completeness"))
                readability = int(item.get("readability"))
            except (TypeError, ValueError):
                semantic, readability = None, None
            passed = ("FAIL" not in values and "UNCERTAIN" not in values and
                      semantic is not None and semantic >= 4 and
                      readability is not None and readability >= 3)
            evaluations[condition] = {"dimensions": dimensions, "semantic": semantic,
                                      "readability": readability, "passed": passed,
                                      "notes": item.get("notes", "")}
        return row, {"response": response, "parsed": parsed, "mapping": mapping,
                     "evaluations": evaluations}

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, record = future.result()
            sid = row["sample_id"]
            output[sid] = record
            append_log(log_path, {"phase": phase, "id": sid, "issue_type": row["issue_type"], **record})
            if index % 20 == 0 or index == len(futures):
                print(f"{phase} general evaluation: {index}/{len(futures)}", flush=True)
    return output


def judge_pairs(samples: list[dict], client, model, log_path: Path) -> dict:
    output = {}
    prompt = ("Compare two Chinese translations of the same English text. Choose the one that preserves the source meaning more "
              "faithfully, especially technical terms, logical relations, negation, modality, conditions, numbers and limits. "
              "Do not choose by fluency alone. Return JSON only: {\"choice\":\"A\"} or {\"choice\":\"B\"}.")
    jobs = []
    for row in samples:
        options = [("correct", row["reference"]), ("incorrect", row["known_incorrect"])]
        random.Random(SEED + sum(ord(char) for char in row["sample_id"]) + 120000).shuffle(options)
        mapping = {"A": options[0][0], "B": options[1][0]}
        payload = {"source": row["source"], "candidate_A": options[0][1], "candidate_B": options[1][1]}
        jobs.append((row, payload, mapping))

    def run(job):
        row, payload, mapping = job
        response = request(client, model,
                           [{"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                           0.0, 1200, True)
        try:
            parsed = json.loads(response["text"]) if response["ok"] else None
        except json.JSONDecodeError:
            parsed = None
        selected = mapping.get(str((parsed or {}).get("choice", ""))) if isinstance(parsed, dict) else None
        return row, {"response": response, "selected": selected,
                     "correct_position": next(k for k, v in mapping.items() if v == "correct")}

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, record = future.result()
            output[row["sample_id"]] = record
            append_log(log_path, {"phase": "locked_test", "id": row["sample_id"],
                                  "issue_type": row["issue_type"], **record})
            if index % 20 == 0 or index == len(futures):
                print(f"Pairwise reference selection: {index}/{len(futures)}", flush=True)
    return {"n": len(samples), "correct": sum(x["selected"] == "correct" for x in output.values()),
            "incorrect": sum(x["selected"] == "incorrect" for x in output.values()),
            "unrated": sum(x["selected"] is None for x in output.values()), "cases": output}


def status(evaluations: dict, sid: str, prompt_id: str) -> str:
    return evaluations.get(sid, {}).get("evaluations", {}).get(prompt_id, {}).get("status", "UNRATED")


def summarize_by_type(samples: list[dict], evaluations: dict, prompt_ids: list[str]) -> dict:
    summary = {}
    for category in CATEGORIES:
        members = [row for row in samples if row["issue_type"] == category]
        if not members:
            continue
        summary[category] = {prompt_id: {
            "n": len(members),
            "fixed": sum(status(evaluations, row["sample_id"], prompt_id) == "FIXED" for row in members),
            "not_fixed": sum(status(evaluations, row["sample_id"], prompt_id) == "NOT_FIXED" for row in members),
            "uncertain_or_unrated": sum(status(evaluations, row["sample_id"], prompt_id) not in {"FIXED", "NOT_FIXED"} for row in members),
        } for prompt_id in prompt_ids}
    return summary


def macro_rate(by_type: dict, prompt_id: str) -> float:
    rates = [row[prompt_id]["fixed"] / row[prompt_id]["n"]
             for row in by_type.values() if row[prompt_id]["n"] >= MIN_DEV_N_FOR_PROMPT_SELECTION]
    return sum(rates) / len(rates) if rates else 0.0


def eval_status(evaluations: dict, sid: str, condition: str) -> str:
    return evaluations.get(sid, {}).get("evaluations", {}).get(condition, {}).get("status", "UNRATED")


def macro_prompt_choice(dev_summary: dict) -> tuple[str, dict]:
    constrained_ids = ["baseline", "fact_map", "self_audit"]
    rates = {prompt_id: macro_rate(dev_summary, prompt_id) for prompt_id in constrained_ids}
    totals = {prompt_id: sum(row[prompt_id]["fixed"] for row in dev_summary.values())
              for prompt_id in constrained_ids}
    # The locked test always compares the best challenger selected on development to baseline.
    selected = max(CHALLENGERS,
                   key=lambda p: (rates[p], totals[p], -len(PROMPTS[p]), -CHALLENGERS.index(p)))
    return selected, {"macro_fixed_rate": rates, "fixed_count": totals}


def generic_summary(samples: list[dict], quality: dict, conditions: list[str]) -> dict:
    result = {}
    for condition in conditions:
        vals = [quality.get(row["sample_id"], {}).get("evaluations", {}).get(condition, {}) for row in samples]
        result[condition] = {"n": len(vals), "passed": sum(x.get("passed") is True for x in vals),
                             "failed_or_unrated": sum(x.get("passed") is not True for x in vals)}
    return result


def selected_pool_condition(generated: dict, quality: dict, sid: str, selector: str) -> str | None:
    pool = [f"pool_{i}" for i in range(1, 5)]
    if selector == "first":
        return pool[0]
    if selector == "medoid":
        def norm(text):
            return "".join(c.lower() for c in text if unicodedata.category(c)[0] in {"L", "N"})
        def similarity(a, b):
            a, b = norm(a), norm(b)
            if not a or not b:
                return 0.0
            ca = Counter(a[i:i+2] for i in range(max(1, len(a)-1)))
            cb = Counter(b[i:i+2] for i in range(max(1, len(b)-1)))
            return 2 * sum((ca & cb).values()) / (sum(ca.values()) + sum(cb.values()))
        return max(pool, key=lambda name: (
            sum(similarity(generated[sid][name]["text"], generated[sid][other]["text"])
                for other in pool if other != name) / 3,
            -pool.index(name)))
    passing = [name for name in pool
               if quality.get(sid, {}).get("evaluations", {}).get(name, {}).get("passed") is True]
    if not passing:
        return None
    return max(passing, key=lambda name: (
        .7 * quality[sid]["evaluations"][name]["semantic"] +
        .3 * quality[sid]["evaluations"][name]["readability"],
        -pool.index(name)))


def pass_at_k(test_samples: list[dict], targeted: dict) -> dict:
    result = {}
    for k in range(1, 5):
        per_sample = []
        for row in test_samples:
            fixed = sum(eval_status(targeted, row["sample_id"], f"pool_{i}") == "FIXED"
                        for i in range(1, 5))
            per_sample.append(1 - math.comb(4 - fixed, k) / math.comb(4, k) if 4 - fixed >= k else 1.0)
        result[str(k)] = {"expected_hits": sum(per_sample), "n": len(test_samples),
                          "rate": sum(per_sample) / len(test_samples)}
    return result


def paired_test(samples: list[dict], targeted: dict, left: str, right: str) -> dict:
    transitions = Counter((eval_status(targeted, row["sample_id"], left),
                           eval_status(targeted, row["sample_id"], right)) for row in samples)
    improved = sum(n for (a, b), n in transitions.items() if a == "NOT_FIXED" and b == "FIXED")
    regressed = sum(n for (a, b), n in transitions.items() if a == "FIXED" and b == "NOT_FIXED")
    p = binomtest(improved, improved + regressed, .5).pvalue if improved + regressed else 1.0
    return {"transitions": {f"{a}->{b}": n for (a, b), n in transitions.items()},
            "improved": improved, "regressed": regressed, "exact_mcnemar_p": p}


def summarize_test(samples: list[dict], generated: dict, targeted: dict, quality: dict,
                   selected: str, pairwise: dict) -> dict:
    conditions = ["direct", "baseline", "selected_single"] + [f"pool_{i}" for i in range(1, 5)]
    targeted_totals = {
        c: {"n": len(samples),
            "fixed": sum(eval_status(targeted, x["sample_id"], c) == "FIXED" for x in samples),
            "not_fixed": sum(eval_status(targeted, x["sample_id"], c) == "NOT_FIXED" for x in samples),
            "uncertain_or_unrated": sum(eval_status(targeted, x["sample_id"], c) not in {"FIXED", "NOT_FIXED"} for x in samples)}
        for c in conditions
    }
    selector_totals = {}
    selector_cases = {}
    for selector, label in [("first", "首份候选"), ("medoid", "相似度中心"), ("quality", "通用质量评审择优")]:
        cases = {}
        for row in samples:
            sid = row["sample_id"]
            choice = selected_pool_condition(generated, quality, sid, selector)
            cases[sid] = {"condition": choice,
                          "status": eval_status(targeted, sid, choice) if choice else "ABSTAIN"}
        selector_cases[label] = cases
        statuses = [case["status"] for case in cases.values()]
        selector_totals[label] = {"n": len(samples), "fixed": statuses.count("FIXED"),
                                  "not_fixed": statuses.count("NOT_FIXED"),
                                  "uncertain": statuses.count("UNCERTAIN"),
                                  "unrated": statuses.count("UNRATED"),
                                  "abstain": statuses.count("ABSTAIN")}
    type_summary = summarize_by_type(samples, targeted, conditions)
    sims = []
    for row in samples:
        sid = row["sample_id"]
        pool = [f"pool_{i}" for i in range(1, 5)]
        vals = []
        def norm(text):
            return "".join(c.lower() for c in text if unicodedata.category(c)[0] in {"L", "N"})
        def similarity(a, b):
            a, b = norm(a), norm(b)
            if not a or not b:
                return 0.0
            ca = Counter(a[i:i+2] for i in range(max(1, len(a)-1)))
            cb = Counter(b[i:i+2] for i in range(max(1, len(b)-1)))
            return 2 * sum((ca & cb).values()) / (sum(ca.values()) + sum(cb.values()))
        for i, left in enumerate(pool):
            for right in pool[i+1:]:
                vals.append(similarity(generated[sid][left]["text"], generated[sid][right]["text"]))
        sims.append({"id": sid, "issue_type": row["issue_type"],
                     "mean_similarity": sum(vals) / len(vals),
                     "fixed_fraction": sum(eval_status(targeted, sid, c) == "FIXED" for c in pool) / 4})
    from scipy.stats import spearmanr
    correlation = spearmanr([x["mean_similarity"] for x in sims], [x["fixed_fraction"] for x in sims])
    dev_vs_selected = paired_test(samples, targeted, "baseline", "selected_single")
    direct_vs_selected = paired_test(samples, targeted, "direct", "selected_single")
    return {"n": len(samples), "targeted_by_condition": targeted_totals,
            "quality_by_condition": generic_summary(samples, quality, conditions),
            "pool_pass_at_k": pass_at_k(samples, targeted),
            "selectors": selector_totals, "selector_cases": selector_cases,
            "by_type": type_summary, "similarity": sims,
            "similarity_spearman_rho": correlation.statistic,
            "similarity_spearman_p": correlation.pvalue,
            "baseline_vs_selected": dev_vs_selected, "direct_vs_selected": direct_vs_selected,
            "known_pair_selection": {k: v for k, v in pairwise.items() if k != "cases"}}


def usage_summary(records: list[dict]) -> dict:
    return {"requests": len(records), "successful": sum(bool(r.get("ok")) for r in records),
            "prompt_tokens": sum(r.get("usage", {}).get("prompt", 0) for r in records),
            "completion_tokens": sum(r.get("usage", {}).get("completion", 0) for r in records),
            "total_tokens": sum(r.get("usage", {}).get("total", 0) for r in records),
            "sum_request_seconds": round(sum(r.get("seconds", 0) for r in records), 2)}


def set_chinese_font():
    candidates = [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
                  Path(r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(path)).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def draw_figures(dev_summary: dict, test_summary: dict, selected: str):
    set_chinese_font()
    names = {"plain": "普通提示", "baseline": "原约束提示",
             "fact_map": "事实映射提示", "self_audit": "自检修订提示"}
    prompts = ["plain", "baseline", "fact_map", "self_audit"]
    x = list(range(len(CATEGORIES)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.24
    for j, prompt_id in enumerate(prompts):
        vals = []
        for cat in CATEGORIES:
            n = dev_summary.get(cat, {}).get(prompt_id, {}).get("n", 0)
            fixed = dev_summary.get(cat, {}).get(prompt_id, {}).get("fixed", 0)
            vals.append(100 * fixed / n if n else float("nan"))
        ax.bar([i + (j - 1.5) * width for i in x], vals, width=width, label=names[prompt_id])
    dev_labels = [f"{cat}\n(n={dev_summary.get(cat, {}).get('baseline', {}).get('n', 0)})"
                  for cat in CATEGORIES]
    ax.set_xticks(x, dev_labels, rotation=12, ha="right")
    ax.set_ylabel("开发集目标错误修复率（%）")
    ax.set_ylim(0, 105)
    ax.set_title("开发集：按错误类型比较候选提示")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure1_development_by_type.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    compare = ["direct", "baseline", "selected_single"]
    labels = ["普通提示", "原约束提示\n（测试组）", f"原约束提示\n（独立复采样）"]
    vals = [100 * test_summary["targeted_by_condition"][p]["fixed"] / max(1, test_summary["n"]) for p in compare]
    bars = ax.bar(labels, vals, color=["#5B8FF9", "#8A78B8", "#61A784"], width=.58)
    for bar, prompt_id in zip(bars, compare):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 1.4,
                f"{test_summary['targeted_by_condition'][prompt_id]['fixed']}/{test_summary['n']}（{y:.1f}%）", ha="center")
    ax.set_ylabel("目标错误修复率（%）")
    ax.set_ylim(0, 105)
    ax.set_title("锁定测试集：提示方案对比")
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout()
    fig.savefig(OUT / "figure2_locked_test_overall.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = .25
    for j, prompt_id in enumerate(compare):
        vals = [100 * test_summary["by_type"].get(cat, {}).get(prompt_id, {}).get("fixed", 0) /
                max(1, test_summary["by_type"].get(cat, {}).get(prompt_id, {}).get("n", 0)) for cat in CATEGORIES]
        label = {"direct": "普通提示", "baseline": "原约束提示（测试组）",
                 "selected_single": "原约束提示（独立复采样）"}[prompt_id]
        ax.bar([i + (j - 1) * width for i in x], vals, width=width, label=label)
    test_labels = [f"{cat}\n(n={test_summary['by_type'].get(cat, {}).get('direct', {}).get('n', 0)})"
                   for cat in CATEGORIES]
    ax.set_xticks(x, test_labels, rotation=12, ha="right")
    ax.set_ylabel("测试集目标错误修复率（%）")
    ax.set_ylim(0, 105)
    ax.set_title("锁定测试集：按错误类型的修复率")
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure3_locked_test_by_type.png", dpi=180)
    plt.close(fig)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    dev, test, split_counts = make_splits()
    if "--plan-only" in sys.argv:
        print(json.dumps({"split": split_counts, "dev_quota": DEV_QUOTA, "test_quota": TEST_QUOTA},
                         ensure_ascii=False, indent=2))
        return

    client, model, endpoint = load_client()
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ["generation_calls.jsonl", "targeted_evaluation_calls.jsonl",
                     "quality_evaluation_calls.jsonl", "pairwise_calls.jsonl"]:
        (OUT / filename).unlink(missing_ok=True)
    write_json(OUT / "sample_manifest.json", {"split_counts": split_counts,
        "development": [{"id": x["sample_id"], "issue_type": x["issue_type"], "source": x["source"]} for x in dev],
        "test": [{"id": x["sample_id"], "issue_type": x["issue_type"], "source": x["source"]} for x in test]})
    generation_log = OUT / "generation_calls.jsonl"
    targeted_log = OUT / "targeted_evaluation_calls.jsonl"
    quality_log = OUT / "quality_evaluation_calls.jsonl"
    pairwise_log = OUT / "pairwise_calls.jsonl"

    print("Development split:", split_counts["development_by_type"], flush=True)
    dev_conditions = {prompt_id: prompt_id for prompt_id in ["plain", "baseline", "fact_map", "self_audit"]}
    dev_generated = generate(dev, dev_conditions, "development", client, model, generation_log)
    dev_evaluations = evaluate(dev, dev_generated, list(dev_conditions), "development", client, model, targeted_log)
    dev_summary = summarize_by_type(dev, dev_evaluations, list(dev_conditions))
    selected, dev_macro = macro_prompt_choice(dev_summary)
    print("Selected using development split only:", selected, dev_macro, flush=True)

    test_conditions = {"direct": "plain", "baseline": "baseline", "selected_single": selected}
    test_conditions.update({f"pool_{i}": selected for i in range(1, 5)})
    test_generated = generate(test, test_conditions, "locked_test", client, model, generation_log)
    test_evaluations = evaluate(test, test_generated, list(test_conditions), "locked_test", client, model, targeted_log)
    quality_evaluations = judge_quality(test, test_generated, list(test_conditions), "locked_test",
                                         client, model, quality_log)
    pairwise = judge_pairs(test, client, model, pairwise_log)
    test_summary = summarize_test(test, test_generated, test_evaluations, quality_evaluations, selected, pairwise)

    generation_records = [json.loads(line) for line in generation_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    targeted_records = [json.loads(line) for line in targeted_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    quality_records = [json.loads(line) for line in quality_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairwise_records = [json.loads(line) for line in pairwise_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_responses = generation_records + [x["response"] for group in
        [targeted_records, quality_records, pairwise_records] for x in group]
    if any(not row.get("ok") for row in all_responses):
        print("Warning: one or more API calls failed; review UNRATED counts and logs.", flush=True)

    development_bundle = {"samples": dev, "generated": dev_generated, "evaluations": dev_evaluations,
                          "by_type": dev_summary, "macro_prompt_selection": dev_macro,
                          "selection_rule": "highest macro-average FIXED rate across represented primary issue categories; tie-break by total FIXED count, then shorter prompt",
                          "selected_prompt": selected}
    test_bundle = {"samples": test, "generated": test_generated, "evaluations": test_evaluations,
                   "quality_evaluations": quality_evaluations, "pairwise_selector": pairwise,
                   "summary": test_summary}
    usage = {"generation": usage_summary(generation_records),
             "targeted_evaluation": usage_summary([x["response"] for x in targeted_records]),
             "quality_evaluation": usage_summary([x["response"] for x in quality_records]),
             "pairwise_selection": usage_summary([x["response"] for x in pairwise_records]),
             "all": usage_summary(all_responses)}
    bundle = {"metadata": {"model": model, "endpoint_host": endpoint.split("//")[-1].split("/")[0],
                            "seed": SEED, "old_pilot_seed_excluded": OLD_SEED,
                            "temperature": TEMP, "generation_thinking_disabled": True,
                            "dev_n": len(dev), "test_n": len(test),
                            "stratification": "primary annotated issue category; category labels are withheld from the generator",
                            "prompt_selection_rule": "select constrained prompt using development set; lock it before final test"},
              "split_counts": split_counts, "development": development_bundle,
              "locked_test": test_bundle, "usage": usage}
    write_json(OUT / "development_data.json", development_bundle)
    write_json(OUT / "test_data.json", test_bundle)
    write_json(OUT / "experiment_data.json", bundle)
    draw_figures(dev_summary, test_summary, selected)
    print(json.dumps({"split_counts": split_counts, "development_macro_selection": dev_macro,
                      "selected_prompt": selected, "test_summary": test_summary, "usage": usage},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
