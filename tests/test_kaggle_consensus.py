"""Consensus extraction + escalation (eval/consensus.py), on invented claims and invented cases only.

No competition file is read, no LLM is called, nothing here prints or stores page text."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import consensus as cons  # noqa: E402


def claim(field, value, quote="q", evidence="claimed"):
    return {"field": field, "value": value, "display": str(value), "quote": quote, "evidence_level": evidence}


def ok(*claims):
    return {"status": "ok", "claims": list(claims)}


# ---------------------------------------------------------------- Task 3: the pre-registration, tally, per-page voting
def test_preregistered_constants():
    assert (cons.MIN_VALID_VOTES, cons.settled_min_for(5), cons.settled_min_for(3)) == (3, 4, 3)
    assert (cons.RA2_MAX_CHANGED, cons.RB0_SAMPLE, cons.RB0_MIN_AGREE_PCT, cons.RB2_MAX_UNRESOLVED, cons.RB0_SEED) == (30, 100, 95, 10, 20261118)
    assert "RA1" in cons.__doc__ and "RB1" in cons.__doc__ and "D2" in cons.__doc__


def test_preregistration_names_every_rule_decision_and_the_label_policy():
    doc = cons.__doc__
    for key in ("R0", "RA1", "RA2", "RB0", "RB1", "RB2", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "Candidate rule"):
        assert key in doc, key
    assert "labels are used only to report" in doc.lower()
    for sample in ("p9s3", "p9s4", "p9s5", "p9h1", "p9h2", "v7n1", "noise1"):
        assert sample in doc, sample


def test_preregistration_has_the_dated_ra1_carrier_sensitivity_section():
    doc = cons.__doc__
    head = "Sensitivity analyses"
    assert head in doc and doc.index(head) > doc.index("Implementation notes")          # its own section, not filed under the notes
    section = doc[doc.index(head):]
    for phrase in ("RA1 carrier sensitivity", "2026-11-19", "every carrier combination", "caveat", "never changes", "limits stay",
                   "void", "preds_A.csv", "rules.json", "aggregate"):
        assert phrase in section, phrase


def test_tally_unanimous_five_is_settled():
    t = cons.tally(["a"] * 5, 4)
    assert (t.valid, t.has_winner, t.winner, t.agree, t.settled) == (5, True, "a", 5, True)


def test_tally_four_of_five_settled_three_of_five_not():
    assert cons.tally(["a"] * 4 + ["b"], 4).settled
    t = cons.tally(["a"] * 3 + ["b"] * 2, 4)
    assert t.has_winner and t.winner == "a" and not t.settled


def test_tally_two_two_one_has_no_winner_and_two_leaders():
    t = cons.tally(["a", "a", "b", "b", "c"], 4)
    assert not t.has_winner and t.leaders == frozenset({"a", "b"}) and not t.settled


def test_tally_fewer_than_three_valid_votes_never_wins():
    t = cons.tally(["a", "a"], 4)
    assert t.valid == 2 and not t.has_winner and not t.settled


def test_tally_absent_is_a_vote_and_can_win():
    t = cons.tally([None, None, None, "a", "a"], 4)
    assert t.has_winner and t.winner is None and t.agree == 3 and not t.settled


def test_tally_three_samples_need_three_of_three():
    assert cons.tally(["a"] * 3, 3).settled and not cons.tally(["a", "a", "b"], 3).settled


def test_failed_sample_is_not_an_absent_vote():
    sso = claim("sso_saml", "all_plans")
    t = cons.page_tallies([ok(sso), None, None, ok(sso), ok()], 4)["sso_saml"]
    assert t.valid == 3 and t.has_winner and t.winner is not None      # if failures counted as absent this would be None


def test_silence_is_a_vote_for_absent():
    t = cons.page_tallies([ok(claim("sso_saml", "all_plans"))] + [ok()] * 4, 4)["sso_saml"]
    assert t.has_winner and t.winner is None


# ---------------------------------------------------------------- Task 4: consensus claims, case records, stage-A numbers
def test_consensus_carries_the_first_sample_whose_signature_wins():
    a, b = claim("sso_saml", "all_plans", quote="A"), claim("sso_saml", "all_plans", quote="B")
    exs = [ok(claim("sso_saml", "none")), ok(a), ok(b), ok(a), ok(b)]
    got = cons.page_claims(exs, cons.page_tallies(exs, 4))
    assert [x["quote"] for x in got["claims"]] == ["A"]        # the signature ignores the quote; sample 1 is the first carrying the winner


def test_no_winner_carries_the_first_valid_samples_claims():
    x, y = claim("sso_saml", "all_plans", quote="X"), claim("sso_saml", "none", quote="Y")
    exs = [None, ok(x), ok(x), ok(y), ok(y)]                   # 4 valid votes, 2-2: no strict majority
    got = cons.page_claims(exs, cons.page_tallies(exs, 4))
    assert [c["quote"] for c in got["claims"]] == ["X"]        # the first VALID sample is index 1


def test_no_winner_and_silent_first_sample_carries_nothing():
    x, y = claim("sso_saml", "all_plans"), claim("sso_saml", "none")
    exs = [ok(), ok(x), ok(x), ok(y), ok(y)]                   # absent 1, x 2, y 2: no strict majority
    assert cons.page_claims(exs, cons.page_tallies(exs, 4))["claims"] == []


def test_page_with_no_valid_sample_has_no_consensus():
    assert cons.page_claims([None, None], cons.page_tallies([None, None], 4)) is None


def test_k1_reproduces_the_sample():                           # R0 on synthetic claims
    cert = {"name": "SOC 2 Type II", "status": "certified", "date": "2026-01-02", "issuer": "Acme Audit LLP", "valid_until": None}
    ex = ok(claim("sso_saml", "all_plans"), claim("dpa_available", "yes"), claim("security_certifications", [cert]))
    got = cons.page_claims([ex], cons.page_tallies([ex], 1))
    assert cons.sample_signatures(got) == cons.sample_signatures(ex) and len(got["claims"]) == 3


def test_page_with_no_valid_sample_is_failed_and_makes_the_case_incomplete():
    recs = [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}]
    out, _, _ = cons.case_records(recs, [{"p1": None}, {}], 4)
    assert out[0]["extract"]["status"] == "failed"


def test_unreadable_pages_are_not_voted_on():
    recs = [{"role": "ignored", "quarantine": [], "page": {"page_id": "p1"}, "extract": None},
            {"role": "official", "quarantine": ["injection"], "page": {"page_id": "p2"}, "extract": None}]
    out, tallies, _ = cons.case_records(recs, [{"p1": ok(claim("sso_saml", "all_plans")), "p2": ok(claim("sso_saml", "all_plans"))}], 1)
    assert tallies == {} and all(r["extract"] is None for r in out)


def test_a_repeated_sample_reproduces_itself_through_the_winner_path():      # R0, winner path (k=5 identical samples)
    ex = ok(claim("sso_saml", "all_plans"), claim("dpa_available", "yes"))
    exs = [ex] * 5
    assert all(t.has_winner and t.settled for t in cons.page_tallies(exs, 4).values())
    assert cons.sample_signatures(cons.page_claims(exs, cons.page_tallies(exs, 4))) == cons.sample_signatures(ex)


PAGE1 = [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}]


def test_equal_signatures_with_different_holders_are_reported():
    def cert(holder):
        return {"name": "ISO 27001", "status": "certified", "date": "2026-01-02", "issuer": "Acme Audit LLP", "valid_until": None, "holder": holder}
    exs = [ok(claim("security_certifications", [cert(h)])) for h in ("Acme Ltd", "Acme Ltd", "Acme Global Pty Ltd", "Acme Ltd", "Acme Ltd")]
    assert cons.page_tallies(exs, 4)["security_certifications"].settled              # the signature cannot see the holder
    _, _, gaps = cons.case_records(PAGE1, [{"p1": e} for e in exs], 4)
    assert gaps["holder_split"] == [("p1", "security_certifications")] and gaps["post_vote_mismatch"] == []


def test_cross_field_normalisation_after_the_vote_is_reported():
    sub = claim("subprocessors", [{"name": "Acme Cloud", "category": "hosting", "country": "Ireland"}])

    def reg(q):
        return claim("data_regions", {"available": ["Ireland"], "planned": []}, quote=q)
    named, plain = "We host in Ireland with Acme Cloud.", "Data stays in Ireland."
    # regions: kept in samples 0, 3, 4 (wins 3 of 5); sub-processors: listed in 1-4 (wins 4 of 5). Sample 0 is the first carrier of the
    # winning region, and its quote names Acme Cloud, which only samples 1-4 list: on the combined claims the region is dropped.
    exs = [ok(reg(named)), ok(reg(named), sub), ok(reg(named), sub), ok(reg(plain), sub), ok(reg(plain), sub)]
    _, _, gaps = cons.case_records(PAGE1, [{"p1": e} for e in exs], 4)
    assert gaps["post_vote_mismatch"] == [("p1", "data_regions")]


def test_mixed_prompts_or_models_are_refused():
    with pytest.raises(ValueError):
        cons.require_one_prompt([cons.Sample("a", "SYS-1", "m", ""), cons.Sample("b", "SYS-2", "m", "")])
    with pytest.raises(ValueError):
        cons.require_one_prompt([cons.Sample("a", "SYS-1", "m1", ""), cons.Sample("b", "SYS-1", "m2", "")])
    cons.require_one_prompt([cons.Sample("a", "SYS-1", "m", ""), cons.Sample("b", "SYS-1", "m", "salt")])      # salts may differ


def test_score_rows_matches_the_metrics_definition():
    gold = {"c__R1": "UNKNOWN", "c__R2": "PASS", "c__OVERALL": "BLOCKED"}
    pred = {"c__R1": "PASS", "c__R2": "FAIL", "c__OVERALL": "BLOCKED"}
    assert cons.score_rows(pred, gold) == {"rows": 3, "wrong": 2, "false_confident": 1}


def test_changed_rows_lists_the_ids_whose_verdict_differs():
    assert cons.changed_rows({"b": "PASS", "a": "FAIL", "c": "PASS"}, {"b": "PASS", "a": "PASS", "c": "UNKNOWN"}) == ["a", "c"]


def _three_cases(quote="q"):
    sso = claim("sso_saml", "all_plans", quote=quote)
    recs = {c: [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}] for c in ("c1", "c2", "c3")}
    says = {"c1": [1, 1, 1, 1, 1], "c2": [1, 0, 0, 0, 0], "c3": [1, 1, 0, 0, 0]}      # who says "sso" in each of 5 samples
    per_sample = [{c: {"p1": ok(sso) if says[c][i] else ok()} for c in says} for i in range(5)]       # per_sample[i][case_id][page_id]

    def assemble(cid, rs):
        return {"verdicts": {"R1": "PASS" if rs[0]["extract"]["claims"] else "UNKNOWN"}, "overall": "CONDITIONAL", "bad": 0, "ours_disagreements": 0}
    return cons.stage_a_numbers(["c1", "c2", "c3"], recs, per_sample, 4, assemble)


def test_stage_a_numbers_on_invented_cases():
    r = _three_cases()
    assert (r["tallies_summary"]["voted"], r["tallies_summary"]["settled"], r["tallies_summary"]["unsettled"]) == (3, 2, 1)   # c1 5/5, c2 4 of 5 absent, c3 3 of 5 absent
    assert [r["preds"][f"{c}__R1"] for c in ("c1", "c2", "c3")] == ["PASS", "UNKNOWN", "UNKNOWN"]
    assert r["changed_vs_first"] == ["c2__R1", "c3__R1"]      # sample 0 alone says "sso" in all three cases
    assert r["fragile"] == ["c3__R1"]                         # drop one "absent" sample: 2-2 tie, the first sample's claim is carried -> PASS
    assert r["incomplete_cases"] == [] and [u["case_id"] for u in r["unsettled"]] == ["c3"]
    assert r["tallies_summary"]["per_field_unsettled"] == {"sso_saml": 1}
    assert len(r["singles"]) == 5 and r["singles"][0]["c2__R1"] == "PASS" and r["singles"][1]["c2__R1"] == "UNKNOWN"


def test_stage_a_outputs_never_contain_claim_text():
    canary = "CANARY-QUOTE-7731"
    r = _three_cases(quote=canary)
    assert canary not in json.dumps(r, default=str) and canary not in cons.format_stage_a(r)


def test_load_extract_reads_the_entry_keyed_by_prompt_model_and_salt(tmp_path):
    import kaggle_policy as kp
    page = {"page_id": "p1", "url": "https://acme.example/s", "title": "t", "text": "x" * 80}
    s = cons.Sample("a", "SYS", "m", "salt1")
    assert cons.load_extract(tmp_path, s, page) is None
    (tmp_path / f"{kp.cache_key('m', page, 'salt1', system='SYS')}.json").write_text(
        json.dumps({"model": "m", "salt": "salt1", "claims": [claim("sso_saml", "all_plans")], "in": 5, "out": 2}), encoding="utf-8")
    got = cons.load_extract(tmp_path, s, page)
    assert got["status"] == "ok" and got["claims"][0]["field"] == "sso_saml"
    assert cons.load_extract(tmp_path, cons.Sample("b", "SYS", "m", "other"), page) is None
    assert s.prompt_version == __import__("hashlib").sha256(b"SYS").hexdigest()[:12]


def test_targets_match_the_preregistered_sample_lists():
    t = cons.TARGETS
    assert [(x, y) for x, y in t["test"]["a"]] == [("s0", ""), ("noise1", "noise1"), ("p9s3", "p9s3"), ("p9s4", "p9s4"), ("p9s5", "p9s5")]
    assert t["test"]["second"] == [("v7", ""), ("v7n1", "v7n1")]
    assert t["holdout"]["a"] == [("s0", ""), ("p9h1", "p9h1"), ("p9h2", "p9h2")] and t["holdout"]["second"] == [("f45", "")]
    assert t["dev"]["a"] == [("s0", "")] and t["dev"]["second"] == []
    assert [(t[k]["split"], t[k]["subset"]) for k in ("test", "holdout", "dev")] == [("test", "all"), ("train", "holdout"), ("train", "dev")]


# ---------------------------------------------------------------- Task 5: rules and the candidate rule
def test_ra1_rounds_the_mean_up():
    s = [{"wrong": 3, "false_confident": 0}, {"wrong": 3, "false_confident": 1}, {"wrong": 4, "false_confident": 1}]   # means 3.33 -> 4, 0.67 -> 1
    assert cons.ra1({"wrong": 4, "false_confident": 1}, s)["status"] == "pass"
    assert cons.ra1({"wrong": 5, "false_confident": 1}, s)["status"] == "fail"
    assert cons.ra1({"wrong": 4, "false_confident": 2}, s)["status"] == "fail"


def test_ra1_with_no_samples_is_not_run():
    assert cons.ra1({"wrong": 0, "false_confident": 0}, [])["status"] == "not_run"


def test_ra2_boundary():
    assert cons.ra2(30)["status"] == "pass" and cons.ra2(31)["status"] == "fail"


def test_rb0_boundary():
    assert cons.rb0(95, 100)["status"] == "pass" and cons.rb0(94, 100)["status"] == "fail"
    assert cons.rb0(19, 20)["status"] == "pass" and cons.rb0(18, 20)["status"] == "fail"      # the threshold is a percentage of n, not of 100


def test_rb0_with_nothing_drawn_is_untested_not_a_pass():
    assert cons.rb0(0, 0)["status"] == "untested"


def test_rb2_boundary():
    assert cons.rb2(10)["status"] == "pass" and cons.rb2(11)["status"] == "fail"


def test_rb1_untested_is_not_a_pass():
    assert cons.rb1(0, 0, 0)["status"] == "untested"
    assert cons.rb1(3, 2, 1)["status"] == "pass" and cons.rb1(3, 1, 1)["status"] == "fail" and cons.rb1(2, 0, 1)["status"] == "fail"


def test_candidate_rule():
    P, F, U, N = "pass", "fail", "untested", "not_run"

    def r(**kw):
        return {k: {"status": v, "detail": ""} for k, v in kw.items()}
    allp = dict(R0=P, RA1=P, RA2=P, RB0=P, RB1=P, RB2=P)
    assert cons.candidate_rule(r(**allp)) == "A+B"
    assert cons.candidate_rule(r(**{**allp, "RB1": U})) == "A"           # B untested -> A-only
    assert cons.candidate_rule(r(**{**allp, "RB0": N, "RB1": N, "RB2": N})) == "A"
    assert cons.candidate_rule(r(**{**allp, "RB2": F})) == "A"            # B not decisive -> A-only
    assert cons.candidate_rule(r(**{**allp, "RA1": F})) == "none"
    assert cons.candidate_rule(r(**{**allp, "R0": N})) == "none"
    assert cons.candidate_rule({}) == "none"                              # nothing recorded yet


def test_rules_json_round_trip(tmp_path):
    p = tmp_path / "sub" / "rules.json"
    cons.save_rule(p, "RA2", cons.ra2(5))
    cons.save_rule(p, "R0", {"status": "pass", "detail": ""})
    assert set(cons.load_rules(p)) == {"RA2", "R0"}
    cons.save_rule(p, "RA2", cons.ra2(99))                                # read-modify-write replaces only that key
    assert cons.load_rules(p)["RA2"]["status"] == "fail" and cons.load_rules(p)["R0"]["status"] == "pass"
    assert cons.load_rules(tmp_path / "missing.json") == {}


def test_save_rule_refuses_unknown_keys_and_statuses(tmp_path):
    with pytest.raises(ValueError):
        cons.save_rule(tmp_path / "r.json", "RX", {"status": "pass", "detail": ""})
    with pytest.raises(ValueError):
        cons.save_rule(tmp_path / "r.json", "R0", {"status": "maybe", "detail": ""})


# ---------------------------------------------------------------- Task 6: stage B logic: triggers, resolution, plan, RB0 draw
def T(votes, k=5):
    return cons.tally(votes, cons.settled_min_for(k))


def test_trigger_one_is_unsettled_trigger_two_is_a_unanimous_disagreeing_second_opinion():
    assert cons.triggers(T(["a"] * 3 + ["b"] * 2), []) == (True, False)
    assert cons.triggers(T(["a"] * 5), ["b", "b"]) == (False, True)
    assert cons.triggers(T(["a"] * 5), ["b", "c"]) == (False, False)      # second opinion not unanimous
    assert cons.triggers(T(["a"] * 5), ["a", "a"]) == (False, False)
    assert cons.triggers(T(["a"] * 5), [None, None]) == (False, True)     # second opinion says "absent" unanimously
    assert cons.triggers(T(["a", "a", "b", "b", "c"]), ["b", "b"]) == (True, False)    # no flash winner: trigger two cannot fire


def test_resolve_confirmed_tie_broken_unresolved():
    assert cons.resolve(T(["a"] * 3 + ["b"] * 2), ["a", "a", "b"]).kind == "confirmed"
    r = cons.resolve(T(["a", "a", "b", "b", "c"]), ["b", "b", "c"])
    assert (r.kind, r.winner) == ("tie_broken", "b")
    assert cons.resolve(T(["a", "a", "b", "b", "c"]), ["c", "c", "c"]).kind == "unresolved"     # c is not a tied leader
    assert cons.resolve(T(["a"] * 4 + ["b"]), ["b", "b", "b"]).kind == "unresolved"             # adjudicator disagrees with consensus
    assert cons.resolve(T(["a"] * 4 + ["b"]), ["a", "b", None]).kind == "unresolved"            # no adjudicator majority
    assert cons.resolve(T(["a"] * 4 + ["b"]), ["a", "a"]).kind == "unresolved"                  # fewer than three adjudicator votes


def test_plan_pages_are_deduplicated():
    tallies = {"c1": {("p1", "f1"): T(["a"] * 3 + ["b"] * 2), ("p1", "f2"): T(["a"] * 3 + ["b"] * 2), ("p2", "f1"): T(["a"] * 5)}}
    plan = cons.b_plan(tallies, {})
    assert [(e["case_id"], e["page_id"], e["fields"]) for e in plan] == [("c1", "p1", ["f1", "f2"])]
    assert plan[0]["triggers"] == {"f1": [1], "f2": [1]}


def test_plan_records_which_trigger_fired_and_sorts_by_case_and_page():
    tallies = {"c2": {("p1", "f"): T(["a"] * 5)},
               "c1": {("p2", "f"): T(["a"] * 3 + ["b"] * 2), ("p1", "f"): T(["a"] * 5)}}
    votes = {("c2", "p1", "f"): ["b", "b"], ("c1", "p2", "f"): ["b", "b"], ("c1", "p1", "f"): ["a", "a"]}
    plan = cons.b_plan(tallies, votes)
    assert [(e["case_id"], e["page_id"], e["triggers"]) for e in plan] == [("c1", "p2", {"f": [1, 2]}), ("c2", "p1", {"f": [2]})]


def test_resolutions_change_the_carried_claims():
    x, y = claim("sso_saml", "all_plans", quote="X"), claim("sso_saml", "none", quote="Y")
    exs = [ok(x), ok(x), ok(y), ok(y), ok()]                   # 2-2-1: no winner, placeholder is sample 0's claim
    t, ysig = cons.page_tallies(exs, 4), cons.sample_signatures(ok(y))["sso_saml"]

    def quotes(res):
        return [c["quote"] for c in cons.page_claims(exs, t, {"sso_saml": res})["claims"]]
    assert quotes(cons.Resolution("tie_broken", ysig)) == ["Y"]
    assert quotes(cons.Resolution("tie_broken", None)) == []   # the adjudicator broke the tie toward "absent"
    assert quotes(cons.Resolution("unresolved", None)) == []
    assert quotes(cons.Resolution("confirmed", None)) == ["X"]


def test_case_records_apply_resolutions_and_do_not_report_overridden_fields_as_gaps():
    x, y = claim("sso_saml", "all_plans", quote="X"), claim("sso_saml", "none", quote="Y")
    exs = [ok(x), ok(x), ok(x), ok(y), ok(y)]                  # x wins 3 of 5, unsettled
    ps = [{"p1": e} for e in exs]
    plain, _, _ = cons.case_records(PAGE1, ps, 4)
    assert [c["quote"] for c in plain[0]["extract"]["claims"]] == ["X"]
    dropped, _, gaps = cons.case_records(PAGE1, ps, 4, {("p1", "sso_saml"): cons.Resolution("unresolved", None)})
    assert dropped[0]["extract"]["claims"] == [] and gaps["post_vote_mismatch"] == []
    kept, _, _ = cons.case_records(PAGE1, ps, 4, {("p1", "sso_saml"): cons.Resolution("confirmed", cons.sample_signatures(ok(x))["sso_saml"])})
    assert [c["quote"] for c in kept[0]["extract"]["claims"]] == ["X"]


def test_rb0_draw_is_seeded_label_free_and_unanimous_only():
    tallies = {"c%d" % i: {("p", "f"): T(["a"] * 5)} for i in range(300)}
    tallies["c0"][("p", "g")] = T([None] * 5)
    tallies["c1"][("p", "h")] = T(["a"] * 4 + ["b"])
    d1, d2 = cons.rb0_draw(tallies), cons.rb0_draw(tallies)
    assert d1 == d2 and len(d1) == 100 and len(set(d1)) == 100
    assert all(f == "f" for _, _, f in d1)                  # not absent-unanimous, not 4-of-5
    assert len(cons.rb0_draw({"c": {("p", "f"): T(["a"] * 5)}})) == 1


# ---------------------------------------------------------------- Task 7: stage A and stage B commands (the pure parts)
PAGE = {"page_id": "p1", "url": "https://acme.example/s", "title": "t", "text": "x" * 80}
PAGE2 = {"page_id": "p2", "url": "https://acme.example/t", "title": "u", "text": "y" * 200}


def test_b_row_changes():
    gold = {"x": "PASS", "y": "FAIL", "z": "PASS", "w": "PASS"}
    a, ab = {"x": "FAIL", "y": "FAIL", "z": "PASS", "w": "FAIL"}, {"x": "PASS", "y": "PASS", "z": "PASS", "w": "UNKNOWN"}
    assert cons.b_row_changes(a, ab, gold) == {"changed": 3, "wrong_to_right": 1, "right_to_wrong": 1}     # w stays wrong


def test_adjudicator_estimate_counts_calls_per_sample():
    from assessor.tracing import estimate_tokens
    import kaggle_policy as kp
    e = cons.adjudicator_estimate([PAGE, PAGE2], "S", 3)
    assert e["calls"] == 6 and e["tokens_in"] == 3 * sum(estimate_tokens("S" + kp.user_prompt(p)) for p in (PAGE, PAGE2))
    assert e["tokens_out"] == round(e["tokens_in"] * __import__("run_kaggle").OUT_PER_IN)
    assert e["minutes"] == (round(6 * 1.1 / 60), round(6 * 2.5 / 60))


def _samples(per_case):
    """per_case: {case_id: [vote of sample 0, sample 1, ...]} with "x" = sso all_plans, "y" = sso none, None = silent."""
    vals = {"x": "all_plans", "y": "none"}
    k = len(next(iter(per_case.values())))
    return [{cid: {"p1": ok(claim("sso_saml", vals[v[i]])) if v[i] else ok()} for cid, v in per_case.items()} for i in range(k)]


def _assemble_first_claim(cid, rs):
    cl = rs[0]["extract"]["claims"]
    return {"verdicts": {"R1": cl[0]["value"] if cl else "UNKNOWN"}, "overall": "CONDITIONAL", "bad": 0, "ours_disagreements": 0}


def _stage_b_inputs():
    recs = {c: [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}] for c in ("c1", "c2", "c3", "c4")}
    flash = _samples({"c1": list("xxxyy"), "c2": list("xxyy") + [None], "c3": list("xxxxx"), "c4": list("xxxyy")})
    second = _samples({"c1": list("xx"), "c2": list("xx"), "c3": list("yy"), "c4": list("xx")})
    adj = _samples({"c1": list("xxy"), "c2": list("yyy"), "c3": list("xxx"), "c4": list("yyy")})
    return recs, flash, second, adj


def test_stage_b_numbers_confirm_break_ties_and_drop():
    recs, flash, second, adj = _stage_b_inputs()
    r = cons.stage_b_numbers(["c1", "c2", "c3", "c4"], recs, flash, second, adj, 4, _assemble_first_claim)
    assert r["counts"] == {"confirmed": 2, "tie_broken": 1, "unresolved": 1}
    assert [(e["case_id"], e["triggers"]) for e in r["plan"]] == [("c1", {"sso_saml": [1]}), ("c2", {"sso_saml": [1]}),
                                                                   ("c3", {"sso_saml": [2]}), ("c4", {"sso_saml": [1]})]
    assert (r["preds_A"]["c1__R1"], r["preds_AB"]["c1__R1"]) == ("all_plans", "all_plans")         # confirmed: nothing changes
    assert (r["preds_A"]["c2__R1"], r["preds_AB"]["c2__R1"]) == ("all_plans", "none")              # tie broken toward a tied leader
    assert (r["preds_A"]["c3__R1"], r["preds_AB"]["c3__R1"]) == ("all_plans", "all_plans")         # second opinion overruled by the adjudicator
    assert (r["preds_A"]["c4__R1"], r["preds_AB"]["c4__R1"]) == ("all_plans", "UNKNOWN")           # adjudicator disagrees with Flash: claim dropped
    assert r["changed"] == ["c2__R1", "c4__R1"] and r["unresolved"] == [("c4", "p1", "sso_saml")]
    assert r["incomplete_cases"] == []


def test_stage_b_without_adjudicator_samples_drops_every_planned_claim_so_the_cli_must_check_coverage_first():
    recs, flash, second, _ = _stage_b_inputs()
    none_drawn = [{c: {"p1": None} for c in recs}] * 3
    r = cons.stage_b_numbers(["c1", "c2", "c3", "c4"], recs, flash, second, none_drawn, 4, _assemble_first_claim)
    assert r["counts"] == {"confirmed": 0, "tie_broken": 0, "unresolved": 4}


def test_rb0_numbers_count_agreement_with_the_unanimous_flash_signature():
    recs = {c: [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}] for c in ("c1", "c2", "c3")}
    flash = _samples({"c1": list("xxxxx"), "c2": list("xxxxx"), "c3": list("xxxxx")})
    tallies = {c: cons.case_records(recs[c], [f[c] for f in flash], 4)[1] for c in recs}
    adj1 = _samples({"c1": ["x"], "c2": ["y"], "c3": ["x"]})[0]
    del adj1["c3"]["p1"]                                                  # c3: the adjudicator never produced this page
    draw = [("c1", "p1", "sso_saml"), ("c2", "p1", "sso_saml"), ("c3", "p1", "sso_saml")]
    assert cons.rb0_numbers(draw, tallies, adj1) == (1, 3)               # a missing page is a disagreement, not a skipped row


def test_coverage_counts_readable_pages_with_a_valid_extract():
    recs = {"c1": [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None},
                   {"role": "official", "quarantine": [], "page": {"page_id": "p2"}, "extract": None},
                   {"role": "ignored", "quarantine": [], "page": {"page_id": "p3"}, "extract": None}]}
    assert cons.coverage({"c1": {"p1": ok(), "p2": None}}, recs) == (1, 2)


def test_stored_verdicts_reads_only_ids_verdicts_and_overall(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text(json.dumps({"case_id": "c1", "verdicts": {"R1": "PASS"}, "overall": "BLOCKED", "facts": {"secret": "ZZZ"}}) + "\n"
                 + json.dumps({"case_id": "c2", "verdicts": {"R1": "FAIL"}, "overall": "CONDITIONAL"}) + "\n", encoding="utf-8")
    assert cons.stored_verdicts(p) == {"c1__R1": "PASS", "c1__OVERALL": "BLOCKED", "c2__R1": "FAIL", "c2__OVERALL": "CONDITIONAL"}


def test_stage_b_report_never_contains_claim_text():
    canary = "CANARY-QUOTE-4417"
    recs = {c: [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}] for c in ("c1", "c2")}
    mk = lambda per: [{cid: {"p1": ok(claim("sso_saml", v, quote=canary)) if v else ok()} for cid, v in {"c1": per[i][0], "c2": per[i][1]}.items()} for i in range(len(per))]  # noqa: E731
    flash = mk([("all_plans", "none")] * 3 + [("none", "all_plans")] * 2)
    r = cons.stage_b_numbers(["c1", "c2"], recs, flash, [], mk([("all_plans", "none")] * 3), 4, _assemble_first_claim)
    text = cons.format_stage_b(r)
    assert canary not in text and canary not in json.dumps(r, default=str)
    assert "unresolved" in text and "tie_broken" in text and "confirmed" in text


def test_r0_rule_needs_all_three_targets_and_zero_differences():
    z = {"k1_diff": 0, "winner_path_diff": 0}
    assert cons.r0_rule({"dev": z, "holdout": z, "test": z})["status"] == "pass"
    assert cons.r0_rule({"dev": z, "holdout": z})["status"] == "not_run"                       # test not checked yet
    assert cons.r0_rule({"dev": z, "holdout": z, "test": {"k1_diff": 0, "winner_path_diff": 2}})["status"] == "fail"
    assert cons.r0_rule({"dev": {"k1_diff": 1, "winner_path_diff": 0}})["status"] == "fail"    # a difference fails at once, whatever is missing


# ---------------------------------------------------------------- Task 8: the candidate file
class _Metric:
    def components(self, solution, submission):
        return {}


_ROWS = [{"id": "c__R1"}, {"id": "c__OVERALL"}]
_BASE = {"c__R1": "PASS", "c__OVERALL": "CONDITIONAL"}                  # stands for submission #1
_A = {"c__R1": "UNKNOWN", "c__OVERALL": "CONDITIONAL"}                  # differs from #1 in one row
_AB = {"c__R1": "FAIL", "c__OVERALL": "BLOCKED"}                        # differs from #1 in two rows


def _rules(**kw):
    allp = dict(R0="pass", RA1="pass", RA2="pass", RB0="pass", RB1="pass", RB2="pass")
    return {k: {"status": v, "detail": ""} for k, v in {**allp, **kw}.items()}


def test_make_candidate_writes_nothing_when_the_rule_says_none(tmp_path):
    r = cons.make_candidate(_rules(RA1="fail"), [], _A, _AB, _ROWS, _Metric(), tmp_path / "out", _BASE)
    assert r["candidate"] == "none" and r["files"] == {} and not (tmp_path / "out").exists()


def test_make_candidate_a_only_writes_one_file_and_counts_the_rows_that_differ(tmp_path):
    r = cons.make_candidate(_rules(RB1="untested"), [], _A, _AB, _ROWS, _Metric(), tmp_path / "out", _BASE)
    assert r["candidate"] == "A" and list(r["files"]) == ["submission.csv"]
    assert r["files"]["submission.csv"] == {"rows": 2, "problems": [], "differs_from_baseline": 1}
    assert (tmp_path / "out" / "submission.csv").read_text().splitlines() == ["id,verdict", "c__R1,UNKNOWN", "c__OVERALL,CONDITIONAL"]


def test_make_candidate_a_plus_b_refuses_a_partial_stage_b(tmp_path):
    # a dry-run `resolve --allow-partial` overwrites preds_AB.csv with claims dropped for the pages the adjudicator never read
    r = cons.make_candidate(_rules(), [], _A, _AB, _ROWS, _Metric(), tmp_path / "out", _BASE, stage_b_partial=True)
    assert "partial" in r["refused"] and r["files"] == {} and not (tmp_path / "out").exists()


def test_make_candidate_a_plus_b_refuses_when_the_stage_b_status_is_unknown(tmp_path):
    r = cons.make_candidate(_rules(), [], _A, _AB, _ROWS, _Metric(), tmp_path / "out", _BASE)      # no stage_b.json was read
    assert "stage B" in r["refused"] and r["files"] == {} and not (tmp_path / "out").exists()


def test_make_candidate_a_only_never_looks_at_stage_b(tmp_path):
    r = cons.make_candidate(_rules(RB1="untested"), [], _A, _AB, _ROWS, _Metric(), tmp_path / "out", _BASE, stage_b_partial=True)
    assert r["candidate"] == "A" and "refused" not in r and list(r["files"]) == ["submission.csv"]


def test_make_candidate_a_plus_b_also_writes_the_a_only_file(tmp_path):
    r = cons.make_candidate(_rules(), [], _A, _AB, _ROWS, _Metric(), tmp_path / "out", _BASE, stage_b_partial=False)
    assert r["candidate"] == "A+B" and set(r["files"]) == {"submission.csv", "submission_A_only.csv"}
    assert r["files"]["submission.csv"]["differs_from_baseline"] == 2 and r["files"]["submission_A_only.csv"]["differs_from_baseline"] == 1
    assert "c__R1,FAIL" in (tmp_path / "out" / "submission.csv").read_text() and "c__R1,UNKNOWN" in (tmp_path / "out" / "submission_A_only.csv").read_text()


def test_make_candidate_refuses_when_a_test_case_is_incomplete(tmp_path):
    r = cons.make_candidate(_rules(), ["c9"], _A, _AB, _ROWS, _Metric(), tmp_path / "out", _BASE)
    assert "incomplete" in r["refused"] and r["files"] == {} and not (tmp_path / "out").exists()


def test_make_candidate_reports_the_self_check_of_a_bad_file(tmp_path):
    r = cons.make_candidate(_rules(RB1="untested"), [], {"c__R1": "MAYBE"}, _AB, _ROWS, _Metric(), tmp_path / "out", _BASE)
    assert r["files"]["submission.csv"]["problems"]


def test_format_rules_lists_every_rule_even_when_not_recorded():
    text = cons.format_rules({"R0": {"status": "pass", "detail": "ok"}})
    for key in cons.RULE_KEYS:
        assert key in text
    assert "not_run" in text and "pass" in text


def test_a_field_no_flash_sample_mentioned_still_triggers_when_the_second_opinion_is_unanimous():
    # spec 5.3, trigger 2: the consensus there is "absent"; an adjudicator majority for a claim then ends unresolved (and counts toward RB2)
    recs = {"c1": [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}]}
    flash = [{"c1": {"p1": ok()}} for _ in range(5)]                                   # every Flash sample is silent on this page
    second = _samples({"c1": list("xx")})                                              # both second-opinion samples find sso_saml
    claim_says = _samples({"c1": list("xxx")})                                         # the adjudicator agrees with the second opinion
    silent = [{"c1": {"p1": ok()}} for _ in range(3)]                                  # the adjudicator agrees with Flash: nothing there
    r = cons.stage_b_numbers(["c1"], recs, flash, second, claim_says, 4, _assemble_first_claim)
    assert [(e["case_id"], e["page_id"], e["triggers"]) for e in r["plan"]] == [("c1", "p1", {"sso_saml": [2]})]
    assert r["counts"] == {"confirmed": 0, "tie_broken": 0, "unresolved": 1} and r["unresolved"] == [("c1", "p1", "sso_saml")]
    assert r["changed"] == []                                                          # B never adds the claim: the verdict stays UNKNOWN
    r2 = cons.stage_b_numbers(["c1"], recs, flash, second, silent, 4, _assemble_first_claim)
    assert r2["counts"] == {"confirmed": 1, "tie_broken": 0, "unresolved": 0}


def test_a_field_only_the_second_opinion_mentions_is_not_in_the_rb0_pool():
    recs = {"c1": [{"role": "official", "quarantine": [], "page": {"page_id": "p1"}, "extract": None}]}
    flash = [{"c1": {"p1": ok()}} for _ in range(5)]
    _, tallies, _, _ = cons.plan_inputs(["c1"], recs, flash, _samples({"c1": list("xx")}), 4, _assemble_first_claim)
    assert ("p1", "sso_saml") in tallies["c1"] and cons.rb0_draw(tallies) == []        # stand-in "absent" consensus is never an RB0 item


# ---------------------------------------------------------------- final review: plan files must match the samples they claim to come from
def _plan(*ids):
    return [{"case_id": c, "page_id": f"{c}-p1", "fields": ["sso_saml"], "triggers": {"sso_saml": [1]}} for c in ids]


def test_plan_file_problems_none_when_the_disk_plan_is_the_recomputed_plan():
    assert cons.plan_file_problems(_plan("c1", "c2"), _plan("c1", "c2")) == []


def test_plan_file_problems_a_missing_plan_file_is_a_problem_not_an_empty_plan():
    probs = cons.plan_file_problems(_plan("c1"), None)
    assert probs and "missing" in probs[0]


def test_plan_file_problems_a_different_plan_is_reported_with_counts_only():
    probs = cons.plan_file_problems(_plan("c1", "c2", "c3"), _plan("c1", "c2"))
    assert len(probs) == 1 and "2" in probs[0] and "3" in probs[0]
    assert not any(cid in probs[0] for cid in ("c1", "c2", "c3"))           # counts, never ids or contents


def test_plan_file_problems_rb0_is_checked_only_when_both_sides_exist():
    now, other = _plan("c1"), _plan("c9")
    assert cons.plan_file_problems(now, now, rb0_now=now, rb0_disk=now) == []
    assert cons.plan_file_problems(now, now, rb0_now=now, rb0_disk=other)               # a drawn RB0 plan from another pool
    assert cons.plan_file_problems(now, now, rb0_now=now, rb0_disk=None) == []          # no rb0_plan.json: RB0 is simply not computed
    assert cons.plan_file_problems(now, now, rb0_now=None, rb0_disk=other) == []        # holdout: no RB0


def test_rb0_pages_group_the_drawn_fields_by_page_and_sort():
    draw = [("c2", "p1", "sso_saml"), ("c1", "p2", "dpa_available"), ("c1", "p2", "data_regions"), ("c1", "p1", "sso_saml")]
    assert cons.rb0_pages_from_draw(draw) == [
        {"case_id": "c1", "page_id": "p1", "fields": ["sso_saml"]},
        {"case_id": "c1", "page_id": "p2", "fields": ["dpa_available", "data_regions"]},     # draw order inside a page is kept
        {"case_id": "c2", "page_id": "p1", "fields": ["sso_saml"]}]


def test_save_plans_a_dry_run_writes_nothing(tmp_path):
    assert cons.save_plans(tmp_path / "out", _plan("c1"), _plan("c1"), full=False) == []
    assert not (tmp_path / "out").exists()


def test_save_plans_a_complete_run_writes_the_plan_and_the_rb0_plan(tmp_path):
    assert cons.save_plans(tmp_path / "out", _plan("c1"), _plan("c2"), full=True) == ["b_plan.json", "rb0_plan.json"]
    assert json.loads((tmp_path / "out" / "b_plan.json").read_text()) == _plan("c1")
    assert cons.save_plans(tmp_path / "out2", _plan("c1"), None, full=True) == ["b_plan.json"]
    assert not (tmp_path / "out2" / "rb0_plan.json").exists()
