"""Pure prompt and parser for the provisional weighted-rubric group diagnostic."""

import json
import math
import re


def rubric_items(rubric):
    items = [
        (int(index), float(weight)) for index, weight in re.findall(r"(?m)^\s*(\d+)\.\s*\[([0-9.]+)\s*pts?\]", rubric)
    ]
    if not items or len({index for index, _ in items}) != len(items):
        raise ValueError("Rubric has missing or duplicate weighted items")
    if not all(math.isfinite(weight) and weight > 0 for _, weight in items) or sum(weight for _, weight in items) != 7:
        raise ValueError("Expected supplied positive weights totaling seven")
    return items


def messages(row, candidate, *, swapped, system_prompt, grading_method="anonymous_pair"):
    if grading_method == "reference_guided":
        if swapped:
            raise ValueError("A labeled single-candidate request has no swapped presentation")
        data = {
            "problem": row["problem"],
            "rubric": row["rubric"],
            "reference_solution": row["answer"],
            "candidate_solution": candidate,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ]
    if grading_method != "anonymous_pair":
        raise ValueError("Unknown grading method")
    answers = {"A": row["answer"], "B": candidate} if swapped else {"A": candidate, "B": row["answer"]}
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {"problem": row["problem"], "rubric": row["rubric"], "answers": answers}, ensure_ascii=False
            ),
        },
    ]


def parse_grade(content, items, *, answer_labels=("A", "B")):
    value = json.loads(content)
    if not answer_labels or len(set(answer_labels)) != len(answer_labels):
        raise ValueError("Expected distinct nonempty answer labels")
    if not isinstance(value, dict) or set(value) != set(answer_labels):
        raise ValueError("Judge must return exactly the requested answer labels")
    output = {}
    for answer in answer_labels:
        decisions = value[answer]
        if not isinstance(decisions, list) or len(decisions) != len(items):
            raise ValueError("Judge returned wrong item count")
        weights = dict(items)
        seen = set()
        score = 0.0
        for decision in decisions:
            index = decision.get("item")
            if type(index) is not int or index not in weights or index in seen:
                raise ValueError("Invalid or duplicate item identity")
            if type(decision.get("satisfied")) is not bool or not isinstance(decision.get("reason"), str):
                raise ValueError("Expected Boolean satisfaction and a reason")
            seen.add(index)
            score += weights[index] * decision["satisfied"]
        output[answer] = {"score": score, "items": decisions}
    return output
