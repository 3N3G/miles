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


def messages(row, candidate, *, swapped, system_prompt):
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


def parse_grade(content, items):
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"A", "B"}:
        raise ValueError("Judge must return both anonymous answers")
    output = {}
    for answer in ("A", "B"):
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
