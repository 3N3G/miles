"""Local VM rubric scorer. Reach it through an SSH tunnel, never a public bind."""

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from experiments.tcs_async_rl.grade_group import Budget, grade_one
from experiments.tcs_async_rl.group_inference import final_answer
from experiments.tcs_async_rl.judge_client import load_credentials
from experiments.tcs_async_rl.reward_hook import question_key


class Scorer:
    def __init__(self, *, input_file, directory, config_path, key_file):
        self.row = json.loads(input_file.read_text())
        self.question_key = question_key(self.row)
        self.config = json.loads(config_path.read_text())
        training = json.loads((config_path.parent / self.config["training_config"]).read_text())
        self.judge = training["judge"]
        self.prompt = (config_path.parent / self.config["prompt_file"]).read_text()
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.key = load_credentials(key_file)[self.judge["api_key_env"]]
        self.budget = Budget(directory, self.config)
        self.semaphore = asyncio.Semaphore(self.config["max_concurrency"])
        self.in_flight = {}
        manifest = {
            "code_commit": subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
            ).strip(),
            "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
            "question_key": self.question_key,
            "judge_config": self.config,
            "judge": self.judge,
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
        }
        (directory / "service-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    async def score(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("response"), str):
            raise ValueError("Expected a question identity and response text")
        if request.get("question_key") != self.question_key:
            raise ValueError("Question differs from the approved smoke input")
        candidate = final_answer(request.get("response", ""))
        if not candidate:
            raise ValueError("No completed final answer; not a zero grade")
        identity = "online-" + hashlib.sha256(candidate.encode()).hexdigest()[:24]
        if identity not in self.in_flight:
            self.in_flight[identity] = asyncio.create_task(self._score(identity, candidate))
        try:
            return await asyncio.shield(self.in_flight[identity])
        except Exception:
            self.in_flight.pop(identity, None)
            raise

    async def _score(self, identity, candidate):
        grades = await asyncio.gather(
            *(
                grade_one(
                    identity=identity,
                    candidate=candidate,
                    row=self.row,
                    swapped=swapped,
                    judge=self.judge,
                    config=self.config,
                    prompt=self.prompt,
                    key=self.key,
                    budget=self.budget,
                    semaphore=self.semaphore,
                    directory=self.directory,
                )
                for swapped in (False, True)
            )
        )
        scores = [grade["candidate"]["score"] for grade in grades]
        result = {
            "reward": statistics.mean(scores),
            "judge_order_gap": abs(scores[0] - scores[1]),
            "scores": scores,
            "identity": identity,
        }
        (self.directory / f"{identity}-score.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {"identity": identity, "reward": result["reward"], "judge_order_gap": result["judge_order_gap"]}
            ),
            flush=True,
        )
        return result


def make_app(scorer):
    # Only the HTTP server needs aiohttp. Pure scorer tests can use stand-in grade calls.
    from aiohttp import web

    async def health(request):
        return web.json_response(
            {
                "ready": True,
                "question_key": scorer.question_key,
                "model": scorer.judge["primary_model"],
                "reasoning_effort": scorer.judge["reasoning_effort"],
            }
        )

    async def score(request):
        try:
            return web.json_response(await scorer.score(await request.json()))
        except ValueError as error:
            return web.json_response({"error_type": type(error).__name__}, status=422)
        except Exception as error:
            return web.json_response({"error_type": type(error).__name__}, status=502)

    app = web.Application(client_max_size=2 * 1024**2)
    app.router.add_get("/health", health)
    app.router.add_post("/score", score)
    return app


def main():
    from aiohttp import web

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("judge_group_v1.json"))
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/api-keys.env")
    parser.add_argument("--port", type=int, default=18491)
    args = parser.parse_args()
    scorer = Scorer(
        input_file=args.input_file, directory=args.directory, config_path=args.config, key_file=args.key_file
    )
    web.run_app(make_app(scorer), host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
