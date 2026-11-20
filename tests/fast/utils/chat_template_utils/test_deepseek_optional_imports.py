"""Non-DeepSeek models can start without optional SGLang family encoders."""

import builtins
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]


class TestOptionalDeepSeekEncoders(unittest.TestCase):
    def test_unrelated_model_does_not_load_optional_encoders(self):
        protocol = types.ModuleType("sglang.srt.entrypoints.openai.protocol")
        protocol.Tool = type("Tool", (), {"model_dump": lambda self: {}})
        templates = types.ModuleType("miles.utils.chat_template_utils.templates")
        templates.encoding_dsv32 = object()
        openai = types.ModuleType("sglang.srt.entrypoints.openai")
        modules = {
            protocol.__name__: protocol,
            templates.__name__: templates,
            openai.__name__: openai,
        }
        original_import = builtins.__import__

        def reject_optional(name, globals=None, locals=None, fromlist=(), level=0):
            if name == openai.__name__:
                raise ImportError("optional family encoder unavailable")
            return original_import(name, globals, locals, fromlist, level)

        with (
            patch.dict(sys.modules, modules),
            patch("builtins.__import__", side_effect=reject_optional),
            tempfile.TemporaryDirectory() as directory,
        ):
            spec = importlib.util.spec_from_file_location(
                "deepseek_optional_import_test", ROOT / "miles/utils/chat_template_utils/deepseek.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            Path(directory, "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
            tokenizer = types.SimpleNamespace(name_or_path=directory)
            self.assertIsNone(module.model_type(tokenizer))
            self.assertIs(module.V32.template, templates.encoding_dsv32)
            # A selected unsupported family must fail instead of changing its template.
            with self.assertRaises(ImportError):
                module.V4.render_messages([{"role": "user", "content": "Hello"}])
            with self.assertRaises(ImportError):
                module.V41.render_messages([{"role": "user", "content": "Hello"}])


if __name__ == "__main__":
    unittest.main()
