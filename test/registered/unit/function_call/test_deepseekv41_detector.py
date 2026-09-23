"""Unit tests for DeepSeekV41Detector spaced DSML tags — no server, no weights."""

from sglang.srt.entrypoints.openai.encoding_dsv41 import (
    dsml_token,
    encode_arguments_to_dsml,
    encode_messages,
    eos_token,
    thinking_end_token,
    tool_call_tag_name,
    tool_calls_block_name,
    tool_parameter_tag_name,
)
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _v41_wrapped(invoke: str) -> str:
    block = tool_calls_block_name
    return f"<{dsml_token}{block}>\n{invoke}\n</{dsml_token}{block}>"


def _v41_invoke(name: str, params: str = "") -> str:
    tag = tool_call_tag_name
    return f'<{dsml_token}{tag} name="{name}">\n{params}\n</{dsml_token}{tag}>'


class TestDeepSeekV41NameResolution(CustomTestCase):
    def test_cli_parser_names_resolve(self):
        self.assertIn("deepseekv41", FunctionCallParser.ToolCallParserEnum)
        self.assertIs(
            FunctionCallParser.ToolCallParserEnum["deepseekv41"], DeepSeekV41Detector
        )
        self.assertIn("deepseek-v41", ReasoningParser.DetectorMap)


class TestDeepSeekV41Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]
        self.params = encode_arguments_to_dsml(
            {"arguments": {"city": "SF"}}
        )

    def test_tokens_use_spaced_v41_names(self):
        detector = DeepSeekV41Detector()
        self.assertEqual(detector.bot_token, f"<{dsml_token}{tool_calls_block_name}>")
        self.assertIn(tool_call_tag_name, detector.invoke_start_token)
        self.assertIn(tool_parameter_tag_name, detector.parameter_regex)
        self.assertNotEqual(detector.bot_token, "<｜DSML｜tool_calls>")
        self.assertNotIn("<｜DSML｜invoke ", detector.invoke_start_token)

    def test_non_streaming_parses_spaced_dsml(self):
        text = "Let me check.\n\n" + _v41_wrapped(
            _v41_invoke("get_weather", self.params)
        )
        result = DeepSeekV41Detector().detect_and_parse(text, self.tools)
        self.assertIn("Let me check.", result.normal_text)
        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])
        self.assertIn("SF", result.calls[0].parameters)

    def test_streaming_agrees_with_one_shot(self):
        text = "Checking.\n\n" + _v41_wrapped(_v41_invoke("get_weather", self.params))
        detector = DeepSeekV41Detector()
        normal, calls = "", []
        for i in range(0, len(text), 7):
            result = detector.parse_streaming_increment(text[i : i + 7], self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        one_shot = DeepSeekV41Detector().detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(normal, one_shot.normal_text)

    def test_unspaced_v4_tags_are_not_v41_calls(self):
        v4 = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            f'<{dsml_token}{tool_parameter_tag_name} name="city" string="true">'
            "SF"
            f"</{dsml_token}{tool_parameter_tag_name}>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        result = DeepSeekV41Detector().detect_and_parse(v4, self.tools)
        self.assertEqual(result.calls, [])

    def test_function_call_parser_name_parses(self):
        text = _v41_wrapped(_v41_invoke("get_weather", self.params))
        parser = FunctionCallParser(self.tools, "deepseekv41")
        result = parser.detector.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])


class TestDeepSeekV41ReasoningParser(CustomTestCase):
    def test_deepseek_v41_splits_think_tags(self):
        parser = ReasoningParser("deepseek-v41")
        reasoning, normal = parser.parse_non_stream(
            "<think>plan the call</think>\n\nanswer"
        )
        self.assertEqual(reasoning, "plan the call")
        self.assertEqual(normal, "\n\nanswer")

    def test_reasoning_then_spaced_tool_call_stays_in_normal(self):
        parser = ReasoningParser("deepseek-v41")
        call = _v41_wrapped(
            _v41_invoke(
                "get_weather",
                encode_arguments_to_dsml({"arguments": {"city": "SF"}}),
            )
        )
        reasoning, normal = parser.parse_non_stream(
            f"<think>need weather</think>\n\n{call}"
        )
        self.assertEqual(reasoning, "need weather")
        self.assertIn(tool_calls_block_name, normal)
        self.assertIn("get_weather", normal)


class TestDsv41StickyPrefix(CustomTestCase):
    def test_trailing_system_reminder_keeps_the_previous_prompt_as_prefix(self):
        """A follow-up that leaves the previous harness reminder where it was
        and appends a new one must keep the first turn's prompt plus the
        assistant text as a prefix. That is the sticky-cache contract: the
        next request's token ids have to start with the pinned sequence."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        system = {
            "role": "system",
            "content": "You are a coding assistant.",
            "tools": tools,
        }
        user = {"role": "user", "content": "Read foo.py and tell me the first line."}
        reminder = {
            "role": "system",
            "content": "<system-reminder>\nThe current date is 2026-09-22.\n</system-reminder>",
        }
        first = encode_messages(
            [system, user, reminder],
            thinking_mode="thinking",
            drop_thinking=False,
        )
        output = "need the file" + thinking_end_token + "\n\nReading it." + eos_token
        assistant = {
            "role": "assistant",
            "reasoning_content": "need the file",
            "content": "\n\nReading it.",
        }
        follow = encode_messages(
            [
                system,
                user,
                reminder,
                assistant,
                {"role": "user", "content": "also check bar.py"},
                {
                    "role": "system",
                    "content": "<system-reminder>\nfoo.py changed.\n</system-reminder>",
                },
            ],
            thinking_mode="thinking",
            drop_thinking=False,
        )
        self.assertTrue(follow.startswith(first + output))


if __name__ == "__main__":
    import unittest

    unittest.main()
