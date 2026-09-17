#!/usr/bin/env python3
"""Unit and offline test suite for the auggie-launch proxy."""

import itertools
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import auggie_launch as main


class TestTokenAndAtomicTruncation(unittest.TestCase):
    def test_estimate_tokens(self):
        text = "def hello():\n    return 'world'"
        tokens = main.truncation.estimate_tokens_heuristic(text)
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len(text))

    def test_group_messages_into_turns_with_paired_tool_calls(self):
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Check files."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "ls", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "pwd", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file1.txt\nfile2.txt"},
            {"role": "tool", "tool_call_id": "call_2", "content": "/home/user/project"},
            {"role": "assistant", "content": "Here are the files."},
            {"role": "user", "content": "Now read file1.txt."},
        ]

        system_msgs, turns = main.truncation.group_messages_into_turns(messages)
        self.assertEqual(len(system_msgs), 1)
        self.assertEqual(system_msgs[0]["content"], "You are a coding assistant.")

        # Turns:
        # Turn 0: user "Check files."
        # Turn 1: assistant with tool_calls + 2 tool results (ATOMIC BLOCK)
        # Turn 2: assistant "Here are the files."
        # Turn 3: user "Now read file1.txt."
        self.assertEqual(len(turns), 4)

        # Verify atomic tool turn contains assistant + 2 tool results
        tool_turn = turns[1]
        self.assertEqual(len(tool_turn.messages), 3)
        self.assertEqual(tool_turn.messages[0]["role"], "assistant")
        self.assertEqual(tool_turn.messages[1]["role"], "tool")
        self.assertEqual(tool_turn.messages[1]["tool_call_id"], "call_1")
        self.assertEqual(tool_turn.messages[2]["role"], "tool")
        self.assertEqual(tool_turn.messages[2]["tool_call_id"], "call_2")

    def test_truncate_preserves_tool_call_integrity(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Old task 1" * 50},
            {"role": "assistant", "content": "Old response 1" * 50},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_x", "type": "function", "function": {"name": "cmd", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_x", "content": "cmd output" * 50},
            {"role": "user", "content": "Latest user query."},
        ]

        truncated = main.truncation.truncate_messages_to_context_limit(messages, max_context_tokens=5000)

        # Verify system message and latest user message are retained
        self.assertEqual(truncated[0]["role"], "system")
        self.assertEqual(truncated[-1]["content"], "Latest user query.")

        # Check that if any message has tool_calls, its corresponding tool result is NEVER lost
        for idx, m in enumerate(truncated):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                self.assertLess(idx + 1, len(truncated), "Assistant tool call cannot be the last message!")
                self.assertEqual(truncated[idx + 1].get("role"), "tool")


class TestToolCallMergingAndJSONRepair(unittest.TestCase):
    def test_merge_parallel_stream_tool_calls(self):
        chunks = [
            {"index": 0, "id": "call_101", "function": {"name": "read_file", "arguments": '{"path": '}},
            {"index": 1, "id": "call_102", "function": {"name": "search", "arguments": '{"query": '}},
            {"index": 0, "function": {"arguments": '"app.py"}'}},
            {"index": 1, "function": {"arguments": '"main"}'}},
        ]
        merged = main.truncation.merge_stream_tool_calls(chunks)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["id"], "call_101")
        self.assertEqual(merged[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(merged[0]["function"]["arguments"]), {"path": "app.py"})

        self.assertEqual(merged[1]["id"], "call_102")
        self.assertEqual(merged[1]["function"]["name"], "search")
        self.assertEqual(json.loads(merged[1]["function"]["arguments"]), {"query": "main"})

    def test_json_repair_unclosed_brackets(self):
        broken = '{"key": "value", "list": [1, 2'
        repaired = main.truncation.repair_json_arguments(broken)
        parsed = json.loads(repaired)
        self.assertEqual(parsed.get("key"), "value")
        self.assertEqual(parsed.get("list"), [1, 2])






class TestAuggieFullInjections(unittest.TestCase):
    def setUp(self):
        self._saved = (main.config.TARGET_BASE_URL, main.config.TARGET_MODEL, main.config.API_KEYS, main.config.REPLY_LANGUAGE)
        main.config.TARGET_BASE_URL = "http://127.0.0.1:50108/v1"
        main.config.TARGET_MODEL = "free"
        main.config.API_KEYS = ["sk-test-mock-key"]
        main.config.REPLY_LANGUAGE = "English"

    def tearDown(self):
        (main.config.TARGET_BASE_URL, main.config.TARGET_MODEL,
         main.config.API_KEYS, main.config.REPLY_LANGUAGE) = self._saved


    def test_build_injected_environment(self):
        proxy_url = "http://127.0.0.1:54321"
        env = main.injections.build_injected_environment(proxy_url)

        # Core Augment variables
        self.assertEqual(env.get("AUGMENT_API_URL"), proxy_url)
        self.assertEqual(env.get("AUGMENT_API_TOKEN"), main.config.LOCAL_TOKEN)
        self.assertEqual(env.get("AUGMENT_DISABLE_AUTO_UPDATE"), "1")
        self.assertEqual(env.get("AUGMENT_MODEL"), "free")

        # Session Auth
        auth = json.loads(env.get("AUGMENT_SESSION_AUTH", "{}"))
        self.assertEqual(auth.get("accessToken"), main.config.LOCAL_TOKEN)
        self.assertEqual(auth.get("tenantURL"), proxy_url)

        # The extra system prompt reaches the CLI through AUGMENT_INSTRUCTIONS
        self.assertIn("English", env.get("AUGMENT_INSTRUCTIONS", ""))

        # Standard upstream provider endpoints
        self.assertEqual(env.get("OPENAI_BASE_URL"), main.config.TARGET_BASE_URL)
        self.assertEqual(env.get("ANTHROPIC_BASE_URL"), main.config.TARGET_BASE_URL)
        self.assertEqual(env.get("OPENAI_API_KEY"), "sk-test-mock-key")

        # Keys come from the environment; an unset one must not be invented.
        self.assertNotIn("TAVILY_API_KEY", env)

        # PATH enhancement
        self.assertIn("/bin", env.get("PATH", ""))

    def test_generate_injected_mcp_config(self):
        mcp_path = main.injections.generate_injected_mcp_config()
        if mcp_path:
            self.assertTrue(os.path.isfile(mcp_path))
            with open(mcp_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("mcpServers", data)


class TestMockedNetworkOperations(unittest.TestCase):
    def setUp(self):
        main.upstream.reset_throttles()
        main.config.TARGET_BASE_URL = "http://127.0.0.1:20128/v1"
        main.config.TARGET_MODEL = "claude-3-7-sonnet"
        main.config.API_KEYS = ["sk-test-mock-key"]

    @patch("auggie_launch.upstream._CONNECTION_POOL.acquire")
    def test_dynamic_models_fetch_mocked(self, mock_acquire):
        mock_conn = MagicMock()
        mock_res = MagicMock()
        mock_res.status = 200
        mock_res.read.return_value = json.dumps({
            "data": [
                {"id": "free"},
                {"id": "fast"},
                {"id": "claude-3-7-sonnet"},
                {"id": "deepseek-r1"},
            ]
        }).encode("utf-8")
        mock_conn.getresponse.return_value = mock_res
        mock_acquire.return_value = mock_conn

        main.config._CACHED_MODELS = []
        main.config._CACHED_MODELS_TIME = 0.0

        models = main.models.fetch_upstream_models()
        self.assertEqual(len(models), 4)
        model_ids = [m["id"] for m in models]
        self.assertIn("free", model_ids)
        self.assertIn("claude-3-7-sonnet", model_ids)
        self.assertIn("deepseek-r1", model_ids)

    @patch("auggie_launch.upstream._CONNECTION_POOL.acquire")
    def test_open_upstream_stream_with_reasoning_mocked(self, mock_acquire):
        mock_conn = MagicMock()
        mock_res = MagicMock()
        mock_res.status = 200
        mock_res.headers = {"X-Router-Provider": "anthropic-bedrock", "X-Model-Used": "claude-3-7-sonnet"}
        mock_res.isclosed.return_value = False

        stream_data = (
            b"data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "Refactoring step..."}}]}).encode("utf-8") + b"\n\n" +
            b"data: " + json.dumps({"choices": [{"delta": {"content": "Final code."}}]}).encode("utf-8") + b"\n\n" +
            b"data: [DONE]\n\n"
        )
        mock_res.__iter__.return_value = stream_data.splitlines(keepends=True)
        mock_conn.getresponse.return_value = mock_res
        mock_acquire.return_value = mock_conn

        req_body = main.upstream.json_bytes({"messages": [{"role": "user", "content": "hi"}], "stream": True})
        wrapper = main.upstream.open_upstream_with_retries(req_body, stream=True, timeout=10, label="test")

        lines = [line.decode("utf-8").strip() for line in wrapper]
        wrapper.close()

        self.assertTrue(any("Refactoring step..." in line for line in lines))
        self.assertTrue(any("Final code." in line for line in lines))

    @patch("auggie_launch.registry.fetch_upstream_models")
    def test_fake_models_dynamic_registry(self, mock_fetch):
        mock_fetch.return_value = [
            {"id": "free"},
            {"id": "gemini-2.5-pro"},
            {"id": "deepseek-r1"},
        ]
        models_data = main.registry.fake_models()
        self.assertEqual(models_data["default_model"], "claude-3-7-sonnet")
        registry_json = models_data["feature_flags"]["model_info_registry"]
        registry = json.loads(registry_json)
        self.assertIn("gemini-2.5-pro", registry)
        self.assertEqual(registry["gemini-2.5-pro"]["context"], 1000000)
        self.assertIn("deepseek-r1", registry)
        self.assertEqual(registry["deepseek-r1"]["context"], 128000)
        self.assertIn("free", registry)
        self.assertIn("free", registry)

    @patch("auggie_launch.upstream._CONNECTION_POOL.acquire")
    def test_auto_parameter_swap_on_400(self, mock_acquire):
        mock_conn = MagicMock()
        res_fail = MagicMock()
        res_fail.status = 400
        res_fail.read.return_value = b'{"error": "max_tokens is not supported with this model. Use max_completion_tokens instead."}'
        res_fail.headers = {}
        res_fail.isclosed.return_value = True

        res_success = MagicMock()
        res_success.status = 200
        res_success.headers = {}
        res_success.isclosed.return_value = False
        res_success.read.return_value = b'{"choices": [{"message": {"content": "pong"}}]}'

        mock_conn.getresponse.side_effect = [res_fail, res_success]
        mock_acquire.return_value = mock_conn

        payload = {"model": "test-model", "messages": [], "max_tokens": 1000}
        wrapper = main.upstream.open_upstream_with_retries(main.upstream.json_bytes(payload), stream=False, timeout=10, label="swap-test")
        self.assertEqual(wrapper.status, 200)
        wrapper.close()

        self.assertEqual(mock_conn.request.call_count, 2)
        second_call_body = mock_conn.request.call_args_list[1][1]["body"]
        second_payload = json.loads(second_call_body.decode("utf-8"))
        self.assertIn("max_completion_tokens", second_payload)
        self.assertNotIn("max_tokens", second_payload)






class TestProviderApiKeyMapping(unittest.TestCase):
    """Provider keys are read from the environment."""

    def _env(self, **env):
        with patch.dict(os.environ, env, clear=True):
            return main.injections.build_injected_environment("http://localhost:50108")

    def test_tavily_key_is_injected(self):
        self.assertEqual(self._env(AUGGIE_LAUNCH_TAVILY_API_KEY="tvly-1").get("TAVILY_API_KEY"), "tvly-1")

    def test_shared_key_is_accepted(self):
        self.assertEqual(self._env(FIRECRAWL_API_KEY="fc-1").get("FIRECRAWL_API_KEY"), "fc-1")

    def test_explicit_prefix_beats_shared(self):
        env = self._env(AUGGIE_LAUNCH_EXA_API_KEY="exa-explicit", EXA_API_KEY="exa-shared")
        self.assertEqual(env.get("EXA_API_KEY"), "exa-explicit")

    def test_unset_key_is_not_injected(self):
        self.assertNotIn("TAVILY_API_KEY", self._env())


class TestModelContextInjection(unittest.TestCase):
    """Context limits come from the cached catalog.
    AUGGIE_LAUNCH_MODEL_CONTEXT_TOKENS still overrides everything."""

    def setUp(self):
        self._saved = (
            main.config.TARGET_MODEL,
            main.config.CACHED_CATALOG,
            main.config.MODEL_CONTEXT_TOKENS,
            main.config.MODEL_CONTEXT_TOKENS_EXPLICIT,
            main.config.MODEL_MAX_OUTPUT_TOKENS,
        )
        main.config.TARGET_MODEL = "mid-model"
        main.config.MODEL_CONTEXT_TOKENS = 200000
        main.config.MODEL_CONTEXT_TOKENS_EXPLICIT = False
        main.config.MODEL_MAX_OUTPUT_TOKENS = 16000
        main.config.CACHED_CATALOG = {
            "big-model": {"contextWindow": 1000000},
            "small-model": {"contextWindow": 32000},
            "mid-model": {"contextWindow": 128000},
        }

    def tearDown(self):
        (
            main.config.TARGET_MODEL,
            main.config.CACHED_CATALOG,
            main.config.MODEL_CONTEXT_TOKENS,
            main.config.MODEL_CONTEXT_TOKENS_EXPLICIT,
            main.config.MODEL_MAX_OUTPUT_TOKENS,
        ) = self._saved

    def test_catalog_supplies_each_context(self):
        self.assertEqual(main.models.model_context_limit("big-model"), 1000000)
        self.assertEqual(main.models.model_context_limit("small-model"), 32000)

    def test_qualified_id_falls_back_to_its_last_segment(self):
        self.assertEqual(main.models.model_context_limit("openai/big-model"), 1000000)

    def test_unknown_model_uses_the_default(self):
        self.assertEqual(main.models.model_context_limit("who-is-this"), 200000)

    def test_explicit_env_override_wins(self):
        main.config.MODEL_CONTEXT_TOKENS_EXPLICIT = True
        main.config.MODEL_CONTEXT_TOKENS = 64000
        self.assertEqual(main.models.effective_context_limit("big-model"), 64000)

    def test_model_list_entry_budgets_track_context(self):
        entry = main.registry.model_list_entry("small-model", 32000)
        self.assertEqual(entry["suggested_prefix_char_count"], 32000)
        self.assertLess(entry["completion_timeout_ms"] / 1000, 300)

    def test_build_openai_request_caps_output_tokens(self):
        request = main.transform.build_openai_request(
            {"model": "small-model", "message": "hi", "max_tokens": 900000},
            stream=False,
        )
        # `small-model` has a 32k window, so the ceiling is a quarter of it --
        # the 16k configured maximum must not be handed out wholesale.
        ceiling = max(256, min(16000, max(1024, 32000 // 4)))
        self.assertEqual(ceiling, 8000)
        granted = request.get("max_tokens", request.get("max_completion_tokens"))
        self.assertEqual(granted, ceiling)


class TestLocalProxyAuthorization(unittest.TestCase):
    def _handler(self, header_value=None):
        handler = main.proxy.AuggieProxy.__new__(main.proxy.AuggieProxy)
        handler.headers = {"Authorization": header_value} if header_value else {}
        handler.send_json = MagicMock()
        return handler

    def test_health_and_token_paths_are_open(self):
        handler = self._handler()
        with patch("auggie_launch.config.REQUIRE_LOCAL_TOKEN", True):
            self.assertTrue(handler.authorized("health"))
            self.assertTrue(handler.authorized("token"))
        handler.send_json.assert_not_called()

    def test_valid_token_allowed(self):
        handler = self._handler("Bearer secret-token")
        with patch("auggie_launch.config.REQUIRE_LOCAL_TOKEN", True), patch("auggie_launch.config.LOCAL_TOKEN", "secret-token"):
            self.assertTrue(handler.authorized("chat-stream"))

    def test_missing_or_wrong_token_rejected(self):
        handler = self._handler("Bearer nope")
        with patch("auggie_launch.config.REQUIRE_LOCAL_TOKEN", True), patch("auggie_launch.config.LOCAL_TOKEN", "secret-token"):
            self.assertFalse(handler.authorized("chat-stream"))
        handler.send_json.assert_called_once()
        self.assertEqual(handler.send_json.call_args.kwargs["status"], 401)

    def test_enforcement_can_be_disabled(self):
        handler = self._handler()
        with patch("auggie_launch.config.REQUIRE_LOCAL_TOKEN", False):
            self.assertTrue(handler.authorized("chat-stream"))


if __name__ == "__main__":
    unittest.main()



class TestCodeGPTGeminiCompat(unittest.TestCase):
    """CodeGPT routes through Vertex/Gemini, which rejects two shapes OpenAI allows:
    non-string enum members and a conversation ending on an assistant turn."""

    def test_gemini_safe_schema_coerces_integer_enum(self):
        schema = {"type": "object", "properties": {"verbosity": {"type": "integer", "enum": [1, 2, 3]}}}
        safe = main.codegpt.gemini_safe_schema(schema)
        verbosity = safe["properties"]["verbosity"]
        self.assertEqual(verbosity["enum"], ["1", "2", "3"])
        self.assertEqual(verbosity["type"], "string")

    def test_gemini_safe_schema_keeps_string_enum(self):
        schema = {"type": "string", "enum": ["a", "b"]}
        self.assertEqual(main.codegpt.gemini_safe_schema(schema), {"type": "string", "enum": ["a", "b"]})

    def test_gemini_safe_messages_appends_user_after_assistant(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        fixed = main.codegpt.gemini_safe_messages(msgs)
        self.assertEqual(fixed[-1]["role"], "user")
        self.assertEqual(len(fixed), 3)

    def test_gemini_safe_messages_handles_repeated_assistant_tail(self):
        msgs = [{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]
        fixed = main.codegpt.gemini_safe_messages(msgs)
        self.assertEqual(fixed[-1]["role"], "user")

    def test_gemini_safe_messages_leaves_user_tail_untouched(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(main.codegpt.gemini_safe_messages(msgs), msgs)

    def test_gemini_safe_messages_empty_input(self):
        fixed = main.codegpt.gemini_safe_messages([])
        self.assertEqual(fixed[0]["role"], "user")

    def test_adapt_request_body_flattens_and_sanitises_tools(self):
        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "calculate",
                    "parameters": {"type": "object", "properties": {"mode": {"enum": [1, 2]}}},
                },
            }],
        }
        with patch("auggie_launch.config.TARGET_MODEL", "deepseek-v4.1-flash"), \
             patch("auggie_launch.config.CODEGPT_SESSION_ID", "sess-1"):
            body = main.codegpt.adapt_request_body(request)
        # The model is addressed as modelId, and no agent is involved.
        self.assertEqual(body["modelId"], "deepseek-v4.1-flash")
        self.assertNotIn("agentId", body)
        self.assertEqual(body["session_id"], "sess-1")
        tool = body["tools"][0]
        self.assertEqual(tool["name"], "calc")
        self.assertNotIn("function", tool)
        self.assertEqual(tool["parameters"]["properties"]["mode"]["enum"], ["1", "2"])


class TestRegistryModelPicker(unittest.TestCase):
    """Auggie's /model menu reads displayName/shortName from the registry and
    crashes when they are absent."""

    def test_every_registry_entry_has_picker_fields(self):
        with patch("auggie_launch.config.DYNAMIC_MODELS", False):
            payload = main.registry.fake_models()
        registry = json.loads(payload["feature_flags"]["model_info_registry"])
        self.assertTrue(registry)
        for name, entry in registry.items():
            self.assertTrue(entry.get("displayName"), f"{name} missing displayName")
            self.assertTrue(entry.get("shortName"), f"{name} missing shortName")

    def test_default_model_is_in_registry(self):
        with patch("auggie_launch.config.DYNAMIC_MODELS", False):
            payload = main.registry.fake_models()
        registry = json.loads(payload["feature_flags"]["model_info_registry"])
        self.assertIn(payload["default_model"], registry)





class TestHistorySummarization(unittest.TestCase):
    """Auggie only compacts a long session when the proxy advertises the
    history-summary feature flags; without them the history grows unbounded."""

    def test_summary_flags_present_and_enabled(self):
        with patch("auggie_launch.config.DYNAMIC_MODELS", False), \
             patch("auggie_launch.config.HISTORY_SUMMARY_ENABLED", True), \
             patch("auggie_launch.config.HISTORY_SUMMARY_MIN_VERSION", "0.0.1"):
            ff = main.registry.fake_models()["feature_flags"]
        self.assertEqual(ff["history_summary_min_version"], "0.0.1")
        params = json.loads(ff["history_summary_params"])
        self.assertIn("trigger_on_total_tokens", params)
        self.assertIn("max_history_chars", params)
        self.assertGreater(params["trigger_on_total_tokens"], 0)

    def test_summary_min_version_empty_when_disabled(self):
        with patch("auggie_launch.config.DYNAMIC_MODELS", False), \
             patch("auggie_launch.config.HISTORY_SUMMARY_ENABLED", False):
            ff = main.registry.fake_models()["feature_flags"]
        # An empty min version is how Auggie decides summarization is off.
        self.assertEqual(ff["history_summary_min_version"], "")

    def test_history_summary_params_is_valid_json(self):
        params = json.loads(main.registry.history_summary_params())
        self.assertEqual(sorted(params), ["input_budget_trigger_ratio", "max_history_chars", "trigger_on_total_tokens"])



class TestGeminiToolMessageFolding(unittest.TestCase):
    """Vertex rejects OpenAI-style tool turns ("number of function response parts
    is not equal to the number of function call parts"), so the proxy folds
    `role: tool` results and assistant tool_calls into plain user/assistant text."""

    def test_tool_messages_become_user_text(self):
        msgs = [
            {"role": "user", "content": "read files"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "view", "arguments": '{"path":"a.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "file a content"},
        ]
        out = main.codegpt.gemini_safe_messages(msgs)
        self.assertFalse(any(m.get("role") == "tool" for m in out))
        self.assertFalse(any(m.get("tool_calls") for m in out))
        joined = " ".join(str(m.get("content") or "") for m in out)
        self.assertIn("file a content", joined)
        self.assertIn("view", joined)

    def test_parallel_tool_calls_are_preserved_as_text(self):
        msgs = [
            {"role": "user", "content": "read both"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "view", "arguments": '{"path":"a.py"}'}},
                {"id": "c2", "type": "function", "function": {"name": "view", "arguments": '{"path":"b.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "A"},
            {"role": "tool", "tool_call_id": "c2", "content": "B"},
        ]
        out = main.codegpt.gemini_safe_messages(msgs)
        joined = " ".join(str(m.get("content") or "") for m in out)
        self.assertIn("a.py", joined)
        self.assertIn("b.py", joined)
        self.assertIn("A", joined)
        self.assertIn("B", joined)

    def test_no_two_consecutive_same_role_from_folding(self):
        msgs = [
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "user", "content": "next"},
        ]
        out = main.codegpt.gemini_safe_messages(msgs)
        roles = [m.get("role") for m in out]
        self.assertEqual(roles, ["user"])
        self.assertEqual(out[0]["content"], "[Tool result: r1]\n[Tool result: r2]\nnext")


class TestParallelToolCallMerge(unittest.TestCase):
    """CodeGPT restarts the delta index at 0 for parallel calls; merging purely by
    index concatenates their arguments into invalid JSON."""

    def test_distinct_ids_on_same_index_stay_separate(self):
        deltas = [
            {"index": 0, "id": "aaa", "function": {"name": "view", "arguments": '{"path":"a.py"}'}},
            {"index": 0, "id": "bbb", "function": {"name": "view", "arguments": '{"path":"b.py"}'}},
        ]
        merged = main.truncation.merge_stream_tool_calls(deltas)
        self.assertEqual(len(merged), 2)
        for call in merged:
            json.loads(call["function"]["arguments"])  # must be valid JSON

    def test_fragments_of_one_call_still_merge(self):
        deltas = [
            {"index": 0, "id": "same", "function": {"name": "view", "arguments": '{"pa'}},
            {"index": 0, "id": "same", "function": {"name": "", "arguments": 'th":"a.py"}'}},
        ]
        merged = main.truncation.merge_stream_tool_calls(deltas)
        self.assertEqual(len(merged), 1)
        self.assertEqual(json.loads(merged[0]["function"]["arguments"]), {"path": "a.py"})






class TestFileSearchRouting(unittest.TestCase):
    """`file_search` must resolve to a tool the proxy can actually serve.

    The tempting target, `codebase-retrieval`, is served by Auggie's own
    /agents endpoint which the proxy only stubs out; routing searches there made
    the model read an empty result, retry forever and eventually crash on a
    malformed call. Searches now resolve to real, locally executable tools.
    """

    AVAILABLE = frozenset({
        "view", "save-file", "str-replace-editor", "launch-process",
        "web-fetch", "remove-files", "tavily_search_tavily",
    })

    def route(self, name, args):
        return main.codegpt.route_tool_call(name, args, self.AVAILABLE)

    def test_search_routes_to_a_locally_runnable_tool(self):
        for name, args in [
            ("file_search", {"pattern": r"def func_0_5\("}),
            ("grep_search", {"pattern": "TODO"}),
            ("file_search", {"query": "where is auth"}),
            ("file_search", {"glob": "**/*.py"}),
        ]:
            resolved, _ = self.route(name, args)
            self.assertIn(resolved, self.AVAILABLE, (name, args))
            self.assertNotEqual(resolved, "codebase-retrieval")

    def test_regex_pattern_becomes_a_real_grep(self):
        name, args = self.route("grep_search", {"pattern": "func_0_5"})
        self.assertEqual(name, "launch-process")
        self.assertIn("grep -rn", args["command"])
        self.assertIn("func_0_5", args["command"])

    def test_search_without_pattern_lists_the_scope(self):
        name, args = self.route("file_search", {"query": "what is in here"})
        self.assertEqual(name, "launch-process")
        self.assertIn("ls -la", args["command"])

    def test_explicit_file_plus_pattern_uses_view(self):
        name, args = self.route("file_search", {"pattern": "x", "path": "module_0.py"})
        self.assertEqual(name, "view")
        self.assertEqual(args["path"], "module_0.py")

    def test_shell_metacharacters_in_pattern_are_quoted(self):
        _name, args = self.route("grep_search", {"pattern": "a; rm -rf /"})
        self.assertIn("'a; rm -rf /'", args["command"])

    def test_unknown_tool_is_not_rewritten_to_a_missing_tool(self):
        name, _ = main.codegpt.route_tool_call("file_search", {"query": "x"}, {"view"})
        self.assertNotEqual(name, "codebase-retrieval")



class TestTimeoutConsistency(unittest.TestCase):
    """Three separate deadlines must line up: the pooled socket timeout, the
    proxy's upstream cutoff, and the deadline Auggie is told about."""

    def test_pooled_connection_adopts_latest_timeout(self):
        import urllib.parse

        from auggie_launch.upstream import ConnectionPool
        pool = ConnectionPool(max_idle_seconds=60.0)
        url = urllib.parse.urlparse("https://example.invalid/v1")
        first = pool.acquire(url, timeout=5.0)
        self.assertEqual(first.timeout, 5.0)
        pool.release(url, first, reusable=True)
        second = pool.acquire(url, timeout=120.0)
        self.assertEqual(second.timeout, 120.0)
        if second.sock is not None:
            self.assertEqual(second.sock.gettimeout(), 120.0)

    def test_auggie_deadline_is_below_proxy_cutoff(self):
        with patch("auggie_launch.config.UPSTREAM_TIMEOUT_SECONDS", 300.0):
            entry = main.registry.model_list_entry("m", 100000)
        self.assertLess(entry["completion_timeout_ms"] / 1000, 300.0)


class TestReplyLanguage(unittest.TestCase):
    def test_language_rule_added_when_set(self):
        with patch("auggie_launch.config.REPLY_LANGUAGE", "English"):
            prompt = main.transform.build_system_prompt()
        self.assertIn("English", prompt)

    def test_no_language_rule_when_unset(self):
        with patch("auggie_launch.config.REPLY_LANGUAGE", ""):
            prompt = main.transform.build_system_prompt()
        self.assertNotIn("Always write your answers", prompt)

    def test_auto_is_treated_as_unset(self):
        with patch("auggie_launch.config.REPLY_LANGUAGE", "auto"):
            prompt = main.transform.build_system_prompt()
        self.assertNotIn("Always write your answers", prompt)


class TestConsecutiveRoleCollapse(unittest.TestCase):
    """Vertex merges consecutive same-role turns, which previously produced a
    stalled stream when a tool-only assistant turn was dropped."""

    def test_tool_only_assistant_turn_kept(self):
        msgs = [
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "view", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "hit"},
            {"role": "user", "content": "next"},
        ]
        out = main.codegpt.gemini_safe_messages(msgs)
        roles = [m["role"] for m in out]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(roles[-1], "user")

    def test_no_consecutive_same_role_after_folding(self):
        msgs = [
            {"role": "tool", "tool_call_id": "c1", "content": "r1"},
            {"role": "tool", "tool_call_id": "c2", "content": "r2"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
        ]
        out = main.codegpt.gemini_safe_messages(msgs)
        roles = [m["role"] for m in out]
        for a, b in itertools.pairwise(roles):
            self.assertNotEqual(a, b)



class TestInclusiveModelBridge(unittest.TestCase):
    """CodeGPT Plus serves its inclusive ("economy") models on
    /chat/tools/<harness>, addressed by `modelId`, with the model's upstream in
    an `X-Provider` header. No agent is involved -- an earlier agent-based
    implementation could not reach these models at all."""

    def test_chat_path_uses_the_harness_suffix(self):
        with patch("auggie_launch.config.CODEGPT_HARNESS", "codegpt"):
            self.assertEqual(main.codegpt.chat_path(), "/chat/tools/codegpt")
            self.assertEqual(main.codegpt.chat_path(has_tools=True), "/chat/tools/codegpt")

    def test_upstream_url_uses_the_bridge_path(self):
        with patch("auggie_launch.config.IS_CODEGPT", True), \
             patch("auggie_launch.config.CODEGPT_HARNESS", "codegpt"), \
             patch("auggie_launch.upstream.active_base_url", return_value="https://api.codegpt.co/api/v1"):
            self.assertEqual(
                main.upstream.upstream_url(True),
                "https://api.codegpt.co/api/v1/chat/tools/codegpt",
            )

    def test_provider_header_is_sent(self):
        with patch("auggie_launch.config.CODEGPT_PROVIDER", "openrouter"), \
             patch("auggie_launch.config.CODEGPT_TOKEN", "tok"), \
             patch("auggie_launch.config.IS_CODEGPT", True):
            headers = main.codegpt.extra_headers()
        self.assertEqual(headers["X-Provider"], "openrouter")
        self.assertEqual(headers["Authorization"], "Bearer tok")
        self.assertEqual(headers["tokens"], "true")

    def test_provider_header_omitted_when_unset(self):
        with patch("auggie_launch.config.CODEGPT_PROVIDER", ""), \
             patch("auggie_launch.config.CODEGPT_TOKEN", "tok"), \
             patch("auggie_launch.config.IS_CODEGPT", True):
            headers = main.codegpt.extra_headers()
        self.assertNotIn("X-Provider", headers)

    def test_body_carries_model_id_and_session_id(self):
        request = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}
        with patch("auggie_launch.config.CODEGPT_SESSION_ID", "sess-abc"):
            body = main.codegpt.adapt_request_body(request)
        self.assertEqual(body["modelId"], "deepseek-v4.1-flash")
        self.assertEqual(body["session_id"], "sess-abc")
        self.assertNotIn("agentId", body)


_REMAP_REQUEST = {"tools": [
    {"type": "function", "function": {"name": "launch-process", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "view", "parameters": {"type": "object", "properties": {}}}},
]}


class TestStreamedToolCallRemap(unittest.TestCase):
    """Remapping must happen after the fragments are merged: rewriting each
    delta on its own produced a full command from an empty argument set, then
    concatenated the real arguments after it -- invalid JSON, and the stream
    died with 'Unexpected non-whitespace character after JSON'."""

    def test_fragmented_arguments_produce_one_valid_call(self):
        deltas = [
            {"index": 0, "id": "call_x", "type": "function", "function": {"name": "grep_search", "arguments": ""}},
            {"index": 0, "function": {"arguments": ""}},
            {"index": 0, "function": {"arguments": '{"pattern":"'}},
            {"index": 0, "function": {"arguments": "func_0_5"}},
            {"index": 0, "function": {"arguments": '"'}},
            {"index": 0, "function": {"arguments": "}"}},
        ]
        with patch("auggie_launch.config.IS_CODEGPT", True):
            resolved = main.proxy.resolve_tool_calls(_REMAP_REQUEST, deltas)
        self.assertEqual(len(resolved), 1)
        name = resolved[0]["function"]["name"]
        args = resolved[0]["function"]["arguments"]
        self.assertEqual(name, "launch-process")
        parsed = json.loads(args)  # must not raise
        self.assertIn("grep -rn", parsed["command"])
        self.assertIn("func_0_5", parsed["command"])

    def test_native_tool_names_are_left_alone(self):
        deltas = [
            {"index": 0, "id": "c1", "type": "function", "function": {"name": "view", "arguments": '{"path":"a.py"}'}},
        ]
        with patch("auggie_launch.config.IS_CODEGPT", True):
            resolved = main.proxy.resolve_tool_calls(_REMAP_REQUEST, deltas)
        self.assertEqual(resolved[0]["function"]["name"], "view")
        self.assertEqual(json.loads(resolved[0]["function"]["arguments"]), {"path": "a.py"})



_CATALOG = {
        "models": {
            "deepseek-v4.1-flash": {"economy": True, "serve": {"upstream": "openrouter"}, "contextWindow": 1048576, "tools": True},
            "gemini-3.8-flash": {"economy": True, "serve": {"upstream": "vertex"}, "contextWindow": 1000000, "tools": True, "vision": True},
            "gpt-5.6-luna": {"economy": False, "serve": {"upstream": "openai"}, "contextWindow": 200000},
        }
    }

class TestDynamicCatalog(unittest.TestCase):
    """The inclusive-model list is read from the CodeGPT extension's catalog so
    it follows the plan instead of being pinned in this repo."""

    def _with_catalog(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "model-catalog.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_CATALOG, fh)
        return path

    def test_only_economy_models_are_returned(self):
        path = self._with_catalog()
        with patch.object(main.codegpt, "_catalog_paths", return_value=[path]):
            models = main.codegpt.load_catalog_models()
        ids = [m["id"] for m in models]
        self.assertIn("deepseek-v4.1-flash", ids)
        self.assertIn("gemini-3.8-flash", ids)
        self.assertNotIn("gpt-5.6-luna", ids)

    def test_context_and_capabilities_are_carried(self):
        path = self._with_catalog()
        with patch.object(main.codegpt, "_catalog_paths", return_value=[path]):
            models = {m["id"]: m for m in main.codegpt.load_catalog_models()}
        self.assertEqual(models["deepseek-v4.1-flash"]["context"], 1048576)
        self.assertTrue(models["gemini-3.8-flash"]["vision"])

    def test_upstream_is_translated_to_a_provider_header(self):
        path = self._with_catalog()
        with patch.object(main.codegpt, "_catalog_paths", return_value=[path]):
            models = {m["id"]: m for m in main.codegpt.load_catalog_models()}
        self.assertEqual(models["deepseek-v4.1-flash"]["provider"], "openrouter")
        # Vertex is the one upstream whose header name differs.
        self.assertEqual(models["gemini-3.8-flash"]["provider"], "vertexai")

    def test_missing_catalog_falls_back_to_the_known_plan_list(self):
        with patch.object(main.codegpt, "_catalog_paths", return_value=["/nonexistent/model-catalog.json"]):
            models = main.codegpt.load_catalog_models()
        ids = [m["id"] for m in models]
        self.assertIn("deepseek-v4.1-flash", ids)
        self.assertTrue(all(m.get("provider") for m in models))

    def test_provider_for_model_prefers_the_explicit_setting(self):
        with patch("auggie_launch.config.CODEGPT_PROVIDER", "anthropic"):
            self.assertEqual(main.codegpt.provider_for_model("anything"), "anthropic")

    def test_provider_for_model_reads_the_catalog(self):
        path = self._with_catalog()
        with patch("auggie_launch.config.CODEGPT_PROVIDER", ""), \
             patch.object(main.codegpt, "_catalog_paths", return_value=[path]):
            self.assertEqual(main.codegpt.provider_for_model("deepseek-v4.1-flash"), "openrouter")
            self.assertEqual(main.codegpt.provider_for_model("nope"), "")

    def test_registry_models_come_from_the_catalog(self):
        path = self._with_catalog()
        with patch.object(main.codegpt, "_catalog_paths", return_value=[path]):
            ids = main.registry.codegpt_model_ids()
        self.assertIn("deepseek-v4.1-flash", ids)


class TestReasoningNotLeaked(unittest.TestCase):
    """Raw  tags must not reach the client transcript: the CLI renders
    reasoning itself, and the tags confused the model on later turns."""

    def test_display_disabled_by_default(self):
        with patch("auggie_launch.config.STREAM_THINKING", False):
            self.assertFalse(main.config.STREAM_THINKING)

    def test_config_default_is_off(self):
        # A regression guard on the shipped default, not just the patched value.
        source = open(main.config.__file__, encoding="utf-8").read()
        self.assertIn('STREAM_THINKING = env_truthy("AUGGIE_LAUNCH_STREAM_THINKING", False)', source)



class TestSearchPatternDetection(unittest.TestCase):
    """Tool names are matched by shape, not by a fixed list. A closed list lost
    `glob_search` (and would lose every new model's spelling), leaving the model
    with "Tool not found" and no way to make progress."""

    AVAILABLE = frozenset({"launch-process", "view", "save-file", "str-replace-editor", "remove-files"})

    def _route(self, name, args=None):
        return main.codegpt.route_tool_call(name, args or {"pattern": "x"}, self.AVAILABLE)

    def test_known_search_names(self):
        for name in ("search", "grep", "glob", "find", "file_search", "grep_search", "glob_search"):
            resolved, _ = self._route(name)
            self.assertEqual(resolved, "launch-process", name)

    def test_unseen_search_spellings_are_recognised(self):
        # Names no list ever enumerated, but obviously search-shaped.
        for name in ("codebase_search", "search_symbols", "semantic_search", "repo_glob", "symbol_find"):
            resolved, _ = self._route(name)
            self.assertEqual(resolved, "launch-process", name)

    def test_every_search_resolves_to_a_servable_tool(self):
        for name in ("glob_search", "codebase_search", "file_search", "find_usages"):
            resolved, _ = self._route(name)
            self.assertIn(resolved, self.AVAILABLE, name)





class TestContextWindowFromCatalog(unittest.TestCase):
    """deepseek-v4.1-flash has a 1M-token window, stated only in the CodeGPT
    catalog. Without reading it the proxy told Auggie 200k, so history was
    compacted and truncated far earlier than the model required."""

    def test_codegpt_catalog_supplies_the_window(self):
        entry = {"id": "deepseek-v4.1-flash", "provider": "openrouter", "context": 1048576, "tools": True}
        with patch("auggie_launch.config.IS_CODEGPT", True), \
             patch.object(main.codegpt, "load_catalog_models", return_value=[entry]):
            self.assertEqual(main.models.lookup_catalog_context("deepseek-v4.1-flash"), 1048576)

    def test_codegpt_lookup_ignores_unknown_models(self):
        entry = {"id": "deepseek-v4.1-flash", "provider": "openrouter", "context": 1048576, "tools": True}
        with patch("auggie_launch.config.IS_CODEGPT", True), \
             patch.object(main.codegpt, "load_catalog_models", return_value=[entry]):
            self.assertEqual(main.models.lookup_catalog_context("no-such-model"), 0)



class TestSummaryThresholdsScaleWithWindow(unittest.TestCase):
    """The summarization trigger and the verbatim tail must follow the model's
    real window; fixed values compacted a 1M model as if it had 200k."""

    def _params(self, window, **overrides):
        config_patches = {
            "CODEGPT_SESSION_ID": "s",
            "HISTORY_SUMMARY_TRIGGER_EXPLICIT": False,
            "HISTORY_SUMMARY_MAX_HISTORY_EXPLICIT": False,
            "HISTORY_SUMMARY_TRIGGER_RATIO": 0.6,
            "HISTORY_SUMMARY_TRIGGER_TOKENS": 120000,
            "HISTORY_SUMMARY_MAX_HISTORY_CHARS": 100000,
        }
        config_patches.update(overrides)
        with patch("auggie_launch.config.IS_CODEGPT", True), \
             patch.object(main.registry, "effective_context_limit", return_value=window):
            with patch.multiple("auggie_launch.config", **config_patches):
                return json.loads(main.registry.history_summary_params())

    def test_large_window_raises_the_trigger(self):
        params = self._params(1_048_576)
        self.assertGreater(params["trigger_on_total_tokens"], 200000)
        self.assertEqual(params["trigger_on_total_tokens"], int(1_048_576 * 0.6))

    def test_keep_chars_is_capped_so_summaries_stay_useful(self):
        params = self._params(1_048_576)
        self.assertLessEqual(params["max_history_chars"], 400000)

    def test_small_window_still_gets_a_floor(self):
        params = self._params(8000)
        self.assertGreaterEqual(params["trigger_on_total_tokens"], 16000)
        self.assertGreaterEqual(params["max_history_chars"], 40000)

    def test_explicit_env_value_wins(self):
        params = self._params(1_048_576, HISTORY_SUMMARY_TRIGGER_EXPLICIT=True,
                              HISTORY_SUMMARY_TRIGGER_TOKENS=50000)
        self.assertEqual(params["trigger_on_total_tokens"], 50000)



class TestGlobVersusContentSearch(unittest.TestCase):
    """A filename glob and a content regex must not be served by the same
    command. `grep -rn '**/*.py'` exits non-zero and prints nothing, which the
    model reads as a dead end and then retries."""

    AVAILABLE = frozenset({"launch-process", "view"})

    def _command(self, pattern):
        _name, args = main.codegpt.route_tool_call("glob_search", {"pattern": pattern}, self.AVAILABLE)
        return args["command"]

    def test_filename_globs_use_find(self):
        for pattern in ("**/*.py", "*.txt", "src/**/*.ts", "*"):
            command = self._command(pattern)
            self.assertTrue(command.startswith("find"), (pattern, command))

    def test_content_patterns_use_grep(self):
        for pattern in ("TODO", "func_0_5", r"def \w+\(", "a/b"):
            command = self._command(pattern)
            self.assertTrue(command.startswith("grep"), (pattern, command))

    def test_shell_payload_is_not_treated_as_a_glob(self):
        command = self._command("a; rm -rf /")
        self.assertTrue(command.startswith("grep"), command)
        self.assertIn("'a; rm -rf /'", command)

    def test_glob_command_is_quoted(self):
        command = self._command("**/*.py")
        self.assertIn("'*.py'", command)



class TestSessionsShortcuts(unittest.TestCase):
    """`--sessions` mirrors the CLI's own picker, which needs a TTY, so the list
    is also readable from a plain shell."""

    def _make_session(self, directory, session_id, workspace, turns=3, name=""):
        payload = {"sessionId": session_id, "workspaceRoot": workspace, "chatHistory": [{}] * turns}
        if name:
            payload["name"] = name
        path = os.path.join(directory, f"{session_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    def test_lists_only_the_current_workspace(self):
        import contextlib
        import io
        home = tempfile.mkdtemp()
        sessions = os.path.join(home, ".augment", "sessions")
        os.makedirs(sessions)
        here = tempfile.mkdtemp()
        elsewhere = tempfile.mkdtemp()
        self._make_session(sessions, "aaaa-1111", here, turns=5, name="mine")
        self._make_session(sessions, "bb-2222", elsewhere, turns=9, name="theirs")

        out = io.StringIO()
        with patch("os.path.expanduser", return_value=home), \
             patch("os.getcwd", return_value=here), \
             contextlib.redirect_stdout(out):
            main.cli.print_sessions()
        text = out.getvalue()
        self.assertIn("aaaa-1111", text)
        self.assertNotIn("bb-2222", text)

    def test_shows_a_date_and_turn_count(self):
        import contextlib
        import io
        home = tempfile.mkdtemp()
        sessions = os.path.join(home, ".augment", "sessions")
        os.makedirs(sessions)
        here = tempfile.mkdtemp()
        self._make_session(sessions, "cccc-3333", here, turns=7)

        out = io.StringIO()
        with patch("os.path.expanduser", return_value=home), \
             patch("os.getcwd", return_value=here), \
             contextlib.redirect_stdout(out):
            main.cli.print_sessions()
        text = out.getvalue()
        # A YYYY-MM-DD HH:MM stamp and the turn count must both be present.
        self.assertRegex(text, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")
        self.assertIn("7", text)

    def test_workspace_without_sessions_says_so(self):
        import contextlib
        import io
        home = tempfile.mkdtemp()
        os.makedirs(os.path.join(home, ".augment", "sessions"))
        out = io.StringIO()
        with patch("os.path.expanduser", return_value=home), \
             patch("os.getcwd", return_value=tempfile.mkdtemp()), \
             contextlib.redirect_stdout(out):
            main.cli.print_sessions()
        self.assertIn("no saved sessions", out.getvalue())


class TestLaunchProcessArguments(unittest.TestCase):
    """`launch-process` needs every required field, and omitting keep_stdin_open
    makes a successful command return empty output -- the model then sees a dead
    end and retries the same command indefinitely."""

    AVAILABLE = frozenset({"launch-process", "view"})

    def _args(self, pattern):
        _name, args = main.codegpt.route_tool_call("grep_search", {"pattern": pattern}, self.AVAILABLE)
        return args

    def test_all_required_fields_are_present(self):
        args = self._args("TODO")
        for field in ("command", "cwd", "wait", "max_wait_seconds"):
            self.assertIn(field, args, field)

    def test_keep_stdin_open_is_explicit(self):
        # Absence of this flag is what swallowed the command output.
        self.assertIs(self._args("TODO")["keep_stdin_open"], False)

    def test_command_is_waitable(self):
        args = self._args("TODO")
        self.assertIs(args["wait"], True)
        self.assertGreater(args["max_wait_seconds"], 0)

    def test_glob_command_also_carries_the_flag(self):
        args = self._args("**/*.py")
        self.assertIs(args["keep_stdin_open"], False)


class TestParallelismFlags(unittest.TestCase):
    """Auggie's agent loop picks its execution strategy from these flags:
    without them it calls executeSequentialTools() and independent tool calls run
    one at a time."""

    def _flags(self):
        with patch("auggie_launch.config.DYNAMIC_MODELS", False):
            return main.registry.fake_models()["feature_flags"]

    def test_parallel_tool_execution_is_advertised(self):
        self.assertIs(self._flags().get("beachheadEnableParallelToolExecution"), True)

    def test_subagent_tool_is_advertised(self):
        self.assertIs(self._flags().get("beachheadEnableSubAgentTool"), True)

    def test_subagent_support_is_advertised(self):
        self.assertIs(self._flags().get("enable_subagent_support"), True)

    def test_subagent_records_are_summarised(self):
        self.assertIs(self._flags().get("cliRecordSummarizationsAndSubagents"), True)

    def test_hindsight_stays_disabled(self):
        # It would upload the workspace to Augment, contradicting INDEXING_MODE.
        self.assertIs(self._flags().get("enable_hindsight"), False)

class TestToolNameAliases(unittest.TestCase):
    """Models invent tool names from training data (`file_search`, `bash`).
    Auggie only knows its own names, so invented calls are remapped."""

    AVAILABLE = frozenset({
        "codebase-retrieval", "view", "save-file", "str-replace-editor",
        "launch-process", "web-fetch", "remove-files", "tavily_search_tavily",
    })

    def test_known_names_pass_through(self):
        for name in self.AVAILABLE:
            self.assertEqual(main.codegpt.resolve_tool_name(name, self.AVAILABLE), name)

    def test_invented_names_map_to_real_tools(self):
        cases = {
            "read_file": "view",
            "bash": "launch-process",
            "write_file": "save-file",
            "edit_file": "str-replace-editor",
            "web_search": "tavily_search_tavily",
            "fetch": "web-fetch",
            "delete_file": "remove-files",
        }
        for invented, expected in cases.items():
            self.assertEqual(main.codegpt.resolve_tool_name(invented, self.AVAILABLE), expected, invented)

    def test_unknown_name_without_match_is_left_alone(self):
        self.assertEqual(main.codegpt.resolve_tool_name("totally_made_up", self.AVAILABLE), "totally_made_up")

    def test_alias_not_applied_when_target_missing(self):
        # `web-fetch` is unavailable here, so the alias must not fire blindly.
        limited = {"view"}
        self.assertEqual(main.codegpt.resolve_tool_name("fetch", limited), "fetch")

    def test_adapt_request_body_rewrites_history_tool_calls(self):
        request = {
            "messages": [
                {"role": "user", "content": "search"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "file_search", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "hits"},
            ],
            "tools": [{"type": "function", "function": {
                "name": "launch-process", "description": "x",
                "parameters": {"type": "object", "properties": {}},
            }}],
        }
        body = main.codegpt.adapt_request_body(request)
        joined = " ".join(str(m.get("content") or "") for m in body["messages"])
        self.assertIn("launch-process", joined)
        self.assertNotIn("file_search", joined)


class TestTerminalCommandAliases(unittest.TestCase):
    """The plain-shell name the model reaches for most often."""

    AVAILABLE = frozenset({"launch-process", "view", "save-file"})

    def test_shell_name_variants_map_to_launch_process(self):
        for name in (
            "execute_terminal_command", "run_terminal_command", "run_terminal_cmd",
            "terminal_command", "execute_shell", "shell_command", "execute_bash",
            "execute_command", "bash", "shell", "sh", "terminal", "run",
        ):
            self.assertEqual(
                main.codegpt.resolve_tool_name(name, self.AVAILABLE),
                "launch-process",
                name,
            )

    def test_unmappable_name_is_left_untouched(self):
        self.assertEqual(
            main.codegpt.resolve_tool_name("totally_unknown_thing", self.AVAILABLE),
            "totally_unknown_thing",
        )


class TestConfigValidation(unittest.TestCase):
    """A mistyped provider or a missing token should surface at startup, not
    halfway through a long turn."""

    def _warnings(self, **overrides):
        base = {
            "TARGET_BASE_URL": "https://api.codegpt.co/api/v1",
            "TARGET_MODEL": "deepseek-v4.1-flash",
            "CODEGPT_PROVIDER": "",
            "CODEGPT_TOKEN": "tok",
            "CODEGPT_SESSION_URL": "",
            "API_KEYS": ["k"],
            "MODEL_CONTEXT_TOKENS": 200000,
            "MODEL_MAX_OUTPUT_TOKENS": 16000,
            "MODEL_CONTEXT_TOKENS_EXPLICIT": False,
        }
        base.update(overrides)
        with patch("auggie_launch.config.IS_CODEGPT", True), \
             patch.object(main.codegpt, "load_catalog_models",
                          return_value=[{"id": "deepseek-v4.1-flash", "provider": "openrouter"}]):
            with patch.multiple("auggie_launch.config", **base):
                return main.config.validate_config()

    def test_wrong_pinned_provider_is_reported(self):
        warnings = self._warnings(CODEGPT_PROVIDER="gemini")
        self.assertTrue(any("gemini" in w and "openrouter" in w for w in warnings), warnings)

    def test_matching_provider_is_silent(self):
        self.assertEqual(self._warnings(CODEGPT_PROVIDER="openrouter"), [])

    def test_unpinned_provider_is_silent(self):
        self.assertEqual(self._warnings(CODEGPT_PROVIDER=""), [])

    def test_missing_token_and_session_url_is_reported(self):
        warnings = self._warnings(CODEGPT_TOKEN="", CODEGPT_SESSION_URL="")
        self.assertTrue(any("token" in w for w in warnings), warnings)

    def test_plain_http_to_a_remote_host_is_reported(self):
        warnings = self._warnings(TARGET_BASE_URL="http://example.com/v1")
        self.assertTrue(any("plain http" in w for w in warnings), warnings)

    def test_localhost_http_is_fine(self):
        warnings = self._warnings(TARGET_BASE_URL="http://localhost:8080/v1")
        self.assertFalse(any("plain http" in w for w in warnings))

    def test_output_budget_larger_than_context_is_reported(self):
        warnings = self._warnings(
            MODEL_CONTEXT_TOKENS_EXPLICIT=True, MODEL_CONTEXT_TOKENS=8000, MODEL_MAX_OUTPUT_TOKENS=16000
        )
        self.assertTrue(any("no room" in w for w in warnings), warnings)

    def test_warnings_print_to_stderr_with_a_stable_prefix(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            main.config.report_config_warnings(["one", "two"])
        text = buf.getvalue()
        self.assertEqual(text.count("[auggie-launch] config warning:"), 2)
