#!/usr/bin/env python3
"""Comprehensive unit and offline test suite for auggie-launch modern LLM & 9router proxy techniques."""

import io
import json
import os
import socket
import unittest
from unittest.mock import MagicMock, patch
import main


class TestTokenAndAtomicTruncation(unittest.TestCase):
    def test_estimate_tokens(self):
        text = "def hello():\n    return 'world'"
        tokens = main.estimate_tokens_heuristic(text)
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

        system_msgs, turns = main.group_messages_into_turns(messages)
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

        truncated = main.truncate_messages_to_context_limit(messages, max_context_tokens=5000)

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
        merged = main.merge_stream_tool_calls(chunks)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["id"], "call_101")
        self.assertEqual(merged[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(merged[0]["function"]["arguments"]), {"path": "app.py"})

        self.assertEqual(merged[1]["id"], "call_102")
        self.assertEqual(merged[1]["function"]["name"], "search")
        self.assertEqual(json.loads(merged[1]["function"]["arguments"]), {"query": "main"})

    def test_json_repair_unclosed_brackets(self):
        broken = '{"key": "value", "list": [1, 2'
        repaired = main.repair_json_arguments(broken)
        parsed = json.loads(repaired)
        self.assertEqual(parsed.get("key"), "value")
        self.assertEqual(parsed.get("list"), [1, 2])


class Test9routerAndModelAdaptation(unittest.TestCase):
    def test_9router_detection(self):
        self.assertTrue(main.detect_9router("http://localhost:20128/v1"))
        self.assertTrue(main.detect_9router("http://127.0.0.1:20128/v1"))
        self.assertTrue(main.detect_9router("https://my-9router.internal:8080/v1"))
        self.assertFalse(main.detect_9router("https://api.openai.com/v1"))

    def test_completion_tokens_parameter_selection(self):
        self.assertTrue(main.should_use_completion_tokens("o1-preview"))
        self.assertTrue(main.should_use_completion_tokens("o3-mini"))
        self.assertTrue(main.should_use_completion_tokens("claude-3-7-sonnet"))
        self.assertTrue(main.should_use_completion_tokens("deepseek-r1"))
        self.assertFalse(main.should_use_completion_tokens("gpt-3.5-turbo"))

    def test_full_jitter_backoff_bounds(self):
        for attempt in range(5):
            delay = main.retry_backoff_seconds(attempt)
            self.assertGreaterEqual(delay, 0.0)
            self.assertLessEqual(delay, main.UPSTREAM_BACKOFF_MAX_SECONDS)

    def test_9router_headers(self):
        main.IS_9ROUTER = True
        main.ROUTER_CAVEMAN_MODE = True
        main.ROUTER_CAVEMAN_LEVEL = "ultra"
        main.ROUTER_PROVIDER = "anthropic"
        headers = main.upstream_headers("test-key", stream=True)
        self.assertEqual(headers.get("X-Source"), "auggie-launch")
        self.assertEqual(headers.get("X-RTK"), "true")
        self.assertEqual(headers.get("X-Caveman-Mode"), "true")
        self.assertEqual(headers.get("X-Caveman-Level"), "ultra")
        self.assertEqual(headers.get("X-Router-Provider"), "anthropic")
        self.assertEqual(headers.get("Connection"), "keep-alive")


class TestLocal9routerDiscovery(unittest.TestCase):
    def test_read_local_9router_state_integration(self):
        state = main.read_local_9router_state()
        if state.installed:
            self.assertTrue(state.installed)
            self.assertTrue(os.path.isfile(state.db_path))
            self.assertIsInstance(state.combos, list)
            self.assertIsInstance(state.model_aliases, dict)
            # Verify "free" combo is present
            combo_names = [c.get("name") for c in state.combos]
            self.assertIn("free", combo_names)
            # Verify aliases present
            self.assertIn("big-pickle", state.model_aliases)
            # Verify provider keys
            self.assertIn("tavily", state.provider_api_keys)
            self.assertIn("firecrawl", state.provider_api_keys)
            # Verify tunnel URL
            self.assertTrue(state.tunnel_url.startswith("https://"))


class TestAuggieFullInjections(unittest.TestCase):
    def setUp(self):
        main.TARGET_BASE_URL = "http://127.0.0.1:20128/v1"
        main.TARGET_MODEL = "free"
        main.API_KEYS = ["sk-test-mock-key"]
        main.ROUTER_CAVEMAN_MODE = True
        main.ROUTER_CAVEMAN_LEVEL = "ultra"

    def test_build_injected_environment(self):
        proxy_url = "http://127.0.0.1:54321"
        env = main.build_injected_environment(proxy_url)

        # Core Augment variables
        self.assertEqual(env.get("AUGMENT_API_URL"), proxy_url)
        self.assertEqual(env.get("AUGMENT_API_TOKEN"), main.LOCAL_TOKEN)
        self.assertEqual(env.get("AUGMENT_DISABLE_AUTO_UPDATE"), "1")
        self.assertEqual(env.get("AUGMENT_MODEL"), "free")

        # Session Auth
        auth = json.loads(env.get("AUGMENT_SESSION_AUTH", "{}"))
        self.assertEqual(auth.get("accessToken"), main.LOCAL_TOKEN)
        self.assertEqual(auth.get("tenantURL"), proxy_url)

        # Caveman system instructions injection
        instructions = env.get("AUGMENT_INSTRUCTIONS", "")
        self.assertIn("Caveman Mode Active (ultra)", instructions)

        # Standard upstream provider endpoints
        self.assertEqual(env.get("OPENAI_BASE_URL"), main.TARGET_BASE_URL)
        self.assertEqual(env.get("ANTHROPIC_BASE_URL"), main.TARGET_BASE_URL)
        self.assertEqual(env.get("OPENAI_API_KEY"), "sk-test-mock-key")

        # 9router provider keys if installed
        if main._LOCAL_9ROUTER.installed:
            if "tavily" in main._LOCAL_9ROUTER.provider_api_keys:
                self.assertEqual(env.get("TAVILY_API_KEY"), main._LOCAL_9ROUTER.provider_api_keys["tavily"])

        # PATH enhancement
        self.assertIn("/bin", env.get("PATH", ""))

    def test_generate_injected_mcp_config(self):
        mcp_path = main.generate_injected_mcp_config()
        if mcp_path:
            self.assertTrue(os.path.isfile(mcp_path))
            with open(mcp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("mcpServers", data)


class TestMockedNetworkOperations(unittest.TestCase):
    def setUp(self):
        main.reset_throttles()
        main.TARGET_BASE_URL = "http://127.0.0.1:20128/v1"
        main.TARGET_MODEL = "claude-3-7-sonnet"
        main.API_KEYS = ["sk-test-mock-key"]
        main.IS_9ROUTER = True

    @patch("main._CONNECTION_POOL.acquire")
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

        main._CACHED_MODELS = []
        main._CACHED_MODELS_TIME = 0.0

        models = main.fetch_upstream_models()
        self.assertEqual(len(models), 4)
        model_ids = [m["id"] for m in models]
        self.assertIn("free", model_ids)
        self.assertIn("claude-3-7-sonnet", model_ids)
        self.assertIn("deepseek-r1", model_ids)

    @patch("main._CONNECTION_POOL.acquire")
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

        req_body = main.json_bytes({"messages": [{"role": "user", "content": "hi"}], "stream": True})
        wrapper = main.open_upstream_with_retries(req_body, stream=True, timeout=10, label="test")

        lines = [line.decode("utf-8").strip() for line in wrapper]
        wrapper.close()

        self.assertTrue(any("Refactoring step..." in l for l in lines))
        self.assertTrue(any("Final code." in l for l in lines))

    @patch("main.fetch_upstream_models")
    def test_fake_models_dynamic_registry(self, mock_fetch):
        mock_fetch.return_value = [
            {"id": "free"},
            {"id": "gemini-2.5-pro"},
            {"id": "deepseek-r1"},
        ]
        models_data = main.fake_models()
        self.assertEqual(models_data["default_model"], "claude-3-7-sonnet")
        registry_json = models_data["feature_flags"]["model_info_registry"]
        registry = json.loads(registry_json)
        self.assertIn("gemini-2.5-pro", registry)
        self.assertEqual(registry["gemini-2.5-pro"]["context"], 1000000)
        self.assertIn("deepseek-r1", registry)
        self.assertEqual(registry["deepseek-r1"]["context"], 128000)
        # Verify 9router local combo is in registry
        self.assertIn("free", registry)
        # Verify 9router local alias is in registry
        self.assertIn("big-pickle", registry)

    @patch("main._CONNECTION_POOL.acquire")
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
        wrapper = main.open_upstream_with_retries(main.json_bytes(payload), stream=False, timeout=10, label="swap-test")
        self.assertEqual(wrapper.status, 200)
        wrapper.close()

        self.assertEqual(mock_conn.request.call_count, 2)
        second_call_body = mock_conn.request.call_args_list[1][1]["body"]
        second_payload = json.loads(second_call_body.decode("utf-8"))
        self.assertIn("max_completion_tokens", second_payload)
        self.assertNotIn("max_tokens", second_payload)




class Test9routerCatalogContext(unittest.TestCase):
    def test_lookup_catalog_context_exact_match(self):
        """Test exact model ID match in catalog."""
        main.CACHED_CATALOG = {"gpt-4o": {"contextWindow": 128000}}
        self.assertEqual(main.lookup_catalog_context("gpt-4o"), 128000)

    def test_lookup_catalog_context_last_segment(self):
        """Test last segment match for namespaced models."""
        main.CACHED_CATALOG = {"gpt-4o": {"contextWindow": 128000}}
        self.assertEqual(main.lookup_catalog_context("openai/gpt-4o"), 128000)

    def test_lookup_catalog_context_missing_returns_zero(self):
        """Test missing model returns 0."""
        main.CACHED_CATALOG = {"gpt-4o": {"contextWindow": 128000}}
        self.assertEqual(main.lookup_catalog_context("unknown-model"), 0)

    def test_lookup_catalog_context_empty_catalog(self):
        """Test empty catalog returns 0."""
        main.CACHED_CATALOG = {}
        self.assertEqual(main.lookup_catalog_context("any-model"), 0)

    def test_model_context_limit_catalog_priority(self):
        """Test catalog context takes priority over heuristics."""
        main.CACHED_CATALOG = {"gpt-4o": {"contextWindow": 200000}}
        # Even though heuristic would give 200000, catalog provides same
        self.assertEqual(main.model_context_limit("gpt-4o"), 200000)

    def test_model_context_limit_heuristic_fallback(self):
        """Test heuristic fallback when model not in catalog."""
        main.CACHED_CATALOG = {}
        self.assertEqual(main.model_context_limit("gemini-2.5-pro"), 1000000)
        self.assertEqual(main.model_context_limit("claude-3.5-sonnet"), 200000)
        self.assertEqual(main.model_context_limit("deepseek-chat"), 128000)


class TestProviderApiKeyMapping(unittest.TestCase):
    def test_build_injected_environment_tavily(self):
        """Test tavily API key mapping."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('main._LOCAL_9ROUTER', main.NineRouterLocalState(provider_api_keys={"tavily": "tvly-test-key"})):
                env = main.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("TAVILY_API_KEY"), "tvly-test-key")

    def test_build_injected_environment_firecrawl(self):
        """Test firecrawl API key mapping."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('main._LOCAL_9ROUTER', main.NineRouterLocalState(provider_api_keys={"firecrawl": "fc-test-key"})):
                env = main.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("FIRECRAWL_API_KEY"), "fc-test-key")

    def test_build_injected_environment_jina_reader(self):
        """Test jina-reader API key mapping (hyphen normalized)."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('main._LOCAL_9ROUTER', main.NineRouterLocalState(provider_api_keys={"jina-reader": "jina-test-key"})):
                env = main.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("JINA_API_KEY"), "jina-test-key")

    def test_build_injected_environment_minimax(self):
        """Test minimax API key mapping."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('main._LOCAL_9ROUTER', main.NineRouterLocalState(provider_api_keys={"minimax": "mm-test-key"})):
                env = main.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("MINIMAX_API_KEY"), "mm-test-key")

    def test_build_injected_environment_generic_fallback(self):
        """Test generic {NORM}_API_KEY fallback for unknown providers."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('main._LOCAL_9ROUTER', main.NineRouterLocalState(provider_api_keys={"custom-provider": "custom-key"})):
                env = main.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("CUSTOM_PROVIDER_API_KEY"), "custom-key")


class TestTunnelFallback(unittest.TestCase):
    @patch("main.socket.socket")
    def test_tunnel_fallback_ping_success(self, mock_socket_class):
        """Test tunnel fallback ping returns True on success."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        state = main.NineRouterLocalState()
        state.tunnel_url = "https://test.example.com"
        result = main._ping_tunnel(state.tunnel_url)
        self.assertTrue(result)
        mock_sock.connect.assert_called_once_with(("test.example.com", 443))

    @patch("main.socket.socket")
    def test_tunnel_fallback_ping_failure(self, mock_socket_class):
        """Test tunnel fallback ping returns False on failure."""
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = socket.timeout
        mock_socket_class.return_value = mock_sock
        state = main.NineRouterLocalState()
        state.tunnel_url = "https://test.example.com"
        result = main._ping_tunnel(state.tunnel_url)
        self.assertFalse(result)

    def test_tunnel_fallback_ping_invalid_url(self):
        """Test tunnel fallback ping returns False for invalid URL."""
        state = main.NineRouterLocalState()
        state.tunnel_url = "not-a-url"
        result = main._ping_tunnel(state.tunnel_url)
        self.assertFalse(result)


class TestRuntimeTunnelFailover(unittest.TestCase):
    def setUp(self):
        self._saved = (main.TARGET_BASE_URL, main.ACTIVE_BASE_URL, main.TUNNEL_BASE_URL)
        main.TARGET_BASE_URL = "http://localhost:20128/v1"
        main.ACTIVE_BASE_URL = ""
        main.TUNNEL_BASE_URL = "https://tunnel.example.com/v1"

    def tearDown(self):
        main.TARGET_BASE_URL, main.ACTIVE_BASE_URL, main.TUNNEL_BASE_URL = self._saved

    def test_active_base_url_defaults_to_target(self):
        self.assertEqual(main.active_base_url(), "http://localhost:20128/v1")
        self.assertEqual(main.upstream_url(), "http://localhost:20128/v1/chat/completions")

    @patch("main._ping_tunnel", return_value=True)
    def test_switch_to_tunnel_when_reachable(self, _ping):
        self.assertTrue(main.switch_to_tunnel("connection refused"))
        self.assertEqual(main.active_base_url(), "https://tunnel.example.com/v1")
        self.assertEqual(main.upstream_url(), "https://tunnel.example.com/v1/chat/completions")
        # second call is a no-op once already switched
        self.assertFalse(main.switch_to_tunnel("connection refused"))

    @patch("main._ping_tunnel", return_value=False)
    def test_no_switch_when_tunnel_unreachable(self, _ping):
        self.assertFalse(main.switch_to_tunnel("connection refused"))
        self.assertEqual(main.active_base_url(), "http://localhost:20128/v1")

    def test_no_switch_without_tunnel_configured(self):
        main.TUNNEL_BASE_URL = ""
        self.assertFalse(main.switch_to_tunnel("connection refused"))
        self.assertEqual(main.active_base_url(), "http://localhost:20128/v1")


if __name__ == "__main__":
    unittest.main()
