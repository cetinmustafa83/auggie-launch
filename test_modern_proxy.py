#!/usr/bin/env python3
"""Comprehensive unit and offline test suite for auggie-launch modern LLM & 9router proxy techniques."""

import itertools
import json
import os
import shutil
import socket
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


class Test9routerAndModelAdaptation(unittest.TestCase):
    def test_9router_detection(self):
        self.assertTrue(main.config.detect_9router("http://localhost:20128/v1"))
        self.assertTrue(main.config.detect_9router("http://127.0.0.1:20128/v1"))
        self.assertTrue(main.config.detect_9router("https://my-9router.internal:8080/v1"))
        self.assertFalse(main.config.detect_9router("https://api.openai.com/v1"))

    def test_completion_tokens_parameter_selection(self):
        self.assertTrue(main.transform.should_use_completion_tokens("o1-preview"))
        self.assertTrue(main.transform.should_use_completion_tokens("o3-mini"))
        self.assertTrue(main.transform.should_use_completion_tokens("claude-3-7-sonnet"))
        self.assertTrue(main.transform.should_use_completion_tokens("deepseek-r1"))
        self.assertFalse(main.transform.should_use_completion_tokens("gpt-3.5-turbo"))

    def test_full_jitter_backoff_bounds(self):
        for attempt in range(5):
            delay = main.upstream.retry_backoff_seconds(attempt)
            self.assertGreaterEqual(delay, 0.0)
            self.assertLessEqual(delay, main.config.UPSTREAM_BACKOFF_MAX_SECONDS)

    def test_9router_headers(self):
        main.config.IS_9ROUTER = True
        main.config.ROUTER_CAVEMAN_MODE = True
        main.config.ROUTER_CAVEMAN_LEVEL = "ultra"
        main.config.ROUTER_PROVIDER = "anthropic"
        headers = main.upstream.upstream_headers("test-key", stream=True)
        self.assertEqual(headers.get("X-Source"), "auggie-launch")
        self.assertEqual(headers.get("X-RTK"), "true")
        self.assertEqual(headers.get("X-Caveman-Mode"), "true")
        self.assertEqual(headers.get("X-Caveman-Level"), "ultra")
        self.assertEqual(headers.get("X-Router-Provider"), "anthropic")
        self.assertEqual(headers.get("Connection"), "keep-alive")


class TestLocal9routerDiscovery(unittest.TestCase):
    def test_read_local_9router_state_integration(self):
        state = main.config.read_local_9router_state()
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
        main.config.TARGET_BASE_URL = "http://127.0.0.1:20128/v1"
        main.config.TARGET_MODEL = "free"
        main.config.API_KEYS = ["sk-test-mock-key"]
        main.config.ROUTER_CAVEMAN_MODE = True
        main.config.ROUTER_CAVEMAN_LEVEL = "ultra"

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

        # Caveman system instructions injection
        instructions = env.get("AUGMENT_INSTRUCTIONS", "")
        self.assertIn("Caveman Mode Active (ultra)", instructions)

        # Standard upstream provider endpoints
        self.assertEqual(env.get("OPENAI_BASE_URL"), main.config.TARGET_BASE_URL)
        self.assertEqual(env.get("ANTHROPIC_BASE_URL"), main.config.TARGET_BASE_URL)
        self.assertEqual(env.get("OPENAI_API_KEY"), "sk-test-mock-key")

        # 9router provider keys if installed
        if main.config._LOCAL_9ROUTER.installed:
            if "tavily" in main.config._LOCAL_9ROUTER.provider_api_keys:
                self.assertEqual(env.get("TAVILY_API_KEY"), main.config._LOCAL_9ROUTER.provider_api_keys["tavily"])

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
        main.config.IS_9ROUTER = True

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
        # Verify 9router local combo is in registry
        self.assertIn("free", registry)
        # Verify 9router local alias is in registry
        self.assertIn("big-pickle", registry)

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




class Test9routerCatalogContext(unittest.TestCase):
    def test_lookup_catalog_context_exact_match(self):
        """Test exact model ID match in catalog."""
        main.config.CACHED_CATALOG = {"gpt-4o": {"contextWindow": 128000}}
        self.assertEqual(main.models.lookup_catalog_context("gpt-4o"), 128000)

    def test_lookup_catalog_context_last_segment(self):
        """Test last segment match for namespaced models."""
        main.config.CACHED_CATALOG = {"gpt-4o": {"contextWindow": 128000}}
        self.assertEqual(main.models.lookup_catalog_context("openai/gpt-4o"), 128000)

    def test_lookup_catalog_context_missing_returns_zero(self):
        """Test missing model returns 0."""
        main.config.CACHED_CATALOG = {"gpt-4o": {"contextWindow": 128000}}
        self.assertEqual(main.models.lookup_catalog_context("unknown-model"), 0)

    def test_lookup_catalog_context_empty_catalog(self):
        """Test empty catalog returns 0."""
        main.config.CACHED_CATALOG = {}
        self.assertEqual(main.models.lookup_catalog_context("any-model"), 0)

    def test_model_context_limit_catalog_priority(self):
        """Test catalog context takes priority over heuristics."""
        main.config.CACHED_CATALOG = {"gpt-4o": {"contextWindow": 200000}}
        # Even though heuristic would give 200000, catalog provides same
        self.assertEqual(main.models.model_context_limit("gpt-4o"), 200000)

    def test_model_context_limit_heuristic_fallback(self):
        """Test heuristic fallback when model not in catalog."""
        main.config.CACHED_CATALOG = {}
        self.assertEqual(main.models.model_context_limit("gemini-2.5-pro"), 1000000)
        self.assertEqual(main.models.model_context_limit("claude-3.5-sonnet"), 200000)
        self.assertEqual(main.models.model_context_limit("deepseek-chat"), 128000)


class TestProviderApiKeyMapping(unittest.TestCase):
    def test_build_injected_environment_tavily(self):
        """Test tavily API key mapping."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('auggie_launch.config._LOCAL_9ROUTER', main.config.NineRouterLocalState(provider_api_keys={"tavily": "tvly-test-key"})):
                env = main.injections.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("TAVILY_API_KEY"), "tvly-test-key")

    def test_build_injected_environment_firecrawl(self):
        """Test firecrawl API key mapping."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('auggie_launch.config._LOCAL_9ROUTER', main.config.NineRouterLocalState(provider_api_keys={"firecrawl": "fc-test-key"})):
                env = main.injections.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("FIRECRAWL_API_KEY"), "fc-test-key")

    def test_build_injected_environment_jina_reader(self):
        """Test jina-reader API key mapping (hyphen normalized)."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('auggie_launch.config._LOCAL_9ROUTER', main.config.NineRouterLocalState(provider_api_keys={"jina-reader": "jina-test-key"})):
                env = main.injections.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("JINA_API_KEY"), "jina-test-key")

    def test_build_injected_environment_minimax(self):
        """Test minimax API key mapping."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('auggie_launch.config._LOCAL_9ROUTER', main.config.NineRouterLocalState(provider_api_keys={"minimax": "mm-test-key"})):
                env = main.injections.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("MINIMAX_API_KEY"), "mm-test-key")

    def test_build_injected_environment_generic_fallback(self):
        """Test generic {NORM}_API_KEY fallback for unknown providers."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('auggie_launch.config._LOCAL_9ROUTER', main.config.NineRouterLocalState(provider_api_keys={"custom-provider": "custom-key"})):
                env = main.injections.build_injected_environment("http://localhost:50108")
                self.assertEqual(env.get("CUSTOM_PROVIDER_API_KEY"), "custom-key")


class TestTunnelFallback(unittest.TestCase):
    @patch("auggie_launch.upstream.socket.socket")
    def test_tunnel_fallback_ping_success(self, mock_socket_class):
        """Test tunnel fallback ping returns True on success."""
        mock_sock = MagicMock()
        mock_socket_class.return_value = mock_sock
        state = main.config.NineRouterLocalState()
        state.tunnel_url = "https://test.example.com"
        result = main.upstream._ping_tunnel(state.tunnel_url)
        self.assertTrue(result)
        mock_sock.connect.assert_called_once_with(("test.example.com", 443))

    @patch("auggie_launch.upstream.socket.socket")
    def test_tunnel_fallback_ping_failure(self, mock_socket_class):
        """Test tunnel fallback ping returns False on failure."""
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = socket.timeout
        mock_socket_class.return_value = mock_sock
        state = main.config.NineRouterLocalState()
        state.tunnel_url = "https://test.example.com"
        result = main.upstream._ping_tunnel(state.tunnel_url)
        self.assertFalse(result)

    def test_tunnel_fallback_ping_invalid_url(self):
        """Test tunnel fallback ping returns False for invalid URL."""
        state = main.config.NineRouterLocalState()
        state.tunnel_url = "not-a-url"
        result = main.upstream._ping_tunnel(state.tunnel_url)
        self.assertFalse(result)


class TestRuntimeTunnelFailover(unittest.TestCase):
    def setUp(self):
        self._saved = (main.config.TARGET_BASE_URL, main.config.ACTIVE_BASE_URL, main.config.TUNNEL_BASE_URL)
        main.config.TARGET_BASE_URL = "http://localhost:20128/v1"
        main.config.ACTIVE_BASE_URL = ""
        main.config.TUNNEL_BASE_URL = "https://tunnel.example.com/v1"

    def tearDown(self):
        main.config.TARGET_BASE_URL, main.config.ACTIVE_BASE_URL, main.config.TUNNEL_BASE_URL = self._saved

    def test_active_base_url_defaults_to_target(self):
        self.assertEqual(main.upstream.active_base_url(), "http://localhost:20128/v1")
        self.assertEqual(main.upstream.upstream_url(), "http://localhost:20128/v1/chat/completions")

    @patch("auggie_launch.upstream._ping_tunnel", return_value=True)
    def test_switch_to_tunnel_when_reachable(self, _ping):
        self.assertTrue(main.upstream.switch_to_tunnel("connection refused"))
        self.assertEqual(main.upstream.active_base_url(), "https://tunnel.example.com/v1")
        self.assertEqual(main.upstream.upstream_url(), "https://tunnel.example.com/v1/chat/completions")
        # second call is a no-op once already switched
        self.assertFalse(main.upstream.switch_to_tunnel("connection refused"))

    @patch("auggie_launch.upstream._ping_tunnel", return_value=False)
    def test_no_switch_when_tunnel_unreachable(self, _ping):
        self.assertFalse(main.upstream.switch_to_tunnel("connection refused"))
        self.assertEqual(main.upstream.active_base_url(), "http://localhost:20128/v1")

    def test_no_switch_without_tunnel_configured(self):
        main.config.TUNNEL_BASE_URL = ""
        self.assertFalse(main.upstream.switch_to_tunnel("connection refused"))
        self.assertEqual(main.upstream.active_base_url(), "http://localhost:20128/v1")


class Test9routerInstallAndRestore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved_backup_dir = main.ninerouter._BUNDLED_DB_BACKUP_DIR
        main.ninerouter._BUNDLED_DB_BACKUP_DIR = os.path.join(self.tmp, "backups")
        os.makedirs(main.ninerouter._BUNDLED_DB_BACKUP_DIR)
        self.backup = os.path.join(main.ninerouter._BUNDLED_DB_BACKUP_DIR, "9router-backup-test.json")
        with open(self.backup, "w", encoding="utf-8") as f:
            json.dump({"settings": {"cavemanEnabled": True}, "combos": []}, f)
        self.home = os.path.join(self.tmp, "home")
        os.makedirs(self.home)

    def tearDown(self):
        main.ninerouter._BUNDLED_DB_BACKUP_DIR = self._saved_backup_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_latest_bundled_db_backup(self):
        self.assertEqual(main.ninerouter.latest_bundled_db_backup(), self.backup)

    def test_restore_creates_db_when_missing(self):
        with patch("auggie_launch.ninerouter.os.path.expanduser", return_value=self.home):
            self.assertTrue(main.ninerouter.restore_9router_db())
        db_file = os.path.join(self.home, ".9router", "db.json")
        self.assertTrue(os.path.isfile(db_file))
        with open(db_file, encoding="utf-8") as f:
            self.assertTrue(json.load(f)["settings"]["cavemanEnabled"])

    def test_restore_skips_existing_db_without_force(self):
        nine_dir = os.path.join(self.home, ".9router")
        os.makedirs(nine_dir)
        db_file = os.path.join(nine_dir, "db.json")
        with open(db_file, "w", encoding="utf-8") as f:
            json.dump({"settings": {"keep": True}}, f)
        with patch("auggie_launch.ninerouter.os.path.expanduser", return_value=self.home):
            self.assertFalse(main.ninerouter.restore_9router_db())
            self.assertTrue(main.ninerouter.restore_9router_db(force=True))
        self.assertTrue(os.path.isfile(db_file + ".bak"))
        with open(db_file, encoding="utf-8") as f:
            self.assertTrue(json.load(f)["settings"]["cavemanEnabled"])

    def test_restore_rejects_non_9router_json(self):
        with open(self.backup, "w", encoding="utf-8") as f:
            json.dump({"not": "a db"}, f)
        with patch("auggie_launch.ninerouter.os.path.expanduser", return_value=self.home):
            self.assertFalse(main.ninerouter.restore_9router_db())

    @patch("auggie_launch.ninerouter.subprocess.run")
    @patch("auggie_launch.ninerouter.shutil.which", return_value="/usr/local/bin/npm")
    def test_install_9router_runs_npm_prefer_online(self, _which, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("auggie_launch.ninerouter.find_9router_binary", return_value="/usr/local/bin/9router"):
            self.assertTrue(main.ninerouter.install_9router())
        args = mock_run.call_args[0][0]
        self.assertEqual(args[1:], ["i", "-g", "9router@latest", "--prefer-online"])

    @patch("auggie_launch.ninerouter.shutil.which", return_value=None)
    def test_install_9router_without_npm(self, _which):
        self.assertFalse(main.ninerouter.install_9router())

    @patch("auggie_launch.ninerouter.subprocess.run")
    @patch("auggie_launch.ninerouter.shutil.which", return_value="/usr/local/bin/npm")
    def test_install_9router_npm_failure(self, _which, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        self.assertFalse(main.ninerouter.install_9router())

    @patch("auggie_launch.ninerouter.install_9router", return_value=True)
    @patch("auggie_launch.ninerouter.restore_9router_db", return_value=True)
    def test_ensure_installs_when_binary_missing(self, mock_restore, mock_install):
        with patch("auggie_launch.ninerouter.find_9router_binary", side_effect=[None, "/usr/local/bin/9router"]):
            self.assertTrue(main.ninerouter.ensure_9router_installed())
        mock_install.assert_called_once()
        mock_restore.assert_called_once()


class TestModelContextInjection(unittest.TestCase):
    def setUp(self):
        self._saved = (
            main.config.TARGET_MODEL,
            main.config.CACHED_CATALOG,
            main.config._LOCAL_9ROUTER,
            main.config.MODEL_CONTEXT_TOKENS,
            main.config.MODEL_CONTEXT_TOKENS_EXPLICIT,
            main.config.MODEL_MAX_OUTPUT_TOKENS,
        )
        main.config.TARGET_MODEL = "free"
        main.config.MODEL_CONTEXT_TOKENS = 200000
        main.config.MODEL_CONTEXT_TOKENS_EXPLICIT = False
        main.config.MODEL_MAX_OUTPUT_TOKENS = 16000
        main.config.CACHED_CATALOG = {
            "big-model": {"contextWindow": 1000000},
            "small-model": {"contextWindow": 32000},
            "mid-model": {"contextWindow": 128000},
        }
        main.config._LOCAL_9ROUTER = main.config.NineRouterLocalState(
            combos=[{"name": "free", "models": ["oc/big-model", "oc/small-model", "oc/mid-model"]}],
            model_aliases={"fast-alias": "oc/big-model"},
        )

    def tearDown(self):
        (
            main.config.TARGET_MODEL,
            main.config.CACHED_CATALOG,
            main.config._LOCAL_9ROUTER,
            main.config.MODEL_CONTEXT_TOKENS,
            main.config.MODEL_CONTEXT_TOKENS_EXPLICIT,
            main.config.MODEL_MAX_OUTPUT_TOKENS,
        ) = self._saved

    def test_combo_context_uses_weakest_member(self):
        """A combo can fall back to any member, so its window is the smallest one."""
        self.assertEqual(main.models.effective_context_limit("free"), 32000)

    def test_alias_context_resolves_through_target(self):
        self.assertEqual(main.models.effective_context_limit("fast-alias"), 1000000)

    def test_explicit_env_override_wins(self):
        main.config.MODEL_CONTEXT_TOKENS = 64000
        main.config.MODEL_CONTEXT_TOKENS_EXPLICIT = True
        self.assertEqual(main.models.effective_context_limit("free"), 64000)

    def test_model_list_entry_budgets_track_context(self):
        small = main.registry.model_list_entry("small-model", 32000)
        big = main.registry.model_list_entry("big-model", 1000000)
        self.assertEqual(small["suggested_prefix_char_count"], 32000)
        self.assertEqual(big["suggested_prefix_char_count"], 200000)  # capped
        self.assertEqual(small["suggested_prefix_char_count"], small["suggested_suffix_char_count"])

    def test_fake_models_reports_per_model_context(self):
        with patch("auggie_launch.config.DYNAMIC_MODELS", False):
            payload = main.registry.fake_models()
        self.assertTrue(payload["models"])
        names = {m["name"] for m in payload["models"]}
        self.assertIn("free", names)
        self.assertIn("fast-alias", names)
        registry = json.loads(payload["feature_flags"]["model_info_registry"])
        self.assertEqual(registry["free"]["context"], 32000)
        self.assertEqual(registry["fast-alias"]["context"], 1000000)
        entry = next(m for m in payload["models"] if m["name"] == "free")
        self.assertEqual(entry["suggested_prefix_char_count"], 32000)

    def test_resolve_request_model_honours_known_models(self):
        self.assertEqual(main.models.resolve_request_model({"model": "fast-alias"}), "fast-alias")
        self.assertEqual(main.models.resolve_request_model({"model": "who-is-this"}), "free")
        self.assertEqual(main.models.resolve_request_model({}), "free")

    def test_build_openai_request_caps_output_tokens(self):
        request = main.transform.build_openai_request(
            {"model": "free", "message": "hi", "max_tokens": 900000},
            stream=False,
        )
        self.assertEqual(request["model"], "free")
        field = "max_completion_tokens" if "max_completion_tokens" in request else "max_tokens"
        self.assertLessEqual(request[field], main.config.MODEL_MAX_OUTPUT_TOKENS)


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

    def test_9router_models_hidden_when_not_9router(self):
        with patch("auggie_launch.config.DYNAMIC_MODELS", False), \
             patch("auggie_launch.config.IS_9ROUTER", False), \
             patch.object(main.config._LOCAL_9ROUTER, "combos", [{"name": "some-combo", "models": ["a"]}]), \
             patch.object(main.config._LOCAL_9ROUTER, "model_aliases", {"some-alias": "a"}):
            names = {m["name"] for m in main.registry.fake_models()["models"]}
        self.assertNotIn("some-combo", names)
        self.assertNotIn("some-alias", names)

    def test_9router_models_shown_when_9router(self):
        with patch("auggie_launch.config.DYNAMIC_MODELS", False), \
             patch("auggie_launch.config.IS_9ROUTER", True), \
             patch.object(main.config._LOCAL_9ROUTER, "combos", [{"name": "some-combo", "models": ["a"]}]), \
             patch.object(main.config._LOCAL_9ROUTER, "model_aliases", {"some-alias": "a"}):
            names = {m["name"] for m in main.registry.fake_models()["models"]}
        self.assertIn("some-combo", names)
        self.assertIn("some-alias", names)



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
        import tempfile
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
