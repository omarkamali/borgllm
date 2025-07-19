import os
import pytest
import yaml
import time
from unittest.mock import patch, MagicMock

from borgllm.borgllm import BorgLLM, LLMProviderConfig

# Define a test config path
TEST_CONFIG_PATH = "test_borg.yaml"
TEST_ENV_PATH = ".env.test"


# Helper function to create dummy config and env files
def _create_dummy_files(config_content, env_content):
    with open(TEST_CONFIG_PATH, "w") as f:
        f.write(config_content)
    with open(TEST_ENV_PATH, "w") as f:
        f.write(env_content)


# Helper function to clean up dummy files
def _cleanup_dummy_files():
    if os.path.exists(TEST_CONFIG_PATH):
        os.remove(TEST_CONFIG_PATH)
    if os.path.exists(TEST_ENV_PATH):
        os.remove(TEST_ENV_PATH)


@pytest.fixture(autouse=True)
def reset_borgllm_config_singleton():
    """
    Resets the BorgLLM singleton instance before each test to ensure test isolation.
    """
    # Reset singleton state
    BorgLLM._instance = None
    BorgLLM._config_initialized = False

    # Clear global built-in providers cache
    from borgllm.borgllm import _GLOBAL_BUILTIN_PROVIDERS, _GLOBAL_BUILTIN_LOCK

    with _GLOBAL_BUILTIN_LOCK:
        _GLOBAL_BUILTIN_PROVIDERS.clear()

    yield

    # Clean up after the test if needed (e.g., in case a test creates files)
    _cleanup_dummy_files()

    # Ensure it's clean for subsequent tests if any issue
    BorgLLM._instance = None
    BorgLLM._config_initialized = False

    # Clear global built-in providers cache again
    with _GLOBAL_BUILTIN_LOCK:
        _GLOBAL_BUILTIN_PROVIDERS.clear()


# Common dummy content
COMMON_DUMMY_CONFIG_CONTENT = """
llm:
  providers:
    - name: "gpt-4.1-nano"
      base_url: "https://api.openai.com/v1"
      model: "gpt-4.1-nano"
      api_key: "${OPENAI_API_KEY}"
      temperature: 0.7
      max_tokens: 1000000
    - name: "gpt-4.1-mini"
      base_url: "https://api.openai.com/v1"
      model: "gpt-4.1-mini"
      api_key: "${OPENAI_API_KEY}"
      temperature: 0.7
      max_tokens: 1000000
    - name: "claude-3.5"
      base_url: "https://api.anthropic.com/v1"
      model: "claude-3-5-sonnet-20240620"
      api_key: "${ANTHROPIC_API_KEY}"
      temperature: 0.7
      max_tokens: 128000
    - name: "qwen-moe_s"
      base_url: "http://10.244.0.110:8000/v1"
      model: "/models/Qwen3-30B-A3B-gptq"
      api_key: "sk-dummy"
      temperature: 0.7
      max_tokens: 4096
    - name: "qwen-dense_c"
      base_url: "https://api.cerebras.ai/v1"
      model: "qwen-3-32b"
      api_key: "${CEREBRAS_API_KEY}"
      temperature: 0.7
      max_tokens: 1000000
    - name: "qwen-dense_g"
      base_url: "https://api.groq.com/openai/v1"
      model: "qwen/qwen3-32b"
      api_key: "${GROQ_API_KEY}"
      temperature: 0.7
      max_tokens: 6000
  virtual:
    - name: "qwen-dense"
      upstreams:
        - name: "qwen-dense_c"
        - name: "qwen-dense_g"
    - name: "qwen-best"
      upstreams:
        - name: "qwen-dense"
        - name: "qwen-moe_s"
  default_model: "qwen-best"
"""
COMMON_DUMMY_ENV_CONTENT = """
OPENAI_API_KEY=test_openai_key
ANTHROPIC_API_KEY=test_anthropic_key
CEREBRAS_API_KEY=test_cerebras_key
GROQ_API_KEY=test_groq_key
"""


# Mocked time class
class MockTime:
    def __init__(self, initial_time=0.0):
        self._time = initial_time

    def time(self):
        return self._time

    def sleep(self, seconds):
        print(f"[MOCKED TIME] Sleeping for {seconds:.2f} seconds...")
        self._time += seconds


def test_load_config_and_env_interpolation():
    try:
        _create_dummy_files(COMMON_DUMMY_CONFIG_CONTENT, COMMON_DUMMY_ENV_CONTENT)
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "interpolated_openai_key",
                "ANTHROPIC_API_KEY": "interpolated_anthropic_key",
                "CEREBRAS_API_KEY": "interpolated_cerebras_key",
                "GROQ_API_KEY": "interpolated_groq_key",
            },
            clear=True,
        ):
            # Use get_instance with _force_reinitialize to ensure a fresh singleton for this test
            provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
            print(provider)

            gpt_nano = provider.get("gpt-4.1-nano")
            assert gpt_nano.api_key == "interpolated_openai_key"

            claude = provider.get("claude-3.5")
            assert claude.api_key == "interpolated_anthropic_key"
    finally:
        _cleanup_dummy_files()


def test_get_real_provider():
    try:
        _create_dummy_files(COMMON_DUMMY_CONFIG_CONTENT, COMMON_DUMMY_ENV_CONTENT)
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_openai_key",
                "ANTHROPIC_API_KEY": "test_anthropic_key",
                "CEREBRAS_API_KEY": "test_cerebras_key",
                "GROQ_API_KEY": "test_groq_key",
            },
            clear=True,
        ):
            # Use get_instance with _force_reinitialize
            provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
            gpt_mini = provider.get("gpt-4.1-mini")
            assert isinstance(gpt_mini, LLMProviderConfig)
            assert gpt_mini.name == "gpt-4.1-mini"
            assert str(gpt_mini.base_url) == "https://api.openai.com/v1"
    finally:
        _cleanup_dummy_files()


def test_get_default_model():
    try:
        _create_dummy_files(COMMON_DUMMY_CONFIG_CONTENT, COMMON_DUMMY_ENV_CONTENT)
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_openai_key",
                "ANTHROPIC_API_KEY": "test_anthropic_key",
                "CEREBRAS_API_KEY": "test_cerebras_key",
                "GROQ_API_KEY": "test_groq_key",
            },
            clear=True,
        ):
            # Use get_instance with _force_reinitialize
            provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
            default_model = provider.get()
            # qwen-best is a virtual provider that resolves to one of its upstreams
            # qwen-best -> qwen-dense or qwen-moe_s
            # qwen-dense -> qwen-dense_c or qwen-dense_g
            # So the resolved default should be one of the actual providers
            assert default_model.name in ["qwen-dense_c", "qwen-dense_g", "qwen-moe_s"]
    finally:
        _cleanup_dummy_files()


def test_virtual_provider_token_approximation():
    try:
        _create_dummy_files(COMMON_DUMMY_CONFIG_CONTENT, COMMON_DUMMY_ENV_CONTENT)
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_openai_key",
                "ANTHROPIC_API_KEY": "test_anthropic_key",
                "CEREBRAS_API_KEY": "test_cerebras_key",
                "GROQ_API_KEY": "test_groq_key",
            },
            clear=True,
        ):
            # Use get_instance with _force_reinitialize
            provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)

            # qwen-dense_c has max_tokens=1000000, qwen-dense_g has max_tokens=6000
            # qwen-moe_s has max_tokens=4096

            # Should pick qwen-dense_g or qwen-dense_c
            qwen_dense_5k = provider.get("qwen-dense", approximate_tokens=5000)
            assert qwen_dense_5k.name in ["qwen-dense_c", "qwen-dense_g"]
            assert qwen_dense_5k.max_tokens >= 5000

            # Should pick qwen-dense_c as qwen-dense_g is too small (6000 < 100000)
            qwen_dense_100k = provider.get("qwen-dense", approximate_tokens=100000)
            assert qwen_dense_100k.name == "qwen-dense_c"
            assert qwen_dense_100k.max_tokens >= 100000

            # qwen-best has qwen-dense and qwen-moe_s as upstreams
            # qwen-moe_s has 4096 max_tokens. qwen-dense can provide 1000000 (qwen-dense_c) or 6000 (qwen-dense_g)

            # Should pick qwen-moe_s (4096) or qwen-dense_g (6000) or qwen-dense_c (1000000)
            qwen_best_3k = provider.get("qwen-best", approximate_tokens=3000)
            assert qwen_best_3k.name in ["qwen-moe_s", "qwen-dense_c", "qwen-dense_g"]
            assert qwen_best_3k.max_tokens >= 3000

            # Should pick qwen-dense_g (6000) or qwen-dense_c (1000000)
            qwen_best_5k = provider.get("qwen-best", approximate_tokens=5000)
            assert qwen_best_5k.name in ["qwen-dense_c", "qwen-dense_g"]
            assert qwen_best_5k.max_tokens >= 5000

            # Should pick only qwen-dense_c (1000000)
            qwen_best_100k = provider.get("qwen-best", approximate_tokens=100000)
            assert qwen_best_100k.name == "qwen-dense_c"
            assert qwen_best_100k.max_tokens >= 100000

    finally:
        _cleanup_dummy_files()


def test_virtual_provider_round_robin():
    # Mock time for this test - patch both time module and borgllm module
    mock_time = MockTime(initial_time=200.0)
    with (
        patch("time.time", mock_time.time),
        patch("time.sleep", mock_time.sleep),
        patch("borgllm.borgllm.time.time", mock_time.time),
        patch("borgllm.borgllm.time.sleep", mock_time.sleep),
        patch("borgllm.langchain.time.sleep", mock_time.sleep),
    ):
        try:
            _create_dummy_files(COMMON_DUMMY_CONFIG_CONTENT, COMMON_DUMMY_ENV_CONTENT)
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test_openai_key",
                    "ANTHROPIC_API_KEY": "test_anthropic_key",
                    "CEREBRAS_API_KEY": "test_cerebras_key",
                    "GROQ_API_KEY": "test_groq_key",
                },
                clear=True,
            ):
                # Use get_instance with _force_reinitialize
                provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)

                # After the change to deterministic selection, it should consistently return the first available
                # For 'qwen-dense', it should always pick 'qwen-dense_c' first
                p1 = provider.get("qwen-dense")
                assert p1.name == "qwen-dense_c"

                # Signal 429 for qwen-dense_c, then it should pick qwen-dense_g
                provider.signal_429("qwen-dense_c", duration=1)
                p2 = provider.get("qwen-dense")
                assert p2.name == "qwen-dense_g"

                # After cooldown, it should pick qwen-dense_c again
                mock_time.sleep(1.1)  # Wait for cooldown to expire
                p3 = provider.get("qwen-dense")
                assert p3.name == "qwen-dense_c"

                # For 'qwen-best', it should try 'qwen-dense' first, which resolves to 'qwen-dense_c'
                p_best1 = provider.get("qwen-best")
                assert p_best1.name == "qwen-dense_c"

                # Signal 429 for qwen-dense_c. qwen-best should still try qwen-dense, which will now resolve to qwen-dense_g.
                provider.signal_429("qwen-dense_c", duration=1)
                p_best2 = provider.get("qwen-best")
                assert p_best2.name == "qwen-dense_g"

                # Signal 429 for qwen-dense_g. Now qwen-best should pick qwen-moe_s.
                provider.signal_429("qwen-dense_g", duration=1)
                p_best3 = provider.get("qwen-best")
                assert p_best3.name == "qwen-moe_s"

                # After all cooldowns expire, it should go back to qwen-dense_c (via qwen-dense)
                mock_time.sleep(1.1)  # Wait for cooldowns to expire
                p_best4 = provider.get("qwen-best")
                assert p_best4.name == "qwen-dense_c"

        finally:
            _cleanup_dummy_files()


def test_signal_429_and_cooldown():
    # Mock time for this test
    mock_time = MockTime(initial_time=100.0)
    with patch("time.time", mock_time.time), patch("time.sleep", mock_time.sleep):

        try:
            _create_dummy_files(COMMON_DUMMY_CONFIG_CONTENT, COMMON_DUMMY_ENV_CONTENT)
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test_openai_key",
                    "ANTHROPIC_API_KEY": "test_anthropic_key",
                    "CEREBRAS_API_KEY": "test_cerebras_key",
                    "GROQ_API_KEY": "test_groq_key",
                },
                clear=True,
            ):
                # Use get_instance with _force_reinitialize
                provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
                initial_provider = provider.get("gpt-4.1-nano")
                assert initial_provider.api_key == "test_openai_key"

                # Signal 429 for this provider
                provider.signal_429("gpt-4.1-nano", duration=10)

                # Attempt to get the provider again, should raise error if await_cooldown is False
                with pytest.raises(
                    ValueError, match="is on cooldown and await_cooldown is false"
                ):
                    provider.get("gpt-4.1-nano", allow_await_cooldown=False)

                # Now, attempt to get it with awaiting allowed, and advance time
                mock_time.sleep(5)  # Advance time, still in cooldown
                with pytest.raises(
                    TimeoutError,
                    match="Timeout waiting for provider gpt-4.1-nano to exit cooldown",
                ):
                    provider.get("gpt-4.1-nano", timeout=2, allow_await_cooldown=True)

                mock_time.sleep(5)  # Advance time past cooldown period
                # Should now be able to get it
                refreshed_provider = provider.get(
                    "gpt-4.1-nano", allow_await_cooldown=True
                )
                assert refreshed_provider.api_key == "test_openai_key"

        finally:
            _cleanup_dummy_files()


def test_non_existent_provider():
    try:
        _create_dummy_files(COMMON_DUMMY_CONFIG_CONTENT, COMMON_DUMMY_ENV_CONTENT)
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_openai_key",
                "ANTHROPIC_API_KEY": "test_anthropic_key",
                "CEREBRAS_API_KEY": "test_cerebras_key",
                "GROQ_API_KEY": "test_groq_key",
            },
            clear=True,
        ):
            # Use get_instance with _force_reinitialize
            provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
            with pytest.raises(
                ValueError, match="LLM provider 'non-existent' not found"
            ):
                provider.get("non-existent")
    finally:
        _cleanup_dummy_files()


def test_virtual_provider_no_usable_upstreams_awaits():
    # Mock time for this test to control sleep and time.time()
    mock_time = MockTime(initial_time=200.0)  # Start time for test
    with patch("time.time", mock_time.time), patch("time.sleep", mock_time.sleep):

        # Create a config where all upstreams of a virtual provider are rate-limited initially
        config_content = """
llm:
  providers:
    - name: "limited_1"
      base_url: "http://dummy.com"
      model: "model_a"
      api_key: "key_a"
      temperature: 0.7
      max_tokens: 1000
    - name: "limited_2"
      base_url: "http://dummy.com"
      model: "model_b"
      api_key: "key_b"
      temperature: 0.7
      max_tokens: 1000
  virtual:
    - name: "virtual_limited"
      upstreams:
        - name: "limited_1"
        - name: "limited_2"
"""
        env_content = ""

        try:
            _create_dummy_files(config_content, env_content)
            # Initialize BorgLLM and simulate initial rate limits
            provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
            provider.signal_429("limited_1", duration=5)
            provider.signal_429(
                "limited_2", duration=10
            )  # limited_2 is on cooldown longer

            # Attempt to get virtual provider, it should wait for limited_1 to expire first
            start_attempt_time = mock_time.time()
            # The virtual provider should successfully return limited_1 after waiting 5s
            resolved_provider = provider.get(
                "virtual_limited", timeout=7, allow_await_cooldown=True
            )

            # Should return limited_1 after waiting for its cooldown to expire
            assert resolved_provider.name == "limited_1"
            # After waiting, limited_1 should no longer be unusable, but limited_2 still is
            assert not provider._is_provider_unusable("limited_1")
            assert provider._is_provider_unusable("limited_2")

            # Advance time so both expire
            mock_time.sleep(5)  # Wait for limited_2's remaining cooldown

            # Now both should be available
            resolved_provider2 = provider.get(
                "virtual_limited", allow_await_cooldown=True
            )
            assert resolved_provider2.name in ["limited_1", "limited_2"]
            assert not provider._is_provider_unusable("limited_1")
            assert not provider._is_provider_unusable("limited_2")

        finally:
            _cleanup_dummy_files()


def test_virtual_provider_no_usable_upstreams_throws_without_await():
    # Mock time for this test
    mock_time = MockTime(initial_time=300.0)
    with patch("time.time", mock_time.time), patch("time.sleep", mock_time.sleep):

        config_content = """
llm:
  providers:
    - name: "limited_3"
      base_url: "http://dummy.com"
      model: "model_c"
      api_key: "key_c"
      temperature: 0.7
      max_tokens: 1000
  virtual:
    - name: "virtual_limited_no_await"
      upstreams:
        - name: "limited_3"
"""
        env_content = ""

        try:
            _create_dummy_files(config_content, env_content)
            provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
            provider.signal_429("limited_3", duration=100)  # Put on long cooldown

            # Attempt to get virtual provider with allow_await_cooldown=False
            with pytest.raises(
                ValueError,
                match="No eligible upstream providers for virtual provider virtual_limited_no_await. All are on cooldown.",
            ):
                provider.get("virtual_limited_no_await", allow_await_cooldown=False)

        finally:
            _cleanup_dummy_files()


def test_virtual_provider_await_timeout():
    # Mock time for this test
    mock_time = MockTime(initial_time=400.0)
    with patch("time.time", mock_time.time), patch("time.sleep", mock_time.sleep):

        config_content = """
llm:
  providers:
    - name: "slow_provider"
      base_url: "http://slow.com"
      model: "slow_model"
      api_key: "slow_key"
      temperature: 0.7
      max_tokens: 1000
    - name: "another_slow_provider"
      base_url: "http://anotherslow.com"
      model: "another_slow_model"
      api_key: "another_slow_key"
      temperature: 0.7
      max_tokens: 1000
  virtual:
    - name: "virtual_timeout"
      upstreams:
        - name: "slow_provider"
        - name: "another_slow_provider"
"""
        env_content = ""

        try:
            _create_dummy_files(config_content, env_content)
            provider = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)

            # Put both upstreams on cooldown, one for 5s, one for 10s
            provider.signal_429("slow_provider", duration=5)
            provider.signal_429("another_slow_provider", duration=10)

            # Attempt to get virtual provider with a timeout less than the longest cooldown
            # Should successfully return slow_provider after waiting 5s
            resolved_provider = provider.get(
                "virtual_timeout", timeout=7, allow_await_cooldown=True
            )
            assert resolved_provider.name == "slow_provider"

            # Verify slow_provider is no longer unusable but another_slow_provider still is
            assert not provider._is_provider_unusable("slow_provider")
            assert provider._is_provider_unusable("another_slow_provider")

        finally:
            _cleanup_dummy_files()


def test_virtual_provider_references_non_existent_upstream():
    config_content = """
llm:
  providers:
    - name: "existing_provider"
      base_url: "http://exists.com"
      model: "model_x"
      api_key: "key_x"
      temperature: 0.7
      max_tokens: 1000
  virtual:
    - name: "bad_virtual_provider"
      upstreams:
        - name: "non_existent_upstream"
        - name: "existing_provider"
"""
    env_content = ""
    try:
        _create_dummy_files(config_content, env_content)
        with pytest.raises(
            ValueError,
            match="Virtual provider 'bad_virtual_provider' references non-existent upstream 'non_existent_upstream'.",
        ):
            BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
    finally:
        _cleanup_dummy_files()


def test_init_with_dict_config():
    initial_config = {
        "llm": {
            "providers": [
                {
                    "name": "dict_provider",
                    "base_url": "https://api.dict.com",
                    "model": "dict_model",
                    "api_key": "dict_key",
                    "temperature": 0.5,
                    "max_tokens": 2000,
                }
            ],
            "default_model": "dict_provider",
        }
    }

    # Use get_instance with _force_reinitialize to ensure a fresh singleton
    # Clear environment variables to prevent built-in providers from being added
    with patch.dict(os.environ, {}, clear=True):
        borgllm_instance = BorgLLM.get_instance(
            config_path="nonexistent.yaml", initial_config_data=initial_config
        )

        assert borgllm_instance.config is not None
        assert len(borgllm_instance.providers) == 1
        assert "dict_provider" in borgllm_instance.providers

        provider = borgllm_instance.get("dict_provider")
        assert provider.name == "dict_provider"
        assert provider.api_key == "dict_key"
        assert borgllm_instance._default_provider_name == "dict_provider"


def test_init_with_dict_config_and_env_vars():
    initial_config = {
        "llm": {
            "providers": [
                {
                    "name": "env_provider",
                    "base_url": "https://api.env.com",
                    "model": "env_model",
                    "api_key": "${TEST_ENV_VAR_KEY}",
                    "temperature": 0.5,
                    "max_tokens": 2000,
                }
            ]
        }
    }

    with patch.dict(os.environ, {"TEST_ENV_VAR_KEY": "env_var_value"}, clear=True):
        borgllm_instance = BorgLLM.get_instance(initial_config_data=initial_config)
        provider = borgllm_instance.get("env_provider")
        assert provider.api_key == "env_var_value"


def test_init_with_dict_config_api_keys_list():
    initial_config = {
        "llm": {
            "providers": [
                {
                    "name": "list_provider",
                    "base_url": "https://api.list.com",
                    "model": "list_model",
                    "api_keys": ["key1", "key2", "key3"],
                    "temperature": 0.5,
                    "max_tokens": 2000,
                }
            ]
        }
    }
    # Clear environment variables to prevent built-in providers from being added
    with patch.dict(os.environ, {}, clear=True):
        borgllm_instance = BorgLLM.get_instance(
            config_path="nonexistent.yaml", initial_config_data=initial_config
        )
        provider = borgllm_instance.get("list_provider")
        assert provider.api_key == "key1"
        assert provider.has_multiple_keys()
        assert provider._api_keys == ["key1", "key2", "key3"]


def test_init_with_dict_config_api_key_comma_separated():
    initial_config = {
        "llm": {
            "providers": [
                {
                    "name": "comma_provider",
                    "base_url": "https://api.comma.com",
                    "model": "comma_model",
                    "api_key": "keyA,keyB",
                    "temperature": 0.5,
                    "max_tokens": 2000,
                }
            ]
        }
    }
    # Clear environment variables to prevent built-in providers from being added
    with patch.dict(os.environ, {}, clear=True):
        borgllm_instance = BorgLLM.get_instance(
            config_path="nonexistent.yaml", initial_config_data=initial_config
        )
        provider = borgllm_instance.get("comma_provider")
        assert provider.api_key == "keyA"
        assert provider.has_multiple_keys()
        assert provider._api_keys == ["keyA", "keyB"]


def test_init_with_dict_config_bad_upstream():
    initial_config = {
        "llm": {
            "providers": [
                {
                    "name": "upstream_exists",
                    "base_url": "http://exists.com",
                    "model": "model_a",
                    "api_key": "key_a",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                }
            ],
            "virtual": [
                {
                    "name": "bad_virtual",
                    "upstreams": [{"name": "non_existent"}],
                }
            ],
        }
    }
    with pytest.raises(
        ValueError,
        match="Virtual provider 'bad_virtual' references non-existent upstream 'non_existent'.",
    ):
        BorgLLM.get_instance(initial_config_data=initial_config)


def test_set_default_provider_overrides_yaml():
    try:
        _create_dummy_files(COMMON_DUMMY_CONFIG_CONTENT, COMMON_DUMMY_ENV_CONTENT)
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test_openai_key",
                "ANTHROPIC_API_KEY": "test_anthropic_key",
                "CEREBRAS_API_KEY": "test_cerebras_key",
                "GROQ_API_KEY": "test_groq_key",
            },
            clear=True,
        ):
            borgllm_instance = BorgLLM.get_instance(config_path=TEST_CONFIG_PATH)
            # Initial default from config should be 'qwen-best'
            assert borgllm_instance.config.default_model == "qwen-best"

            # Set a programmatic default
            borgllm_instance.set_default_provider("gpt-4.1-mini")
            default_model = borgllm_instance.get()
            assert default_model.name == "gpt-4.1-mini"

            # Verify that calling set_default_provider with a non-existent provider raises an error
            with pytest.raises(
                ValueError,
                match="Provider 'non_existent_provider' not found. Cannot set as default.",
            ):
                borgllm_instance.set_default_provider("non_existent_provider")

    finally:
        _cleanup_dummy_files()
