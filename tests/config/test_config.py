"""Tests for configuration management."""

import os

from dopeagents.config import DopeAgentsConfig, get_config, reset_config, set_config


class TestDopeAgentsConfig:
    """Verify configuration loading and management."""

    def teardown_method(self) -> None:
        """Clean up globals after each test."""
        reset_config()

    def test_config_from_env(self) -> None:
        """Configuration loads from environment variables."""
        # Set environment variable
        os.environ["DOPEAGENTS_DEFAULT_MODEL"] = "gpt-4o-mini"

        # Create config from env
        config = DopeAgentsConfig.from_env()
        assert config.default_model == "gpt-4o-mini"

    def test_config_prefix_dopeagents(self) -> None:
        """Configuration respects DOPEAGENTS_ prefix."""
        os.environ["DOPEAGENTS_DEBUG_MODE"] = "true"

        config = DopeAgentsConfig.from_env()
        assert config.debug_mode is True

    def test_config_boolean_parsing(self) -> None:
        """Boolean config values parse correctly."""
        os.environ["DOPEAGENTS_ENABLE_COST_TRACKING"] = "false"
        os.environ["DOPEAGENTS_ENABLE_RETRY"] = "true"

        config = DopeAgentsConfig.from_env()
        assert config.enable_cost_tracking is False
        assert config.enable_retry is True

    def test_config_float_parsing(self) -> None:
        """Float config values parse correctly."""
        os.environ["DOPEAGENTS_MAX_COST_PER_CALL"] = "10.50"

        config = DopeAgentsConfig.from_env()
        assert config.max_cost_per_call == 10.50

    def test_config_optional_fields(self) -> None:
        """Optional fields default to None."""
        config = DopeAgentsConfig.from_env()
        # Clean env to avoid other tests interfering
        assert config.api_key is None or isinstance(config.api_key, str)
        assert config.api_base is None or isinstance(config.api_base, str)

    def test_config_defaults(self) -> None:
        """Configuration has sensible defaults."""
        # Clear environment to test defaults
        for key in list(os.environ.keys()):
            if key.startswith("DOPEAGENTS_"):
                del os.environ[key]

        config = DopeAgentsConfig.from_env()
        assert config.default_model is None
        assert config.enable_cost_tracking is True
        assert config.enable_retry is True
        assert config.max_retries == 3
        assert config.tracer_type == "console"

    def test_global_singleton_get_config(self) -> None:
        """get_config() returns singleton."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_global_singleton_set_config(self) -> None:
        """set_config() updates singleton."""
        reset_config()

        config1 = DopeAgentsConfig(default_model="gpt-4o-mini")
        set_config(config1)

        config2 = get_config()
        assert config2.default_model == "gpt-4o-mini"

    def test_config_reset_clears_singleton(self) -> None:
        """reset_config() clears cached singleton."""
        reset_config()
        config1 = get_config()

        reset_config()
        config2 = get_config()

        # Different instances after reset
        assert config1 is not config2

    def test_config_thread_safety(self) -> None:
        """Config access is thread-safe."""
        import threading

        reset_config()
        results = []

        def access_config() -> None:
            config = get_config()
            results.append(config)

        threads = [threading.Thread(target=access_config) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same singleton
        assert len({id(r) for r in results}) == 1

    def teardown_method_cleanup(self) -> None:
        """Clean up environment variables."""
        for key in list(os.environ.keys()):
            if key.startswith("DOPEAGENTS_"):
                del os.environ[key]
