#!/usr/bin/env python3
"""Test if configured LLM model is accessible and detect capabilities."""

import sys

from dopeagents.config import get_config

try:
    import litellm

    # Get and resolve active model
    config = get_config()
    active_model = config.resolve_model()
    has_key = config.has_api_key()

    print("=" * 60)
    print("CONFIG ANALYSIS")
    print("=" * 60)
    print("✓ Config loaded successfully")
    print(f"✓ Active model: {active_model}")
    print(f"✓ Has API key: {has_key}")

    if not has_key:
        print("✗ ERROR: No API key configured!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MODEL ACCESSIBILITY TEST")
    print("=" * 60)

    # Extract provider from model string
    provider_prefix = active_model.split("/")[0]
    print(f"📍 Detected provider: {provider_prefix}")

    # Get provider config
    provider_config = config.get_provider_config(provider_prefix)
    print(f"📋 Provider config: {provider_config}")

    # Test API call with minimal prompt
    print(f"\n🔄 Testing API call to {active_model}...")
    print("   (using minimal prompt for quick connectivity check)")

    response = litellm.completion(
        model=active_model,
        messages=[{"role": "user", "content": "Say 'OK'"}],
        max_tokens=10,
        temperature=0.1,
        **provider_config,  # Pass provider-specific config
    )

    print("✅ SUCCESS: Model is accessible!")

    # Initialize capabilities dict
    capabilities = {
        "thinking": False,
        "tool_calls": False,
        "function_calling": False,
        "vision": False,
        "streaming": False,
        "extended_thinking": False,
    }

    # Parse initial response for capabilities
    response_dict = {}

    # Check for capability indicators in response
    response_str = str(response).lower()
    if "thinking" in response_str or "thought" in response_str:
        capabilities["thinking"] = True
    if "tool" in response_str or "function_call" in response_str:
        capabilities["tool_calls"] = True

    # Check choices for tool/function calls
    if (
        response
        and response.choices
        and len(response.choices) > 0
        and hasattr(response.choices[0], "message")
    ):
        message = response.choices[0].message
        if hasattr(message, "tool_calls") and message.tool_calls:
            capabilities["tool_calls"] = True
        if hasattr(message, "function_call") and message.function_call:
            capabilities["function_calling"] = True

    # Inspect raw response for additional metadata
    if hasattr(response, "__dict__"):
        response_dict = {
            k: v
            for k, v in response.__dict__.items()
            if not k.startswith("_") and k not in ["choices", "usage"]
        }

    # TEST EXTENDED THINKING CAPABILITY FIRST (before displaying capabilities)
    print("\n🔄 Testing extended thinking capability...")
    try:
        thinking_response = litellm.completion(
            model=active_model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+2? Think step by step.",
                }
            ],
            max_tokens=100,
            temperature=0.1,
            extra_body={
                "reasoning_budget": 1024,
                "chat_template_kwargs": {"enable_thinking": True},
            },
            **provider_config,
        )

        if thinking_response.choices and thinking_response.choices[0].message:
            msg = thinking_response.choices[0].message
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                capabilities["thinking"] = True
                capabilities["extended_thinking"] = True

        # Check usage for thinking tokens
        if (
            hasattr(thinking_response, "usage")
            and thinking_response.usage
            and hasattr(thinking_response.usage, "thinking_tokens")
        ):
            thinking_tokens = thinking_response.usage.thinking_tokens
            if thinking_tokens and thinking_tokens > 0:
                capabilities["thinking"] = True
                capabilities["extended_thinking"] = True

    except Exception:
        pass  # Thinking not supported, proceed with other tests

    # Display basic response metadata
    print("\n📊 Response Metadata:")
    if hasattr(response, "usage") and response.usage:
        print(f"   - Tokens: {response.usage.completion_tokens} completion tokens")

    if hasattr(response, "model"):
        print(f"   - Model endpoint: {response.model}")

    if (
        response
        and response.choices
        and len(response.choices) > 0
        and hasattr(response.choices[0], "message")
    ):
        content = response.choices[0].message.content
        if content:
            print(f"   - Response: {content[:50]}...")
        else:
            print("   - Response: (empty)")
    print("\n" + "=" * 60)
    print("🧠 MODEL CAPABILITIES")
    print("=" * 60)

    capability_symbols = {
        "thinking": "🤔",
        "tool_calls": "🔧",
        "function_calling": "⚙️",
        "vision": "👁️",
        "streaming": "📡",
        "extended_thinking": "💭",
    }

    for cap, enabled in capabilities.items():
        symbol = capability_symbols.get(cap, "•")
        status = "✅" if enabled else "❌"
        print(f"   {symbol} {cap.replace('_', ' ').title():.<30} {status}")

    # Display raw metadata if available
    if response_dict:
        print("\n📋 Additional Response Metadata:")
        for key, value in response_dict.items():
            if value is not None and not callable(value):
                # Truncate long values
                val_str = str(value)
                if len(val_str) > 50:
                    val_str = val_str[:50] + "..."
                print(f"   - {key}: {val_str}")

    print("\n" + "=" * 60)

except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print("✗ ERROR: Model is NOT accessible")
    print("\nError details:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {e!s}")
    sys.exit(1)
