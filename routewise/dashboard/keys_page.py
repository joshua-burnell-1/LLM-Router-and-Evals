"""Models & keys: which models are ready to call, and a safe place to add API keys."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from routewise.dashboard.ui import callout, hero, inject_css, kpis, render
from routewise.models.registry import PROVIDER_LABELS, load_models
from routewise.settings import (
    KEY_PROVIDERS,
    env_file,
    key_status,
    model_status,
    save_key,
    use_key_for_session,
)

inject_css()
render(
    hero(
        "Models & keys",
        "Connect the models you want to test",
        "Bring your own API keys. Use a key for this session only, or save it to a local .env "
        "file that never leaves this machine and is never committed. Claude models also run on "
        "your Claude plan through the claude CLI, with no key at all.",
    )
)

registry = load_models(include_disabled=True)
statuses = {m: model_status(spec) for m, spec in registry.items()}
n_ready = sum(s[0] == "ready" for s in statuses.values())
n_key = sum(s[0] == "needs key" for s in statuses.values())
render(
    kpis(
        [
            ("Ready to call", str(n_ready), "usable in the studio, blind tests and router"),
            ("Need a key", str(n_key), "add one below"),
            ("Providers", str(len({s.provider for s in registry.values()})), "in the catalogue"),
        ]
    )
)

st.markdown("### Model catalogue")
STATE_LABEL = {"ready": "Ready", "needs key": "Needs a key", "off": "Off"}
table = pd.DataFrame(
    [
        {
            "Model": spec.name or m,
            "Provider": PROVIDER_LABELS.get(spec.provider, spec.provider),
            "Status": STATE_LABEL[statuses[m][0]],
            "Connected via": statuses[m][1],
            "Input $ / 1M tokens": spec.input_price,
            "Output $ / 1M tokens": spec.output_price,
            "Price checked": "yes" if spec.price_verified else "not yet",
        }
        for m, spec in registry.items()
    ]
)
st.dataframe(
    table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Input $ / 1M tokens": st.column_config.NumberColumn(format="$%.2f"),
        "Output $ / 1M tokens": st.column_config.NumberColumn(format="$%.2f"),
    },
)
st.caption(
    "Prices are list prices used for cost estimates. Add or edit models in config/models.yaml; "
    "any provider LiteLLM supports will work."
)

st.markdown("### API keys")
render(
    callout(
        "Keys are only ever shown as their last four characters. Saved keys go to "
        f"{env_file().name} with owner-only permissions; it is listed in .gitignore.",
    )
)
ss = st.session_state
for provider, env_name in KEY_PROVIDERS.items():
    label = PROVIDER_LABELS.get(provider, provider)
    status = key_status(provider)
    with st.container(border=True):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.markdown(f"**{label}** · `{env_name}`")
        c2.markdown(f":green[{status}]" if status else ":gray[no key yet]")
        with st.form(f"key-form-{provider}", clear_on_submit=True, border=False):
            key = st.text_input(
                f"{label} API key",
                type="password",
                key=f"key-{provider}",
                placeholder="Paste a key",
                label_visibility="collapsed",
            )
            b1, b2, _ = st.columns([1, 1, 2])
            session_only = b1.form_submit_button("Use for this session", use_container_width=True)
            persist = b2.form_submit_button(
                "Save to .env", type="primary", use_container_width=True
            )
        if session_only or persist:
            try:
                if persist:
                    save_key(provider, key)
                    ss["keys-flash"] = f"Saved the {label} key to {env_file().name}."
                else:
                    use_key_for_session(provider, key)
                    ss["keys-flash"] = f"Using the {label} key until the app restarts."
                st.rerun()
            except ValueError as exc:
                st.error(f"Could not use that key: {exc}.")
if "keys-flash" in ss:
    st.toast(ss.pop("keys-flash"), icon=":material/check_circle:")
