# ========= Sectie: Arbeidsmarkt =========
st.subheader("Arbeidsmarkt")

arb_cols = [c for c in ["unemployment","payrolls","init_claims"] if c in dfp.columns]
if arb_cols:
    dfl = dfp[["date"] + arb_cols].copy()
    dfl = maybe_daily(dfl)

    fig3 = go.Figure()

    if view_mode.startswith("Genormaliseerd"):
        # Single axis (index = 100)
        stack = []
        for i, c in enumerate(arb_cols):
            s = normalize_100(dfl[c])
            stack.append(s)
            fig3.add_trace(go.Scatter(
                x=dfl["date"], y=s, mode="lines", name=c,
                line=dict(width=2, color=PALETTE[i % len(PALETTE)])
            ))
        yr = padded_range(pd.concat(stack, axis=0)) if stack else None
        fig3.update_layout(
            height=400, legend=dict(orientation="h"),
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(title="Datum"),
            yaxis=dict(title="Index (=100)", range=yr)
        )
    else:
        # Dual-axis: payrolls rechts, unemployment + init_claims links
        left_cols  = [c for c in ["unemployment","init_claims"] if c in arb_cols]
        right_col  = "payrolls" if "payrolls" in arb_cols else None

        # Linker-as: unemployment / init_claims
        left_stack = []
        for i, c in enumerate(left_cols):
            s = pd.to_numeric(dfl[c], errors="coerce")
            left_stack.append(s)
            fig3.add_trace(go.Scatter(
                x=dfl["date"], y=s, mode="lines", name=c, yaxis="y",
                line=dict(width=2, color=PALETTE[i % len(PALETTE)])
            ))
        y_left = padded_range(pd.concat(left_stack, axis=0)) if left_stack else None

        # Rechter-as: payrolls (indien aanwezig)
        y_right = None
        if right_col:
            s_r = pd.to_numeric(dfl[right_col], errors="coerce")
            y_right = padded_range(s_r)
            # pak een duidelijk andere kleur (groen) en streepjeslijn
            fig3.add_trace(go.Scatter(
                x=dfl["date"], y=s_r, mode="lines", name=right_col, yaxis="y2",
                line=dict(width=3, dash="dash", color=PALETTE[3])
            ))

        fig3.update_layout(
            height=420, legend=dict(orientation="h"),
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(title="Datum"),
            yaxis=dict(title="Niveau (werkloosheid/claims)", range=y_left),
            yaxis2=dict(title="Niveau (payrolls)", overlaying="y", side="right", range=y_right)
        )

    st.plotly_chart(fig3, use_container_width=True)

    # Dynamische conclusie (laatste Δ)
    last2 = dfl[[c for c in arb_cols if c in dfl.columns]].dropna().tail(2)
    if len(last2) == 2 and last2.shape[1] > 0:
        changes = {c: float(last2[c].iloc[-1] - last2[c].iloc[-2]) for c in last2.columns}
        tone, summary = dynamic_conclusion(changes)
        info_card("Dynamische conclusie (arbeidsmarkt)", [summary], tone=tone)

    # Dagelijkse delta's onder elkaar (groen/rood bars)
    st.markdown("**Dagelijkse delta per serie**")
    for c in arb_cols:
        st.caption(f"Δ {c}")
        tiny_delta_chart(dfl["date"], dfl[c], c)
else:
    st.info("Geen arbeidsmarkt-kolommen gevonden (unemployment/payrolls/init_claims).")

st.divider()
