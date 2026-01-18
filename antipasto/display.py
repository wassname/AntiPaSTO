"""Display helpers for Jupyter notebooks.

Rich HTML output with score-colored formatting for steering demos.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML, display
from typing import List
from antipasto.gen import ScaleAdapter
from antipasto.train.train_adapter import generate_example_output


# Diverging colormap: red (negative) ↔ white (neutral) ↔ blue (positive)
CMAP = plt.cm.RdBu
NORM = mcolors.Normalize(vmin=-15, vmax=15)


def score_to_hex(score: float) -> str:
    """Map score to hex color using matplotlib colormap."""
    rgba = CMAP(NORM(score))
    return mcolors.to_hex(rgba)


def format_steered_output(
    coeff: float,
    direction: str,
    q: str,
    a: str,
    score: float,
    nll: float,
    pmass: float,
    show_q: bool = False,
    max_text_len: int = 40000,
    bool_q: bool = True,
) -> str:
    """Format a single steered output as HTML string.
    
    Args:
        coeff: Steering coefficient used
        direction: Name of the steering direction (e.g., "honest")
        q: Question/prompt text
        a: Answer/response text
        score: Log-ratio score (log P(yes)/P(no))
        nll: Negative log-likelihood of the response
        pmass: Probability mass on Yes/No tokens
        show_q: Whether to include the prompt
        max_text_len: Maximum characters to show from response
        bool_q: Whether this is a Yes/No evaluation question (shows P(Yes) etc. if True)
        
    Returns:
        HTML string (not displayed, for batching)
    """
    color = score_to_hex(score) if bool_q else "#888"
    
    # Build coeff label with direction (e.g., "1× honest" or "-0.5× honest")
    if coeff > 0:
        coeff_label = f"+{coeff:g}× {direction}"
    elif coeff < 0:
        coeff_label = f"{coeff:g}× {direction}"
    else:
        coeff_label = "baseline (0×)"
    
    parts = []
    
    if show_q:
        # Clean prompt: show user turn, then any model forcing prefix
        q_clean = q.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
        q_clean = q_clean.replace("user\n", "👤 ").replace("model\n", "\n🤖 ")
        parts.append(f"""
        <div style="background: #e3f2fd; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
            <b>📝 Prompt:</b>
            <pre style="margin: 6px 0 0 0; white-space: pre-wrap; font-family: monospace; font-size: 0.95em; color: #333;">{q_clean.strip()}</pre>
        </div>
        """)
    
    # Clean chat markers from response (the generated part only)
    a_clean = a.replace("user\n", "👤 ").replace("model\n", "🤖 ").replace("<end_of_turn>", "").replace("<start_of_turn>", "")
    
    # Build metrics badges (only if bool_q)
    metrics_html = ""
    if bool_q:
        prob_yes = 1 / (1 + np.exp(-score))
        score_bg = score_to_hex(score)
        luminance = 0.299 * CMAP(NORM(score))[0] + 0.587 * CMAP(NORM(score))[1] + 0.114 * CMAP(NORM(score))[2]
        text_color = "#fff" if luminance < 0.05 else "#000"
        pmass_color = "#d32f2f" if pmass < 0.99 else "#4caf50"
        pmass_text = "#fff" if pmass < 0.99 else "#000"
        
        metrics_html = f"""
            <span style="background: {score_bg}; color: {text_color}; padding: 2px 6px; border-radius: 3px; margin-left: 8px; font-weight: bold;">
                P(Yes): {prob_yes:.0%}
            </span>
            <span style="color: #999; margin-left: 8px; font-size: 0.85em;">
                log-ratio: {score:+.2f} | nll: {nll:.2f}
            </span>
            <span style="background: {pmass_color}; color: {pmass_text}; padding: 2px 6px; border-radius: 3px; margin-left: 8px; font-size: 0.85em;">
                pmass: {pmass:.1%}
            </span>
        """
    
    parts.append(f"""
    <div style="border-left: 4px solid {color}; padding: 8px 12px; margin: 8px 0; background: #fafafa; border-radius: 4px;">
        <div style="margin-bottom: 6px;">
            <b>🤖 steer = {coeff_label}</b>{metrics_html}
        </div>
        <div style="color: #555; font-family: monospace; white-space: pre-wrap; font-size: 0.95em;">{a_clean[:max_text_len].strip()}</div>
    </div>
    """)
    
    return "\n".join(parts)


def display_steered_output(
    coeff: float,
    direction: str,
    q: str,
    a: str,
    score: float,
    nll: float,
    pmass: float,
    show_q: bool = False,
    max_text_len: int = 40000,
):
    """Display a single steered output with colored formatting.
    
    Args:
        coeff: Steering coefficient used
        direction: Name of the steering direction (e.g., "honest")
        q: Question/prompt text
        a: Answer/response text
        score: Log-ratio score (log P(yes)/P(no))
        nll: Negative log-likelihood of the response
        pmass: Probability mass on Yes/No tokens
        show_q: Whether to display the prompt
        max_text_len: Maximum characters to show from response
    """
    html = format_steered_output(coeff, direction, q, a, score, nll, pmass, show_q, max_text_len)
    display(HTML(html))


def run_steering_demo(
    model,
    tokenizer,
    choice_ids,
    direction: str = None,
    coeffs=[-1, 0, 1],
    max_new_tokens: int = 48,
    max_text_len: int = 40000,
    skip_special_tokens=True,
    model_name: str = None,
    warn_low_pmass: bool = False,
    bool_q=True,
     **kwargs
):
    """Run demo for multiple coefficients with colored output.
    
    Args:
        model: The model with adapter loaded
        tokenizer: Tokenizer for the model
        choice_ids: Token IDs for Yes/No choices (from get_choice_ids)
        direction: Name of the steering direction (e.g., "honest"). Auto-detected from training config if None.
        coeffs: List of steering coefficients to test
        max_new_tokens: Maximum tokens to generate
        max_text_len: Maximum characters to display from response
        model_name: Model name for header (e.g., "Gemma-3-12B"). Auto-detected if None.
        
    Returns:
        List of (coeff, q, a, score, seq_nll, pmass) tuples
    """
    # Auto-detect direction and personas from training config
    pos_persona = None
    neg_persona = None
    if hasattr(model, 'antipasto_training_config'):
        tc = model.antipasto_training_config
        personas = tc.get('PERSONAS', [])
        if len(personas) >= 2:
            pos_persona = personas[0][0] if personas[0] else None  # e.g., "an honest"
            neg_persona = personas[1][0] if personas[1] else None  # e.g., "a dishonest"
            if direction is None and pos_persona:
                # Extract direction from persona (e.g., "an honest" -> "honest")
                direction = pos_persona.replace("an ", "").replace("a ", "").strip()
    
    if direction is None:
        direction = "steering"  # Fallback
    
    # Auto-detect model name if not provided
    if model_name is None:
        if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
            model_name = model.config._name_or_path.split('/')[-1]
        else:
            model_name = "Model"
    
    # Collect all results first, then build HTML
    results = []
    for coeff in coeffs:
        with ScaleAdapter(model, coeff=coeff):
            q, a, score, seq_nll, pmass = generate_example_output(
                model, tokenizer, choice_ids=choice_ids, max_new_tokens=max_new_tokens,
                skip_special_tokens=skip_special_tokens, warn_low_pmass=warn_low_pmass, **kwargs
            )
        results.append((coeff, q, a, score, seq_nll, pmass))
    
    # Build single HTML block
    html_parts = []
    
    # Header with branding
    html_parts.append(f"""
    <div style="background: linear-gradient(135deg, #ff9a56, #ff6b6b); color: white; padding: 10px 15px; 
                border-radius: 6px 6px 0 0; font-weight: bold;">
        🍝 AntiPaSTO | <span style="font-weight: normal;">{model_name}</span> 
        <span style="float: right; font-weight: normal; opacity: 0.9;">steering: {direction}</span>
    </div>
    <div style="border: 1px solid #ddd; border-top: none; border-radius: 0 0 6px 6px; padding: 12px; background: #fff;">
    """)
    
    # Add each result with persona labels
    # Always use the positive direction for clarity: +1× honest = more honest, -1× honest = less honest
    for i, (coeff, q, a, score, seq_nll, pmass) in enumerate(results):
        label = direction  # Use consistent label (the positive direction)
            
        html_parts.append(format_steered_output(
            coeff, label, q, a, score, seq_nll, pmass,
            show_q=(i == 0), max_text_len=max_text_len, bool_q=bool_q
        ))
    
    html_parts.append("</div>")
    
    # Display as single block
    display(HTML("\n".join(html_parts)))
    
    return results
