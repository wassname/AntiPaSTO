#!/usr/bin/env python3
"""Test prompting baseline on Daily Dilemmas eval.

Evaluates models with honest/dishonest persona prompts on Daily Dilemmas dataset.
"""

from loguru import logger
import sys
logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")

from antipasto.train.train_adapter import (
    evaluate_daily_dilemma,
    evaluate_model,
    load_model,
    load_labels,
    TrainingConfig,
    get_choice_ids,
    load_and_process_daily_dilemmas_eval_dataset,
    process_daily_dilemma_results,
    generate_example_output,
)
from antipasto.config import EVAL_BASELINE_MODELS, PROMPT, PERSONAS, proj_root
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import pandas as pd
import gc
from tqdm.auto import tqdm
from antipasto.train.daily_dilemas import format_main_results_table
import re
from pathlib import Path
import gc
import tyro
import time

def sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in filenames."""
    return model_id.replace('/', '_')


def main(config):
    # Config setup
    if config.quick:
        _EVAL_BASELINE_MODELS = EVAL_BASELINE_MODELS[:1]
        config.eval_max_dilemmas = 64
    else:
        _EVAL_BASELINE_MODELS = EVAL_BASELINE_MODELS

    results = []

    for model_name in tqdm(_EVAL_BASELINE_MODELS, desc="Evaluating models"):
        if "0.6B" in model_name:
            config.model_name = model_name
            config.quantization_type = "none"
        else:
            config.model_name = model_name
            config.quantization_type = "4bit"
        model_id = config.model_name
        
        # Check if cache exists for this model
        model_safe = sanitize_model_id(model_id)
        if config.quick:
            model_safe += "_QUICK"
        cache_path = Path(proj_root) / "outputs" / f"baselines/prompting/{model_safe}.parquet"
        cache_path.parent.mkdir(exist_ok=True, parents=True)
    
        if cache_path.exists():
            logger.info(f"Loading cached results from {cache_path}")
            df_cached = pd.read_parquet(cache_path)
            results.append(df_cached)
            continue
        
        # No cache, evaluate the model
        logger.info(f"No cache found for {model_id}, evaluating...")
        base_model, tokenizer = load_model(model_id, quantization_type=config.quantization_type)

        choice_ids = get_choice_ids(tokenizer)

        prompts = [
            PROMPT.format(persona=PERSONAS[0][0]),
            "", # PROMPT.format(persona="a normal").replace("  ", " "),
            PROMPT.format(persona=PERSONAS[1][0]),
        ]
        coeffs = [1.0, 0, -1.0]
        print(f"Using prompts {list(zip(coeffs, prompts))}")
        prompts1 = list(zip(coeffs, prompts))
        
        # Quick test to see if prompting works
        logger.info(f"Quick test of prompting... with model {model_id}")        
        for coeff, prompt in prompts1:            
            t0 = time.time()
            (q, a, score, seq_nll, pmass) = generate_example_output(
                base_model, 
                tokenizer, 
                choice_ids=choice_ids, 
                max_new_tokens=46,
                instructions=prompt  # Match eval loop format
            )
            t1 = time.time()
            if coeff == 1:
                logger.info('='*40+f"\nQ: {q}")
            logger.info(f"Prompt: Coeff={coeff:+.1f}, score={score:.3f}, nll={seq_nll:.3f}, pmass={pmass:.3f}, time={t1-t0:.3f}s\n{a}\n"+'-'*40)
        
        model_results = []
        for coeff, prompt in prompts1:
            dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
                tokenizer,
                instructions=prompt,
                max_tokens=config.eval_max_tokens + 32,  # for prompt tokens
                eval_max_n_dilemmas=config.eval_max_dilemmas
            )
            df_labels = load_labels(dataset_dd)

            d = evaluate_daily_dilemma(
                base_model,
                dataset_dd_pt,
                tokenizer,
                choice_ids,
                batch_size=max(32, config.bs),
            )
            d['model_id'] = model_id
            d['prompt'] = prompt
            d['coeff'] = coeff
            d['method'] = 'prompting'
            model_results.append(d)
        
        # Save per-model cache immediately after evaluation
        df_model = pd.concat(model_results)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        df_model.to_parquet(cache_path)
        logger.info(f"Saved results to {cache_path}")
        results.append(df_model)
        
        # Clean up model from memory
        del base_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Combine all results and show summary
    df_all = pd.concat(results, ignore_index=True)
    logger.info(f"Total results: {len(df_all)} rows from {len(df_all['model_id'].unique())} models")

    # Process and display results for each model
    model_name = _EVAL_BASELINE_MODELS[0]
    _, tokenizer = load_model(model_name, quantization_type="none")
    
    dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
        tokenizer,
        instructions="",
        max_tokens=config.eval_max_tokens + 32,  # for prompt tokens
        eval_max_n_dilemmas=config.eval_max_dilemmas
    )
    df_labels = load_labels(dataset_dd)
    df_labeled = process_daily_dilemma_results(df_all, dataset_dd, df_labels)[0]

    df_scores = []
    for model_name in _EVAL_BASELINE_MODELS:
        config.model_name = model_name
        df_model = df_labeled[df_labeled["model_id"] == model_name]
        if len(df_model) == 0:
            continue

        print(f"\n\n## {model_name} [effect in score*label units]")
        cols_labels = [c for c in df_model.columns if c.startswith("score_")]
        df_res_pv = df_model.groupby(["method", "coeff"])[cols_labels].mean().T
        df_res_pv.index = [s.lstrip("score_") for s in df_res_pv.index]

        # reorder so truthfulness at top, then all ones starting with Virtue/ then MFT, then Emotion
        df_res_pv = df_res_pv.reindex(
            sorted(
                df_res_pv.index,
                key=lambda x: (
                    not x.startswith("Value/Honesty"),
                    not x.startswith("Virtue/"),
                    not x.startswith("MFT/"),
                    x,
                ),
            ),
            axis=0,
        )
        print(df_res_pv.head(3).round(3).to_markdown())

        print(f"\n\n## {model_name} [effect in logscore]")
        
        md_table, df_eff_sz, main_score = format_main_results_table(
            df_model,
            # 
            config=config,
            target_method='prompting',
            show_alt_measures=False,
        )
        print(md_table)
        df_scores.append(dict(main_score=main_score, model_name=model_name, method="prompting"))
    df_scores_all = pd.DataFrame(df_scores)
    print("\n\n### Summary of main scores ###")
    print(df_scores_all.sort_values("main_score", ascending=False).to_markdown(index=False))

    output_file = cache_path = Path(proj_root) / "outputs" / 'prompting_results.csv'
    df_scores_all.to_csv(output_file, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig, use_underscores=True)
    main(config)



