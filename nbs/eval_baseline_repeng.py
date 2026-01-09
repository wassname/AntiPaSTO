#!/usr/bin/env python3
"""Test repeng baseline on our training data and Daily Dilemmas eval.

Tests the original vgel/repeng library (PCA-based control vectors) on our
honest/dishonest training data and evaluates on Daily Dilemmas.
"""

import gc
import sys
from pathlib import Path
import tyro
import numpy as np
import pandas as pd
import torch
from loguru import logger
from repeng import ControlModel, ControlVector
from repeng.control import model_layer_list
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from antipasto import make_dataset
from antipasto.config import EVAL_BASELINE_MODELS, TrainingConfig, proj_root
from antipasto.train.daily_dilemas import (
    evaluate_daily_dilemma,
    format_main_results_table,
    load_and_process_daily_dilemmas_eval_dataset,
    load_labels,
    process_daily_dilemma_results,
)
from antipasto.train.train_adapter import (
    create_train_dataset,
    generate_example_output,
    get_choice_ids,
    load_model,
)

logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")




def load_baselines():
    files = list((Path(proj_root) / "outputs" / "baselines" / "repeng").glob("*.parquet"))
    results = []
    for f in files:
        df = pd.read_parquet(f)
        results.append(df)
    return results

def sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in filenames."""
    return model_id.replace('/', '_')


def main(config):
    # Config
    # config = TrainingConfig(
    #     eval_batch_size=32,
    #     # dataset_max_samples=800,
    # )

    if config.quick:
        # layers 2
        _EVAL_BASELINE_MODELS = EVAL_BASELINE_MODELS[:1]
        config.eval_max_dilemmas = 64
        config.max_samples = 100
    else:
        _EVAL_BASELINE_MODELS = EVAL_BASELINE_MODELS

    eval_batch_size = max(32, config.bs)

    results = []

    for model_name in tqdm(EVAL_BASELINE_MODELS, desc="Evaluating models"):
        # Set quantization based on model size (same as prompting baseline)
        # we don't support 8bit and 4bit bnb yet
        config.model_name = model_name
        config.quantization_type = "none"

        # Check if cache exists for this model
        model_safe = sanitize_model_id(model_name)
        cache_path = Path(proj_root) / "outputs" / f"baselines/repeng/{model_safe}.parquet"
        if config.quick:
            model_safe += "_QUICK"
        

        if cache_path.exists():
            logger.info(f"Loading cached results from {cache_path}")
            df_cached = pd.read_parquet(cache_path)
            results.append(df_cached)
            continue

        # No cache, evaluate the model
        logger.info(f"No cache found for {model_name}, evaluating...")
        base_model, tokenizer = load_model(model_name, quantization_type=config.quantization_type)

        # repeng uses layers relative to end: [-5, -6, -7, ...]
        try:
            N = base_model.config.num_hidden_layers
        except AttributeError:
            # gemma models don't have config.num_hidden_layers
            # print(base_model)
            from repeng.control import model_layer_list
            N = len(model_layer_list(base_model))
        repeng_layers = list(range(-5, -N // 2, -1))  # last half layers
        if config.quick:
            repeng_layers = repeng_layers[:2]

        model = ControlModel(base_model, repeng_layers)

        logger.info(f"Loaded model: {model_name}, repeng layers: {repeng_layers}")

        train_honest, train_dataset_pt, val_honest, val_dataset_pt = create_train_dataset(
            config,
            tokenizer,
            max_size=config.max_samples
        )

        logger.info(f"Created dataset with {len(train_honest)} pairs")

        # Train control vector
        logger.info("Training control vector with repeng...")
        control_vector = ControlVector.train(
            model,
            tokenizer,
            train_honest,
            batch_size=eval_batch_size,
            hidden_layers=repeng_layers,
        )
        logger.info("Control vector trained")

        model.reset()

        choice_ids = get_choice_ids(tokenizer)

        # Quick test
        logger.info("Quick test of PCA A-steering vectors...")
        for coeff in [-1.0, 0.0, 1.0]:
            model.reset()
            if coeff != 0.0:
                model.set_control(control_vector, coeff)

            (q, a, score, seq_nll, pmass) = generate_example_output(
                model,
                tokenizer,
                choice_ids=choice_ids,
                max_new_tokens=128,
            )
            logger.info(f"Coeff={coeff:+.1f}, score={score:.3f}, nll={seq_nll:.3f} , pmass={pmass:.3f}\n{a}")

        # Load eval dataset
        dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
            tokenizer,
            instructions="",
            max_tokens=config.eval_max_tokens,
            eval_max_n_dilemmas=config.eval_max_dilemmas,
        )
        df_labels = load_labels(dataset_dd)

        # Evaluate at different coefficients
        model_results = []

        for coeff in [-1.0, 0.0, 1.0]:
            logger.info(f"Evaluating repeng at coeff={coeff}")

            # Set control vector
            model.reset()
            if coeff != 0.0:
                model.set_control(control_vector, coeff)

            # Evaluate
            d = evaluate_daily_dilemma(
                model,
                dataset_dd_pt,
                tokenizer,
                choice_ids,
                batch_size=eval_batch_size,
            )

            d["model_id"] = model_name
            d["coeff"] = coeff
            d["method"] = "repeng"
            model_results.append(d)

        # Save per-model cache immediately after evaluation
        df_model = pd.concat(model_results)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        df_model.to_parquet(cache_path)
        logger.info(f"Saved results to {cache_path}")
        results.append(df_model)

        # Clean up model from memory
        del base_model, tokenizer, model, control_vector
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info("Done with evaluation! Processing results...")
    df_labeled = process_and_display_results(results, config)
    df_scores = []
    for model_name in EVAL_BASELINE_MODELS:
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

        print(f"\n\n## {model_name} [effect in score*label units]")
        md_table, df_eff_sz, main_score = format_main_results_table(
            df_model,
            config=config,
            target_method="repeng",
        )
        print(md_table)
        df_scores.append(dict(main_score=main_score, model_name=model_name, method="repeng"))
    df_scores_all = pd.DataFrame(df_scores)
    print("\n\n### Summary of main scores ###")
    print(df_scores_all.sort_values("main_score", ascending=False).to_markdown(index=False))
    output_file = cache_path = Path(proj_root) / "outputs" / 'repeng_results.csv'
    df_scores_all.to_csv(output_file, index=False)
    logger.info("Done!")

def process_and_display_results(results: list[pd.DataFrame], config: TrainingConfig = None, ):
    """Load cached results and display formatted tables for each model.
    
    Args:
        config: Training configuration
        results: List of DataFrames with evaluation results (optional, will load from cache if empty)
    """
    if config is None:
        config = TrainingConfig()
    # Load all cached results if not provided
    if not results:
        results = []
        for model_name in EVAL_BASELINE_MODELS:
            model_safe = sanitize_model_id(model_name)
            if config.quick:
                model_safe += "_QUICK"
            cache_path = Path(proj_root) / "outputs" / f"baselines/repeng/{model_safe}.parquet"
            
            if cache_path.exists():
                df_cached = pd.read_parquet(cache_path)
                results.append(df_cached)
    
    if not results:
        logger.warning("No results to process")
        return
    
    # Combine all results and show summary
    df_all = pd.concat(results, ignore_index=True)
    logger.info(f"Total results: {len(df_all)} rows from {len(df_all['model_id'].unique())} models")

    # Load tokenizer and dataset (needed for processing)
    model_name = df_all['model_id'].iloc[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_dd, dataset_dd_pt = load_and_process_daily_dilemmas_eval_dataset(
        tokenizer,
        instructions="",
        max_tokens=config.eval_max_tokens,
        eval_max_n_dilemmas=config.eval_max_dilemmas
    )

    df_labels = load_labels(dataset_dd)
    df_labeled = process_daily_dilemma_results(df_all, dataset_dd, df_labels)[0]
    logger.info("Processed results with labels")
    return df_labeled




if __name__ == "__main__":
    from antipasto.config import default_configs
    config = tyro.cli(TrainingConfig, use_underscores=True  )
    main(config)