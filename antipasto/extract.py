import dataclasses
import os
import typing
import warnings
from typing import Callable, Literal, OrderedDict
from torch import Tensor
import gguf
import numpy as np
from sklearn.decomposition import PCA
import torch
from jaxtyping import Float, Int
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import tqdm
from baukit import TraceDict

from .dataset import DatasetEntry



@dataclasses.dataclass
class ControlVector:
    model_type: str
    directions: dict[str, torch.Tensor]

    @classmethod
    def train(
        cls,
        model: "PreTrainedModel",
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[DatasetEntry],
        hidden_layers: typing.Iterable[str] | None = None,
        batch_size: int = 32,
        **kwargs,
    ) -> "ControlVector":
        """
        Train a ControlVector for a given model and tokenizer using the provided dataset.

        Args:
            model (PreTrainedModel | ControlModel): The model to train against.
            tokenizer (PreTrainedTokenizerBase): The tokenizer to tokenize the dataset.
            dataset (list[DatasetEntry]): The dataset used for training.
            **kwargs: Additional keyword arguments.
                batch_size (int, optional): The maximum batch size for training.
                    Defaults to 32. Try reducing this if you're running out of memory.
                method (str, optional): The training method to use. Options:
                    - "pca_diff": PCA on difference vectors (default)
                    - "pca_center": PCA on centered vectors
                    - "pca_diff_weighted": PCA with weighted diff vectors
                    - "pca_center_weighted": PCA with weighted centered vectors
                    - "umap": UMAP dimensionality reduction
                    - "svd_gradient": SVD on gradients from DPO loss
        Returns:
            ControlVector: The trained vector.
        """
        # the order is [positive, negative, positive, negative, ...]
        train_strs = [s for ex in dataset for s in (ex.positive, ex.negative)]

        # Determine if we need gradients based on method
        method = kwargs.get('method', 'pca_diff')
        needs_grads = any(m in method for m in ['gradient', 'fisher', 'hvp'])
        
        if needs_grads:
            # # Full gradient collection for gradient-based methods
            # act, logprobs, grads, feat_grad_norms = _collect_activations_grads(
            #     model, tokenizer, train_strs, hidden_layers, batch_size
            # )
            1/0  # WIP: gradient collection disabled for now
        else:
            # Lightweight activation-only collection for PCA methods
            act, logprobs = _collect_activations_only(
                model, tokenizer, train_strs, hidden_layers, batch_size
            )
            grads = None
            feat_grad_norms = None

        # compute directions
        dirs = read_representations(
            act, logprobs, grads, feat_grad_norms,
            **kwargs,
        )

        # init class
        return cls(model_type=model.config.model_type, directions=dirs)

    def export_gguf(self, path: os.PathLike[str] | str):
        """
        Export a trained ControlVector to a llama.cpp .gguf file.
        Note: This file can't be used with llama.cpp yet. WIP!

            vector = ControlVector.train(...)
            vector.export_gguf("path/to/write/vector.gguf")
        """

        arch = "controlvector"
        writer = gguf.GGUFWriter(path, arch)
        writer.add_string(f"{arch}.model_hint", self.model_type)
        writer.add_uint32(f"{arch}.layer_count", len(self.directions))
        for layer in self.directions.keys():
            writer.add_tensor(f"direction.{layer}", self.directions[layer])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    @classmethod
    def import_gguf(cls, path: os.PathLike[str] | str) -> "ControlVector":
        reader = gguf.GGUFReader(path)

        archf = reader.get_field("general.architecture")
        if not archf or not len(archf.parts):
            warnings.warn(".gguf file missing architecture field")
        else:
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "controlvector":
                warnings.warn(
                    f".gguf file with architecture {arch!r} does not appear to be a control vector!"
                )

        modelf = reader.get_field("controlvector.model_hint")
        if not modelf or not len(modelf.parts):
            raise ValueError(".gguf file missing controlvector.model_hint field")
        model_hint = str(bytes(modelf.parts[-1]), encoding="utf-8")

        directions = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            layer = tensor.name[len("direction.") :]
            if not layer:
                raise ValueError(
                    f".gguf file has invalid direction field name: {tensor.name}"
                )
            directions[layer] = tensor.data
        return cls(model_type=model_hint, directions=directions)

    def _helper_combine(
        self, other: "ControlVector", other_coeff: float
    ) -> "ControlVector":
        if self.model_type != other.model_type:
            warnings.warn(
                "Trying to add vectors with mismatched model_types together, this may produce unexpected results."
            )

        model_type = self.model_type
        directions: dict[str, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = self.directions[layer]
        for layer in other.directions:
            other_layer = other_coeff * other.directions[layer]
            if layer in directions:
                directions[layer] = directions[layer] + other_layer
            else:
                directions[layer] = other_layer
        return ControlVector(model_type=model_type, directions=directions)

    def __eq__(self, other: "ControlVector") -> bool:
        if self is other:
            return True

        if self.model_type != other.model_type:
            return False
        if self.directions.keys() != other.directions.keys():
            return False
        for k in self.directions.keys():
            if (self.directions[k] != other.directions[k]).any():
                return False
        return True

    def __add__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for +: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, 1)

    def __sub__(self, other: "ControlVector") -> "ControlVector":
        if not isinstance(other, ControlVector):
            raise TypeError(
                f"Unsupported operand type(s) for -: 'ControlVector' and '{type(other).__name__}'"
            )
        return self._helper_combine(other, -1)

    def __neg__(self) -> "ControlVector":
        directions: dict[str, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = -self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __mul__(self, other: int | float | np.number) -> "ControlVector":
        directions: dict[str, np.ndarray] = {}
        for layer in self.directions:
            directions[layer] = other * self.directions[layer]
        return ControlVector(model_type=self.model_type, directions=directions)

    def __rmul__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(other)

    def __truediv__(self, other: int | float | np.number) -> "ControlVector":
        return self.__mul__(1 / other)
    

def PCAWeighted(train, weights=None, n_components=1) -> torch.Tensor:
    """
    Weighted PCA via SVD. Returns first principal component direction.
    
    Args:
        train: [n, d] tensor
        weights: [n] tensor of sample weights (normalized to sum=1 internally)
    
    Returns:
        [d] tensor, first PC direction
    
    https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis
    """
    if weights is not None:
        weights_flat = weights.flatten().clone()
        # Normalize weights to sum to 1
        weights_norm = weights_flat / weights_flat.sum()
    else:
        weights_norm = torch.ones_like(train)[:, 0]
    
    # Weighted mean and centering
    weighted_mean = torch.sum(train * weights_norm.unsqueeze(-1), dim=0) / weights_norm.sum()
    train = train - weighted_mean
    
    # Apply sqrt of weights
    train_weighted_torch = train * torch.sqrt(weights_norm).unsqueeze(-1)
    
    # Torch SVD (full_matrices=False for efficiency)
    U, S, Vt = torch.linalg.svd(train_weighted_torch, full_matrices=False)
    
    # First PC direction
    direction = Vt[:n_components] # [k, d]
    
    return direction


def choose_sign_from_hiddens(direction: torch.Tensor, hiddens: torch.Tensor) -> torch.Tensor:
    """Flip direction so positives project higher than negatives on average."""
    projected_hiddens = project_onto_direction(hiddens, direction)
    
    # order is [positive, negative, positive, negative, ...]
    # Compare pos/neg pairs directly (more efficient than list comprehension)
    pos_proj = projected_hiddens[::2]  # [num_pairs]
    neg_proj = projected_hiddens[1::2]  # [num_pairs]
    
    # Fraction of pairs where positive projects higher than negative
    frac_correct = (pos_proj > neg_proj).float().mean()
    
    # Flip if majority of pairs are backwards (pos projects lower than neg)
    return direction if frac_correct >= 0.5 else -direction

def read_representations(
    act: dict[str, Tensor],
    logprobs: Tensor,
    grads: dict[str, Tensor | None] | None,
    feat_grad_norms: dict[str, Tensor | None] | None = None,
    method: typing.Literal["pca_diff", "pca_center", "umap", "pca_diff_weighted", "pca_center_weighted", "svd_gradient", "fisher_steer", "hvp_steer"] = "pca_diff",
    n_components: int = 1,
    use_scipy: bool = True,  # SciPy is often faster than PyTorch GPU SVD for moderate matrix sizes
) -> dict[str, np.ndarray]:
    
    hidden_layers= list(act.keys())
    
    # Check if gradient-based method is used without gradients
    needs_grads = any(m in method for m in ['gradient', 'fisher', 'hvp'])
    if needs_grads and grads is None:
        raise ValueError(f"Method '{method}' requires gradients, but grads=None. Use _collect_activations_grads.")

    # B. Compute directions
    directions: OrderedDict[str, torch.Tensor] = OrderedDict()
    for layer in tqdm.tqdm(hidden_layers, desc='pca'):
        h = act[layer]#.clone()

        
        if method == "svd_gradient":
            raise NotImplementedError("Gradient-based methods are currently disabled.")
        
        else: # PCA-based methods
            # run PCA on difference vectors between positive and negative examples
            train = h[::2] - h[1::2]
            if 'weighted' in method:
                # importance sampling weights from logprobs, higher prob = more online data
                # Normalize logprobs to [0, 2]: Higher prob = higher weight (focus on coherent samples)
                # For pairs, use difference or average; here, weight by pos logprob (more honest = higher weight) 
                pair_probs = logprobs.view(-1, 2).mean(1)  # Average logprob per pair [n_pairs]
                weights = torch.softmax(pair_probs, dim=0) * 2.0  # [0, 2] range
                weights = torch.clamp(weights, min=0.0)  # Non-negative
                # Reshape to match train (interleave for pos/neg, but since train is diff, use pair weights)
                # train will be defined below; just repeat to match pairwise diffs length
                # weights used with train = h[::2] - h[1::2], so shapes match (n_pairs,)
                components = PCAWeighted(train, weights=weights, n_components=n_components)
            else:
                # Doesn't have weighting but is otherwise better (faster, tracks signs, 
                pca = PCA(n_components=n_components)  # NEW: Use n_components
                components = pca.fit(train.numpy()).components_[:n_components]  # (K, d)
                components = torch.from_numpy(components)
            
            # NEW: Multi-comp sign flip (per component)
            for i in range(n_components):
                components[i] = choose_sign_from_hiddens(components[i], h)

            directions[layer] = components  # Keep as (k, d) - baukit_dir_add_hook handles it

    return directions

@torch.no_grad()
def _collect_activations_only(
    model,
    tokenizer,
    inputs: list[str],
    layers_to_edit: list[str],
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Lightweight collection of hidden states and logprobs without gradients.
    Used for PCA-based methods that don't need gradient information.
    
    Returns:
        hidden_states: {layer: [batch, hidden_dim]}
        completion_lprob: [batch]
    """
    assert batch_size % 2 == 0, "batch_size must be even for pos/neg pairs"
    batched_inputs = [inputs[p : p + batch_size] for p in range(0, len(inputs), batch_size)]
    
    hidden_states: dict[str, list[torch.Tensor]] = {layer: [] for layer in layers_to_edit}
    completion_lprob: list[torch.Tensor] = []
    
    model.eval()
    
    for bi, batch in enumerate(tqdm.tqdm(batched_inputs, desc=f"Getting act for modules={len(layers_to_edit)}")):
        encoded_batch = tokenizer(batch, padding=True, return_tensors="pt", padding_side="left").to(model.device)
        attention_mask = encoded_batch["attention_mask"]
        
        if bi % 10 == 0:
            torch.cuda.empty_cache()
        
        with torch.inference_mode():
            with TraceDict(
                model,
                layers=layers_to_edit,
                retain_output=True,
            ) as ret:
                outputs = model(**encoded_batch, output_hidden_states=True)
                
                # Compute logprobs
                lprobs = outputs.logits[:, :-1].log_softmax(-1)
                labels = encoded_batch["input_ids"][:, 1:, None]
                lprobs_for_inputs = torch.gather(input=lprobs, dim=-1, index=labels).squeeze(-1)
                
                label_mask = attention_mask[:, 1:]
                avg_logp_completion = (lprobs_for_inputs * label_mask).sum(-1) / label_mask.sum(-1)
                
                # Get last non-padded token index
                # attention_mask is [batch, seq_len] with 1s for real tokens, 0s for padding
                # For left padding: [0,0,0,1,1,1] -> last index is seq_len-1
                # For right padding: [1,1,1,0,0,0] -> need to find last 1
                # Flip and argmax finds first 1 from right, subtract from end
                last_valid_idx = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[-1]).argmax(dim=-1).cpu()
                
                # Collect activations from each layer
                for layer in layers_to_edit:
                    hs = ret[layer].output.detach().float().cpu()
                    last_hs = hs[range(len(last_valid_idx)), last_valid_idx]
                    hidden_states[layer].append(last_hs)
                
                completion_lprob.append(avg_logp_completion.detach().cpu().float())
        
        del outputs, lprobs, lprobs_for_inputs, ret
        torch.cuda.empty_cache()
    
    # Stack results
    hidden_states = {k: torch.vstack(v) for k, v in hidden_states.items()}
    completion_lprob = torch.cat(completion_lprob)
    
    return hidden_states, completion_lprob




def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    mag = torch.linalg.norm(direction)
    assert not torch.isinf(mag)
    return (H @ direction) / mag

