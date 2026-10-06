from typing import Callable, Dict, Tuple, Union, List
import logging
import os
import torch
from jaxtyping import Float
from transformers import AutoModel, AutoTokenizer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookedRootModule, HookPoint
import transformer_lens.utils as utils
from mechir.modelling.patched import PatchedMixin
from mechir.modelling.sae import SAEMixin
from mechir.modelling.hooked.loading_from_pretrained import get_official_model_name
from mechir.util import batched_dot_product, batched_cos_sim, linear_rank_function, sort_docs_by_score_batch, normalize_outputs, create_attn_mask
from mechir.modelling.architectures import HookedEncoder

logger = logging.getLogger(__name__)


def cls_pooling(token_embeddings, attention_mask=None):
    return token_embeddings[:, 0, :]

def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

POOLING = {
    "cls": cls_pooling,
    "mean": mean_pooling,
}


class Dot(HookedRootModule, PatchedMixin, SAEMixin):
    def __init__(
        self,
        model_name_or_path: str,
        pooling_type: str = "cls",
        sim_func_type: str = "dot",
        tokenizer=None,
        special_token: str = "a",
        return_cache: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = (
            AutoTokenizer.from_pretrained(model_name_or_path)
            if tokenizer is None
            else tokenizer
        )
        self.special_token = special_token
        torch.set_grad_enabled(False)
        self._device = utils.get_device()
        self.model_name_or_path = get_official_model_name(model_name_or_path)
        self.__hf_model = (
            AutoModel.from_pretrained(model_name_or_path).eval().to(self._device)
        )
        self._model = HookedEncoder.from_pretrained(
            self.model_name_or_path, device=self._device, hf_model=self.__hf_model
        )

        self._pooling_type = pooling_type
        self._pooling = POOLING[pooling_type]
        self._return_cache = return_cache
        self._score_func = batched_dot_product if sim_func_type == "dot" else batched_cos_sim
        self._normalize = False if sim_func_type == "dot" else True

        self.setup()

    # def setup(self):
    #     super().setup()

    #     # Rename hook points
    #     for name, module in self.named_modules():
    #         if isinstance(module, HookPoint):
    #             if module.name.startswith("_model."):
    #                 module.name = module.name[len("_model."):]

    #     # Rebuild hook dict
    #     self.hook_dict = {
    #         module.name: module
    #         for module in self.modules()
    #         if isinstance(module, HookPoint)
    #     }

    def forward(
        self,
        input_ids: Float[torch.Tensor, "batch seq"],
        attention_mask: Float[torch.Tensor, "batch seq"],
        token_type_ids: Float[torch.Tensor, "batch seq"] = None,
    ):
        model_output = self._model(
            input=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_type="embeddings",
        )
        pooling = self._pooling(model_output, attention_mask)
        if self._normalize:
            return normalize_outputs(pooling)
        return pooling


    def get_act_patch_block_every(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        reps_q: Float[torch.Tensor, "batch pos model"],
        patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], float],
        scores: Float[torch.Tensor, "batch pos"],
        scores_p: Float[torch.Tensor, "batch pos"],
        **kwargs,
    ) -> Float[torch.Tensor, "3 layer pos"]:
        """
        Returns an array of results of patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        """

        batch_size, seq_len = corrupted_tokens["input_ids"].size()
        results = torch.zeros(
            batch_size,
            3,
            self._model.cfg.n_layers,
            seq_len,
            device=self._device,
            dtype=torch.float32,
        )

        for index, output in self._get_act_patch_block_every(
            corrupted_tokens=corrupted_tokens, clean_cache=clean_cache
        ):
            output = self._score_func(reps_q, output)
            results[(slice(None),) + index] = patching_metric(output, scores, scores_p) #.mean()

        return results

    def get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        reps_q: Float[torch.Tensor, "batch pos model"],
        patching_metric: Callable,
        scores: Float[torch.Tensor, "batch pos"],
        scores_p: Float[torch.Tensor, "batch pos"],
        **kwargs,
    ) -> Float[torch.Tensor, "layer head"]:
        """
        Returns an array of results of patching at all positions for each head in each
        layer, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's embedding output.
        """
        batch_size, _ = corrupted_tokens["input_ids"].size()
        results = torch.zeros(
            batch_size,
            self._model.cfg.n_layers,
            self._model.cfg.n_heads,
            device=self._device,
            dtype=torch.float32,
        )

        for index, output in self._get_act_patch_attn_head_out_all_pos(
            corrupted_tokens=corrupted_tokens, clean_cache=clean_cache
        ):
            output = self._score_func(reps_q, output)
            results[(slice(None),) + index] = patching_metric(output, scores, scores_p) #.mean()

        return results

    def get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        reps_q: Float[torch.Tensor, "batch pos model"],
        clean_cache: ActivationCache,
        layer_head_list,
        patching_metric: Callable,
        scores: Float[torch.Tensor, "batch pos"],
        scores_p: Float[torch.Tensor, "batch pos"],
        **kwargs,
    ) -> Float[torch.Tensor, "layer pos head"]:

        _, seq_len = corrupted_tokens["input_ids"].size()
        results = torch.zeros(
            2, len(layer_head_list), seq_len, device=self._device, dtype=torch.float32
        )

        for index, output in self._get_act_patch_attn_head_by_pos(
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            layer_head_list=layer_head_list,
        ):
            output = self._score_func(reps_q, output)
            results[index] = patching_metric(output, scores, scores_p).mean()

        return results
    
    def get_path_patch(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_tokens: Float[torch.Tensor, "batch pos"],
        corrupted_scores: Float[torch.Tensor, "batch pos"],
        clean_scores: Float[torch.Tensor, "batch pos"],
        reps_q: Float[torch.Tensor, "batch pos model"],
        patching_metric: Callable,
        sender_type: str,
        receiver_type: str,
        receiver_node: Union[Tuple[int, int], List[int], None],
        corrupted_cache: ActivationCache = None,
        clean_cache: ActivationCache = None,
        direct_includes_mlps: bool = False,
        sender_attn_component: str = "z",
        receiver_attn_component: str = "v", 
        **kwargs,
    ) -> Float[torch.Tensor, "layer head"]:
        
        if receiver_type == "resid_post":
            max_layer = self._model.cfg.n_layers
        elif receiver_type == "head":
            max_layer = receiver_node[0]
        elif receiver_type == "mlp":
            max_layer = receiver_node[0] + 1
        
        batch_size = clean_tokens["input_ids"].shape[0]
        if sender_type == "head": 
            results = torch.zeros(
                batch_size,
                max_layer,
                self._model.cfg.n_heads,
                device=self._device,
                dtype=torch.float32,
            )
        else: # mlp
            results = torch.zeros(
                batch_size,
                max_layer, 
                device=self._device, 
                dtype=torch.float32
            )

        for sender_node, output in self._get_path_patch(
            model_type="bi",
            max_layer=max_layer,
            sender_type=sender_type,
            receiver_node=receiver_node,
            receiver_type=receiver_type,
            corrupted_tokens=corrupted_tokens,
            clean_tokens=clean_tokens,
            corrupt_cache=corrupted_cache,
            clean_cache=clean_cache,
            direct_includes_mlps=direct_includes_mlps,
            sender_attn_component=sender_attn_component,
            receiver_attn_component=receiver_attn_component,
        ):
            output = self._score_func(reps_q, output)

            if sender_type == "head":
                layer, head = sender_node
                results[:, layer, head] = patching_metric(output, corrupted_scores, clean_scores) #.mean()
            elif sender_type == "mlp":
                layer = sender_node
                results[:, layer] = patching_metric(output, corrupted_scores, clean_scores) #.mean()
        
        return results

    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        cache_as_dict: bool = False,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule. This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            if not cache_as_dict:
                cache = ActivationCache(
                    cache_dict, self, has_batch_dim=not remove_batch_dim
                )
            return out, cache
        else:
            return out, None

    def score(
        self,
        queries: dict,
        documents: dict,
        reps_q=None,
        cache=False,
        cache_as_dict=False,
    ):
        if reps_q is None:
            reps_q = self.forward(queries["input_ids"], queries["attention_mask"])
        if cache:
            reps_d, cache_d = self.run_with_cache(
                documents["input_ids"],
                documents["attention_mask"],
                cache_as_dict=cache_as_dict,
            )
            return self._score_func(reps_q, reps_d), reps_q, reps_d, cache_d
        reps_d = self.forward(documents["input_ids"], documents["attention_mask"])
        return self._score_func(reps_q, reps_d), reps_q, reps_d, None

    def activation_patch(
        self,
        queries: dict,
        documents: dict, # Lower scoring
        documents_p: dict, # Higher scoring
        patch_type: str = "block_all",
        layer_head_list: list = [],
        patching_metric: Callable = linear_rank_function,
    ):
        assert (
            patch_type in self._patch_funcs
        ), f"Patch type {patch_type} not recognized. Choose from {self._patch_funcs.keys()}"
        scores, reps_q, _, _ = self.score(queries, documents)
        scores_p, _, _, cache_d = self.score(
            queries, documents_p, cache=True, reps_q=reps_q
        )

        # Regroup documents based on score differences
        higher_score_docs, lower_score_docs, higher_scores, lower_scores = sort_docs_by_score_batch(
            documents, documents_p, scores, scores_p
        )

        _, higher_score_cache = self.run_with_cache(
            higher_score_docs["input_ids"],
            higher_score_docs["attention_mask"],
        )

        patching_kwargs = {
            "corrupted_tokens": lower_score_docs, #documents,
            "clean_cache": higher_score_cache, #cache_d,
            "reps_q": reps_q,
            "patching_metric": patching_metric,
            "layer_head_list": layer_head_list,
            "scores": lower_scores, #scores,
            "scores_p": higher_scores, #scores_p,
        }

        patched_output = self._patch_funcs[patch_type](**patching_kwargs)
        if self._return_cache:
            return patched_output, cache_d
        return patched_output, None  # PatchingOutput(output, scores, scores_p)
    
    def path_patch(
        self,
        queries: dict,
        documents: dict, # Lower scoring
        documents_p: dict, # Higher scoring
        sender_type: str = "head",
        receiver_type: str = "resid_post",
        receiver_node: Union[Tuple[int, int], List[int], None] = None,
        recompute_mlps: bool = False,
        sender_attn_component: str = "z",
        receiver_attn_component: str = "v",
        patching_metric: Callable = linear_rank_function,
    ):
        allowed_types = {
            "sender_type": {"head", "mlp"},
            "receiver_type": {"resid_post", "head", "mlp"},
            "sender_attn_component": {"z", "q", "k", "v"},
            "receiver_attn_component": {"z", "q", "k", "v"},
        }
        input_vars = {
            "sender_type": sender_type,
            "receiver_type": receiver_type,
            "sender_attn_component": sender_attn_component,
            "receiver_attn_component": receiver_attn_component,
        }
        
        for var_name, current_value in input_vars.items():
            allowed_set = allowed_types.get(var_name)
            assert (
                current_value in allowed_set
            ), f"Invalid value for {var_name}: '{current_value}'. Must be one of {allowed_set}."


        scores, reps_q, _, cache_d = self.score(
            queries, documents, cache=True
        )

        scores_p, _, _, cache_p = self.score(
            queries, documents_p, cache=True, reps_q=reps_q
        )  

        # Regroup documents based on score differences
        higher_score_docs, lower_score_docs, higher_scores, lower_scores = sort_docs_by_score_batch(
            documents, documents_p, scores, scores_p
        )

        patching_kwargs = {
            "corrupted_tokens": lower_score_docs, # Lower scoring
            "clean_tokens": higher_score_docs, # Higher scoring
            # "corrupted_cache": cache_d,
            # "clean_cache": cache_p,
            "corrupted_scores": lower_scores,
            "clean_scores": higher_scores,
            "reps_q": reps_q,
            "patching_metric": patching_metric,
            "sender_type": sender_type,
            "receiver_type": receiver_type,
            "receiver_node": receiver_node,
            "direct_includes_mlps": recompute_mlps,
            "sender_attn_component": sender_attn_component,
            "receiver_attn_component": receiver_attn_component,
        }

        patched_output = self._patch_funcs["path_patch"](**patching_kwargs)

        return patched_output, None # have same format as activation_patch
    

    def ablate_attn_and_mlp(
        self,
        queries: dict,
        documents: dict,
        documents_p: dict,
        heads_to_ablate: List[Tuple[int,int]] = None,
        mlps_to_ablate: List[int] = None,
        ablation_type: str = "zero",
        head_means: Dict[Tuple[int, int], torch.Tensor] = None,
    ):
        ablated_reps = self._get_attn_and_mlp_ablation(documents, heads_to_ablate, mlps_to_ablate, ablation_type, head_means)
        ablated_reps_p = self._get_attn_and_mlp_ablation(documents_p, heads_to_ablate, mlps_to_ablate, ablation_type, head_means)

        reps_q = self.forward(queries["input_ids"], queries["attention_mask"])
        ablated_scores = self._score_func(reps_q, ablated_reps)
        ablated_scores_p = self._score_func(reps_q, ablated_reps_p)
        
        return ablated_scores, ablated_scores_p, ablated_reps, ablated_reps_p
    

    def mask_attn(
        self,
        queries: dict,
        documents: dict,
        heads_to_mask: List[Tuple[int,int]] = None,
        mask_type: str = "all_to_sep",
    ):
        masks = create_attn_mask(documents, mask_type)
        masked_reps = self._get_mask_attn(documents, heads_to_mask, masks)

        reps = self.forward(documents["input_ids"], documents["attention_mask"])
        reps_q = self.forward(queries["input_ids"], queries["attention_mask"])

        og_scores = self._score_func(reps_q, reps)
        masked_scores = self._score_func(reps_q, masked_reps)
        
        return og_scores, masked_scores
