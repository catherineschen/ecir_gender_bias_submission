import torch
from jaxtyping import Float
from typing import List, Optional, Tuple, Union, Dict
from transformer_lens.ActivationCache import ActivationCache
import transformer_lens.utils as utils
from abc import ABC, abstractmethod
from functools import partial
import itertools
from transformer_lens.hook_points import HookPoint
from mechir.util import get_hook_layer, normalize_outputs


class PatchedMixin(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = None

    @property
    def _patch_funcs(self):
        return {
            ##### activation patching functions ######
            "block_all": self.get_act_patch_block_every,
            "head_all": self.get_act_patch_attn_head_out_all_pos,
            "head_by_pos": self.get_act_patch_attn_head_by_pos,

            ###### path patching #######
            "path_patch": self.get_path_patch,
        }

    ################# ACTIVATION PATCHING FUNCTIONS ##################

    def _patch_residual_component(
        self,
        corrupted_component: Float[torch.Tensor, "batch pos d_model"],
        hook: HookPoint,
        pos: int,
        clean_cache: ActivationCache,
    ):
        """
        Patches a given sequence position in the residual stream, using the value
        from the clean cache.
        """
        corrupted_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
        return corrupted_component

    def _patch_head_vector(
        self,
        corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,  #: HookPoint,
        head_index: int,
        clean_cache: ActivationCache,
        **kwargs,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        """
        Patches the output of a given head (before it's added to the residual stream) at
        every sequence position, using the value from the clean cache.
        """

        corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][
            :, :, head_index
        ]
        return corrupted_head_vector

    def _patch_head_vector_by_pos_pattern(
        self,
        corrupted_activation: Float[torch.Tensor, "batch pos head_index pos_q pos_k"],
        hook,  #: HookPoint,
        pos,
        head_index: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        corrupted_activation[:, head_index, pos, :] = clean_cache[hook.name][
            :, head_index, pos, :
        ]
        return corrupted_activation

    def _patch_head_vector_by_pos(
        self,
        corrupted_activation: Float[torch.Tensor, "batch pos head_index d_head"],
        hook,  #: HookPoint,
        pos,
        head_index: int,
        clean_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        corrupted_activation[:, pos, head_index] = clean_cache[hook.name][
            :, pos, head_index
        ]
        return corrupted_activation

    def _get_act_patch_block_every(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        **kwargs,
    ) -> Float[torch.Tensor, "3 layer pos"]:
        """
        Returns an array of results of patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        """

        self._model.reset_hooks()
        _, seq_len = corrupted_tokens["input_ids"].size()
        # send tokens to device if not already there
        corrupted_tokens["input_ids"] = corrupted_tokens["input_ids"].to(self._device)
        corrupted_tokens["attention_mask"] = corrupted_tokens["attention_mask"].to(
            self._device
        )

        for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
            for layer in range(self._model.cfg.n_layers):
                for position in range(seq_len):
                    hook_fn = partial(
                        self._patch_residual_component,
                        pos=position,
                        clean_cache=clean_cache,
                    )
                    patched_outputs = self.run_with_hooks(
                        corrupted_tokens["input_ids"],
                        attention_mask=corrupted_tokens["attention_mask"],
                        fwd_hooks=[
                            ("_model." + utils.get_act_name(component, layer), hook_fn)
                        ],
                    )
                    yield (component_idx, layer, position), patched_outputs

    def _get_act_patch_attn_head_out_all_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        **kwargs,
    ) -> Float[torch.Tensor, "layer head"]:
        """
        Returns an array of results of patching at all positions for each head in each
        layer, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's embedding output.
        """

        self._model.reset_hooks()
        for layer in range(self._model.cfg.n_layers):
            for head in range(self._model.cfg.n_heads):
                hook_fn = partial(
                    self._patch_head_vector, head_index=head, clean_cache=clean_cache
                )
                patched_outputs = self.run_with_hooks(
                    corrupted_tokens["input_ids"],
                    attention_mask=corrupted_tokens["attention_mask"],
                    fwd_hooks=[("_model." + utils.get_act_name("z", layer), hook_fn)],
                )
                yield (layer, head), patched_outputs

    def _get_act_patch_attn_head_by_pos(
        self,
        corrupted_tokens: Float[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        layer_head_list,
        **kwargs,
    ) -> Float[torch.Tensor, "layer pos head"]:
        self._model.reset_hooks()
        _, seq_len = corrupted_tokens["input_ids"].size()

        for component_idx, component in enumerate(["z", "pattern"]):
            for i, layer_head in enumerate(layer_head_list):
                layer = layer_head[0]
                head = layer_head[1]
                for position in range(seq_len):
                    patch_fn = (
                        self._patch_head_vector_by_pos_pattern
                        if component == "pattern"
                        else self._patch_head_vector_by_pos
                    )
                    hook_fn = partial(
                        patch_fn, pos=position, head_index=head, clean_cache=clean_cache
                    )
                    patched_outputs = self.run_with_hooks(
                        corrupted_tokens["input_ids"],
                        attention_mask=corrupted_tokens["attention_mask"],
                        fwd_hooks=[
                            ("_model." + utils.get_act_name(component, layer), hook_fn)
                        ],
                    )
                    yield (component_idx, i, position), patched_outputs

    ################# PATH PATCHING FUNCTIONS ####################

    def _patch_or_freeze_head_vectors(
        self,
        orig_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint, 
        new_cache: ActivationCache,
        orig_cache: ActivationCache,
        head_to_patch=None, #: Tuple[int, int], 
        pos_to_patch: Union[int, List[int], None] = None,
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        '''
        This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them
        to their values in orig_cache), except for head_to_patch (if it's in this layer) which
        we patch with the value from new_cache.

        head_to_patch: tuple of (layer, head)
            we can use hook.layer() to check if the head to patch is in this layer
        '''
        # Setting using ..., otherwise changing orig_head_vector will edit cache value too
        orig_head_vector[...] = orig_cache[hook.name][...]

        if head_to_patch and head_to_patch[0] == get_hook_layer(hook.name):

            if pos_to_patch:
                if isinstance(pos_to_patch, int):
                    pos_to_patch = [pos_to_patch]
                
                for pos in pos_to_patch:
                    orig_head_vector[:, pos, head_to_patch[1]] = new_cache[hook.name][:, pos, head_to_patch[1]]
            else:
                orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
                
        return orig_head_vector

    def _patch_or_freeze_mlps(
        self,
        orig_mlp: Float[torch.Tensor, "batch pos d_mlp"],
        hook: HookPoint,
        new_cache: ActivationCache,
        orig_cache: ActivationCache,
        mlp_to_patch=None, # layer
    ):  
        orig_mlp[...] = orig_cache[hook.name][...]
        if mlp_to_patch is not None and mlp_to_patch == get_hook_layer(hook.name):
            orig_mlp[...] = new_cache[hook.name][...]
        return orig_mlp

    def _patch_head_input(
        self,
        orig_activation: Float[torch.Tensor, "batch pos head_idx d_head"],
        hook: HookPoint,
        patched_cache: ActivationCache,
        head_list, #: List[Tuple[int, int]],
    ) -> Float[torch.Tensor, "batch pos head_idx d_head"]:
        '''
        Function which can patch any combination of heads in layers,
        according to the heads in head_list.
        '''
        heads_to_patch = [head for layer, head in head_list if layer == get_hook_layer(hook.name)]
        orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
        return orig_activation


    def _patch_mlp_input(
        self,
        orig_mlp: Float[torch.Tensor, "batch pos d_mlp"],
        hook: HookPoint,
        patched_cache: ActivationCache,
    ) -> Float[torch.Tensor, "batch pos d_mlp"]:
        '''
        '''
        orig_mlp[...] = patched_cache[hook.name][...]
        return orig_mlp


    def _get_path_patch(
        self,
        model_type: str,
        max_layer: int,
        sender_type: str, # head or mlp
        # sender_nodes: Union[List[Tuple[int, int]], List[int], None],  # Heads (layer, head) or MLPs (layers), all if None
        receiver_node: Union[Tuple[int, int], List[int], None],  # Heads, MLPs, or None (for final resid_post)
        receiver_type: str,
        corrupted_tokens: Dict[str, torch.Tensor],
        clean_tokens: Dict[str, torch.Tensor],
        corrupt_cache: Optional[ActivationCache] = None,
        clean_cache: Optional[ActivationCache] = None,
        direct_includes_mlps: bool = False, # If False, freezes MLPs, otherwise recomputes them
        sender_attn_component: str = "z",
        receiver_attn_component: str = "v", 
        pos_to_patch: Union[int, List[int], None] = None,
        **kwargs,
    ) -> Union[Float[torch.Tensor, "layer head"], Float[torch.Tensor, "layer"]]:
        self._model.reset_hooks()

        # Get receiver hook name(s) and define filter(s)
        if receiver_type == "resid_post":
            receiver_input = "resid_post"
            receiver_layer = self._model.cfg.n_layers - 1
        else:
            if receiver_type == "head":
                # Handle the single head receiver node
                receiver_input = receiver_attn_component if sender_type == "head" else "z"
                receiver_layer, _ = receiver_node
            elif receiver_type == "mlp":
                # Handle the single MLP receiver node
                receiver_input = "pre"
                receiver_layer = receiver_node

        receiver_hook_name = "_model." + utils.get_act_name(receiver_input, receiver_layer)
        receiver_hook_name_filter = lambda name: name == receiver_hook_name

        # ========== Step 1 ==========
        # Get activations if not provided

        # Only cache components that we need for efficiency
        if direct_includes_mlps:
            name_filter = lambda name: name.endswith(sender_attn_component)
        else:
            name_filter = lambda name: name.endswith("mlp_out") or name.endswith(sender_attn_component)

        # Define common arguments
        common_args = {
            "names_filter": name_filter,
        }

        if clean_cache is None:
            _, clean_cache = self.run_with_cache(
                input_ids=clean_tokens["input_ids"],
                attention_mask=clean_tokens["attention_mask"],
                **common_args
            )
        if corrupt_cache is None:
            _, corrupt_cache = self.run_with_cache(
                input_ids=corrupted_tokens["input_ids"],
                attention_mask=corrupted_tokens["attention_mask"],
                **common_args
            )

        # ========== Step 2 ==========
        # Run on x_corrupt, with sender component patched from x_clean, every other component (head and/or mlp) frozen

        sender_iter = itertools.product(range(max_layer), range(self._model.cfg.n_heads)) if sender_type == "head" else range(max_layer)

        forward_pass_args = {
            # "input_ids": corrupted_tokens["input_ids"],
            "attention_mask": corrupted_tokens["attention_mask"],
            "names_filter": receiver_hook_name_filter,
        }

        for sender_node in list(sender_iter):

            # Define Attention (Head) Hook
            if sender_type == "head":
                sender_layer, sender_head = sender_node
                head_patch_tuple = (sender_layer, sender_head)
                attn_name_filter = lambda name: name.endswith(sender_attn_component)
            else: # sender_type == "mlp"
                head_patch_tuple = None # Freeze all
                attn_name_filter = lambda name: name.endswith("z") # Freeze all attention components

            # Define the partial function and apply the hook
            attn_hook_fn = partial(
                self._patch_or_freeze_head_vectors,
                new_cache=clean_cache,
                orig_cache=corrupt_cache,
                head_to_patch=head_patch_tuple,
                pos_to_patch=pos_to_patch,
            )
            self._model.add_hook(attn_name_filter, attn_hook_fn, level=1)

            # Define MLP Hook
            # MLP hooks are needed if sender is MLP, OR if MLPs must be frozen (direct_includes_mlps=False)
            if sender_type == "mlp" or not direct_includes_mlps:
                
                # Determine the layer to patch (only needed if sender_type is mlp)
                mlp_to_patch = sender_node if sender_type == "mlp" else None
                
                mlp_hook_fn = partial(
                    self._patch_or_freeze_mlps,
                    new_cache=clean_cache,
                    orig_cache=corrupt_cache,
                    mlp_to_patch=mlp_to_patch,
                )
                mlp_name_filter = lambda name: name.endswith("mlp_out")
                self._model.add_hook(mlp_name_filter, mlp_hook_fn, level=1)


            # Forward pass
            _, patched_cache = self.run_with_cache(
                corrupted_tokens["input_ids"], **forward_pass_args
            )
            assert set(patched_cache.keys()) == set([receiver_hook_name])


            # ========== Step 3 ==========
            # Run on x_corrupt, patching in the receiver node(s) from the previously cached value

            if receiver_type == "resid_post":
                # Post-processing is done directly on the cached activation
                final_activations = self._model.blocks[-1].ln2(patched_cache[receiver_hook_name]) #[:, 0, :]
                
                if model_type == "bi":
                    # Final output is the embedding (after LN and selecting CLS token)
                    patched_outputs = self._pooling(final_activations, corrupted_tokens["attention_mask"]) #.squeeze(0)
                    if self._normalize:
                        patched_outputs = normalize_outputs(patched_outputs)
                elif model_type == "cross":
                    # Final output requires the classifier
                    patched_outputs = self._model.classifier(final_activations)[:, 1] #.squeeze(0)
                    
            else: # receiver_type is "head" or "mlp" (Intermediate patch)
                
                # Define the single injection hook based on receiver_type
                if receiver_type == "head":
                    hook_fn = partial(
                        self._patch_head_input, 
                        patched_cache=patched_cache, 
                        head_list=[receiver_node]
                    )
                else: # receiver_type == "mlp"
                    hook_fn = partial(
                        self._patch_mlp_input, 
                        patched_cache=patched_cache, 
                    )

                patched_outputs = self.run_with_hooks(
                    corrupted_tokens["input_ids"],
                    attention_mask=corrupted_tokens["attention_mask"],
                    fwd_hooks=[(receiver_hook_name_filter, hook_fn)],
                )
            
            yield sender_node, patched_outputs


    ################### ABLATION FUNCTIONS ######################
    
    def _head_ablation_hook(
        self,
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint,
        heads_to_ablate: List,
        head_means: Dict[Tuple[int, int], torch.Tensor] = None,
        ablation_type: str = "zero",
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:

        hook_layer = get_hook_layer(hook.name)
        for layer, head in heads_to_ablate:
            if layer == hook_layer:
                if ablation_type == "zero":
                    value[:, :, head] = 0.
                elif ablation_type == "mean":
                    mean_vec = head_means[(layer, head)]
                    value[:, :, head] = mean_vec.to(value.device, value.dtype)
                else:
                    raise NotImplementedError(
                        f"ablation_type=[{ablation_type}] is currently not supported."
                    )

        return value
    
    def _mlp_ablation_hook(
        self,
        value: Float[torch.Tensor, "batch pos d_mlp"],
        hook: HookPoint,
        mlps_to_ablate: List[int],
        ablation_type: str = "zero",
    ) -> Float[torch.Tensor, "batch pos d_mlp"]:
        
        hook_layer = get_hook_layer(hook.name)
        if hook_layer in mlps_to_ablate:
            if ablation_type != "zero":
                raise NotImplementedError
            value[:] = 0.

        return value
    
    def _head_mask_hook(
        self,
        value: Float[torch.Tensor, "batch head_index q_pos k_pos"],
        hook: HookPoint, # expects attn_scores
        heads_to_mask: List,
        masks: torch.Tensor, 
    ):
        hook_layer = get_hook_layer(hook.name)
        for layer, head in heads_to_mask:
            if layer == hook_layer:
                # apply mask to head
                value[:,head,:,:] = value[:,head,:,:] + masks

        return
    
    def _get_attn_and_mlp_ablation(
        self,
        document_tokens: Dict[str, torch.Tensor],
        heads_to_ablate: List[Tuple[int,int]] = None,
        mlps_to_ablate: List[int] = None,
        ablation_type: str = "zero",
        head_means: Dict[Tuple[int, int], torch.Tensor] = None,
        **kwargs
    ):
        self._model.reset_hooks()

        fwd_hooks = []

        if heads_to_ablate:
            attn_hook_fn = partial(
                self._head_ablation_hook,
                heads_to_ablate=heads_to_ablate,
                ablation_type=ablation_type,
                head_means=head_means,
            )
            attn_name_filter = lambda name: name.endswith("z")
            fwd_hooks.append((attn_name_filter, attn_hook_fn))

        if mlps_to_ablate:
            mlp_hook_fn = partial(
                self._mlp_ablation_hook,
                mlps_to_ablate=mlps_to_ablate,
                ablation_type=ablation_type,
            )
            mlp_name_filter = lambda name: name.endswith("mlp_out")
            fwd_hooks.append((mlp_name_filter, mlp_hook_fn))

        ablated_outputs = self.run_with_hooks(
            document_tokens["input_ids"],
            attention_mask=document_tokens["attention_mask"],
            fwd_hooks=fwd_hooks,
            **kwargs,
        )
        return ablated_outputs
    
    def _get_mask_attn(
        self,
        document_tokens: Dict[str, torch.Tensor],
        heads_to_apply_mask: List[Tuple[int,int]],
        masks,
        act_name: str = "attn_scores",
        **kwargs
    ):
        
        self._model.reset_hooks()

        mask_hook_fn = partial(
            self._head_mask_hook,
            heads_to_mask=heads_to_apply_mask,
            masks=masks,
        )
        attn_name_filter = lambda name: name.endswith(act_name)

        masked_outputs = self.run_with_hooks(
            document_tokens["input_ids"],
            attention_mask=document_tokens["attention_mask"],
            fwd_hooks=[(attn_name_filter, mask_hook_fn)],
            **kwargs,
        )

        return masked_outputs


    @abstractmethod
    def forward(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _model_forward method"
        )

    @abstractmethod
    def run_with_cache(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _model_run_with_cache method"
        )

    @abstractmethod
    def run_with_hooks(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _model_run_with_hooks method"
        )

    @abstractmethod
    def get_act_patch_attn_head_out_all_pos(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the get_act_patch_attn_head_out_all_pos method"
        )

    @abstractmethod
    def get_act_patch_attn_head_by_pos(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the get_act_patch_attn_head_by_pos method"
        )

    @abstractmethod
    def get_act_patch_block_every(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the get_act_patch_block_every method"
        )
    
    @abstractmethod
    def get_path_patch(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _get_path_patch method"
        )
    
    @abstractmethod
    def ablate_attn_and_mlp(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _get_attn_and_mlp_ablation method"
        )
    
    @abstractmethod
    def mask_attn(*args, **kwargs):
        raise NotImplementedError(
            "Instantiate a subclass of PatchedMixin and implement the _get_mask_attn method"
        )
