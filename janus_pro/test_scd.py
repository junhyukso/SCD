# janus_pro_sjd_infer_v2.py
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import GenerationConfig
from transformers import LogitsProcessorList
import argparse
from utils import set_logger
from absl import logging

torch.set_grad_enabled(False)

def get_jacobi_param_dict(img_size=384, patch_size=16, image_token_size=None, vl_chat_processor=None, 
                          gen_head=None, model=None, coupling_mode='none', max_num_new_tokens=16, 
                          seed=None, iteration=0, gsd_size=1,):

    return dict(
        jacobi_loop_interval_l=1,
        jacobi_loop_interval_r=(img_size // patch_size) ** 2,
        max_num_new_tokens=max_num_new_tokens,
        guidance_scale=5,
        seed= seed,
        multi_token_init_scheme="random",
        do_cfg=True,
        image_top_k=1000,
        text_top_k=10,
        prefix_token_sampler_scheme="speculative_jacobi",
        image_token_size=image_token_size,
        gen_head=gen_head,
        patch_width=(img_size // patch_size),
        processor=vl_chat_processor,
        all_model=model,
        coupling_mode=coupling_mode,
        i=iteration,
        gsd_size=gsd_size,
    )
    
class MaxlenCriteria(StoppingCriteria):
    def __init__(self, max_seq_length, prompt_len):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.prompt_len = prompt_len
    
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[-1] >= self.max_seq_length
    
class JanusSJDSolver:
    def __init__(self, model, processor, image_top_k=2000, text_top_k=10): #model: MultiModalityCausalLM
        self.model = model
        self.processor = processor
        self.image_top_k = image_top_k
        self.text_top_k = text_top_k
        
    def _init_model_kwargs(self, prefill_num, mask=None, device='cuda', parallel_size=1, *args, **kwargs):
        model_kwargs = dict(
            use_cache = True,
            attention_mask = mask[0, -1:, :prefill_num] if mask is not None else torch.ones((parallel_size, prefill_num), device=device),
            past_key_values = transformers.DynamicCache(),
            cache_position=prefill_num,
        )
        return model_kwargs

    def create_logits_processor(self):
        start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_start_token)
        end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_end_token)
        
        from scheduler.logit_processor_3dim import MultiTokensVLLogitsProcessor, MultiTokensInterleavedTopKLogitsWarper
        processors = LogitsProcessorList()
        
        candidate_processor = MultiTokensVLLogitsProcessor(
            image_start_token_id=start_token_id,
            image_end_token_id=end_token_id,
            patch_size=16,
            voc_size=16384,
            device=self.model.device,
        )
        
        topk_processor = MultiTokensInterleavedTopKLogitsWarper(
            image_top_k=self.image_top_k,
            text_top_k=self.text_top_k,
            image_start_token_id=start_token_id,
            image_end_token_id=end_token_id,
        )
        
        processors.append(candidate_processor)
        processors.append(topk_processor)
        
        return processors
        
    def generate(self, input_ids, max_new_tokens, prompt_len, parallel_size, **kwargs):
        stopping_criteria = StoppingCriteriaList([
            MaxlenCriteria(max_new_tokens, prompt_len)
        ])
        
        prompt_len = input_ids.shape[-1]
        max_seq_length = prompt_len + max_new_tokens
        
        model_kwargs = self._init_model_kwargs(
            prefill_num=prompt_len,
            mask=None,
            device=self.model.device,
            parallel_size=parallel_size,
        )
        
        temperature = 1.0
        synced_gpus = False
        
        generation_config = GenerationConfig(
            max_new_tokens=max_seq_length,
            max_length=max_seq_length,
            temperature=temperature,
            top_k=None,
            do_sample=True,
            _pad_token_tensor=None,
            return_dict_in_generate=False,
        )
        
        logits_processor = self.create_logits_processor()
        
        outputs = self.model._sample(
            input_ids=input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=None,
            logits_warper=None,
            **model_kwargs,
        )
        return outputs if isinstance(outputs, torch.Tensor) else outputs.sequences


def generate_janus_sjd(
    args,
    model_path="deepseek-ai/Janus-Pro-7B",
    prompt_input="A British Shorthair cat lounging on a wooden windowsill with city skyline bokeh, analog photo, photoart_style, realistic, film grain, 4k/8k, exquisite fur detail, 50-mm-lens, sharp-focus, f/1.8, ISO 400, shutter-speed 1/125, soft window light, small catchlight, perfect symmetry, elegant composition.",
    parallel_size=1,
    image_token_num_per_image=576,
    img_size=384,
    patch_size=16,
):
    print(f"coupling mode: {args.coupling_mode}")
    device = "cuda"

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).to(torch.bfloat16).to(device).eval()
    
    prompt = f"Generate an image of {img_size}x{img_size} according to the following prompt:\n" + prompt_input
    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    text_with_image_tag = sft_format + vl_chat_processor.image_start_tag
    
    input_ids = tokenizer.encode(text_with_image_tag, return_tensors="pt").to(device)
    input_ids = input_ids.expand(parallel_size, -1).contiguous()
    prompt_len = input_ids.shape[-1]

    language_model = model.language_model

    from scheduler.jacobi_iteration_lumina_mgpt import renew_sampler
    
    image_token_size = model.config.gen_vision_config.params.image_token_size
    coupling_mode = args.coupling_mode
    max_num_new_tokens = args.max_num_new_tokens
    seed = args.seed
    
    save_path = f"generated_samples/{coupling_mode}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    set_logger(log_level='info', fname=os.path.join(save_path, 'gen_img_latency.log'))
    
    logging.info(f"prompt: {prompt_input}")
    
    jacobi_param_dict = get_jacobi_param_dict(img_size, patch_size, image_token_size, vl_chat_processor, 
                                              model.gen_head, model, coupling_mode, max_num_new_tokens, 
                                              seed, args.i,)
    language_model.__class__ = renew_sampler(language_model.__class__)
    language_model._init_new_params(**jacobi_param_dict)

    solver = JanusSJDSolver(language_model, vl_chat_processor, image_top_k=jacobi_param_dict["image_top_k"])

    outputs = solver.generate(
        input_ids=input_ids,
        max_new_tokens=image_token_num_per_image,
        prompt_len = prompt_len,
        use_cache=True,
        parallel_size=parallel_size,
    )
    img_token_ids = outputs[:, -image_token_num_per_image:]
    print(img_token_ids.shape)

    dec = model.gen_vision_model.decode_code(
        img_token_ids.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    
    for i in range(parallel_size):
        Image.fromarray(dec[i]).save(f"{save_path}/{max_num_new_tokens}_{seed}_img_{prompt_input[:20]}.jpg")
    print("Done. Images saved to generated_samples/")
    return outputs, img_token_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupling_mode", type=str, default="sjd", choices=["sjd", "maximal", "gumbel"])
    parser.add_argument("--max_num_new_tokens", type=int, default=16, choices=[1, 16, 32, 64, 128])
    parser.add_argument("--i", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5,)
    parser.add_argument("--prompt", type=str, default="A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair")
    args = parser.parse_args()
    
    generate_janus_sjd(args, prompt_input=args.prompt)
