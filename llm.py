import config

import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import LLM_MODEL, JUDGE_TOP_N, JUDGE_THRESHOLD, W_RETRIEVAL, W_JUDGE

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=dtype)
        _model.to(device)
        _model.eval()
    return _tokenizer, _model


def get_tokenizer_and_model():
    if _tokenizer is None or _model is None:
        return _load_model()
    return _tokenizer, _model


def _build_judge_prompt(query_context: dict, candidate: dict) -> str:
    query_label = query_context.get("predicted_label", "unknown")
    query_text = query_context.get("user_text", "") or ""
    
    candidate_label = candidate.get("caption", "unknown")
    candidate_conf = candidate.get("confidence", 0.0)
    candidate_score = candidate.get("score", 0.0)
    
    prompt = f"""You are an expert image retrieval judge.
Determine if the candidate image is relevant to the query.

Query image predicted label: {query_label}
Query text: {query_text if query_text else "N/A"}

Candidate filename: {candidate.get('filename', 'unknown')}
Candidate predicted label: {candidate_label}
Candidate confidence: {candidate_conf:.4f}
Candidate similarity score: {candidate_score:.4f}

Answer with Yes or No:"""
    
    return prompt


def _compute_judge_score(prompt: str) -> float:
    tokenizer, model = _load_model()
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False)
        logits = outputs.logits
    
    vocab_size = tokenizer.vocab_size
    
    def get_token_id(token_str: str):
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id is None or token_id == tokenizer.unk_token_id:
            token_id = tokenizer.convert_tokens_to_ids(token_str.strip())
        if token_id is None or token_id >= vocab_size:
            variants = [
                f" {token_str.strip().capitalize()}",
                f" {token_str.strip().lower()}",
                token_str.strip().capitalize(),
                token_str.strip().lower(),
            ]
            for variant in variants:
                token_id = tokenizer.convert_tokens_to_ids(variant)
                if token_id is not None and token_id < vocab_size:
                    return token_id
        return token_id
    
    yes_token_id = get_token_id("Yes")
    no_token_id = get_token_id("No")
    
    yes_logit = -100.0
    no_logit = -100.0
    
    if yes_token_id is not None and yes_token_id < vocab_size:
        yes_logit = logits[0, -1, yes_token_id].item()
    if no_token_id is not None and no_token_id < vocab_size:
        no_logit = logits[0, -1, no_token_id].item()
    
    scores_tensor = torch.tensor([yes_logit, no_logit])
    probs = F.softmax(scores_tensor, dim=0)
    judge_score = probs[0].item()
    
    return judge_score


def judge_candidates(query_context: dict, candidates: list, 
                     top_n: int = JUDGE_TOP_N, threshold: float = JUDGE_THRESHOLD) -> list:
    candidates_to_judge = candidates[:top_n] if len(candidates) > top_n else candidates
    
    tokenizer, model = _load_model()
    device = next(model.parameters()).device
    
    judged = []
    for prompt, candidate in zip(
        [_build_judge_prompt(query_context, c) for c in candidates_to_judge], 
        candidates_to_judge
    ):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=False)
            logits = outputs.logits
        
        vocab_size = tokenizer.vocab_size
        
        def get_token_id(token_str: str):
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None or token_id == tokenizer.unk_token_id:
                token_id = tokenizer.convert_tokens_to_ids(token_str.strip())
            if token_id is None or token_id >= vocab_size:
                variants = [
                    f" {token_str.strip().capitalize()}",
                    f" {token_str.strip().lower()}",
                    token_str.strip().capitalize(),
                    token_str.strip().lower(),
                ]
                for variant in variants:
                    token_id = tokenizer.convert_tokens_to_ids(variant)
                    if token_id is not None and token_id < vocab_size:
                        return token_id
            return token_id
        
        yes_token_id = get_token_id("Yes")
        no_token_id = get_token_id("No")
        
        yes_logit = -100.0
        no_logit = -100.0
        
        if yes_token_id is not None and yes_token_id < vocab_size:
            yes_logit = logits[0, -1, yes_token_id].item()
        if no_token_id is not None and no_token_id < vocab_size:
            no_logit = logits[0, -1, no_token_id].item()
        
        scores_tensor = torch.tensor([yes_logit, no_logit])
        probs = F.softmax(scores_tensor, dim=0)
        judge_score = probs[0].item()
        
        verdict = "relevant" if judge_score >= threshold else "rejected"
        
        result = dict(candidate)
        result["judge_score"] = judge_score
        result["verdict"] = verdict
        judged.append(result)
    
    judged.sort(key=lambda x: W_RETRIEVAL * x.get("score", 0) + W_JUDGE * x["judge_score"], reverse=True)
    judged = [r for r in judged if r["verdict"] != "rejected"]
    
    return judged


def generate_answer(question: str, results: list) -> str:
    if not results:
        return "not enough information"
    
    tokenizer, model = _load_model()
    device = next(model.parameters()).device
    
    context_lines = []
    for i, item in enumerate(results, 1):
        caption = item.get("caption", "unknown")
        score = item.get("score", 0.0)
        context_lines.append(f"{i}. {caption} (score: {score:.2f})")
    
    context = "\n".join(context_lines)
    
    system_prompt = (
        "You are a helpful vision analytics assistant. "
        "Answer using ONLY the retrieved image results provided. "
        "Cite result numbers when making claims. "
        "If the information is not enough to answer confidently, say 'not enough information'."
    )
    
    user_prompt = f"Retrieved results:\n{context}\n\nQuestion: {question}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
        )
    
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    answer = full_text[prompt_len:].strip()
    
    return answer if answer else full_text.strip()