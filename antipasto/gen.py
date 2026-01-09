import torch
from antipasto.peft_utils.adapter_scaling import ScaleAdapter
from textwrap import wrap, indent
from antipasto.eval import gen_with_choices, get_choice_ids
from antipasto.config import PROMPT, PERSONAS
from torch.nn import functional as F
import matplotlib.pyplot as plt

@torch.no_grad()
def gen(model, tokenizer, prompt, coeffs=[-200, -20, -2, -1, 0, 1, 2, 20, 200, None], max_new_tokens=128):
     model.eval()
     # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
     inputs = tokenizer.apply_chat_template([
          {'role': 'system', 'content': ""},
          {"role": "user", "content": prompt}], return_tensors="pt", return_dict=True).to(model.device)

     question = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
     N = inputs["input_ids"].shape[1]
     print('='*40+'\n'+f"Question: {question}"+'\n'+'='*40)
     for coeff in coeffs:
          with ScaleAdapter(model, coeff=coeff):
               with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)  # reduce control jank
                    s = tokenizer.decode(outputs[0, N:], skip_special_tokens=False)

                    s = "\n".join(wrap(s, width=120))
                    print(f"coeff={coeff}:\n{s}")
                    print('-'*40)
                    yield coeff, s

@torch.no_grad()
def gen_with_ans(model, tokenizer, prompt, coeffs=[-200, -20, -2, -1, 0, 1, 2, 20, 200, None], max_new_tokens=128, plot=False):
     prompt = prompt
     model.eval()
     # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
     inputs = tokenizer.apply_chat_template([
          {'role': 'system', 'content': ""},
          {"role": "user", "content": prompt}], return_tensors="pt", return_dict=True, return_attention_mask=True).to(model.device)

     question = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
     N = inputs["input_ids"].shape[1]
     print('='*40+'\n'+f"Question: {question}"+'\n'+'='*40)

     res = []
     choice_ids = get_choice_ids(tokenizer)
     for coeff in coeffs:
          with ScaleAdapter(model, coeff=coeff):
               with torch.autocast("cuda", dtype=torch.bfloat16):
                    # outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)  # reduce control jank
                    out, seq_nll, logp_choices, logratios = gen_with_choices(model, 
                    tokenizer, inputs['input_ids'], inputs['attention_mask'], choice_ids, continue_n_tokens=max_new_tokens)
                    outputs = out.sequences
                    s = tokenizer.decode(outputs[0, N:], skip_special_tokens=False)
                    s = "\n".join(wrap(s, width=120))
                    p = torch.sigmoid(logratios[0]).item()
                    print(f"coeff={coeff}, ans={p:.2%} yes, [logratio={logratios[0]:.4f}]:\n{s}")
                    print('-'*40)
                    res.append(dict(coeff=coeff, text=s, prob_yes=p, logratio=logratios[0].item()))
                    yield coeff, logratios, logratios

     if plot:        
        # coeffs_ = [r['coeff'] if r['coeff'] is not None else 0 for r in res]
        # lprobs_ = [r['logratio'] for r in res]
        # plt.figure(figsize=(6,4))
        # plt.plot(coeffs_, lprobs_, marker='o')
        # plt.xscale('symlog', linthresh=1)
        # plt.yscale('linear')
        # plt.xlabel('Adapter Scaling Coefficient')
        # plt.ylabel('Log Ratio')
        # plt.title('Effect of Adapter Scaling on Log Ratio')
        # plt.grid(True)
        # plt.show()

        coeffs_ = [r['coeff'] if r['coeff'] is not None else 0 for r in res]
        lprobs_ = [r['prob_yes'] for r in res]
        plt.figure(figsize=(6,4))
        plt.plot(coeffs_, lprobs_, marker='o')
        plt.xscale('symlog', linthresh=1)
        plt.yscale('log')
        plt.xlabel('Adapter Scaling Coefficient')
        plt.ylabel('p(yes)')
        plt.title('Effect of Adapter Scaling on p(yes)')
        plt.grid(True)
        plt.show()