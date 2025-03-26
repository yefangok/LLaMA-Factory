import re
import numpy as np

def fim_transform(ds:dict,fim_rate=0.5)->str:
    content = ds["content"]
    # 只有生成的ose数据才做fim
    if ds["data_type"] == "ose_code":
        end_idx = [m.end() for m in re.finditer(r"\bendfun(?:c|ction)?\b",content,flags=re.DOTALL)]
        end_idx = np.array(end_idx)

        np_rng = np.random.RandomState(seed=123)
        if len(end_idx) > 0 and np_rng.binomial(1, fim_rate):
            up_bound = np_rng.randint(low=0, high=end_idx[-1])
            down_bond = np_rng.choice(end_idx[end_idx>up_bound])
            prefix = content[:up_bound]
            middle = content[up_bound:down_bond]
            suffix = content[down_bond:]
            content = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"
        else:
            ...
    return content