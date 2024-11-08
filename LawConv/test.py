import pandas as pd
from LawConv.utils import LLMClient
from omegaconf import OmegaConf

if __name__ == "__main__":
    data = pd.read_excel(
        "LawConv/test/意图识别测试样例.xlsx", sheet_name="模糊类别案例"
    )
    intent_detect_prompt = open("LawConv/config/lawyer/work/intent_detect.md").read()
    config_file = "LawConv/config/pipeline.yaml"
    cfg = OmegaConf.load(config_file)
    print(cfg.gpt)
    llm_client = LLMClient(**cfg.gpt)
    for index, line in data.iterrows():
        question, label = line
        msgs = [
            {
                "role": "system",
                "content": f"{intent_detect_prompt}",
            },
            {"role": "user", "content": f"{question}"},
        ]

        print(msgs)
        intent_detect_response = llm_client.response_json_output(msgs)
        print(intent_detect_response, label)
