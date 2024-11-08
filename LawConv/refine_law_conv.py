import json
import re
from typing import List
from LawConv.middleware import Pipeline
from LawConv.agents import PRE_AGENTS, POST_AGENTS
from LawConv.utils import LLMClient
from LawConv.farui import call_with_messages


class RefineConvPipeline(Pipeline):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.pre_cfg = self.cfg.pre_cfg
        self.post_cfg = self.cfg.post_cfg
        self.llm_cfg = self.cfg.llm
        self.gpt_cfg = self.cfg.gpt

        self.pre_agents = {}
        for key in self.pre_cfg:
            if key in PRE_AGENTS:
                self.pre_agents[key] = PRE_AGENTS[key](**self.pre_cfg[key])

        self.post_agents = {}
        for key in self.post_cfg:
            if key in POST_AGENTS:
                self.post_agents[key] = POST_AGENTS[key](**self.post_cfg[key])

        self.llm_client = LLMClient(**self.gpt_cfg)
        # self.gpt_client = LLMClient(**self.gpt_cfg)

    def run_test(self, conv: List[dict], type: str):

        mode_output = self.pre_agents["mode_switch"].run(conv, type, self.llm_client)
        if mode_output["mode"]:
            # 主动咨询模式
            server_character_output = self.pre_agents["sever_character"].run(
                conv, type, self.llm_client, None
            )
            conv_trigger_info = server_character_output
        else:
            # 专业模式
            lawyer_output = self.pre_agents["lawyer"].run(
                conv, type, self.llm_client, None
            )
            conv_trigger_info = lawyer_output

        print(conv_trigger_info)
        return conv, conv_trigger_info

    def run(self, conv: List[dict], type: str):

        conv = call_with_messages(conv)
        conv_trigger_info = {}
        return conv, conv_trigger_info


if __name__ == "__main__":
    demo_convs = json.load(open("LawConv/test/law_conv_with_error.json", "r"))
    pipeline_config = "LawConv/config/pipeline.yaml"
    refine_conv_processor = RefineConvPipeline(pipeline_config)
    conv, conv_trigger_info = refine_conv_processor.run(demo_convs, "work")
    print(conv, conv_trigger_info)
