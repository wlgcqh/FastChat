import json
import re
from typing import List
from LawConv.middleware import Pipeline
from LawConv.agents import PRE_AGENTS, POST_AGENTS
from LawConv.utils import LLMClient


class RefineConvPipeline(Pipeline):
    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)
        self.pre_cfg = self.cfg.pre_cfg
        self.post_cfg = self.cfg.post_cfg
        self.llm_cfg = self.cfg.llm
        self.gpt_cfg = self.cfg.gpt
        print(self.cfg)

        self.pre_agents = {}
        for key in self.pre_cfg:
            if key in PRE_AGENTS:
                self.pre_agents[key] = PRE_AGENTS[key](**self.pre_cfg[key])

        self.post_agents = {}
        for key in self.post_cfg:
            if key in POST_AGENTS:
                self.post_agents[key] = POST_AGENTS[key](**self.post_cfg[key])

        self.llm_client = LLMClient(**self.llm_cfg)
        # self.gpt_client = LLMClient(**self.gpt_cfg)

    def run(self, conv: List[dict]):
        ## limit_conv_turn
        limit_conv_turn_output = self.pre_agents["limit_conv_turn"].run(
            conv, self.llm_client
        )
        end_conv = limit_conv_turn_output["result"]["end_conv"]
        conv = limit_conv_turn_output["data"]
        if end_conv:
            return conv, {}

        # extract the telephone numbers in user's content
        extract_telephone_output = self.pre_agents["extract_telephone"].run(
            conv, self.llm_client
        )
        ask_tel_again = extract_telephone_output["result"]["ask_tel_again"]
        wrong_phone_num = extract_telephone_output["result"]["wrong_phone_num"]
        conv = extract_telephone_output["data"]
        if ask_tel_again:
            return conv, {}

        # judgment of character
        server_character_output = self.pre_agents["sever_character"].run(
            conv, self.llm_client
        )
        ai_character = server_character_output["result"]["ai_character"]
        conv = server_character_output["data"]

        # validate the trigger of similification

        simplify_response_output = self.post_agents["simplify_response"].run(
            conv, self.llm_client
        )
        simplification = simplify_response_output["result"]["simplified"]
        conv = simplify_response_output["data"]

        # count dedup times
        max_dedup_output = self.post_agents["max_dedup"].run(conv, self.llm_client)
        turn_dedup = max_dedup_output["result"]["turn_dedup"]
        conv = simplify_response_output["data"]

        conv_trigger_info = {
            "simplification": simplification,
            "ask_tel_again": ask_tel_again,
            "wrong_phone_num": wrong_phone_num,
            "turn_dedup": turn_dedup,
        }

        return conv, conv_trigger_info

    def simple_run(self, conv: List[dict]):

        output = self.pre_agents["sever_character"].simple_run(conv, self.llm_client)
        # gpt_response = self.pre_agents["sever_character"].simple_run(
        #     conv, self.gpt_client
        # )
        return output
        # return str({"意图": intent_response, "角色回答": llm_response})


if __name__ == "__main__":
    demo_convs = json.load(open("LawConv/test/law_conv_with_error.json", "r"))
    pipeline_config = "LawConv/config/pipeline.yaml"
    refine_conv_processor = RefineConvPipeline(pipeline_config)
    conv = refine_conv_processor.simple_run(demo_convs)
    # print(conv)
