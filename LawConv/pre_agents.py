#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2024-07-15 20:00:07
@Author : su.zhu
@Desc   : 
"""

import json
import re
from typing import List
from collections import defaultdict
from LawConv.utils import LLMClient, convert_chat_format
import time
import copy


def constrct_intent_input(conv):
    intent_input = []
    for i, msg in enumerate(conv):
        if msg["role"] == "user":
            intent_input.append(f"用户: {msg['content']}")
        else:
            intent_input.append(f"客服: {msg['content']}")
    intent_input = "\n".join(intent_input)
    # return {"role": "user", "content": intent_input}
    return intent_input


def construct_summary_input(conv, output):
    aug_conv = copy.deepcopy(conv)

    summary_input = f"用户输入为：{aug_conv[-1]['content']}。"

    if output["emotion_output"]["emotion_type"]:
        summary_input += (
            f"需要安抚用户情绪,安抚话术为：{output['emotion_output']['reply']}。"
        )
    else:
        summary_input += f"不需要安抚用户情绪。"
    if output["is_recommend_output"]["is_recommend"]:
        summary_input += f"需要推荐服务,推荐话术为: 我们可以代笔写劳动仲裁申请书，点击 服务 即可进入到法律文书撰写页面。也可以点击 律师VIP 服务，由我们线下的同事来为您解决问题。"
    else:
        summary_input += f"不需要推荐服务。"

    if output["ask_question_output"]["is_ask"]:
        summary_input += (
            f"需要追问问题,追问问题为：{output['ask_question_output']['question']}。"
        )
    else:
        summary_input += f"不需要追问问题。"

    aug_conv[-1]["content"] = summary_input
    return aug_conv


class PreAgent:
    def __init__(*args, **kwargs):
        ## pass
        pass

    def run(self, conv: List[dict], type: str, llm_client: LLMClient):
        return {"result": {}}


class IsRecommendPreAgent(PreAgent):
    def __init__(self, min_turns: int, **kwargs) -> None:
        self.min_turns = min_turns
        with open(kwargs["prompt"], "r") as f:
            self.prompt = f.read()

    def run(self, conv: List[dict], type: type, llm_client: LLMClient):
        end_conv = False
        is_recommend = False
        if len(conv) // 2 == self.min_turns:
            is_recommend = True
            return {"is_recommend": is_recommend, "recommend_intent": ""}
        else:
            msgs = [
                {
                    "role": "system",
                    "content": f"{self.prompt}",
                }
            ]
            # intent_format_conv = constrct_intent_input(conv)
            msgs.extend(conv)
            print(msgs)
            intent_response = llm_client.response_json_output(msgs)
            intent_json_ouput = eval(intent_response)
            print(intent_json_ouput)
            intent = intent_json_ouput["intention"]
            reason = intent_json_ouput["reason"]

            return {"is_recommend": intent, "recommend_intent": reason}


class ModeSwitchPreAgent(PreAgent):
    def __init__(self, key_word) -> None:
        self.key_word = key_word

    def run(self, conv: List[dict], type: type, llm_client: LLMClient):
        # mode = 0 表示专业律师模式， mode = 1 表示主动咨询模式
        mode = 1
        for msg in conv:
            if self.key_word in msg["content"]:
                mode = 0
                break
        return {"mode": mode}


class EmotionPreAgent(PreAgent):
    def __init__(self, max_turns: int, **kwargs) -> None:
        self.max_turns = max_turns
        with open(kwargs["prompt"], "r") as f:
            self.prompt = f.read()

    def run(self, conv: List[dict], type: type, llm_client: LLMClient):
        emotion_type = False
        if len(conv) // 2 > self.max_turns:
            return {"emotion_type": emotion_type, "reply": ""}
        else:
            msgs = [
                {
                    "role": "system",
                    "content": f"{self.prompt}",
                }
            ]
            msgs.extend(conv)
            print(msgs)
            intent_response = llm_client.response_json_output(msgs)
            intent_json_ouput = eval(intent_response)
            print(intent_json_ouput)
            emotion_type = intent_json_ouput["emotion_type"]
            reply = intent_json_ouput["reply"]
            return {"emotion_type": emotion_type, "reply": reply}


class AskQuestionPreAgent(PreAgent):
    def __init__(self, max_turns: int, **kwargs) -> None:
        self.max_turns = max_turns
        with open(kwargs["prompt"], "r") as f:
            self.prompt = f.read()

    def run(self, conv: List[dict], type: type, llm_client: LLMClient):
        is_ask = False
        if len(conv) // 2 > self.max_turns:
            return {"is_ask": is_ask, "question": ""}
        else:
            msgs = [
                {
                    "role": "system",
                    "content": f"{self.prompt}",
                }
            ]

            msgs.extend(conv)
            print(msgs)
            intent_response = llm_client.response_json_output(msgs)
            intent_json_ouput = eval(intent_response)
            print(intent_json_ouput)
            question = intent_json_ouput["question"]
            is_ask = True
            return {"is_ask": is_ask, "question": question}


class ExtractTelephonePreAgent(PreAgent):
    def __init__(self, activated: bool) -> None:
        self.activated = activated

    def run(self, conv: List[dict], llm_client: LLMClient):
        if self.activated and len(conv) > 2:
            ask_tel_again = False
            ask_tel = self._detect_asktel_intent(conv[-1]["content"], llm_client)
            phone_list = []

            tel_pattern = "(?<!\d)(1\d{10})(?!\d)"
            phone_list = re.compile(tel_pattern).findall(conv[-2]["content"])

            wrong_tel_pattern = "(?<!\d)(1\d{9})(?!\d)"
            wrong_phone_list = re.compile(wrong_tel_pattern).findall(
                conv[-2]["content"]
            )
            wrong_tel_pattern = "(?<!\d)(1\d{11})(?!\d)"
            wrong_phone_list = (
                re.compile(wrong_tel_pattern).findall(conv[-2]["content"])
                + wrong_phone_list
            )
            wrong_phone_num = False

            if ask_tel and len(phone_list) == 0:
                ask_tel_again = True
                if len(wrong_phone_list) > 0:
                    wrong_phone_num = True

            ## 电话号码非法，需要用户重新输入
            if ask_tel_again:
                conv.append(
                    {"role": "assistant", "content": "电话号码有误，请重新输入"}
                )

        else:
            ask_tel_again = False
            wrong_phone_num = False

        return {
            "data": conv,
            "result": {
                "ask_tel_again": ask_tel_again,
                "wrong_phone_num": wrong_phone_num,
            },
        }

    def _detect_asktel_intent(self, assistant: str, llm_client: LLMClient):
        detect_asktel_prompt = """请检测下面客服回复中是否包含询问客户电话号码的意图，输出要求如下：
* 如果你认为客服回复中没有询问用户电话号码意图，请输出有；
* 如果你认为客服回复中有询问用户电话号码意图，请输出无；
* 请直接输出0或者1，不要输出任何其他解释内容；
客服回复如下：
{assistant}
客服回复中有无询问用户电话号码意图：
"""
        intent = llm_client.response_from_question(
            detect_asktel_prompt.format(assistant=assistant)
        )
        if "有" in intent:
            return True

        return False


class SeverCharacterPreAgent(PreAgent):

    def __init__(self, *args, **kwargs):

        self.law_types = ["work"]
        self.roles_prompt = defaultdict(dict)
        self.recommend = open(kwargs["recommend"], "r").read()
        self.ask = open(kwargs["ask"], "r").read()
        for law_type in self.law_types:
            for prompt_name, file_path in kwargs[law_type].items():
                self.roles_prompt[law_type][prompt_name] = open(file_path, "r").read()

    def run(self, conv: List[dict], law_type: str, llm_client: LLMClient, output: dict):
        intent_detect_prompt = self.roles_prompt[law_type]["intent_detect"]
        msgs = [
            {
                "role": "system",
                "content": f"{intent_detect_prompt}",
            }
        ]
        msgs.extend(conv)
        print(msgs)
        intent_detect_response = llm_client.response_json_output(msgs)

        intent_detect_json_ouput = eval(intent_detect_response)
        print(intent_detect_json_ouput)
        intent = intent_detect_json_ouput["intention"]
        reason = intent_detect_json_ouput["reason"]

        history_conv = convert_chat_format(conv)
        law_prompt = self.roles_prompt[law_type][intent]
        print(history_conv)
        print(law_prompt)
        print(law_prompt.format(input=history_conv))
        msgs = [
            {
                "role": "system",
                "content": f"{law_prompt.format(input=history_conv)}",
            }
        ]
        # msgs.extend(conv)
        print(msgs)
        law_output = llm_client.response_json_output(msgs)
        law_json_output = eval(law_output)
        print(law_json_output)
        intent = law_json_output["intention"]
        answer = law_json_output["answer"]
        if intent == 1:
            msgs = [
                {
                    "role": "system",
                    "content": f"{self.recommend}",
                }
            ]
            msgs.extend(conv)
            print(msgs)
            output = llm_client.response_from_list(msgs)
        else:

            msgs = [
                {
                    "role": "system",
                    "content": f"{self.ask.format(input=answer)}",
                }
            ]
            msgs.extend(conv)
            print(msgs)
            output = llm_client.response_from_list(msgs)
        print(output)
        conv.append({"role": "assistant", "content": f"{output}"})
        return {
            "intent_detect": intent_detect_response,
        }


class LawyerPreAgent(PreAgent):

    def __init__(self, prompt, *args, **kwargs):
        self.prompt = open(prompt, "r").read()

    def run(self, conv: List[dict], law_type: str, llm_client: LLMClient, output: dict):
        msgs = [
            {
                "role": "system",
                "content": f"{self.prompt}",
            }
        ]
        msgs.extend(conv)
        print(msgs)
        law_response = llm_client.response_from_list(msgs)
        conv.append({"role": "assistant", "content": f"{law_response}"})
        return {}
