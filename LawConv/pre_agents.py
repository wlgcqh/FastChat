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
from LawConv.utils import LLMClient


class PreAgent:
    def __init__(*args, **kwargs):
        ## pass
        pass

    def run(self, conv: List[dict], llm_client: LLMClient):
        return {"data": conv, "result": {}}


class LimitConvTurnPreAgent(PreAgent):
    def __init__(self, max_num: int) -> None:
        self.max_num = max_num

    def run(self, conv: List[dict], llm_client: LLMClient):
        end_conv = False
        if len(conv) // 2 > self.max_num:
            # 对话轮次大于5轮，就调用一个新的prompt, 让大模型去 流畅地转移到 结束话术。
            system_prompt = """# 角色
你是一名专业且耐心的客服。
## 技能  
当前与用户的对话轮次已经足够，需要流畅的结束对话让用户购买我们的法律咨询服务
"""
            msgs = [{"role": "system", "content": f"{system_prompt}"}]
            msgs.extend(conv)
            response = llm_client.response_from_list(msgs)
            conv.append({"role": "assistant", "content": f"{response}"})
            end_conv = True

        return {"data": conv, "result": {"end_conv": end_conv}}


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
        with open(kwargs["prompt"], "r") as f:
            self.role_prompt = f.read()
        self.ai_characters = ["lawyer", "psychologist", "middle_role", "clerk"]
        self.roles_prompt = {}
        for ai_character in self.ai_characters:
            self.roles_prompt[ai_character] = open(kwargs[ai_character], "r").read()

    def run(self, conv: List[dict], llm_client: LLMClient):
        intent_response = llm_client.response_from_question(
            self.role_prompt.format(conv=conv[-1]["content"])
        )
        try:
            index = int(re.findall(r"\[return\] (.+?)", intent_response)[0])
            ai_character = self.ai_characters[index - 1]
        except:
            ai_character = "lawyer"

        system_prompt = self.roles_prompt[ai_character]
        msgs = [{"role": "system", "content": f"{system_prompt}"}]
        msgs.extend(conv)
        response = llm_client.response_from_list(msgs)

        conv.append({"role": "assistant", "content": f"{response}"})

        return {
            "data": conv,
            "result": {"ai_character": ai_character, "intent_detect": intent_response},
        }

    def simple_run(self, conv: List[dict], llm_client: LLMClient):
        intent_response = llm_client.response_from_question(
            self.role_prompt.format(conv=conv[-1]["content"])
        )
        try:
            index = int(re.findall(r"\[return\] (.+?)", intent_response)[0])
            print(index, self.ai_characters)
            ai_character = self.ai_characters[index - 1]
        except:
            ai_character = "lawyer"

        llm_response = {}
        for role in self.ai_characters:
            system_prompt = self.roles_prompt[role]
            msgs = [{"role": "system", "content": f"{system_prompt}"}]
            msgs.extend(conv)
            response = llm_client.response_from_list(msgs)
            llm_response[role] = response

        conv.append({"role": "assistant", "content": f"{llm_response[ai_character]}"})
        return {
            "data": conv,
            "result": {
                "ai_character": ai_character,
                "intent_detect": intent_response,
                "llm_response": llm_response,
            },
        }
