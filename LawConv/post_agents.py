#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from typing import List

from LawConv.utils import LLMClient, ngram_similarity


class PostAgent:
    def __init__(*args, **kwargs):
        ## pass
        pass

    def run(self, conv: List[dict], llm_client: LLMClient):
        return {"data": conv, "result": {}}


class SimplifyResponsePostAgent(PostAgent):
    def __init__(self, max_length_in_char: int) -> None:
        self.max_length_in_char = max_length_in_char

    def run(self, conv: List[dict], llm_client: LLMClient):
        try:
            if len(conv[-1]["content"]) > self.max_length_in_char:
                conv[-1]["content"] = self._simplify_assistant_response(
                    conv[-1]["content"], llm_client
                )
                processed = True
            else:
                processed = False
        except:
            processed = False

        return {"data": conv, "result": {"simplified": processed}}

    def _simplify_assistant_response(self, assistant: str, llm_client: LLMClient):
        simplify_prompt = """请简化下面客服回复中内容，删除关于之前对话内容的描述，直接给出本轮客服回复的重要内容。
原本客服回复如下：
{assistant}
简化客服回复：
"""
        simplified_res = llm_client.response_from_question(
            simplify_prompt.format(assistant=assistant)
        )
        return simplified_res


class MaxDedupPostAgent(PostAgent):
    def __init__(self, max_dedup: int) -> None:
        self.max_dedup = max_dedup

    def run(self, conv: List[dict], type: str, llm_client: LLMClient):
        last_turn = conv[-1]
        dedup_count = 0
        for turn in conv[1:-1:2]:
            if ngram_similarity(last_turn["content"], turn["content"]) > 0.85:
                dedup_count += 1

        turn_dedup = dedup_count >= self.max_dedup

        if turn_dedup:
            ## todo
            system_prompt = """# 角色
你是一名专业且耐心的客服。
## 技能  
当前与用户的对话存在重复，需要重新提问
"""
            msgs = [{"role": "system", "content": f"{system_prompt}"}]
            msgs.extend(conv)
            response = llm_client.response_from_list(msgs)
            conv[-1] = {"role": "assistant", "content": f"{response}"}

        return {
            "data": conv,
            "result": {"turn_dedup": turn_dedup, "dedup_count": dedup_count},
        }
