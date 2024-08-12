#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2024-07-15 19:58:39
@Author : su.zhu
@Desc   : 
"""

from LawConv.post_agents import (
    MaxDedupPostAgent,
    SimplifyResponsePostAgent,
)


from LawConv.pre_agents import (
    IsRecommendPreAgent,
    ExtractTelephonePreAgent,
    SeverCharacterPreAgent,
)

#################################################################################################
POST_AGENTS = {
    "max_dedup": MaxDedupPostAgent,
    "simplify_response": SimplifyResponsePostAgent,
}

PRE_AGENTS = {
    "is_recommend": IsRecommendPreAgent,
    "extract_telephone": ExtractTelephonePreAgent,
    "sever_character": SeverCharacterPreAgent,
}
