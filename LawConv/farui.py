import random
from http import HTTPStatus
import dashscope
from dashscope import Generation

dashscope.api_key = "sk-297a0d9ca96d4a2eac2349de2b3e9f05"


def call_with_messages(conv):
    messages = [
        {
            "role": "system",
            "content": "你是一个法律助手,专注于法律问答，回复简洁一些。",
        },
    ]
    messages.extend(conv)
    print(messages)
    response = dashscope.Generation.call(
        "farui-plus",
        messages=messages,
        result_format="message",  # set the result to be "message" format.
    )
    # print(response)
    # print(response.output.choices[0].message)
    res = response.output.choices[0].message.content

    return res


if __name__ == "__main__":
    call_with_messages()
