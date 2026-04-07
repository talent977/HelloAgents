from hello_agents import ReActAgent, SearchTool
from hello_agents.core.llm import HelloAgentsLLM

def test_llm():
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = ''.join(llmClient.think(exampleMessages))
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)


def test_react_agent():
    llm = HelloAgentsLLM()
    agent = ReActAgent(name='reAct', llm=llm, system_prompt="你是一个智能助手，可以使用工具来帮助用户。")
    agent.add_tool(SearchTool())
    ss = agent.run("你可以查下当前的北京标准时间来确认最新时间，然后告诉我今天A股华海药业的涨幅")
    print(ss)

if __name__ == '__main__':
    test_react_agent()