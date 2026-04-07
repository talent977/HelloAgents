#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typer.cli import state

from hello_agents.utils.exception import Resp
from hello_agents.utils.logging import logger


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def registry(self, name, description, func):
        if name in self.tools:
            logger.info(f"错误：工具 '{name}' 已存在")
        else:
            self.tools[name] = {'desc': description, 'func': func}

    def execute(self, name, **kwargs):
        tool = self.tools.get(name)
        if not tool:
            return Resp.fail(f"错误：工具 '{name}' 未找到")
        try:
            result = tool['func'](**kwargs)
        except Exception as e:
            err = f"错误：执行工具时出错 - {e}"
            logger.info(err)
            return Resp.fail(err)
        return result

    def get_description(self,name):
        if name not in self.tools:
            return Resp.fail(f"错误：工具 '{name}' 未找到")
        return Resp.success(self.tools[name]['desc'])

    def get_tool(self,name):
        if name not in self.tools:
            return Resp.fail(f"错误：工具 '{name}' 未找到")
        return Resp.success(self.tools[name])

    def get_tool_list(self):
        return list(self.tools.keys())


def get_weather(city):
    """天气查询工具（模拟数据）"""
    weather_data = {
        "北京": "晴，15-25℃，空气质量良",
        "济南": "多云，18-28℃，空气质量优",
        "广州": "阵雨，22-30℃，空气质量良"
    }
    return weather_data.get(city, f"未找到{city}的天气信息")


# 可选：增加更多工具以扩展Agent能力
def calculate(expression):
    """计算器工具（简化版）"""
    try:
        # 安全警告：实际应用需使用安全评估库
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

class MockLLM():
    def __init__(self,tool_registry,max_steps=10):
        self.tool_registry = tool_registry
        self.max_steps = max_steps

    def generate_thought(self, state):
        """根据当前状态生成思考"""
        # 简单规则：根据关键词选择工具
        if "天气" in state:
            # 提取城市名（简化处理，默认北京）
            city = "北京"
            if "上海" in state:
                city = "上海"
            elif "广州" in state:
                city = "广州"
            return f"用户需要查询{city}的天气，我应该调用get_weather工具，参数为city={city}"
        elif "计算" in state or "+" in state or "*" in state:
            return "用户需要进行数学计算，我应该调用calculate工具"
        else:
            return "我无法理解用户需求，需要请求澄清"

    def parse_action(self, thought):
        """从思考中解析出行动指令"""
        if "get_weather" in thought:
            # 提取城市参数
            city = "北京"
            if "上海" in thought:
                city = "上海"
            elif "广州" in thought:
                city = "广州"
            return {
                "action": "get_weather",
                "action_input": {"city": city}
            }
        elif "calculate" in thought:
            # 提取表达式（简化）
            return {
                "action": "calculate",
                "action_input": {"expression": "3+5*2"}
            }
        else:
            return {
                "action": "final_answer",
                "action_input": {"answer": "我无法处理这个请求"}
            }

class ReActAgent():
    def __init__(self,tool_registry,llm,max_steps=10):
        self.llm = llm
        self.max_steps = max_steps
        self.tool_registry = tool_registry

    def run(self,query):
        state = query
        history = []
        for i in range(self.max_steps):
            print(f"第{i+1}轮对话：")
            thought = self.llm.generate_thought(state)
            print(f"思考：{thought}")
            action = self.llm.parse_action(thought)
            if action['action'] == 'final_answer':
                print(f"最终回答：{action['action_input']['answer']}")
                break
            result = self.tool_registry.execute(action['action'], **action['action_input'])
            print(f"工具执行结果：{result}")
            state = f"{state}\n思考: {thought}\n行动: {action}\n观察: {result}"
            history.append({'thought': thought, 'action': action, 'result': result})
        return history

def main():
    # 1. 初始化组件
    tool_registry = ToolRegistry()
    tool_registry.registry("get_weather", "查询城市天气", get_weather)
    tool_registry.registry("calculate", "数学表达式计算", calculate)

    llm = MockLLM(tool_registry)
    agent = ReActAgent(tool_registry, llm, max_steps=5)

    # 2. 运行测试
    print("=== ReAct Agent 测试 ===")
    print("输入任务：查询北京的天气")
    history = agent.run("查询北京的天气")

    print(f"\n=== 循环结束 ===")
    print(f"共执行{len(history)}步")

if __name__ == "__main__":
    main()