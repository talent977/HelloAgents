#!/usr/bin/env python
# -*- coding: utf-8 -*-
from hello_agents.utils.exception import Resp
from hello_agents.utils.logging import logger


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def registry(self, name, description, func):
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
        return self.tools.get(name, {}).get('desc', '')
